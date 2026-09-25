#!/usr/bin/env python3
"""Stage-by-stage numerical check of dfn_torch.DFTorch against libdf, then end-to-end against df.enhance.enhance,
then a gradient check (loss on the enhanced waveform backpropagates to the input waveform and to DFN3 weights)."""
import pathlib
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from dfn_torch import DFTorch, enhance_torch  # noqa: E402
from df.enhance import df_features, enhance, init_df  # noqa: E402
from df.utils import get_norm_alpha  # noqa: E402
from libdf import erb, erb_norm, unit_norm  # noqa: E402

torch.manual_seed(0)
model, st, _ = init_df(log_level="ERROR")
model.eval()
SR, NFFT, HOP = st.sr(), st.fft_size(), st.hop_size()
nb_df = getattr(model, "nb_df", 96)
a = get_norm_alpha(False)
dft = DFTorch(SR, NFFT, HOP, st.erb_widths(), nb_df, alpha=a)
print(f"sr={SR} fft={NFFT} hop={HOP} nb_erb={len(st.erb_widths())} nb_df={nb_df} alpha={a}")

mix = sorted(p for p in pathlib.Path(sys.argv[1]).glob("*/*.wav") if not p.name.endswith(".clean.wav"))[:3]
up = torchaudio.transforms.Resample(16000, SR)
ok = True


def rel(a, b):
    a, b = torch.as_tensor(a), torch.as_tensor(b)
    return float((a - b).abs().max() / (b.abs().max() + 1e-12))


for p in mix:
    y = up(torch.from_numpy(sf.read(str(p), dtype="float32")[0])[None])[:, : SR * 6]
    y = y[:, : y.shape[-1] // HOP * HOP]
    # stage 1: analysis
    ref_spec = torch.as_tensor(st.analysis(y.numpy()))
    my_spec = dft.analysis(y)
    e1 = rel(my_spec, ref_spec)
    # stage 2: erb + erb_norm
    ref_erb = torch.as_tensor(erb_norm(erb(ref_spec.numpy(), st.erb_widths()), a))
    my_erb = dft.erb_norm(dft.erb(ref_spec))
    e2 = rel(my_erb, ref_erb)
    # stage 3: unit_norm
    ref_un = torch.as_tensor(unit_norm(ref_spec[..., :nb_df].numpy(), a))
    my_un = dft.unit_norm(ref_spec[..., :nb_df])
    e3 = rel(my_un, ref_un)
    # stage 4: synthesis
    ref_syn = torch.as_tensor(st.synthesis(ref_spec.numpy().copy()))              # libdf scales its input in place
    my_syn = dft.synthesis(ref_spec)
    e4 = rel(my_syn, ref_syn)
    # stage 5: end to end
    with torch.no_grad():
        ref_out = enhance(model, st, y.clone())
        my_out = enhance_torch(model, dft, y.clone())
    L = min(ref_out.shape[-1], my_out.shape[-1])
    e5 = rel(my_out[..., :L], ref_out[..., :L])
    good = max(e1, e2, e3, e4, e5) < 1e-3
    ok &= good
    print(f"{'OK  ' if good else 'DIFF'} {p.parent.name}/{p.stem}: analysis {e1:.1e} | erb_norm {e2:.1e} | unit_norm {e3:.1e} "
          f"| synthesis {e4:.1e} | end-to-end {e5:.1e} (max relative error)")

# gradient check
x = up(torch.from_numpy(sf.read(str(mix[0]), dtype="float32")[0])[None])[:, : SR * 3]
x = x[:, : x.shape[-1] // HOP * HOP].requires_grad_(True)
model.train(False)
for q in model.parameters():
    q.requires_grad_(True)
out = enhance_torch(model, dft, x)
loss = (out ** 2).mean()
loss.backward()
gx = float(x.grad.abs().sum())
gw = sum(float(q.grad.abs().sum()) for q in model.parameters() if q.grad is not None)
print(f"gradient: d loss/d input |.|={gx:.3e}, d loss/d weights |.|={gw:.3e}, finite={bool(torch.isfinite(x.grad).all())}")
ok &= gx > 0 and gw > 0
print("ALL OK" if ok else "MISMATCH")
