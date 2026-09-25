#!/usr/bin/env python3
"""Enhance exported HumDial mixtures with DeepFilterNet3 (48 kHz model; 16k -> 48k -> 16k) and report the
SI-SNR gain of DeepFilterNet3 and of the frozen RNNoise-style suppressor on the same evaluation mixtures.

    dfn_venv/bin/python dfn_enhance.py --mix mix/demand_m15 --out enh/dfn3_demand_m15 \
        --rnnoise .../common/rnnoise_pretrained_final.pt --rnnoise-code .../common
"""
import argparse
import math
import pathlib
import sys
import time

import soundfile as sf
import torch
import torchaudio
from df.enhance import enhance, init_df

ap = argparse.ArgumentParser()
ap.add_argument("--mix", type=pathlib.Path, required=True)
ap.add_argument("--out", type=pathlib.Path, required=True)
ap.add_argument("--rnnoise", default=None)
ap.add_argument("--rnnoise-code", default=None)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

model, df_state, _ = init_df()                     # DeepFilterNet3 (default model of deepfilternet 0.5.x)
SR = df_state.sr()
up = torchaudio.transforms.Resample(16000, SR)
down = torchaudio.transforms.Resample(SR, 16000)
fe = None
if a.rnnoise:
    sys.path.insert(0, a.rnnoise_code)
    from rnnoise_torch import RNNoiseTorch
    fe = RNNoiseTorch().eval(); fe.load_state_dict(torch.load(a.rnnoise, map_location="cpu", weights_only=True))


def sisnr(e, r):
    r = r - r.mean(); e = e - e.mean(); t = (e * r).sum() / ((r ** 2).sum() + 1e-9) * r
    return 10 * math.log10(float((t ** 2).sum()) / (float(((e - t) ** 2).sum()) + 1e-9) + 1e-9)


files = sorted(p for p in a.mix.glob("*/*.wav") if not p.name.endswith(".clean.wav"))
if a.limit:
    files = files[: a.limit]
g_dfn, g_rn, t0 = [], [], time.time()
for i, p in enumerate(files):
    y = torch.from_numpy(sf.read(str(p), dtype="float32")[0])
    x = torch.from_numpy(sf.read(str(p.with_suffix("").with_suffix(".clean.wav")), dtype="float32")[0])
    with torch.no_grad():
        e48 = enhance(model, df_state, up(y[None]))
        e = down(e48)[0][: len(y)]
        if len(e) < len(y):
            e = torch.nn.functional.pad(e, (0, len(y) - len(e)))
    od = a.out / p.parent.name; od.mkdir(parents=True, exist_ok=True)
    sf.write(str(od / p.name), e.numpy(), 16000, subtype="FLOAT")
    b = sisnr(y, x); g_dfn.append(sisnr(e, x) - b)
    if fe is not None:
        with torch.no_grad():
            r = fe(y[None])[0]
        g_rn.append(sisnr(r, x[: r.shape[-1]]) - b)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(files)}  {(time.time()-t0)/(i+1):.2f}s/clip  DFN3 gain {sum(g_dfn)/len(g_dfn):+.1f} dB"
              + (f"  RNNoise gain {sum(g_rn)/len(g_rn):+.1f} dB" if g_rn else ""), flush=True)
print(f"done {len(files)} -> {a.out} | SI-SNR gain on the evaluation mixtures: DFN3 {sum(g_dfn)/len(g_dfn):+.1f} dB"
      + (f", frozen RNNoise {sum(g_rn)/len(g_rn):+.1f} dB" if g_rn else ""))
