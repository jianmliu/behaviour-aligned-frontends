#!/usr/bin/env python3
"""SI-SNR gain of RNNoise-style checkpoints on exported evaluation mixtures (<mix>/<scen>/<sid>.wav + .clean.wav)."""
import math, pathlib, sys, torch, soundfile as sf
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rnnoise_torch import RNNoiseTorch
mix = pathlib.Path(sys.argv[1]); cks = sys.argv[2:]
M = {}
for c in cks:
    m = RNNoiseTorch().eval(); m.load_state_dict(torch.load(c, map_location="cpu", weights_only=True)); M[c] = m
def sisnr(e, r):
    r = r - r.mean(); e = e - e.mean(); t = (e * r).sum() / ((r ** 2).sum() + 1e-9) * r
    return 10 * math.log10(float((t ** 2).sum()) / (float(((e - t) ** 2).sum()) + 1e-9) + 1e-9)
G = {c: [] for c in M}
for p in sorted(q for q in mix.glob("*/*.wav") if not q.name.endswith(".clean.wav")):
    y = torch.from_numpy(sf.read(str(p), dtype="float32")[0]); x = torch.from_numpy(sf.read(str(p.with_suffix("").with_suffix(".clean.wav")), dtype="float32")[0]); b = sisnr(y, x)
    with torch.no_grad():
        for c, m in M.items():
            e = m(y[None])[0]; G[c].append(sisnr(e, x[: e.shape[-1]]) - b)
print(f"SI-SNR gain on {len(next(iter(G.values())))} evaluation mixtures in {mix}:")
for c, v in G.items(): print(f"  {c}: {sum(v)/len(v):+.1f} dB")
