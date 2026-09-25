#!/usr/bin/env python3
"""SI-SNR gain of front-end checkpoints on FDB clips mixed with a noise corpus at a given SNR range (seeded)."""
import argparse, math, pathlib, random, sys
import soundfile as sf, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rnnoise_torch import RNNoiseTorch
ap = argparse.ArgumentParser()
ap.add_argument("--fdb", type=pathlib.Path, required=True); ap.add_argument("--noise-dir", type=pathlib.Path, required=True)
ap.add_argument("--noise-glob", default="ch01.wav"); ap.add_argument("--snr", type=float, nargs=2, default=(-20.0, -10.0))
ap.add_argument("--n", type=int, default=60); ap.add_argument("ckpts", nargs="+")
a = ap.parse_args(); random.seed(0)
wavs = sorted(a.fdb.glob("*/*/input.wav")); random.shuffle(wavs); wavs = wavs[: a.n]
noises = [torch.from_numpy(sf.read(str(f), dtype="float32", always_2d=True)[0].mean(1)) for f in sorted(a.noise_dir.rglob(a.noise_glob)) if sf.info(str(f)).samplerate == 16000]
def rd(p):
    x, sr = sf.read(str(p), dtype="float32", always_2d=True); x = torch.from_numpy(x.mean(1))
    if sr != 16000:
        import torchaudio; x = torchaudio.transforms.Resample(sr, 16000)(x)
    return x[: 12 * 16000]
def sisnr(e, r):
    r = r - r.mean(); e = e - e.mean(); t = (e * r).sum() / ((r ** 2).sum() + 1e-9) * r
    return 10 * math.log10(float((t ** 2).sum()) / (float(((e - t) ** 2).sum()) + 1e-9) + 1e-9)
models = {}
for c in a.ckpts:
    m = RNNoiseTorch().eval(); m.load_state_dict(torch.load(c, map_location="cpu", weights_only=True)); models[c] = m
acc = {c: [] for c in models}; base = []
for i, w in enumerate(wavs):
    x = rd(w); nz = noises[i % len(noises)]; K = len(x); off = random.randrange(max(len(nz) - K, 1)); n = nz[off:off + K]
    if len(n) < K: n = n.repeat(-(-K // max(len(n), 1)))[:K]
    snr = a.snr[0] + random.random() * (a.snr[1] - a.snr[0]); y = x + n * math.sqrt((float((x**2).mean()) + 1e-9) / (float((n**2).mean()) + 1e-9) / 10 ** (snr / 10))
    b = sisnr(y, x); base.append(b)
    with torch.no_grad():
        for c, m in models.items():
            e = m(y[None])[0]; acc[c].append(sisnr(e, x[: e.shape[-1]]) - b)
print(f"noisy SI-SNR {sum(base)/len(base):.1f} dB on {a.noise_dir.name} U{list(a.snr)}")
for c, v in acc.items(): print(f"  {c}: gain {sum(v)/len(v):+.1f} dB")
