#!/usr/bin/env python3
"""Export the exact noisy HumDial mixtures that fo_placement.py feeds the model (same sample list, same
per-sample seeded noise, same SNR rule), without loading the model. Used to run an external enhancer
(e.g. DeepFilterNet3) and then evaluate its output with fo_placement.py --pre-enhanced.

    python fo_dump_mix.py --humdial data/humdial/unz/test/en_test_nondev --per-scen 100 \
        --noise-dir data/demand --snr -15 -15 --out mix/demand_m15
Writes <out>/<scenario>/<sid>.wav (16 kHz float32) plus <out>/<scenario>/<sid>.clean.wav for SI-SNR.
"""
import argparse
import math
import pathlib

import soundfile as sf
import torch
import torchaudio

ap = argparse.ArgumentParser()
ap.add_argument("--humdial", type=pathlib.Path, required=True)
ap.add_argument("--out", type=pathlib.Path, required=True)
ap.add_argument("--per-scen", type=int, default=20)
ap.add_argument("--noise-dir", type=pathlib.Path, required=True)
ap.add_argument("--noise-glob", default="ch01.wav")
ap.add_argument("--snr", type=float, nargs=2, default=(0.0, 15.0))
ap.add_argument("--max-sec", type=float, default=40.0)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

SCENS = ["ask", "backchannel", "deny", "others_talk_to_user_after", "others_talk_to_user_before",
         "pause", "repeat", "shift", "talk_to_others", "wait"]            # same order as fo_placement.py


def rd(p):                                                                 # identical to fo_placement.py
    x, sr = sf.read(str(p), dtype="float32", always_2d=True); x = torch.from_numpy(x.mean(1))
    if sr != 16000:
        x = torchaudio.transforms.Resample(sr, 16000)(x)
    return x


noises = []
for f in sorted(a.noise_dir.rglob(a.noise_glob)):
    x, sr = sf.read(str(f), dtype="float32", always_2d=True)
    if sr != 16000:
        continue
    noises.append(torch.from_numpy(x.mean(1)))
assert noises


def add_noise(x, key):                                                     # identical to fo_placement.py
    seed = int.from_bytes(key.encode(), "little") % (2**31)
    rng = torch.Generator().manual_seed(seed)
    nz = noises[int(torch.randint(len(noises), (1,), generator=rng))]
    K = len(x)
    off = int(torch.randint(max(len(nz) - K, 1), (1,), generator=rng))
    n = nz[off:off + K]
    if len(n) < K:
        n = n.repeat(-(-K // max(len(n), 1)))[:K]
    snr = a.snr[0] + float(torch.rand(1, generator=rng)) * (a.snr[1] - a.snr[0])
    ps, pn = float((x ** 2).mean()) + 1e-9, float((n ** 2).mean()) + 1e-9
    return x + n * math.sqrt(ps / pn / (10 ** (snr / 10))), snr


samples = []
for sc in SCENS:
    samples += [(sc, w) for w in sorted((a.humdial / sc).glob("*.wav"))[: a.per_scen]]
if a.limit:
    samples = samples[: a.limit]
for sc, w in samples:
    sid = w.stem
    x = rd(w)
    if len(x) > a.max_sec * 16000:
        x = x[: int(a.max_sec * 16000)]
    y, _ = add_noise(x, f"{sc}/{sid}")
    od = a.out / sc; od.mkdir(parents=True, exist_ok=True)
    sf.write(str(od / f"{sid}.wav"), y.numpy(), 16000, subtype="FLOAT")
    sf.write(str(od / f"{sid}.clean.wav"), x.numpy(), 16000, subtype="FLOAT")
print(f"dumped {len(samples)} mixtures -> {a.out}")
