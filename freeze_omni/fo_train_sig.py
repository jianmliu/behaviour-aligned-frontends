#!/usr/bin/env python3
"""Matched-SNR signal-retrained control for the Freeze-Omni harsh regimes.

Identical to fo_train_soft.py in data (same FDB clips and target filter, same 12 s cut), noise draw
(same seeded generator, same SNR range), optimiser (AdamW, lr, clip) and number of steps; the only
difference is the loss: negative SI-SNR of the enhanced clip against the clean clip. No LLM is loaded.
This isolates "trained at the evaluation SNR" from "trained on the model's decision".

    python fo_train_sig.py --targets nz/fo_fdb_ann --fdb data/fdb/v1.0/unz --noise-dir data/demand \
        --init final.pt --snr -20 -10 --max-sec 12 --epochs 2 --out exp/fo_sig_m15
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
import time

import soundfile as sf
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rnnoise_torch import RNNoiseTorch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--targets", type=pathlib.Path, required=True)
ap.add_argument("--fdb", type=pathlib.Path, required=True)
ap.add_argument("--noise-dir", type=pathlib.Path, required=True)
ap.add_argument("--noise-glob", default="ch01.wav")
ap.add_argument("--init", required=True)
ap.add_argument("--out", type=pathlib.Path, required=True)
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--snr", type=float, nargs=2, default=(-20.0, -10.0))
ap.add_argument("--max-sec", type=float, default=12.0)
ap.add_argument("--clip", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="mps")
ap.add_argument("--log-every", type=int, default=50)
ap.add_argument("--max-steps", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed); random.seed(a.seed)

items = []
for p in sorted(a.fdb.glob("*/*/input.wav")):
    task, sid = p.parent.parent.name, p.parent.name
    if (a.targets / task / sid / "events.json").exists():
        items.append(p)
print(f"[data] {len(items)} clips", flush=True)

noises = []
for f in sorted(a.noise_dir.rglob(a.noise_glob)):
    x, sr = sf.read(str(f), dtype="float32", always_2d=True)
    if sr == 16000:
        noises.append(torch.from_numpy(x.mean(1)))
assert noises, f"empty noise set: {a.noise_dir}"
print(f"[noise] {len(noises)} files from {a.noise_dir} ({a.noise_glob}), SNR U[{a.snr[0]},{a.snr[1]}]", flush=True)
gen = torch.Generator().manual_seed(a.seed)


def add_noise(x):
    nz = noises[int(torch.randint(len(noises), (1,), generator=gen))]
    K = len(x); off = int(torch.randint(max(len(nz) - K, 1), (1,), generator=gen))
    n = nz[off:off + K]
    if len(n) < K:
        n = n.repeat(-(-K // max(len(n), 1)))[:K]
    snr = a.snr[0] + float(torch.rand(1, generator=gen)) * (a.snr[1] - a.snr[0])
    ps, pn = float((x ** 2).mean()) + 1e-9, float((n ** 2).mean()) + 1e-9
    return x + n * math.sqrt(ps / pn / (10 ** (snr / 10)))


def rd(p):
    x, sr = sf.read(str(p), dtype="float32", always_2d=True); x = torch.from_numpy(x.mean(1))
    if sr != 16000:
        import torchaudio
        x = torchaudio.transforms.Resample(sr, 16000)(x)
    return x[: int(a.max_sec * 16000)]


def neg_sisnr(est, ref, eps=1e-8):
    ref = ref - ref.mean(); est = est - est.mean()
    t = (est * ref).sum() / ((ref ** 2).sum() + eps) * ref
    e = est - t
    return -10 * torch.log10((t ** 2).sum() / ((e ** 2).sum() + eps) + eps)


fe = RNNoiseTorch().to(a.device)
fe.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=True))
opt = torch.optim.AdamW(fe.parameters(), lr=a.lr)
a.out.mkdir(parents=True, exist_ok=True)
(a.out / "args.json").write_text(json.dumps(vars(a), default=str, indent=1))
logf = open(a.out / "train_log.csv", "a")

step, t0 = 0, time.time()
for ep in range(1, a.epochs + 1):
    random.shuffle(items)
    run = 0.0
    for p in items:
        clean = rd(p)
        noisy = add_noise(clean)
        enh = fe(noisy.to(a.device)[None])[0]
        loss = neg_sisnr(enh, clean.to(a.device)[: enh.shape[-1]])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(fe.parameters(), a.clip)
        opt.step(); step += 1; run += float(loss)
        if step % a.log_every == 0:
            print(f"[fo-sig {time.strftime('%T')}] ep {ep} step {step}: -SI-SNR {run / a.log_every:+.2f} dB "
                  f"{(time.time() - t0) / step:.2f}s/step", flush=True)
            logf.write(f"{ep},{step},{run / a.log_every}\n"); logf.flush(); run = 0.0
        if a.max_steps and step >= a.max_steps:
            print(f"[smoke] {step} steps loss={float(loss):+.2f}", flush=True); sys.exit(0)
    torch.save(fe.state_dict(), a.out / f"epoch_{ep:03d}.pt")
print(f"[done] {step} steps -> {a.out}", flush=True)
