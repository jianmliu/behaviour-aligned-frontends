#!/usr/bin/env python3
"""Lychee-FD × RNNoiseTorch：黑盒 RL（antithetic ES）——与 fo_train_rl 同一配方，rollout 换成
Lychee 官方离线全双工解码（fdb_infer_lychee 同路径），reward 只用其首响事件（不碰 logits）。

    python train_rnnoise_es.py --lychee-repo ... --lychee-ckpt ... --fdb-root $WORK/data/fdb/v1.0 \
        --targets $WORK/nz/fdb_ann --noise-dir $WORK/data/demand --init $WORK/rnnoise_pretrain/final.pt \
        --out $WORK/exp/rnj_es_bands --space bands --pop 6 --m 6 --gens 30
rollout 成本：Lychee 离线解码 ~9 s/条（H100）→ 每代 (2*pop+1)*m 条 ≈ 13*6*9 s ≈ 12 min；30 代 ≈ 6 h。
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
import time

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rnnoise_torch import RNNoiseTorch   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--lychee-repo", type=pathlib.Path, required=True)
ap.add_argument("--lychee-ckpt", type=pathlib.Path, required=True)
ap.add_argument("--fdb-root", type=pathlib.Path, required=True)
ap.add_argument("--targets", type=pathlib.Path, required=True, help="events.json 目标树（fdb_targets.py）")
ap.add_argument("--noise-dir", type=pathlib.Path, required=True)
ap.add_argument("--init", required=True)
ap.add_argument("--out", type=pathlib.Path, required=True)
ap.add_argument("--space", choices=["bands", "full"], default="bands")
ap.add_argument("--pop", type=int, default=6)
ap.add_argument("--m", type=int, default=6)
ap.add_argument("--gens", type=int, default=30)
ap.add_argument("--sigma", type=float, default=0.05)
ap.add_argument("--lr", type=float, default=0.05)
ap.add_argument("--snr", type=float, nargs=2, default=(-5.0, 20.0))
ap.add_argument("--max-sec", type=float, default=20.0)
ap.add_argument("--w-pre", type=float, default=3.0)
ap.add_argument("--w-miss", type=float, default=1.0)
ap.add_argument("--w-false", type=float, default=3.0)
ap.add_argument("--on-slack", type=float, default=1.0)
ap.add_argument("--pre-slack", type=float, default=1.0)
ap.add_argument("--late-slope", type=float, default=0.25)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda")
a = ap.parse_args()
torch.manual_seed(a.seed); random.seed(a.seed)

sys.path.insert(0, str(a.lychee_repo))
from lychee_fd.runtime.hf_v9_generation import SingleTurnGenerationFramework   # noqa: E402

items = []
for p in sorted(a.fdb_root.glob("*/*/input.wav")):
    task, sid = p.parent.parent.name, p.parent.name
    tj = a.targets / task / sid / "events.json"
    if not tj.exists():
        continue
    if sf.info(str(p)).duration > a.max_sec:
        continue
    ev = json.loads(tj.read_text()).get("events", [])
    items.append({"wav": p, "t_star": float(ev[0]["start_time"]) if ev else None})
print(f"[data] {len(items)} 条", flush=True)

noises = []
for f in sorted(a.noise_dir.rglob("ch01.wav")):
    x, sr = sf.read(str(f), dtype="float32", always_2d=True)
    if sr == 16000:
        noises.append(torch.from_numpy(x.mean(1)))
assert noises


def rd(p):
    x, sr = sf.read(str(p), dtype="float32", always_2d=True); x = x.mean(1)
    if sr != 16000:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, 16000); x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    return torch.from_numpy(x)


def add_noise(x, gen):
    nz = noises[int(torch.randint(len(noises), (1,), generator=gen))]
    K = len(x); off = int(torch.randint(max(len(nz) - K, 1), (1,), generator=gen))
    n = nz[off:off + K]
    if len(n) < K:
        n = n.repeat(-(-K // max(len(n), 1)))[:K]
    snr = a.snr[0] + float(torch.rand(1, generator=gen)) * (a.snr[1] - a.snr[0])
    ps, pn = float((x ** 2).mean()) + 1e-9, float((n ** 2).mean()) + 1e-9
    return x + n * math.sqrt(ps / pn / (10 ** (snr / 10)))


def placement_reward(first_t, t_star):
    if t_star is None:
        return 0.0 if first_t is None else -a.w_false
    if first_t is None:
        return -a.w_miss
    d = first_t - t_star
    if d < -a.pre_slack:
        return -a.w_pre
    if d <= a.on_slack:
        return 1.0
    return -a.late_slope * (d - a.on_slack)


fw = SingleTurnGenerationFramework(model_type="V9", model_path=str(a.lychee_ckpt), device=a.device,
                                   torch_dtype=torch.bfloat16, allowing_backchannel=True)
fe = RNNoiseTorch().eval()
fe.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=True))
for p_ in fe.parameters():
    p_.requires_grad_(False)

if a.space == "bands":
    theta = torch.zeros(23)
    def apply_policy(th):
        fe.gain_bias = th[:22]; fe.gain_exp = float(th[22].exp())
else:
    base = torch.nn.utils.parameters_to_vector(fe.parameters()).detach().clone()
    theta = torch.zeros_like(base)
    def apply_policy(th):
        torch.nn.utils.vector_to_parameters(base + th, fe.parameters())


@torch.no_grad()
def first_response(wav):
    """Lychee 官方离线解码 → 首个非 backchannel response 的 start_time（秒），无 → None。"""
    res = fw.full_chunk_stream_offline_generation(wav, return_raw_trace=False)
    evs = [e for e in res.get("events", []) if e.get("type") != "backchannel"]
    return float(evs[0]["start_time"]) if evs else None


@torch.no_grad()
def rollout(th, batch):
    apply_policy(th)
    R = 0.0
    for it, gen in batch:
        x = add_noise(rd(it["wav"]), gen)
        enh = fe(x[None])[0]
        R += placement_reward(first_response(enh), it["t_star"])
    return R / len(batch)


m_t = torch.zeros_like(theta); v_t = torch.zeros_like(theta)
a.out.mkdir(parents=True, exist_ok=True)
(a.out / "args.json").write_text(json.dumps(vars(a), default=str, indent=1))
logf = open(a.out / "es_log.csv", "a")
best, t0 = -1e9, time.time()
for g in range(1, a.gens + 1):
    batch = [(random.choice(items), torch.Generator().manual_seed(a.seed * 100000 + g * 100 + j))
             for j in range(a.m)]
    eps = torch.randn(a.pop, theta.numel()) * a.sigma
    Rp = torch.tensor([rollout(theta + e, batch) for e in eps])
    Rm = torch.tensor([rollout(theta - e, batch) for e in eps])
    R0 = rollout(theta, batch)
    allR = torch.cat([Rp, Rm]); ranks = allR.argsort().argsort().float()
    util = ranks / (len(allR) - 1) - 0.5
    grad = ((util[: a.pop] - util[a.pop:])[:, None] * eps).sum(0) / (a.pop * a.sigma)
    b1, b2 = 0.9, 0.999
    m_t = b1 * m_t + (1 - b1) * grad; v_t = b2 * v_t + (1 - b2) * grad ** 2
    theta = theta + a.lr * (m_t / (1 - b1 ** g)) / ((v_t / (1 - b2 ** g)).sqrt() + 1e-8)
    if R0 > best:
        best = R0; torch.save({"theta": theta, "space": a.space, "R": R0}, a.out / "best.pt")
    torch.save({"theta": theta, "space": a.space, "gen": g}, a.out / "last.pt")
    print(f"[es {time.strftime('%T')}] gen {g}: R0 {R0:+.3f} pop {allR.mean():+.3f}±{allR.std():.3f} "
          f"best {best:+.3f} |θ| {theta.norm():.3f} {(time.time()-t0)/g/60:.1f}min/代", flush=True)
    logf.write(f"{g},{R0},{allR.mean()},{allR.std()},{best}\n"); logf.flush()
