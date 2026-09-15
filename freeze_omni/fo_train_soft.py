#!/usr/bin/env python3
"""Freeze-Omni × RNNoiseTorch：软落点事件级目标训练（mini/MPS 可跑；7B 冻结，只训 78K 前端）。

数据：FDB v1.0 727 条 input.wav（≤20s）+ DEMAND ch01 加噪；目标 t* = Freeze-Omni 自己在
干净音频上的首个 ss chunk（nz/fo_fdb_clean 的 events.json，fo_placement --fdb 产出），
无响应样本 = refrain（惩罚任何触发）。与 Lychee 线 soft_placement 同一套数学/旋钮。

    python fo_train_soft.py --targets nz/fo_fdb_clean --fdb data/fdb/v1.0/unz \
        --noise-dir data/demand --init final.pt --out exp/fo_s0 --w-pre 3 --w-miss 1
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
from fo_soft_forward import FreezeOmniPlacement, load_pipeline, CHUNK_SAMPLES   # noqa: E402
from soft_placement import first_trigger_dist, placement_cost                    # noqa: E402
from rnnoise_torch import RNNoiseTorch                                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--targets", type=pathlib.Path, required=True)
ap.add_argument("--fdb", type=pathlib.Path, required=True)
ap.add_argument("--noise-dir", type=pathlib.Path, required=True)
ap.add_argument("--init", default=None)
ap.add_argument("--out", type=pathlib.Path, required=True)
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--snr", type=float, nargs=2, default=(-5.0, 20.0))
ap.add_argument("--max-sec", type=float, default=12.0)
ap.add_argument("--pre-ramp", type=float, default=20.0, help="premature 斜坡长度(s)；0=平顶（默认覆盖全区）")
ap.add_argument("--w-margin", type=float, default=1.0, help="logit 边距项权重（premature 区压 start 逻辑值）")
ap.add_argument("--margin", type=float, default=2.0)
ap.add_argument("--w-pre-min", type=float, default=1.0)
ap.add_argument("--w-pre", type=float, default=3.0)
ap.add_argument("--w-miss", type=float, default=1.0)
ap.add_argument("--w-false", type=float, default=3.0)
ap.add_argument("--on-slack", type=float, default=1.0)
ap.add_argument("--pre-slack", type=float, default=1.0)
ap.add_argument("--late-slope", type=float, default=0.25)
ap.add_argument("--clip", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-steps", type=int, default=0)
ap.add_argument("--log-every", type=int, default=5)
ap.add_argument("--save-every", type=int, default=50)
a = ap.parse_args()
torch.manual_seed(a.seed); random.seed(a.seed)
CHUNK_SEC = CHUNK_SAMPLES / 16000

# 样本清单 + 目标
items = []
for p in sorted(a.fdb.glob("*/*/input.wav")):
    task, sid = p.parent.parent.name, p.parent.name
    tj = a.targets / task / sid / "events.json"
    if not tj.exists():
        continue
    ev = json.loads(tj.read_text()).get("events", [])
    t_star = float(ev[0]["start_time"]) if ev else None          # 秒；None = refrain
    items.append({"wav": p, "t_star": t_star, "key": f"{task}/{sid}"})
n_ref = sum(i["t_star"] is None for i in items)
print(f"[data] {len(items)} 条  respond {len(items)-n_ref} / refrain {n_ref}", flush=True)
assert items

noises = []
for f in sorted(a.noise_dir.rglob("ch01.wav")):
    x, sr = sf.read(str(f), dtype="float32", always_2d=True)
    if sr != 16000:
        continue
    noises.append(torch.from_numpy(x.mean(1)))
assert noises, f"噪声库为空: {a.noise_dir}"
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
    if sr != 16000:                                   # FDB 合成子集非 16k
        import torchaudio
        x = torchaudio.transforms.Resample(sr, 16000)(x)
    return x[: int(a.max_sec * 16000)]


pipe = load_pipeline()
fop = FreezeOmniPlacement(pipe.model)
fe = RNNoiseTorch().to("mps")
if a.init:
    fe.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=True))
for p_ in pipe.model.parameters():
    p_.requires_grad_(False)
opt = torch.optim.AdamW(fe.parameters(), lr=a.lr)
a.out.mkdir(parents=True, exist_ok=True)
(a.out / "args.json").write_text(json.dumps(vars(a), default=str, indent=1))
logf = open(a.out / "train_log.csv", "a")

step, t0 = 0, time.time()
for ep in range(1, a.epochs + 1):
    random.shuffle(items)
    for it in items:
        x = add_noise(rd(it["wav"])).to("mps")
        enh = fe(x[None])[0]
        p1, Tc = fop.state_probs(enh, need_grad=True)
        w, miss = first_trigger_dist(p1[None])
        excess = fop.last_margin_excess                       # (Tc,) relu(z1 - zmax_other + margin)
        if it["t_star"] is None:
            loss = a.w_false * (1 - miss).mean() + a.w_margin * excess.mean()
        else:
            ts_c = it["t_star"] / CHUNK_SEC
            cost = placement_cost(Tc, ts_c, w_pre=a.w_pre,
                                  on_slack=a.on_slack / CHUNK_SEC, pre_slack=a.pre_slack / CHUNK_SEC,
                                  late_slope=a.late_slope * CHUNK_SEC, device=p1.device,
                                  pre_ramp=(a.pre_ramp / CHUNK_SEC) if a.pre_ramp else None,
                                  w_pre_min=a.w_pre_min)
            pre_mask = (torch.arange(Tc, device=p1.device, dtype=torch.float32) < ts_c - a.pre_slack / CHUNK_SEC).float()
            loss = (w * cost[None]).sum(-1).mean() + a.w_miss * miss.mean() \
                 + a.w_margin * (excess * pre_mask).sum() / pre_mask.sum().clamp_min(1.0)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(fe.parameters(), a.clip)
        opt.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % a.log_every == 0:
            print(f"[fo-soft {time.strftime('%T')}] ep {ep} step {step}: loss {float(loss):.3f} "
                  f"trig {float(1-miss.mean()):.2f} ‖g‖ {float(gn):.2f} {(time.time()-t0)/step:.1f}s/步",
                  flush=True)
            logf.write(f"{ep},{step},{float(loss)},{float(1-miss.mean())},{float(gn)}\n"); logf.flush()
        if step % a.save_every == 0:
            torch.save(fe.state_dict(), a.out / "last.pt")
        if a.max_steps and step >= a.max_steps:
            print(f"[smoke] {a.max_steps} 步完成 loss={float(loss):.3f}", flush=True)
            torch.save(fe.state_dict(), a.out / "last.pt"); sys.exit(0)
    torch.save(fe.state_dict(), a.out / f"epoch_{ep:03d}.pt")
    print(f"[fo-soft] epoch {ep} 完成", flush=True)
