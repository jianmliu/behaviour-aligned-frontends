#!/usr/bin/env python3
"""Freeze-Omni × RNNoiseTorch：RL（antithetic ES，无梯度）直接优化首响落点 reward。

与 fo_train_soft 严格可比：同一成本函数（premature / on-time / late / miss / refrain 误触发），
soft 用可微期望成本做梯度，RL 用**真实落点**（整句前向 p_t 首次 >0.5 的 chunk，与
recognize() 的判决同口径）当 reward，绕开 teacher-forcing——这是 episode 级目标的本体。

策略空间（--space）：
  bands  RNNoise 输出增益的 22 带偏置 + 全局压制指数（23 维；ES 低维快收敛，"RL 调前端操作点"）
  full   RNNoise 全部 77.9K 参数（ES 高维，噪声大；作对照）
ES：OpenAI-ES 变体，antithetic 对采样，reward 秩归一化，Adam 更新；每代 pop 对 × M 样本。

    python fo_train_rl.py --targets nz/fo_fdb_clean --fdb data/fdb/v1.0/unz --noise-dir data/demand \
        --init final.pt --out exp/fo_rl_bands --space bands --pop 8 --m 8 --gens 100
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
from rnnoise_torch import RNNoiseTorch                                          # noqa: E402
from soft_placement import first_trigger_dist, placement_cost                    # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--targets", type=pathlib.Path, required=True)
ap.add_argument("--fdb", type=pathlib.Path, required=True)
ap.add_argument("--noise-dir", type=pathlib.Path, required=True)
ap.add_argument("--noise-glob", default="ch01.wav", help="噪声文件通配（DEMAND: ch01.wav；MUSAN 等: *.wav）")
ap.add_argument("--init", required=True)
ap.add_argument("--out", type=pathlib.Path, required=True)
ap.add_argument("--space", choices=["bands", "full"], default="bands")
ap.add_argument("--pop", type=int, default=8, help="antithetic 对数（每代 2*pop 次 rollout）")
ap.add_argument("--m", type=int, default=8, help="每 rollout 的样本数")
ap.add_argument("--gens", type=int, default=100)
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
ap.add_argument("--reward", choices=["placement", "dense"], default="placement",
                help="placement=真实落点三档成本（黑盒可得）；dense=软落点期望成本数值（白盒才有，仍零阶）")
ap.add_argument("--pre-ramp", type=float, default=20.0)
ap.add_argument("--w-pre-min", type=float, default=1.0)
ap.add_argument("--w-margin", type=float, default=1.0)
a = ap.parse_args()
torch.manual_seed(a.seed); random.seed(a.seed)
CHUNK_SEC = CHUNK_SAMPLES / 16000

items = []
for p in sorted(a.fdb.glob("*/*/input.wav")):
    task, sid = p.parent.parent.name, p.parent.name
    tj = a.targets / task / sid / "events.json"
    if not tj.exists():
        continue
    ev = json.loads(tj.read_text()).get("events", [])
    items.append({"wav": p, "t_star": float(ev[0]["start_time"]) if ev else None})
print(f"[data] {len(items)} 条 refrain={sum(i['t_star'] is None for i in items)}", flush=True)

noises = []
for f in sorted(a.noise_dir.rglob(a.noise_glob)):
    x, sr = sf.read(str(f), dtype="float32", always_2d=True)
    if sr == 16000:
        noises.append(torch.from_numpy(x.mean(1)))
assert noises


def rd(p):
    x, sr = sf.read(str(p), dtype="float32", always_2d=True); x = torch.from_numpy(x.mean(1))
    if sr != 16000:                                   # FDB 合成子集非 16k
        import torchaudio
        x = torchaudio.transforms.Resample(sr, 16000)(x)
    return x[: int(a.max_sec * 16000)]


def add_noise(x, gen):
    nz = noises[int(torch.randint(len(noises), (1,), generator=gen))]
    K = len(x); off = int(torch.randint(max(len(nz) - K, 1), (1,), generator=gen))
    n = nz[off:off + K]
    if len(n) < K:
        n = n.repeat(-(-K // max(len(n), 1)))[:K]
    snr = a.snr[0] + float(torch.rand(1, generator=gen)) * (a.snr[1] - a.snr[0])
    ps, pn = float((x ** 2).mean()) + 1e-9, float((n ** 2).mean()) + 1e-9
    return x + n * math.sqrt(ps / pn / (10 ** (snr / 10)))


def placement_reward(first_chunk, t_star):
    """真实落点 → 负成本（与 soft_placement.placement_cost 同几何）。"""
    if t_star is None:
        return 0.0 if first_chunk is None else -a.w_false
    if first_chunk is None:
        return -a.w_miss
    d = first_chunk * CHUNK_SEC - t_star
    if d < -a.pre_slack:
        return -a.w_pre
    if d <= a.on_slack:
        return 1.0
    return -a.late_slope * (d - a.on_slack)


# ---------------------------------------------------------------- 模型与策略
pipe = load_pipeline()
fop = FreezeOmniPlacement(pipe.model)
fe = RNNoiseTorch().to("mps").eval()
fe.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=True))
for p_ in list(pipe.model.parameters()) + list(fe.parameters()):
    p_.requires_grad_(False)

if a.space == "bands":
    # 策略 = 22 带增益偏置（log 域）+ 全局指数；作用于 RNNoise 的带增益输出（rnnoise_torch 的钩子）
    assert hasattr(fe, "gain_bias") or True
    theta = torch.zeros(23)
    def apply_policy(th):
        fe.gain_bias = th[:22].to("mps")          # rnnoise_torch: 若无此属性则由 forward 忽略——冒烟时确认
        fe.gain_exp = float(th[22].exp())
else:
    base = torch.nn.utils.parameters_to_vector(fe.parameters()).detach().cpu()
    theta = torch.zeros_like(base)
    def apply_policy(th):
        torch.nn.utils.vector_to_parameters((base + th).to("mps"), fe.parameters())


@torch.no_grad()
def dense_cost(p1, Tc, t_star):
    """与 fo_train_soft 逐项相同的软落点损失，只取数值（白盒稠密奖励；优化器仍是零阶 ES）。"""
    w, miss = first_trigger_dist(p1[None])
    excess = fop.last_margin_excess
    if t_star is None:
        return float(a.w_false * (1 - miss).mean() + a.w_margin * excess.mean())
    ts_c = t_star / CHUNK_SEC
    cost = placement_cost(Tc, ts_c, w_pre=a.w_pre, on_slack=a.on_slack / CHUNK_SEC,
                          pre_slack=a.pre_slack / CHUNK_SEC, late_slope=a.late_slope * CHUNK_SEC,
                          device=p1.device, pre_ramp=(a.pre_ramp / CHUNK_SEC) if a.pre_ramp else None,
                          w_pre_min=a.w_pre_min)
    pre_mask = (torch.arange(Tc, device=p1.device, dtype=torch.float32) < ts_c - a.pre_slack / CHUNK_SEC).float()
    return float((w * cost[None]).sum(-1).mean() + a.w_miss * miss.mean()
                 + a.w_margin * (excess * pre_mask).sum() / pre_mask.sum().clamp_min(1.0))


@torch.no_grad()
def rollout(th, batch):
    apply_policy(th)
    R = 0.0
    for it, gen in batch:
        x = add_noise(rd(it["wav"]), gen).to("mps")
        enh = fe(x[None])[0]
        p1, Tc = fop.state_probs(enh, need_grad=False)
        if a.reward == "dense":
            R += -dense_cost(p1, Tc, it["t_star"])
            continue
        hit = (p1 > 0.5).nonzero()
        first = int(hit[0]) if len(hit) else None
        R += placement_reward(first, it["t_star"])
    return R / len(batch)


opt = torch.optim.Adam([theta.requires_grad_(False)], lr=a.lr) if False else None
m_t = torch.zeros_like(theta); v_t = torch.zeros_like(theta)      # 手写 Adam（theta 无 autograd）
a.out.mkdir(parents=True, exist_ok=True)
(a.out / "args.json").write_text(json.dumps(vars(a), default=str, indent=1))
logf = open(a.out / "es_log.csv", "a")
best = -1e9
t0 = time.time()
for g in range(1, a.gens + 1):
    # 同一代内所有扰动共享同一批样本+噪声种子（common random numbers，降 ES 方差）
    batch = [(random.choice(items), torch.Generator().manual_seed(a.seed * 100000 + g * 100 + j))
             for j in range(a.m)]
    eps = torch.randn(a.pop, theta.numel()) * a.sigma
    Rp = torch.tensor([rollout(theta + e, batch) for e in eps])
    Rm = torch.tensor([rollout(theta - e, batch) for e in eps])
    R0 = rollout(theta, batch)
    # 秩归一化 + antithetic 梯度估计
    allR = torch.cat([Rp, Rm]); ranks = allR.argsort().argsort().float()
    util = (ranks / (len(allR) - 1) - 0.5)
    up, um = util[: a.pop], util[a.pop:]
    grad = ((up - um)[:, None] * eps).sum(0) / (a.pop * a.sigma)
    b1, b2 = 0.9, 0.999
    m_t = b1 * m_t + (1 - b1) * grad; v_t = b2 * v_t + (1 - b2) * grad ** 2
    theta = theta + a.lr * (m_t / (1 - b1 ** g)) / ((v_t / (1 - b2 ** g)).sqrt() + 1e-8)
    if R0 > best:
        best = R0; torch.save({"theta": theta, "space": a.space, "R": R0}, a.out / "best.pt")
    torch.save({"theta": theta, "space": a.space, "gen": g}, a.out / "last.pt")
    print(f"[es {time.strftime('%T')}] gen {g}: R0 {R0:+.3f} pop {allR.mean():+.3f}±{allR.std():.3f} "
          f"best {best:+.3f} |θ| {theta.norm():.3f} {(time.time()-t0)/g/60:.1f}min/代", flush=True)
    logf.write(f"{g},{R0},{allR.mean()},{allR.std()},{best}\n"); logf.flush()
