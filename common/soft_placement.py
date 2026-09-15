#!/usr/bin/env python3
"""软落点可微 loss：事件级行为目标，攻击 control-CE 的两个结构问题。

动机（docs/11、2026-08-30 消融定盘）
------------------------------------
control-CE 是帧级代理：keep 多数类主导（靠 mask_control_label 手工下采样），且
"克制↔可听性"的折中不可调——测过的混合（低 lr、SI-SNR 锚）全部塌回信号解吸引子。
本模块把目标改到**事件级**：首响落点的期望成本。落点分类（premature/on-time/miss）
恰是评测口径（humdial_metrics3），成本形状 = 显式操作点旋钮，帧级稀疏问题定义上消失
（只约束事件，不约束每帧）。这是 RL（episode reward）之前该试的可微版本：若它打开
CE 打不开的中间地带，RL 非必要；若不能，RL 的必要性论证成立。

构造
----
决策概率 p_t = P(start-speaking | chunk t)，从 control_logits 在官方 clamp 段
[ctrl_min, start_bc+1) 内 softmax 取 start_speaking 分量，采样于 chunk 末位帧
（官方标签就只在 chunk 末位有效——build_control_label 同款网格）。
首触发分布  w_t = p_t · Π_{s<t}(1 − p_s)，   miss 质量 m = 1 − Σ w_t。
loss = Σ_t w_t · c_t + m · c_miss，c_t 围绕干净轨迹首响帧 t*（事件级自蒸馏——
与 CE 的自蒸馏哲学同源，目标从帧 token 换成落点）：

    c_t = w_pre                （t < t* − pre_slack：premature 区，平顶）
        = 0                    （|t − t*| ≤ on_slack：on-time 窗）
        = late_slope·(t−t*)Δ   （其后线性 late；Δ=chunk 秒）

推荐搭配小权重原 CE 正则（train 侧 --lambda-ce），防止非首响行为（终止/附和）漂移。

★ 上机冒烟必校验（本地无法定）：logits→帧的 shift 约定。校验法：干净重放下
  argmax(control_logits[..., 段]) 应基本重现落盘 generated_control_ids 的 suffix 段；
  用 `verify_shift()`，冒烟时对 3 条轨迹跑，取一致率最高的 shift（0 或 1）。
"""
from __future__ import annotations

import json
import pathlib

import torch

FRAME_HZ = 25.0     # control 帧率（events.json 时间戳同网格）


# ---------------------------------------------------------------- 目标提取
def target_frame_from_events(events_path: pathlib.Path | str) -> int | None:
    """干净轨迹 → 首个非 backchannel response 的 25Hz 帧号（suffix 轴）。无响应 → None。"""
    d = json.loads(pathlib.Path(events_path).read_text())
    evs = [e for e in d.get("events", []) if e.get("type") != "backchannel"]
    if not evs:
        return None
    return int(round(float(evs[0]["start_time"]) * FRAME_HZ))


# ---------------------------------------------------------------- 核心数学
def first_trigger_dist(p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """p (B,T) ∈ [0,1] → 首触发分布 w (B,T) 与 miss 质量 (B,)。
    w_t = p_t · Π_{s<t}(1−p_s)；Σw + miss = 1（数值上用 log 累积防下溢）。"""
    eps = 1e-6                                                # fp32 下 1-1e-8==1.0 → log1p(-1)=-inf → NaN
    log_keep = torch.log1p(-(p.clamp(max=1 - eps)))          # log(1-p)
    cum = torch.cumsum(log_keep, dim=-1)
    prior = torch.cat([torch.zeros_like(cum[..., :1]), cum[..., :-1]], dim=-1)
    w = p * prior.exp()
    miss = cum[..., -1].exp()
    return w, miss


def placement_cost(T: int, t_star_chunk: float, *, w_pre: float, on_slack: float,
                   pre_slack: float, late_slope: float, device=None,
                   pre_ramp: float | None = None, w_pre_min: float = 1.0) -> torch.Tensor:
    """chunk 网格上的每步成本向量 (T,)。t_star_chunk 与 slack/slope/ramp 单位均为 chunk 步。
    pre_ramp=None：premature 平顶 w_pre（原版）；给定 ramp：成本从边界处 w_pre_min 线性升到
    ramp 步以外的 w_pre——平顶对"提前很多开口"的模型没有梯度（09-11 Freeze-Omni 冒烟：
    loss 恒 3.0、‖g‖ 0），斜坡让"晚一点开口"有可微的收益。"""
    t = torch.arange(T, dtype=torch.float32, device=device)
    d = t - float(t_star_chunk)
    cost = torch.zeros(T, device=device)
    if pre_ramp is None:
        cost = torch.where(d < -pre_slack, torch.full_like(cost, w_pre), cost)
    else:
        depth = ((-pre_slack - d) / max(pre_ramp, 1e-6)).clamp(0.0, 1.0)     # 0 在边界，1 在 ramp 外
        pre_cost = w_pre_min + (w_pre - w_pre_min) * depth
        cost = torch.where(d < -pre_slack, pre_cost, cost)
    late = late_slope * (d - on_slack).clamp_min(0.0)
    cost = torch.where(d > on_slack, late, cost)
    return cost


def soft_placement_loss(
    control_logits: torch.Tensor,      # (B, L, V) 全长（prefix+suffix）
    cfg,                               # llm.config：control_token_ids_min / start_bc_token_id /
                                       #             start_speaking_token_id / control_token_chunk_size
    prefix_len: int,
    t_star_frame: int | None,          # 干净首响帧（suffix 25Hz 轴）；None = refrain 样本
                                       # （干净解码无响应，727 条中占 24%）——目标改为
                                       # 全程不触发，w_false 惩罚任何触发质量
    *,
    w_false: float = 3.0,
    shift: int = 1,                    # ★冒烟校验：logits[t] 预测 label[t+shift]
    w_pre: float = 3.0,                # premature 平顶成本（克制旋钮）
    w_miss: float = 1.0,               # miss 成本（可听性旋钮）
    on_slack_sec: float = 1.0,
    pre_slack_sec: float = 1.0,
    late_slope_per_sec: float = 0.25,
    pre_ramp_sec: float | None = None,
    w_pre_min: float = 1.0,
    w_margin: float = 0.0,             # logit 边距项：premature 区压 start 逻辑值，绕过 softmax 饱和
    margin: float = 2.0,
) -> tuple[torch.Tensor, dict]:
    """返回 (loss 标量, 诊断 dict)。B=1 起步（与现训练管线一致）。"""
    ctrl_min = int(cfg.control_token_ids_min)
    ctrl_hi = int(cfg.start_bc_token_id) + 1            # 与官方 clamp_loss 同段
    start_id = int(cfg.start_speaking_token_id)
    C = int(cfg.control_token_chunk_size)
    chunk_sec = C / FRAME_HZ

    # suffix 段 + shift 对齐：预测帧 i 的 logits 在 prefix_len + i - shift
    L = control_logits.shape[1]
    n_suffix = L - prefix_len
    seg = control_logits[:, :, ctrl_min:ctrl_hi].float()
    probs = torch.softmax(seg, dim=-1)[..., start_id - ctrl_min]   # (B, L)

    # chunk 末位帧的决策概率（官方标签网格）：suffix 帧 i = C-1, 2C-1, ...
    idx_suffix = torch.arange(C - 1, n_suffix, C, device=control_logits.device)
    idx = (idx_suffix + prefix_len - shift).clamp(min=0, max=L - 1)
    p = probs[:, idx]                                              # (B, Tc)

    w, miss = first_trigger_dist(p)
    z = seg[:, idx, :]                                        # (B, Tc, K) 段内 logits
    z1 = z[..., start_id - ctrl_min]
    z_other = z.clone(); z_other[..., start_id - ctrl_min] = -1e4
    zmax_other = z_other.max(-1).values
    excess = torch.relu(z1 - zmax_other + margin)              # >0 表示 start 逻辑值过高
    if t_star_frame is None:                    # refrain：触发质量即错误
        loss = w_false * (1 - miss).mean()
        if w_margin > 0:
            loss = loss + w_margin * excess.mean()
        with torch.no_grad():
            diag = {"miss_mass": float(miss.mean()), "t_soft_chunk": -1.0,
                    "t_star_chunk": -1.0, "trigger_mass": float(1 - miss.mean())}
        return loss, diag
    t_star_chunk = t_star_frame / C
    cost = placement_cost(
        p.shape[-1], t_star_chunk,
        w_pre=w_pre,
        on_slack=on_slack_sec / chunk_sec,
        pre_slack=pre_slack_sec / chunk_sec,
        late_slope=late_slope_per_sec * chunk_sec,
        device=p.device,
        pre_ramp=(pre_ramp_sec / chunk_sec) if pre_ramp_sec else None, w_pre_min=w_pre_min,
    )
    loss = (w * cost[None, :]).sum(-1).mean() + w_miss * miss.mean()
    if w_margin > 0:
        tc = torch.arange(p.shape[-1], device=p.device, dtype=torch.float32)
        pre_mask = (tc < t_star_chunk - pre_slack_sec / chunk_sec).float()
        loss = loss + w_margin * (excess * pre_mask[None]).sum() / pre_mask.sum().clamp_min(1.0)
    with torch.no_grad():
        t_soft = (w * torch.arange(p.shape[-1], device=p.device)[None, :]).sum(-1) \
                 / (1 - miss).clamp_min(1e-6)
        diag = {"miss_mass": float(miss.mean()),
                "t_soft_chunk": float(t_soft.mean()),
                "t_star_chunk": float(t_star_chunk),
                "trigger_mass": float(1 - miss.mean())}
    return loss, diag


# ---------------------------------------------------------------- 冒烟校验
@torch.no_grad()
def verify_shift(control_logits: torch.Tensor, control_ids: torch.Tensor,
                 cfg, prefix_len: int) -> dict:
    """干净重放下，各 shift 的 argmax(control 段) vs 落盘 control token 一致率。
    control_ids: (B, L_suffix) —— pack 里切过 prefix 的序列。上机对 3 条轨迹跑，
    一致率最高者即正确 shift（预期 ~0 或 ~1，一致率应显著高于随机）。"""
    ctrl_min = int(cfg.control_token_ids_min)
    ctrl_hi = int(cfg.start_bc_token_id) + 1
    seg = control_logits[:, :, ctrl_min:ctrl_hi]
    pred = seg.argmax(-1) + ctrl_min                    # (B, L)
    out = {}
    n = control_ids.shape[1]
    for shift in (0, 1):
        pr = pred[:, prefix_len - shift: prefix_len - shift + n]
        m = min(pr.shape[1], n)
        out[shift] = float((pr[:, :m] == control_ids[:, :m]).float().mean())
    return out
