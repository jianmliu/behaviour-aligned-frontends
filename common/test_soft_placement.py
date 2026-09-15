#!/usr/bin/env python3
"""soft_placement 本地单测（Mac，无模型）：数学性质 + 梯度方向 + 真轨迹目标提取。"""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from soft_placement import (first_trigger_dist, placement_cost, soft_placement_loss,
                            target_frame_from_events)


class Cfg:  # 模拟 llm.config 的 control 常量（数值任意，段内关系正确即可）
    control_token_ids_min = 100
    start_speaking_token_id = 103
    start_bc_token_id = 105          # 段 = [100, 106) → 6 类
    control_token_chunk_size = 10    # 0.4s @ 25Hz


def logits_with_p(T_chunks, hot, cfg, L=None, prefix=10, hi=8.0):
    """构造 (1,L,V) logits：指定 chunk 的 start-speaking 打高分，其余压低。"""
    C = cfg.control_token_chunk_size
    L = L or prefix + T_chunks * C
    V = 120
    g = torch.zeros(1, L, V)
    g[:, :, cfg.control_token_ids_min] = 10.0           # 默认类（keep）——真实模型
                                                        # keep 主导时 start 概率 <1e-3
    for tc in hot:
        fr = prefix + tc * C + C - 1 - 1                # chunk 末位帧 − shift(=1)
        g[:, fr, cfg.start_speaking_token_id] = hi
        g[:, fr, cfg.control_token_ids_min] = 0.0
    return g


def main():
    torch.manual_seed(0)
    cfg = Cfg()

    # 1. 首触发分布守恒：Σw + miss = 1
    p = torch.rand(4, 50)
    w, miss = first_trigger_dist(p)
    assert torch.allclose(w.sum(-1) + miss, torch.ones(4), atol=1e-5), "守恒失败"
    # 全零 → 全 miss；首位=1 → w0=1
    w0, m0 = first_trigger_dist(torch.zeros(1, 20))
    assert m0.item() > 0.999
    w1, m1 = first_trigger_dist(torch.tensor([[1.0] + [0.5] * 19]))
    assert w1[0, 0].item() > 0.999 and m1.item() < 1e-6
    print("✓ 首触发分布守恒/边界")

    # 2. 成本几何：premature 平顶、on-time 零、late 线性
    c = placement_cost(40, 20.0, w_pre=3.0, on_slack=2.5, pre_slack=2.5, late_slope=0.1)
    assert c[10] == 3.0 and c[20] == 0.0 and c[19] == 0.0
    assert 0 < c[30] < c[39]
    print("✓ 成本几何")

    # 3. 端到端：on-time 触发 loss ≈ 0；premature 触发 loss 大；无触发 → miss 项
    t_star_fr = 20 * cfg.control_token_chunk_size       # t* = chunk 20
    lo_on, d_on = soft_placement_loss(logits_with_p(40, [20], cfg), cfg, 10, t_star_fr)
    lo_pre, d_pre = soft_placement_loss(logits_with_p(40, [5], cfg), cfg, 10, t_star_fr)
    lo_miss, d_miss = soft_placement_loss(logits_with_p(40, [], cfg), cfg, 10, t_star_fr)
    assert lo_on.item() < 0.1, f"on-time loss 应≈0，得 {lo_on.item():.3f}"
    assert lo_pre.item() > 1.0, f"premature loss 应大，得 {lo_pre.item():.3f}"
    assert d_miss["miss_mass"] > 0.9 and lo_miss.item() > 0.5
    print(f"✓ 端到端: on={lo_on.item():.3f} pre={lo_pre.item():.3f} miss={lo_miss.item():.3f}")

    # 4. 梯度方向：premature 场景下，降 loss 应压低 premature 帧的 start 概率、
    #    抬高 t* 帧的（trigger_mass 通道）
    g = logits_with_p(40, [5], cfg).requires_grad_(True)
    lo, _ = soft_placement_loss(g, cfg, 10, t_star_fr)
    lo.backward()
    C = cfg.control_token_chunk_size
    fr_pre = 10 + 5 * C + C - 1 - 1
    fr_star = 10 + 20 * C + C - 1 - 1
    g_pre = g.grad[0, fr_pre, cfg.start_speaking_token_id].item()
    g_star = g.grad[0, fr_star, cfg.start_speaking_token_id].item()
    assert g_pre > 0, f"premature 帧梯度应为正（压低），得 {g_pre:.2e}"
    assert g_star < 0, f"t* 帧梯度应为负（抬高），得 {g_star:.2e}"
    print(f"✓ 梯度方向: ∂L/∂pre={g_pre:.2e} ∂L/∂t*={g_star:.2e}")

    # 5. 旋钮单调性：w_pre 越大，premature 场景 loss 越大
    l1, _ = soft_placement_loss(logits_with_p(40, [5], cfg), cfg, 10, t_star_fr, w_pre=1.0)
    l5, _ = soft_placement_loss(logits_with_p(40, [5], cfg), cfg, 10, t_star_fr, w_pre=5.0)
    assert l5.item() > l1.item()
    print("✓ 操作点旋钮单调")

    # 5c. premature 斜坡：在触发 chunk 上，斜坡给出比平顶更强的"往后挪"梯度
    #    （触发点之后的 chunk 本无质量，那里的梯度≈0 是正确行为）
    fr5 = 10 + 5 * C + C - 1 - 1
    g_flat = logits_with_p(40, [5], cfg).requires_grad_(True)
    soft_placement_loss(g_flat, cfg, 10, t_star_fr)[0].backward()
    g_ramp = logits_with_p(40, [5], cfg).requires_grad_(True)
    soft_placement_loss(g_ramp, cfg, 10, t_star_fr, pre_ramp_sec=5.0)[0].backward()
    gf = g_flat.grad[0, fr5, cfg.start_speaking_token_id].item()
    gr = g_ramp.grad[0, fr5, cfg.start_speaking_token_id].item()
    assert gr > 0 and gr >= gf - 1e-9, f"斜坡梯度应为正且不弱于平顶: flat={gf:.2e} ramp={gr:.2e}"
    print(f"✓ premature 斜坡: 触发点梯度 平顶={gf:.2e} 斜坡={gr:.2e}")

    # 6. 真轨迹目标提取（727 条备份）
    root = pathlib.Path.home() / "rspeech_lychee_backup/fdb_out/LycheeFD_clean"
    if root.exists():
        ts, none_n = [], 0
        for p_ in sorted(root.glob("*/*/events.json")):
            t = target_frame_from_events(p_)
            if t is None: none_n += 1
            else: ts.append(t)
        ts_t = torch.tensor(ts, dtype=torch.float32)
        print(f"✓ 真轨迹 t*: n={len(ts)} 无响应={none_n} "
              f"中位={ts_t.median()/25:.1f}s p10={ts_t.quantile(0.1)/25:.1f}s "
              f"p90={ts_t.quantile(0.9)/25:.1f}s")
        assert len(ts) > 500 and none_n > 100 and (ts_t >= 0).all()
    else:
        print(f"⚠ 备份轨迹不存在（{root}），跳过第 6 项")

    print("\n全部通过")


if __name__ == "__main__":
    main()
