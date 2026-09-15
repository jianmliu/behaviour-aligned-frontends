# 批量离线解码（评测/rollout 提速 ~10-16×）— 设计立项 2026-09-11

## 问题
`full_chunk_stream_offline_generation`（hf_v9_generation.py:490）单样本串行：自回归
逐 token = GEMV，带宽受限（每 token 读全部 12.8B 权重）。1000 条评测 H100 2.5h；
mini M4 上 ~20-50h；RL rollout 同病。批量化（GEMV→GEMM）是软落点/RL 迭代的基建。

## 源码侦察结论（批量化的有利与不利面）
**有利**
- 外层循环按 0.4s chunk 网格推进，**离线评测所有样本天然共享同一时间轴**——
  批内 chunk 边界自动同步，这是最大简化
- 状态只有三个：'l' 听（每 chunk 只生成 1 个 control 决策 token）/'s' 说/'b' 附和
  （chunk 内自回归至边界）
- 行为统计：92% 样本单响应、绝大多数 chunk 处于听态 → **只批量化听态就拿走大头收益**

**不利**
- `multi_head_generate` 与外层循环全是单样本索引（`[0, -1]` 散布各处）
- 四套 KV cache（text/stoken/control/merge）需按样本分组重排（index_select）
- 说话态 `do_sample T=0.7`：批量化改变 RNG 流 → 对拍只能到"听态贪婪段 bit 级一致 +
  说话段行为统计一致"，不能全程 bit 级

## 分期方案
**v1（2-3 天）：听态批量 + 说话态串行退化**
- 每 chunk：批内按状态分组；L 组拼 batch 一次 forward 出全组决策（pad+detect 构造
  向量化）；S/B 组逐样本沿用原实现跑到 chunk 边界；组间 KV 按 index_select 重排
- 预期：听态占比 ~90%+ → 端到端 ~7-10×；batch=16 时 1000 条 H100 ~20-25 分钟
- 对拍：3 样本 batch=1 vs batch=3，听态决策序列 bit 级一致（贪婪）

**v2（+2 天）：说话态也批量**
- chunk 内自回归子循环全批推进，完成掩码 + pad；每样本独立 torch.Generator 保采样
  语义；→ 额外 ~1.5-2×，且 RL 的"多 rollout 同样本"场景（GRPO K 采样）直接受益

## 定位与时机
- 不在 ICASSP 关键路径（论文数字已定盘）；截稿（9-16）后动工
- 服务对象：软落点 s0/s1 评测、RL rollout（H100 一晚数万 rollout 成为可能）、
  mini 上 200 条 pilot（→ ~2h 可跑）
- 纪律：改完必须过对拍冒烟（[[smoke-with-real-forward-before-gpu]] 同款教训——
  判据是解码输出一致，不是 import 通过）
