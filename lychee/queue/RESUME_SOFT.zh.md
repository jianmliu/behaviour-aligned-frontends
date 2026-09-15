# 软落点（soft-placement）上机手册 — 2026-09-11 备妥

代码已全部本地开发+单测通过（`scripts/test_soft_placement.py`，7 项含 refrain 分支、
梯度方向、727 条真轨迹 t\* 提取：respond 554 / refrain 173）。上机只剩执行。

## 一次收割流程（新盒子）
1. `bash bootstrap_v3.sh <ip> [user]`（老三样：备份/监控/盯梢一起切）
2. 增量上传本篇新文件（bootstrap 之外）：
   ```
   scp scripts/{soft_placement.py,train_rnnoise_joint.py,noise_hook.py} <box>:/ephemeral/work/nscripts/
   scp scripts/noise_queue/{29-rnj-soft-train.sh,30-hd-Rjoint-s0.sh,31-hd-Rjoint-s1.sh} <box>:/ephemeral/work/queue/
   ```
3. 队列自动跑：29（冒烟→s0/s1 训练，~10 分钟）→ 30（s0 评测 2.5h）→ 31（s1 评测 2.5h）
   共 ~5.5h ≈ $21。

## 内建护栏（不需要人工盯）
- 29 号先跑 `--max-steps 2` 冒烟；训练脚本 soft 模式首步自动 `verify_shift`
  （干净重放 argmax 对落盘 control token 的一致率），shift 不对齐直接拒训进 failed
- 判活标准照旧：loss 出数（`[rnj ... soft X trig Y]` 行），不是进程存在

## 判读（本地，评测树同步后）
```
python3 scripts/humdial_metrics3.py --root .../ev_Rjoint_s0 --humdial .../en_test_nondev
```
对照表：L0 5.4/59.4/27.4 | N0 3.2/46.4/44.5 | R-indep 6.2/60.1/26.6 |
r0(纯CE) 7.0/43.3/44.7（on-time/premature/miss）
- **成功判据**（打开中间地带）：premature 显著低于 R-indep（<55）且 miss 显著低于
  r0（<38），on-time 不低于 6
- s0/s1 两点若真的分开 → "成本形状=操作点旋钮" 成立，事件级目标优于帧级 CE，
  RL（episode reward）作为下一级写入回声篇
- 若两点都塌向某一吸引子 → 中间地带需要真 rollout（RL 必要性论证成立）

## 已知待验证点（冒烟自动覆盖）
- logits→帧 shift（0 或 1）：verify_shift 自动定，不匹配 `--ctrl-shift` 会拒训
- chunk 网格假设：control_token_chunk_size 帧每 chunk、末位为决策位（与官方
  mask_control_label 同款；若一致率低会在 verify_shift 暴露）

## 2026-09-11 晚补充：Lychee 两制度对照（与 Freeze-Omni 同配方）
上传清单加：`scripts/{fdb_targets.py,train_rnnoise_es.py,humdial_prep.py,rnnoise_torch.py}`（后两者已改：
--policy 钩子 / 带增益策略钩子）→ `/ephemeral/work/nscripts/`；作业 `32-rnj-soft-ann.sh`（标注目标软落点
训练+评测）、`33-rnj-es.sh`（黑盒 ES 训练 ~6h + 评测）→ `/ephemeral/work/queue/`。
两作业合计 ~9.5h ≈ $37（H100）。评测树：ev_s_ann、ev_es；与 ev_Rjoint(r0)/ev_Rindep 同表比较。

## 2026-09-12 凌晨：Freeze-Omni 过夜链收官，论文零 todo
- 结论：Freeze-Omni 的急躁是语言性、电平无关的；冻结/软落点（标注、自蒸馏）/ES 三种前端、干净与加噪、
  −10/−20 dB 探针，落点全部在 200 条抽样噪声内；克制轴（20/20、16/20）从未动过。两制度一致 = "无杠杆"。
- 投稿前清单（截稿 09-16）：① ~~refs.bib 作者名核实~~（09-12 完成：fdbv3/smartglasses2026/uaf2026，bib 零警告）；
  ② 4+1 页数复核（正文止于第 4 页）；③ ~~premature 口径辩护段~~（09-12 已写入协议节）；④ 复现包：scripts/ +
  noise_queue/ + 本手册；⑤ 可选：Lychee ES 臂（作业 33，H100 ~6h/$25）——只在想补"两制度也在 Lychee
  上并排"时做，现稿已自洽。

## 2026-09-12 午后：恶劣制度两制度对照收官（Freeze-Omni）
- −15 dB：无前端 25.5/57.0｜冻结 24.0/61.0（更早 74:30）｜软落点 31.5/52.5（vs 冻结更晚 81:26）｜ES 22.0/63.0（=冻结，比软落点更早 83:30）
- −10 dB：无前端 33.0/49.5｜冻结 26.0/56.0（更早 64:32）｜软落点 35.5/45.5（vs 冻结更晚 73:33）｜ES 24.5/57.5（=冻结 p=0.11，比软落点更早 76:25）
- 配对符号检验口径：首响时刻差 |Δ|≤0.2 s 视为同（脚本 scratchpad/signtest.py，tol=0.2）；论文所有计数同此。
- 论文零 todo；5 页、References 第 4 页起。剩余：ES@−10 与 es_more 出数后可各补一句（须复核页数）。
- es_more：full 空间 ES@−15 同预算 16.0/68.5（比冻结更早 84:48 p=0.002；比软落点 118:30）→ 已入稿；bands 120 代待出。
- refs.bib humdial2026 作者由 'Wang, et al.' 改为 9 位真名（09-12 17:40 PT 发现 PDF 显示 '[1] et al. Wang'）。
- es_more：bands 120 代 ES@−15 23.0/60.0（=冻结 p=0.25；比软落点早 74:38）→ 已入稿。mini 恶劣制度全链收官 09-12 20:45 PT。
- 09-12 20:55 PT 起 mini 跑 harsh1000.sh（−15 后 −10 dB 四系统 per-scen 100 → n=1000，skip 已有 200；~7h/制度；旗标 FO_HARSH1000_m15_DONE/m10_DONE/ALL_DONE，表 logs/fo_tables_harsh1000_m*.txt）。心跳 v6 bffi7tttn。
- 09-13 03:10 PT −15 dB n=1000：N 28.4/56.5｜Ri 24.3/61.2（340:174）｜s_ann 35.8/49.0（vs N 309:173 p=6e-10；vs Ri 432:139）｜rl 24.7/60.8（=Ri p=0.27；vs s_ann 421:154）→ 论文 −15 dB 段升到 n=1000。
- 09-13 06:20 PT −10 dB n=1000：N 32.9/51.2｜Ri 28.8/55.7（294:168）｜s_ann 36.9/47.1（vs N 240:151 p=8e-6；vs Ri 356:150）｜rl 27.1/57.6（vs Ri 223:173 p=0.014；vs s_ann 392:145）→ 论文 −10 dB 句、剂量句升到 n=1000；协议句 200 条例外仅剩 ES/自蒸馏变体、干净、探针行。FO_HARSH1000_ALL_DONE。
- 09-13 −5 dB 四系统 200 条：N 40.0/42.0｜Ri 33.5/49.5（51:32 p=0.048）｜s_ann 38.0/44.5（=N；vs Ri 52:35 p=0.09）｜rl 30.5/52.5（vs s_ann 58:31 p=0.006）。harsh1000.sh -5 在跑（FO_HARSH1000_m5_DONE / FO_M5_ALL_DONE）。备弹。
- 09-13 14:00 PT −5 dB n=1000：N 37.7/46.3｜Ri 32.4/51.9（252:139 p=1e-8）｜s_ann 38.2/46.1（=N p=0.72；vs Ri 249:138 p=2e-8）｜rl 30.8/53.5（vs Ri 175:127 p=0.007；vs s_ann 292:131 p=3e-15）。FO_M5_ALL_DONE；三剂量点四系统 n=1000 齐；−5 dB 入稿一句。表：logs/fo_tables_harsh1000_m5.txt。
- 09-13 14:20 PT 起 mini 跑 es_more1000.sh（full/120 代 ES@−15 扩到 n=1000，FO_ES_MORE1000_DONE，表 logs/fo_tables_es_more1000.txt）→ harsh.sh -20 -25 -15 → harsh1000.sh -20（FO_M20_ALL_DONE）。−20 dB 是剂量曲线远端边界探针，备弹。心跳 v9 bfefc694b。
- 09-13 16:40 PT ES 加强臂 n=1000：full 17.4/68.7（vs Ri 403:225 p=1e-12）；120 代 24.6/60.5（vs Ri 288:228 略晚 p=0.009；vs s_ann 403:180 p=1e-20）→ 已入稿，稿中不再有 200 条注脚（协议句例外仅 ES/自蒸馏干净变体、干净、探针行）。−20 dB 链在跑（FO_M20_ALL_DONE）。
- 09-13 −20 dB 200 条：N 22.0/63.5｜Ri 18.5/67.5（77:45 p=0.005）｜s_ann 28.5/55.5（vs N 75:39 p=1e-3；vs Ri 93:36 p=5e-7）｜rl 14.0/73.5（vs Ri 80:52 p=0.018；vs s_ann 108:40 p=2e-8）。harsh1000.sh -20 在跑。备弹。
- 09-13 23:00 PT −20 dB n=1000：N 23.1/63.0｜Ri 20.0/66.6（342:217 p=1e-7）｜s_ann 30.7/54.5（vs N 354:168 p=3e-16；vs Ri 453:146 p=2e-37）｜rl 16.3/70.9（vs Ri 375:272 p=6e-5；vs s_ann 535:189 p=4e-39）。FO_M20_ALL_DONE。四剂量点×四系统 n=1000 网格完整；mini 链全部结束。
- 09-14 12:10 PT 起 mini 跑 dense_es.sh：fo_train_rl.py 新增 --reward dense（软落点损失数值当 reward，零阶 ES，白盒才有的稠密信号）；bands 40 代 + full 40 代 @−15，各评 n=1000 → nz/fo_rl_dense_m15、fo_rl_dense_full_m15，表 logs/fo_tables_dense.txt，旗标 FO_DENSE_DONE。目的：把'失去梯度的价格'拆成奖励粗糙 vs 方向缺失。心跳 v10。
- 09-14 14:30 PT 稠密奖励 ES bands n=1000：25.9/59.5；vs Ri 172:120 p=0.003；vs rl（三档）262:237 p=0.28；vs s_ann 383:132 p=2e-29。结论：价格=方向，非奖励粗糙。full 稠密在跑。
- 09-14 16:40 PT 稠密 ES full n=1000：23.6/61.5（=Ri p=0.83；vs 黑盒 full 399:219；vs s_ann 431:164）。FO_DENSE_DONE。稿中 ES 段改为'the price is the direction, not the reward's resolution or the budget'。
- 09-14 17:10 PT 起 mini 跑 gen_text.sh：fo_gen_text.py（fo_placement 副本，触发后沿用流式状态生成首响文本 ≤48 token 或首句，写各树 text.json，含 prev_first 对拍）→ nz/fo_N_m15、fo_Ri_m15、fo_s_ann_m15（−15 dB，n=1000）；旗标 FO_GENTEXT_DONE。目的：任务级指标（首响内容相关性 LLM 评判）。心跳 v11。
- 09-15 任务级：三棵树首响文本 + Qwen 评判收官。相关性 N 0.825 / Ri 0.769 / s_ann 0.844；s_ann vs Ri 170:132 p=0.033；s_ann vs N p=0.95；中文幻觉 136/131/94（s_ann vs N 81:39 p=2e-4）。脚本 fo_gen_text.py / fo_judge_items.py / fo_judge_qwen.py。入稿一句。
- 09-15 决定：不跑 Lychee ES 臂（作业 33）。贡献 (1) 与摘要收窄到 Freeze-Omni（"Two regimes, one testbed (Freeze-Omni, 7B)"；摘要 "On one open model (Freeze-Omni) the two are compared directly"；"at the budgets tried"）；局限段加 "the black-box arm is run on Freeze-Omni only"。投稿候选版：5 页、References 第 4 页起、零 todo、bib 零警告。截稿 09-16。
- 09-15 通读修订：① 局限段"one model family fine-tuned"→"front-ends trained through two models"；② SI-SNR 句改为实测 "kept 6.4 of the frozen model's 8.6 dB gain on its training mixtures"（旧 13.2/15.4 不可复现；实测脚本内联于会话，冻结 +8.6/+8.5/+12.8 dB @U[−5,20]/5/U[−20,−10]，s_ann +6.4/+6.5/+11.3，s_ann_m15 +5.4/+5.3/+10.6）；③ \S IV-C→\ref{sec:abl}；④ VoiceChat 不再叫"second model"；⑤ 补干净音频半句（三前端 200 条均 n.s.）；⑥ "every wrong-place metric"→"wrong-place total 67→48% (p<1e-23)"（摘要/贡献/结论一致）；⑦ 删"at any scale"；⑧ 结论加 Freeze-Omni 一句，讨论段压缩。5 页合规。
