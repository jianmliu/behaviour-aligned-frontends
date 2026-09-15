# 噪声篇计划（ICASSP 2027 主投，2026-08-29 立项）

## 一句话
在 HumDial-FDBench（官方 Test）上量化噪声对全双工行为的破坏（假触发病理），
并用行为目标（Lychee control-CE）联合调试 RNNoise-style 前端；
对照 = 独立训练的同一前端 / 无前端。

## 为什么这次联合优化胜算高（vs 回声篇教训）
噪声是**外生**的——没有"模型输出回到输入"的反馈环，teacher forcing 的曝光鸿沟
（回声篇三大机理中最深的一个）在此天然不存在；"噪声时别假触发"是普通条件分类，
teacher-forced control-CE 的监督覆盖充分。前端仅 78K 参数（假设空间小，代理退化空间也小）。

## 系统矩阵（刻意小）
| 行 | 说明 |
|---|---|
| L0 | 干净 HumDial-Test（对表其排行榜行为口径） |
| N0 | +噪声，无前端（崩坏基线；噪声谱系：稳态 DEMAND/瞬态 DNS 类/背景语音） |
| R-indep | RNNoise-torch，信号 loss 预训练（DNS 风格数据），冻结 |
| R-joint | 同一预训练起点 → control-CE 穿冻结 Lychee 微调（回声篇 E41 配方） |
| 消融 | pitch 特征、λ_text、瞬态 vs 稳态分噪声型报告 |

## 数据
- 评测：HumDial Track2-Test（HF 公开 zip，官方测试集）
- 训练输入：FDB 727 用户音频（跨语料零污染）±公开语料扩充;
  训练集邮件申请并行（asd6404112a@mail.nwpu.edu.cn）,拿到则升级
- 噪声库：DEMAND（稳态）+ 瞬态类（键盘/关门等,DNS/freesound）——瞬态正是
  HumDial 自己点名的假触发源（其论文原话 "transient noises causing false activations"）

## 资产复用（全部现成）
- Lychee 推理/评测链（fdb_infer_lychee/lychee_event_metrics/by_subset）
- 联合训练管线（train_lychee_joint + lychee_hook,前端模块换 RNNoiseTorch）
- 误差棒方法、G1/G2 打法、bootstrap_v2、坑手册 B7
- rnnoise_torch.py：77.9K 参数,保长+可微 ✓（scratchpad,待入 src）

## 18 天倒排（截稿 9-16）
D1-2 HumDial test 结构核实+噪声协议; RNNoise 信号 loss 预训练（本地/GPU）
D3-4 开机：L0/N0 行为测量（G1/G2 打法）
D5-7 R-joint 训练矩阵+评测
D8-16 写作 paper_noise + 补实验;回声篇资产转第二篇（Interspeech 2027,
     系统谱系 Speex+RNNoise → Speex+joint-RNNoise → LC-AEC）

## 与回声篇（paper_mono）的关系
paper_mono 暂停投 ICASSP,其全部内容+P1b 两实验转入第二篇;
噪声篇成为 ICASSP 主投。已确认的跨模型/阶梯结论在第二篇兑现。
