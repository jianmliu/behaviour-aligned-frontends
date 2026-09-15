# 噪声篇（ICASSP 2027）复现包索引

- 协议与指标：`scripts/humdial_prep.py`（HumDial 加噪/增强物化，按样本种子）、`scripts/humdial_metrics3.py`（首响落点口径）
- Lychee-FD 线：`geic_nemo/scripts/fdb_infer_lychee.py`（离线全双工解码）、`scripts/train_rnnoise_joint.py`
  （control-CE / 软落点 `--loss-mode soft` / SI-SNR 锚 `--lambda-sig`）、`scripts/train_rnnoise_es.py`（黑盒 ES）、
  `scripts/rnnoise_torch.py`（78K 前端，含带增益策略钩子）、`scripts/soft_placement.py`（事件级 loss + 单测）
- Freeze-Omni 线（Mac mini/MPS）：`scripts/fo_placement.py`（听态落点评测 + 探针）、`scripts/fo_soft_forward.py`
  （可微整句前向 + 流式对拍）、`scripts/fo_train_soft.py`、`scripts/fo_train_rl.py`、`scripts/fdb_targets.py`
- 作业队列：本目录 `05…33-*.sh`（H100）；mini 过夜链 `~/freeze_omni/overnight{,2,3,5}.sh`（备份于
  `~/rspeech_lychee_backup/freeze_omni/scripts/`）
- 上机手册：`RESUME_SOFT.md`；论文：`paper_noise/main.tex`
