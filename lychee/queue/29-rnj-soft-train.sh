#!/usr/bin/env bash
# 噪声篇 29：软落点可微 loss 训练（RL 前哨——事件级目标能否打开 CE 打不开的中间地带）
# 两个操作点：s0 克制倾向(w_pre3/w_miss1)、s1 可听性倾向(w_pre1/w_miss2)。
# 内建护栏：--max-steps 2 冒烟（含 verify_shift 首步校验），过了才真训。
set -euo pipefail
export WORK=/ephemeral/work
[ -f $WORK/rnnoise_pretrain/final.pt ] || { echo "预训练 ckpt 未传，让队"; exit 75; }
n=$(set +o pipefail; ls $WORK/fdb_out/LycheeFD_clean/*/*/events.json 2>/dev/null | wc -l); [ "$n" -ge 700 ] || { echo "FDB 轨迹未传，让队"; exit 75; }
source /ephemeral/work/venv/bin/activate
export PYTHONPATH=$WORK/geic_nemo/src:$WORK/nscripts
cd $WORK/nscripts && source $WORK/geic_nemo/scripts/gpu_guard.sh
gpu_lock
COMMON="--lychee-repo $WORK/third_party/Lychee-FD --lychee-ckpt $WORK/models/lychee-fd/lychee_full_duplex \
  --traces $WORK/fdb_out/LycheeFD_clean --fdb-root $WORK/data/fdb/v1.0 \
  --noise-dir $WORK/data/demand --epochs 2 --seed 0 \
  --rnnoise-init $WORK/rnnoise_pretrain/final.pt --lr 1e-4 --loss-mode soft --lambda-ce 0.1"
# 冒烟：2 步出 loss + shift 校验（校验失败脚本自 exit 非零 → 本作业进 failed）
python -u train_rnnoise_joint.py $COMMON --max-steps 2 --out-dir /tmp/soft_smoke
echo "== 冒烟通过，开训 =="
python -u train_rnnoise_joint.py $COMMON --w-pre 3.0 --w-miss 1.0 \
  --out-dir $WORK/exp/rnj/s0 > $WORK/logs/rnj_s0.log 2>&1
python -u train_rnnoise_joint.py $COMMON --w-pre 1.0 --w-miss 2.0 \
  --out-dir $WORK/exp/rnj/s1 > $WORK/logs/rnj_s1.log 2>&1
ls $WORK/exp/rnj/s0/epoch_002.pt $WORK/exp/rnj/s1/epoch_002.pt
