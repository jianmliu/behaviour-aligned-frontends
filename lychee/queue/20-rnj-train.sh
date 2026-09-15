#!/usr/bin/env bash
# 噪声篇 20：R-joint 训练矩阵（卡0-2 三配置）+ 其余卡留给消融/评测
set -euo pipefail
export WORK=/ephemeral/work
[ -f $WORK/rnnoise_pretrain/final.pt ] || { echo "预训练 ckpt 未传，让队"; exit 75; }
n=$(set +o pipefail; ls $WORK/fdb_out/LycheeFD_clean/*/*/events.json 2>/dev/null | wc -l); [ "$n" -ge 700 ] || { echo "FDB 轨迹未传，让队"; exit 75; }
source /ephemeral/work/venv/bin/activate
export PYTHONPATH=$WORK/geic_nemo/src:$WORK/nscripts
cd $WORK/nscripts && source $WORK/geic_nemo/scripts/gpu_guard.sh
gpu_lock
NG=$(nvidia-smi -L | wc -l)
COMMON="--lychee-repo $WORK/third_party/Lychee-FD --lychee-ckpt $WORK/models/lychee-fd/lychee_full_duplex \
  --traces $WORK/fdb_out/LycheeFD_clean --fdb-root $WORK/data/fdb/v1.0 \
  --noise-dir $WORK/data/demand --epochs 2 --seed 0"
declare -A CFG
CFG[0]="--rnnoise-init $WORK/rnnoise_pretrain/final.pt --lr 1e-4"
CFG[1]="--rnnoise-init $WORK/rnnoise_pretrain/final.pt --lr 3e-5"
CFG[2]="--lr 1e-4"   # scratch 对照
# 单卡时只跑主配置 r0（变体等主结果后按需补）；多卡时全并行
[ "$NG" -ge 3 ] && RUNS="0 1 2" || RUNS="0"
pids=()
for i in $RUNS; do
  CUDA_VISIBLE_DEVICES=$((i % NG)) python -u train_rnnoise_joint.py $COMMON ${CFG[$i]} \
    --out-dir $WORK/exp/rnj/r$i > $WORK/logs/rnj_r$i.log 2>&1 & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || true; done
ls $WORK/exp/rnj/r*/epoch_002.pt 2>/dev/null | wc -l
