#!/usr/bin/env bash
# 噪声篇 05：HumDial 英文子集干净树物化（CPU）+ L0 推理（8 卡分片）
set -euo pipefail
export WORK=/ephemeral/work
[ -d $WORK/humdial/en_test_nondev ] || { echo "HumDial 未传，让队"; exit 75; }
source /ephemeral/work/venv/bin/activate
export PYTHONPATH=$WORK/geic_nemo/src:$WORK/nscripts
cd $WORK/geic_nemo && source scripts/gpu_guard.sh
gpu_lock
NG=$(nvidia-smi -L | wc -l)
python -u $WORK/nscripts/humdial_prep.py --humdial $WORK/humdial/en_test_nondev \
  --mode clean --out $WORK/nz/hd_clean --per-scen 100
pids=()
for i in $(seq 0 $((NG-1))); do
  CUDA_VISIBLE_DEVICES=$i python -u scripts/fdb_infer_lychee.py \
    --fdb_root $WORK/nz/hd_clean --lychee_repo $WORK/third_party/Lychee-FD \
    --model_path $WORK/models/lychee-fd/lychee_full_duplex \
    --out_root $WORK/nz/ev_L0 --shard $i --num-shards $NG \
    > $WORK/logs/nz_L0_$i.log 2>&1 & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || true; done
n=$(set +o pipefail; ls $WORK/nz/ev_L0/*/*/events.json 2>/dev/null | wc -l)
python -u $WORK/nscripts/humdial_metrics3.py --humdial $WORK/humdial/en_test_nondev --root $WORK/nz/ev_L0 --out $WORK/exp/nz/L0.json | tail -14
echo "=== L0 完成 $n ==="; [ "$n" -ge 950 ] || exit 1
