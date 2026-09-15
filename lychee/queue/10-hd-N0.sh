#!/usr/bin/env bash
# 噪声篇 10：N0（加噪无前端）物化+推理+指标（含与 L0 配对）
set -euo pipefail
export WORK=/ephemeral/work
n=$(set +o pipefail; ls $WORK/nz/ev_L0/*/*/events.json 2>/dev/null | wc -l); [ "$n" -ge 950 ] || { echo "L0 未完，让队"; exit 75; }
source /ephemeral/work/venv/bin/activate
export PYTHONPATH=$WORK/geic_nemo/src:$WORK/nscripts
cd $WORK/geic_nemo && source scripts/gpu_guard.sh
gpu_lock
NG=$(nvidia-smi -L | wc -l)
python -u $WORK/nscripts/humdial_prep.py --noise-dir $WORK/data/demand --humdial $WORK/humdial/en_test_nondev \
  --mode noise --out $WORK/nz/hd_N0 --per-scen 100
pids=()
for i in $(seq 0 $((NG-1))); do
  CUDA_VISIBLE_DEVICES=$i python -u scripts/fdb_infer_lychee.py \
    --fdb_root $WORK/nz/hd_N0 --lychee_repo $WORK/third_party/Lychee-FD \
    --model_path $WORK/models/lychee-fd/lychee_full_duplex \
    --out_root $WORK/nz/ev_N0 --shard $i --num-shards $NG \
    > $WORK/logs/nz_N0_$i.log 2>&1 & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || true; done
python -u $WORK/nscripts/humdial_metrics3.py --humdial $WORK/humdial/en_test_nondev --root $WORK/nz/ev_N0  --out $WORK/exp/nz/N0.json | tail -16
