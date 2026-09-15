#!/usr/bin/env bash
# 噪声篇 33：Lychee 黑盒 ES（bands 策略，标注目标 reward）训练 + 评测 —— 与 Freeze-Omni fo_rl_ann 同配方
set -euo pipefail
export WORK=/ephemeral/work
[ -d $WORK/nz/fdb_ann ] || { echo "标注目标树未建（32 先跑），让队"; exit 75; }
source /ephemeral/work/venv/bin/activate
export PYTHONPATH=$WORK/geic_nemo/src:$WORK/nscripts
cd $WORK/nscripts && source $WORK/geic_nemo/scripts/gpu_guard.sh
gpu_lock
python -u train_rnnoise_es.py --lychee-repo $WORK/third_party/Lychee-FD --lychee-ckpt $WORK/models/lychee-fd/lychee_full_duplex \
  --fdb-root $WORK/data/fdb/v1.0 --targets $WORK/nz/fdb_ann --noise-dir $WORK/data/demand \
  --init $WORK/rnnoise_pretrain/final.pt --out $WORK/exp/rnj_es_bands --space bands --pop 6 --m 6 --gens 30 > $WORK/logs/rnj_es.log 2>&1
cd $WORK/geic_nemo
python -u $WORK/nscripts/humdial_prep.py --humdial $WORK/humdial/en_test_nondev --mode noise --noise-dir $WORK/data/demand \
  --rnnoise $WORK/rnnoise_pretrain/final.pt --policy $WORK/exp/rnj_es_bands/best.pt --out $WORK/nz/hd_es --per-scen 100
NG=$(nvidia-smi -L | wc -l); pids=()
for i in $(seq 0 $((NG-1))); do
  CUDA_VISIBLE_DEVICES=$i python -u scripts/fdb_infer_lychee.py --fdb_root $WORK/nz/hd_es --lychee_repo $WORK/third_party/Lychee-FD \
    --model_path $WORK/models/lychee-fd/lychee_full_duplex --out_root $WORK/nz/ev_es --shard $i --num-shards $NG > $WORK/logs/nz_es_$i.log 2>&1 & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || true; done
n=$(set +o pipefail; ls $WORK/nz/ev_es/*/*/events.json 2>/dev/null | wc -l); echo "ev_es $n"; [ "$n" -ge 950 ]
