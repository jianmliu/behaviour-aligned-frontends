#!/usr/bin/env bash
# 噪声篇 32：Lychee 软落点·标注目标（与 Freeze-Omni fo_s_ann 同配方）训练 + 评测
set -euo pipefail
export WORK=/ephemeral/work
[ -f $WORK/rnnoise_pretrain/final.pt ] || { echo "预训练 ckpt 未传，让队"; exit 75; }
source /ephemeral/work/venv/bin/activate
export PYTHONPATH=$WORK/geic_nemo/src:$WORK/nscripts
cd $WORK/nscripts && source $WORK/geic_nemo/scripts/gpu_guard.sh
gpu_lock
python fdb_targets.py --fdb $WORK/data/fdb/v1.0 --out $WORK/nz/fdb_ann
COMMON="--lychee-repo $WORK/third_party/Lychee-FD --lychee-ckpt $WORK/models/lychee-fd/lychee_full_duplex \
  --traces $WORK/fdb_out/LycheeFD_clean --fdb-root $WORK/data/fdb/v1.0 --noise-dir $WORK/data/demand \
  --epochs 2 --seed 0 --rnnoise-init $WORK/rnnoise_pretrain/final.pt --lr 1e-4 --loss-mode soft --lambda-ce 0.1 --pre-ramp 5 \
  --targets-dir $WORK/nz/fdb_ann"
python -u train_rnnoise_joint.py $COMMON --max-steps 2 --out-dir /tmp/soft_ann_smoke
python -u train_rnnoise_joint.py $COMMON --w-pre 3.0 --w-miss 1.0 --out-dir $WORK/exp/rnj/s_ann > $WORK/logs/rnj_s_ann.log 2>&1
python -c "import torch; torch.save(torch.load('$WORK/exp/rnj/s_ann/epoch_002.pt', map_location='cpu', weights_only=True)['fe'], '$WORK/exp/rnj/s_ann/fe_final.pt')"
cd $WORK/geic_nemo
python -u $WORK/nscripts/humdial_prep.py --humdial $WORK/humdial/en_test_nondev --mode noise --noise-dir $WORK/data/demand \
  --rnnoise $WORK/exp/rnj/s_ann/fe_final.pt --out $WORK/nz/hd_s_ann --per-scen 100
NG=$(nvidia-smi -L | wc -l); pids=()
for i in $(seq 0 $((NG-1))); do
  CUDA_VISIBLE_DEVICES=$i python -u scripts/fdb_infer_lychee.py --fdb_root $WORK/nz/hd_s_ann --lychee_repo $WORK/third_party/Lychee-FD \
    --model_path $WORK/models/lychee-fd/lychee_full_duplex --out_root $WORK/nz/ev_s_ann --shard $i --num-shards $NG > $WORK/logs/nz_sann_$i.log 2>&1 & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || true; done
n=$(set +o pipefail; ls $WORK/nz/ev_s_ann/*/*/events.json 2>/dev/null | wc -l); echo "ev_s_ann $n"; [ "$n" -ge 950 ]
