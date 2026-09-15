#!/usr/bin/env bash
# 噪声篇 30：R-joint 软落点 s0(克制倾向) 评测
set -euo pipefail
export WORK=/ephemeral/work
CK=$WORK/exp/rnj/s0/epoch_002.pt
[ -f "$CK" ] || { echo "R-joint 训练未完，让队"; exit 75; }
source /ephemeral/work/venv/bin/activate
export PYTHONPATH=$WORK/geic_nemo/src:$WORK/nscripts
cd $WORK/geic_nemo && source scripts/gpu_guard.sh
gpu_lock
# 训练 ckpt 是 {"fe": sd, ...} 包装；humdial_prep --rnnoise 吃裸 state_dict
python -c "import torch; torch.save(torch.load('$CK', map_location='cpu', weights_only=True)['fe'], '$WORK/exp/rnj/s0/fe_final.pt')"
NG=$(nvidia-smi -L | wc -l)
CUDA_VISIBLE_DEVICES=0 python -u $WORK/nscripts/humdial_prep.py --humdial $WORK/humdial/en_test_nondev \
  --mode noise --noise-dir $WORK/data/demand \
  --rnnoise $WORK/exp/rnj/s0/fe_final.pt --out $WORK/nz/hd_Rjoint_s0 --per-scen 100
pids=()
for i in $(seq 0 $((NG-1))); do
  CUDA_VISIBLE_DEVICES=$i python -u scripts/fdb_infer_lychee.py \
    --fdb_root $WORK/nz/hd_Rjoint_s0 --lychee_repo $WORK/third_party/Lychee-FD \
    --model_path $WORK/models/lychee-fd/lychee_full_duplex \
    --out_root $WORK/nz/ev_Rjoint_s0 --shard $i --num-shards $NG \
    > $WORK/logs/nz_Rjs0_$i.log 2>&1 & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || true; done
n=$(set +o pipefail; ls $WORK/nz/ev_Rjoint_s0/*/*/events.json 2>/dev/null | wc -l)
echo "ev_Rjoint_s0 完成 $n 条"
[ "$n" -ge 950 ]
