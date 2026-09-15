#!/bin/bash
# 恶劣噪声制度下的两制度对照：用法 ./harsh.sh <SNR_eval> <SNR_train_lo> <SNR_train_hi>
# 例：./harsh.sh -10 -15 -5  → 训练噪声 U[-15,-5]，评测固定 -10 dB；冻结/软落点(标注)/ES 三前端 + 无前端基线
cd ~/freeze_omni
SE=$1; TL=$2; TH=$3; TAG=$(echo $SE | tr -d -)
HD=~/freeze_omni/data/humdial/unz/test/en_test_nondev
COMMON="--fdb data/fdb/v1.0/unz --noise-dir data/demand --init final.pt --snr $TL $TH --max-sec 12"
EV="--humdial $HD --per-scen 20 --noise-dir data/demand --snr $SE $SE"
[ -d nz/fo_N_m$TAG ] || ./venv/bin/python -u fo_placement.py $EV --out nz/fo_N_m$TAG > logs/h_N.log 2>&1
./venv/bin/python -u fo_placement.py $EV --out nz/fo_Ri_m$TAG --rnnoise final.pt > logs/h_Ri.log 2>&1
./venv/bin/python -u fo_train_soft.py $COMMON --targets nz/fo_fdb_ann --out exp/fo_s_ann_m$TAG --w-pre 3 --w-miss 1 --epochs 2 --pre-ramp 20 --w-margin 1.0 > logs/h_s_ann.log 2>&1
./venv/bin/python -u fo_placement.py $EV --out nz/fo_s_ann_m$TAG --rnnoise exp/fo_s_ann_m$TAG/epoch_002.pt > logs/h_ev_s_ann.log 2>&1
./venv/bin/python -u fo_train_rl.py $COMMON --targets nz/fo_fdb_ann --out exp/fo_rl_m$TAG --space bands --pop 6 --m 6 --gens 40 > logs/h_rl.log 2>&1
./venv/bin/python -u fo_placement.py $EV --out nz/fo_rl_m$TAG --rnnoise final.pt --policy exp/fo_rl_m$TAG/last.pt > logs/h_ev_rl.log 2>&1
for t in fo_N_m$TAG fo_Ri_m$TAG fo_s_ann_m$TAG fo_rl_m$TAG; do echo "== $t"; ./venv/bin/python humdial_metrics3.py --root nz/$t --humdial $HD 2>&1 | tail -14; done > logs/fo_tables_harsh_m$TAG.txt
echo DONE > logs/FO_HARSH_m${TAG}_DONE
