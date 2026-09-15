#!/bin/bash
# 恶劣制度四系统扩到 n=1000（per-scen 100；前 20 条已存在自动 skip）。用法 ./harsh1000.sh <SNR>
cd ~/freeze_omni
SE=$1; TAG=$(echo $SE | tr -d -)
HD=~/freeze_omni/data/humdial/unz/test/en_test_nondev
EV="--humdial $HD --per-scen 100 --noise-dir data/demand --snr $SE $SE"
./venv/bin/python -u fo_placement.py $EV --out nz/fo_N_m$TAG > logs/h1000_N_m$TAG.log 2>&1
./venv/bin/python -u fo_placement.py $EV --out nz/fo_Ri_m$TAG --rnnoise final.pt > logs/h1000_Ri_m$TAG.log 2>&1
./venv/bin/python -u fo_placement.py $EV --out nz/fo_s_ann_m$TAG --rnnoise exp/fo_s_ann_m$TAG/epoch_002.pt > logs/h1000_s_ann_m$TAG.log 2>&1
./venv/bin/python -u fo_placement.py $EV --out nz/fo_rl_m$TAG --rnnoise final.pt --policy exp/fo_rl_m$TAG/last.pt > logs/h1000_rl_m$TAG.log 2>&1
for t in fo_N_m$TAG fo_Ri_m$TAG fo_s_ann_m$TAG fo_rl_m$TAG; do echo "== $t"; ./venv/bin/python humdial_metrics3.py --root nz/$t --humdial $HD 2>&1 | tail -14; done > logs/fo_tables_harsh1000_m$TAG.txt
touch FO_HARSH1000_m${TAG}_DONE
