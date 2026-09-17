#!/bin/bash
# 第二噪声语料（MUSAN noise）交叉复现 @−15 dB，n=1000：DEMAND 上训好的四系统直接换噪声评测（泛化检验）
cd ~/freeze_omni
HD=~/freeze_omni/data/humdial/unz/test/en_test_nondev
EV="--humdial $HD --per-scen 100 --noise-dir data/musan/noise --noise-glob *.wav --snr -15 -15"
./venv/bin/python -u fo_placement.py $EV --out nz/mu_N_m15 > logs/mu_N.log 2>&1
./venv/bin/python -u fo_placement.py $EV --out nz/mu_Ri_m15 --rnnoise final.pt > logs/mu_Ri.log 2>&1
./venv/bin/python -u fo_placement.py $EV --out nz/mu_s_ann_m15 --rnnoise exp/fo_s_ann_m15/epoch_002.pt > logs/mu_s_ann.log 2>&1
./venv/bin/python -u fo_placement.py $EV --out nz/mu_rl_m15 --rnnoise final.pt --policy exp/fo_rl_m15/last.pt > logs/mu_rl.log 2>&1
for t in mu_N_m15 mu_Ri_m15 mu_s_ann_m15 mu_rl_m15; do echo "== $t"; ./venv/bin/python humdial_metrics3.py --root nz/$t --humdial $HD 2>&1 | tail -14; done > logs/fo_tables_musan_xcorpus.txt
touch FO_MUSAN_XCORPUS_DONE
