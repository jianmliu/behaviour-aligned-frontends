#!/bin/bash
# 第二噪声语料（MUSAN noise）上重训两制度 @−15 dB（训练噪声 U[−20,−10] 来自 MUSAN），评测 n=1000
cd ~/freeze_omni
HD=~/freeze_omni/data/humdial/unz/test/en_test_nondev
COMMON="--fdb data/fdb/v1.0/unz --noise-dir data/musan/noise --noise-glob *.wav --init final.pt --snr -20 -10 --max-sec 12 --targets nz/fo_fdb_ann"
EV="--humdial $HD --per-scen 100 --noise-dir data/musan/noise --noise-glob *.wav --snr -15 -15"
./venv/bin/python -u fo_train_soft.py $COMMON --out exp/mu_s_ann_m15 --w-pre 3 --w-miss 1 --epochs 2 --pre-ramp 20 --w-margin 1.0 > logs/mu_train_s_ann.log 2>&1
./venv/bin/python -u fo_placement.py $EV --out nz/mu_s_ann_tr_m15 --rnnoise exp/mu_s_ann_m15/epoch_002.pt > logs/mu_ev_s_ann_tr.log 2>&1
./venv/bin/python -u fo_train_rl.py $COMMON --out exp/mu_rl_m15 --space bands --pop 6 --m 6 --gens 40 > logs/mu_train_rl.log 2>&1
./venv/bin/python -u fo_placement.py $EV --out nz/mu_rl_tr_m15 --rnnoise final.pt --policy exp/mu_rl_m15/last.pt > logs/mu_ev_rl_tr.log 2>&1
for t in mu_N_m15 mu_Ri_m15 mu_s_ann_tr_m15 mu_rl_tr_m15; do echo "== $t"; ./venv/bin/python humdial_metrics3.py --root nz/$t --humdial $HD 2>&1 | tail -14; done > logs/fo_tables_musan_retrain.txt
touch FO_MUSAN_RETRAIN_DONE
