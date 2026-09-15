#!/bin/bash
# −15 dB 三棵树（无前端/冻结/梯度）首响文本生成（写入各树 text.json，events.json 不动）
cd ~/freeze_omni
HD=~/freeze_omni/data/humdial/unz/test/en_test_nondev
EV="--humdial $HD --per-scen 100 --noise-dir data/demand --snr -15 -15 --max-tokens 48"
./venv/bin/python -u fo_gen_text.py $EV --out nz/fo_N_m15 > logs/gen_N_m15.log 2>&1
./venv/bin/python -u fo_gen_text.py $EV --out nz/fo_Ri_m15 --rnnoise final.pt > logs/gen_Ri_m15.log 2>&1
./venv/bin/python -u fo_gen_text.py $EV --out nz/fo_s_ann_m15 --rnnoise exp/fo_s_ann_m15/epoch_002.pt > logs/gen_s_ann_m15.log 2>&1
touch FO_GENTEXT_DONE
