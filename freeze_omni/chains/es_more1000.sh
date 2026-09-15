#!/bin/bash
# ES 加强臂（full 空间 / bands 120 代）@−15 dB 扩到 n=1000（skip 已有 200）
cd ~/freeze_omni
HD=~/freeze_omni/data/humdial/unz/test/en_test_nondev
EV="--humdial $HD --per-scen 100 --noise-dir data/demand --snr -15 -15"
./venv/bin/python -u fo_placement.py $EV --out nz/fo_rl_full_m15 --rnnoise final.pt --policy exp/fo_rl_full_m15/last.pt > logs/h1000_rl_full_m15.log 2>&1
./venv/bin/python -u fo_placement.py $EV --out nz/fo_rl_long_m15 --rnnoise final.pt --policy exp/fo_rl_long_m15/last.pt > logs/h1000_rl_long_m15.log 2>&1
for t in fo_rl_full_m15 fo_rl_long_m15; do echo "== $t"; ./venv/bin/python humdial_metrics3.py --root nz/$t --humdial $HD 2>&1 | tail -14; done > logs/fo_tables_es_more1000.txt
touch FO_ES_MORE1000_DONE
