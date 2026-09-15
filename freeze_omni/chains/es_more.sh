#!/bin/bash
# ES 加强臂（−15 dB）：全参数空间同预算 / 23 维策略 3 倍预算
cd ~/freeze_omni
HD=~/freeze_omni/data/humdial/unz/test/en_test_nondev
until [ -f logs/FO_HARSH_ALL_DONE ]; do sleep 120; done
COMMON="--fdb data/fdb/v1.0/unz --noise-dir data/demand --init final.pt --snr -20 -10 --max-sec 12 --targets nz/fo_fdb_ann"
EV="--humdial $HD --per-scen 20 --noise-dir data/demand --snr -15 -15"
./venv/bin/python -u fo_train_rl.py $COMMON --out exp/fo_rl_full_m15 --space full --pop 6 --m 6 --gens 40 --sigma 0.02 > logs/es_full.log 2>&1
./venv/bin/python -u fo_placement.py $EV --out nz/fo_rl_full_m15 --rnnoise final.pt --policy exp/fo_rl_full_m15/last.pt > logs/es_full_ev.log 2>&1
./venv/bin/python -u fo_train_rl.py $COMMON --out exp/fo_rl_long_m15 --space bands --pop 6 --m 6 --gens 120 > logs/es_long.log 2>&1
./venv/bin/python -u fo_placement.py $EV --out nz/fo_rl_long_m15 --rnnoise final.pt --policy exp/fo_rl_long_m15/last.pt > logs/es_long_ev.log 2>&1
for t in fo_rl_full_m15 fo_rl_long_m15; do echo "== $t"; ./venv/bin/python humdial_metrics3.py --root nz/$t --humdial $HD 2>&1 | tail -3; done > logs/fo_tables_es_more.txt
echo DONE > logs/FO_ES_MORE_DONE
