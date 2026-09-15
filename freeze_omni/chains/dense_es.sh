#!/bin/bash
# 白盒稠密奖励 ES @−15 dB：同一软落点成本的数值当 reward，优化器仍为零阶 ES；bands 与 full 各一次，评测 n=1000
cd ~/freeze_omni
HD=~/freeze_omni/data/humdial/unz/test/en_test_nondev
COMMON="--fdb data/fdb/v1.0/unz --noise-dir data/demand --init final.pt --snr -20 -10 --max-sec 12 --targets nz/fo_fdb_ann --reward dense"
EV="--humdial $HD --per-scen 100 --noise-dir data/demand --snr -15 -15"
./venv/bin/python -u fo_train_rl.py $COMMON --out exp/fo_rl_dense_m15 --space bands --pop 6 --m 6 --gens 40 > logs/es_dense.log 2>&1
./venv/bin/python -u fo_placement.py $EV --out nz/fo_rl_dense_m15 --rnnoise final.pt --policy exp/fo_rl_dense_m15/last.pt > logs/h1000_rl_dense_m15.log 2>&1
./venv/bin/python -u fo_train_rl.py $COMMON --out exp/fo_rl_dense_full_m15 --space full --pop 6 --m 6 --gens 40 --sigma 0.02 > logs/es_dense_full.log 2>&1
./venv/bin/python -u fo_placement.py $EV --out nz/fo_rl_dense_full_m15 --rnnoise final.pt --policy exp/fo_rl_dense_full_m15/last.pt > logs/h1000_rl_dense_full_m15.log 2>&1
for t in fo_rl_dense_m15 fo_rl_dense_full_m15; do echo "== $t"; ./venv/bin/python humdial_metrics3.py --root nz/$t --humdial $HD 2>&1 | tail -14; done > logs/fo_tables_dense.txt
touch FO_DENSE_DONE
