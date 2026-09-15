#!/bin/bash
cd ~/freeze_omni
HD=~/freeze_omni/data/humdial/unz/test/en_test_nondev
until [ -f logs/FO_NIGHT_DONE ]; do sleep 120; done
./venv/bin/python -u fo_placement.py --humdial $HD --per-scen 20 --out nz/fo_s_ann_clean --rnnoise exp/fo_s_ann/epoch_002.pt > logs/fo_eval_s_ann_clean.log 2>&1
./venv/bin/python -u fo_placement.py --humdial $HD --per-scen 20 --out nz/fo_rl_ann_clean --rnnoise final.pt --policy exp/fo_rl_ann/best.pt > logs/fo_eval_rl_ann_clean.log 2>&1
./venv/bin/python -u fo_placement.py --humdial $HD --per-scen 20 --out nz/fo_Rindep_clean --rnnoise final.pt > logs/fo_eval_Rindep_clean.log 2>&1
for t in fo_s_ann_clean fo_rl_ann_clean fo_Rindep_clean; do echo "== $t"; ./venv/bin/python humdial_metrics3.py --root nz/$t --humdial $HD 2>&1 | tail -3; done > logs/fo_tables_clean.txt
echo DONE > logs/FO_NIGHT2_DONE
