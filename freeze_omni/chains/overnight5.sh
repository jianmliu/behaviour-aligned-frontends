#!/bin/bash
# 链 5：四棵关键树扩到 100/场景（fo_placement 断点续跑，已有 20/场景自动跳过）
cd ~/freeze_omni
HD=~/freeze_omni/data/humdial/unz/test/en_test_nondev
./venv/bin/python -u fo_placement.py --humdial $HD --per-scen 100 --out nz/fo_L0 > logs/fo_L0_1000.log 2>&1
./venv/bin/python -u fo_placement.py --humdial $HD --per-scen 100 --out nz/fo_N0 --noise-dir data/demand > logs/fo_N0_1000.log 2>&1
./venv/bin/python -u fo_placement.py --humdial $HD --per-scen 100 --out nz/fo_s_ann --noise-dir data/demand --rnnoise exp/fo_s_ann/epoch_002.pt > logs/fo_s_ann_1000.log 2>&1
./venv/bin/python -u fo_placement.py --humdial $HD --per-scen 100 --out nz/fo_Rindep --noise-dir data/demand --rnnoise final.pt > logs/fo_Rindep_1000.log 2>&1
for t in fo_L0 fo_N0 fo_Rindep fo_s_ann; do echo "== $t"; ./venv/bin/python humdial_metrics3.py --root nz/$t --humdial $HD 2>&1 | tail -14; done > logs/fo_tables_1000.txt
echo DONE > logs/FO_NIGHT5_DONE
