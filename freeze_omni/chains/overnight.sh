#!/bin/bash
cd ~/freeze_omni
HD=~/freeze_omni/data/humdial/unz/test/en_test_nondev
until [ -f logs/FO_STAGE2_DONE ]; do sleep 120; done
if ! ./venv/bin/python - << "PY"
import re, sys
ok = True; n = 0
for line in open("logs/fo_check.log", errors="ignore"):
    m = re.search(r"mean=([0-9.]+) max=([0-9.]+)\s+first>0.5 stream=(\S+) full=(\S+)", line)
    if m:
        n += 1
        if float(m.group(1)) > 0.15 or m.group(3) != m.group(4): ok = False   # 决策一致 + 均值Δ
print("check", "PASS" if ok and n else "FAIL", n); sys.exit(0 if ok and n else 1)
PY
then echo "GATE_FAIL_CHECK" > logs/OVERNIGHT_GATE; exit 1; fi
grep -a -q "\[smoke\] 3 步完成" logs/fo_train_smoke.log || { echo "GATE_FAIL_SMOKE" > logs/OVERNIGHT_GATE; exit 1; }
echo GATE_PASS > logs/OVERNIGHT_GATE
COMMON="--fdb data/fdb/v1.0/unz --noise-dir data/demand --init final.pt"
EV="--humdial $HD --per-scen 20 --noise-dir data/demand"
./venv/bin/python -u fo_train_soft.py $COMMON --targets nz/fo_fdb_ann --out exp/fo_s_ann --w-pre 3 --w-miss 1 --epochs 2 --max-sec 12 --pre-ramp 20 --w-margin 1.0 > logs/fo_s_ann.log 2>&1
./venv/bin/python -u fo_placement.py $EV --out nz/fo_s_ann --rnnoise exp/fo_s_ann/epoch_002.pt > logs/fo_eval_s_ann.log 2>&1
./venv/bin/python -u fo_train_rl.py $COMMON --targets nz/fo_fdb_ann --out exp/fo_rl_ann --space bands --pop 6 --m 6 --gens 40 --max-sec 12 > logs/fo_rl.log 2>&1
./venv/bin/python -u fo_placement.py $EV --out nz/fo_rl_ann --rnnoise final.pt --policy exp/fo_rl_ann/best.pt > logs/fo_eval_rl.log 2>&1
./venv/bin/python -u fo_train_soft.py $COMMON --targets nz/fo_fdb_clean --out exp/fo_s_self --w-pre 3 --w-miss 1 --epochs 2 --max-sec 12 --pre-ramp 20 --w-margin 1.0 > logs/fo_s_self.log 2>&1
./venv/bin/python -u fo_placement.py $EV --out nz/fo_s_self --rnnoise exp/fo_s_self/epoch_002.pt > logs/fo_eval_s_self.log 2>&1
for t in fo_L0 fo_N0 fo_Rindep fo_s_ann fo_rl_ann fo_s_self; do echo "== $t"; ./venv/bin/python humdial_metrics3.py --root nz/$t --humdial $HD 2>&1 | tail -3; done > logs/fo_tables_night.txt
echo DONE > logs/FO_NIGHT_DONE
