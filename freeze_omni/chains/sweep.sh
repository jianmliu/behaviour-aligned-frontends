#!/bin/bash
# SNR 剂量扫描：找 Freeze-Omni 失败出现声学成分的区间（固定 SNR，每场景 20 条）
cd ~/freeze_omni
HD=~/freeze_omni/data/humdial/unz/test/en_test_nondev
for snr in -10 -5 -15; do
  tag=$(echo $snr | tr -d -); 
  ./venv/bin/python -u fo_placement.py --humdial $HD --per-scen 20 --out nz/fo_N_m${tag} --noise-dir data/demand --snr $snr $snr > logs/fo_N_m${tag}.log 2>&1
done
for t in fo_N_m10 fo_N_m5 fo_N_m15; do echo "== $t"; ./venv/bin/python humdial_metrics3.py --root nz/$t --humdial $HD 2>&1 | tail -14; done > logs/fo_tables_sweep.txt
echo DONE > logs/FO_SWEEP_DONE
