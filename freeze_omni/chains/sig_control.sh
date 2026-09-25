#!/bin/bash
# Matched-SNR signal-retrained control (-15 dB): DEMAND then MUSAN. Same data/steps/optimiser as the gradient arm; loss = -SI-SNR.
cd ~/freeze_omni
HD=~/freeze_omni/data/humdial/unz/test/en_test_nondev
P=./venv/bin/python
COMMON="--targets nz/fo_fdb_ann --fdb data/fdb/v1.0/unz --init final.pt --snr -20 -10 --max-sec 12 --epochs 2"
# DEMAND
$P -u fo_train_sig.py $COMMON --noise-dir data/demand --out exp/fo_sig_m15 > logs/sig_train_demand.log 2>&1
$P fo_sisnr.py --fdb data/fdb/v1.0/unz --noise-dir data/demand --snr -20 -10 final.pt exp/fo_sig_m15/epoch_002.pt exp/fo_s_ann_m15/epoch_002.pt > logs/sig_sisnr_demand.txt 2>&1
$P -u fo_placement.py --humdial $HD --per-scen 100 --noise-dir data/demand --snr -15 -15 --out nz/fo_sig_m15 --rnnoise exp/fo_sig_m15/epoch_002.pt > logs/sig_ev_demand.log 2>&1
touch FO_SIG_DEMAND_DONE
# MUSAN
$P -u fo_train_sig.py $COMMON --noise-dir data/musan/noise --noise-glob '*.wav' --out exp/mu_sig_m15 > logs/sig_train_musan.log 2>&1
$P fo_sisnr.py --fdb data/fdb/v1.0/unz --noise-dir data/musan/noise --noise-glob '*.wav' --snr -20 -10 final.pt exp/mu_sig_m15/epoch_002.pt exp/mu_s_ann_m15/epoch_002.pt > logs/sig_sisnr_musan.txt 2>&1
$P -u fo_placement.py --humdial $HD --per-scen 100 --noise-dir data/musan/noise --noise-glob '*.wav' --snr -15 -15 --out nz/mu_sig_tr_m15 --rnnoise exp/mu_sig_m15/epoch_002.pt > logs/sig_ev_musan.log 2>&1
# tables + sign tests
{ for t in fo_N_m15 fo_Ri_m15 fo_sig_m15 fo_s_ann_m15; do echo "== $t"; $P humdial_metrics3.py --root nz/$t --humdial $HD 2>&1 | tail -1; done
  cd nz; for b in fo_N_m15 fo_Ri_m15 fo_s_ann_m15; do ../venv/bin/python ../fo_signtest.py $b fo_sig_m15; done; cd ..
  for t in mu_N_m15 mu_Ri_m15 mu_sig_tr_m15 mu_s_ann_tr_m15; do echo "== $t"; $P humdial_metrics3.py --root nz/$t --humdial $HD 2>&1 | tail -1; done
  cd nz; for b in mu_N_m15 mu_Ri_m15 mu_s_ann_tr_m15; do ../venv/bin/python ../fo_signtest.py $b mu_sig_tr_m15; done; cd ..
} > logs/fo_tables_sig_control.txt
touch FO_SIG_ALL_DONE
