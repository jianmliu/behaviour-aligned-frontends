#!/bin/bash
# Reordered controls: (1) let the running DEMAND signal-control evaluation finish, (2) DeepFilterNet3 (gate + n=1000),
# (3) MUSAN signal control last. Same commands as sig_control.sh / dfn_control.sh.
cd ~/freeze_omni
HD=~/freeze_omni/data/humdial/unz/test/en_test_nondev
P=./venv/bin/python
until [ "$(find nz/fo_sig_m15 -name events.json | wc -l | tr -d ' ')" -ge 1000 ] && ! pgrep -f "fo_placemen[t].py.*fo_sig_m15" >/dev/null; do sleep 60; done
{ for t in fo_N_m15 fo_Ri_m15 fo_sig_m15 fo_s_ann_m15; do echo "== $t"; $P hm3_repo.py --root nz/$t --humdial $HD 2>&1 | tail -1; done
  cd nz; for b in fo_N_m15 fo_Ri_m15 fo_s_ann_m15; do ../venv/bin/python ../fo_signtest.py $b fo_sig_m15; done; cd ..
} > logs/fo_tables_sig_demand.txt
touch FO_SIG_DEMAND_DONE
# DeepFilterNet3
until [ -f mix/dfn3_ready ]; do sleep 60; done
rm -rf nz/gate_raw_m15
$P -u fo_placement.py --humdial $HD --per-scen 100 --limit 30 --pre-enhanced mix/demand_m15 --out nz/gate_raw_m15 > logs/dfn_gate.log 2>&1
$P - <<'PY' > logs/dfn_gate_check.txt 2>&1
import json, pathlib, sys
def first(p):
    ev=[e for e in json.loads(p.read_text()).get("events",[]) if e.get("type")=="response"]; return min(e["start_time"] for e in ev) if ev else None
g=pathlib.Path("nz/gate_raw_m15"); ref=pathlib.Path("nz/fo_N_m15"); n=bad=0
for p in g.rglob("events.json"):
    q=ref/p.relative_to(g); a,b=first(p),first(q); n+=1
    if (a is None)!=(b is None) or (a is not None and abs(a-b)>1e-6): bad+=1; print("mismatch",p.parent,a,b)
print(f"gate: {n} samples, {bad} mismatches"); sys.exit(1 if bad or n<30 else 0)
PY
if [ $? -ne 0 ]; then touch FO_DFN_GATE_FAILED; else
  $P -u fo_placement.py --humdial $HD --per-scen 100 --pre-enhanced enh/dfn3_demand_m15 --out nz/fo_dfn3_m15 > logs/dfn_ev.log 2>&1
  { for t in fo_N_m15 fo_Ri_m15 fo_sig_m15 fo_s_ann_m15 fo_dfn3_m15; do echo "== $t"; $P hm3_repo.py --root nz/$t --humdial $HD 2>&1 | tail -1; done
    cd nz; for b in fo_N_m15 fo_Ri_m15 fo_sig_m15 fo_s_ann_m15; do ../venv/bin/python ../fo_signtest.py $b fo_dfn3_m15; done; cd ..
  } > logs/fo_tables_dfn_control.txt
  touch FO_DFN_DONE
fi
# MUSAN signal control
COMMON="--targets nz/fo_fdb_ann --fdb data/fdb/v1.0/unz --init final.pt --snr -20 -10 --max-sec 12 --epochs 2"
$P -u fo_train_sig.py $COMMON --noise-dir data/musan/noise --noise-glob '*.wav' --out exp/mu_sig_m15 > logs/sig_train_musan.log 2>&1
$P fo_sisnr.py --fdb data/fdb/v1.0/unz --noise-dir data/musan/noise --noise-glob '*.wav' --snr -20 -10 final.pt exp/mu_sig_m15/epoch_002.pt exp/mu_s_ann_m15/epoch_002.pt > logs/sig_sisnr_musan.txt 2>&1
$P -u fo_placement.py --humdial $HD --per-scen 100 --noise-dir data/musan/noise --noise-glob '*.wav' --snr -15 -15 --out nz/mu_sig_tr_m15 --rnnoise exp/mu_sig_m15/epoch_002.pt > logs/sig_ev_musan.log 2>&1
{ for t in mu_N_m15 mu_Ri_m15 mu_sig_tr_m15 mu_s_ann_tr_m15; do echo "== $t"; $P hm3_repo.py --root nz/$t --humdial $HD 2>&1 | tail -1; done
  cd nz; for b in mu_N_m15 mu_Ri_m15 mu_s_ann_tr_m15; do ../venv/bin/python ../fo_signtest.py $b mu_sig_tr_m15; done; cd ..
} > logs/fo_tables_sig_musan.txt
touch FO_SIG_ALL_DONE
