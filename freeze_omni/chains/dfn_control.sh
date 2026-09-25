#!/bin/bash
# DeepFilterNet3 control (-15 dB DEMAND, n=1000): waits for the signal-control chain and for the enhanced audio,
# gates on exact reproduction of the no-front-end run from the exported raw mixtures, then evaluates DFN3 output.
cd ~/freeze_omni
HD=~/freeze_omni/data/humdial/unz/test/en_test_nondev
P=./venv/bin/python
until [ -f FO_SIG_ALL_DONE ]; do sleep 60; done
until [ -f mix/dfn3_ready ]; do sleep 60; done
rm -rf nz/gate_raw_m15
$P -u fo_placement.py --humdial $HD --per-scen 100 --limit 30 --pre-enhanced mix/demand_m15 --out nz/gate_raw_m15 > logs/dfn_gate.log 2>&1
$P - <<'PY' > logs/dfn_gate_check.txt 2>&1 || { echo GATE_FAILED; touch FO_DFN_GATE_FAILED; exit 1; }
import json, pathlib, sys
def first(p):
    ev=[e for e in json.loads(p.read_text()).get("events",[]) if e.get("type")=="response"]; return min(e["start_time"] for e in ev) if ev else None
g=pathlib.Path("nz/gate_raw_m15"); ref=pathlib.Path("nz/fo_N_m15"); n=bad=0
for p in g.rglob("events.json"):
    q=ref/p.relative_to(g); a,b=first(p),first(q); n+=1
    if (a is None)!=(b is None) or (a is not None and abs(a-b)>1e-6): bad+=1; print("mismatch",p.parent,a,b)
print(f"gate: {n} samples, {bad} mismatches"); sys.exit(1 if bad or n<30 else 0)
PY
$P -u fo_placement.py --humdial $HD --per-scen 100 --pre-enhanced enh/dfn3_demand_m15 --out nz/fo_dfn3_m15 > logs/dfn_ev.log 2>&1
{ for t in fo_N_m15 fo_Ri_m15 fo_sig_m15 fo_s_ann_m15 fo_dfn3_m15; do echo "== $t"; $P hm3_repo.py --root nz/$t --humdial $HD 2>&1 | tail -1; done
  cd nz; for b in fo_N_m15 fo_Ri_m15 fo_sig_m15 fo_s_ann_m15; do ../venv/bin/python ../fo_signtest.py $b fo_dfn3_m15; done; cd ..
} > logs/fo_tables_dfn_control.txt
touch FO_DFN_DONE
