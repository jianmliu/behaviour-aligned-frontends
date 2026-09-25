#!/usr/bin/env python3
"""Recompute every Lychee-FD number stated in the paper from the per-sample event trees.

Uses the exact classification of humdial_metrics3.py, per sample, so paired tests can be recomputed.
Run on the mini:  python verify_lychee_claims.py <dir with ev_L0 ev_N0 ev_Rindep ev_Rjoint ev_Rjoint{1,2,3}>
"""
import json
import pathlib
import sys
from math import comb

HD = pathlib.Path("data/humdial/unz/test/en_test_nondev")
ROOT = pathlib.Path(sys.argv[1])
TOL, W = 1.0, 6.0
OTHER_SEG = {"others_talk_to_user_before": 0, "others_talk_to_user_after": 1, "talk_to_others": 2}
Q_SEG = {"others_talk_to_user_before": 1}


def first_event(p):
    evs = [e for e in json.loads(p.read_text()).get("events", []) if e.get("type") != "backchannel"]
    return float(evs[0]["start_time"]) if evs else None


def classify(tree):
    out = {}
    for ev in (ROOT / tree).glob("*/*/events.json"):
        scen, sid = ev.parent.parent.name, ev.parent.name
        ann = HD / scen / f"{sid}.json"
        if not ann.exists():
            continue
        segs = json.loads(ann.read_text()).get("speech_segments", [])
        if not segs:
            continue
        st = first_event(ev)
        key = f"{scen}/{sid}"
        if st is None:
            out[key] = "miss"; continue
        q = segs[Q_SEG.get(scen, 0)] if Q_SEG.get(scen, 0) < len(segs) else segs[0]
        oi = OTHER_SEG.get(scen)
        if oi is not None and oi < len(segs) and segs[oi]["xmin"] - 0.3 <= st <= segs[oi]["xmax"] + 2.0:
            out[key] = "mis_addr"; continue
        if scen == "pause":
            ts = HD / scen / f"{sid}_timestamp.json"
            if ts.exists():
                ch = json.loads(ts.read_text()).get("chunks", [])
                best, gap = 0, None
                for i in range(len(ch) - 1):
                    g0, g1 = ch[i]["timestamp"][1], ch[i + 1]["timestamp"][0]
                    if g1 - g0 > best:
                        best, gap = g1 - g0, (g0, g1)
                if gap and best >= 0.8 and gap[0] - 0.2 <= st <= gap[1] + 0.3:
                    out[key] = "break_preempt"; continue
        out[key] = "premature" if st < q["xmax"] - TOL else ("on_time" if st <= q["xmax"] + W else "late")
    return out


def pct(c, cls, scen=None):
    ks = [k for k in c if scen is None or k.startswith(scen + "/")]
    return 100 * sum(c[k] == cls for k in ks) / len(ks), len(ks)


def mcnemar(a, b, pred):
    ks = set(a) & set(b)
    x = sum(pred(a[k]) and not pred(b[k]) for k in ks); y = sum(pred(b[k]) and not pred(a[k]) for k in ks)
    n = x + y; k = min(x, y)
    return x, y, min(1.0, sum(comb(n, i) for i in range(k + 1)) * 2 / 2 ** n) if n else 1.0


C = {t: classify(t) for t in ["ev_L0", "ev_N0", "ev_Rindep", "ev_Rjoint", "ev_Rjoint1", "ev_Rjoint2", "ev_Rjoint3"]}
print("n per tree:", {t: len(v) for t, v in C.items()})
paper = {"ev_L0": (5.4, 59.4, 27.4, 5.8, 18), "ev_N0": (3.2, 46.4, 44.5, 4.4, 10),
         "ev_Rindep": (6.2, 60.1, 26.6, 4.8, 20), "ev_Rjoint": (7.0, 43.3, 44.7, 3.7, 10)}
ok = []
for t, want in paper.items():
    got = (round(pct(C[t], "on_time")[0], 1), round(pct(C[t], "premature")[0], 1), round(pct(C[t], "miss")[0], 1),
           round(pct(C[t], "mis_addr")[0], 1), round(pct(C[t], "break_preempt", "pause")[0]))
    good = all(abs(a - b) < 0.06 for a, b in zip(want, got)); ok.append(good)
    print(f"{'OK  ' if good else 'DIFF'} Table 1 {t}: paper {want} | recomputed {got}")
for t, lab in [("ev_Rjoint1", "r1 lr3e-5 (paper 6.2/62/25)"), ("ev_Rjoint2", "r2 scratch (paper on-time 3.6)"), ("ev_Rjoint3", "r3 SI-SNR anchor (paper ~frozen)")]:
    print(f"     ablation {lab}: on-time {pct(C[t],'on_time')[0]:.1f} premature {pct(C[t],'premature')[0]:.1f} miss {pct(C[t],'miss')[0]:.1f} mis-addr {pct(C[t],'mis_addr')[0]:.1f} break {pct(C[t],'break_preempt','pause')[0]:.0f}")
print(f"     clean premature on talk_to_others: {pct(C['ev_L0'],'premature','talk_to_others')[0]:.0f}% (paper 86)")
print(f"     clean others_before mis-addressed: {pct(C['ev_L0'],'mis_addr','others_talk_to_user_before')[0]:.0f}% (paper 58)")
scens = sorted({k.split('/')[0] for k in C["ev_L0"]})
rise = {s: (pct(C['ev_L0'], 'miss', s)[0], pct(C['ev_N0'], 'miss', s)[0]) for s in scens}
print("     misses clean->noise per scenario:", {s: f"{a:.0f}->{b:.0f}" for s, (a, b) in rise.items()})
print(f"     misses rise in all ten: {all(b > a for a, b in rise.values())}; largest noise miss: {max(rise, key=lambda s: rise[s][1])} (paper: repeat 42->59)")
R, J = C["ev_Rindep"], C["ev_Rjoint"]
x, y, p = mcnemar(R, J, lambda c: c == "premature"); print(f"     McNemar premature frozen-only {x} vs behav-only {y}, p={p:.1e} (paper 252 vs 86, 5e-20)")
x, y, p = mcnemar(R, J, lambda c: c == "miss"); print(f"     McNemar miss frozen-only {x} vs behav-only {y}, p={p:.1e} (paper 2e-22)")
wp = lambda c: c in ("premature", "mis_addr", "break_preempt")
ks = set(R) & set(J)
print(f"     wrong-place: frozen {100*sum(wp(R[k]) for k in R)/len(R):.1f}% behav {100*sum(wp(J[k]) for k in J)/len(J):.1f}% (paper 66.9/48.0)")
x, y, p = mcnemar(R, J, wp); print(f"     McNemar wrong-place {x} vs {y}, p={p:.1e} (paper 4e-24)")
for a, lab in [("ev_Rindep", "frozen"), ("ev_L0", "clean")]:
    x, y, p = mcnemar(C[a], J, lambda c: c == "on_time"); print(f"     on-time behav vs {lab}: {y} vs {x}, p={p:.2f} (paper n.s.)")
print(f"\n{sum(ok)}/{len(ok)} table rows match")
