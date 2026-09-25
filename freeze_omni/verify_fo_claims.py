#!/usr/bin/env python3
"""Recompute every Freeze-Omni number stated in the paper from the per-sample trees and compare.

Run on the mini from ~/freeze_omni. Prints OK / DIFF per claim; nothing is taken from summary logs.
"""
import json
import math
import re
import subprocess
import sys
from math import comb
from pathlib import Path

HD = "data/humdial/unz/test/en_test_nondev"
NZ = Path("nz")


def summary(tree):
    out = subprocess.run([sys.executable, "humdial_metrics3.py", "--root", str(NZ / tree), "--humdial", HD],
                         capture_output=True, text=True).stdout
    m = re.search(r"on-time ([\d.]+)%\s+premature ([\d.]+)%\s+miss ([\d.]+)%", out)
    rows = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 8 and parts[1].isdigit():
            rows[parts[0]] = [int(x) for x in parts[1:]]
    n = len(list((NZ / tree).rglob("events.json")))
    return (float(m.group(1)), float(m.group(2)), float(m.group(3))) if m else None, rows, n


def first(p):
    ev = [e for e in json.loads(p.read_text()).get("events", []) if e.get("type") == "response"]
    return min(e["start_time"] for e in ev) if ev else None


def load(tree):
    return {str(p.parent.relative_to(NZ / tree)): first(p) for p in (NZ / tree).rglob("events.json")}


def sign(a, b, tol=0.2):
    A, B = load(a), load(b)
    e = l = 0
    for k in set(A) & set(B):
        x, y = A[k], B[k]
        if x is None or y is None or abs(y - x) <= tol:
            continue
        e += y < x; l += y > x
    n = e + l; k = min(e, l)
    p = sum(comb(n, i) for i in range(k + 1)) * 2 / 2 ** n if n else 1.0
    return e, l, min(p, 1.0)


results = []


def claim(name, paper, got, tol=0.05):
    ok = all(abs(float(a) - float(b)) <= tol for a, b in zip(paper, got)) if isinstance(paper, (list, tuple)) \
        else abs(float(paper) - float(got)) <= tol
    results.append(ok)
    print(f"{'OK  ' if ok else 'DIFF'} {name}: paper {paper} | recomputed {got}")


def ptxt(p):
    return f"{p:.1e}"


S = {}
for t in ["fo_L0", "fo_N0", "fo_Rindep", "fo_s_ann", "fo_s_self", "fo_burst10", "fo_burst20",
          "fo_Rindep_clean", "fo_s_ann_clean", "fo_N_m5", "fo_N_m10", "fo_N_m15", "fo_N_m20",
          "fo_Ri_m5", "fo_Ri_m10", "fo_Ri_m15", "fo_Ri_m20", "fo_s_ann_m5", "fo_s_ann_m10", "fo_s_ann_m15",
          "fo_s_ann_m20", "fo_rl_m15", "fo_rl_long_m15", "fo_rl_full_m15", "fo_rl_dense_m15",
          "fo_rl_dense_full_m15", "mu_N_m15", "mu_Ri_m15", "mu_s_ann_m15", "mu_s_ann_tr_m15", "mu_rl_tr_m15"]:
    if (NZ / t).exists():
        S[t] = summary(t)
    else:
        print(f"MISSING tree {t}")

ot = lambda t: S[t][0][:2]
restr = lambda t: (S[t][1]["others_talk_to_user_before"][4], S[t][1]["pause"][5])
print("== sample counts:", {t: S[t][2] for t in S})
claim("clean 40.6/43.8", (40.6, 43.8), ot("fo_L0"))
claim("clean restraint 99/57", (99, 57), restr("fo_L0"), 0)
claim("protocol noise 40.6/43.4", (40.6, 43.4), ot("fo_N0"))
claim("protocol noise restraint 100/60", (100, 60), restr("fo_N0"), 0)
claim("frozen 38.8/45.1", (38.8, 45.1), ot("fo_Rindep"))
claim("frozen restraint 100/61", (100, 61), restr("fo_Rindep"), 0)
claim("soft placement 39.7/44.3", (39.7, 44.3), ot("fo_s_ann"))
claim("soft placement restraint 100/60", (100, 60), restr("fo_s_ann"), 0)
claim("self-distillation 43.0/39.0 (n=200)", (43.0, 39.0), ot("fo_s_self"))
e, l, p = sign("fo_N0", "fo_Rindep"); claim("frozen vs noise earlier:later 172:115", (172, 115), (e, l), 0); print("   p", ptxt(p))
e, l, p = sign("fo_N0", "fo_s_ann"); claim("trained vs noise earlier:later 156:91", (156, 91), (e, l), 0); print("   p", ptxt(p))
claim("bursts +10 dB 42.5/40.0", (42.5, 40.0), ot("fo_burst10"))
claim("bursts +20 dB 39.5/43.5", (39.5, 43.5), ot("fo_burst20"))
for t in ["fo_Rindep_clean", "fo_s_ann_clean"]:
    e, l, p = sign("fo_L0", t); print(f"     clean-audio {t} vs L0: {e}:{l} p={p:.2f} (paper: n.s.)"); results.append(p > 0.05)
for s, v in zip([5, 10, 15, 20], [46.3, 51.2, 56.5, 63.0]):
    claim(f"dose premature -{s} dB", v, S[f"fo_N_m{s}"][0][1])
claim("-15 none 28.4/56.5", (28.4, 56.5), ot("fo_N_m15"))
claim("-15 frozen 24.3/61.2", (24.3, 61.2), ot("fo_Ri_m15"))
e, l, p = sign("fo_N_m15", "fo_Ri_m15"); claim("-15 frozen vs none 340:174", (340, 174), (e, l), 0); print("   p", ptxt(p), "(paper 2e-13)")
claim("-15 aligned 35.8/49.0", (35.8, 49.0), ot("fo_s_ann_m15"))
e, l, p = sign("fo_N_m15", "fo_s_ann_m15"); claim("-15 aligned vs none later 309:173", (173, 309), (e, l), 0); print("   p", ptxt(p), "(paper 6e-10)")
e, l, p = sign("fo_Ri_m15", "fo_s_ann_m15"); claim("-15 aligned vs frozen later 432:139", (139, 432), (e, l), 0); print("   p", ptxt(p), "(paper <1e-35)")
claim("-15 ES 24.7/60.8", (24.7, 60.8), ot("fo_rl_m15"))
e, l, p = sign("fo_Ri_m15", "fo_rl_m15"); print(f"     ES vs frozen {e}:{l} p={p:.2f} (paper 0.27)"); results.append(abs(p - 0.27) < 0.02)
e, l, p = sign("fo_s_ann_m15", "fo_rl_m15"); claim("ES earlier than aligned 421:154", (421, 154), (e, l), 0)
claim("ES 120 gens 24.6/60.5", (24.6, 60.5), ot("fo_rl_long_m15"))
claim("ES all-param 17.4/68.7", (17.4, 68.7), ot("fo_rl_full_m15"))
e, l, p = sign("fo_Ri_m15", "fo_rl_full_m15"); print(f"     ES all-param vs frozen {e}:{l} p={p:.1e} (paper <1e-11)"); results.append(p < 1e-11)
claim("ES dense 25.9/59.5", (25.9, 59.5), ot("fo_rl_dense_m15"))
claim("ES dense all-param 23.6/61.5", (23.6, 61.5), ot("fo_rl_dense_full_m15"))
for s in (10, 20):
    ps = [sign(f"fo_N_m{s}", f"fo_Ri_m{s}")[2], sign(f"fo_N_m{s}", f"fo_s_ann_m{s}")[2], sign(f"fo_Ri_m{s}", f"fo_s_ann_m{s}")[2]]
    print(f"     -{s} dB sign-test p (frozen/none, aligned/none, aligned/frozen): {[f'{x:.1e}' for x in ps]} (paper: all <1e-5)")
    results.append(max(ps) < 1e-5)
print(f"     -5 dB: none {ot('fo_N_m5')} frozen {ot('fo_Ri_m5')} aligned {ot('fo_s_ann_m5')}; aligned vs none p={sign('fo_N_m5','fo_s_ann_m5')[2]:.2f} (paper: cancels harm)")
claim("MUSAN none 13.0/75.6", (13.0, 75.6), ot("mu_N_m15"))
claim("MUSAN frozen 14.2/74.3", (14.2, 74.3), ot("mu_Ri_m15"))
print(f"     MUSAN frozen vs none p={sign('mu_N_m15','mu_Ri_m15')[2]:.2f} (paper 0.24)")
claim("MUSAN DEMAND-trained 15.8/71.2", (15.8, 71.2), ot("mu_s_ann_m15"))
print(f"     MUSAN DEMAND-trained vs none p={sign('mu_N_m15','mu_s_ann_m15')[2]:.2f} (paper 0.22)")
e, l, p = sign("mu_N_m15", "mu_s_ann_tr_m15"); claim("MUSAN retrained later 275:151", (151, 275), (e, l), 0); print("   p", ptxt(p), "(paper 2e-9)")
ps = [sign("mu_N_m15", "mu_rl_tr_m15")[2], sign("mu_s_ann_tr_m15", "mu_rl_tr_m15")[2]]
print(f"     MUSAN retrained ES between: p vs none {ps[0]:.1e}, vs aligned {ps[1]:.1e} (paper p<=0.004)"); results.append(max(ps) <= 0.0045)
print(f"\n{sum(results)}/{len(results)} checks pass")
