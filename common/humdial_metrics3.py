#!/usr/bin/env python3
"""HumDial 行为指标 v3：首响落点分析（定稿口径）。

实证前提（L0 干净基线，n=235 中期）：99% 样本单响应、92% 首响为句中抢答——
第二轮探针（continue/stop）不可测；改测「唯一响应的落点质量」，它对噪声敏感且语义清晰。

落点分类（首个 response 事件的 start_time 相对标注段）：
  premature   落在第一问句内部（< seg1.xmax - TOL）
  on-time     落在第一问句尾/其后合理窗（seg1.xmax-TOL .. +W）
  late/other  其后（含第二段之后——罕见）
  mis-addr    落在「非对模型语音」窗口内（others_*/talk_to_others 的他人段）= 假触发
  miss        无响应
pause 场景另计 break-preempt：落在 [break] 间隙内。
"""
import argparse, json, pathlib, collections

TOL = 1.0
W = 6.0

ap = argparse.ArgumentParser()
ap.add_argument("--root", type=pathlib.Path, required=True)
ap.add_argument("--humdial", type=pathlib.Path, required=True)
ap.add_argument("--out", type=pathlib.Path, default=None)
a = ap.parse_args()

OTHER_SEG = {  # 场景 → 「他人语音」段索引（mis-addressed 判定窗）
    "others_talk_to_user_before": 0,
    "others_talk_to_user_after": 1,
    "talk_to_others": 2,
}
Q_SEG = {"others_talk_to_user_before": 1}          # 真问句段索引（默认 0）

def first_event(p):
    d = json.loads(p.read_text())
    evs = [e for e in d.get("events", []) if e.get("type") != "backchannel"]
    return float(evs[0]["start_time"]) if evs else None

per = collections.defaultdict(collections.Counter)
for ev in a.root.glob("*/*/events.json"):
    scen, sid = ev.parent.parent.name, ev.parent.name
    ann_p = a.humdial / scen / f"{sid}.json"
    if not ann_p.exists():
        continue
    segs = json.loads(ann_p.read_text()).get("speech_segments", [])
    if not segs:
        continue
    st = first_event(ev)
    c = per[scen]
    c["n"] += 1
    if st is None:
        c["miss"] += 1
        continue
    qi = Q_SEG.get(scen, 0)
    q = segs[qi] if qi < len(segs) else segs[0]
    oi = OTHER_SEG.get(scen)
    if oi is not None and oi < len(segs):
        o = segs[oi]
        if o["xmin"] - 0.3 <= st <= o["xmax"] + 2.0:
            c["mis_addr"] += 1
            continue
    if scen == "pause":
        ts = a.humdial / scen / f"{sid}_timestamp.json"
        if ts.exists():
            ch = json.loads(ts.read_text()).get("chunks", [])
            best, gap = 0, None
            for i in range(len(ch) - 1):
                g0, g1 = ch[i]["timestamp"][1], ch[i + 1]["timestamp"][0]
                if g1 - g0 > best:
                    best, gap = g1 - g0, (g0, g1)
            if gap and best >= 0.8 and gap[0] - 0.2 <= st <= gap[1] + 0.3:
                c["break_preempt"] += 1
                continue
    if st < q["xmax"] - TOL:
        c["premature"] += 1
    elif st <= q["xmax"] + W:
        c["on_time"] += 1
    else:
        c["late"] += 1

cols = ["n", "on_time", "premature", "miss", "mis_addr", "break_preempt", "late"]
print(f"{'scenario':<30}" + "".join(f"{c:>10}" for c in cols))
tot = collections.Counter()
out = {}
for scen in sorted(per):
    c = per[scen]
    print(f"{scen:<30}" + "".join(f"{c.get(k,0):>10}" for k in cols))
    for k in cols:
        tot[k] += c.get(k, 0)
    out[scen] = dict(c)
print("-" * 100)
print(f"{'TOTAL':<30}" + "".join(f"{tot.get(k,0):>10}" for k in cols))
n = max(tot["n"], 1)
print(f"\non-time {100*tot['on_time']/n:.1f}%  premature {100*tot['premature']/n:.1f}%  "
      f"miss {100*tot['miss']/n:.1f}%  mis-addressed {100*tot['mis_addr']/n:.1f}%")
if a.out:
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
