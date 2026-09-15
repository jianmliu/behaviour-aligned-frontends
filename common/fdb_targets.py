#!/usr/bin/env python3
"""FDB v1.0 人工标注 → 落点目标树（events.json，与 fo_placement 自蒸馏树同格式）。

t*（该开口的时刻）：
  candor_turn_taking       turn_taking.json 的 [TURN-TAKING] 窗起点
  *_pause_handling / icc   transcription.json 末词结束（用户话轮真实结束；停顿区间在其前，
                           premature 成本自然惩罚停顿内抢答）
  synthetic_user_interruption  interrupt.json 窗终点（用户打断说完）
无法确定 → 跳过（不写文件）。自蒸馏目标教"抗噪"，标注目标教"克制"——两树并用。

    python fdb_targets.py --fdb data/fdb/v1.0/unz --out nz/fo_fdb_ann
"""
import argparse
import json
import pathlib

ap = argparse.ArgumentParser()
ap.add_argument("--fdb", type=pathlib.Path, required=True)
ap.add_argument("--out", type=pathlib.Path, required=True)
a = ap.parse_args()

n = skip = 0
for d in sorted(p.parent for p in a.fdb.glob("*/*/input.wav")):
    task, sid = d.parent.name, d.name
    t = None
    if (d / "turn_taking.json").exists():
        j = json.loads((d / "turn_taking.json").read_text())
        if j:
            t = float(j[0]["timestamp"][0])
    elif (d / "interrupt.json").exists():
        j = json.loads((d / "interrupt.json").read_text())
        if j:
            t = float(j[0]["timestamp"][1])
    elif (d / "transcription.json").exists():
        j = json.loads((d / "transcription.json").read_text())
        if j:
            t = max(float(w["timestamp"][1]) for w in j)
    if t is None:
        skip += 1
        continue
    od = a.out / task / sid
    od.mkdir(parents=True, exist_ok=True)
    (od / "events.json").write_text(json.dumps(
        {"events": [{"type": "response", "start_time": t, "end_time": -1}], "source": "annotation"}))
    n += 1
print(f"目标树 {n} 条（跳过 {skip}）→ {a.out}")
