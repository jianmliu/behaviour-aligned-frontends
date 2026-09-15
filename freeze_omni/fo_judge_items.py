#!/usr/bin/env python3
"""把一棵树的 text.json + HumDial 元数据拼成评判条目 jsonl：用户到触发时刻为止说了什么、完整问题、模型首响文本。"""
import json, pathlib, sys
tree = pathlib.Path(sys.argv[1]); hd = pathlib.Path(sys.argv[2]); out = pathlib.Path(sys.argv[3])
n = 0
with out.open("w") as f:
    for tj in sorted(tree.glob("*/*/text.json")):
        sc, sid = tj.parent.parent.name, tj.parent.name
        d = json.loads(tj.read_text())
        meta = json.loads((hd / sc / f"{sid}.json").read_text())
        ts = hd / sc / f"{sid}_timestamp.json"
        heard = None
        if ts.exists() and d["first"] is not None:
            ch = json.loads(ts.read_text()).get("chunks", [])
            heard = " ".join(c["text"] for c in ch if c["timestamp"][1] is not None and c["timestamp"][1] <= d["first"])
        segs = meta.get("speech_segments", [])
        f.write(json.dumps({"scenario": sc, "sid": sid, "first": d["first"], "heard_prefix": heard,
                            "segments": [{"t0": s["xmin"], "t1": s["xmax"], "text": s["text"]} for s in segs],
                            "response": d["text"]}, ensure_ascii=False) + "\n"); n += 1
print(n)
