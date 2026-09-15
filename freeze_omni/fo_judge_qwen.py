#!/usr/bin/env python3
"""用本地 Qwen2-7B-Instruct（Freeze-Omni 的文本主干，权重已在 repo/）做首响相关性评判。
输入 items.jsonl（fo_judge_items.py 产出），输出 scores.jsonl：score∈{0,1,2}，
2=回应了用户的实际问题，1=泛泛/部分相关，0=无关或答了别的问题。
贪心解码、温度 0；三棵树用同一评判器 → 配对比较。"""
import json
import pathlib
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

inp, outp = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
mp = "repo/Qwen2-7B-Instruct"
tok = AutoTokenizer.from_pretrained(mp)
model = AutoModelForCausalLM.from_pretrained(mp, torch_dtype=torch.bfloat16).to("mps").eval()
SYS = ("You grade the OPENING of a voice assistant reply. You are given the full question of the user, "
       "the prefix of it that the assistant had actually heard when it started talking, and the opening words "
       "of the assistant. Score 2 if the opening clearly addresses the actual question of the user; "
       "1 if it is generic or only partly related; 0 if it is unrelated, answers a different question, or is empty. "
       "Reply with JSON only: {\"score\": <0|1|2>, \"why\": \"<5 words>\"}.")
done = set()
if outp.exists():
    done = {json.loads(l)["key"] for l in outp.open()}
fo = outp.open("a")
n = 0
t0 = time.time()
for line in inp.open():
    it = json.loads(line)
    key = f"{it.get('tree','')}/{it['scenario']}/{it['sid']}"
    if key in done or it["first"] is None:
        continue
    q = it["segments"][0]["text"] if it["segments"] else ""
    user = (f"Full question of the user: {q}\n"
            f"Heard prefix when the assistant started: {it['heard_prefix'] or ''}\n"
            f"Assistant opening: {it['response'].strip() or '(empty)'}")
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("mps")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=20, do_sample=False)
    txt = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    try:
        sc = int(json.loads(txt[txt.find("{"): txt.rfind("}") + 1])["score"])
    except Exception:
        sc = -1
    fo.write(json.dumps({"key": key, "scenario": it["scenario"], "score": sc, "raw": txt},
                        ensure_ascii=False) + "\n")
    fo.flush()
    n += 1
    if n % 50 == 0:
        print(f"  {n} judged  {(time.time() - t0) / n:.2f}s/item", flush=True)
print(f"done {n}")
