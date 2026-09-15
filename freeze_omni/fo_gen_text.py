#!/usr/bin/env python3
"""Freeze-Omni 首响落点评测（听态 pass，无 token 生成）——与 Lychee 线同一口径。

每条 HumDial 样本：可选加噪（DEMAND ch01，按样本 seed 固定 → 系统间配对，与
humdial_prep 同分布）→ 可选前端（RNNoiseTorch 波形域）→ 160ms chunk 逐块喂
Freeze-Omni，读**模型自己的**状态判断（demo 里被强制 'cl' 覆盖，这里不覆盖）。
首次 stat=='ss' 的时刻 = 首响落点；全程未 ss = miss。
落盘 events.json 与 fdb_infer_lychee 同结构（events[0].start_time），humdial_metrics3
原样复用；另存每 chunk 的 state_1 概率轨迹（soft-placement / RL reward 的原料）。

    python fo_placement.py --humdial data/humdial/unz/test/en_test_nondev --out nz/fo_L0 \
        --per-scen 20 [--noise-dir data/demand --snr 0 15] [--rnnoise ckpt.pt]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
import types

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.compliance.kaldi as k

REPO = pathlib.Path.home() / "freeze_omni/repo_mps"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

ap = argparse.ArgumentParser()
ap.add_argument("--humdial", type=pathlib.Path, required=True)
ap.add_argument("--out", type=pathlib.Path, required=True)
ap.add_argument("--per-scen", type=int, default=20)
ap.add_argument("--noise-dir", type=pathlib.Path, default=None)
ap.add_argument("--snr", type=float, nargs=2, default=(0.0, 15.0))
ap.add_argument("--rnnoise", default=None, help="RNNoiseTorch state_dict；None=无前端")
ap.add_argument("--max-sec", type=float, default=40.0)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--max-tokens", type=int, default=48, help="首响文本最多生成 token 数")
ap.add_argument("--burst", type=int, default=0, help="瞬态噪声探针：随机插入 N 个 300ms 爆发（每样本 seed 固定）")
ap.add_argument("--burst-db", type=float, default=10.0, help="爆发相对语音功率的 dB")
ap.add_argument("--gain-db", type=float, default=0.0, help="探针：全局增益 dB（波形域）")
ap.add_argument("--lowpass", type=float, default=0.0, help="探针：低通截止 Hz（0=不用）")
ap.add_argument("--policy", default=None, help="fo_train_rl 的 best.pt/last.pt（bands: 设 gain_bias/gain_exp）")
ap.add_argument("--fdb", action="store_true", help="--humdial 指向 FDB 根（task/id/input.wav 布局）")
a = ap.parse_args()

from models.pipeline import inferencePipeline   # noqa: E402


class AudioProc:                                   # 与 bin/inference.py 同款 fbank 流
    def __init__(self):
        self.chunk_size, self.chunk_overlap, self.feat_dim = 16, 3, 80
        self.frame_size, self.frame_shift = 400, 160
        self.frame_overlap = self.frame_size - self.frame_shift
        self.CHUNK = self.frame_shift * self.chunk_size          # 2560 样本 = 160ms
        self.reset()

    def reset(self):
        self.input_chunk = torch.zeros([1, self.chunk_size + self.chunk_overlap, self.feat_dim])
        self.input_sample = torch.zeros([1, self.CHUNK + self.frame_overlap, 1])

    def process(self, audio):
        with torch.no_grad():
            s = audio.reshape(1, -1, 1) * 32768
            self.input_sample[:, :self.frame_overlap] = self.input_sample[:, -self.frame_overlap:].clone()
            self.input_sample[:, self.frame_overlap:] = s
            xs = k.fbank(waveform=self.input_sample.squeeze(-1), dither=0, frame_length=25,
                         frame_shift=10, num_mel_bins=self.feat_dim)
            self.input_chunk[:, :self.chunk_overlap] = self.input_chunk[:, -self.chunk_overlap:].clone()
            self.input_chunk[:, self.chunk_overlap:] = xs.squeeze(0)
        return self.input_chunk.clone()


def rd(p):
    x, sr = sf.read(str(p), dtype="float32", always_2d=True); x = torch.from_numpy(x.mean(1))
    if sr != 16000:
        x = torchaudio.transforms.Resample(sr, 16000)(x)
    return x


# ---------------------------------------------------------------- 噪声（与 humdial_prep 同分布/同 seed 规则）
noises = []
if a.noise_dir:
    for f in sorted(a.noise_dir.rglob("ch01.wav")):          # DEMAND: <ENV>_16k/<ENV>/ch01.wav
        x, sr = sf.read(str(f), dtype="float32", always_2d=True)
        if sr != 16000:                                          # 跳过 48k 版本
            continue
        noises.append(torch.from_numpy(x.mean(1)))
    assert noises, f"噪声库为空: {a.noise_dir}"


def add_noise(x, key: str):
    seed = int.from_bytes(key.encode(), "little") % (2**31)
    rng = torch.Generator().manual_seed(seed)
    nz = noises[int(torch.randint(len(noises), (1,), generator=rng))]
    K = len(x)
    off = int(torch.randint(max(len(nz) - K, 1), (1,), generator=rng))
    n = nz[off:off + K]
    if len(n) < K:
        n = n.repeat(-(-K // max(len(n), 1)))[:K]
    snr = a.snr[0] + float(torch.rand(1, generator=rng)) * (a.snr[1] - a.snr[0])
    ps, pn = float((x ** 2).mean()) + 1e-9, float((n ** 2).mean()) + 1e-9
    return x + n * math.sqrt(ps / pn / (10 ** (snr / 10))), snr


# ---------------------------------------------------------------- 前端
fe = None
if a.rnnoise:
    from rnnoise_torch import RNNoiseTorch
    fe = RNNoiseTorch().eval()
    fe.load_state_dict(torch.load(a.rnnoise, map_location="cpu", weights_only=True))
    if a.policy:
        pol = torch.load(a.policy, map_location="cpu", weights_only=True)
        th = pol["theta"]
        if pol.get("space") == "bands":
            fe.gain_bias = th[:22]; fe.gain_exp = float(th[22].exp())
        else:
            base = torch.nn.utils.parameters_to_vector(fe.parameters()).detach()
            torch.nn.utils.vector_to_parameters(base + th, fe.parameters())
        print(f"[policy] {a.policy} space={pol.get('space')} |θ|={float(th.norm()):.3f}", flush=True)


# ---------------------------------------------------------------- 模型
args = types.SimpleNamespace(model_path=str(REPO.parent / "models/checkpoints"),
                             llm_path=str(REPO / "Qwen2-7B-Instruct"),
                             top_k=5, top_p=0.8, temperature=0.7)
t0 = time.time(); pipe = inferencePipeline(args); print(f"[load] {time.time()-t0:.0f}s", flush=True)
proc = AudioProc(); C = proc.CHUNK
SCENS = ["ask", "backchannel", "deny", "others_talk_to_user_after", "others_talk_to_user_before",
         "pause", "repeat", "shift", "talk_to_others", "wait"]

samples = []
if a.fdb:
    samples = [(p.parent.parent.name, p) for p in sorted(a.humdial.glob("*/*/input.wav"))]
else:
    for sc in SCENS:
        wavs = sorted((a.humdial / sc).glob("*.wav"))[: a.per_scen]
        samples += [(sc, w) for w in wavs]
if a.limit:
    samples = samples[: a.limit]
print(f"[data] {len(samples)} 条  noise={bool(noises)} fe={bool(fe)}", flush=True)

done = skip = 0
for sc, w in samples:
    sid = w.parent.name if a.fdb else w.stem
    od = a.out / sc / sid; oj = od / "events.json"; tj = od / "text.json"
    if tj.exists():
        skip += 1; continue
    x = rd(w)
    if len(x) > a.max_sec * 16000:
        x = x[: int(a.max_sec * 16000)]
    snr = None
    if noises:
        x, snr = add_noise(x, f"{sc}/{sid}")
    if fe is not None:
        with torch.no_grad():
            x = fe(x[None])[0]
    if a.burst:
        rng = torch.Generator().manual_seed(int.from_bytes(f"burst/{sc}/{sid}".encode(), "little") % (2**31))
        ps = float((x ** 2).mean()) + 1e-9; L = int(0.3 * 16000)
        for _ in range(a.burst):
            st_ = int(torch.randint(max(len(x) - L, 1), (1,), generator=rng))
            nz = noises[int(torch.randint(len(noises), (1,), generator=rng))] if noises else torch.randn(L, generator=rng)
            off = int(torch.randint(max(len(nz) - L, 1), (1,), generator=rng)); b = nz[off:off + L]
            if len(b) < L:
                b = torch.nn.functional.pad(b, (0, L - len(b)))
            b = b * math.sqrt(ps * 10 ** (a.burst_db / 10) / (float((b ** 2).mean()) + 1e-9))
            x[st_:st_ + L] = x[st_:st_ + L] + b
    if a.gain_db:
        x = x * (10 ** (a.gain_db / 20))
    if a.lowpass:
        x = torchaudio.functional.lowpass_biquad(x, 16000, a.lowpass)
    proc.reset()
    out = pipe.speech_dialogue(None, stat="pre", role="You are a helpful assistant.")
    n = math.ceil(len(x) / C) * C
    xp = torch.zeros(n); xp[: len(x)] = x
    stats, first = [], None
    t1 = time.time()
    for i in range(0, n, C):
        out = pipe.speech_dialogue(proc.process(xp[i:i + C]), **out)
        s = out["stat"]; stats.append(s)
        if s == "ss":
            first = i / 16000; break
        if s not in ("cl", "sl"):
            out["stat"] = "cl"
    el = time.time() - t1
    # 与树中已有落点对拍（同 seed 应逐条一致）
    prev = json.loads(oj.read_text())["events"] if oj.exists() else None
    prev_first = (prev[0]["start_time"] if prev else None) if prev is not None else "n/a"
    text, ntok, tg = "", 0, 0.0
    if first is not None:
        t2 = time.time()
        with torch.no_grad():
            out = pipe.speech_dialogue(None, **out)           # Stage 3: start speak（沿用流式状态，同 server.py）
            last = ""
            suffix = ("。", "：", "？", "！", ".", "?", "!", "\n")
            while True:
                if len(out.get("past_tokens", [])) > a.max_tokens:
                    break
                out.pop("text", None); out.pop("hidden_state", None)
                out = pipe.speech_dialogue(None, **out)
                if out["stat"] == "cs":
                    text += out["text"][len(last):]
                    last = out["text"]
                    if text.strip().endswith(suffix) and len(out["past_tokens"]) >= 8:
                        break
                if out["stat"] == "sl":
                    break
            ntok = len(out.get("past_tokens", []))
        tg = time.time() - t2
    od.mkdir(parents=True, exist_ok=True)
    tj.write_text(json.dumps({"first": first, "prev_first": prev_first, "text": text, "n_tokens": ntok,
                              "gen_sec": tg, "snr": snr}, ensure_ascii=False), encoding="utf-8")
    if prev_first != "n/a" and prev_first != first:
        print(f"  [mismatch] {sc}/{sid} prev={prev_first} now={first}", flush=True)
    done += 1
    if done % 10 == 0:
        print(f"  {done}/{len(samples)}  最近: {sc}/{sid} first={first} gen={tg:.1f}s ntok={ntok} {text[:40]!r}", flush=True)
print(f"完成 done={done} skip={skip} → {a.out}")
