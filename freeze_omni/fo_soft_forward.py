#!/usr/bin/env python3
"""Freeze-Omni 可微整句前向：波形 →(前端)→ fbank → 编码器(非流式,chunk 掩码) → 子采样
adapter → [role][eod+prefix][25 prompt][audio tokens] → Qwen2 → predictor_head → 每 chunk
的 start-speaking 概率 p_t（4 类头取前 3 类 softmax 的第 1 类，与 recognize() 同口径）。

用途：soft_placement 训练（p_t 直接喂 first_trigger_dist）；RL 的可微替身。
★ --check：对同一样本，流式 recognize（forward hook 抓 predictor_head 输出）vs 整句前向
  的逐 chunk p_t 对拍——这是本模型的 verify_shift；不通过不得训练。

帧率链（train.yaml）：fbank 10ms → subsampling ×4 (40ms) → encoder chunk_size=4 帧 = 160ms
→ CNNSubsampling ×2 → 每 160ms chunk 2 个 LLM token，chunk 末 token 出状态。
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
import types

import torch
import torch.nn.functional as F
import torchaudio.compliance.kaldi as k

REPO = pathlib.Path.home() / "freeze_omni/repo_mps"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

CHUNK_SAMPLES = 2560          # 160ms @16k（bin/inference.py audioEncoderProcessor）
FRAME_OVERLAP = 240           # 400-160：流式 fbank 的前置零填充，整句版复现同一帧网格
ROLE = "You are a helpful assistant."


def load_pipeline():
    from models.pipeline import inferencePipeline
    args = types.SimpleNamespace(model_path=str(REPO.parent / "models/checkpoints"),
                                 llm_path=str(REPO / "Qwen2-7B-Instruct"),
                                 top_k=5, top_p=0.8, temperature=0.7)
    return inferencePipeline(args)


class FreezeOmniPlacement:
    """整句可微前向。model = pipeline.model（audioLLM）。"""

    def __init__(self, model, device="mps"):
        self.m = model
        self.dev = device
        tok = model.tokenizer
        role_ids = tok(["<|im_start|>system\n" + ROLE], return_tensors="pt")["input_ids"]
        prefix = torch.cat([torch.tensor([[tok.eod_id]]), model.chat_template["prefix"]], 1)
        self.role_ids = role_ids.to(device)
        self.prefix_ids = prefix.to(device)
        self.n_prompt = int(model.prompt_ids.numel()) if getattr(model, "prompt_finetune", False) else 0
        self.tokens_per_chunk = None          # 首次前向时由长度比推定，check 模式核验

    def fbank_full(self, wav: torch.Tensor) -> torch.Tensor:
        """wav (K,) fp32 → (1, T, 80)，帧网格与流式 AudioProc 一致（前置 240 零、按 chunk 补齐）。"""
        n = math.ceil(len(wav) / CHUNK_SAMPLES) * CHUNK_SAMPLES
        xp = torch.zeros(FRAME_OVERLAP + n, device=wav.device, dtype=wav.dtype)
        xp[FRAME_OVERLAP:FRAME_OVERLAP + len(wav)] = wav
        feats = k.fbank(waveform=(xp * 32768)[None], dither=0, frame_length=25, frame_shift=10,
                        num_mel_bins=80)
        return feats[None]

    def state_probs(self, wav: torch.Tensor, need_grad: bool = True):
        """返回 p1 (Tc,)：每 chunk 的 start-speaking 概率（可微到 wav），及 Tc。"""
        m = self.m
        feats = self.fbank_full(wav)                                  # (1,T,80)
        ilens = torch.tensor([feats.shape[1]], device=feats.device)
        enc, enc_mask = m.encoder(feats.to(self.dev), ilens.to(self.dev),
                                  decoding_chunk_size=4, num_decoding_left_chunks=16)
        ad = m.adpter(enc, enc_mask)
        audio_emb, audio_mask = ad[0], ad[1]                          # (1, Ta, D)
        Ta = audio_emb.shape[1]
        n_chunks = feats.shape[1] // 16
        tpc = self.tokens_per_chunk or max(1, round(Ta / max(n_chunks, 1)))
        self.tokens_per_chunk = tpc

        wte = m.llm_decoder.transformer.wte
        parts = [wte(self.role_ids), wte(self.prefix_ids)]
        if self.n_prompt:
            pid = m.prompt_ids.to(self.dev)
            parts.append(m.prompt_embeddings(pid[None] if pid.dim() == 1 else pid))
        parts.append(audio_emb)
        emb = torch.cat(parts, 1).half()
        att = torch.ones(emb.shape[:2], dtype=torch.bool, device=self.dev)
        ctx = torch.enable_grad() if need_grad else torch.no_grad()
        with ctx, torch.autocast(device_type="mps", dtype=torch.bfloat16):
            out = m.llm_decoder.model(inputs_embeds=emb, attention_mask=att)
            h = out["last_hidden_state"][0]                           # (L, D)
            logits = m.predictor_head(h)                              # (L, 4)
            prob = F.softmax(logits[:, :-1].float(), dim=-1)          # 与 recognize 同：去掉最后一类
        off = self.role_ids.shape[1] + self.prefix_ids.shape[1] + self.n_prompt
        idx = torch.tensor([off + tpc * (c + 1) - 1 for c in range(n_chunks)], device=self.dev)
        idx = idx.clamp(max=prob.shape[0] - 1)
        p1 = prob[idx, 1]
        z = logits[idx, :-1].float()                              # (Tc, 3) 供边距项（绕过饱和）
        self.last_margin_excess = torch.relu(z[:, 1] - torch.stack([z[:, 0], z[:, 2]], -1).max(-1).values + 2.0)
        return p1, n_chunks


# ---------------------------------------------------------------- 对拍
class AudioProc:                                   # 与 bin/inference.py 同款 fbank 流（内置，避免
    def __init__(self):                            # import fo_placement 触发其模块级 argparse）
        self.chunk_size, self.chunk_overlap, self.feat_dim = 16, 3, 80
        self.frame_size, self.frame_shift = 400, 160
        self.frame_overlap = self.frame_size - self.frame_shift
        self.CHUNK = self.frame_shift * self.chunk_size
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


def streaming_probs(pipe, wav: torch.Tensor):
    """流式 recognize 逐 chunk 跑，forward hook 抓 predictor_head 输出 → 每 chunk p1。"""
    caught = []
    def hook(_m, _i, o):
        prob = F.softmax(o[0, -1, :-1].float(), dim=-1)
        caught.append(float(prob[1]))
    hd = pipe.model.predictor_head.register_forward_hook(hook)
    proc = AudioProc(); C = proc.CHUNK
    out = pipe.speech_dialogue(None, stat="pre", role=ROLE)
    n = math.ceil(len(wav) / C) * C
    xp = torch.zeros(n); xp[: len(wav)] = wav
    for i in range(0, n, C):
        out = pipe.speech_dialogue(proc.process(xp[i:i + C]), **out)
        out["stat"] = "cl"                    # 对拍需要走完全程（不在 ss 处停）
    hd.remove()
    return torch.tensor(caught[1:])       # 第 0 个来自 set_system_role 的 "sl" 步（伪触发），丢弃


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=pathlib.Path, nargs="+", help="wav 样本，流式 vs 整句对拍")
    a = ap.parse_args()
    import soundfile as sf
    pipe = load_pipeline()
    fop = FreezeOmniPlacement(pipe.model)
    for w in a.check or []:
        x, sr = sf.read(str(w), dtype="float32", always_2d=True); x = torch.from_numpy(x.mean(1))
        assert sr == 16000
        x = x[: 20 * 16000]
        ps = streaming_probs(pipe, x)
        pf, nc = fop.state_probs(x, need_grad=False)
        pf = pf.cpu()
        n = min(len(ps), len(pf))
        d = (ps[:n] - pf[:n]).abs()
        print(f"{w.name}: chunks stream={len(ps)} full={len(pf)} tokens/chunk={fop.tokens_per_chunk} "
              f"|Δp1| mean={d.mean():.4f} max={d.max():.4f}  "
              f"first>0.5 stream={int((ps>0.5).nonzero()[0]) if (ps>0.5).any() else None} "
              f"full={int((pf>0.5).nonzero()[0]) if (pf>0.5).any() else None}")
        fs = int((ps>0.5).nonzero()[0]) if (ps>0.5).any() else None
        lo = max(0, (fs or 0) - 3); hi = min(n, (fs or 0) + 3)
        print("  越界附近 stream:", [round(float(v), 3) for v in ps[lo:hi]])
        print("  越界附近 full  :", [round(float(v), 3) for v in pf[lo:hi]])
