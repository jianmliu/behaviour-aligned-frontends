"""噪声篇联合训练接缝：RNNoiseTorch × 冻结 Lychee（继承回声版 LycheeJointFrontend）。

与回声版差异仅两处：前端无参考信号（forward(wav)），场景是加噪不是回声。
训练目标复用 FDB 727 的干净轨迹（fdb_out/LycheeFD_clean 的 events+audio_input_ids 重放）
——与回声篇同一套 pack 逻辑（train_lychee_joint.lychee_hook_pack）。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from geic_nemo.frontend import lychee_hook
from geic_nemo.frontend.lychee_hook import LycheeJointFrontend, build_control_label, IGNORE  # noqa


class NoiseJointFrontend(LycheeJointFrontend):
    """geic 参数位传入 RNNoiseTorch 实例；scene = {"noisy": (B,K)}。"""

    def enhanced_mel(self, scene: dict):
        dev = next(self.geic.parameters()).device
        wav = self.geic(scene["noisy"].float().to(dev))            # RNNoise: (B,K)->(B,K)
        mels = [self._log_mel(w, n_mels=128) for w in wav]
        feats = torch.nn.utils.rnn.pad_sequence(
            [m.t() for m in mels], batch_first=True).transpose(1, 2)
        lens = torch.tensor([m.shape[1] - 2 for m in mels], dtype=torch.int32,
                            device=wav.device)
        return feats, lens


class NoiseScene:
    """近端 + 噪声 @随机 SNR（与 pretrain_rnnoise 同分布；每样本 seed 固定可复现）。"""

    def __init__(self, noise_files, snr_range=(-5.0, 20.0)):
        import soundfile as sf
        import numpy as np
        self.noises = []
        for f in noise_files:
            x, sr = sf.read(str(f), dtype="float32", always_2d=True)
            assert sr == 16000, f
            self.noises.append(torch.from_numpy(x.mean(1)))
        assert self.noises
        self.snr_range = snr_range

    def __call__(self, near: torch.Tensor, gen: torch.Generator):
        B, K = near.shape
        out = near.clone()
        for b in range(B):
            nz = self.noises[int(torch.randint(len(self.noises), (1,), generator=gen))]
            off = int(torch.randint(max(len(nz) - K, 1), (1,), generator=gen))
            n = nz[off: off + K]
            if len(n) < K:
                reps = -(-K // max(len(n), 1))
                n = n.repeat(reps)[:K]
            snr = self.snr_range[0] + float(torch.rand(1, generator=gen)) * \
                (self.snr_range[1] - self.snr_range[0])
            ps = float((near[b] ** 2).mean()) + 1e-9
            pn = float((n ** 2).mean()) + 1e-9
            out[b] = near[b] + n * (ps / pn / (10 ** (snr / 10))) ** 0.5
        return {"noisy": out, "clean": near}
