"""可微 RNNoise-style 噪声抑制前端（噪声篇 P1，docs/11 计划）。

设计（论文表述：learned-gain band suppressor, RNNoise-style~\\cite{valin2018rnnoise}）：
- 特征：22 个 Bark 尺度带的 log 能量 + BFCC(DCT) + 一阶/二阶差分 → 全可微。
  官方 42 维里的 pitch 相关 6 维置零（RNNoise 论文自述其贡献有限；留作消融）。
- 网络：官方拓扑 Dense(24)→GRU(24)→GRU(48)→GRU(96)→Dense(22, sigmoid) ≈ 0.09M 参数。
- 应用：22 带增益线性插值到 STFT bin，乘幅度谱（保相位），ISTFT——保长、因果（帧内）。
- 训练：整链可微；信号 loss 预训练（回声篇 E41 配方：先信号先验）→ control-CE 微调。

STFT 口径与 RNNoise 一致量级：48k 原版是 20ms 窗/10ms hop；我们 16k 用 512 窗/160 hop（10ms）。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

SR = 16000
N_FFT = 512
HOP = 160
N_BANDS = 22


def bark_band_edges(n_fft=N_FFT, sr=SR, n_bands=N_BANDS):
    """近似 Bark/opus 带划分：低频密、高频疏（RNNoise 用 opus 带）。返回每带的 bin 边界。"""
    # opus 风格：前几带线性(~200Hz)，之后近似指数
    edges_hz = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 2000, 2400, 2800,
                3200, 4000, 4800, 5600, 6400, 8000, 9600, 12000, 15600, 20000, 24000]
    edges_hz = [min(h, sr // 2) for h in edges_hz[: n_bands + 1]]
    bins = [int(round(h / (sr / 2) * (n_fft // 2))) for h in edges_hz]
    # 单调化去重
    for i in range(1, len(bins)):
        bins[i] = max(bins[i], bins[i - 1] + 1)
    bins[-1] = n_fft // 2 + 1
    return bins


class RNNoiseTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(N_FFT))
        edges = bark_band_edges()
        # 带矩阵 (n_bands, n_bins)：三角/矩形权重（矩形足够，保持简单）
        n_bins = N_FFT // 2 + 1
        M = torch.zeros(N_BANDS, n_bins)
        for b in range(N_BANDS):
            M[b, edges[b]: edges[b + 1]] = 1.0 / max(edges[b + 1] - edges[b], 1)
        self.register_buffer("band_mat", M)                     # 分析：bin→band 均值
        # 合成：band→bin 线性插值权重（转置的归一化版）
        S = torch.zeros(n_bins, N_BANDS)
        centers = [(edges[b] + edges[b + 1] - 1) / 2 for b in range(N_BANDS)]
        for k in range(n_bins):
            # 找左右带中心线性插值
            b = 0
            while b < N_BANDS - 1 and centers[b + 1] < k:
                b += 1
            if k <= centers[0]:
                S[k, 0] = 1.0
            elif k >= centers[-1]:
                S[k, -1] = 1.0
            else:
                c0, c1 = centers[b], centers[b + 1]
                w = (k - c0) / max(c1 - c0, 1e-6)
                S[k, b], S[k, b + 1] = 1 - w, w
        self.register_buffer("interp_mat", S)                   # band 增益→bin 增益

        # 特征维：22 log 能量 + 22 BFCC + 22 Δ + 22 ΔΔ = 88 →（官方 42 里含 pitch 6，
        # 我们用自己的可微特征集，网络入口宽度按特征定）
        feat_dim = N_BANDS * 4
        self.dense_in = nn.Sequential(nn.Linear(feat_dim, 24), nn.Tanh())
        self.gru_vad = nn.GRU(24, 24, batch_first=True)
        self.gru_noise = nn.GRU(24 + 24, 48, batch_first=True)
        self.gru_denoise = nn.GRU(24 + 24 + 48, 96, batch_first=True)
        self.dense_out = nn.Sequential(nn.Linear(96, N_BANDS), nn.Sigmoid())
        self.dense_vad = nn.Sequential(nn.Linear(24, 1), nn.Sigmoid())
        # DCT 矩阵（BFCC）
        D = torch.zeros(N_BANDS, N_BANDS)
        for i in range(N_BANDS):
            for j in range(N_BANDS):
                D[i, j] = math.cos(math.pi * i * (j + 0.5) / N_BANDS)
        self.register_buffer("dct", D * math.sqrt(2.0 / N_BANDS))

    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def features(self, mag2):
        """mag2: (B, T, bins) 功率谱 → (B, T, 88) 特征（全可微）。"""
        be = mag2 @ self.band_mat.t()                           # (B,T,22) 带能量
        lbe = torch.log10(be + 1e-8)
        bfcc = lbe @ self.dct.t()
        d1 = torch.diff(lbe, dim=1, prepend=lbe[:, :1])
        d2 = torch.diff(d1, dim=1, prepend=d1[:, :1])
        return torch.cat([lbe, bfcc, d1, d2], dim=-1)

    def forward(self, wav):
        """wav: (B, K) → (B, K) 增强波形；保长。"""
        B, K = wav.shape
        spec = torch.stft(wav, N_FFT, HOP, window=self.window, return_complex=True,
                          center=True, pad_mode="constant")     # (B, bins, T)
        spec = spec.transpose(1, 2)                             # (B, T, bins)
        feats = self.features(spec.abs() ** 2)
        h = self.dense_in(feats)
        v, _ = self.gru_vad(h)
        n_, _ = self.gru_noise(torch.cat([h, v], -1))
        d, _ = self.gru_denoise(torch.cat([h, v, n_], -1))
        gains = self.dense_out(d)                               # (B, T, 22)
        # 可选策略钩子（fo_train_rl --space bands）：22 带 log 偏置 + 全局指数，作用在带增益上
        gb, ge = getattr(self, "gain_bias", None), getattr(self, "gain_exp", None)
        if gb is not None:
            gains = (gains * gb.to(gains.device).exp()).clamp(0.0, 1.0)
        if ge is not None:
            gains = gains.clamp_min(1e-6) ** float(ge)
        g_bin = gains @ self.interp_mat.t()                     # (B, T, bins)
        out = torch.istft((spec * g_bin).transpose(1, 2), N_FFT, HOP,
                          window=self.window, center=True, length=K)
        return out

    def vad(self, wav):
        spec = torch.stft(wav, N_FFT, HOP, window=self.window, return_complex=True,
                          center=True, pad_mode="constant").transpose(1, 2)
        h = self.dense_in(self.features(spec.abs() ** 2))
        v, _ = self.gru_vad(h)
        return self.dense_vad(v).squeeze(-1)


if __name__ == "__main__":
    m = RNNoiseTorch()
    print(f"可训参数: {m.n_trainable()/1e3:.1f}K")
    x = torch.randn(2, 32000)
    y = m(x)
    assert y.shape == x.shape, (y.shape, x.shape)
    loss = ((y - x) ** 2).mean()
    loss.backward()
    g = sum(p.grad.norm() for p in m.parameters() if p.grad is not None)
    print(f"前向保长 ✓  反传梯度范数 {float(g):.4f} ✓")
