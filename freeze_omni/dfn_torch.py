#!/usr/bin/env python3
"""Differentiable PyTorch re-implementation of DeepFilterNet's libdf feature pipeline, so gradients can flow
from a downstream loss through DeepFilterNet3 back to the waveform.

Mirrors df.enhance.enhance():  pad -> analysis (STFT) -> [erb -> erb_norm], [unit_norm on first nb_df bins]
-> model -> synthesis (ISTFT) -> un-pad. Every stage is checked numerically against libdf in test_dfn_torch.py.
"""
import math

import torch
import torch.nn.functional as F


def vorbis_window(n_fft: int) -> torch.Tensor:
    h = n_fft // 2
    i = torch.arange(n_fft, dtype=torch.float64)
    s = torch.sin(0.5 * math.pi * (i + 0.5) / h)
    return torch.sin(0.5 * math.pi * s * s).float()


class DFTorch(torch.nn.Module):
    """Stateless (per-call) differentiable analysis/features/synthesis matching libdf.DF."""

    def __init__(self, sr=48000, fft_size=960, hop_size=480, erb_widths=None, nb_df=96, alpha=None,
                 mean_norm_init=(-60.0, -90.0), unit_norm_init=(0.001, 0.0001)):
        super().__init__()
        self.sr, self.n_fft, self.hop, self.nb_df = sr, fft_size, hop_size, nb_df
        self.register_buffer("win", vorbis_window(fft_size))
        self.wnorm = 1.0 / (fft_size ** 2 / (2 * hop_size))
        w = torch.as_tensor(erb_widths, dtype=torch.long)
        self.nb_erb = len(w)
        fb = torch.zeros(fft_size // 2 + 1, self.nb_erb)
        start = 0
        for b, width in enumerate(w.tolist()):
            fb[start:start + width, b] = 1.0 / width                    # band mean of |X|^2
            start += width
        self.register_buffer("erb_fb", fb)
        self.alpha = alpha if alpha is not None else math.exp(-hop_size / sr / 1.0)
        self.register_buffer("mean_init", torch.linspace(mean_norm_init[0], mean_norm_init[1], self.nb_erb))
        self.register_buffer("unit_init", torch.linspace(unit_norm_init[0], unit_norm_init[1], nb_df))

    # ---- STFT / ISTFT with libdf's streaming convention (frame t uses input up to (t+1)*hop) ----
    def analysis(self, x):                                              # x: [B, T] -> complex [B, Tf, F]
        B, T = x.shape
        assert T % self.hop == 0
        xp = F.pad(x, (self.n_fft - self.hop, 0))                       # analysis memory starts at zero
        frames = xp.unfold(-1, self.n_fft, self.hop)                    # [B, Tf, n_fft]
        return torch.fft.rfft(frames * self.win, dim=-1) * self.wnorm

    def synthesis(self, spec):                                          # complex [B, Tf, F] -> [B, Tf*hop]
        B, Tf, _ = spec.shape
        frames = torch.fft.irfft(spec, n=self.n_fft, dim=-1) * self.n_fft * self.win  # libdf irfft is unnormalised
        out = F.fold(frames.transpose(1, 2), output_size=(1, (Tf - 1) * self.hop + self.n_fft),
                     kernel_size=(1, self.n_fft), stride=(1, self.hop)).view(B, -1)
        return out[:, : Tf * self.hop]                                  # streaming output: first hop of each frame

    # ---- features ----
    def erb(self, spec):                                                # power per ERB band (mean over bins)
        return (spec.real ** 2 + spec.imag ** 2) @ self.erb_fb

    def erb_norm(self, e):                                              # 10log10 -> exp mean norm -> /40
        x = 10 * torch.log10(e + 1e-10)
        out, state = [], self.mean_init.expand(x.shape[0], -1)
        for t in range(x.shape[1]):
            state = x[:, t] * (1 - self.alpha) + state * self.alpha
            out.append((x[:, t] - state) / 40.0)
        return torch.stack(out, 1)

    def unit_norm(self, s):                                             # complex [B,Tf,nb_df]
        out, state = [], self.unit_init.expand(s.shape[0], -1)
        mag = s.abs()
        for t in range(s.shape[1]):
            state = mag[:, t] * (1 - self.alpha) + state * self.alpha
            out.append(s[:, t] / torch.sqrt(state))
        return torch.stack(out, 1)

    def features(self, x):
        spec = self.analysis(x)
        erb_feat = self.erb_norm(self.erb(spec)).unsqueeze(1)                        # [B,1,Tf,nb_erb]
        spec_feat = torch.view_as_real(self.unit_norm(spec[..., : self.nb_df])).unsqueeze(1)
        return torch.view_as_real(spec).unsqueeze(1), erb_feat, spec_feat


def enhance_torch(model, dft: DFTorch, audio48, pad=True):
    """Differentiable counterpart of df.enhance.enhance for audio48 [B, T] at the model rate."""
    orig = audio48.shape[-1]
    if pad:
        audio48 = F.pad(audio48, (0, dft.n_fft))
    rem = (-audio48.shape[-1]) % dft.hop
    if rem:
        audio48 = F.pad(audio48, (0, rem))
    if hasattr(model, "reset_h0"):
        model.reset_h0(batch_size=audio48.shape[0], device=audio48.device)
    spec, erb_feat, spec_feat = dft.features(audio48)
    enh = model(spec.clone(), erb_feat, spec_feat)[0]                                  # [B,1,Tf,F,2]
    out = dft.synthesis(torch.view_as_complex(enh.squeeze(1).contiguous()))
    if pad:
        d = dft.n_fft - dft.hop
        out = out[:, d: orig + d]
    return out
