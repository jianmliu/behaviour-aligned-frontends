#!/usr/bin/env python3
"""HumDial 评测树物化（噪声篇）：把 en_test_nondev/<scen>/<id>.wav 规范化为
<scen>/<id>/input.wav 结构（fdb_infer_lychee 直接可跑），可选加噪声/前端增强。

    # 干净树（L0 用）
    python humdial_prep.py --mode clean --out .../hd_clean --per-scen 100
    # 噪声树（N0）：混合噪声分布（环境×SNR 按样本 seed 固定 → 系统间配对）
    python humdial_prep.py --mode noise --out .../hd_N0 --per-scen 100
    # 前端增强树（R-*）：噪声后过 RNNoiseTorch
    python humdial_prep.py --mode noise --rnnoise <ckpt> --out .../hd_R --per-scen 100
"""
import argparse, hashlib, math, pathlib, random, sys

import numpy as np
import soundfile as sf
import torch

SP = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
SR = 16000

ap = argparse.ArgumentParser()
ap.add_argument("--humdial", type=pathlib.Path,
                default=SP / "humdial/test/test/en_test_nondev")
ap.add_argument("--noise-dir", type=pathlib.Path, default=SP / "demand")
ap.add_argument("--mode", choices=["clean", "noise"], required=True)
ap.add_argument("--rnnoise", default=None, help="RNNoiseTorch ckpt（加噪后增强）")
ap.add_argument("--policy", default=None, help="train_rnnoise_es 的 best.pt（bands: gain_bias/gain_exp 钩子）")
ap.add_argument("--snr", type=float, nargs=2, default=(0.0, 15.0), help="SNR 采样范围 dB")
ap.add_argument("--out", type=pathlib.Path, required=True)
ap.add_argument("--per-scen", type=int, default=100)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--num-shards", type=int, default=1)
a = ap.parse_args()

model = None
if a.rnnoise:
    from rnnoise_torch import RNNoiseTorch
    model = RNNoiseTorch()
    model.load_state_dict(torch.load(a.rnnoise, map_location="cpu", weights_only=True))
    if a.policy:
        pol = torch.load(a.policy, map_location="cpu", weights_only=True); th = pol["theta"]
        if pol.get("space") == "bands":
            model.gain_bias = th[:22]; model.gain_exp = float(th[22].exp())
        else:
            base = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
            torch.nn.utils.vector_to_parameters(base + th, model.parameters())
        print(f"[policy] {a.policy} space={pol.get('space')} |θ|={float(th.norm()):.3f}", flush=True)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

noises = []
if a.mode == "noise":
    for p in sorted(a.noise_dir.glob("*/ch01.wav")):
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        noises.append(x.mean(1))
    assert noises, "噪声库为空"

samples = []
for scen_dir in sorted(a.humdial.iterdir()):
    if not scen_dir.is_dir():
        continue
    wavs = sorted(scen_dir.glob("*.wav"))[: a.per_scen]
    samples += [(scen_dir.name, w) for w in wavs]
samples = samples[a.shard :: a.num_shards]
print(f"[shard {a.shard}/{a.num_shards}] {len(samples)} 条  mode={a.mode} rnnoise={bool(model)}")

done = 0
for scen, wav_p in samples:
    sid = wav_p.stem
    out = a.out / scen / sid / "input.wav"
    if out.exists():
        continue
    x, sr = sf.read(str(wav_p), dtype="float32", always_2d=True)
    x = x.mean(1)
    assert sr == SR, (wav_p, sr)
    if a.mode == "noise":
        rng = random.Random(int(hashlib.sha1(f"{scen}/{sid}".encode()).hexdigest()[:8], 16))
        nz = noises[rng.randrange(len(noises))]
        off = rng.randrange(0, max(len(nz) - len(x), 1))
        n = nz[off: off + len(x)]
        if len(n) < len(x):
            n = np.tile(n, math.ceil(len(x) / max(len(n), 1)))[: len(x)]
        snr = rng.uniform(*a.snr)
        ps, pn = (x ** 2).mean() + 1e-9, (n ** 2).mean() + 1e-9
        x = x + n * math.sqrt(ps / pn / (10 ** (snr / 10)))
    if model is not None:
        with torch.no_grad():
            t = torch.from_numpy(x)[None]
            if next(model.parameters()).is_cuda:
                t = t.cuda()
            x = model(t)[0].cpu().numpy()
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), x, SR)
    done += 1
    if done % 100 == 0:
        print(done, flush=True)
print(f"done {done}")
