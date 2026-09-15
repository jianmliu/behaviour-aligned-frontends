#!/usr/bin/env python3
"""R-joint：RNNoiseTorch × 冻结 Lychee 的 control-CE 微调（噪声篇主训练）。

数据流：FDB 727 干净轨迹重放（LycheeFD_clean）+ 近端加噪（NoiseScene）→ RNNoise 增强
→ 冻结 12.8B → control CE → 只更 RNNoise（77.9K）。
起点 = 信号 loss 预训练 ckpt（rnnoise_pretrain/final.pt）。

    python train_rnnoise_joint.py --lychee-repo ... --lychee-ckpt ... \
        --traces $WORK/fdb_out/LycheeFD_clean --fdb-root $WORK/data/fdb/v1.0 \
        --noise-dir $WORK/data/demand --rnnoise-init $WORK/rnnoise_pretrain/final.pt \
        --out-dir $WORK/exp/rnj/run1 --epochs 2
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time

import numpy as np
import soundfile as sf
import torch

SP = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SP))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lychee-repo", type=pathlib.Path, required=True)
    ap.add_argument("--lychee-ckpt", type=pathlib.Path, required=True)
    ap.add_argument("--traces", type=pathlib.Path, required=True)
    ap.add_argument("--fdb-root", type=pathlib.Path, required=True)
    ap.add_argument("--noise-dir", type=pathlib.Path, required=True)
    ap.add_argument("--rnnoise-init", default=None, help="信号预训练 ckpt；None=scratch")
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-sec", type=float, default=20.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--loss-mode", choices=["ce", "soft"], default="ce",
                    help="soft = 软落点事件级 loss（soft_placement.py）+ λ_ce·CE 正则："
                         "目标从帧 token 换成首响落点期望成本，refrain 样本(24%%)监督克制。"
                         "攻击 CE 的帧级稀疏与两吸引子不可调和问题（08-30 消融定盘）")
    ap.add_argument("--lambda-ce", type=float, default=0.1,
                    help="soft 模式下原 control-CE 正则权重（防终止/附和等非首响行为漂移）")
    ap.add_argument("--w-pre", type=float, default=3.0, help="premature 成本（克制旋钮）")
    ap.add_argument("--w-miss", type=float, default=1.0, help="miss 成本（可听性旋钮）")
    ap.add_argument("--w-false", type=float, default=3.0, help="refrain 样本误触发成本")
    ap.add_argument("--pre-ramp", type=float, default=0.0, help="premature 斜坡长度(s)，0=平顶")
    ap.add_argument("--max-steps", type=int, default=0, help="冒烟用：>0 时跑 N 步即退")
    ap.add_argument("--targets-dir", type=pathlib.Path, default=None,
                    help="soft 模式外部目标树（fdb_targets.py 的标注 t*）；默认用自蒸馏（干净轨迹首响）")
    ap.add_argument("--ctrl-shift", type=int, default=1,
                    help="logits→帧对齐；★上机冒烟先跑 soft_placement.verify_shift 定值")
    ap.add_argument("--lambda-sig", type=float, default=0.0,
                    help="SI-SNR 信号锚权重；>0 时 loss = ctrl-CE + λ·(-SI-SNR)。检验假设：r0 的高"
                         " miss 是否因纯 CE 无信号锚、前端弃守可听性（CE 多数类先验压向 keep）")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--lambda-text", type=float, default=0.0)
    ap.add_argument("--snr-range", type=float, nargs=2, default=(-5.0, 20.0))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--resume", default=None)
    return ap.parse_args()


def main() -> int:
    a = parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    sys.path.insert(0, "/ephemeral/work/geic_nemo/src")

    from rnnoise_torch import RNNoiseTorch
    from geic_nemo.frontend import lychee_hook
    lychee_hook.set_repo(a.lychee_repo)
    import noise_hook
    # train_lychee_joint 的 pack 逻辑（复用同一实现，避免两份）
    import importlib.util as iu
    spec = iu.spec_from_file_location(
        "tlj", "/ephemeral/work/geic_nemo/scripts/train_lychee_joint.py")
    tlj = iu.module_from_spec(spec)
    tlj.__dict__["__name__"] = "tlj"
    src = open("/ephemeral/work/geic_nemo/scripts/train_lychee_joint.py").read()
    src = src.split("if __name__")[0]                    # 只要函数定义
    exec(compile(src, "tlj", "exec"), tlj.__dict__)

    fe = RNNoiseTorch()
    if a.rnnoise_init:
        fe.load_state_dict(torch.load(a.rnnoise_init, map_location="cpu", weights_only=True))
    model = noise_hook.NoiseJointFrontend(fe, str(a.lychee_ckpt), lambda_text=a.lambda_text)
    fe.to(model.llm.device)          # enhanced_mel 的输出设备跟 fe 走，必须与 llm 同卡
    print(f"[model] RNNoise 可训 {sum(p.numel() for p in fe.parameters())/1e3:.1f}K | "
          f"Lychee 冻结 | init={a.rnnoise_init}")

    # 样本清单：干净轨迹 + 近端 wav（far-end 不需要）
    import hashlib
    items = []
    for p in sorted(a.traces.glob("*/*/events.json")):
        task, sid = p.parent.parent.name, p.parent.name
        near = a.fdb_root / task / sid / "input.wav"
        if not near.exists():
            continue
        if sf.info(str(near)).duration > a.max_sec:
            continue
        items.append({"task": task, "sid": sid, "trace": p, "near": near})
    def is_val(it):
        h = int(hashlib.sha1(f"{it['task']}_{it['sid']}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return h < a.val_frac
    train_items = [i for i in items if not is_val(i)]
    print(f"[data] train {len(train_items)} / val {len(items)-len(train_items)}")

    scenes = noise_hook.NoiseScene(sorted(a.noise_dir.glob("*/ch01.wav")),
                                   snr_range=tuple(a.snr_range))
    opt = torch.optim.AdamW(fe.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s+1)/max(a.warmup,1)))
    gen = torch.Generator(); gen.manual_seed(a.seed)

    start_ep, step0 = 1, 0
    if a.resume and pathlib.Path(a.resume).exists():
        ck = torch.load(a.resume, map_location="cpu", weights_only=True)
        fe.load_state_dict(ck["fe"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_ep, step0 = int(ck["epoch"]), int(ck["step"])
        print(f"[resume] ep {start_ep} step {step0}")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "args.json").write_text(json.dumps(vars(a), default=str, indent=1))
    logf = open(a.out_dir / "train_log.csv", "a")

    def rd(p):
        x, sr = sf.read(str(p), dtype="float32", always_2d=True); x = x.mean(1)
        if sr != 16000:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sr, 16000); x = resample_poly(x, 16000//g, sr//g).astype("float32")
        return torch.from_numpy(x)

    if a.loss_mode == "soft":
        from soft_placement import soft_placement_loss, target_frame_from_events
        model.return_control_logits = True
        for it in train_items:                      # 事件级目标：respond 样本带 t*，
            src = (a.targets_dir / it["task"] / it["sid"] / "events.json") if a.targets_dir else it["trace"]
            it["t_star"] = target_frame_from_events(src) if src.exists() else None   # refrain 样本 None
        n_ref = sum(1 for it in train_items if it["t_star"] is None)
        print(f"[soft] respond {len(train_items)-n_ref} / refrain {n_ref}")

    step, t0 = step0, time.time()
    for ep in range(start_ep, a.epochs + 1):
        random.shuffle(train_items)
        for it in train_items:
            trace = json.loads(it["trace"].read_text())
            pack = tlj.lychee_hook_pack(trace, model.llm.device, model.llm.config, a.lychee_ckpt)
            scene = scenes(rd(it["near"])[None], gen)
            loss, parts = model(scene, pack)
            if a.loss_mode == "soft":
                logits = parts.pop("control_logits")
                if step == step0:                # 内建护栏：首步校验 shift 约定
                    from soft_placement import verify_shift
                    vs = verify_shift(logits, pack["control_input_ids"],
                                      model.llm.config, pack["prefix_input_ids"].shape[1])
                    print(f"[soft] verify_shift={vs}（用 --ctrl-shift {a.ctrl_shift}）", flush=True)
                    if max(vs.values()) < 0.5:
                        raise SystemExit(f"shift 校验失败 {vs}：logits/token 对齐错，勿训")
                    if vs.get(a.ctrl_shift, 0) < max(vs.values()) - 0.05:
                        raise SystemExit(f"--ctrl-shift {a.ctrl_shift} 非最优 {vs}，改参重跑")
                sp_loss, sp_diag = soft_placement_loss(
                    logits, model.llm.config,
                    pack["prefix_input_ids"].shape[1], it["t_star"],
                    shift=a.ctrl_shift, w_pre=a.w_pre, w_miss=a.w_miss, w_false=a.w_false,
                    pre_ramp_sec=(a.pre_ramp or None))
                loss = sp_loss + a.lambda_ce * loss          # CE 降为正则
                parts["soft"] = float(sp_loss); parts.update(sp_diag)
            if a.lambda_sig > 0:
                dev = next(fe.parameters()).device
                enh = fe(scene["noisy"].float().to(dev))
                ref = scene["clean"].float().to(dev)
                # SI-SNR（scale-invariant）：零均值化→投影→残差
                e, r = enh - enh.mean(-1, keepdim=True), ref - ref.mean(-1, keepdim=True)
                t = (e * r).sum(-1, keepdim=True) * r / (r.pow(2).sum(-1, keepdim=True) + 1e-8)
                sisnr = 10 * torch.log10(t.pow(2).sum(-1) / ((e - t).pow(2).sum(-1) + 1e-8) + 1e-8)
                loss = loss + a.lambda_sig * (-sisnr.mean())
                parts["sisnr"] = float(sisnr.mean())
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(fe.parameters(), a.clip)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if a.max_steps and step - step0 >= a.max_steps:
                print(f"[smoke] {a.max_steps} 步完成，soft={parts.get('soft')}", flush=True)
                return 0
            if step % a.log_every == 0:
                extra = (f" soft {parts['soft']:.3f} trig {parts['trigger_mass']:.2f}"
                         if "soft" in parts else "")
                print(f"[rnj {time.strftime('%T')}] ep {ep} step {step}: "
                      f"ctrl {parts['control']:.4f}{extra} ‖g‖ {float(gn):.3f} "
                      f"{(time.time()-t0)/max(step-step0,1):.1f}s/步", flush=True)
                logf.write(f"{ep},{step},{parts['control']},{float(gn)},"
                           f"{parts.get('soft','')},{parts.get('trigger_mass','')}\n"); logf.flush()
            if step % a.save_every == 0:
                torch.save({"fe": fe.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "step": step, "epoch": ep},
                           a.out_dir / "last.pt")
        torch.save({"fe": fe.state_dict(), "step": step, "epoch": ep},
                   a.out_dir / f"epoch_{ep:03d}.pt")
        print(f"[rnj] epoch {ep} 完成", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
