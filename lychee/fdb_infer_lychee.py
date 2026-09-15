#!/usr/bin/env python3
"""Lychee-FD 在 FDB 上的干净基线推理——验证其发表数字可复现（第二模型候选的第一步）。

背景（docs/04 E37）
------------------
Lychee-FD（HITsz-TMG，10B，Whisper-v3-large 连续入口，24 共享层 + text4/speech4/control2 头）
就是在 Full-Duplex-Bench 1.0/1.5 上评测的——与本项目同一基准家族。对表目标（其论文 Table 2）：

    FDB 1.0:  I-TOR 94.5   B-Freq 14.6   B-TOR 23.4   T-TOR 98.3   P-TOR 10.0   Stop 840ms
    FDB 1.5:  IRR 78.0     BRR 69.0      Stop 570ms   Lat 826ms

本脚本对 FDB 树逐条跑其官方离线全双工状态机
（`SingleTurnGenerationFramework.full_chunk_stream_offline_generation`），存事件流。
事件自带 25 Hz 网格时间戳与类型（response/backchannel），TOR 类指标从 control 轨迹直接可算，
不需要 VAD，也不必先合成音频（token2wav 只在需要听效果时跑）。

前置
----
- 权重：models/lychee-fd/{lychee_full_duplex, token2wav}（本机已下载，rsync 到 GPU 盒子）
- 代码：third_party/Lychee-FD（含我们补的 lychee_fd/models/ shim——官方 repo 打包漏了这个
  模块；类接口若有出入，冒烟时对着报错在 shim 里补）
- 环境：他们 environment.yml 基于 torch/cuda12.1；先尝试直接用 voicechat 环境（torch 版本
  大概率兼容，缺什么 pip 补什么），不行再建独立 env

用法（GPU 盒子）
---------------
    # 冒烟：3 条样本，验证 shim + 状态机跑通、事件流合理
    python scripts/fdb_infer_lychee.py --fdb_root $WORK/data/fdb/v1.0 \
        --lychee_repo $WORK/third_party/Lychee-FD \
        --model_path $WORK/models/lychee-fd/lychee_full_duplex \
        --out_root $WORK/fdb_out/LycheeFD_clean --limit 3
    # 全量 727 条（预计与 VoiceChat L0 同量级，~1-2h）
    ... --limit 0
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fdb_root", type=pathlib.Path, required=True)
    ap.add_argument("--lychee_repo", type=pathlib.Path, required=True)
    ap.add_argument("--model_path", type=pathlib.Path, required=True)
    ap.add_argument("--out_root", type=pathlib.Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0, help="本进程分片号（8 卡并行：0..7）")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow_backchannel", action="store_true", default=True)
    a = ap.parse_args()

    sys.path.insert(0, str(a.lychee_repo))
    import numpy as np
    import soundfile as sf
    import torch
    from lychee_fd.runtime.hf_v9_generation import SingleTurnGenerationFramework

    t0 = time.time()
    fw = SingleTurnGenerationFramework(
        model_type="V9",
        model_path=str(a.model_path),
        device=a.device,
        torch_dtype=torch.bfloat16,
        allowing_backchannel=a.allow_backchannel,
    )
    print(f"[load] {time.time()-t0:.0f}s", flush=True)

    samples = sorted(p.parent for p in a.fdb_root.glob("*/*/input.wav"))
    if a.limit:
        samples = samples[: a.limit]
    samples = samples[a.shard:: a.num_shards]          # 8 卡数据并行：每卡一片
    print(f"[shard {a.shard}/{a.num_shards}] {len(samples)} 条", flush=True)
    done = skip = fail = 0
    for k, d in enumerate(samples):
        task, sid = d.parent.name, d.name
        out_dir = a.out_root / task / sid
        out_json = out_dir / "events.json"
        if out_json.exists():                      # 断点续跑
            skip += 1
            continue
        audio, sr = sf.read(str(d / "input.wav"), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sr != 16_000:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sr, 16_000)
            audio = resample_poly(audio, 16_000 // g, sr // g).astype(np.float32)
        try:
            with torch.inference_mode():
                res = fw.full_chunk_stream_offline_generation(
                    torch.from_numpy(audio), return_raw_trace=True)
        except Exception as e:                     # 单条失败不倒全局，记账后继续
            print(f"  ✗ {task}/{sid}: {type(e).__name__}: {e}", flush=True)
            fail += 1
            continue
        # 官方返回结构冒烟时确认：期望含 decoder() 的事件列表（start/end_time、text、type）
        # 与 raw trace（text/stoken/control 三序列）。原样落盘，指标离线算。
        # ★ raw trace 必须完整保留（generated_input_ids / generated_stoken_ids /
        #   generated_control_ids / audio_input_ids / prefix 长度）——联合训练的自蒸馏
        #   目标就是这套序列的 teacher-forcing 重放（docs/04 E38：三通道 embedding 相加、
        #   chunk 对齐、prefix 构造全部由模型自己生成的序列保证，不重新构造）。
        out_dir.mkdir(parents=True, exist_ok=True)

        def _ser(o):
            if isinstance(o, torch.Tensor):
                return o.tolist()
            raise TypeError(str(type(o)))

        out_json.write_text(json.dumps(res, default=_ser, ensure_ascii=False), encoding="utf-8")
        done += 1
        if (k + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  {k+1}/{len(samples)}  done={done} fail={fail}  {el/60:.0f}min", flush=True)

    print(f"\n完成 done={done} skip={skip} fail={fail} → {a.out_root}")
    return 0 if fail * 10 < max(done, 1) else 1    # 失败率 >10% 视为整体失败


if __name__ == "__main__":
    raise SystemExit(main())
