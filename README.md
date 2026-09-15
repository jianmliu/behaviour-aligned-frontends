# Behaviour-Aligned Front-Ends for Full-Duplex Speech LLMs: Signal Restoration Is Not Behaviour Restoration

Reproduction package for the ICASSP 2027 submission of the same title (`paper/main.pdf`).

The paper asks one question: when a 78k-parameter RNNoise-style suppressor sits in front of a
frozen full-duplex speech LLM, should it be trained to restore the *signal* or to serve the
model's *turn-taking decision*? The gradient-free (evolution-strategies) arm is kept as a probe.
Everything here is organised around the arms of that comparison.

| Arm | Model | Objective | Optimiser | Where it runs |
|---|---|---|---|---|
| Lychee-FD (12.8B) | frozen | control-token cross-entropy through the model | Adam (gradients) | 1×H100, `lychee/` |
| Freeze-Omni (7B) | frozen | expected first-response placement cost | Adam (gradients) | Mac mini M4 32GB, MPS, `freeze_omni/` |
| Freeze-Omni (7B) | frozen | realised placement reward (decisions only) | antithetic ES | same |

Both arms share the front-end (`common/rnnoise_torch.py`), the placement metric
(`common/humdial_metrics3.py`), the noise protocol (`common/humdial_prep.py` /
`freeze_omni/fo_placement.py`) and the event-level loss (`common/soft_placement.py`).

## Layout

```
paper/            main.tex, refs.bib, figure, style files, compiled PDF
common/           front-end, placement metric, noise protocol, soft-placement loss (+ unit tests), sign test
common/rnnoise_pretrained_final.pt   signal-pretrained RNNoise-style checkpoint (the shared starting point)
lychee/           Lychee-FD arm: offline full-duplex decoding, joint training (CE / soft / SI-SNR anchor), ES
lychee/queue/     the exact H100 job scripts, in execution order (05 … 33) + the run manual (zh)
lychee/results/   per-scenario placement counts for every table row (JSON)
lychee/checkpoints/rnj_r{0..3}/   behaviour-trained front-ends: r0 = R-joint, r1/r2/r3 = ablations
freeze_omni/      Freeze-Omni arm: placement evaluation, differentiable whole-utterance forward,
                  soft-placement training, ES training, first-response text generation, LLM judge
freeze_omni/chains/      the shell chains that produced every Freeze-Omni number (dose sweep, harsh regimes,
                         ES budget variants, dense-reward ES, text generation)
freeze_omni/checkpoints/ trained front-ends (epoch_002.pt) and ES policies (last.pt) per regime
freeze_omni/results/     summary tables (fo_tables_*.txt), judge items and scores (jsonl)
freeze_omni/freeze_omni_mps.patch   diff that makes the official Freeze-Omni repo run on Apple MPS
artifacts/        per-sample outputs (events.json / text.json per HumDial sample) for both arms, as tarballs
docs/             the original experiment plan and the batched-decoding design note (zh)
```

Symlinks inside `lychee/` and `freeze_omni/` point at `common/` so every script runs from its own directory.

## Data and weights (not included)

| Item | Source | Used for |
|---|---|---|
| HumDial English test set (`en_test_nondev`, 4,550 recordings, 10 scenarios) | ICASSP 2026 HumDial challenge release | evaluation (100/scenario; 20/scenario for the rows marked n=200) |
| Full-Duplex-Bench v1.0 user audio (727 recordings) | Full-Duplex-Bench | front-end training input; turn-end annotations are the Freeze-Omni placement targets |
| DEMAND (16 kHz, `ch01.wav`) | DEMAND corpus | noise; environments held out between pretraining and test |
| Lychee-FD weights | HIT-TMG/Lychee-FD | `lychee/` |
| Freeze-Omni checkpoints + Qwen2-7B-Instruct | VITA-MLLM/Freeze-Omni | `freeze_omni/` (Qwen2-7B-Instruct is also the relevance judge) |

Per-sample noise is seeded from the sample id, so every system sees bit-identical mixtures; all
comparisons in the paper are paired on those samples.

## Environment

```
pip install -r requirements.txt
```

Freeze-Omni on Apple silicon additionally needs the official repo patched:

```
git clone https://github.com/VITA-MLLM/Freeze-Omni repo && cp -r repo repo_mps
cd repo_mps && patch -p1 < ../freeze_omni/freeze_omni_mps.patch
```

The patch rewrites `cuda`→`mps`, adds the missing `make_pad_mask` import in the encoder, and silences
a per-chunk print. `transformers` must stay at 4.45.2 (5.x changes the KV-cache type the repo assumes).
The exact mini environment is in `freeze_omni/requirements-mini.txt`. Audio is truncated to 12 s for
training on 32 GB (20 s costs ~60 s/step on MPS).

## Reproducing the paper

### Lychee-FD (Table 1, Sec. 3.1–3.2, ablations 3.4)

Run the queue scripts in numeric order on one H100 (`lychee/gpu_queue.sh` is the daemon that does this
unattended). Each script states its inputs and outputs in its header.

| Row / result | Script |
|---|---|
| clean (L0) | `queue/05-hd-clean.sh` |
| noise, no front-end (N0) | `queue/10-hd-N0.sh` |
| +RN-t frozen (R-indep) | `queue/15-hd-Rindep.sh` |
| R-joint training (r0) and ablations r1/r2/r3 | `queue/20-rnj-train.sh`, `26`, `27`, `28` |
| +RN-t behaviour-trained evaluation | `queue/25-hd-Rjoint.sh` |
| soft-placement and ES variants on Lychee (prepared, not run for the paper) | `queue/29`–`33` |

Placement counts per scenario are produced by `common/humdial_metrics3.py --root <eval tree> --humdial <HumDial>`;
the JSON files in `lychee/results/` are those outputs. Paired McNemar tests are computed from the per-sample
trees in `artifacts/lychee_per_sample_outputs.tgz`.

### Freeze-Omni (Sec. 3.3)

Everything ran on the Mac mini from `freeze_omni/`, with `repo_mps/`, `data/` and `final.pt`
(= `common/rnnoise_pretrained_final.pt`) in the same directory. The chains in `freeze_omni/chains/`
are the literal commands; the table below maps paper statements to them.

| Paper statement | Chain / command |
|---|---|
| clean profile, protocol-SNR noise, frozen and trained front-ends at 0–15 dB (n=1000) | `chains/overnight*.sh` (evaluations via `fo_placement.py`) |
| dose curve −5/−10/−15/−20 dB, no front-end | `chains/sweep.sh` then `chains/harsh1000.sh <SNR>` |
| four systems per harsh regime (no front-end / frozen / gradient / ES), n=200 then n=1000 | `chains/harsh.sh <SNR> <train_lo> <train_hi>` then `chains/harsh1000.sh <SNR>` |
| ES budget variants at −15 dB: all-parameter space, 120 generations | `chains/es_more.sh`, `chains/es_more1000.sh` |
| ES with the gradient arm's own dense loss as reward | `chains/dense_es.sh` (`fo_train_rl.py --reward dense`) |
| first-response text at −15 dB for three systems | `chains/gen_text.sh` (`fo_gen_text.py`) |
| relevance judge (Qwen2-7B-Instruct, 0/1/2) | `fo_judge_items.py` then `fo_judge_qwen.py` |
| paired sign tests on first-response instants (|Δ| ≤ 0.2 s counts as tied) | `common/fo_signtest.py <tree A> <tree B>` |
| differentiable whole-utterance forward vs streaming decode agreement | `fo_soft_forward.py --check` |

Summary tables for every regime are in `freeze_omni/results/fo_tables_*.txt`; the judge inputs and
outputs are `items_*_m15.jsonl` and `scores_all_m15.jsonl` (scores are recovered from the `raw` field with
`"score": <n>` when the JSON was truncated). Per-sample outputs are in
`artifacts/freeze_omni_per_sample_outputs.tgz`.

Training budgets used in the paper: soft placement 2 epochs over 727 FDB recordings (~1.4k steps,
`--w-pre 3 --w-miss 1 --pre-ramp 20 --w-margin 1.0`); ES `--pop 6 --m 6 --gens 40` (3k rollouts) in the
23-dimensional band-gain space, `--sigma 0.02` for the all-parameter space, 120 generations for the long run.
Training noise for a regime evaluated at S dB is drawn from U[S−5, S+5] (U[−20, −10] for −15 dB, etc.).

## Unit tests

```
python common/test_soft_placement.py
```

covers the first-trigger distribution, the placement-cost geometry, the refrain branch, the premature-plateau
ramp and the logit-margin term, and target extraction from real trajectories.

## Notes on what is and is not here

- The figure `paper/fig_placement.pdf` is included; the one-off script that drew it from
  `lychee/results/*.json` was not preserved.
- Lychee-FD ES and soft-placement jobs (`queue/29`–`33`) are complete and smoke-tested but were not run for
  the submitted paper; the gradient-free probe is reported on Freeze-Omni only, as the paper states.
- The Freeze-Omni relevance judge runs the 7B text model on MPS at ~3.4 s/item; 3,000 items take ~3 h.

## Citation

```
@inproceedings{liu2027behaviour,
  title     = {Behaviour-Aligned Front-Ends for Full-Duplex Speech {LLM}s: Signal Restoration Is Not Behaviour Restoration},
  author    = {Liu, Jianming},
  booktitle = {Proc. ICASSP},
  year      = {2027}
}
```

MIT license (see `LICENSE`). Third-party code is not redistributed: the Freeze-Omni repo is patched in place,
Lychee-FD is imported from its own package, and RNNoise's band layout is re-implemented in
`common/rnnoise_torch.py`.
