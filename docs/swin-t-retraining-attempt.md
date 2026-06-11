# Retraining attempts report: Swin-T backbone swap (and R50 prompt retraining)

> **Role of this document.** Before settling on the eval-only experiments reported in the README
> (threshold sensitivity + logit-delta search), we attempted to *retrain* parts of ECLIPSE on Colab.
> Both attempts failed to approach the released checkpoints within a student compute budget, and
> that failure is what redirected the project toward inference-time experiments. The raw logs are
> preserved in [`../results/logs/`](../results/logs/). Negative results are reported here in full
> because they shaped the project and because they independently support the paper's premise (see
> "What this taught us" at the end).

## Summary

We attempted to improve the ECLIPSE ADE20K-Panoptic continual segmentation reproduction by replacing the original ResNet-50 backbone with Swin-T, while keeping evaluation single-scale and without test-time augmentation.

The goal was to test whether a stronger frozen backbone could improve PQ without changing the continual prompt-learning premise of ECLIPSE. To keep the experiment practical, the first Swin-T base run used a shortened 32k-iteration base schedule instead of a full paper-length training budget.

The early results were substantially lower than the original ResNet-50 ECLIPSE article result, especially for the 100-50 setting. Because of that, we stopped the Swin-T 32k attempt instead of continuing to the full 100-10 and 100-5 scenarios or producing the final comparison graph.

## Files Found

The relevant recovered files were:

- `C:/Users/danie/Downloads/ade_ps_swin_t_100_step0_32k.pth`
- `C:/Users/danie/Downloads/output_swin_t_base100_32k.txt`
- `C:/Users/danie/Downloads/output_swin_t_base100.txt`
- `C:/Users/danie/Downloads/output_swin_t_100_50.txt`

`output_swin_t_base100.txt` and `output_swin_t_base100_32k.txt` are the same base run log under two different names.

## Experiment Setup

Backbone:

- Swin-Tiny
- Config: `configs/ade20k/panoptic-segmentation/swin/maskformer2_swin_t_bs16_160k.yaml`
- No TTA
- Single GPU
- `SOLVER.IMS_PER_BATCH = 8`
- Torch CUDA runtime shown in logs: CUDA 12.8

Base step:

- `CONT.TASK = 0`
- `CONT.BASE_CLS = 100`
- `CONT.INC_CLS = 50`
- `CONT.NUM_PROMPTS = 0`
- `SOLVER.MAX_ITER = 32000`
- `SOLVER.CHECKPOINT_PERIOD = 32000`
- `TEST.EVAL_PERIOD = 32000`
- `SOLVER.BASE_LR = 0.0001`
- Output checkpoint: `ade_ps_swin_t_100_step0_32k.pth`

100-50 prompt step:

- `CONT.TASK = 1`
- `CONT.BASE_CLS = 100`
- `CONT.INC_CLS = 50`
- `CONT.NUM_PROMPTS = 50`
- `CONT.WEIGHTS = results/ade_ps_swin_t_100_step0.pth`
- `SOLVER.MAX_ITER = 8000`
- `SOLVER.CHECKPOINT_PERIOD = 500000`
- `TEST.EVAL_PERIOD = 8000`
- `SOLVER.BASE_LR = 0.0005`

## Results

### Base 100-Class Swin-T Run

From `output_swin_t_base100_32k.txt`:

| Metric | Value |
|---|---:|
| All PQ | 14.6626 |
| SQ | 35.2695 |
| RQ | 17.3524 |
| PQ_base | 21.9939 |
| PQ_new | 0.0000 |
| PQ_novel | 0.0000 |

The base run completed successfully and saved:

`results/ade_ps_swin_t_100_step0.pth`

Recovered local file:

`C:/Users/danie/Downloads/ade_ps_swin_t_100_step0_32k.pth`

### Swin-T 100-50 Run

From `output_swin_t_100_50.txt`, the final evaluation block reported:

| Metric | Value |
|---|---:|
| All PQ | 18.5832 |
| SQ | 52.6457 |
| RQ | 22.4895 |
| PQ_base | 25.4424 |
| PQ_new | 4.8647 |
| PQ_novel | 4.8647 |

The log also contained an earlier evaluation block:

| Metric | Value |
|---|---:|
| All PQ | 17.3085 |
| PQ_base | 23.3412 |
| PQ_new | 5.2433 |
| PQ_novel | 5.2433 |

The final 100-50 result was therefore:

`All PQ = 18.5832`

## Comparison Against Original ECLIPSE Result

The original ECLIPSE article value used in our comparison notebooks for ADE20K-Panoptic 100-50 was:

`100-50 All PQ = 35.6`

Our Swin-T 32k-base attempt reached:

`100-50 All PQ = 18.5832`

Difference:

| Run | 100-50 All PQ |
|---|---:|
| Original ECLIPSE article / ResNet-50 reference | 35.6 |
| Swin-T 32k base + 8k 100-50 | 18.5832 |
| Absolute gap | -17.0168 |

This was far below the ResNet-50 reference, so the result did not support continuing the 32k Swin-T run into the full 100-10 and 100-5 experiments.

## Decision

We stopped the Swin-T 32k attempt because the first meaningful continual result, 100-50, was much lower than the ResNet-50 reference result.

This does not prove Swin-T is a bad backbone for ECLIPSE. It only shows that this specific Swin-T setup, using a 32k base schedule and 8k 100-50 prompt stage, was not competitive enough to justify further compute at that time.

Likely reasons include:

- The 32k base schedule may have been too short for Swin-T to learn a strong ADE20K panoptic base model.
- The Swin-T integration may need more tuning than a direct ResNet-50 replacement.
- The prompt stages depend heavily on the quality of the frozen base model.
- The original ECLIPSE ResNet-50 reference had a much stronger established training recipe.

## Conclusion

The Swin-T 32k experiment was useful as an early feasibility check. It successfully trained and produced a base checkpoint and a 100-50 result, but the PQ was too low compared with the original ResNet-50 ECLIPSE reference.

Because the 100-50 All PQ was only `18.5832` compared with the reference `35.6`, we stopped this attempt and shifted focus toward other backbone strategies and more robust checkpointing workflows.

## Addendum: R50 prompt-retraining attempt (100-10)

Separately from the Swin-T swap, we also attempted to retrain the **100-10 prompt stages on the
original R50 architecture** from a self-trained base checkpoint
(log: [`../results/logs/output_100_10.txt`](../results/logs/output_100_10.txt)). The first launch
crashed immediately with a real lesson about Detectron2's CLI:

```
AssertionError: Override list has odd length: ['OUTPUT_DIR', 'results/ade_ps', ...,
'WANDB', 'False', '--eval-only']; it must be a list of pairs
```

(`--eval-only` was placed *after* the config overrides, so yacs swallowed it into the key/value
list — flags must come before the `opts`.) After fixing the launch, the prompt stages trained, but
the final evaluation reached only **PQ 17.48** — roughly half the released checkpoint's 33.84 —
because our self-trained base step was far weaker than the authors' full 160k-iteration base model.

## Compute cost (why we pivoted)

| Run | Iterations | Wall-clock (Colab GPU) | Outcome |
|---|---:|---:|---|
| Swin-T base, 100 classes | 32,000 | **6 h 01 m** | PQ 14.66 — too weak a base |
| Swin-T 100-50 prompt step | 8,000 | 35 m | PQ 18.58 (novel 4.86) |
| R50 100-10 prompt retraining | 16,000/task | hours/task × 5 tasks | PQ 17.48 — abandoned |
| **Any eval-only experiment** | — | **~5 m per scenario** | basis of all final results |

A *single* under-trained base consumed a full Colab session; the paper's actual base schedule
(160k iterations) is ~5× that, per backbone, before any continual step. That arithmetic is what
moved the project to eval-only science: threshold sensitivity and logit-delta search deliver
measured, reproducible findings at ~5 minutes per run.

## What this taught us (and what it says about ECLIPSE)

The Swin-T failure mode is informative, not just disappointing: with a weak frozen base
(PQ 14.66), the prompt stage could only reach **4.86 novel PQ** — the continual mechanism had
nothing solid to build on. This is direct, independent evidence for the paper's central premise:
**ECLIPSE's prompt tuning inherits the quality of the frozen base model**. The method's published
numbers depend on a well-trained base, exactly as the authors' design assumes. Together with our
reproduction (±0.1 PQ) and the delta-search negative result (the authors' hand-tuned deltas were
already optimal within our grid), the retraining attempts complete a consistent picture: the
ECLIPSE results are real, well-tuned, and not easy to improve on a student budget.

