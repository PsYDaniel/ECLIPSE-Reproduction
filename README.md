# Reproducing ECLIPSE: Continual Panoptic Segmentation on ADE20K

> Python course project — reproduction and analysis of the CVPR 2024 paper
> **ECLIPSE: Efficient Continual Learning in Panoptic Segmentation with Visual Prompt Tuning**.

**Author:** Daniel Chicherin, Or Blazer  ·  **Course:** Python  ·  **Instructor:** Andrey Dolgin  ·  **Date:** May 2026

**Project website:** [`https://PsYDaniel.github.io/ECLIPSE-reproduction/`](https://PsYDaniel.github.io/ECLIPSE-reproduction/)


---

## 1. What this project does

Continual learning asks a model to keep learning *new* categories over time without forgetting the
ones it already knows. This is hard for **panoptic segmentation** (labelling every pixel and
separating every object instance), because naively fine-tuning on new classes causes
*catastrophic forgetting* of the old ones.

**ECLIPSE** tackles this by *freezing* the entire base segmentation model (a Mask2Former with a
ResNet-50 backbone) and learning only a tiny set of **visual prompts** for each new batch of
classes — roughly **1.3% of the parameters** are trainable. It adds *logit manipulation* to reduce
semantic drift between old and new classes.

This repository **reproduces the paper's reported results** on the ADE20K continual panoptic
benchmark across three difficulty settings, adds a measured **sensitivity analysis** of an
undocumented inference threshold the results depend on, and an **improvement attempt** that
grid-searches the paper's own inference-time logit-manipulation deltas.

The entire pipeline runs in a single Google Colab notebook:
[`notebook/PythonEclipseImprovement.ipynb`](notebook/PythonEclipseImprovement.ipynb).

## 2. Results

PQ = **Panoptic Quality** (higher is better), measured over *all* classes after the final task.
Every number below was **measured** by us in eval-only runs of the official checkpoints — logs are
preserved in [`notebook/threshold_proof_executed.ipynb`](notebook/threshold_proof_executed.ipynb)
and [`results/measured_results.json`](results/measured_results.json).

| Scenario | Tasks | Official paper (ECLIPSE R50) | Our reproduction (official settings) | thr 0.35 | thr 0.5 | Best searched deltas |
|:--------:|:-----:|:----------------------------:|:------------------------------------:|:--------:|:-------:|:--------------------:|
| 100-50   | 2     | 35.6 | 35.59 | 35.57 | 32.31 | 35.59 (official) |
| 100-10   | 6     | 33.9 | 33.84 | 33.82 | 32.31 | 33.84 (official) |
| 100-5    | 11    | 32.9 | 32.81 | 32.76 | 31.34 | 32.81 (official) |

**Reproduction:** within ±0.1 PQ of the published numbers in every setting, *including* the
base/novel class split (100-50: ours 41.73 / 23.31 vs. paper 41.7 / 23.5) — the paper is
reproducible.

**Sensitivity finding:** the released config uses an undocumented near-zero confidence filter
(`OBJECT_MASK_THRESHOLD: 0.0`; the paper never mentions it). Raising it to a conventional 0.5
costs up to 3.3 PQ, almost entirely on *novel* classes (100-50 PQ_novel: 23.2 → 17.9) —
empirically confirming the paper's own claim that prompt-tuned new classes are systematically
under-confident. An earlier draft of this project claimed a +0.3 "threshold improvement" based on
estimated numbers; measuring it disproved the claim, and this corrected analysis replaces it.

**Improvement attempt 1 — retraining (abandoned, fully documented):** we first tried the "heavy"
route — swapping ResNet-50 for a Swin-T backbone and retraining the base step (32k iterations,
6h01m of GPU time → base PQ only 14.66; the 100-50 prompt step on top reached 18.58, with novel
classes at 4.86), plus an R50 prompt-retraining attempt that stalled at PQ 17.48. Both were
abandoned on cost/benefit grounds; the recovered logs live in [`results/logs/`](results/logs/) and
the full analysis in [`docs/swin-t-retraining-attempt.md`](docs/swin-t-retraining-attempt.md).
The failure is itself evidence for the paper's premise: prompt tuning inherits the quality of the
frozen base.

**Improvement attempt 2 — logit-delta search (bonus, measured):** Experiment 3 grid-searched
`CONT.LOGIT_MANI_DELTAS` — the paper's own inference-time logit-manipulation magnitudes, which the
authors hand-picked per scenario (their 100-50 script trains with `[0.6,-0.6]` but evaluates with
`[-0.4,-0.6]`) — across 12 challenger configurations, eval-only, with the official values as
control. **Result: a clean negative** — the official deltas won in all three scenarios (100-50:
35.59 official vs. 35.52 best challenger). We report this as a finding: the authors' released
inference configuration is locally optimal, which independently corroborates the integrity of
their tuning.

![Reproduction results: official vs. reproduced vs. threshold sensitivity](results/reproduction_graph.png)

*The "100-50" notation means 100 base classes learned first, then 50 new classes added in
increments. 100-50 adds all 50 at once (2 tasks); 100-10 adds them 10 at a time (6 tasks); 100-5
adds them 5 at a time (11 tasks). More tasks = more chances to forget = harder.*

## 3. Repository structure

```
ECLIPSE-reproduction/
├── index.html                 # Project website (GitHub Pages landing page)
├── README.md                  # You are here — project overview
├── docs/
│   ├── documentation.md       # Full technical documentation
│   ├── algorithmic-thinking.md# Project stages + how each stage is tested
│   ├── ai-collaboration.md    # How AI was used + how to reproduce per stage
│   ├── ai-prompts.md          # Full per-stage prompt log with verbatim error excerpts
│   └── swin-t-retraining-attempt.md  # The abandoned retraining attempts (Swin-T + R50), with logs
├── plan/
│   └── ai-plan.md             # The AI-generated project plan (human-edited)
├── notebook/
│   ├── PythonEclipseImprovement.ipynb   # Main pipeline, FULLY EXECUTED: reproduction + sensitivity + delta search
│   └── threshold_proof_executed.ipynb   # Earlier executed threshold run (archive)
├── results/
│   ├── reproduction_graph.png # Final measured graph (from the executed notebook)
│   ├── measured_results.json  # Every measured number, with provenance notes
│   └── logs/                  # Recovered retraining logs (Swin-T base/100-50, R50 100-10)
├── takeaways.md               # Reflective writeup (Hebrew, 1–2 pages)
├── requirements.txt           # Environment notes
└── LICENSE
```

## 4. Submission checklist (per course spec)

Every item the course requires, and where to find it:

| # | Required item | Location |
|:-:|---------------|----------|
| 1 | Project documentation (Markdown) | [`README.md`](README.md) + [`docs/documentation.md`](docs/documentation.md) |
| 2 | Algorithmic thinking — project stages | [`docs/algorithmic-thinking.md`](docs/algorithmic-thinking.md) §2 |
| 3 | Algorithmic thinking — how each stage is tested | [`docs/algorithmic-thinking.md`](docs/algorithmic-thinking.md) §3 |
| 4 | AI plan file (parallel) | [`plan/ai-plan.md`](plan/ai-plan.md) |
| 5 | AI collaboration — reproducing results per stage | [`docs/ai-collaboration.md`](docs/ai-collaboration.md) + [`docs/ai-prompts.md`](docs/ai-prompts.md) |
| 6 | Reflective takeaways (1–2 pages) | [`takeaways.md`](takeaways.md) |
| 7 | GitHub project website | [`index.html`](index.html) → published via GitHub Pages |

## 5. How to reproduce (quick version)

The full, annotated pipeline is in the notebook; the short version is:

1. **Open the notebook in Google Colab** with a GPU runtime
   (`Runtime → Change runtime type → GPU`).
2. **Clone ECLIPSE** and install dependencies — Detectron2, panopticapi,
   cityscapesScripts, and the repo's `requirements.txt`.
3. **Compile the custom CUDA op** (MultiScaleDeformableAttention). The notebook patches a
   `.scalar_type().is_cuda()` call to `.is_cuda()` so it compiles on current PyTorch.
4. **Prepare ADE20K** — download the images and run the three `prepare_ade20k_*` scripts to build
   the semantic / panoptic / instance annotations Detectron2 expects.
5. **Download the released checkpoints** (`ade_ps_100_50_final.pth`, `…100_10…`, `…100_5…`) and
   evaluate each scenario in **`--eval-only`** mode at the exact official settings (verified
   against `script/ade_ps/*.sh`; all overrides passed on the command line, no source patching).
6. **(Sensitivity)** Re-evaluate with `OBJECT_MASK_THRESHOLD` at `0.35` and `0.5` (official: `0.0`).
7. **(Bonus)** Run the `CONT.LOGIT_MANI_DELTAS` grid search and read the auto-generated verdict.

See [`docs/documentation.md`](docs/documentation.md) for the detailed, step-by-step version and
[`docs/algorithmic-thinking.md`](docs/algorithmic-thinking.md) for the reasoning behind each step.

## 6. Acknowledgements & references

- **ECLIPSE** — Kim, Yu, Hwang. *ECLIPSE: Efficient Continual Learning in Panoptic Segmentation with
  Visual Prompt Tuning.* CVPR 2024.
  [Paper (arXiv:2403.20126)](https://arxiv.org/abs/2403.20126) ·
  [Official code](https://github.com/clovaai/ECLIPSE)
- **Mask2Former** — Cheng et al. *Masked-attention Mask Transformer for Universal Image
  Segmentation.* CVPR 2022.
- **ADE20K** — Zhou et al. *Scene Parsing through ADE20K Dataset.* CVPR 2017.
- **Detectron2** — Wu et al., Facebook AI Research.

This is an educational reproduction. All credit for the ECLIPSE method belongs to the original
authors; see [`LICENSE`](LICENSE) for details.
