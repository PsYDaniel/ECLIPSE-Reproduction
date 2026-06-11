# AI Prompt Log — Full Working Record

**Project:** Reproducing ECLIPSE (CVPR 2024, arXiv:2403.20126) · **Authors:** Daniel Chicherin, Or Blazer
**Course:** Python (B.Sc., 3rd year) · **Companion documents:** [`ai-collaboration.md`](ai-collaboration.md) (methodology), [`swin-t-retraining-attempt.md`](swin-t-retraining-attempt.md) (retraining attempts)

This document is the complete, stage-by-stage record of the prompts we issued to AI assistants
during the project, the errors we encountered (quoted from the actual run logs), how each AI answer
was verified before being trusted, and what each exchange produced. Prompts are reconstructed
faithfully from our working sessions; error excerpts are copied verbatim from the preserved logs in
[`../results/logs/`](../results/logs/) and the executed notebook outputs.

**Working protocol.** Every exchange below followed the same loop: (1) state the environment
precisely, (2) paste the *real* error, (3) ask for an explanation before a fix, (4) demand a
verification command, (5) run it and feed the result back. Where the AI was wrong — and it was
wrong at several decisive moments, documented below — the verification step is what caught it.

---

## Stage 0 — Environment and repository setup

**Context.** Google Colab GPU runtime, Python 3.12, project storage on Google Drive.

> **User:** "I'm on Google Colab with a GPU runtime, Python 3.12. I want to reproduce the CVPR 2024
> ECLIPSE paper (clovaai/ECLIPSE on GitHub). Give me notebook cells that mount Google Drive, clone
> the repository into Drive, and confirm the clone — and explain why Drive vs. local disk matters
> in Colab before I commit to a layout."

**Assistant output:** the mount + clone cells, plus the key trade-off: Drive persists between
sessions but has very slow I/O; local disk (`/content`) is fast but ephemeral.

**Error encountered** (on re-running the cell in a second session):

```
fatal: destination path 'ECLIPSE' already exists and is not an empty directory.
```

> **User:** "Re-running the clone cell gives `fatal: destination path 'ECLIPSE' already exists`.
> I don't want to delete and re-download every session. What is the idempotent pattern for
> clone-or-reuse in a notebook?"

**Resolution:** guard the clone with an existence check on a sentinel file (`train_inc.py`) and
add a `FORCE_FRESH_CLONE` flag for when a pristine checkout is required. **Verification:** the
cell prints either the clone progress or "Using existing ECLIPSE checkout" and `git rev-parse
--short HEAD` prints a commit hash. **Lesson:** every setup cell must be safe to re-run, because
Colab sessions die.

---

## Stage 1 — Dependency installation

> **User:** "Give me the install cells for the ECLIPSE repo on Colab: Detectron2, panopticapi,
> cityscapesScripts, opencv, plus the repo's own requirements.txt — in the right order, and tell me
> which of these must be compiled against the runtime's PyTorch/CUDA rather than installed as a
> wheel, and why."

**Assistant output:** install Detectron2 from source (`pip install
'git+https://github.com/facebookresearch/detectron2.git'`) because prebuilt wheels target specific
torch/CUDA pairs that Colab's rolling updates break; the rest from git/PyPI.

**Warning encountered** (representative, from the executed notebook logs):

```
DEPRECATION: Loading egg at /root/.local/lib/python3.12/site-packages/
MultiScaleDeformableAttention-1.0-py3.12-linux-x86_64.egg is deprecated.
pip 24.3 will enforce this behaviour change.
```

> **User:** "pip prints a deprecation warning about loading an `.egg` for
> MultiScaleDeformableAttention. Is this blocking, and should I change how the CUDA op is
> installed?"

**Resolution:** non-blocking for our pipeline; we later switched the op build to
`setup.py build_ext --inplace` (Stage 2), which sidesteps the egg path entirely.
**Verification:** `python -c "import detectron2; print('detectron2 import ok')"` exits 0.
**Lesson:** distinguish warnings from errors before spending time on them — but record them, since
deprecations predict *future* breakage.

---

## Stage 2 — Compiling the custom CUDA operator (the project's critical fix)

**Context.** Mask2Former requires the `MultiScaleDeformableAttention` CUDA extension, written in
2021-era PyTorch C++ API. Colab's runtime had moved to PyTorch with CUDA 12.8.

**Build environment warnings** (verbatim from our build log):

```
W0525 12:38:01.969000 3676 torch/utils/cpp_extension.py:659] Attempted to use ninja as the
BuildExtension backend but we could not find ninja.. Falling back to using the slow distutils backend.
W0525 12:38:01.975000 3676 torch/utils/cpp_extension.py:535] There are no x86_64-linux-gnu-g++
version bounds defined for CUDA version 12.8
```

**Compile error** (representative excerpt — the legacy API call at the heart of the failure):

```
src/cuda/ms_deform_attn_cuda.cu: in function 'ms_deform_attn_cuda_forward':
error: 'c10::ScalarType' has no member named 'is_cuda'
       AT_ASSERTM(value.scalar_type().is_cuda(), "value must be a CUDA tensor");
```

> **User:** "Running the build for MultiScaleDeformableAttention fails. I'm on Colab, PyTorch with
> CUDA 12.8, Python 3.12. Here is the full compiler output: \<paste\>. Before proposing a fix:
> what changed in the PyTorch C++ API that makes `.scalar_type().is_cuda()` invalid, and what is
> the smallest patch that preserves the operator's semantics?"

**Assistant output:** in modern PyTorch the CUDA-ness check lives on the *tensor*
(`tensor.is_cuda()`), not on its scalar type; the minimal patch is `.scalar_type().is_cuda()` →
`.is_cuda()`. Follow-up rounds added three more compatibility patches:
`.type().is_cuda()` → `.is_cuda()`, `AT_DISPATCH_FLOATING_TYPES(value.type(), …)` →
`AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), …)`, and `.data<…>` → `.data_ptr<…>`.

> **User:** "The patch compiles. Give me a command that proves the extension actually loads and
> exposes the forward/backward symbols — I don't want to discover a broken op an hour into
> evaluation."

**Verification (as suggested, and adopted into the notebook permanently):**

```python
import MultiScaleDeformableAttention as MSDA
assert hasattr(MSDA, 'ms_deform_attn_forward')
assert hasattr(MSDA, 'ms_deform_attn_backward')
```

**Lesson:** research code rots at the speed of its fastest-moving dependency. The fix was one line;
*finding* it required reading the first compiler error, not the last.

---

## Stage 3 — ADE20K dataset preparation

> **User:** "ADE20K prep through the three scripts (`prepare_ade20k_sem_seg.py`, `_pan_seg.py`,
> `_ins_seg.py`) is unbearably slow when the dataset sits on Google Drive. Design a layout where
> the images and prep run on Colab's local disk, only durable artifacts go to Drive, and the cell
> skips all work if the prepared folders already exist."

**Assistant output:** `wget` + `unzip` directly to `/content`, run the three prep scripts there,
existence-check on `myade20k_panoptic_{train,val}` as the skip condition.

**Verification:** progress bars reach `100% 20210/20210` (train) and `100% 2000/2000` (val); the
four required folders exist (asserted in the cell). **Lesson:** I/O placement, not compute, was the
bottleneck — measured at minutes (local) vs. hours (Drive).

---

## Stage 4 — Checkpoints and the official run scripts

> **User:** "Download the released checkpoints (`ade_ps_100_50_final.pth` etc.) from the ECLIPSE
> GitHub releases. Add a guard against the classic failure where wget 'succeeds' but saves an HTML
> error page. Then print `script/ade_ps/100_50.sh` — I want to read the exact evaluation arguments
> before running anything."

**Assistant output:** a `valid_checkpoint()` size check (≥50 MB) plus a Drive cache so checkpoints
survive sessions; reading the script surfaced the eval section's arguments — including
`CONT.LOGIT_MANI_DELTAS`, which later became Experiment 3.

**Verification:** checkpoint sizes print in the hundreds of MB; the script's `--eval-only` line is
visible. **Lesson:** reading the authors' script line-by-line is primary-source research; two of
our three experiments came from details noticed here.

---

## Stage 5 — Evaluation harness (and the first integrity failure)

**Context.** Our first bonus implementation `sed`-patched threshold values across the repo and —
critically — plotted *estimated* "improved" numbers that had never been measured.

> **User:** "Review this notebook cell critically. It changes `OBJECT_MASK_THRESHOLD` with sed and
> then plots optimized_scores = [33.1, 34.2, 35.95]. Are these values actually produced by any
> evaluation in this notebook?"

**AI review finding (the project's turning point):** the values were hardcoded; the comment in our
own cell read `# הערכים המשוערים` ("the estimated values"). No re-evaluation followed the sed. The
claimed +0.3 PQ improvement was unsupported.

> **User:** "Rebuild this as an eval-only proof harness: same checkpoints, same dataset, the
> threshold passed as a config value; parse PQ only from the Detectron2 `copypaste` log lines;
> build the table and the graph exclusively from parsed logs; and add a verdict cell that prints an
> honest negative result if the change does not improve every scenario."

**Outcome:** the harness measured what the estimates had asserted — and **disproved them** (see
Stage 6). A second design iteration replaced file patching entirely with command-line config
overrides (`MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD …`), plus a clean-checkout assertion, so
every run is the official commit + an explicit, logged set of overrides. **Lesson:** the most
valuable prompt in the project was the one that asked the AI to attack our own work.

---

## Stage 6 — The threshold investigation (where two AIs disagreed and the source decided)

**Context.** Our premise was that the paper's default threshold is 0.5. One AI assistant
*confirmed* this premise and supplied line numbers ("maskformer_model.py, around lines 2504–2506")
as evidence. The file is ~700 lines long; the line numbers were fabricated.

> **User:** "Two claims are in conflict: (a) the default `OBJECT_MASK_THRESHOLD` for ECLIPSE's
> ADE20K panoptic eval is 0.5; (b) it is something else. Do not reason from memory. Fetch
> `configs/ade20k/panoptic-segmentation/maskformer2_R50_bs16_160k.yaml` from the official
> clovaai/ECLIPSE repository and quote the TEST block verbatim."

**Primary-source answer (verbatim from the official config):**

```yaml
TEST:
  SEMANTIC_ON: False
  INSTANCE_ON: False
  PANOPTIC_ON: True
  MASK_BG: False
  OVERLAP_THRESHOLD: 0.0
  OBJECT_MASK_THRESHOLD: 0.0
```

The official default is **0.0** — a setting the paper itself never mentions. Our measured
sensitivity results then fell into place (all PQ, eval-only, official checkpoints):

| Scenario | thr 0.0 (official) | thr 0.35 | thr 0.5 |
|:--:|:--:|:--:|:--:|
| 100-50 | 35.59 | 35.57 | 32.31 |
| 100-10 | 33.84 | 33.82 | 32.31 |
| 100-5  | 32.81 | 32.76 | 31.34 |

At 0.5 the damage concentrates in novel classes (100-50 PQ_novel 23.3 → 17.9), empirically
confirming the paper's claim that prompt-tuned new classes are under-confident.
**Lesson (two-sided):** an AI confirmed our wrong assumption with invented evidence, and a
different AI review caught our own unmeasured numbers. Neither humans nor AIs were the authority —
the primary source and the measurement were.

---

## Stage 7 — Retraining attempts (Swin-T backbone swap; R50 prompt retraining)

> **User:** "ECLIPSE freezes the base model, so a stronger frozen backbone should lift all
> scenarios. Create a notebook variant that swaps ResNet-50 for Swin-T using
> `maskformer2_swin_t_bs16_160k.yaml`, retrains the 100-class base step once (shortened to 32k
> iterations to fit Colab), then trains the 100-50 prompt step on top. Single-scale eval, no TTA."

**Real launch error** (verbatim from [`../results/logs/output_100_10.txt`](../results/logs/output_100_10.txt)):

```
AssertionError: Override list has odd length: ['OUTPUT_DIR', 'results/ade_ps', ...,
'WANDB', 'False', '--eval-only']; it must be a list of pairs
```

> **User:** "train_inc.py crashes with `Override list has odd length … it must be a list of
> pairs`, and the last element of the list is `--eval-only`. Why is an argparse flag ending up
> inside the yacs override list?"

**Resolution:** Detectron2's launcher treats everything after the named flags as `opts` key/value
pairs; `--eval-only` must precede the overrides. **Measured outcomes** (full analysis in
[`swin-t-retraining-attempt.md`](swin-t-retraining-attempt.md)):

- Swin-T base, 32k iterations, **6h01m** wall-clock → PQ **14.66** (too weak a base).
- Swin-T 100-50 prompt step → PQ **18.58**, novel classes only **4.86**.
- R50 100-10 prompt retraining from a self-trained base → PQ **17.48** vs. 33.84 released.

> **User:** "Given base PQ 14.66 after six hours and 100-50 PQ 18.58 vs. the 35.6 reference, give
> me an honest cost/benefit analysis: continue scaling the Swin-T schedule, or pivot to
> inference-time experiments? Justify in terms of expected PQ per GPU-hour."

**Decision:** pivot. A paper-faithful base schedule (160k iterations) extrapolates to ~30 GPU-hours
per backbone before any continual step; eval-only experiments cost ~5 minutes per scenario and
produce defensible, measured findings. **Lesson:** the failure itself validated ECLIPSE's premise —
prompt tuning inherits the frozen base's quality (novel PQ 4.86 on a weak base).

---

## Stage 8 — Improvement attempt: logit-manipulation delta search

> **User:** "The official eval scripts hand-tune `CONT.LOGIT_MANI_DELTAS` per scenario — 100-50
> even trains with `[0.6,-0.6]` but evaluates with `[-0.4,-0.6]`, so it's a post-hoc knob.
> Generate an eval-only grid search over a small neighbourhood of the official deltas (4 candidates
> per scenario), cache logs to Drive so a disconnect can resume, rank candidates by log-parsed PQ
> with the official values as control, and write a verdict cell that distinguishes improvement /
> mixed / negative outcomes."

**Measured outcome (negative result, reported as such):** the official deltas won in **all three
scenarios** — e.g. 100-50: official `[-0.4,-0.6]` → 35.59 vs. best challenger `[-0.3,-0.5]` →
35.52; every one of the 12 challenger configurations scored below its official control.
**Lesson:** a negative result with full logs is a *finding*: the authors' hand-tuned inference
configuration is locally optimal, which independently supports the integrity of their reported
tuning.

---

## Stage 9 — Final report generation

> **User:** "Regenerate the README results section, the comparison graph, and the documentation
> from the measured logs only. Remove every number that was ever estimated. Where an earlier claim
> was disproven, do not silently delete it — state what was claimed, what was measured, and why
> the correction matters."

**Outcome:** the current repository state. The graph
([`../results/reproduction_graph.png`](../results/reproduction_graph.png)) contains only measured
bars; [`../results/measured_results.json`](../results/measured_results.json) carries every number
with provenance.

---

## Summary table — errors encountered and their resolutions

| # | Stage | Error / failure (verbatim source) | Root cause | Resolution | Verified by |
|:-:|:-----:|------------------------------------|------------|------------|-------------|
| 1 | 0 | `fatal: destination path 'ECLIPSE' already exists` | non-idempotent clone cell | clone-or-reuse guard + `FORCE_FRESH_CLONE` | commit hash prints |
| 2 | 1 | pip `.egg` deprecation warning | legacy egg install of CUDA op | in-place `build_ext` | `import detectron2` ok |
| 3 | 2 | `'c10::ScalarType' has no member named 'is_cuda'` (+ ninja/g++ warnings) | PyTorch C++ API drift vs. 2021 code | 4 source patches before build | symbol asserts on import |
| 4 | 3 | hours-long dataset prep, stalled progress | Drive I/O bottleneck | local-disk layout + skip guard | progress bars reach 100% |
| 5 | 4 | risk: HTML page saved as `.pth` | wget saves error pages silently | ≥50 MB size assertion | checkpoint sizes print |
| 6 | 5 | **estimated numbers presented as results** | plotting unmeasured values | log-parsed-only harness + verdict cell | every figure traceable to a log |
| 7 | 6 | wrong default-threshold premise; fabricated line numbers from an AI | reasoning from memory | fetch official config; quote verbatim | `OBJECT_MASK_THRESHOLD: 0.0` |
| 8 | 7 | `Override list has odd length … '--eval-only'` | flag placed after yacs `opts` | flags before overrides | training launches |
| 9 | 7 | Swin-T PQ 18.58 vs. 35.6 reference | 32k base schedule too short; compute budget | documented pivot to eval-only | cost table, preserved logs |
| 10 | 8 | (no error — negative result) | official deltas already optimal | reported honestly as a finding | 15 logged runs |

---

*Every error excerpt above is reproduced from the project's preserved logs or executed notebook
outputs. The companion file [`ai-collaboration.md`](ai-collaboration.md) describes the prompting
methodology; this file is the record of its application.*
