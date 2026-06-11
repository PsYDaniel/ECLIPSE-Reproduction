# Working with AI — and how to reproduce these results

This document explains **how an AI chat assistant was used** throughout the project and, for each
stage, **how you could prompt an assistant to reach the same result**. The point is not "the AI did
it" — it is to make the collaboration *transparent and repeatable*. Every prompt below was followed
by reading the output critically, running it, and feeding the real error messages back in.

> **Honest framing.** The AI was most useful for *environment plumbing* (install errors, CUDA
> compilation, dataset paths) and for *explaining* unfamiliar concepts. It was least reliable when it
> guessed at exact file paths or API details it couldn't see — those always had to be verified
> against the actual repository and error output.

## How to get good, reproducible answers

A few habits made the assistant's output trustworthy:

1. **Give it the real error, verbatim.** Pasting the full traceback beats describing it.
2. **Tell it the environment.** "Google Colab, Python 3.12, CUDA GPU, PyTorch from the default
   image" changes the answer completely.
3. **Ask for a check, not just a fix.** "…and give me a command to confirm it worked" turns every
   step into something verifiable.
4. **Make it explain before it edits.** Asking "why does this fail?" before "fix it" caught wrong
   guesses early.
5. **Keep the loop tight.** One change → run → paste result back. Large multi-step answers were
   harder to debug than small ones.

## Stage-by-stage prompts

### Stage 0–1 — Setup & dependencies
> *"I'm on Google Colab with a GPU. I want to run the clovaai/ECLIPSE repo. Give me the cells to
> clone it and install Detectron2, panopticapi, cityscapesScripts and the repo requirements, in the
> right order, and a command to confirm Detectron2 imports."*

Follow-up when an install failed: paste the `pip` error and ask *"what does this dependency conflict
mean and what's the minimal fix?"*

### Stage 2 — Compiling the CUDA op (the hard one)
> *"Running `make.sh` for MultiScaleDeformableAttention fails with this compiler error: \<paste full
> error\>. I'm on PyTorch \<version\>. What changed in the API and what's the smallest patch?"*

This is where the assistant identified that `.scalar_type().is_cuda()` is no longer valid and
suggested the one-line `sed` patch to `.is_cuda()`. **Verification it suggested:** re-run `make.sh`
and then `import MultiScaleDeformableAttention` in Python.

### Stage 3 — Dataset preparation
> *"ADE20K prep is extremely slow on Google Drive. The scripts are `prepare_ade20k_sem_seg.py`,
> `prepare_ade20k_pan_seg.py`, `prepare_ade20k_ins_seg.py`. How do I download to Colab's local disk
> instead, run them there, and copy only the results back to Drive?"*

This produced the `wget`/`unzip`/`rsync --exclude` strategy. **Verification:** check that the
`myade20k_panoptic_{train,val}` folders appear and the progress bars hit 100%.

### Stage 4 — Checkpoints & the run script
> *"Download `ade_ps_100_50_final.pth` from the ECLIPSE GitHub releases into `checkpoints/`, then
> show me the contents of `script/ade_ps/100_50.sh` so I can see how evaluation is launched."*

Reading the script together, the assistant pointed out the `--eval-only` line and which `CONT.*`
arguments must stay untouched. **Verification:** confirm the `.pth` size is hundreds of MB.

### Stages 5–7 — Evaluating each scenario
> *"I want to evaluate the released weights, not train. In `script/ade_ps/100_50.sh`, programmatically
> point the data root at my local dataset, use my downloaded checkpoint, and comment out the
> `python train_inc.py …` line that is **not** `--eval-only`. Then run it."*

The assistant produced the small Python snippet that rewrites the script (`text.replace(...)` and
commenting out the training line). **Verification:** a final `PQ` value prints; compare to the
paper. Re-used almost verbatim for 100-10 and 100-5 by swapping the checkpoint and script name.

### Stage 8 — Sensitivity experiment (and the mistake that shaped the project)
> *"Hypothesis: novel-class masks are lower-confidence, so the `OBJECT_MASK_THRESHOLD` filter
> discards correct masks. Build an eval-only harness that runs every scenario at thresholds 0.0,
> 0.35 and 0.5, passes the value as a command-line config override, and builds the table/graph
> only from parsed Detectron2 logs."*

An early draft of this experiment plotted *estimated* numbers for the 0.35 threshold without
running it. An AI review of the notebook caught the hardcoded values, and we rebuilt the cell so it
can only report log-parsed results. The measurement then **disproved our own claim**: the official
threshold turned out to be 0.0 (not 0.5 as we had assumed), so 0.35 was a slight degradation, not
an improvement — while threshold 0.5 cost up to 3.3 PQ, almost all of it on novel classes.

### Stage 9 — Delta search improvement attempt
> *"`CONT.LOGIT_MANI_DELTAS` is the paper's inference-time logit-manipulation knob and the authors
> hand-picked the values per scenario. Generate an eval-only grid search over a small neighbourhood
> of the official deltas, cache logs to Drive so the run is resumable, and write a verdict cell
> that honestly reports improvement, mixed, or negative results."*

**Verification:** every candidate's PQ is parsed from its own log; the official deltas are included
in the ranking as the control.

## When the AI was wrong (and how we caught it)

- It occasionally invented file paths inside the repo. **Caught by** `ls`-ing the path before
  trusting it.
- It sometimes proposed reinstalling packages that were already fine, masking the real error.
  **Caught by** reading the *first* error in the traceback, not the last.
- Early plans under-estimated how long dataset prep and CUDA compilation would take. **Caught by**
  actually timing them and promoting both to their own checkpointed stages.
- One assistant claimed the paper's default mask threshold was 0.5 with invented line numbers as
  evidence; another assumed it too. **Caught by** fetching the official config from the ECLIPSE
  repository, which sets `OBJECT_MASK_THRESHOLD: 0.0`. This single check rewrote our entire bonus
  claim — and conversely, *our own notebook* was caught presenting estimated numbers as measured
  ones by an AI review. Verification ran in both directions.

The broader lesson — that AI output is a *draft to verify*, not a final answer — is the main subject
of the [reflective takeaways](../takeaways.md).

> **Full prompt log.** This file describes the *methodology*; the complete stage-by-stage record —
> every prompt, every verbatim error excerpt, every verification step, including the retraining
> attempts and the negative delta-search result — is in [`ai-prompts.md`](ai-prompts.md).
