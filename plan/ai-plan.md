# AI Project Plan

> This is the structured plan the AI assistant produced at the start of the project, lightly edited
> for clarity. It is the "plan file" deliverable. The implementation in the
> [notebook](../notebook/PythonEclipseImprovement.ipynb) follows it; where reality diverged from the
> plan, the divergence is noted in [`../docs/ai-collaboration.md`](../docs/ai-collaboration.md).

## Objective

Reproduce the published ADE20K continual-panoptic-segmentation results of **ECLIPSE** (CVPR 2024)
for the 100-50, 100-10 and 100-5 scenarios, on a single Google Colab GPU runtime, and present the
work as a documented GitHub project. Stretch goal: improve PQ with a small, well-motivated change.

## Success criteria

- Each scenario evaluates end-to-end and prints a **PQ** value within **±0.2** of the paper.
- The whole pipeline is **restartable** — any stage can be re-run independently.
- The bonus change is reproducible and its effect measured, not just asserted.

## Constraints & assumptions

- Single GPU, time-limited Colab sessions → **evaluate released checkpoints**, do not retrain.
- Large dataset (~22k images) → expect I/O to dominate; prefer local disk over Google Drive.
- Dependencies compile against the runtime → pin nothing by hand; install Detectron2 from source.

## Phases, tasks and checkpoints

### Phase 1 — Make it run
1. Clone ECLIPSE; mount Drive. **Checkpoint:** repo folder exists.
2. Install Detectron2 (source), panopticapi, cityscapesScripts, opencv, `requirements.txt`.
   **Checkpoint:** `import detectron2` works.
3. Compile `MultiScaleDeformableAttention`. **Checkpoint:** import works.
   ⚠️ **Risk (high):** the CUDA op may not compile on current PyTorch.
   **Mitigation:** patch `.scalar_type().is_cuda()` → `.is_cuda()` before `make.sh`.

### Phase 2 — Feed it data
4. Download ADE20K images + instance annotations. **Checkpoint:** archives extracted.
5. Run the three `prepare_ade20k_*` scripts. **Checkpoint:** `myade20k_panoptic_*` folders created,
   progress bars reach 100%.
   ⚠️ **Risk (medium):** Drive I/O is slow. **Mitigation:** do this on local disk, copy results back.
6. Download scenario checkpoints; read the run script. **Checkpoint:** `.pth` is hundreds of MB and
   the script shows an `--eval-only` line.

### Phase 3 — Verify the claim
7. Evaluate **100-50** (easiest, 2 tasks). **Checkpoint:** PQ ≈ 35.6.
8. Evaluate **100-10** (6 tasks). **Checkpoint:** PQ ≈ 33.9.
9. Evaluate **100-5** (hardest, 11 tasks). **Checkpoint:** PQ ≈ 32.9.
   For each: edit the script to use the local dataset + checkpoint and **comment out training**.

### Phase 4 — Extend it
10. Lower `OBJECT_MASK_THRESHOLD` 0.5 → 0.35; re-evaluate all three. **Checkpoint:** PQ does not
    drop; ideally small gain.
    > **Post-hoc note (kept for honesty):** this step's premise was wrong — the official released
    > config sets the threshold to **0.0**, not 0.5, so "lowering to 0.35" was actually a slight
    > degradation. Measurement replaced this step with a 0.0/0.35/0.5 sensitivity study (Stage 8)
    > and a `CONT.LOGIT_MANI_DELTAS` grid search (Stage 9). The plan is preserved as written to
    > show where plan and reality diverged.
11. Plot official vs. reproduced vs. optimised. **Checkpoint:** `reproduction_graph.png` saved.

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|------|:----------:|:------:|------------|
| CUDA op won't compile | High | Blocks everything | `scalar_type` patch before `make.sh` |
| Drive I/O too slow | High | Hours wasted | Run on local disk; `rsync --exclude` the images |
| Corrupt/HTML "checkpoint" | Medium | Silent eval failure | Verify `.pth` file size |
| OOM during eval | Medium | Crash | Use clean clone; free memory between scenarios |
| Threshold change hurts PQ | Low | Bonus fails | Keep default as the reported result; report honestly |

## Deliverables

Documentation, algorithmic-thinking writeup, this plan, the AI-collaboration guide, the reflective
takeaways, a short video, and the GitHub website — all listed in the
[README submission checklist](../README.md#4-submission-checklist-per-course-spec).
