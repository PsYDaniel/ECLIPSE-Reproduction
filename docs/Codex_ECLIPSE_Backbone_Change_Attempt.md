# Codex ECLIPSE Backbone change attempt


## Conversation

### 1. Improve the Original ECLIPSE Reproduction

User: "Please improve the ECLIPSE reproduction notebook by adding inference-only test-time augmentation. Use multi-scale inference at 0.75, 1.0, and 1.25, plus horizontal flip, and average predictions at the mask or logit level. This should require no retraining. Also consider smarter prompt initialization for new tasks, either by initializing new prompt embeddings from semantically nearest base-class prompts or by using CLIP/SigLIP text embeddings projected into prompt space. Keep the notebook cells ordered so someone can run them one by one in VS Code with a Colab runtime."

Codex: Reviewed the original ECLIPSE reproduction notebook and updated it with inference-time TTA logic and smarter prompt-initialization support while preserving the run order.

Files involved:

- `ECLIPSE_Reproduction_Worked.ipynb`

Outcome:

- Added no-retraining TTA support.
- Added prompt initialization improvements.
- Kept the notebook structured for sequential execution.

### 2. Create a Stronger Backbone Notebook Without TTA

User: "Please create a new notebook based on the same code, but without TTA. Instead, test a stronger frozen backbone. Swap ResNet-50 for a stronger model such as Swin-T, ConvNeXt-T, or DINOv2. Since ECLIPSE freezes the base model, the idea is to retrain the 100-class base step once with the new backbone, then train continual prompts on top. Also add a graph comparing the original article results with our final results. The notebook should run in VS Code using a Google Colab runtime."

Codex: Created a separate Swin-T no-TTA Colab notebook for the stronger frozen-backbone experiment.

Files involved:

- `ECLIPSE_Stronger_Backbone_SwinT_No_TTA_Colab.ipynb`

Outcome:

- Added Swin-T backbone setup.
- Generated Swin-T Detectron2 config and training scripts.
- Kept evaluation single-scale and no-TTA.
- Added graphing code comparing article PQ values against the new run.

### 3. Prefer ConvNeXt-V2-Tiny Instead of Swin-T

User: "Please create another new file. Instead of Swin-T, use ConvNeXt-V2-Tiny because Swin-T may take too long to retrain."

Codex: Created a separate ConvNeXt-V2-Tiny no-TTA Colab notebook.

Files involved:

- `ECLIPSE_Stronger_Backbone_ConvNeXtV2_Tiny_No_TTA_Colab.ipynb`

Outcome:

- Added a Detectron2-compatible ConvNeXt-V2-Tiny backbone wrapper using `timm`.
- Added ConvNeXt-V2-Tiny config generation.
- Added train/eval scripts for base, 100-50, 100-10, and 100-5.
- Added final PQ extraction and article-comparison graphing.

### 4. Make the Swin-T Schedule More Practical

User: "In the Swin-T notebook, please adjust the iterations so the run is more likely to improve PQ but still finishes in a reasonable amount of time. I tried running it on a Google Colab A100 40 GB runtime and the ETA was about two days, which is not reasonable."

Codex: Updated the Swin-T Colab notebook schedule to reduce runtime while keeping the training meaningful.

Files involved:

- `ECLIPSE_Stronger_Backbone_SwinT_No_TTA_Colab.ipynb`

Outcome:

- Reduced the base step from the original long schedule to a stronger but more practical Colab schedule.
- Set base training to `75,000` iterations.
- Set prompt stages to practical iteration counts.
- Reduced repeated validation overhead by evaluating mainly at the end of each stage.

### 5. Add Swin-T Checkpoint Persistence

User: "Please make the Swin-T notebook more robust. If ResNet-50 uses 100k iterations, try 75k for Swin-T. Also add a cell that saves training checkpoints so I do not need to retrain if I rerun the notebook."

Codex: Updated the Swin-T notebook with Drive checkpoint restore/save support.

Files involved:

- `ECLIPSE_Stronger_Backbone_SwinT_No_TTA_Colab.ipynb`

Outcome:

- Set Swin-T base training to `75,000` iterations.
- Added `Cell 8.5` to restore/save Swin-T checkpoints through Google Drive.
- Added checkpoint manifest entries for base, 100-50, 100-10, and 100-5.
- Patched training cells to save checkpoints after each stage.
- Verified the notebook had no TTA markers and no stale 32k assignment.

### 6. Make Swin-T Work Locally

User: "Please make the Swin-T model work locally on my laptop. Organize the cells in order and add cells that download and remember checkpoints, so I do not need to retrain if I have already trained. Make this a separate file."

Codex: Created a separate local-first Swin-T notebook designed for VS Code with a Linux or WSL2 Jupyter kernel.

Files involved:

- `ECLIPSE_Stronger_Backbone_SwinT_No_TTA_Local.ipynb`

Outcome:

- Created a local workspace flow under `~/eclipse_swin_local`.
- Added local download caches for ADE20K and Swin-T checkpoints.
- Added local checkpoint restore/save support.
- Blocked native Windows Python because Detectron2 and ECLIPSE CUDA extensions are more reliable through Linux/WSL2.
- Verified the notebook compiles and has no Colab or TTA leftovers.

### 7. Add ConvNeXt Checkpoint Restore and Save Cells

User: "In the ConvNeXt model, please add checkpoint creation and restoration for ConvNeXt training. Make the cells explain in comments what they do and how they work."

Codex: Added a checkpoint restore/save cell to the ConvNeXt-V2-Tiny notebook and wired it into training.

Files involved:

- `ECLIPSE_Stronger_Backbone_ConvNeXtV2_Tiny_No_TTA_Colab.ipynb`

Outcome:

- Added `Cell 8.5` for ConvNeXt-V2-Tiny checkpoint restore/save through Google Drive.
- Added clear comments explaining the purpose and mechanics of the checkpoint system.
- Added checkpoint manifest paths for base, 100-50, 100-10, and 100-5.
- Patched training cells to save after each completed stage.

### 8. Save Base Checkpoints Every 10k Iterations

User: "Please ensure the ConvNeXt base training creates checkpoints every 10k iterations, and make it save those checkpoints to Google Drive while training is still running."

Codex: Updated the ConvNeXt base training flow to checkpoint every 10k iterations and live-sync checkpoints to Drive.

Files involved:

- `ECLIPSE_Stronger_Backbone_ConvNeXtV2_Tiny_No_TTA_Colab.ipynb`

Outcome:

- Set base training `SOLVER.CHECKPOINT_PERIOD` to `10000`.
- Added `--resume` to base training.
- Added live background sync for base checkpoints.
- Synced `model_*.pth`, `model_final.pth`, and `last_checkpoint` to Drive while base training runs.
- Added restore logic for intermediate base checkpoints and `last_checkpoint`.

### 9. Make Detectron2 More Reliable

User: "Please make sure Detectron2 does not cause issues."

Codex: Hardened the Detectron2 setup cells in the ConvNeXt notebook.

Files involved:

- `ECLIPSE_Stronger_Backbone_ConvNeXtV2_Tiny_No_TTA_Colab.ipynb`

Outcome:

- Reworked the Detectron2 source build cell.
- Added CUDA, Torch, `CUDA_HOME`, `nvcc`, and GPU capability checks.
- Added clean rebuild of Detectron2 with stale `.so` and `.pyd` removal.
- Used `FORCE_CUDA=1`.
- Used `TORCH_CUDA_ARCH_LIST`.
- Installed Detectron2 with `--no-build-isolation --no-deps`.
- Added strict import verification for `detectron2._C`.
- Forced the notebook and shell cells to use `/content/detectron2_src`.

### 10. Clean the ConvNeXt Notebook Run Order

User: "Please remove unnecessary cells and make the notebook clear to run from Cell 0 to the final cell. Also make sure base training and 100-50 training are continuously saved to Drive while training. Add a clearly labeled restoration cell to run before base training or when continuing training."

Codex: Reorganized the ConvNeXt notebook into a clear sequential run order.

Files involved:

- `ECLIPSE_Stronger_Backbone_ConvNeXtV2_Tiny_No_TTA_Colab.ipynb`

Outcome:

- Reduced the notebook to a clear 19-cell run order.
- Removed unnecessary detour cells.
- Added a clearly labeled restore/continue checkpoint cell.
- Added live Drive syncing for both base training and 100-50 training.
- Set both base and 100-50 checkpointing to 10k intervals.
- Confirmed `--resume` is used in generated training scripts.
- Verified no TTA markers were introduced.

### 11. Resolve Notebook Cell Count Confusion

User: "Why does my file contain 36 cells? It still has all the diagnostics. Did you not change that file?"

Codex: Checked the exact file path and confirmed the file on disk had 19 cells, while VS Code was likely showing a stale open buffer.

Files involved:

- `ECLIPSE_Stronger_Backbone_ConvNeXtV2_Tiny_No_TTA_Colab.ipynb`
- `ECLIPSE_Stronger_Backbone_ConvNeXtV2_Tiny_No_TTA_Colab_CLEAN_RUN_ORDER.ipynb`

Outcome:

- Verified the original file on disk had 19 cells and no old diagnostic cells.
- Created a fresh clean-copy notebook to avoid VS Code stale-buffer confusion.
- Confirmed both notebooks had restore logic and live sync for base and 100-50.

### 12. Add ADE20K Restore Progress and ETA

User: "The ADE20K restore step takes far too long. Please add progress or ETA. I have been running it for 85 minutes and it is still not finished."

Codex: Replaced the silent ADE20K Drive restore cell with a progress-aware restore and cache system.

Files involved:

- `ECLIPSE_Stronger_Backbone_ConvNeXtV2_Tiny_No_TTA_Colab.ipynb`
- `ECLIPSE_Stronger_Backbone_ConvNeXtV2_Tiny_No_TTA_Colab_CLEAN_RUN_ORDER.ipynb`

Outcome:

- Replaced silent `shutil.copytree()` restore with progress-aware commands.
- Added `pv` progress, speed, and ETA for tar archive restore.
- Added `rsync --info=progress2` for old folder-cache restore.
- Added a single-file Drive tar cache:
  `/content/drive/MyDrive/ECLIPSE_CACHE/ADEChallengeData2016_prepared.tar`
- Added detection and cleanup for incomplete local `/content` dataset restores.
- Removed a reappeared duplicate CUDA-op visibility cell.
- Cleared saved notebook outputs so the main file was lightweight again.

## Final Notebook State

Primary ConvNeXt notebook:

- `ECLIPSE_Stronger_Backbone_ConvNeXtV2_Tiny_No_TTA_Colab.ipynb`

Clean copy:

- `ECLIPSE_Stronger_Backbone_ConvNeXtV2_Tiny_No_TTA_Colab_CLEAN_RUN_ORDER.ipynb`

Important current features:

- Clear Cell 0 to final-cell run order.
- Hardened Detectron2 build.
- ConvNeXt-V2-Tiny backbone patch.
- No TTA.
- Base training checkpoint every 10k iterations.
- 100-50 checkpoint every 10k iterations.
- Live Drive sync while base and 100-50 training are running.
- Restore/continue cell before training.
- ADE20K restore with progress and ETA.
- Article-vs-run PQ graphing.

**NOTE**
- The ultimate result was that it was too time consuming and expensive to continue.
- We decided to stick with the original backbone.
