# legacy/ — early code (no longer maintained)

Kept for reference; **do not use it for any new analysis or evaluation**.
For the current pipeline see [`../pipeline/`](../pipeline/) and [`../experiments/`](../experiments/).

| Directory | Period | Description |
|---|---|---|
| `pilot/` | Earliest stage of the project | Pilot with 3–4 subjects. Attempts with various architectures such as LSTM / Attention |
| `v1_0624/` | 2026-06 | First complete pipeline, superseded by the `*_FIXED.py` files in `pipeline/` |

## ⚠️ Why it can't be run directly

1. **Paths are placeholders.** The hard-coded absolute paths of the development machine were replaced with `<DATASET_ROOT>`, `<DOWNLOADS>` and
   `<CLEAN_DIR>` (they leaked the user account and directory structure). You have to fill them in to run anything.
2. **The data format has changed.** These programs correspond to an old dataset layout (`0513Combined/`, `0519Combined/`, etc.);
   the column definitions and file-name format of the current `Dataset/Combined/` are different.
3. **Known methodological problems.** The early versions contain several errors discovered later:
   * splitting by file instead of by subject → inflated evaluation
   * MVC from a single-point peak instead of a moving-window stable peak → systematically lowered %MVC
   * knee angle from a two-axis projection only → flexion systematically underestimated by 20–30°
   * the trunk forward-lean angle actually measured side-to-side sway → that compensation label was effectively dead

   The fix for each item is described in the docstring of the corresponding `*_FIXED.py` in `pipeline/`.

## Contents of `v1_0624/`

The complete old pipeline: `emg_process_segment.py` → `Features_insert.py` →
`Global_Training.py` → `phase2_finetune.py`, plus `MAX_MVC.py`, `MAX_segment.py`,
`CheckFrame.py`, `FindFrame.py` (LED sync-point detection) and `TrainModel.py`.

It also includes two result figures from that time: `my_model_results.png` and `prediction_vs_actual.png`.

## Contents of `pilot/`

| Subdirectory | Content |
|---|---|
| `Preprocess/` | EMG preprocessing, FFT, MDF median frequency, video capture, LED sync-point detection, skeleton merging |
| `TrainingModel/` | Various architecture attempts: `MIA.py` (Conv1D + MultiHeadAttention), `MSE.py`, `TrainModel.py` (BiLSTM) |
| `TestModel/` | `predict_batch_csv.py` (batch prediction on CSVs), `predict_realtime_mediapipe.py` (real-time MediaPipe inference), plotting tools |

> Early subject codes were `S01` / `PILOT_A` / `PILOT_B` / `TEST1`,
> which **do not correspond to the current S01–S10 numbering** (different people, recruited at a different time).
