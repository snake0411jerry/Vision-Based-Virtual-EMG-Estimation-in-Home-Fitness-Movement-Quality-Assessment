# pipeline/ — core pipeline

From raw EMG and OpenCap TRCs all the way to trained models.
Requires `Dataset/` (not in this repo; see the [root README](../README.md#3-dataset-path-only-needed-to-rerun-the-pipeline)).

File names with `_FIXED` are corrected versions relative to the old code in [`../legacy/v1_0624/`](../legacy/v1_0624/);
each program's docstring lists what was changed and why. **The old versions are kept only for reference; do not use them.**

---

## 🔄 Execution order

```
1. emg_process_segment_FIXED.py   Raw EMG → filtering/rectification/envelope, outputs dual tracks at 60FPS and 1000Hz
2. MAX_MVC_FIXED.py               compute the MVC reference (used by the next step)
3. create subjects.csv            ★ contains personal data, must be created yourself, see below
4. Features_insert_FIXED.py       merge skeleton TRC with EMG + feature engineering → Dataset/Combined/
5. Global_Training_FIXED.py       global model training → models/global/
6. phase2_finetune_FIXED.py       personalized fine-tuning → models/personalized/from_global/
```

> ⚠️ **Steps 5–6 produce models of the “global training” route, which cannot be used to report generalization scores** (they saw everyone during training).
> For the honest baseline used in the paper, go through [`../experiments/loso_train_and_save.py`](../experiments/loso_train_and_save.py)
> → `phase2_finetune_loso.py`; see [`../models/README.md`](../models/README.md#-which-one-should-i-use) for why.

On Windows, run with `PYTHONIOENCODING=utf-8`:

```powershell
$env:PYTHONIOENCODING = "utf-8"
python pipeline/emg_process_segment_FIXED.py
```

---

## 📄 Files

| File | Purpose |
|---|---|
| `emg_process_segment_FIXED.py` | EMG preprocessing: remove DC → 60Hz notch → 20Hz high-pass → full-wave rectification → 5Hz low-pass envelope. Outputs dual tracks at **60FPS** (for training) and **1000Hz** (for fatigue analysis). Scans the whole Raw folder and processes all subjects automatically |
| `MAX_MVC_FIXED.py` | Normalization denominator. ★ Uses `mvc_denominator()`: the **99.9th percentile of the subject's envelope pooled over all segments**, on the same scale as the numerator (unsmoothed instantaneous values). The old “maximum after a 500ms moving-window average” has been withdrawn — its scale did not match the numerator, structurally inflating the ratio by 1.5–2×; see [Normalization-7][Normalization-8] in the file header |
| `Features_insert_FIXED.py` | Merges skeleton TRC with EMG, feature engineering. **Needs `subjects.csv`** |
| `build_skeleton_dataset.py` | Produces `Dataset/Skeleton/*.npz` (joint-based data for ST-GCN, 20 nodes) |
| `Global_Training_FIXED.py` | Global model training + LOSO evaluation |
| `phase2_finetune_FIXED.py` | Personalized fine-tuning (starting point = global model) |
| `eval_utils.py` | **Shared evaluation utilities**: leak-proof splitting, windowing, Pearson r, nRMSE, compensation thresholds |
| `subjects.example.csv` | Column template for `subjects.csv` |
| `align_mediapipe_trc.py`, `build_combined_mediapipe.py`, `build_combined_trunkfix.py` | Single-camera (MediaPipe) experiment data; see [`../experiments/MEDIAPIPE_EXPERIMENT.md`](../experiments/MEDIAPIPE_EXPERIMENT.md) |

### `eval_utils.py` — the shared copy

The compensation thresholds are centralized in `COMP_THRESHOLDS` + `add_comp_labels()`,
shared by `Global_Training_FIXED.py` and `phase2_finetune_FIXED.py`,
putting an end to “three generations of copies each drifting apart”.

| Compensation | Current threshold | AUC | Notes |
|---|---|---|---|
| `heel_raise` | 0.020 | 0.616 | Weak AUC, conservative first |
| `knee_valgus` | **0.758** | 0.768 | Changed from 0.92 — the old value misclassified **56.3% of frames** of the normal bodyweight group |
| `trunk_lean` | 40.0° | 0.621 | Only 0.7% of the normal group misclassified |

> ⚠️ These are **this dataset's ROC optimal points, not literature values**. The paper must not say “according to the XX standard”.
> ⚠️ A per-frame ROC systematically underestimates separability (compensation only occurs during part of each rep).
> Once rep segmentation is done, rerun it with “peak per rep”, and revisit the thresholds then.

---

## 🔒 `subjects.csv` (contains personal data, not version-controlled)

`Features_insert_FIXED.py` needs this file. Create it yourself with the columns of `subjects.example.csv`:

```csv
subject_key,raw_csv,emg_env_csv,trc_dir,age,height_cm,weight_kg,gender,squat_1rm_kg
S01,SUBJECT01_Raw_DATA.csv,SUBJECT01_CLEAN_SYNCED_Env_60FPS.csv,SUBJECT01_MarkerData,21,175.0,70.0,1,90.0
```

* `gender`: 1 = male, 0 = female
* `raw_csv` / `emg_env_csv` / `trc_dir` are the actual file names and may differ from `subject_key`
  (the folder names chosen when uploading to OpenCap often don't match the EMG file names)
* **It is listed in `.gitignore`; do not remove that rule or force-add it with `git add -f`**

---

## ⚠️ Methodological pitfalls (all of them hit before)

1. **Verify TRC–EMG pairing with a “duration fingerprint”.**
   Never infer recording order from file modification times — the later-recorded `_two` files have timestamps in the middle, misaligning the whole sequence.
   Correctly paired EMG segments and TRCs should differ in duration by less than **±0.05 s**; `Features_insert_FIXED.py` has a built-in automatic check (warns above 0.5 s).
   > This once misaligned the entire pairing of 3 subjects; LOSO dropped from 0.73 to 0.32 and nobody noticed.

2. **TRC duration must not be taken from the header's `NumFrames`.**
   The header is not updated after a file is cropped; the actual `Time` column must be read (last row minus first row).

3. **The built-in health check only compares total duration and has a blind spot.**
   It cannot catch “total duration right, but head and tail each off a little”.
   When needed, use [`../experiments/check_event_alignment.py`](../experiments/check_event_alignment.py) for an event-alignment check.

4. **When changing the segment structure of a Clean CSV, always check `segment_ids` in `SUBJECTS` as well**,
   and delete the Combined files with the old numbering — otherwise orphaned leftover files with a different number of columns are produced.

5. **Never use `Load_1RM_Ratio` as a feature** — it is unknown at deployment and, being highly correlated with muscle activation, leaks the answer.

6. **`EMG_*_MVC` is not a true %MVC**; it is normalized to the maximum activation observed for that subject.
   See item 6 of the [root README](../README.md#-read-before-citing-numbers) for details.
