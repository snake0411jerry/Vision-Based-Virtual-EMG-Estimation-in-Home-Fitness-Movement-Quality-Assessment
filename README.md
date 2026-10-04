# Vision-Based Virtual EMG Estimation in Home Fitness Movement Quality Assessment

*English edition of the sEMG-Teensy-Collector project code.*

Predicting **muscle activation** during squats from **markerless motion capture** (OpenCap skeleton coordinates),
with the goal of **not needing electrodes** at deployment.

> ⚠️ "Activation" here is not a true %MVC (this project has no MVC recordings);
> it is normalized to **the maximum activation observed for that subject**. See item 6 of [Read before citing numbers](#-read-before-citing-numbers) for details and reasons.

Two muscles are measured: the main muscle (Raw0) and a synergist (Raw1). The data are 10 subjects, squats, multiple loads, 89 segments in total.
The project contains Teensy 4.1 acquisition firmware, Python signal preprocessing, and the deep learning models with the full set of experiments.

> 📦 **About this repository.** This is the English edition of the project's code. It ships the code, documentation, figures and
> **trained model weights**, but **no data**: the dataset, subject information and all CSV files (including the experiment result
> tables under `results/`) are excluded. Scripts that read result CSVs will regenerate them when you rerun the experiments.

---

## ⚡ I want to do X — where to go

| I want to… | Go here | Notes |
|---|---|---|
| **Run inference with the trained models** | [`models/global/`](models/) | See [Using the trained models directly](#using-the-trained-models-directly) |
| **Know which models exist and how they differ** | [`models/README.md`](models/README.md) | Grouped by **training method**, with scores and intended use for each |
| **See the numbers for an experiment** | [`results/README.md`](results/README.md) | Which program produces each output and how to read it |
| **Rerun an experiment** | [`experiments/README.md`](experiments/README.md) | Purpose, parameters and outputs of each program |
| **Rerun the whole pipeline from raw data** | [`pipeline/README.md`](pipeline/README.md) | Needs `Dataset/` (not in this repo) |
| **Modify the hardware / flash the firmware** | [`firmware/README.md`](firmware/README.md) | Teensy 4.1 + Grove EMG |
| **Check the environment is set up** | `python tools/smoke_test.py` | No dataset needed, 30 seconds |
| **Check for personal-data leaks before uploading** | `python tools/check_no_pii.py` | See [Subject personal data](#-subject-personal-data-read-this-first) |
| **Know the pitfalls / how to cite numbers** | [Read before citing numbers](#-read-before-citing-numbers) | **Read it in full before writing a paper** |
| Find early old code | [`legacy/`](legacy/) | No longer maintained, kept for reference |

---

## 🗂️ Directory structure

```text
sEMG-Teensy-Collector/
├── paths.py              ★ all paths are centralized here — never write absolute paths again
├── requirements.txt
│
├── firmware/             Teensy 4.1 firmware (.ino), including past versions
├── pipeline/             ★ core pipeline: preprocessing → feature merging → training → personalized fine-tuning
├── experiments/          paper experiments and diagnostics (only .py; all outputs go to results/)
├── results/              experiment outputs: figures (result CSVs are not included in this repo)
├── models/               ★ trained models, grouped by "training method"
│   ├── global/               global training (everyone trained together)
│   ├── loso_zeroshot/        each LOSO fold (the only honest cross-subject baseline)
│   ├── personalized/         personalized fine-tuning
│   └── ablation/             ablations and control groups
├── tools/                smoke_test.py / check_no_pii.py
└── legacy/               early code (pilot + v1), no longer maintained
```

**Not in this repo**: `Dataset/` (about 762MB, contains raw subject signals) and
`pipeline/subjects.csv` (subject personal data). How to obtain each is described in the sections below.

---

## 🚀 Quick start

### 1. Get the code and environment

```bash
git clone https://github.com/snake0411jerry/Vision-Based-Virtual-EMG-Estimation-in-Home-Fitness-Movement-Quality-Assessment.git
cd Vision-Based-Virtual-EMG-Estimation-in-Home-Fitness-Movement-Quality-Assessment

conda create -n emg_env python=3.10
conda activate emg_env
pip install -r requirements.txt
```

### 2. Check everything works

```bash
python tools/smoke_test.py
```

This checks path derivation and that every kind of model file is present, and actually loads the models and runs inference once.
**No dataset needed**, so it can be run right after cloning.

### 3. Dataset path (only needed to rerun the pipeline)

`Dataset/` is not under version control. By default it is looked up in the repo's **parent directory**:

```text
<project-root>/
├── Code/          ← this repo
└── Dataset/       ← default location
```

If it lives elsewhere, just set an environment variable — **no code changes needed**:

```powershell
# Windows PowerShell
$env:SEMG_DATASET = "E:\my\Dataset"
```
```bash
# Linux / macOS
export SEMG_DATASET=/mnt/data/Dataset
```

> 💡 On Windows, run Python with `PYTHONIOENCODING=utf-8`; otherwise non-ASCII output (e.g. emoji) can fail under the cp950 console encoding.

---

## Using the trained models directly

```python
import os, joblib, tensorflow as tf, numpy as np
import paths                                    # paths.py in the repo root

d    = paths.GLOBAL_MODEL_DIR
model = tf.keras.models.load_model(os.path.join(d, 'global_fitness_model.keras'))
s_ts  = joblib.load(os.path.join(d, 'global_scaler_ts.pkl'))
s_st  = joblib.load(os.path.join(d, 'global_scaler_static.pkl'))
spec  = joblib.load(os.path.join(d, 'global_feature_spec.pkl'))   # ts_cols / static_cols

# input: ts_input (N, 40, 16) skeleton time series, static_input (N, 4) subject static features
# output: (N, 2) → [main-muscle normalized activation, synergist normalized activation]
pred = model.predict({'ts_input': X_ts, 'static_input': X_static})
```

* The input window is **40 frames** (about 0.67 s @ 60FPS); the **last frame of the window** is the prediction target.
* The feature column order **must** follow `ts_cols` in `global_feature_spec.pkl`; do not order them yourself.
* `EMG_*_MVC` stores a **ratio from 0 to 1.5**: normalized to the 99.9th percentile of the subject's pooled envelope over all segments,
  **not a true %MVC** (see item 6 below). Multiplying by 100 only changes the scale to 0–150; the meaning is unchanged, and it must not be compared with %MVC values in the literature.

⚠️ **To evaluate generalization, do not use `models/global/`** — it saw every subject during training.
Use `models/loso_zeroshot/` instead; see [Read before citing numbers](#-read-before-citing-numbers) for why.

---

## 📊 Main results

10 subjects, squats, two channels (main muscle and synergist). **After the coordinate-axis fixes, 16-dim features**.

| | Zero-shot LOSO<br>(has not seen this person) | After personalized fine-tuning | Single-subject upper bound<br>(not a deployment scenario) |
|---|---|---|---|
| **Main muscle r** | 0.735 ± 0.013 | **0.830 ± 0.006** | 0.895 ± 0.002 |
| Main muscle nRMSE | 0.803 ± 0.019 | 0.607 | — |
| **Synergist r** | 0.462 ± 0.017 | 0.524 ± 0.008 | 0.682 ± 0.004 |
| Synergist nRMSE | 0.986 ± 0.022 | 0.816 | — |

> The zero-shot synergist numbers **exclude S09** (that channel likely had poor electrode contact); the main muscle uses all 10 people.
> The fine-tuned main-muscle value is the mean ± sd over **8 complete pipeline runs**.

**Other completed control experiments** (details in [`experiments/README.md`](experiments/README.md)):

| Experiment | Conclusion |
|---|---|
| Split method (Experiment A) | Random splitting inflates by **+0.164** (main muscle) / **+0.302** (synergist) |
| Feature ablation (Experiment C) | Adding v/a **does nothing for EMG regression** (p=0.99); 16 dims are enough |
| Load recognition (%1RM) | But adding v/a **helps load recognition significantly** (within-subject r 0.590 → 0.753) |
| Architecture comparison | TCN-Transformer has 4.8× the parameters with **no difference** in score → keep Conv1D |
| ST-GCN student network | Main muscle r 0.729 → **0.768** (p=0.0098), but **no improvement at all for the synergist** |
| Heterogeneous knowledge distillation (Experiment C) | **No measurable benefit** (+0.005, p=0.60, some of the 5 seeds go the other way). But the teacher's privileged-information advantage of **+0.081** is extremely stable — see below |

---

## ⚠️ Read before citing numbers

These rules were established after hitting the pitfalls. **Read them all before writing a paper or making comparisons.**

1. **Always split by subject (LOSO), never by file or by window.**
   Measured inflation: "the person was seen during training" **+0.081**, and "adjacent overlapping windows in both training and test" another **+0.082**;
   temporal leakage is even more dramatic for the synergist: **+0.225** from that alone.
   > 📌 Under the honest protocol the synergist nRMSE = 1.10 (**worse than guessing the mean**); with a random window split it becomes 0.657 and r jumps to 0.718.
   > That 0.30 of correlation **contains no kinematic information; the model simply memorized adjacent frames**.

2. **The base model for personalized fine-tuning must exclude that subject**, otherwise the "before fine-tuning" score is inflated.
   This is why `models/loso_zeroshot/` exists.

3. **Report r and nRMSE together.** r is immune to linear transforms (scaling all predictions by ×0.5 leaves r unchanged) and cannot reveal systematic over/underestimation;
   **nRMSE = 1 means "no better than guessing the mean"**, a pass line with physical meaning.

4. **A single run cannot resolve differences of order 0.02.** The sd across seeds is about 0.015–0.018, with a range of 0.039.
   Any claim of "changing X improved things by this much" needs multiple seeds (use `experiments/loso_multiseed.py`).

5. **Numbers from different runs cannot be subtracted directly.** cuDNN kernel nondeterminism means even a fixed seed is not bit-reproducible.
   For the same 16-dim LOSO, three independent runs gave 0.740 / 0.729 / 0.723 — **the dispersion across runs is larger than within a run**.
   → **Any comparison must run both arms inside the same job.**
   (But **inference is fully reproducible** — given a set of saved models, the evaluation numbers can be reproduced reliably.)

6. **`EMG_*_MVC` is not a true %MVC** — this project has no MVC recordings (the segment counts in `Dataset/Raw` map
   one-to-one to `Dataset/Combined`, and all of them are squat segments). The denominator is the **99.9th percentile of the subject's
   envelope pooled over all segments** (`mvc_denominator()` in `pipeline/MAX_MVC_FIXED.py`), meaning "normalized to
   the maximum activation observed for that subject". The paper must not say %MVC, nor compare directly with %MVC values in the literature
   (the column name `EMG_*_MVC` is kept only for compatibility with existing code).
   > Versions before 2026-08-05 used "the maximum after a 500ms moving-window average" as the denominator, which was on a different scale
   > from the numerator (unsmoothed instantaneous values), structurally inflating the ratio by 1.5–2×; this has been withdrawn.

7. **Never use `Load_1RM_Ratio` as a feature** — it is unknown at deployment and, being highly correlated with muscle activation, leaks the answer.

8. **Always use S01–S10 in public; real names must never appear.**

9. The compensation thresholds (`COMP_THRESHOLDS` in `pipeline/eval_utils.py`) are **this dataset's ROC optimal points, not literature values**.
   The paper **must not say "according to the XX standard"**.

<details>
<summary>📌 Known limitations (click to expand)</summary>

* **`models/personalized/from_loso/` is a low draw**: r=0.814 after fine-tuning, the lowest of 9 observations (−2.56 sd).
  Analyses that load them directly (waveform plots, calibration curves) are conservative by about 0.016.
  Before making figures or demos, rerun `experiments/loso_train_and_save.py` to get models closer to the centre of the distribution.
* **S02 has abnormal between-segment dispersion**: between-segment sd 0.162 (others 0.030–0.113), zero-shot r 0.632, the lowest of all.
  This is not a single-segment problem; this subject's data are inconsistent overall.
* **S09's synergist** likely had poor electrode contact and is excluded from evaluation (the main-muscle quality is normal and fully retained).
* **LOSO overfitting**: training curves show val bottoming out around epochs 3–5 and then worsening; 30 epochs is currently too many.
  Tuning this requires nested LOSO; never look at the LOSO val directly (that is the test set).
* **Monocular depth error** is the biggest unknown for "removing the electrodes": 5 of the 16 features use the depth axis, and MediaPipe's monocular z is unreliable.
  (See [`experiments/MEDIAPIPE_EXPERIMENT.md`](experiments/MEDIAPIPE_EXPERIMENT.md) for the single-camera experiments.)
* **The EMG amplitude itself may be unobservable.** Experiment C measured: when the teacher additionally sees the subject's true EMG,
  its score jumps from 0.758 to **0.840** (consistent across all 5 seeds); but distilling this advantage into a student that only sees the skeleton
  **transfers nothing** (+0.005, p=0.60). The content of the privileged signal is "this person's EMG amplitude and gain",
  information that skeleton kinematics **in principle** does not contain — it is not that the model is too small or the training tricks are wrong.
  This draws a ceiling for the "remove the electrodes" route.

</details>

---

## 🔒 Subject personal data (read this first)

The subjects of this project are real people; **de-identification is a hard requirement**.

### Rules

* Code, file names, figures and commit messages always use **S01–S10**; **real names or nicknames must never appear**.
* Subjects' age / height / weight / sex / 1RM are stored in `pipeline/subjects.csv`, which is **gitignored and never version-controlled**.
* The S01–S10 ↔ real-name mapping table is kept **outside the repo** and must not be copied in.

### To run the full pipeline

`pipeline/Features_insert_FIXED.py` needs `pipeline/subjects.csv`.
Create it yourself with the columns of [`pipeline/subjects.example.csv`](pipeline/subjects.example.csv):

```csv
subject_key,raw_csv,emg_env_csv,trc_dir,age,height_cm,weight_kg,gender,squat_1rm_kg
S01,SUBJECT01_Raw_DATA.csv,SUBJECT01_CLEAN_SYNCED_Env_60FPS.csv,SUBJECT01_MarkerData,21,175.0,70.0,1,90.0
```

(`gender`: 1 = male, 0 = female)

**Do not remove the `subjects.csv` rule from `.gitignore`, and do not force-add it with `git add -f`.**

### Self-check before uploading

```bash
python tools/check_no_pii.py          # scan git-tracked files
python tools/check_no_pii.py --all    # also scan untracked files
```

It detects: a custom list of real names, absolute paths of the development machine (including the user account), `subjects.csv` or the mapping table accidentally under version control,
and suspected national ID numbers / mobile phone numbers.
The list of real names lives in `tools/pii_names.txt` (**gitignored**); see
[`tools/pii_names.example.txt`](tools/pii_names.example.txt) for the format.

> ℹ️ This English repository was created with a **fresh git history** and has never contained subject personal data.

---

## 💻 Hardware

* **Microcontroller**: Teensy 4.1
* **Sensors**: Grove EMG Sensor ×2 (main muscle Raw0, synergist Raw1)
* **Communication**: Serial; see the code in `firmware/` for the baud rate
* For pins and SD card settings see [`firmware/README.md`](firmware/README.md)

---

## 👥 Development team

* 林寬泓
* 林士閔
* 吳羽絜
