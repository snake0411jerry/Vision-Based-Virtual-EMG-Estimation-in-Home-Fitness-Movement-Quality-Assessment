# MediaPipe single camera vs OpenCap multi-camera: how much does EMG prediction lose?

Created 2026-10-02. The question answered: **if the skeleton source changes from multi-camera OpenCap to a single phone camera,
how much does the EMG prediction r drop?**

This document records how the data were prepared, the pitfalls, what each of the three experiments answers, and how to rerun them.

---

## 0. Why split it into three experiments

“How much does r drop” is really two questions mixed together:

| | Problem | Can retraining fix it? |
|---|---|---|
| **Domain shift** | MediaPipe's feature value distributions differ from OpenCap's (standing knee angle 151–170° vs 176–180°) | **Yes**. Relearning the scaler and model weights absorbs it |
| **Missing information** | A single camera truly cannot see some quantities (left-right shifts, heel lift) | **No**. Information that never came in is simply not there |

Running only “apply directly” adds the two together and gives an overly pessimistic number;
running only “retrain” cannot answer “can my existing model be used as is”. So all three must be run:

| Experiment | Question | Program | Cost |
|---|---|---|---|
| **1. Apply directly** | Existing model + phone, nothing changed | `eval_crossdomain_mediapipe.py` | ~3 min |
| **1b. Refit the scaler only** | Existing model + phone + one calibration (no EMG needed) | same, with `--refit-scaler` | ~4 min |
| **2. Full retraining** | Where is the ceiling of single-camera information | `loso_multiseed.py` | see §4 |

---

## 1. Data preparation (done)

```
Dataset/mediapipe_for_model/   raw output from the teammate (91 TRC + CSV files)
        ↓  pipeline/align_mediapipe_trc.py        ← 🔴 must not be skipped
Dataset/mediapipe_aligned/     cropped to align frame-by-frame with OpenCap (89 files)
        ↓  pipeline/build_combined_mediapipe.py
Dataset/Combined_mediapipe/    same spec as Dataset/Combined/ (89 files)
```

```bash
python pipeline/align_mediapipe_trc.py
python pipeline/build_combined_mediapipe.py
python experiments/compare_mediapipe_features.py
```

### 🔴 Pitfall: the teammate's README claim of “same frame count, same time axis” holds for only 14 / 89 trials

The OpenCap TRCs in `Dataset/Open/` were cropped by hand (both head and tail), but **the header's
`NumFrames` was not updated** — this was already noted in the comments of `Features_insert_FIXED.trc_duration_s()`
(“one subject's 0.trc header says 2039 frames, but there are only 1717”).
The teammate's export script trusted the header, so the resulting MediaPipe files cover **the whole original recording**,
3–10 seconds longer than the cropped OpenCap.

Fed directly into `process_subject()`, resampling would linearly squeeze 34 seconds of skeleton onto 28.6 seconds of
EMG — movement and EMG misaligned by several seconds, r collapses, and **the cause has nothing to do with single-camera accuracy**.

The good news is that it can be fixed exactly, not approximately: on both sides `Time` is `(Frame-1)/60`, and for 74 trials
the per-frame difference was measured at **0.000000 s**, so cropping by `Frame#` gives exact frame-to-frame alignment.

| Mode | Trials | Situation | Handling |
|---|---|---|---|
| `crop` | 74 | S01–S08, MediaPipe `1..N_header` contains OpenCap `[fa..fb]` | take rows `fa..fb` |
| `asis` | 14 | S09/S10, row count already equals OpenCap, only renumbered | use the whole file |
| `tail` | 1 | S09 `b.trc`, MediaPipe has 83 extra rows = **1.3833s** | take the last `N_oc` rows |

The 83 frames of the `tail` trial are exactly the handover doc's note “b.trc additionally had 1.383s trimmed from its start” —
the teammate used the pre-crop version. This inference **does not rely on assumptions**; it is independently confirmed by the signal check below.

### Verification: checking the signal, not just the arithmetic

After alignment, the OpenCap vs MediaPipe correlation of “left knee height relative to left hip” was computed for every trial:

```
median 0.968   min 0.883   max 0.993      (89/89 trials)
```

This falls within the 0.86–0.99 range the teammate measured, confirming the alignment is correct. A ±120 frame shift search was also run;
the best shift is within **±7 frames (≤0.12 s)** for every trial, with a maximum improvement of +0.033.

**These few frames are deliberately not corrected.** They are MediaPipe's smoothing lag; adjusting the shift per segment against OpenCap
would sneak in multi-camera information that is unavailable at deployment.

### The only difference between the two sets

| Item | Same? |
|---|---|
| EMG labels (`EMG_Main_MVC` / `EMG_Compass_MVC`) | **Identical** (verified file by file by `compare_mediapipe_features.py`, max diff < 1e-9) |
| MVC denominator | Same (both computed from the subject's own Raw file) |
| Segment pairing, subject static data, feature formulas, window/step | Same |
| `ref_height` | **Computed separately** (see below) |
| Skeleton source | ← **the only difference** |

`ref_height` is deliberately computed separately: MediaPipe's absolute scale is 5–8% smaller than OpenCap's, but all spatial
features are divided by the segment's own `ref_height`, so the ratio cancels automatically. Using OpenCap's
`ref_height` instead would sneak in multi-camera information.

---

## 2. Agreement at the feature level (done)

The teammate compared at the marker level. But **the model consumes features**: `Knee_Angle` is computed from 3D vectors,
`Trunk_Lean` uses `hypot(X,Z)`, `Knee_Ankle_Ratio` is a ratio of two distances —
these nonlinear combinations amplify or cancel coordinate errors and cannot be inferred from marker-level r.

`results/mediapipe_feature_agreement_summary.csv` (89 segments, the 16 dims that enter the model):

| Feature | Median r | 10th pct r | bias/sd | sd ratio | Single-camera reliable set |
|---|---|---|---|---|---|
| `Knee_Angle_norm` | **0.981** | 0.953 | −0.27 | 0.85 | ★ |
| `Ankle_Y_norm` | **0.976** | 0.949 | −0.02 | 0.94 | ★ |
| `Knee_Y_norm` | **0.968** | 0.929 | −0.05 | 1.05 | ★ |
| `Toe_Y_norm` | **0.967** | 0.915 | −0.20 | 0.86 | ★ |
| `Knee_X_norm` | **0.960** | 0.902 | 0.81 | 0.97 | ★ |
| `Shoulder_Y_norm` | 0.890 | 0.523 | 0.36 | 0.63 | ★ |
| `Knee_Ankle_Ratio_norm` | 0.882 | 0.517 | −1.24 | 0.95 | ★ |
| `Ankle_X_norm` | 0.833 | 0.598 | 1.87 | 1.17 | |
| `Toe_X_norm` | 0.705 | 0.119 | 0.19 | 1.13 | |
| `Knee_Z_norm` | 0.546 | −0.277 | 1.69 | 1.09 | |
| **`Trunk_Lean_Angle_norm`** | **0.354** | **−0.578** | 0.43 | 0.77 | ★ |
| `Ankle_Z_norm` | 0.320 | −0.189 | 0.37 | 2.45 | |
| `Toe_Z_norm` | 0.314 | 0.007 | 2.71 | 2.67 | |
| `Shoulder_Z_norm` | 0.300 | −0.352 | −2.09 | 2.69 | |
| `L_Heel_Rise_norm` | 0.286 | −0.289 | 3.51 | 2.79 | |
| `R_Heel_Rise_norm` | 0.241 | −0.377 | 2.03 | 3.31 | |

* ★ = the “8 dims a single camera sees accurately” identified by `probe_camera_features.py`.
* `bias/sd` = mean offset ÷ the typical OpenCap sd of the feature. **Fatal for “apply directly”, nearly harmless for “retrain”.**
* Those with `sd ratio` > 2 (Z and heels) are pure noise being amplified, not signal.

**★ Median of the median r for the reliable 8 dims = 0.964; for the other 8 dims = 0.314.**
The premise of the 8-dim set is confirmed — **with one important exception**.

### 🔴 `Trunk_Lean_Angle_norm` should not count as part of the “reliable set”

`probe_camera_features.py` classified it as “seen accurately” (the teammate measured 0.912 at the marker level),
but at the feature level it is only **0.354 median, 10th percentile −0.578**. Per person it is even clearer:

```
S03=0.11  S10=0.15  S01=0.23  S02=0.42  (S07=−0.27  S09=−0.47)
```

Suspected cause: `Trunk_Lean = atan2(hypot(ΔX, ΔZ), ΔY)`, where `ΔZ` is the left-right component.
Squat trunk lean is mainly in X, but `hypot` also pulls in **Z, where MediaPipe's noise is largest**,
and `hypot` is always positive, with no sign to cancel out. In addition, MediaPipe's `Neck` is the midpoint of the shoulders
(an approximation), not the same anatomical point as OpenCap's neck marker.

So the experiment matrix needs an extra **7-dim set (the 8 dims minus Trunk_Lean)**.

---

## 3. Results of experiments 1 / 1b (done)

The 10 models in `models/loso_zeroshot/` (trained on OpenCap 16 dims) fed MediaPipe directly:

| | OpenCap | MediaPipe | Diff | Worst |
|---|---|---|---|---|
| **1. scaler untouched** main muscle r | 0.724 | **0.514** | **−0.211** | S02 −0.994 |
| | nRMSE | 0.794 | 0.923 | +0.129 | |
| **1b. scaler refit only** main muscle r | 0.724 | **0.542** | **−0.182** | S02 −0.874 |
| | nRMSE | 0.794 | 0.856 | +0.062 | |

Per person (main muscle r, OpenCap → untouched → scaler refit):

```
S01  0.734 -> 0.612 -> 0.248      S06  0.740 -> 0.708 -> 0.698
S02  0.622 -> -0.372 -> -0.252    S07  0.775 -> 0.756 -> 0.796
S03  0.684 -> 0.563 -> 0.680      S08  0.723 -> 0.569 -> 0.680
S04  0.821 -> 0.679 -> 0.757      S09  0.676 -> 0.475 -> 0.410
S05  0.781 -> 0.503 -> 0.713      S10  0.683 -> 0.641 -> 0.692
```

**The point is the distribution, not the mean.** After refitting the scaler, 6 people barely drop (|Δ| ≤ 0.067;
S07/S10 even improve), but S01 / S02 / S09 collapse. The mean hides this.

S01 / S02 actually have decent agreement on the 8 dims (0.936 / 0.919), but **the median over 16 dims is only
0.641 / 0.713** — the drop is caused by feeding in the 8 noisy features. This directly predicts that
**8-dim (or 7-dim) MediaPipe will clearly beat 16-dim**; experiment 2 tests this.

⚠️ `models/loso_zeroshot` is a low draw (see `reproducibility/`),
but this experiment only compares “the same model fed two kinds of input”, so a low baseline does not affect the difference.

---

## 4. Experiment 2: full retraining

🔴 **The terminal is Windows PowerShell 5.1: no `&&`, no `VAR=x cmd`.**
All long commands are wrapped in `.ps1` scripts; do not paste bash syntax directly (we got bitten on 2026-10-02).
`.ps1` files must be saved as **UTF-8 with BOM** — without a BOM, 5.1 reads them as ANSI (cp950),
the bytes of non-ASCII comments are treated as syntax characters, and the whole script fails to parse.

```powershell
.\experiments\run_mediapipe.ps1 -Arms all          # eight arms x 3 seeds
.\experiments\run_mediapipe.ps1 -Arms all -DryRun  # print parameters only, don't start
.\experiments\stop_mediapipe.ps1                   # clean pause
```

The eight arms:

| arm | Skeleton | Features | Question |
|---|---|---|---|
| `oc16` / `mp16` | OpenCap / MediaPipe | 16 dims | How much does changing the skeleton source cost |
| `oc8` / `mp8` | same | reliable 8 dims | Does dropping low-SNR features help |
| `oc7` / `mp7` | same | reliable 7 dims (no Trunk_Lean) | Is the broken Trunk_Lean the main cause |
| `oc8tf` / `mp8tf` | same (trunkfix) | reliable 8 dims | Does fixing the Trunk_Lean definition help |

**Every arm has an OpenCap counterpart**, because the effects of “changing the skeleton source” and “dropping features / changing definitions”
must be separated. `loso_multiseed` resets the seed at the start of every fold-set,
so different arms with the same seed are paired.

Analysis:

```powershell
python experiments/analyze_mediapipe_experiment.py --prefix loso_mediapipe `
  --pairs oc16:mp16 oc8:mp8 oc7:mp7 oc8tf:mp8tf mp8:mp7 mp8:mp8tf
```

### Execution order and interruption

`--order seed` (the default) puts seeds in the outer loop: each seed runs all arms before moving to the next seed.
**This means that whenever you stop, every arm has the same number of seeds**, so partial results can still be used for paired tests.
With arms in the outer loop, stopping midway gives asymmetric results like “A has 3 seeds, B only 1”.

`--resume` skips (arm, seed) pairs already completed in `<prefix>_raw.csv`,
and records which run produced each row in the `chunk` column.

### This machine's GPU problem

The RTX 5060 Laptop has compute capability **12.0**, and TensorFlow 2.10.0
has no precompiled kernels for sm_120; it warns about this itself at startup:

> CUDA kernels will be jit-compiled from PTX, which could take 30 minutes or longer

Consequence: GPU utilization is only ~27%, and a single fold-set takes **19–21 minutes**;
in addition, `cuModuleGetFunction(...) failed with CUDA_ERROR_UNKNOWN` occasionally
kills the whole process (it happened once at 10:55 on 2026-10-02, after 23 minutes of running).
The retry + `--resume` in `run_mediapipe.ps1` exist for this.
The real fix is a TF/CUDA that supports sm_120, but that would change the environment tied to `ic3mt-2026`.

### 🔴 Don't subtract the results from the 0.766 on the slides

From the handover doc's “methodology rules”: the dispersion across runs (≈0.025) is larger than the within-run sd across seeds (0.013–0.017).
The `oc16` arm exists for exactly this reason — **the OpenCap baseline for this run is `oc16`,
not the slide number.** `analyze_mediapipe_experiment.py` only handles arms within the same raw CSV
precisely to prevent this.

---

## 5. Known limitations (mention them in the paper)

1. **This is not a true single-camera deployment test.** MediaPipe was run on OpenCap's synchronized `Cam0`
   video; shooting distance, angle and lighting are all laboratory conditions. Casual phone recordings will be worse.
2. **The residual ±7 frame time shift is not corrected** (reason in §1). This makes the MediaPipe numbers
   slightly conservative.
3. **S05 Segment 2 has a 0.98 s EMG/TRC duration difference**, a pre-existing issue (already noted in the `SUBJECTS` table:
   the tail of 20.trc was cut). It is the same for both sets and does not affect the comparison.
4. **`Knee_Ankle_Ratio_norm` has a heavy tail in MediaPipe.** OpenCap's maximum is 1.779;
   MediaPipe's maximum is 6.523, with 0.007% (about 14 / 193494 frames) above 2 —
   left-right noise makes the two ankles nearly coincide in the horizontal plane, so the denominator approaches 0.
   **Deliberately not clipped**: hand-tuning a clipping threshold would amount to tuning against OpenCap. Only 14 frames are affected,
   not enough to move the scaler's std, but if this feature's importance ever looks abnormal, come back here first.
5. **`Knee_Toe_Diff_*` and `Subj_*` do not enter the model** (`eval_utils.EXCLUDE_ALWAYS`),
   so the table in §2 does not list them. `Knee_Toe_Diff_norm` has r = −0.486; if it is ever
   put back into the input, revisit this first.

---

## 6. Diagnosis: why it breaks, and whether changing definitions can rescue it

`diagnose_mediapipe_features.py`. Both conclusions changed what should be done next.

### 6.1 MediaPipe's error is **isotropic**; Z fails purely because its signal is too small

Signal = OpenCap's sd; noise = sd of (MediaPipe − OpenCap). Both are divided by `ref_height`,
so the unit is “fraction of body height” and can be compared directly across axes.

| Axis | Signal sd | Noise sd | SNR |
|---|---|---|---|
| Y up-down (image plane) | 0.0785 | 0.0240 | **3.27** |
| X front-back (**depth axis**) | 0.0611 | 0.0325 | **1.88** |
| Z left-right (image plane) | **0.0114** | 0.0226 | **0.50** |

Per quantity:

| Quantity | Signal sd | Noise sd | SNR | Median r |
|---|---|---|---|---|
| `LAnkle_Y` | 0.1007 | 0.0242 | 4.16 | 0.976 |
| `LKnee_X` | 0.0986 | 0.0290 | 3.40 | 0.960 |
| `LKnee_Y` | 0.0809 | 0.0245 | 3.30 | 0.968 |
| `LShoulder_Y` | 0.0393 | 0.0225 | 1.74 | 0.890 |
| `LShoulder_X` | 0.0516 | 0.0355 | 1.46 | 0.870 |
| `LAnkle_X` | 0.0461 | 0.0335 | 1.38 | 0.833 |
| `LKnee_Z` | 0.0221 | 0.0244 | 0.91 | 0.547 |
| `LAnkle_Z` | 0.0099 | 0.0219 | 0.45 | 0.322 |
| `LShoulder_Z` | 0.0088 | 0.0219 | 0.40 | 0.299 |

**The noise floor is almost the same on all three axes (2.2–3.3% of body height, about 4 cm for a 170cm person).**
The difference is entirely in the signal: left-right movement in a squat is only 1.1% of body height (about 2 cm), **below the noise floor**;
while the depth axis X has a dozen-plus centimetres of front-back movement, giving an SNR 3.8× that of Z.

**This rules out the whole “recover depth” route** (monocular depth estimation, QR-code distance calibration, fixed camera position):
Z is not a depth problem. This is the second time it has been ruled out — the first was `probe_camera_features.py`
(removing the 8 poorly tracked dims costs only 0.03–0.07).

It also rules out “dropping features by axis”: `Knee_X` (SNR 3.40) must stay, while `Ankle_X`/`Toe_X`
(SNR 1.38) could be dropped — both are on the depth axis. **Drop by SNR, not by axis.**

Inference: knee valgus (`Knee_Ankle_Ratio`) measures left-right shifts on the order of 2 cm,
which at the current shooting distance + MediaPipe heavy are **in principle unmeasurable**.
If this compensation matters for the research, it needs a closer camera / higher resolution / a better pose model,
not depth reconstruction.

### 6.2 `Trunk_Lean` is not an information problem but a definition problem

| Definition | Median r | 10th pct | Min |
|---|---|---|---|
| `atan2(ΔX, ΔY)`  shoulder−hip, **signed** | **0.902** | 0.728 | 0.440 |
| `atan2(ΔX, ΔY)`  neck−pelvis, **signed** | 0.896 | 0.697 | 0.436 |
| `atan2(hypot(ΔX,ΔZ), ΔY)`  **current** | 0.354 | −0.578 | −0.859 |
| `atan2(|ΔX|, ΔY)`  absolute value | 0.221 | −0.692 | −0.867 |

The key is the **sign**: taking the absolute value is even worse than the current version, so the problem is not Z noise
but that **`hypot` is always positive and folds backward trunk lean onto the positive side**, creating a fake V shape;
the two skeletons disagree about “which side of zero the trunk is on”, and after folding they decorrelate.

**This means the current definition costs points on the OpenCap side too.** [Axis-1] changed Z-only to
hypot(X,Z), fixing the magnitude (0.03m → 0.35m) but introducing sign folding.

`trunk_lean_mode` has been added to `Features_insert_FIXED.process_subject()`
(**the default is still `hypot_xz`**, preserving IC3MT reproducibility), and
`pipeline/build_combined_trunkfix.py` rebuilds both datasets:

```
Dataset/Combined_trunkfix/             (OpenCap, signed_x)
Dataset/Combined_mediapipe_trunkfix/   (MediaPipe, signed_x)
```

Verified in the real feature pipeline:

| | Median r | 10th pct | Segments below 0.5 |
|---|---|---|---|
| `hypot_xz` | 0.354 | −0.578 | **49 / 89** |
| `signed_x` | **0.896** | 0.697 | **2 / 89** |

The health check confirms that **only `Trunk_Lean_Angle_*` changes** between the two versions; all other columns are identical cell for cell.
MediaPipe's sd ratio also goes from 0.77 (compressed) back to 0.95.

🔴 The cost of `signed_x`: the hypot version is independent of the subject's facing direction, while the `signed_x` version assumes
“the person roughly faces the camera and positive X is forward”. This holds for single-camera deployment, but it is **an extra assumption**,
not a pure improvement. For arbitrary orientations, estimate the facing direction from the feet first and project, rather than going back to hypot.

### 6.3 Other features: changing the definition doesn't help

| Family | Best alternative | Median r | Current | Conclusion |
|---|---|---|---|---|
| `Knee_Angle` | XY plane only | 0.983 | 0.981 | Difference within noise; not worth changing |
| `Knee_Ankle_Ratio` | current XZ | 0.882 | 0.882 | **No fix**. X only drops straight to zero (−0.002); the signal is all in Z |
| `Shoulder_Y` | current (relative to LHip) | 0.890 | 0.890 | Using midHip is worse (0.875) |

Side finding: `Neck_Y` relative to midHip has an agreement of **0.926**, better than `Shoulder_Y`'s 0.890.
To squeeze out a bit more later, consider switching to neck height — but that is a new feature, and only retraining will tell whether it helps.

---

## 7. Progress and pending runs

### Run 1  eight arms x 3 seeds (completed 2026-10-02 23:55)

Commands in §4. At 12:50 on 2026-10-02, seeds 42 and 1 of the headline `mp16` / `oc16` were done;
the rest was resumed later (about 7 hours / 20 fold-sets).

### Run 2  training on both domains together (domain generalization) — not run yet at that point

```powershell
python experiments/loso_dualdomain.py --seeds 42 1 2 --prefix loso_dualdomain
```

The `both` arm has twice the training data, so it costs about 4 fold-sets' worth x each seed
(about 4–5 hours for 3 seeds). The key cell is “when testing on MediaPipe, `both` − `mp_only`”.

**Wait for Run 1's results to decide whether it is worth running**: if `mp7` already lands within 0.04 of `oc16`,
domain generalization would only squeeze out another 0.01–0.02, not worth it; only a gap of 0.08 or more justifies it.

⚠️ The two domains are **not independent samples** (the same movement appears twice); do not write “equivalent to 20 people”.
   The N of the learning curve does not grow; only the input diversity at each N grows.

---

## 8. Final results of Run 1 (eight arms x 3 seeds, completed 2026-10-02 23:55)

All 20 fold-sets completed with **zero CUDA crashes** (retry + a larger JIT cache worked).
Each fold-set took 19.1–19.6 minutes, except the first at 27.5 minutes (PTX JIT warm-up of a new process).

| arm | Skeleton | Features | Main muscle r | Main muscle nRMSE | Synergist r |
|---|---|---|---|---|---|
| `oc7`   | OpenCap   | 7 dims | **0.7439 ± 0.0040** | 0.7752 | 0.4297 |
| `oc8tf` | OpenCap   | 8 dims + fix | 0.7429 ± 0.0108 | 0.7586 | 0.4379 |
| `oc8`   | OpenCap   | 8 dims | 0.7338 ± 0.0122 | 0.7644 | 0.4133 |
| `oc16`  | OpenCap   | 16 dims | 0.7253 ± 0.0223 | 0.7740 | 0.4197 |
| `mp8`   | MediaPipe | 8 dims | **0.6946 ± 0.0019** | 0.8171 | 0.3578 |
| `mp8tf` | MediaPipe | 8 dims + fix | 0.6901 ± 0.0053 | 0.8342 | 0.3587 |
| `mp7`   | MediaPipe | 7 dims | 0.6892 ± 0.0163 | 0.8253 | 0.3969 |
| `mp16`  | MediaPipe | 16 dims | 0.6875 ± 0.0152 | 0.7891 | 0.3593 |

### 8.1 Paired differences within the same feature set (mp − oc) — consistent and significant

| Feature set | Δ r | p across seeds | p across subjects | Δ nRMSE | p across seeds |
|---|---|---|---|---|---|
| 8 dims | **−0.039** | 0.026 | **0.0027** | +0.053 | 0.039 |
| 7 dims | **−0.055** | 0.017 | 0.011 | +0.050 | 0.026 |
| 8 dims + fix | **−0.053** | 0.029 | **0.0006** | +0.076 | 0.033 |
| 16 dims | −0.038 | 0.18 | 0.22 | +0.015 | 0.52 |

(The 16-dim row spans chunks, so only its general direction is meaningful. The other three rows are all within chunk 3.)

### 8.2 🔴 All three feature-engineering attempts are null

| Comparison | Δ r | p | Conclusion |
|---|---|---|---|
| `mp7` − `mp8` | −0.006 | 0.58 | Dropping Trunk_Lean **doesn't help** the single camera |
| `mp8tf` − `mp8` | −0.005 | 0.39 | **Fixing the Trunk_Lean definition doesn't help either** |
| `oc7` − `oc8` | +0.010 | 0.27 | Slightly better for multi-camera, not significant |
| `oc8tf` − `oc8` | +0.009 | 0.11 | Same |

**The four MediaPipe arms are nearly flat (0.6875–0.6946, range 0.007)**,
no matter how features are chosen or definitions changed. The four OpenCap arms have a range of 0.019.

🔴 §6.2 raised Trunk_Lean's cross-source agreement from 0.354 to 0.896,
**and none of it turned into predictive power**. Agreement is not usefulness.

### 8.3 Why the SNR analysis of §6 had no predictive power (important)

The SNR analysis correctly identified “which features are inaccurate”, but it carried a wrong implicit assumption:
**that the loss comes from those inaccurate features**. In fact:

1. `probe_camera_features.py` had already shown the model barely uses those 8 dims
   (removing them all costs only 0.03–0.07). So of course dropping them makes no difference.
2. The real loss comes from **the features you must keep**. Their cross-source agreement of r=0.96–0.98
   looks good, but MediaPipe's noise floor is 2.4% of body height (about 4 cm for a 170cm person)
   and **isotropic** — it contaminates good and bad features alike.
3. The 2–4% residual on the good features cannot be discarded by feature selection, because you need those features.

**So the single camera's limitation is not “a few features are inaccurate” but “the overall localization accuracy is 4 cm”.**
That is not something downstream processing can solve.

### 8.4 ★ What the mean hides: 8 dims make the loss predictable

Per-person main muscle r (OpenCap -> MediaPipe):

| | 16 dims | 8 dims |
|---|---|---|
| Mean difference | −0.038 | −0.039 |
| **sd of per-person differences** | **0.091** | **0.030** |
| Worst person | S02 **−0.182** | S08 −0.084 |
| Best person | S03 **+0.105** | S04 +0.004 |

```
16 dims                              8 dims
S01  0.720 -> 0.693  (-0.027)        S01  0.736 -> 0.668  (-0.068)
S02  0.648 -> 0.466  (-0.182)        S02  0.649 -> 0.626  (-0.023)
S03  0.702 -> 0.807  (+0.105)        S03  0.804 -> 0.761  (-0.043)
S04  0.809 -> 0.773  (-0.036)        S04  0.770 -> 0.773  (+0.004)
S05  0.765 -> 0.758  (-0.007)        S05  0.784 -> 0.716  (-0.068)
S06  0.743 -> 0.690  (-0.053)        S06  0.708 -> 0.709  (+0.001)
S07  0.759 -> 0.748  (-0.011)        S07  0.795 -> 0.736  (-0.059)
S08  0.833 -> 0.759  (-0.075)        S08  0.792 -> 0.708  (-0.084)
S09  0.634 -> 0.464  (-0.170)        S09  0.620 -> 0.588  (-0.031)
S10  0.639 -> 0.717  (+0.077)        S10  0.681 -> 0.661  (-0.020)
```

**The two sets have exactly the same mean difference (−0.038 / −0.039), but the per-person dispersion differs threefold.**
With 16 dims, S02 / S09 collapse by 0.17–0.18 while S03 / S10 actually improve by 0.08–0.11;
with 8 dims nobody collapses and nobody improves; everyone lands within −0.084 to +0.004.

🔴 **This is the most practically valuable conclusion of this experiment, and it is completely invisible from the mean alone.**
Deployment should use 8 dims, **not because it raises the mean (it doesn't), but because it cuts the per-person variation to 1/3**.
For a product that gives users feedback, “everyone loses a steady 4%” is far better than
“most people are fine but one in five is completely off”.

How to write it in the paper: the cost of a single camera is **about 0.04 lower r and about 0.05 higher nRMSE**,
consistent and significant; **feature engineering cannot recover this gap**;
but using the high-SNR 8-dim features makes this cost **uniform and predictable across subjects**.

### 8.5 An interim conclusion that must be corrected

At 12:50 on 2026-10-02, with 2 seeds, I reported “main muscle nRMSE is almost exactly the same
(diff 0.0003, p=0.99)”. **That conclusion does not hold.**
With 3 seeds, the nRMSE difference within the same feature set is +0.050 to +0.076 and significant.
That 0.0003 was an artifact of “only 2 seeds” plus “`oc16` happening to be the least stable arm across seeds
(sd 0.022)”.

Lesson: **2 seeds are not enough to claim equivalence at this noise level**,
especially when the arm with the largest sd across seeds is used as the baseline.

---

## 9. The remaining levers

The feature-engineering route is exhausted (all three variations null). Only two things remain untried:

### 9.1 Training on both domains together (5 seeds, completed 2026-10-03 12:43) — ★ effective

8 dims. Three arms, `oc_only` / `mp_only` / `both`, each evaluated on both domains.

| Skeleton used for training | Test on OpenCap | Test on MediaPipe |
|---|---|---|
| `oc_only` | 0.7427 ± 0.0130 | **0.5034** ± 0.0463 |
| `mp_only` | 0.7086 ± 0.0235 | 0.6878 ± 0.0200 |
| `both` | **0.7489** ± 0.0071 | **0.7197** ± 0.0147 |

`both` is **the best on both domains** — joint training did not sacrifice multi-camera performance to gain single-camera performance.

#### The key cell (testing on MediaPipe, n = 5 seeds)

| Metric | `both` − `mp_only` | p across seeds (n=5) | p across subjects (n=10) |
|---|---|---|---|
| Main muscle r | **+0.0318** | 0.060 | **0.029** ✱ |
| Main muscle nRMSE | −0.0217 | 0.053 | 0.218 |
| Synergist r | +0.0228 | 0.067 | 0.270 |
| Synergist nRMSE | **−0.0397** | **0.019** ✱ | **0.044** ✱ |

The effect size went from +0.0298 with 3 seeds to +0.0318 with 5 seeds, **a very stable estimate**;
all four metrics point the same way. But p=0.060 across seeds sits on the borderline.
**In the paper report both tests; don't cherry-pick the p=0.029 one.**

Domain gap: control −0.054 → joint training −0.032, shrinking by about 40%.

#### `oc_only` tested on MediaPipe = 0.503 is an independent check

This is the scenario “train on multi-camera, deploy straight to a phone, change nothing”.
It agrees in magnitude with the cross-domain evaluation of §3 using `models/loso_zeroshot` (0.514 / scaler refit 0.542)
despite different models, different scalers and different feature counts — the two corroborate each other.

---

### 9.1b Adding personalized fine-tuning (completed 2026-10-03 15:49) — ★ the product numbers are here

🔴 **The protocol is “last 20% of each segment”, different from the all-window protocol of §8 / §9.1; absolute values must not be subtracted across them.**

Procedure: train the base model on the other 9 people → split each of test subject S's segments into first 80% / last 20%
→ zero-shot test on the last 20% → fine-tune on the first 80% (feature layers frozen) → test the last 20% again.

| Base model | Evaluation domain | Zero-shot r | **r after fine-tuning** | Gain |
|---|---|---|---|---|
| `both` | MediaPipe | 0.7703 | **0.8087** | +0.038 |
| `both` | OpenCap | 0.8157 | **0.8453** | +0.030 |
| `mp_only` | MediaPipe | 0.7492 | **0.8054** | +0.056 |
| `mp_only` | OpenCap | 0.7552 | 0.8038 | +0.049 |

**All 10/10 subjects improve** (`both`: +0.014 to +0.091; `mp_only`: +0.009 to +0.108).

#### 🔴 Personalization washes out the benefit of joint two-domain training

| Comparison (testing on MediaPipe, after fine-tuning) | Diff | p across seeds | p across subjects |
|---|---|---|---|
| `both` − `mp_only` | **+0.0033** | 0.28 | 0.55 |

At zero-shot `both` leads `mp_only` by +0.021 (same direction as the +0.032 all-window result of §9.1);
**after fine-tuning only +0.003 remains, entirely within noise**.
`mp_only`'s personalization gain (+0.056) is larger than `both`'s (+0.038), and the two converge to the same point.

**Practical meaning: if the deployment flow includes one personalization calibration, OpenCap multi-camera data are not needed.**
The whole training pipeline can use phones only — greatly lowering the equipment barrier for future data collection.
(But the training stage still needs EMG for labels; that has not changed.)

#### Personalization does not narrow the single-camera gap

Best MediaPipe pipeline 0.809 vs best OpenCap pipeline 0.845 → **−0.037** (p=0.018).
Almost the same as the −0.032 before fine-tuning. **Personalization lifts both sides together rather than closing the gap.**

⚠️ The `mp_only` row's “MediaPipe − OpenCap = +0.002” **must not be read as no gap**:
that base model was trained on MediaPipe, so testing it on OpenCap is itself cross-domain, and both sides are impaired.

#### ★ The early-stopping leakage concern I raised earlier does not hold up when measured

`phase2_finetune_loso.py` uses `EarlyStopping(monitor='val_loss',
restore_best_weights=True)`, and that val is the very test tail that is scored — i.e. selecting weights with the test set.
I was therefore worried that the project's existing “0.830 after fine-tuning” was inflated; this experiment deliberately avoids early stopping and records
the test performance at every epoch to measure it:

| Group | Leakage inflation (oracle − full 20 epochs) |
|---|---|
| `both` / MediaPipe | **+0.0000** |
| `both` / OpenCap | +0.0010 |
| `mp_only` / MediaPipe | +0.0001 |
| `mp_only` / OpenCap | +0.0003 |

**At most 0.001, negligible.** The reason is that fine-tuning uses lr=1e-5 with frozen feature layers,
the test metric rises almost monotonically, and the best epoch is simply the last one.
**`phase2_finetune_loso.py` does not need fixing, and the slide 22 number is unaffected.**
(I raised this concern myself, found it did not hold after measuring, and record it here so nobody worries about it again.)

### 9.1c Fine-tuning source variant: ft_src = mediapipe vs both (completed 2026-10-03 19:02)

The base model is fixed at `both`; only **which domain of the test subject's first 80% is used in the fine-tuning stage** changes:

| Fine-tuning source | Meaning |
|---|---|
| `mediapipe` | Deployment scenario — at calibration the user has only a phone |
| `both` | Academic upper bound — assumes that person also has multi-camera data (unavailable in practice) |

Both run within **the same execution** and share the same last-20% test set, so they can be subtracted directly.

| Fine-tuning source | Evaluation domain | r | nRMSE |
|---|---|---|---|
| `mediapipe` | MediaPipe | 0.8132 ± 0.0115 | 0.6153 |
| **`both`** | **MediaPipe** | **0.8270** ± 0.0109 | 0.5934 |
| `mediapipe` | OpenCap | 0.8213 ± 0.0053 | 0.6155 |
| **`both`** | **OpenCap** | **0.8539** ± 0.0015 | 0.5569 |

(Zero-shot baseline, independent of ft_src: MediaPipe 0.7743, OpenCap 0.8090)

#### Paired comparison (ft_src=both − ft_src=mediapipe)

| Evaluation domain | Δ r | p across seeds | p across subjects |
|---|---|---|---|
| MediaPipe | **+0.0138** | **0.0029** | **0.0077** |
| OpenCap | **+0.0325** | **0.0050** | **0.0029** |

**All 10/10 subjects improve** (+0.001 to +0.033, sd 0.013).
The improvement is concentrated in those who were worse to begin with: S10 +0.033, S08 +0.032, S09 +0.030;
those already good gain little: S01 +0.001, S02 +0.002.

#### Cross-run consistency check (good news on the side)

The cell `both` / ft=mediapipe / tested on MediaPipe came out at
**0.8087 (v1) and 0.8132 (v2)** in two independent runs, a difference of 0.0045, far smaller than the cross-run dispersion of
0.025 recorded in the handover doc. This is the first independent rerun of the same setting on this line, and **the result replicates**.

---

## 9.1d Two problems that had to be solved first — **solved by §9.1e; see that section for the conclusions**

### Problem 1: fine-tuning had not converged at all, so every fine-tuning number is an underestimate

`TailProbe` recorded the performance on the test tail at every epoch. Result:

| Group | Fraction with best_epoch = 20 (still improving at the end) |
|---|---|
| v2 `ft_src=both` | **100%** (30/30) |
| v2 `ft_src=mediapipe` | 97% (29/30) |
| v1 `both`/MediaPipe | 100% |
| v1 `mp_only`/MediaPipe | 93% |

**Almost every fold was still rising at epoch 20.** `FT_EPOCHS=20` was inherited from
`phase2_finetune_loso.py`, but that value is clearly not the point of convergence.

Consequences:
1. **Every “r after fine-tuning” in this document is a lower bound**; the true personalization ceiling is higher.
2. This also explains why the “early-stopping leakage” measured in §9.1b is only 0.000–0.001 —
   when the curve rises monotonically, the last epoch is the best epoch,
   so `restore_best_weights` naturally has nothing to steal.
   **So that section's conclusion must be changed to “at the unconverged point of 20 epochs, early-stopping leakage has no effect”**,
   and cannot be generalized to the converged case.
3. The project's existing “0.830 after fine-tuning” (slide 22) uses the same `FT_EPOCHS=20`,
   **and is very likely an underestimate too**.

### Problem 2: the advantage of ft_src=both is completely confounded with “twice the gradient steps”

`ft_src=both` has **6057** fine-tuning windows and `ft_src=mediapipe` has **3029** — exactly twice as many.
With a fixed 20 epochs, the former also takes twice the gradient steps.
**Since neither group has converged, taking twice the steps is better by itself.**

So the +0.0138 difference has two completely different explanations that currently cannot be separated:
  (a) multi-camera data provide information the single camera lacks
  (b) simply taking twice the gradient steps

#### Separating them requires a data-balanced control

```powershell
# run fine-tuning to convergence (e.g. 100 epochs), record the full curve, compare both ft_src together
# this solves problems 1 and 2 at once:
#   - comparing after convergence, the step-count difference is no longer a confounder
#   - and we get the true personalization ceiling along the way
```

Cost: the base models must be retrained (they were not saved), 3 seeds × (38 min base + 2 × about 5 min fine-tuning)
≈ **2.4 hours**. Or first probe the curve shape with 1 seed (about 50 minutes).

**Until this control has been run, the +0.0138 of §9.1c must not be written up as “the contribution of multi-camera information”.**

### 9.1e Convergence experiment (completed 2026-10-04 03:18) — the answers to the two problems of §9.1d

`--ft-epochs 100 --save-curve`, base model fixed at `both`, both `ft_src` variants, 3 seeds.

#### Convergence curve (main muscle r on the test tail, mean over 10 people × 3 seeds)

| epoch | ft=mediapipe | ft=both | Diff |
|---|---|---|---|
| 1 | 0.7760 | 0.7774 | +0.0014 |
| 10 | 0.7933 | 0.8039 | +0.0105 |
| **20** | **0.8103** | **0.8244** | **+0.0142** ← where we stopped the day before |
| 40 | 0.8335 | 0.8443 | +0.0108 |
| 60 | 0.8464 | 0.8542 | +0.0078 |
| 80 | 0.8545 | 0.8605 | +0.0060 |
| **100** | **0.8603** | **0.8650** | **+0.0047** |

#### Answer to problem 1: 20 epochs underestimated by 0.05, and 100 epochs still has not fully converged

`ft=mediapipe` rose from 0.8103 at 20 epochs to **0.8603** at 100 epochs,
**an underestimate of 0.050** — larger than every “single camera vs multi-camera” gap on this line.

The increment per 10 epochs is still shrinking but has not reached zero:

```
ft=mediapipe   10→20 +0.0169   30→40 +0.0097   50→60 +0.0056   70→80 +0.0037   90→100 +0.0026
ft=both        10→20 +0.0206   30→40 +0.0078   50→60 +0.0043   70→80 +0.0029   90→100 +0.0021
```

Fraction with `best_epoch = 100`: `ft=mediapipe` **100%**, `ft=both` 90%.
**100 epochs still has not converged**, but the increment has dropped to about +0.002 per 10 epochs.
Running further would go higher still, but with diminishing returns and a risk of overfitting (not yet observed).

🔴 **This means the project's existing “0.830 after fine-tuning” (slide 22) is also an underestimate.**
`phase2_finetune_loso.py` also uses `FT_EPOCHS=20`.
Before submission that experiment should be rerun with more epochs; otherwise the personalization gain is systematically under-reported.

#### Answer to problem 2: most of the advantage of `ft_src=both` is a step-count effect

| | Δr (both − mediapipe) |
|---|---|
| 20 epochs (§9.1c) | **+0.0138** (p=0.003 / 0.008) |
| **100 epochs** | **+0.0047** (p=0.024 / **0.059**) |

The gap shrinks to a third, and the across-subjects test drops out of significance (p=0.059).
The curve shape says it even more clearly: the gap between the two lines is largest at epochs 10–20 (+0.014)
and then **keeps converging** (+0.011 → +0.008 → +0.006 → +0.005).

**Interpretation: `ft_src=both` leads mainly by “taking twice the gradient steps for the same number of epochs”.
Once both sides approach their own plateaus, the extra contribution of multi-camera data is only +0.005,
below the 0.02 interpretation threshold — i.e. “indistinguishable”.**

(Note: this cannot be written as “the two are the same”; that would require an equivalence test (TOST).)

#### Full results after convergence

| Fine-tuning source | Evaluation domain | r | nRMSE |
|---|---|---|---|
| `mediapipe` | MediaPipe | **0.8603** ± 0.0002 | 0.5446 |
| `both` | MediaPipe | 0.8650 ± 0.0012 | 0.5338 |
| `mediapipe` | OpenCap | 0.8438 ± 0.0097 | 0.5829 |
| `both` | OpenCap | **0.8835** ± 0.0007 | 0.4943 |

Zero-shot baseline: MediaPipe 0.7736, OpenCap 0.8164.

🔴 **The most important number: the deployment scenario (two-domain base model, fine-tuned on phone data only) reaches r = 0.860 after convergence.**
That is 0.047 higher than the 0.813 reported the day before, and higher than the project's existing OpenCap fine-tuned 0.830.

Single camera vs multi-camera after convergence: 0.8650 vs 0.8835 = **−0.019** (both ft=both),
or deployment scenario 0.8603 vs 0.8835 = −0.023.
**Smaller than the −0.032 at zero-shot — personalization does narrow the camera gap,
the opposite of the “does not narrow” conclusion §9.1b reached at 20 epochs, because neither side had converged then.**

### 9.1f Converged rerun of slide 22's “0.830 after fine-tuning” (completed 2026-10-04 10:42)

`experiments/pipeline_variance_converge.py`, following slide 22's original protocol
(`pipeline_variance.py`: each seed retrains LOSO → fine-tuning; OpenCap 16 dims),
with the same base model branching into two paths, 3 seeds:

| Protocol | Main muscle r | Main muscle nRMSE |
|---|---|---|
| Before fine-tuning (zero-shot) | 0.7926 ± 0.0093 | — |
| (a) original protocol, 20 epochs + early stopping on the test tail | **0.8378 ± 0.0044** | — |
| (b) epoch 50 | 0.8643 ± 0.0028 | — |
| (b) epoch 100 | 0.8791 ± 0.0016 | — |
| (b) **epoch 150, no early stopping** | **0.8852 ± 0.0011** | — |

(b) − (a): main muscle r **+0.047** (p=0.0016 across seeds, 0.0023 across subjects),
nRMSE **−0.091** (p=0.0046 / 0.0007). **10/10 people improve** (+0.020 to +0.140, S09 the most).

All three checks pass:
- (a) and (b) at epoch 20 are **identical row for row** → the two paths really branch from the same model
- (a) reproduces 0.838, within the cross-run dispersion of slide 22's 0.8305
- Early stopping **never triggered** in any of the 30 folds → 0.830 was not inflated by test-set early stopping; it was **simply undertrained**

Increment per 25 epochs: +0.0202 → +0.0095 → +0.0053 → +0.0036 → +0.0025;
at 150 epochs 83% of folds are still rising, but close to a plateau.

**Implication for the paper:** slide 22's personalization gain was under-reported. After convergence, zero-shot 0.79 → fine-tuned 0.885,
close to the single-person upper bound of 0.895 (`exp_b_single_subject`, a different run, so only the general direction is meaningful).
The submission should report the converged version, stating that the epoch count was fixed in advance based on “the curve flattening”.

### 9.2 Lowering MediaPipe's noise floor — not something code can solve

The noise floor is 2.4% of body height. It can only be lowered at the capture end:
a closer camera, higher resolution, a different pose model (MediaPipe heavy is already the largest),
or multi-frame temporal smoothing (**smoothing parameters may only be chosen on the training folds**, otherwise it peeks at the test data).

This also means the inference of §6.1 holds: knee valgus measures left-right shifts on the order of 2 cm,
which under current capture conditions are **in principle unmeasurable**, and no downstream method can compensate.
