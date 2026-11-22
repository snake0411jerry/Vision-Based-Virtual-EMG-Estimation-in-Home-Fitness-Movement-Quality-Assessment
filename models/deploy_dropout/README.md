# deploy_dropout — load-dropout model for the app

**In one sentence:** give it 40 frames (60 fps, about 0.67 s) of phone-skeleton features, the set's load (optional) and the
user's basic info, and it returns the activation envelope of the main muscle and the synergist.

Produced by [`experiments/train_deploy_dropout.py`](../../experiments/train_deploy_dropout.py).

## Files

| File | Purpose |
|---|---|
| `dropout_conv8_load.keras` | The model (Keras, TF 2.10) |
| `dropout_conv8_load.tflite` | Same model as TFLite for on-device use. Inputs `serving_default_ts_input:0` (1, 40, 11) and `serving_default_static_input:0` (1, 4); output (1, 2); one window per call. Verified against the Keras model (max difference < 1e-6) |
| `scaler_ts.pkl`, `scaler_static.pkl` | sklearn StandardScaler (for Python) |
| `deploy_config.json` | Column order, scaler mean/scale and load formulas — **a non-Python app can preprocess from this file alone** |

## How accurate is it

Leave-one-subject-out results: the test subject was never trained on, the skeleton comes from a single phone (MediaPipe),
3 random seeds × 10 subjects.

| Case | Main-muscle r |
|---|---|
| Load entered | **0.785** |
| Load not entered | 0.727 |
| (reference) skeleton-only model | 0.704 |

This deployment model is trained on all data (OpenCap 20 subjects + MediaPipe 10 subjects), so it **has seen every subject
and must not be used to compute scores**. Quote the table above.

## Inputs

### 1. `ts_input`: shape (N, 40, 11)

11 numbers per frame, in this fixed order:

| # | Column | Source |
|---|---|---|
| 0–7 | `Knee_Angle_norm`, `Ankle_Y_norm`, `Knee_Y_norm`, `Toe_Y_norm`, `Knee_X_norm`, `Shoulder_Y_norm`, `Knee_Ankle_Ratio_norm`, `Trunk_Lean_Angle_norm` | Computed from the skeleton with `pipeline/Features_insert_FIXED.py` (MediaPipe is first converted to OpenCap TRC format, see `pipeline/build_combined_mediapipe.py`) |
| 8 | `Load_pct1RM` | total set load in kg (**including the 20 kg bar**) ÷ 1RM in kg; bodyweight squat = 0 |
| 9 | `Load_per_BW` | total set load in kg (including the bar) ÷ body weight in kg; bodyweight squat = 0 |
| 10 | missing flag | load entered = 0, not entered = 1 |

- **Standardization:** apply `(x − mean) / scale` to columns 0–9, using `deploy_config.json` → `inputs.ts_input`. Do not standardize the flag in column 10.
- **Load not entered:** standardize as usual, then set the **standardized** columns 8 and 9 to 0 and column 10 to 1.
- **1RM unknown:** estimate it as "the heaviest set of about 10 reps to failure ÷ 0.7", the same estimate used for the training data. A ±30% error in 1RM barely changes the accuracy.

### 2. `static_input`: shape (N, 4)

Age ÷ 100, height in cm ÷ 100, body weight in kg ÷ 100, sex (male = 1, female = 0), standardized with `inputs.static_input`.

### 3. Windows

- The skeleton must be **60 fps**; resample it first if it is not.
- Cut a 40-frame window every 5 frames (about 0.083 s).
- Each window returns one value, the activation at the window's **last frame**.

## Output

Shape (N, 2): `[main muscle, synergist]`, an activation envelope normalized to the subject's own maximum (roughly 0 to 1; the last layer is a sigmoid).

- **The main muscle is the reliable one** (r about 0.79). The synergist is only about 0.45, so show it as a rough reference at most.
- **It predicts the shape of the effort**: which rep was harder and which set was heavier. Absolute levels are not comparable between people, since they depend on things like subcutaneous fat.

## Notes

- `ref_height` in the features is the 95th percentile of Shoulder_Y − Ankle_Y over the **whole set**, so the simplest approach is to process a set after it is recorded. For a real-time version, estimate ref_height from the first few reps.
- The training data contain squats only: standard squats, 3 compensation patterns, and bodyweight, 20 kg empty bar and 2 heavier back-squat loads. Other movements are out of scope.
