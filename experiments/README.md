# experiments/ — paper experiments and diagnostics

Only `.py` files live here. All outputs (CSVs, figures) are written to [`../results/`](../results/);
the path is set by `RESULTS_DIR` in [`paths.py`](../paths.py).

> 📦 This English repository does not include the result CSVs listed below; running a program regenerates its outputs in `results/`.

Before running, make sure the dataset is in place (see the [root README](../README.md#3-dataset-path-only-needed-to-rerun-the-pipeline)).
On Windows, always set `PYTHONIOENCODING=utf-8`.

---

## 🧪 Main paper experiments

| Program | Question answered | Conclusion | Output |
|---|---|---|---|
| **`loso_train_and_save.py`** | Cross-subject zero-shot ability? | Main muscle r 0.735 | `loso_zeroshot_metrics.csv` + `models/loso_zeroshot/` |
| **`phase2_finetune_loso.py`** | How much does personalized fine-tuning add? | +0.033 (leak-free protocol) | `finetune_results.csv` + `models/personalized/from_loso/` |
| **`exp_a_split_compare.py`** | **Experiment A**: how much does the split method matter? | Random splitting inflates by **+0.164 / +0.302** | `exp_a_split_{raw,summary}.csv` |
| **`exp_b_single_subject.py`** | **Experiment B**: upper bound of single-person modelling? | Main muscle 0.895, synergist 0.682 | `exp_b_single_subject_{raw,summary}.csv` |
| **`loso_multiseed.py`** | **Experiment C** and any other two-arm comparison | General tool, see below | `{prefix}_{raw,summary}.csv` |
| **`exp_load_regression.py`** | Can load (%1RM) be recognized? | Within-subject r 0.590→0.753 (with v/a) | `exp_load_{16d,43d}_*.csv` |
| **`loso_stgcn_compare.py`** | Does the ST-GCN student network help? | Main muscle 0.729→**0.768** (p=0.0098) | `stgcn_compare_{raw,summary}.csv` |
| **`loso_arch_compare.py`** | Conv1D vs TCN-Transformer? | No difference, but 4.8× the parameters | `arch_compare_{raw,summary}.csv` |
| **`exp_distill_compare.py`** | **Experiment C**: does heterogeneous knowledge distillation help? | **No measurable benefit** (+0.005, p=0.60, some seeds reversed at n=5); see below | `exp_distill_{raw,summary}.csv` |
| `probe_teacher_privilege.py` | What should the privileged teacher's `priv_frames` be? | **39** (with other settings the privileged channel might as well not exist) | `probe_teacher_privilege_*.csv` |
| `compare_mediapipe_features.py` and related | Single phone camera (MediaPipe) vs OpenCap | See [`MEDIAPIPE_EXPERIMENT.md`](MEDIAPIPE_EXPERIMENT.md) | `loso_mediapipe_*.csv` etc. |

### `exp_distill_compare.py` — heterogeneous knowledge distillation (Experiment C)

Corresponds to step 2, "heterogeneous teacher-student network", and ablation experiment C in step 4 of the NSTC proposal.
Related files: [`models_teacher.py`](models_teacher.py) (teacher), [`distill_utils.py`](distill_utils.py) (projection layers and losses).

**Teacher = privileged-information model.** During training it sees the true sEMG, while the student only sees the skeleton — exactly
the gap that "removing the electrodes" has to bridge. Two design choices are essential, otherwise the whole thing degenerates:

| Mechanism | What happens without it |
|---|---|
| **Drop the last frame of the window inside the model** (`priv_drop_target`) | The last frame is the prediction target; if it enters the input, the teacher just copies out the answer |
| **Randomly mask 50% of the EMG input** | The teacher takes only the EMG shortcut, its latent representation loses its grounding in movement, and the student has nothing to imitate |

**Four student arms are used to decompose where the gain comes from** — running only baseline vs ours,
even a win could not tell whether the credit belongs to the "teacher" or the "reconstruction loss", and the paper claims the former:

| arm | Loss | Meaning |
|---|---|---|
| `student_only` | task | The proposal's Baseline |
| `student_recon` | + reconstruction | Dense supervision, **no teacher information** |
| `student_hint` | + hint + soft labels | **Only** teacher information ← ★ what the paper wants to claim |
| `student_kd` | all three | The proposal's Ours |

```
recon − only   contribution of dense supervision
hint  − only   contribution of the teacher        ★
kd    − recon  marginal contribution of the teacher on top of dense supervision
```

**Check the precondition diagnostic before running.** At the end of each seed it prints `teacher_visible` vs `teacher_masked`:
if the gap is small, the teacher is not using the privileged channel at all, so its latent representation contains nothing the student cannot learn —
**distillation is then bound to fail; fix the teacher before running the real experiment**.

#### 🔴 Results (n=5): all four contrasts non-significant, all with reversals

```
recon − only  -0.0011  p=0.84      hint − only  +0.0013  p=0.87
kd    − recon +0.0064  p=0.53      kd   − only  +0.0054  p=0.60
```

**But the precondition holds**: privileged teacher 0.8397 vs vision-only 0.7584 (+0.081, consistent across all 5 seeds).
The teacher really does hold information the student lacks, yet **none of it transfers** — because that information is
"this subject's EMG amplitude and gain", a quantity skeleton kinematics **in principle cannot observe**.
For the full numbers and interpretation see [`../results/README.md`](../results/README.md).

⚠️ At n=3 the result was +0.0144, same sign for all three seeds, p=0.056, seemingly just short of significance;
**after adding a 4th seed it reversed outright**. A live lesson in why multiple seeds are necessary.

#### Why `priv_frames` is 39 (don't change it casually)

Scanned by `probe_teacher_privilege.py` (mean of 2 folds, details in `results/probe_teacher_privilege_*.csv`):

| `priv_frames` | Gap to target | Vision-only r | Privileged r | Gain |
|---|---|---|---|---|
| 25 | 0.25 s | 0.763 | 0.769 | +0.006 ❌ |
| 30 | 0.17 s | 0.758 | 0.759 | +0.001 ❌ |
| 35 | 0.08 s | 0.779 | 0.785 | +0.006 ❌ |
| **39** | **0.02 s** | 0.776 | **0.847** | **+0.071** ✅ |

**It is a cliff, not a slope.** After the 5Hz low-pass the EMG envelope is very smooth, so "recent EMG" either **equals the answer**
or **carries no information**; there is no usable middle ground.

⚠️ Another detail that must be kept: the EMG branch **only pools** (GAP + GMP); never add a readout head such as "flatten the last N frames".
When one was added, the teacher's r shot up to **0.993** (⚠️ this number was measured **before** the structural guard
`Lambda(z[:, :priv_frames, :])` was added; under the current implementation `priv_frames=39`
was measured independently twice at around 0.84, with no degeneration — see the header of `probe_teacher_privilege.py`)
— it copied the answer directly, the latent representation lost its meaning,
and distillation went to zero (hint arm 0.759 vs baseline 0.763). Pooling throws away the endpoint information,
so the model only gets "the activation level of the whole window", which is exactly what we want it to learn.

⚠️ `teacher_visible` can see the true EMG, so it is **not a fair comparison**; it is used only for this diagnostic.
⚠️ The projection layers and reconstruction head are training-time scaffolding and are not used at inference. The four student arms have **exactly the same deployed parameter count**
(29,734); the CSV records both `deployed_params` and `training_params`.

### `loso_multiseed.py` — general tool for any comparison

**Any claim of "changing X improved things by this much" should be verified with it**, because a single run cannot resolve differences of order 0.02.

```bash
# Experiment C: 16-dim vs 43-dim features
python loso_multiseed.py --arms spatial=Combined@spatial kinematic=Combined@spatial+kinematic \
                         --seeds 42 1 2 3 4 --prefix exp_c_ablation

# before vs after the coordinate-axis fix
python loso_multiseed.py --arms before=Combined_before_axisfix after=Combined \
                         --seeds 42 1 2 3 4 --prefix loso_multiseed
```

⚠️ **Both arms must run inside the same job** — numbers from different runs cannot be subtracted directly (see rule 5 in the root README).

---

## 🔬 Stability and variance estimation

| Program | What it measures | Output |
|---|---|---|
| `pipeline_variance.py` | **Total variance of the whole pipeline** (each seed reruns LOSO→fine-tuning from scratch) + per-segment residuals | `pipeline_variance_{loso,finetune,segments,segments_summary}.csv` |
| `phase2_finetune_multiseed.py` | Noise of the fine-tuning stage only (base not retrained, models not overwritten), sd ≤0.0002 | `phase2_multiseed_{raw,summary}.csv` |
| `diagnose_finetune_gap.py` | Diagnoses the source of the 0.832 vs 0.814 gap | `finetune_gap_{raw,summary}.csv` |
| `synthetic_ablation.py` | Synthetic ablation, verifies that r is immune to amplitude | `synthetic_ablation.csv` + `.png` |

> 📌 `pipeline_variance.py` takes about 25 min/seed and is currently the only variance estimate covering the whole pipeline.
> The sd from `phase2_finetune_multiseed.py` **underestimates** the total variance (the base is fixed, not retrained);
> when the paper reports absolute values after fine-tuning, the uncertainty should be taken from the base model's sd (±0.013–0.018).

---

## 🩺 Data quality diagnostics

| Program | Purpose | Output |
|---|---|---|
| `check_axis_convention.py` | Coordinate-axis diagnosis (precondition check before the fix) | `axis_diagnosis.csv` |
| `verify_axisfix.py` | Verifies the coordinate-axis fix actually took effect | `axisfix_verify.csv` |
| `check_event_alignment.py` | **Event-alignment health check**: TRC squat count/timing vs EMG burst count/timing | `event_alignment_report.csv` |
| `compare_loso.py` | Old vs new single-seed LOSO comparison (superseded by `loso_multiseed.py`) | printed directly |
| `calibration_curve.py` | Calibration curve: **how many sets must be recorded for fine-tuning to be enough** | `calibration_curve.csv` + `.png` |

> ⚠️ The peak detection in `check_event_alignment.py` is itself noisy; **per-segment numbers must not be used as evidence on their own**.
> Its purpose is to flag suspicious segments for manual review. Before comparing, always filter out duplicate detections less than 1.5s apart (bottom bounce),
> and make sure both sides have the same number of events — a count mismatch shifts everything by one beat and produces a fake 2–4 second misalignment.

---

## 🧬 Control-group training

| Program | What it produces |
|---|---|
| `loso_masked_synergist.py` | Training with the S09 synergist masked → `models/ablation/masked_synergist/` |
| `loso_exclude_s09.py` | 9-person LOSO excluding S09 entirely → `models/ablation/exclude_s09/` |
| `train_holdout_two.py` | Global model with `_two` as the validation set → `models/ablation/holdout_two/` |
| `finetune_predict_two.py` | **Closest to deployment**: after personalization, predict the later-recorded `_two` sets → `models/personalized/from_loso_nontwo/` |

---

## 🏗️ Model architecture definitions

| File | Description |
|---|---|
| `models_tcn_transformer.py` | TCN residual stack + Pre-LN Transformer (107,554 parameters). Same interface as `build_model`, can be swapped in directly |
| `models_stgcn.py` | ST-GCN student network (distance partitioning + edge importance), 20 nodes / 19 edges |

Both can be run directly to print the parameter count of each configuration:

```bash
python models_stgcn.py
```

> ST-GCN needs `Dataset/Skeleton/*.npz`, produced by [`pipeline/build_skeleton_dataset.py`](../pipeline/build_skeleton_dataset.py).

---

## 📈 Plotting

| Program | Figures produced |
|---|---|
| `plot_zeroshot_vs_finetuned.py` | Zero-shot vs fine-tuned waveforms (also imported by other programs) |
| `plot_selected_subjects.py` | Waveforms of selected subjects (currently S01/S09/S10) |
| `plot_training_curves.py` | Training curves for LOSO + fine-tuning |
| `plot_holdout_two_curves.py` | Training curves for the `_two` holdout |
| `make_slide_figures.py` | Three figures for slides (`slide_*.png`) |

> ⚠️ `EMG_*_MVC` stores a ratio from 0 to 1.5; multiplying by 100 only changes the scale to 0–150,
> it is **not a true %MVC** (this project has no MVC recordings), so plots / captions must not be labelled "%MVC".
> See item 6 of the [root README](../README.md#-read-before-citing-numbers) for details.

---

## ⚠️ Read before running experiments

1. **Control groups must run in the same job.** cuDNN kernel nondeterminism means even a fixed seed is not bit-reproducible;
   the real noise scale is "between runs", not "between seeds". Numbers from different sessions cannot be subtracted directly.
2. **A single run cannot resolve differences of order 0.02** (sd across seeds 0.015–0.018). Use at least 5 seeds.
3. **Always split by subject**; sliding windows must not cross segment boundaries.
4. Inference is **fully reproducible** — given a set of saved models, any evaluation number can be reproduced reliably.
   Uncertainty comes only from nondeterminism during training.
