# results/ — experiment outputs

The project's results consist of 49 CSVs + 17 figures. Every file is produced by a program under [`../experiments/`](../experiments/).

> 📦 **This English repository includes only the figures.** The result CSVs indexed below are not included;
> each one is regenerated in this folder when the program that produces it is run. The index is kept so you know
> which program to run and how to read its output.

Naming conventions:
* `*_raw.csv` — per seed × per subject details
* `*_summary.csv` — aggregated mean ± sd and test results
* `*_metrics.csv` — single-run evaluation results per subject

> ⚠️ **Check the evaluation protocol of every number before citing it.** Zero-shot LOSO evaluates **all windows** of the subject,
> while fine-tuning evaluates the held-out windows from the **last 20% of each segment's time** — **the two cannot be subtracted from each other**.

---

## ⭐ The most frequently used

| File | Content | Produced by |
|---|---|---|
| `loso_zeroshot_metrics.csv` | **Per-subject zero-shot scores** (r / MAE / RMSE / nRMSE, main muscle and synergist) | `loso_train_and_save.py` |
| `finetune_results.csv` | **Before/after personalized fine-tuning**, per subject | `phase2_finetune_loso.py` |
| `exp_a_split_summary.csv` | Decomposition of split-method inflation (LOSO / random segment / random window) | `exp_a_split_compare.py` |
| `pipeline_variance_loso.csv` | Total variance of the whole pipeline | `pipeline_variance.py` |
| `stgcn_compare_summary.csv` | ST-GCN vs Conv1D | `loso_stgcn_compare.py` |

---

## 📋 Full index

### Main results

| File | Description |
|---|---|
| `loso_zeroshot_metrics.csv` | Zero-shot LOSO, per subject |
| `finetune_results.csv` | Before/after personalized fine-tuning, per subject |
| `training_history.csv` | Training curves of each LOSO fold |
| `calibration_curve.csv` | Calibration curve: how many sets must be recorded for fine-tuning to be enough |
| `two_finetuned_metrics.csv` | After personalization, predicting the later-recorded `_two` sets (closest to deployment) |
| `holdout_two_metrics.csv` / `holdout_two_history.csv` | Global model with `_two` as the validation set |

### Paper experiments

| File | Corresponding experiment |
|---|---|
| `exp_a_split_{raw,summary}.csv` | **Experiment A** split-method comparison |
| `exp_b_single_subject_{raw,summary}.csv` | **Experiment B** single-subject upper bound |
| `exp_c_ablation_{raw,summary}.csv` | **Experiment C** 16-dim vs 43-dim features |
| `exp_load_16d_*.csv` / `exp_load_43d_*.csv` | Load recognition (`_raw` per window / `_segment` per segment / `_comp` compensation segments / `_summary` aggregated) |
| `stgcn_compare_{raw,summary}.csv` | ST-GCN student network comparison (3 arms × 5 seeds) |
| `arch_compare_{raw,summary}.csv` | Conv1D vs TCN-Transformer |
| `exp_distill_{raw,summary}.csv` | **Experiment C** heterogeneous knowledge distillation (6 arms × 3 seeds) |
| `probe_teacher_privilege_{raw,summary}.csv` | Scan of the privileged teacher's `priv_frames` (basis for the setting) |
| `loso_mediapipe_*`, `loso_dualdomain_*`, `dualdomain_finetune_*`, `mediapipe_*` | Single-camera (MediaPipe) experiments; see [`../experiments/MEDIAPIPE_EXPERIMENT.md`](../experiments/MEDIAPIPE_EXPERIMENT.md) |

#### 🔴 Experiment C results (2026-08-04, n=5 seeds): **distillation has no measurable benefit**

| arm | Main muscle r | Main muscle nRMSE |
|---|---|---|
| `student_only` (Baseline) | 0.7683 ± 0.0111 | 0.7863 |
| `student_recon` | 0.7672 ± 0.0093 | 0.7853 |
| `student_hint` | 0.7696 ± 0.0076 | 0.7977 |
| `student_kd` (Ours) | 0.7736 ± 0.0134 | 0.7864 |
| *teacher_masked* (vision only) | *0.7584 ± 0.0053* | *0.7787* |
| *teacher_visible* (🚫 not citable) | *0.8397 ± 0.0047* | *0.6236* |

**Paired t-tests (n=5 seeds) — all four non-significant, all with reversals**

| Comparison | Δ main muscle r | p | dz | Per seed |
|---|---|---|---|---|
| ① `recon − only` (dense supervision) | −0.0011 | 0.84 | −0.10 | ⚠️ reversals |
| ② `hint − only` (teacher) ★ | +0.0013 | 0.87 | +0.08 | ⚠️ reversals |
| ③ `kd − recon` (teacher's marginal) | +0.0064 | 0.53 | +0.31 | ⚠️ reversals |
| ④ `kd − only` (total gain) | +0.0054 | 0.60 | +0.25 | ⚠️ reversals |

`kd − only` per seed: `+0.0164  +0.0194  −0.0317  +0.0152  +0.0075`

> ⚠️ **This result is a live lesson in “why multiple seeds are necessary”.**
> At n=3 (seeds 42/1/2) it showed **+0.0144, same sign for all three seeds, dz=2.34, p=0.056**,
> seemingly just short of significance. **After adding seed 3 it reversed outright** (−0.0317), and at n=5 it converged to +0.0054, p=0.60.
> Stopping at n=3 and claiming “the trend supports distillation” would have been wrong.

**★ But the precondition holds — and that is the truly valuable finding:**

```
teacher privileged mode  0.8397 ± 0.0047
teacher vision only      0.7584 ± 0.0053     diff +0.081, consistent across all 5 seeds
student baseline         0.7683 ± 0.0111
```

The teacher **really does hold information the student lacks** (+0.081, extremely stable across seeds), yet **none of it transfers**.
The content of the privileged signal is “this subject's EMG amplitude and gain” —
**a quantity skeleton kinematics in principle cannot observe**; it is not a matter of model capacity or training tricks.

📌 This directly quantifies the ceiling of the whole “remove the electrodes” route, echoing the leakage decomposition of Experiment A.

#### Notes on reading `exp_distill_*`

The `arm` column has six values, which **must not be treated alike**:

| arm | Citable? |
|---|---|
| `teacher_visible` | 🚫 **Never cite it as a score** — its input contains the true EMG; it is only used to confirm the privileged channel works |
| `teacher_masked` | ⚠️ The teacher's vision-only ability; usable as a reference, but it is not the model to be deployed |
| `student_only` | ✅ Baseline |
| `student_recon` / `student_hint` / `student_kd` | ✅ Three loss combinations, used to decompose where the gain comes from |

`deployed_params` is the parameter count actually needed at deployment (29,734 for all four student arms);
`training_params` includes the training-time projection layers and reconstruction head, which are not loaded at inference. **Report the former in the paper.**

### Variance and stability

| File | Description |
|---|---|
| `pipeline_variance_loso.csv` | Whole pipeline, LOSO stage |
| `pipeline_variance_finetune.csv` | Whole pipeline, fine-tuning stage |
| `pipeline_variance_segments{,_summary}.csv` | **Per-segment residuals**, for cross-checking suspicious segments |
| `phase2_multiseed_{raw,summary}.csv` | Multi-seed stability of the fine-tuning stage (sd ≤0.0002) |
| `loso_multiseed_{raw,summary}.csv` | Multi-seed comparison before vs after the coordinate-axis fix |
| `finetune_gap_{raw,summary}.csv` | Diagnosis of the 0.832 vs 0.814 gap |

### Data quality diagnostics

| File | Description |
|---|---|
| `axis_diagnosis.csv` | Coordinate-axis diagnosis (all 89 TRCs) |
| `axisfix_verify.csv` | Verification that the coordinate-axis fix took effect (only 4 of 51 columns changed) |
| `axisfix_loso_comparison.csv` | LOSO comparison before vs after the fix |
| `event_alignment_report.csv` | Event-alignment health check, all 89 segments |
| `synthetic_ablation.csv` | Synthetic ablation (verifies r is immune to amplitude) |

### Control groups

| File | Description |
|---|---|
| `loso_masked_metrics.csv` / `finetune_masked_results.csv` | Training with the S09 synergist masked |
| `loso_9subj_metrics.csv` / `finetune_9subj_results.csv` | 9-person version excluding S09 entirely |
| `loso_shoulderx_{raw,summary}.csv` | Adding Shoulder_X (17 dims) — experiments showed **no benefit; not adopted** |

### Historical versions (**do not cite**; for reference only)

| File | Description |
|---|---|
| `loso_zeroshot_metrics_before_axisfix.csv` | Before the coordinate-axis fix |
| `finetune_results_before_axisfix.csv` | Before the coordinate-axis fix |
| `loso_zeroshot_metrics_20250730_stale.csv` | Outdated intermediate version |

---

## 🖼️ Figures

The figures in this repository were redrawn with English labels from the saved models (personalized fine-tuning is redone in memory,
so values printed on a figure can differ from the original run in the third decimal place).
Figures marked “not included” need a full training run to redraw; run the listed program to produce them.

| File | Content | In this repo |
|---|---|---|
| `zs_vs_ft_summary.png` | Summary comparison of zero-shot vs fine-tuned | ✅ |
| `zs_vs_ft_waveform_{main,syn}.png` | Same, as waveforms (main muscle / synergist) | ✅ |
| `selected_{scatter,waveform_main,waveform_syn}.png` | Selected subjects (S01/S09/S10) | ✅ |
| `slide_{waveform,loso,calibration}.png` | For slides | ✅ |
| `synthetic_ablation.png` | Synthetic ablation | ✅ |
| `learning_curve_subjects.png` | Learning curve over the number of training subjects | ✅ |
| `holdout_two_{scatter,training_curves,waveform_main,waveform_syn}.png` | The `_two` holdout group | not included — `train_holdout_two.py`, `plot_holdout_two_curves.py` |
| `two_finetuned_waveform_{main,syn}.png` | After personalization, predicting `_two` | not included — `finetune_predict_two.py` |
| `calibration_curve.png` | Calibration curve | not included — `calibration_curve.py` |
| `conf_waveform_demo.png` | Waveform demo for a conference talk | not included |

> ⚠️ Several early figures plotted the 0–1.5 ratio of `EMG_*_MVC` **directly as a percentage** (missing the ×100).
> 5 figures have been fixed; **keep this in mind when producing new figures**.
> ⚠️ Also, `EMG_*_MVC` itself is **not a true %MVC** (this project has no MVC recordings);
> it is normalized to the maximum activation observed for that subject, and ×100 only changes the scale, not the meaning.
> Figure axes / captions must not be labelled “%MVC”; see item 6 of the [root README](../README.md#-read-before-citing-numbers) for details.

---

## `_cache/`

Temporary files from runs (`.log`, base64 image JSON), **gitignored**, so they never appear in a cloned repo.
