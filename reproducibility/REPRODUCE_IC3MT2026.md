# IC3MT 2026 reproducibility freeze point (tag `ic3mt-2026`)

This document ties **every number in the IC3MT 2026 oral presentation** to “which program, which CSV, which row”.
When someone later asks “where did this 0.766 come from”, the answer is here, with no need to dig through chat logs or guess.

Created: 2026-09-07  |  Slides: `簡報與逐字稿/0823研討會.pptx` (48 pages)
Paper: `大專生計畫與IC3MT/035S_IC3MT2026.pdf`

> 📦 **Note for this English repository.** It was created with a fresh git history, so it does not carry the `ic3mt-2026` tag,
> and, like every CSV, the data manifest and result CSVs referenced below are not included. The tag, manifest and result files
> live in the project's original repository. This document is kept so the provenance of every reported number stays traceable.

---

## 1. What this tag freezes

| Item | How it is frozen |
|---|---|
| Code | git tag `ic3mt-2026` of `Code/` |
| Model weights and scalers | same (193 files in `models/` are version-controlled) |
| Experiment result CSVs | same (88 files in `results/`) |
| **Dataset** | **`reproducibility/DATA_MANIFEST_ic3mt-2026.csv`** — SHA-256 of 310 files |
| Public-release code | git tag `ic3mt-2026` of `public_release/` |

`Dataset/` (962 MB) is not under version control, so the tag itself cannot cover it. If the data are regenerated or overwritten,
the checked-out code still runs but the numbers silently stop matching. The manifest exists to make that **audible**:

```bash
cd Code
python tools/make_data_manifest.py --check reproducibility/DATA_MANIFEST_ic3mt-2026.csv
```

All paths in the manifest use `S01`–`S10`; nicknames are replaced when it is written (the mapping table `pipeline/subjects.csv`
is gitignored). So the manifest itself **contains no personal data** and can safely be version-controlled.

## 2. Environment

```
Python   3.10.19        (conda env: pose_ai)
TensorFlow 2.10.0
numpy    1.26.4
scipy    1.15.3
scikit-learn 1.7.2
GPU      RTX 5060
```

⚠️ `set_random_seed` **is not fully reproducible** on this machine's GPU (see the handover doc). Rerunning the same seed
gives small differences, of roughly the same size as the sd across seeds (main muscle r ±0.015).
So the standard for “reproduced” is **landing in the same interval**, not bit-for-bit identity.

## 3. Reported numbers → sources

The mean/sd below are computed fresh from the CSVs (5 seeds: 1, 2, 3, 4, 42) and placed side by side with the slides.

### Main result (slide 21)

| Quantity | Slide says | Actual CSV | Source |
|---|---|---|---|
| Conv1D baseline main muscle r | 0.729 | **0.7288 ± 0.0172** | `results/stgcn_compare_summary.csv` arm=`baseline` |
| ST-GCN main muscle r | 0.766 | **0.7676 ± 0.0081** (`stgcn_posva`) | same |
| Paired t-test p | 0.0098 | **0.0098** (d=+0.0389, t=4.625, n=5) | difference of the two above |
| baseline nRMSE | 0.812 | **0.8122 ± 0.0165** | same |
| ST-GCN nRMSE | 0.790 | 0.7777 (`stgcn_posva`) / **0.7902** (see below) | — |

> 🔴 **The 0.766 / 0.790 on slide 21 actually mix two runs.**
> In the `stgcn_compare` run, ST-GCN is 0.7676 / 0.7777;
> the slide's 0.766 / 0.7902 match the `teacher` arm of `exp_distill_compress_summary.csv`
> (0.7658 / 0.7902) — the same numbers used for “before compression” on slide 26.
> **Both versions lead to the same conclusion** (Δr is about +0.039 either way; p=0.0098 comes from `stgcn_compare`),
> but the handover doc's “methodology rules” state explicitly that numbers from different runs must not be subtracted.
> **For a journal submission, consistently use the `stgcn_compare` run**, and tie the p-value and the means to the same run.

Program:
```bash
python experiments/loso_stgcn_compare.py --seeds 42 1 2 3 4
```

### Personalized fine-tuning (slide 22)

| Stage | Slide | CSV | Source |
|---|---|---|---|
| Zero-shot | 0.766 | 0.7885 ± 0.0070 | `r_main_before` in `results/pv_mvcfix_finetune.csv` |
| After fine-tuning | 0.83 | **0.8305 ± 0.0048** | `r_main_after` in `results/pv_mvcfix_finetune.csv` |
| Single-person upper bound | 0.895 | **0.8951 ± 0.0021** | `results/exp_b_single_subject_summary.csv` |

> 🔴 **Correction of 2026-10-04**: this table originally said “0.830 ± 0.006 after fine-tuning comes from
> `phase2_multiseed_summary.csv`” — **that was wrong**. Since 2026-08-04
> (including when this tag was made) that file has always shown **0.814**, because it uses the fixed `models/loso_zeroshot`
> weights (see §4 item 3: a low draw).
> The correct source is `pv_mvcfix_finetune.csv` (`experiments/pipeline_variance.py`,
> **each seed retrains the 10 LOSO folds and then fine-tunes**, 5 seeds = 42/1/2/3/4).
> That is the total variance of the whole pipeline, and the number that matches 0.83.
> The sibling file `pipeline_variance_finetune.csv` (before the MVC fix) gives 0.8320 ± 0.0037.
>
> 🔴 **This 0.830 is an underestimate**: that protocol uses `FT_EPOCHS=20`. On 2026-10-03 the MediaPipe
> experiments found 20 epochs is far from converged (a similar setting run to 100 epochs is about 0.05 higher),
> and the early-stopping val is the test tail itself. For the converged rerun see
> `experiments/pipeline_variance_converge.py` and `MEDIAPIPE_EXPERIMENT.md`.
>
> **Converged rerun results (2026-10-04, `results/pv_converge_raw.csv`, 3 seeds):**
>
> | Protocol | Main muscle r |
> |---|---|
> | Before fine-tuning (zero-shot) | 0.7926 ± 0.0093 |
> | Original protocol, 20 epochs + early stopping on the test tail | 0.8378 ± 0.0044 (reproduces 0.83) |
> | **150 epochs, no early stopping** | **0.8852 ± 0.0011** |
>
> Difference **+0.047** (paired p=0.0016 across seeds / 0.0023 across subjects), nRMSE −0.091, 10/10 people improve.
> Early stopping **never triggered** in any of the 30 folds, so 0.830 was not inflated by early stopping; it was **undertrained**.
> At 150 epochs 83% of folds are still rising (about +0.0025 per 25 epochs), close to a plateau.
> **For submission, personalized fine-tuning should be reported in its converged version**, stating that the epoch count was fixed in advance based on “the curve flattening”, not chosen by looking at the test set.

⚠️ The evaluation set for these three cells is “the last 20% of each segment”, **a different protocol from slide 21's all-window evaluation; they cannot be subtracted**.
The slides use slide 21's 0.766 as the starting point here, which strictly speaking switches protocols;
Experiment B's own zero-shot value is 0.790. Pick one and state it clearly in the submission.

### Ablation ①: feature dimensionality (slide 23)

| Metric | 16 dims | 43 dims | p |
|---|---|---|---|
| Main muscle r | 0.7397 ± 0.0104 | 0.7396 ± 0.0103 | 0.989 |

Source: `results/exp_c_ablation_summary.csv`
```bash
python experiments/loso_multiseed.py --arms spatial=Combined@spatial kinematic=Combined@spatial+kinematic --seeds 42 1 2 3 4
```

### Ablation ③: task dependence (slide 25)

| Metric | 16 dims | 43 dims | p |
|---|---|---|---|
| Within-subject r (load recognition) | 0.590 ± 0.120 | **0.753 ± 0.041** | **0.015** |

Source: `results/exp_load_16d_summary.csv`, `exp_load_43d_summary.csv`
```bash
python experiments/exp_load_regression.py
```

### Compression (slide 26)

| Metric | Before compression: teacher | After compression: small_only | p |
|---|---|---|---|
| Parameters | 29,734 | 10,070 | 2.95× |
| Main muscle r | **0.7658 ± 0.0056** | **0.7654 ± 0.0061** | **0.9165** |
| Main muscle nRMSE | **0.7902 ± 0.0293** | **0.8187 ± 0.0427** | **0.0500** |

Source: `results/exp_distill_compress_summary.csv`

> ⚠️ p = 0.92 is **not** evidence that “the two are the same”, only that “there is no evidence they differ”.
> Claiming equivalence requires an equivalence test (TOST). This must be added before submission (second-tier to-do).

### Distillation (slide 27)

| Teacher design | Δr | p | Source |
|---|---|---|---|
| ① privileged EMG teacher | +0.0054 | 0.60 | `results/exp_distill_summary.csv` |
| ② enlarged same-architecture teacher | −0.0003 | 0.87 | `results/exp_distill_arch_summary.csv` |
| ③ heterogeneous Transformer teacher | +0.0052 | 0.44 | same |

### External comparison: GBT (slide 28)

| | Slide says | Handover doc says | Committed result file |
|---|---|---|---|
| GBT main muscle r | 0.775 | 0.776 | **none** |
| GBT nRMSE | 0.722 | 0.745 | **none** |

> 🔴 **This is the one hole in this freeze point that cannot be filled.**
> The teammate's audit folder (outside this repo) has the program `baseline_gbt/baseline_gbt.py`, but **its output was never version-controlled**,
> and the nRMSE on the slides (0.722) and in the handover doc (0.745) disagree — by 0.023,
> above the 0.02 interpretation threshold, which affects the claim of “how much GBT wins on nRMSE”.
> **Next step**: rerun it once, put the CSV into `results/`, then update this section and the handover doc.
> Until then, the paper should not cite GBT's absolute nRMSE.

## 4. Known reproducibility limits

1. **GPU nondeterminism** — rerunning the same seed differs by about ±0.015 (see §2).
2. **Cross-run drift** — for the same 16-dim LOSO, three independent runs gave 0.740 / 0.729 / 0.723;
   the dispersion across runs (≈0.025) is larger than the within-run sd across seeds (0.013–0.017).
   **Numbers from different runs cannot be subtracted**; to compare, run both in the same job. nRMSE is much more stable than r.
3. **`models/` is a low draw** — the existing weights give r=0.814 after fine-tuning, the lowest of 9 observations
   (−2.56 sd). Figures made by loading them directly are conservative by about 0.016.
4. **GBT: see above**.

## 5. How to return to this state

```bash
git -C Code checkout ic3mt-2026
git -C public_release checkout ic3mt-2026
cd Code && python tools/make_data_manifest.py --check reproducibility/DATA_MANIFEST_ic3mt-2026.csv
```

Passing the manifest check = code, models, results and data are all back to their state on the day of the presentation.
