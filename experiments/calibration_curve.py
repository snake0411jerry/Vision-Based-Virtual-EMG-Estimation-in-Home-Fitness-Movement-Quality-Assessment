"""
calibration_curve.py — calibration curve: how much personal data is worth collecting?
=========================================================================
[Rationale]
-------------------------------------------------------------------------
Motivation: personalized fine-tuning requires a new user to put on electrodes and record EMG first.
            That is a hassle, so the practical question is not "does fine-tuning help" but
            "**how many sets** must be recorded? After which set is there no marginal benefit?"

The approach treats the amount of personal calibration data as the independent variable and
plots performance against it:

    k = 0  →  no calibration at all (zero-shot, LOSO model as-is)
    k = 1  →  fine-tune on this subject's first set only
    k = 2  →  fine-tune on the first 2 sets
    ...
    k = N  →  fine-tune on all available sets

For every k, fine-tuning restarts from **the same zero-shot model** (not continuing from the
previous k, which would silently accumulate more gradient updates and distort the curve), and
evaluation uses a test set that is **completely fixed and never takes part in any k's
fine-tuning**.

[Key leakage-prevention design]
-------------------------------------------------------------------------
1. Starting model = loso_without_{S}.keras, trained with subject S fully excluded.
2. Test set = that subject's "last N_TEST sets"; these never enter the calibration pool for any k.
   (The temporal tail is not used as the test set because, if the calibration pool contained the
    early part of the same set, it would become "same set, different time" rather than "a brand-new
    set", overestimating the calibration benefit.)
3. Calibration pool = the remaining sets, taking k sets in Segment order (≈ recording order),
   simulating the real scenario "the user comes in, records set 1, then set 2, ...".
4. Each k is repeated N_REPEAT times (different weight initialization and data order) and averaged,
   reducing the jaggedness caused by single-run training randomness.

[How to read the plot]
-------------------------------------------------------------------------
- Curve flattens after k=k* → more than k* sets brings no marginal benefit; record k* sets in practice.
- Curve almost flat (k=0 already near the top) → zero-shot is enough, personalization unnecessary
  (good news for deployment).
- Curve keeps rising up to k=N → not enough data yet; worth recording more.
- Look at r and nRMSE together: r may saturate early while nRMSE (which includes systematic bias)
  often keeps falling — meaning calibration mainly corrects the individual's gain/offset rather than
  improving the temporal shape.

Output: calibration_curve.csv + calibration_curve.png
"""
# --- Paths: always managed centrally by paths.py at the repo root; do not revert to absolute paths ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import (  # noqa: E402,F401
    REPO_ROOT, PIPELINE_DIR, EXPERIMENTS_DIR, RESULTS_DIR,
    DATASET_DIR, RAW_DIR, CLEAN_DIR, CLEAN_60FPS_DIR, CLEAN_1000HZ_DIR,
    COMBINED_DIR, COMBINED_BEFORE_AXISFIX_DIR, OPEN_DIR, SKELETON_DIR,
    GLOBAL_MODEL_DIR, LOSO_MODEL_DIR,
    PERSONALIZED_FROM_LOSO_DIR, PERSONALIZED_NONTWO_DIR, PERSONALIZED_FROM_GLOBAL_DIR,
    ABL_BEFORE_AXISFIX_DIR, ABL_MASKED_DIR, ABL_EXCLUDE_S09_DIR, ABL_HOLDOUT_TWO_DIR,
    SUBJECTS_CSV, SUBJECTS_EXAMPLE_CSV,
    ensure_results_dir, require_dataset, require_subjects_csv,
)
# --- End of path setup ---


import os
import sys
import glob

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

FIXED_DIR = PIPELINE_DIR
sys.path.insert(0, FIXED_DIR)
from eval_utils import subject_id_from_path, make_windows, full_metrics  # noqa: E402

DATA_DIR = COMBINED_DIR
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = LOSO_MODEL_DIR          # per-fold LOSO models + scalers
N_TEST_SEG = 2        # the last few sets are held out entirely as the test set
N_REPEAT = 2          # number of repeats per k (averaged to reduce noise)
WINDOW_SIZE = 40
STEP_SIZE = 5
FT_EPOCHS = 20
FT_BATCH = 16
FT_LR = 1e-5
LABEL_EMG_COLS = ['EMG_Main_MVC', 'EMG_Compass_MVC']


def seg_num(path):
    import re
    m = re.search(r'_Seg_(\d+)_', os.path.basename(path))
    return int(m.group(1)) if m else 0


def run_subject(subject, ts_cols, static_cols):
    files = sorted(glob.glob(os.path.join(DATA_DIR, f"{subject}_Seg_*_Combined_Features.csv")),
                   key=seg_num)
    if len(files) < N_TEST_SEG + 2:
        print(f"⚠️ {subject}: not enough sets, skipping")
        return []

    base_path = os.path.join(MODEL_DIR, f'loso_without_{subject}.keras')
    s_ts = joblib.load(os.path.join(MODEL_DIR, f'loso_scaler_ts_{subject}.pkl'))
    s_st = joblib.load(os.path.join(MODEL_DIR, f'loso_scaler_static_{subject}.pkl'))

    calib_files = files[:-N_TEST_SEG]
    test_files = files[-N_TEST_SEG:]
    print(f"\n===== {subject} | calibration pool {len(calib_files)} sets / test {len(test_files)} sets "
          f"(Seg {[seg_num(f) for f in test_files]}) =====")

    test_dfs = [pd.read_csv(f) for f in test_files]
    Xte, Ste, Yte, _ = make_windows(test_dfs, ts_cols, static_cols, LABEL_EMG_COLS,
                                    None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)

    rows = []
    for k in range(0, len(calib_files) + 1):
        reps = []
        for rep in range(N_REPEAT if k > 0 else 1):
            tf.keras.utils.set_random_seed(1000 + rep)
            model = tf.keras.models.load_model(base_path)   # ★ reload the zero-shot starting point for every k

            n_windows = 0
            if k > 0:
                cdfs = [pd.read_csv(f) for f in calib_files[:k]]
                Xc, Sc, Yc, _ = make_windows(cdfs, ts_cols, static_cols, LABEL_EMG_COLS,
                                             None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
                n_windows = len(Xc)
                if n_windows > 0:
                    for layer in model.layers:
                        layer.trainable = ('feature' not in layer.name)
                    model.compile(optimizer=tf.keras.optimizers.Adam(FT_LR),
                                  loss={'out_emg': 'mse'}, loss_weights={'out_emg': 3.0},
                                  metrics={'out_emg': 'mae'})
                    model.fit({'ts_input': Xc, 'static_input': Sc}, {'out_emg': Yc},
                              epochs=FT_EPOCHS, batch_size=FT_BATCH, verbose=0)

            pred = model.predict({'ts_input': Xte, 'static_input': Ste}, verbose=0)
            pred = pred[0] if isinstance(pred, list) else pred
            reps.append((full_metrics(Yte, pred), n_windows))
            tf.keras.backend.clear_session()

        ms = [r[0] for r in reps]
        row = {'Subject': subject, 'k_segments': k,
               'n_calib_windows': reps[0][1]}
        for i, mu in enumerate(['main', 'syn']):
            row[f'r_{mu}'] = float(np.mean([m['r'][i] for m in ms]))
            row[f'mae_{mu}'] = float(np.mean([m['mae'][i] for m in ms]))
            row[f'nrmse_{mu}'] = float(np.mean([m['nrmse'][i] for m in ms]))
        rows.append(row)
        print(f"   k={k:>2}  main r={row['r_main']:.3f} nRMSE={row['nrmse_main']:.3f}"
              f" | synergist r={row['r_syn']:.3f} nRMSE={row['nrmse_syn']:.3f}")
    return rows


def plot(df):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    subjects = sorted(df['Subject'].unique())
    cmap = plt.get_cmap('tab10')

    panels = [('r_main', 'Main muscle Pearson r', 0, 0), ('r_syn', 'Synergist Pearson r', 0, 1),
              ('nrmse_main', 'Main muscle nRMSE (lower is better)', 1, 0), ('nrmse_syn', 'Synergist nRMSE (lower is better)', 1, 1)]
    for col, title, i, j in panels:
        ax = axes[i][j]
        for n, s in enumerate(subjects):
            d = df[df['Subject'] == s]
            ax.plot(d['k_segments'], d[col], marker='o', ms=4, lw=1.2,
                    color=cmap(n % 10), alpha=0.55, label=s)
        mean = df.groupby('k_segments')[col].mean()
        ax.plot(mean.index, mean.values, color='black', lw=2.6, marker='s', ms=6,
                label='Mean', zorder=10)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Personal calibration data k (number of sets)')
        ax.grid(alpha=0.25)
    axes[0][0].legend(fontsize=7, ncol=2)
    fig.suptitle('Calibration curve: personalized performance vs amount of personal data (k=0 is zero-shot)', fontsize=13)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, 'calibration_curve.png')
    fig.savefig(out, dpi=150)
    print(f"📊 Figure saved to {out}")


def main():
    for f in ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']:
        try:
            matplotlib.rcParams['font.sans-serif'] = [f]
            matplotlib.rcParams['axes.unicode_minus'] = False
            break
        except Exception:
            continue

    spec = joblib.load(os.path.join(MODEL_DIR, 'feature_spec.pkl'))
    ts_cols, static_cols = spec['ts_cols'], spec['static_cols']

    all_files = glob.glob(os.path.join(DATA_DIR, "*_Combined_Features.csv"))
    subjects = sorted(set(subject_id_from_path(p) for p in all_files))

    rows = []
    for s in subjects:
        rows += run_subject(s, ts_cols, static_cols)

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, 'calibration_curve.csv')
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print(f"\n✅ Saved {out}")

    print("\n=== Cross-subject mean for each k ===")
    agg = df.groupby('k_segments')[['r_main', 'r_syn', 'nrmse_main', 'nrmse_syn']].mean()
    print(agg.round(3).to_string())
    plot(df)


if __name__ == '__main__':
    main()
