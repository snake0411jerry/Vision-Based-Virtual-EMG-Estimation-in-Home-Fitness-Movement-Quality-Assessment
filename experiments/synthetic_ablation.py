"""
synthetic_ablation.py — a synthetic control experiment that artificially reduces synergist activation
=========================================================================
[⚠️ First a mathematical fact, which decides how the experiment is designed]
-------------------------------------------------------------------------
Your original idea was "multiply %MVC by 0.5 and see whether the model degrades in a way similar to S04".
But Pearson r is **completely immune** to positive linear transformations:

    r(a·y, ŷ) ≡ r(y, ŷ)      for any a > 0

That is, simply multiplying the ground truth by 0.5 leaves r unchanged to the last decimal. So this operation
**cannot** explain why S04's r is low — but it is a very valuable control in its own right,
because it directly proves that "a small synergist amplitude" **alone cannot** cause a low r.
(MAE shrinks proportionally and nRMSE is unchanged, because the std in the denominator is scaled too.)

So where does S04's low r come from? A sound physical explanation is **signal-to-noise ratio**:
the weaker the muscle output, the closer the true physiological signal is to the electrode/instrument noise floor,
the larger the share drowned in noise, and the less of it is predictable. This is the classic
measurement-error attenuation phenomenon.

[Hence this script builds two synthetic cases]
-------------------------------------------------------------------------
Variant A (scaling only, your original idea)   y' = a·y
    Expected: r unchanged, nRMSE unchanged, MAE shrinks proportionally.
    Role: control, proving that amplitude itself is not the main cause.

Variant B (scaling + fixed noise floor)        y' = a·y + N(0, σ)
    σ is fixed and does not shrink with a (the noise floor is set by the instrument, independent of output).
    Expected: smaller a → worse SNR → r decreases monotonically.
    Role: test the causal path "low activation → low SNR → low r",
          and find which (a, σ) reproduces S04's (mean activation, zero-shot r).

⚠️ 2026-10-05: the reference points of the real subjects (including S04) are now computed on the fly
   by real_subject_points() from the current data and LOSO models. The old hard-coded values such as
   (16.6%, r=0.278) were measured before the MVC denominator fix (2026-08-05) and were on a different
   scale from the corrected synthetic cases.

[What this experiment can and cannot answer]
-------------------------------------------------------------------------
✔ Can answer: whether S04's low r can be explained by the measurement effect of "small signal + fixed noise",
          without assuming the model has a structural flaw for low-activation people.
✘ Cannot answer: whether the model "fails to learn" the motion–EMG mapping of low-activation people.
          That would require retraining with modified labels (or more subjects);
          this script only changes the ground truth used for evaluation, not the model predictions.
          Scope the paper's conclusions accordingly.

Output: synthetic_ablation.csv + synthetic_ablation.png
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
from eval_utils import make_windows, full_metrics  # noqa: E402

DATA_DIR = COMBINED_DIR
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = LOSO_MODEL_DIR          # per-fold LOSO models + scalers
# use subjects with normal activation as "prototypes", reduce them artificially and see whether they degrade to look like S04
DONOR_SUBJECTS = ['S07', 'S03', 'S08']
# reference points of the real subjects are computed by real_subject_points(), not hard-coded (see the 2026-10-05 note in the header)
REAL_SUBJECTS = ['S04', 'S06', 'S01', 'S05', 'S08', 'S02', 'S03', 'S07']

SCALES = [1.0, 0.7, 0.5, 0.3, 0.2]
NOISE_SIGMAS = [0.0, 0.02, 0.05]     # in %MVC units (0.02 = 2% MVC)
WINDOW_SIZE, STEP_SIZE = 40, 5
LABEL_EMG_COLS = ['EMG_Main_MVC', 'EMG_Compass_MVC']
SYN_IDX = 1
CLIP_HI = 1.5
RNG = np.random.default_rng(42)


def real_subject_points(ts_cols, static_cols):
    """Each real subject's (mean synergist activation ×100, zero-shot synergist r), computed from current data.

    Exactly the same protocol as the synthetic cases: same windows, last frame of the window as the target,
    predictions from the same loso_without_{S}.keras. Reads no CSV from results/.
    """
    out = []
    for subj in REAL_SUBJECTS:
        files = sorted(glob.glob(os.path.join(DATA_DIR, f"{subj}_Seg_*_Combined_Features.csv")))
        base = os.path.join(MODEL_DIR, f'loso_without_{subj}.keras')
        if not files or not os.path.exists(base):
            print(f"⚠️ skipping {subj}")
            continue
        s_ts = joblib.load(os.path.join(MODEL_DIR, f'loso_scaler_ts_{subj}.pkl'))
        s_st = joblib.load(os.path.join(MODEL_DIR, f'loso_scaler_static_{subj}.pkl'))
        dfs = [pd.read_csv(f) for f in files]
        X, S, Y, _ = make_windows(dfs, ts_cols, static_cols, LABEL_EMG_COLS,
                                  None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
        model = tf.keras.models.load_model(base)
        pred = model.predict({'ts_input': X, 'static_input': S}, verbose=0)
        pred = pred[0] if isinstance(pred, list) else pred
        tf.keras.backend.clear_session()
        r = float(np.corrcoef(Y[:, SYN_IDX], pred[:, SYN_IDX])[0, 1])
        out.append({'subj': subj, 'mvc': float(Y[:, SYN_IDX].mean() * 100), 'r': r})
    return pd.DataFrame(out)


def main():
    for f in ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']:
        matplotlib.rcParams['font.sans-serif'] = [f]
        matplotlib.rcParams['axes.unicode_minus'] = False
        break

    spec = joblib.load(os.path.join(MODEL_DIR, 'feature_spec.pkl'))
    ts_cols, static_cols = spec['ts_cols'], spec['static_cols']

    real = real_subject_points(ts_cols, static_cols)
    s04 = real.set_index('subj').loc['S04']
    S04_REF = {'mean_mvc_pct': s04['mvc'], 'r_syn': s04['r']}
    print('Real-subject reference points (current data):')
    print(real.round(3).to_string(index=False))

    rows = []
    for subj in DONOR_SUBJECTS:
        files = sorted(glob.glob(os.path.join(DATA_DIR, f"{subj}_Seg_*_Combined_Features.csv")))
        base = os.path.join(MODEL_DIR, f'loso_without_{subj}.keras')
        if not files or not os.path.exists(base):
            print(f"⚠️ skipping {subj}")
            continue

        s_ts = joblib.load(os.path.join(MODEL_DIR, f'loso_scaler_ts_{subj}.pkl'))
        s_st = joblib.load(os.path.join(MODEL_DIR, f'loso_scaler_static_{subj}.pkl'))
        dfs = [pd.read_csv(f) for f in files]
        X, S, Y, _ = make_windows(dfs, ts_cols, static_cols, LABEL_EMG_COLS,
                                  None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)

        model = tf.keras.models.load_model(base)
        pred = model.predict({'ts_input': X, 'static_input': S}, verbose=0)
        pred = pred[0] if isinstance(pred, list) else pred
        tf.keras.backend.clear_session()

        y_syn = Y[:, SYN_IDX]
        p_syn = pred[:, SYN_IDX]
        print(f"\n===== {subj} | original synergist mean={y_syn.mean()*100:.1f}%MVC  "
              f"r={np.corrcoef(y_syn, p_syn)[0,1]:.3f} =====")

        for a in SCALES:
            for sigma in NOISE_SIGMAS:
                y_mod = a * y_syn
                if sigma > 0:
                    y_mod = y_mod + RNG.normal(0, sigma, size=y_mod.shape)
                y_mod = np.clip(y_mod, 0, CLIP_HI)

                m = full_metrics(np.column_stack([Y[:, 0], y_mod]),
                                 np.column_stack([pred[:, 0], p_syn]))
                variant = 'A_scale_only' if sigma == 0 else f'B_scale+noise σ={sigma}'
                rows.append({
                    'Subject': subj, 'variant': variant, 'scale_a': a, 'sigma': sigma,
                    'mean_mvc_pct': y_mod.mean() * 100,
                    'std_mvc_pct': y_mod.std() * 100,
                    'r_syn': m['r'][SYN_IDX],
                    'mae_syn': m['mae'][SYN_IDX],
                    'nrmse_syn': m['nrmse'][SYN_IDX],
                })
                print(f"   a={a:<4} σ={sigma:<5} → mean={y_mod.mean()*100:5.1f}%MVC "
                      f"r={m['r'][SYN_IDX]:+.3f}  MAE={m['mae'][SYN_IDX]:.3f}  "
                      f"nRMSE={m['nrmse'][SYN_IDX]:.3f}")

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, 'synthetic_ablation.csv')
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print(f"\n✅ Saved {out}")

    # --- invariance check of variant A (the key control the paper can cite directly) ---
    print("\n" + "=" * 68)
    print("[Variant A, scaling only] Does r change with amplitude?")
    a_only = df[df['sigma'] == 0]
    for subj in a_only['Subject'].unique():
        d = a_only[a_only['Subject'] == subj]
        print(f"  {subj}: r range = {d['r_syn'].min():.4f} ~ {d['r_syn'].max():.4f}"
              f"  (change {d['r_syn'].max()-d['r_syn'].min():.6f})"
              f" | mean %MVC dropped from {d['mean_mvc_pct'].max():.1f} to {d['mean_mvc_pct'].min():.1f}")
    print("  → amplitude reduced to 1/5 and r barely moves: proof that low amplitude by itself does not lower r.")

    print(f"\n[Variant B, scaling + noise] Can it reproduce S04's (mean activation {S04_REF['mean_mvc_pct']:.1f}, r={S04_REF['r_syn']:.3f})?")
    b = df[df['sigma'] > 0].copy()
    b['dist'] = ((b['mean_mvc_pct'] - S04_REF['mean_mvc_pct']) / 10) ** 2 + \
                (b['r_syn'] - S04_REF['r_syn']) ** 2
    print(b.nsmallest(5, 'dist')[
        ['Subject', 'variant', 'scale_a', 'sigma', 'mean_mvc_pct', 'r_syn', 'nrmse_syn']
    ].round(3).to_string(index=False))

    # --- figure ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for (subj, var), d in df.groupby(['Subject', 'variant']):
        d = d.sort_values('scale_a')
        ls = '--' if var.startswith('A') else '-'
        ax.plot(d['scale_a'], d['r_syn'], marker='o', ms=4, ls=ls, label=f"{subj} {var}", alpha=0.8)
    ax.axhline(S04_REF['r_syn'], color='crimson', lw=1.6, ls=':', label=f"S04 measured r={S04_REF['r_syn']:.3f}")
    ax.set_xlabel('Amplitude scaling factor a'); ax.set_ylabel('Synergist Pearson r')
    ax.set_title('Variant A (dashed, r unchanged) vs variant B (solid, r falls with SNR)')
    ax.invert_xaxis(); ax.grid(alpha=0.25); ax.legend(fontsize=6.5, ncol=2)

    ax = axes[1]
    bb = df[df['sigma'] > 0]
    ax.scatter(bb['mean_mvc_pct'], bb['r_syn'], s=26, alpha=0.45, label='Synthetic cases (variant B)')
    ax.scatter(real['mvc'], real['r'], s=95, marker='*', color='crimson',
               edgecolor='k', linewidth=0.5, label='Actual subjects', zorder=5)
    for _, rr in real.iterrows():
        ax.annotate(rr['subj'], (rr['mvc'], rr['r']), fontsize=7,
                    xytext=(3, 3), textcoords='offset points')
    ax.set_xlabel('Mean synergist muscle activation'); ax.set_ylabel('Synergist Pearson r')
    ax.set_title('Synthetic cases vs actual subjects')
    ax.grid(alpha=0.25); ax.legend(fontsize=8)

    fig.tight_layout()
    p = os.path.join(RESULTS_DIR, 'synthetic_ablation.png')
    fig.savefig(p, dpi=150)
    print(f"\n📊 Figure saved to {p}")


if __name__ == '__main__':
    main()
