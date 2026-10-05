"""
plot_zeroshot_vs_finetuned.py — waveform comparison of zero-shot vs personalized fine-tuning
=========================================================================
[What the plot compares]
  black  measured EMG (ground truth)
  orange zero-shot prediction: model loaded from loso_without_{S}.keras, which **never saw this person** during training
  green  fine-tuned prediction: the same model, further calibrated on the subject's "first 80% of each segment"

  The evaluation interval is always "the last 20% time tail of each segment", unseen by both models,
  so the orange–green difference is the true personalization gain (no leakage).

  Segment tails are drawn one after another on the same time axis; grey dashed lines mark the seams.

[Output]
  zs_vs_ft_waveform_main.png   main muscle
  zs_vs_ft_waveform_syn.png    synergist
  zs_vs_ft_summary.png         per-subject r / nRMSE before vs after + scatter plot
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
from eval_utils import (subject_id_from_path, make_windows,        # noqa: E402
                        temporal_split_df, full_metrics)

DATA_DIR = COMBINED_DIR
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = LOSO_MODEL_DIR          # per-fold LOSO models + scalers
VAL_TAIL_FRAC = 0.2
WINDOW_SIZE = 40
STEP_TRAIN = 5
FT_EPOCHS = 20
FT_BATCH = 16
FT_LR = 1e-5
FPS = 60.0

STATIC_COLS = ['Subj_Age', 'Subj_Height_m', 'Subj_Weight_norm', 'Subj_Gender']
LABEL_EMG_COLS = ['EMG_Main_MVC', 'EMG_Compass_MVC']
MUSCLE = ['Main muscle (Main)', 'Synergist (Synergist)']

C_TRUE, C_ZS, C_FT = '#15181a', '#e0623c', '#0f766e'


def dense_pred(model, df, ts_cols, s_ts, s_st):
    ts = s_ts.transform(df[ts_cols].values)
    st = s_st.transform(df[STATIC_COLS].values)
    y = df[LABEL_EMG_COLS].values
    X, S, idx = [], [], []
    for j in range(0, len(ts) - WINDOW_SIZE):
        X.append(ts[j:j + WINDOW_SIZE, :])
        t = j + WINDOW_SIZE - 1
        S.append(st[t, :]); idx.append(t)
    if not X:
        return None
    p = model.predict({'ts_input': np.array(X), 'static_input': np.array(S)},
                      verbose=0, batch_size=512)
    p = p[0] if isinstance(p, list) else p
    return np.array(idx), y[np.array(idx), :], p


def run_subject(subject, ts_cols):
    files = sorted(glob.glob(os.path.join(DATA_DIR, f"{subject}_Seg_*_Combined_Features.csv")))
    base = os.path.join(MODEL_DIR, f'loso_without_{subject}.keras')
    if not files or not os.path.exists(base):
        return None
    s_ts = joblib.load(os.path.join(MODEL_DIR, f'loso_scaler_ts_{subject}.pkl'))
    s_st = joblib.load(os.path.join(MODEL_DIR, f'loso_scaler_static_{subject}.pkl'))

    dfs = [pd.read_csv(f) for f in files]
    tr_parts, va_parts = [], []
    for d in dfs:
        a, b = temporal_split_df(d, VAL_TAIL_FRAC)
        tr_parts.append(a); va_parts.append(b)

    # ---- zero-shot ----
    model = tf.keras.models.load_model(base)
    zs = [dense_pred(model, d, ts_cols, s_ts, s_st) for d in va_parts]

    # ---- fine-tuned ----
    Xtr, Str, Ytr, _ = make_windows(tr_parts, ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                    None, WINDOW_SIZE, STEP_TRAIN, s_ts, s_st)
    Xva, Sva, Yva, _ = make_windows(va_parts, ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                    None, WINDOW_SIZE, STEP_TRAIN, s_ts, s_st)
    for l in model.layers:
        l.trainable = ('feature' not in l.name)
    model.compile(optimizer=tf.keras.optimizers.Adam(FT_LR),
                  loss={'out_emg': 'mse'}, loss_weights={'out_emg': 3.0},
                  metrics={'out_emg': 'mae'})
    model.fit({'ts_input': Xtr, 'static_input': Str}, {'out_emg': Ytr},
              epochs=FT_EPOCHS, batch_size=FT_BATCH,
              validation_data=({'ts_input': Xva, 'static_input': Sva}, {'out_emg': Yva}),
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                                          restore_best_weights=True)],
              verbose=0)
    ft = [dense_pred(model, d, ts_cols, s_ts, s_st) for d in va_parts]
    tf.keras.backend.clear_session()

    segs, yt, pz, pf, bounds = [], [], [], [], []
    off = 0
    for i, (a, b) in enumerate(zip(zs, ft)):
        if a is None or b is None:
            continue
        n = len(a[1])
        yt.append(a[1]); pz.append(a[2]); pf.append(b[2])
        segs.append(i); off += n; bounds.append(off)
    if not yt:
        return None
    yt = np.concatenate(yt); pz = np.concatenate(pz); pf = np.concatenate(pf)
    return {'subject': subject, 'y': yt, 'zs': pz, 'ft': pf, 'bounds': bounds[:-1],
            'm_zs': full_metrics(yt, pz), 'm_ft': full_metrics(yt, pf)}


def main():
    for f in ['Microsoft JhengHei', 'SimHei']:
        matplotlib.rcParams['font.sans-serif'] = [f]
        matplotlib.rcParams['axes.unicode_minus'] = False
        break

    spec = joblib.load(os.path.join(MODEL_DIR, 'feature_spec.pkl'))
    ts_cols = spec['ts_cols']

    subjects = sorted(set(subject_id_from_path(p)
                          for p in glob.glob(os.path.join(DATA_DIR, "*_Combined_Features.csv"))))
    res = []
    for s in subjects:
        print(f"Processing {s} ...")
        r = run_subject(s, ts_cols)
        if r:
            res.append(r)
            print(f"   main r {r['m_zs']['r'][0]:.3f} → {r['m_ft']['r'][0]:.3f} | "
                  f"synergist r {r['m_zs']['r'][1]:.3f} → {r['m_ft']['r'][1]:.3f}")

    # ---------- waveform plots ----------
    for mi, mname in enumerate(MUSCLE):
        n = len(res)
        fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(17, 2.7 * ((n + 1) // 2)),
                                 squeeze=False)
        for k, r in enumerate(res):
            ax = axes[k // 2][k % 2]
            t = np.arange(len(r['y'])) / FPS
            ax.plot(t, r['y'][:, mi], lw=1.0, color=C_TRUE, label='Measured EMG', zorder=3)
            ax.plot(t, r['zs'][:, mi], lw=1.0, color=C_ZS, alpha=0.85,
                    label='Zero-shot (never seen this person)', zorder=2)
            ax.plot(t, r['ft'][:, mi], lw=1.0, color=C_FT, alpha=0.85,
                    label='Fine-tuned (personalized)', zorder=2)
            for b in r['bounds']:
                ax.axvline(b / FPS, color='#999', ls=':', lw=0.7, alpha=0.6)
            rz, rf = r['m_zs']['r'][mi], r['m_ft']['r'][mi]
            nz, nf = r['m_zs']['nrmse'][mi], r['m_ft']['nrmse'][mi]
            ax.set_title(f"{r['subject']}    r {rz:.3f} → {rf:.3f} ({rf-rz:+.3f})   "
                         f"nRMSE {nz:.3f} → {nf:.3f}", fontsize=9.5)
            ax.set_xlabel('Time (s, segment tails joined)', fontsize=8)
            ax.set_ylabel('Muscle activation', fontsize=8)
            ax.tick_params(labelsize=7); ax.grid(alpha=0.2)
            if k == 0:
                ax.legend(fontsize=7.5, loc='upper right', ncol=3)
        for k in range(len(res), ((n + 1) // 2) * 2):
            axes[k // 2][k % 2].axis('off')
        fig.suptitle(f'{mname}: zero-shot vs personalized fine-tuning (evaluation interval is a time tail unseen by both)',
                     fontsize=13)
        fig.tight_layout()
        tag = 'main' if mi == 0 else 'syn'
        p = os.path.join(RESULTS_DIR, f'zs_vs_ft_waveform_{tag}.png')
        fig.savefig(p, dpi=140); plt.close(fig)
        print(f"📊 Saved {p}")

    # ---------- summary figure ----------
    subs = [r['subject'] for r in res]
    x = np.arange(len(subs)); w = 0.36
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for mi, mname in enumerate(MUSCLE):
        ax = axes[0][mi]
        rz = [r['m_zs']['r'][mi] for r in res]
        rf = [r['m_ft']['r'][mi] for r in res]
        ax.bar(x - w/2, rz, w, label='Zero-shot', color=C_ZS, alpha=0.85)
        ax.bar(x + w/2, rf, w, label='Fine-tuned', color=C_FT, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(subs, rotation=30, fontsize=8)
        ax.set_ylabel('Pearson r'); ax.set_title(f'{mname}  Pearson r (higher is better)', fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.2, axis='y')

        ax = axes[1][mi]
        nz = [r['m_zs']['nrmse'][mi] for r in res]
        nf = [r['m_ft']['nrmse'][mi] for r in res]
        ax.bar(x - w/2, nz, w, label='Zero-shot', color=C_ZS, alpha=0.85)
        ax.bar(x + w/2, nf, w, label='Fine-tuned', color=C_FT, alpha=0.85)
        ax.axhline(1.0, color='crimson', ls='--', lw=1.2, label='=1 level of guessing the mean')
        ax.set_xticks(x); ax.set_xticklabels(subs, rotation=30, fontsize=8)
        ax.set_ylabel('nRMSE'); ax.set_title(f'{mname}  nRMSE (lower is better)', fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.2, axis='y')
    fig.suptitle('Zero-shot vs personalized fine-tuning: per-subject comparison', fontsize=13)
    fig.tight_layout()
    p = os.path.join(RESULTS_DIR, 'zs_vs_ft_summary.png')
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"📊 Saved {p}")


if __name__ == '__main__':
    main()
