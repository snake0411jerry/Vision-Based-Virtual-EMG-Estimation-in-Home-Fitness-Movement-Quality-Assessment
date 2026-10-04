"""
plot_holdout_two_curves.py — training curves of the train_holdout_two model
=========================================================================
Reruns the training setup of train_holdout_two.py (62 non-_two files for training / 12 _two files for validation),
additionally recording per-epoch loss and MAE and plotting them.

[Role of val]
  The _two sets are also the evaluation set of train_holdout_two.py. This script passes them to fit() as
  validation_data, but **attaches no val-dependent callback** (no EarlyStopping, no ModelCheckpoint),
  so val is used purely to compute metrics and does not affect weight updates;
  the original evaluation results remain valid.

[What this split measures — different from LOSO]
  S01/S03/S05/S07/S09/S10 have no _two data and go entirely into the training set;
  for the 4 subjects with _two (S02/S04/S06/S08), their non-_two segments are in the training set too.
  So this is "the same person, unseen re-recorded sets" (within-subject, cross-trial),
  **not** cross-subject generalization — cite LOSO for cross-person numbers.

Output:
  holdout_two_training_curves.png
  holdout_two_history.csv
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
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FIXED_DIR = PIPELINE_DIR
sys.path.insert(0, FIXED_DIR)

import train_holdout_two as T                                    # noqa: E402
from eval_utils import build_ts_cols, make_windows               # noqa: E402

DATA_DIR = T.DATA_DIR
WINDOW_SIZE, STEP_TRAIN = T.WINDOW_SIZE, T.STEP_TRAIN
EPOCHS, BATCH, SEED = T.EPOCHS, T.BATCH, T.SEED
STATIC_COLS = T.STATIC_COLS
LABEL_EMG_COLS = T.LABEL_EMG_COLS

C_TR, C_VA = '#e0623c', '#0f766e'

PANELS = [
    ('loss',     'MSE Loss'),
    ('mae',      'MAE (both channels)'),
    ('mae_main', 'Main muscle MAE'),
    ('mae_syn',  'Synergist MAE'),
]


def mae_main(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true[:, 0] - y_pred[:, 0]))


def mae_syn(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true[:, 1] - y_pred[:, 1]))


def build_model(n_ts, n_static):
    """Same architecture as train_holdout_two.build_model, plus per-channel MAE metrics."""
    from tensorflow.keras.layers import (Input, Dense, Dropout, Conv1D,
                                         BatchNormalization, GlobalAveragePooling1D,
                                         Concatenate)
    i_ts = Input(shape=(WINDOW_SIZE, n_ts), name='ts_input')
    i_st = Input(shape=(n_static,), name='static_input')
    x = Conv1D(64, 3, padding='causal', activation='relu', name='feature_conv1')(i_ts)
    x = BatchNormalization(name='feature_bn1')(x)
    x = Conv1D(64, 3, padding='causal', dilation_rate=2, activation='relu', name='feature_conv2')(x)
    f = GlobalAveragePooling1D(name='feature_pool')(x)
    z = Concatenate(name='fusion_concat')([f, i_st])
    z = Dense(64, activation='relu', name='fusion_dense')(z)
    z = Dropout(0.2, name='fusion_dropout')(z)
    z = Dense(32, activation='relu', name='emg_dense')(z)
    o = Dense(2, activation='sigmoid', name='out_emg')(z)
    m = tf.keras.Model([i_ts, i_st], [o])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse',
              metrics=['mae', mae_main, mae_syn])
    return m


def main():
    tf.keras.utils.set_random_seed(SEED)
    for f in ['Microsoft JhengHei', 'SimHei']:
        matplotlib.rcParams['font.sans-serif'] = [f]
        matplotlib.rcParams['axes.unicode_minus'] = False
        break

    tmap = T.two_segment_map()
    two_keys = {(s, seg) for s, lst in tmap.items() for seg, _ in lst}

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_Combined_Features.csv")))
    va_files = [p for p in files if (T.subj_of(p), T.seg_of(p)) in two_keys]
    tr_files = [p for p in files if (T.subj_of(p), T.seg_of(p)) not in two_keys]
    print(f"📂 training {len(tr_files)} files (non-_two) / validation {len(va_files)} files (_two)")
    print(f"   validation sets: {sorted({T.subj_of(p) for p in va_files})}")

    tr_dfs = [pd.read_csv(p) for p in tr_files]
    va_dfs = [pd.read_csv(p) for p in va_files]
    ts_cols = build_ts_cols(tr_dfs[0].columns.tolist(), T.FEATURE_MODE, False)

    big = pd.concat(tr_dfs, ignore_index=True)
    s_ts = StandardScaler().fit(big[ts_cols].values)
    s_st = StandardScaler().fit(big[STATIC_COLS].values)

    Xtr, Str, Ytr, _ = make_windows(tr_dfs, ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                    None, WINDOW_SIZE, STEP_TRAIN, s_ts, s_st)
    Xva, Sva, Yva, _ = make_windows(va_dfs, ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                    None, WINDOW_SIZE, STEP_TRAIN, s_ts, s_st)
    print(f"🧩 time-series features={len(ts_cols)}  training windows={len(Xtr)}  validation windows={len(Xva)}")
    print(f"⏱  training {EPOCHS} epochs …")

    model = build_model(len(ts_cols), len(STATIC_COLS))
    # ★ no val-dependent callback → val is for monitoring only
    h = model.fit({'ts_input': Xtr, 'static_input': Str}, Ytr,
                  validation_data=({'ts_input': Xva, 'static_input': Sva}, Yva),
                  epochs=EPOCHS, batch_size=BATCH, verbose=0)
    hist = h.history

    pd.DataFrame({'epoch': np.arange(1, EPOCHS + 1), **hist}).to_csv(
        os.path.join(RESULTS_DIR, 'holdout_two_history.csv'), index=False, encoding='utf-8-sig')

    # ---------- plotting ----------
    ep = np.arange(1, EPOCHS + 1)
    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.7 * len(PANELS), 4.4))
    for ax, (key, title) in zip(axes, PANELS):
        tr, va = hist[key], hist['val_' + key]
        ax.plot(ep, tr, color=C_TR, lw=2.0, marker='o', ms=2.6, label='train')
        ax.plot(ep, va, color=C_VA, lw=2.0, marker='o', ms=2.6, label='val (_two)')
        best = int(np.argmin(va))
        ax.axvline(best + 1, color=C_VA, ls=':', lw=1.4, alpha=.75)
        ax.plot(best + 1, va[best], marker='*', ms=14, color=C_VA,
                markeredgecolor='k', markeredgewidth=.5, zorder=5)
        ax.annotate(f'val best\nep{best+1} = {va[best]:.4f}',
                    (best + 1, va[best]), fontsize=8.5, color=C_VA,
                    xytext=(6, 10), textcoords='offset points')
        ax.set_title(f'{title}\n(lower is better)', fontsize=11)
        ax.set_xlabel('Epoch')
        ax.grid(alpha=.25)
        ax.set_xlim(1, EPOCHS)
    axes[0].set_ylabel('Value')
    axes[0].legend(fontsize=10)
    fig.suptitle('train_holdout_two model training curves  '
                 '(62 non-_two for training / 12 _two for validation)\n'
                 'val is for monitoring only; no callbacks attached, so it does not affect weight updates',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    p = os.path.join(RESULTS_DIR, 'holdout_two_training_curves.png')
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"📊 Saved {p}")

    print(f"\n=== Convergence summary ===")
    for key, name in PANELS:
        tr, va = hist[key], hist['val_' + key]
        b = int(np.argmin(va))
        print(f"  {name:<14} train {tr[0]:.4f} → {tr[-1]:.4f} | "
              f"val {va[0]:.4f} → {va[-1]:.4f}  best ep{b+1}={va[b]:.4f}"
              f" (final vs best {'+' if va[-1]>va[b] else ''}{va[-1]-va[b]:.4f})")
    print(f"\n✅ Saved holdout_two_history.csv")


if __name__ == '__main__':
    main()
