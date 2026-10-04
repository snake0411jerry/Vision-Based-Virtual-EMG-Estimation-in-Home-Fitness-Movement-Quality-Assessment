"""
plot_training_curves.py — training curves (Loss / MAE vs Epoch)
=========================================================================
Records per-epoch metrics of LOSO training and personalized fine-tuning and plots them.

[Relationship to the evaluation pipeline — important]
  This script passes the holdout to fit() as validation_data, but **attaches no val-dependent
  callback** (no EarlyStopping, no ModelCheckpoint),
  so val is only used to "compute and record metrics" and never affects weight updates.
  The LOSO evaluation therefore remains valid, and the val curve here is an honest generalization curve.

  The only exception is the fine-tuning stage: it already had EarlyStopping(patience=5),
  matching phase2_finetune_loso.py, so the actual training process is reproduced faithfully.

[The model has a single EMG output head]
  The compensation-classification (comp) task is off by default (its labels come from geometric thresholds, which is circular),
  so this figure has only EMG-related curves, additionally broken down into per-channel MAE for the main muscle / synergist.

Output:
  training_curves_loso.png      LOSO training (10 folds overlaid + mean)
  training_curves_finetune.png  personalized fine-tuning (10 subjects overlaid + mean)
  training_history.csv          raw per-epoch values
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
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

FIXED_DIR = PIPELINE_DIR
sys.path.insert(0, FIXED_DIR)
from eval_utils import (subject_id_from_path, build_ts_cols,      # noqa: E402
                        make_windows, temporal_split_df)

DATA_DIR = COMBINED_DIR
HERE = os.path.dirname(os.path.abspath(__file__))

FEATURE_MODE = "spatial"
WINDOW_SIZE = 40
STEP_SIZE = 5
EPOCHS = 30              # consistent with loso_train_and_save.py
BATCH_SIZE = 32
RANDOM_STATE = 42

VAL_TAIL_FRAC = 0.2
FT_EPOCHS = 20
FT_BATCH = 16
FT_LR = 1e-5

STATIC_COLS = ['Subj_Age', 'Subj_Height_m', 'Subj_Weight_norm', 'Subj_Gender']
LABEL_EMG_COLS = ['EMG_Main_MVC', 'EMG_Compass_MVC']

C_TR, C_VA = '#e0623c', '#0f766e'


def mae_main(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true[:, 0] - y_pred[:, 0]))


def mae_syn(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true[:, 1] - y_pred[:, 1]))


def build_model(n_ts, n_static, lr=1e-3):
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
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse',
              metrics=['mae', mae_main, mae_syn])
    return m


PANELS = [
    ('loss',     'MSE Loss',        'lower is better'),
    ('mae',      'MAE (both channels)',    'lower is better'),
    ('mae_main', 'Main muscle MAE',       'lower is better'),
    ('mae_syn',  'Synergist MAE',       'lower is better'),
]


def plot_curves(hist_list, labels, title, outpath, epochs):
    """hist_list: [{key: [per-epoch values]}]; overlay every fold + a thick mean line."""
    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.6 * len(PANELS), 4.2))
    ep = np.arange(1, epochs + 1)

    for ax, (key, title_txt, sub) in zip(axes, PANELS):
        tr_all, va_all = [], []
        for h in hist_list:
            tr = h.get(key, [])
            va = h.get('val_' + key, [])
            if len(tr) == epochs:
                tr_all.append(tr)
                ax.plot(ep, tr, color=C_TR, alpha=.18, lw=.9)
            if len(va) == epochs:
                va_all.append(va)
                ax.plot(ep, va, color=C_VA, alpha=.18, lw=.9)

        if tr_all:
            ax.plot(ep, np.mean(tr_all, axis=0), color=C_TR, lw=2.6, label='train (mean)')
        if va_all:
            ax.plot(ep, np.mean(va_all, axis=0), color=C_VA, lw=2.6, label='val (mean)')

        ax.set_title(f'{title_txt}\n({sub})', fontsize=11)
        ax.set_xlabel('Epoch')
        ax.grid(alpha=.25)
        ax.set_xlim(1, epochs)
    axes[0].set_ylabel('Value')
    axes[0].legend(fontsize=9.5)
    fig.suptitle(title, fontsize=13.5, fontweight='bold')
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"📊 Saved {outpath}")


def main():
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    for f in ['Microsoft JhengHei', 'SimHei']:
        matplotlib.rcParams['font.sans-serif'] = [f]
        matplotlib.rcParams['axes.unicode_minus'] = False
        break

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_Combined_Features.csv")))
    groups = [subject_id_from_path(p) for p in files]
    subjects = sorted(set(groups))
    ts_cols = build_ts_cols(pd.read_csv(files[0]).columns.tolist(), FEATURE_MODE, False)
    print(f"📂 {len(files)} files / {len(subjects)} subjects  🧩 time-series features={len(ts_cols)}\n")

    loso_hists, ft_hists, rows = [], [], []

    for hold in subjects:
        tr_files = [p for p, g in zip(files, groups) if g != hold]
        va_files = [p for p, g in zip(files, groups) if g == hold]
        print(f"🔁 fold {hold} …", end=' ', flush=True)

        tr_dfs = [pd.read_csv(p) for p in tr_files]
        va_dfs = [pd.read_csv(p) for p in va_files]
        big = pd.concat(tr_dfs, ignore_index=True)
        s_ts = StandardScaler().fit(big[ts_cols].values)
        s_st = StandardScaler().fit(big[STATIC_COLS].values)

        Xtr, Str, Ytr, _ = make_windows(tr_dfs, ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                        None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
        Xva, Sva, Yva, _ = make_windows(va_dfs, ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                        None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)

        model = build_model(len(ts_cols), len(STATIC_COLS))
        # ★ no val-dependent callback → val is purely for monitoring and does not affect training
        h = model.fit({'ts_input': Xtr, 'static_input': Str}, Ytr,
                      validation_data=({'ts_input': Xva, 'static_input': Sva}, Yva),
                      epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        loso_hists.append(h.history)
        for e in range(EPOCHS):
            rows.append({'stage': 'loso', 'subject': hold, 'epoch': e + 1,
                         **{k: v[e] for k, v in h.history.items()}})
        print(f"loss {h.history['loss'][-1]:.4f} / val {h.history['val_loss'][-1]:.4f}", end='')

        # ---- personalized fine-tuning (with EarlyStopping, matching the formal pipeline) ----
        tr_parts, va_parts = [], []
        for d in va_dfs:
            a, b = temporal_split_df(d, VAL_TAIL_FRAC)
            tr_parts.append(a); va_parts.append(b)
        Xf, Sf, Yf, _ = make_windows(tr_parts, ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                     None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
        Xv, Sv, Yv, _ = make_windows(va_parts, ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                     None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
        if len(Xv):
            for layer in model.layers:
                layer.trainable = ('feature' not in layer.name)
            model.compile(optimizer=tf.keras.optimizers.Adam(FT_LR), loss='mse',
                          metrics=['mae', mae_main, mae_syn])
            hf = model.fit({'ts_input': Xf, 'static_input': Sf}, Yf,
                           validation_data=({'ts_input': Xv, 'static_input': Sv}, Yv),
                           epochs=FT_EPOCHS, batch_size=FT_BATCH, verbose=0)
            ft_hists.append(hf.history)
            for e in range(len(hf.history['loss'])):
                rows.append({'stage': 'finetune', 'subject': hold, 'epoch': e + 1,
                             **{k: v[e] for k, v in hf.history.items()}})
            print(f"  |  fine-tune loss {hf.history['loss'][-1]:.4f}")
        else:
            print()
        tf.keras.backend.clear_session()

    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, 'training_history.csv'),
                              index=False, encoding='utf-8-sig')

    plot_curves(loso_hists, subjects,
                'LOSO training curves (10 folds overlaid; thin lines are folds, thick line is the mean)\n'
                'val = the subject held out in that fold, for monitoring only, not used for weight updates',
                os.path.join(RESULTS_DIR, 'training_curves_loso.png'), EPOCHS)
    plot_curves(ft_hists, subjects,
                'Personalized fine-tuning training curves (10 subjects overlaid)\n'
                'val = last 20% of each segment (temporal held-out)',
                os.path.join(RESULTS_DIR, 'training_curves_finetune.png'), FT_EPOCHS)

    print("\n=== Convergence summary (LOSO, mean over 10 folds) ===")
    for key, name, _ in PANELS:
        tr = np.mean([h[key] for h in loso_hists], axis=0)
        va = np.mean([h['val_' + key] for h in loso_hists], axis=0)
        print(f"  {name:<14} train {tr[0]:.4f} → {tr[-1]:.4f}   val {va[0]:.4f} → {va[-1]:.4f}"
              f"   (val/train = {va[-1]/tr[-1]:.2f})")
    print(f"\n✅ Saved training_history.csv")


if __name__ == '__main__':
    main()
