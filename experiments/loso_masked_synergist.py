"""
loso_masked_synergist.py — exclude a given subject's synergist labels from the "training loss"
=========================================================================
[Why this script exists]
-------------------------------------------------------------------------
S09's synergist signal is suspected of poor electrode contact:
  · coefficient of variation only 0.31 (others 0.65–0.92) — the signal barely follows the movement
  · MVC denominator only 0.91 (others 2.7–6.05) — that session's maximal contraction was not captured
  · the model systematically underestimates by 17 percentage points — the labels are inflated by about 2×

Simply "leaving it out of the average" only fixes the reported numbers; **the bad labels are still in the other 9
folds' training sets** and may teach the model a wrong motion→synergist mapping. This script masks that subject's
synergist out of **the loss function**, while keeping the main-muscle channel (which is of normal quality, r=0.626).

[Method]
-------------------------------------------------------------------------
The model output is still 2-dimensional (main + synergist), but a custom loss is used:
    y_true carries 3 columns = [main, synergist, synergist mask]
    loss = Σ(error² × mask) / Σ(mask)
For samples whose mask is 0, the synergist error does not enter the gradient; the main muscle is always 1.
When every mask is 1, this loss is exactly equivalent to the original MSE.

[Output]
  models_masked/               masked-version LOSO models and scalers
  loso_masked_metrics.csv      zero-shot metrics
  finetune_masked_results.csv  metrics before and after fine-tuning
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

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

FIXED_DIR = PIPELINE_DIR
sys.path.insert(0, FIXED_DIR)
from eval_utils import (subject_id_from_path, build_ts_cols,      # noqa: E402
                        make_windows, full_metrics, temporal_split_df)

DATA_DIR = COMBINED_DIR
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = ABL_MASKED_DIR
# ★ subjects whose synergist labels are untrustworthy and must be excluded from the training loss
MASK_SYNERGIST = {'S09'}

FEATURE_MODE = "spatial"
WINDOW_SIZE = 40
STEP_SIZE = 5
EPOCHS = 30
BATCH_SIZE = 32
RANDOM_STATE = 42

# fine-tuning settings (same as phase2_finetune_loso.py)
VAL_TAIL_FRAC = 0.2
FT_EPOCHS = 20
FT_BATCH = 16
FT_LR = 1e-5

STATIC_COLS = ['Subj_Age', 'Subj_Height_m', 'Subj_Weight_norm', 'Subj_Gender']
LABEL_EMG_COLS = ['EMG_Main_MVC', 'EMG_Compass_MVC']
MASK_COL = '_syn_valid'


def load_with_mask(fp):
    """Read the file and attach a synergist mask column (0 if this subject is masked)."""
    df = pd.read_csv(fp)
    subj = subject_id_from_path(fp)
    df[MASK_COL] = 0.0 if subj in MASK_SYNERGIST else 1.0
    return df


def masked_mse(y_true, y_pred):
    """y_true = [main, synergist, synergist mask]; y_pred = [main, synergist].

    Synergist entries with mask 0 are excluded from the loss. Equivalent to standard MSE when every mask is 1.
    """
    target = y_true[:, :2]
    syn_mask = y_true[:, 2]
    mask = tf.stack([tf.ones_like(syn_mask), syn_mask], axis=1)
    sq = tf.square(target - y_pred)
    return tf.reduce_sum(sq * mask) / (tf.reduce_sum(mask) + 1e-8)


def build_model(n_ts_feat, n_static, lr=1e-3):
    from tensorflow.keras.layers import (Input, Dense, Dropout, Conv1D,
                                         BatchNormalization, GlobalAveragePooling1D,
                                         Concatenate)
    i_ts = Input(shape=(WINDOW_SIZE, n_ts_feat), name='ts_input')
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
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=masked_mse)
    return m


def windows_with_mask(dfs, ts_cols, s_ts, s_st):
    """Return X, S, Y3 (including the mask column)."""
    X, S, Y3, _ = make_windows(dfs, ts_cols, STATIC_COLS,
                               LABEL_EMG_COLS + [MASK_COL], None,
                               WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
    return X, S, Y3


def main():
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    os.makedirs(MODEL_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_Combined_Features.csv")))
    groups = [subject_id_from_path(p) for p in files]
    subjects = sorted(set(groups))
    print(f"📂 files={len(files)}  subjects={subjects}")
    print(f"🎭 synergist masked: {sorted(MASK_SYNERGIST)} (their main muscle still takes part in training)\n")

    ts_cols = build_ts_cols(load_with_mask(files[0]).columns.tolist(), FEATURE_MODE, False)
    # the mask column must not enter the input features
    ts_cols = [c for c in ts_cols if c != MASK_COL]
    print(f"🧩 time-series features={len(ts_cols)}")

    zs_rows, ft_rows = [], []
    for hold in subjects:
        tr_files = [p for p, g in zip(files, groups) if g != hold]
        va_files = [p for p, g in zip(files, groups) if g == hold]
        print(f"\n{'='*62}\n🔁 holding out {hold} (train {len(tr_files)} / holdout {len(va_files)})\n{'='*62}")

        tr_dfs = [load_with_mask(p) for p in tr_files]
        va_dfs = [load_with_mask(p) for p in va_files]

        big = pd.concat(tr_dfs, ignore_index=True)
        s_ts = StandardScaler().fit(big[ts_cols].values)
        s_st = StandardScaler().fit(big[STATIC_COLS].values)

        Xtr, Str, Ytr3 = windows_with_mask(tr_dfs, ts_cols, s_ts, s_st)
        masked_frac = 1.0 - Ytr3[:, 2].mean()
        print(f"   training windows={len(Xtr)} ({masked_frac*100:.1f}% of them have the synergist masked)")

        model = build_model(len(ts_cols), len(STATIC_COLS))
        model.fit({'ts_input': Xtr, 'static_input': Str}, Ytr3,
                  epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

        # ---- zero-shot evaluation ----
        Xva, Sva, Yva3 = windows_with_mask(va_dfs, ts_cols, s_ts, s_st)
        pred = model.predict({'ts_input': Xva, 'static_input': Sva}, verbose=0)
        pred = pred[0] if isinstance(pred, list) else pred
        m = full_metrics(Yva3[:, :2], pred)
        note = "(labels untrustworthy, for reference only)" if hold in MASK_SYNERGIST else ""
        print(f"   [zero-shot] main r={m['r'][0]:.3f} nRMSE={m['nrmse'][0]:.3f}"
              f" | synergist r={m['r'][1]:.3f} nRMSE={m['nrmse'][1]:.3f} {note}")

        model.save(os.path.join(MODEL_DIR, f'loso_without_{hold}.keras'))
        joblib.dump(s_ts, os.path.join(MODEL_DIR, f'loso_scaler_ts_{hold}.pkl'))
        joblib.dump(s_st, os.path.join(MODEL_DIR, f'loso_scaler_static_{hold}.pkl'))
        zs_rows.append({'Subject': hold,
                        'r_main': m['r'][0], 'r_syn': m['r'][1],
                        'mae_main': m['mae'][0], 'mae_syn': m['mae'][1],
                        'nrmse_main': m['nrmse'][0], 'nrmse_syn': m['nrmse'][1]})

        # ---- personalized fine-tuning (starting from this very zero-shot model) ----
        tr_parts, va_parts = [], []
        for d in va_dfs:
            a, b = temporal_split_df(d, VAL_TAIL_FRAC)
            tr_parts.append(a); va_parts.append(b)

        Xf, Sf, Yf3 = windows_with_mask(tr_parts, ts_cols, s_ts, s_st)
        Xv, Sv, Yv3 = windows_with_mask(va_parts, ts_cols, s_ts, s_st)
        if len(Xv) == 0:
            print("   ⚠️ 0 validation windows, skipping fine-tuning")
            continue

        p_before = model.predict({'ts_input': Xv, 'static_input': Sv}, verbose=0)
        p_before = p_before[0] if isinstance(p_before, list) else p_before
        mb = full_metrics(Yv3[:, :2], p_before)

        for layer in model.layers:
            layer.trainable = ('feature' not in layer.name)
        model.compile(optimizer=tf.keras.optimizers.Adam(FT_LR), loss=masked_mse)
        model.fit({'ts_input': Xf, 'static_input': Sf}, Yf3,
                  epochs=FT_EPOCHS, batch_size=FT_BATCH,
                  validation_data=({'ts_input': Xv, 'static_input': Sv}, Yv3),
                  callbacks=[tf.keras.callbacks.EarlyStopping(
                      monitor='val_loss', patience=5, restore_best_weights=True)],
                  verbose=0)

        p_after = model.predict({'ts_input': Xv, 'static_input': Sv}, verbose=0)
        p_after = p_after[0] if isinstance(p_after, list) else p_after
        ma = full_metrics(Yv3[:, :2], p_after)
        print(f"   [fine-tuned] main r={mb['r'][0]:.3f}→{ma['r'][0]:.3f}"
              f" | synergist r={mb['r'][1]:.3f}→{ma['r'][1]:.3f}"
              f"  nRMSE {mb['nrmse'][1]:.3f}→{ma['nrmse'][1]:.3f}")

        model.save(os.path.join(MODEL_DIR, f'{hold}_personalized.keras'))
        ft_rows.append({'Subject': hold,
                        'r_main_before': mb['r'][0], 'r_main_after': ma['r'][0],
                        'r_syn_before': mb['r'][1], 'r_syn_after': ma['r'][1],
                        'nrmse_main_before': mb['nrmse'][0], 'nrmse_main_after': ma['nrmse'][0],
                        'nrmse_syn_before': mb['nrmse'][1], 'nrmse_syn_after': ma['nrmse'][1]})
        tf.keras.backend.clear_session()

    zs = pd.DataFrame(zs_rows); ft = pd.DataFrame(ft_rows)
    zs.to_csv(os.path.join(RESULTS_DIR, 'loso_masked_metrics.csv'), index=False, encoding='utf-8-sig')
    ft.to_csv(os.path.join(RESULTS_DIR, 'finetune_masked_results.csv'), index=False, encoding='utf-8-sig')
    joblib.dump({'ts_cols': ts_cols, 'static_cols': STATIC_COLS,
                 'feature_mode': FEATURE_MODE, 'masked_synergist': sorted(MASK_SYNERGIST)},
                os.path.join(MODEL_DIR, 'feature_spec.pkl'))

    valid = zs[~zs['Subject'].isin(MASK_SYNERGIST)]
    vf = ft[~ft['Subject'].isin(MASK_SYNERGIST)]
    print(f"\n{'='*62}\n📈 Masked-version results (synergist statistics exclude {sorted(MASK_SYNERGIST)})")
    print(f"   LOSO       main r={zs['r_main'].mean():.3f}  synergist r={valid['r_syn'].mean():.3f}"
          f"  synergist nRMSE={valid['nrmse_syn'].mean():.3f}")
    print(f"   fine-tuned main r={ft['r_main_after'].mean():.3f}  synergist r={vf['r_syn_after'].mean():.3f}"
          f"  synergist nRMSE={vf['nrmse_syn_after'].mean():.3f}")
    print(f"✅ Saved loso_masked_metrics.csv / finetune_masked_results.csv")


if __name__ == '__main__':
    main()
