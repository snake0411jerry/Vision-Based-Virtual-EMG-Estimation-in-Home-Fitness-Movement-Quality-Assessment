"""
loso_exclude_s09.py — 9-subject LOSO training + personalized fine-tuning with S09 fully excluded (including the main muscle)
=========================================================================
[Difference from the previous two versions]
-------------------------------------------------------------------------
This project handled S09's synergist quality problem in three ways, from weakest to strongest:

  1. Exclude from evaluation only (loso_zeroshot_metrics.csv + post-hoc filtering)
     All of S09's data still takes part in training; its synergist numbers are just left out of the average.

  2. Training mask (loso_masked_synergist.py)
     S09's synergist is masked from the loss; the main muscle still takes part in training.
     ⟶ Measured: differs from (1) by ≤0.015, confirming that S09's bad labels
        did not contaminate the other subjects' training (it is only 9.2% of the training data).

  3. Full exclusion (this script)
     None of S09's data enters training or evaluation; the dataset goes back to 9 subjects / 82 files.
     ⟶ Most conservative, at the cost of losing S09's main-muscle data, which is of normal quality.

This script implements approach (3), for comparison with the other two to decide how S09 should be handled.

[Output]
  models_9subj/                    9-subject LOSO models and scalers
  loso_9subj_metrics.csv           zero-shot metrics
  finetune_9subj_results.csv       metrics before and after fine-tuning
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
MODEL_DIR = ABL_EXCLUDE_S09_DIR
# ★ fully excluded subjects (all channels; take part in neither training nor evaluation)
EXCLUDE_SUBJECTS = {'S09'}

FEATURE_MODE = "spatial"
WINDOW_SIZE = 40
STEP_SIZE = 5
EPOCHS = 30
BATCH_SIZE = 32
RANDOM_STATE = 42

VAL_TAIL_FRAC = 0.2
FT_EPOCHS = 20
FT_BATCH = 16
FT_LR = 1e-5

STATIC_COLS = ['Subj_Age', 'Subj_Height_m', 'Subj_Weight_norm', 'Subj_Gender']
LABEL_EMG_COLS = ['EMG_Main_MVC', 'EMG_Compass_MVC']


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
    m.compile(optimizer=tf.keras.optimizers.Adam(lr),
              loss={'out_emg': 'mse'}, loss_weights={'out_emg': 3.0},
              metrics={'out_emg': 'mae'})
    return m


def main():
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    os.makedirs(MODEL_DIR, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_Combined_Features.csv")))
    files = [p for p in all_files if subject_id_from_path(p) not in EXCLUDE_SUBJECTS]
    groups = [subject_id_from_path(p) for p in files]
    subjects = sorted(set(groups))
    print(f"📂 {len(all_files)} files originally → {len(files)} files after excluding {sorted(EXCLUDE_SUBJECTS)}")
    print(f"👥 subjects ({len(subjects)}): {subjects}\n")

    ts_cols = build_ts_cols(pd.read_csv(files[0]).columns.tolist(), FEATURE_MODE, False)
    print(f"🧩 time-series features={len(ts_cols)}")

    zs_rows, ft_rows = [], []
    for hold in subjects:
        tr_files = [p for p, g in zip(files, groups) if g != hold]
        va_files = [p for p, g in zip(files, groups) if g == hold]
        print(f"\n{'='*62}\n🔁 holding out {hold} (train {len(tr_files)} / holdout {len(va_files)})\n{'='*62}")

        tr_dfs = [pd.read_csv(p) for p in tr_files]
        va_dfs = [pd.read_csv(p) for p in va_files]

        big = pd.concat(tr_dfs, ignore_index=True)
        s_ts = StandardScaler().fit(big[ts_cols].values)
        s_st = StandardScaler().fit(big[STATIC_COLS].values)

        Xtr, Str, Ytr, _ = make_windows(tr_dfs, ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                        None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
        print(f"   training windows={len(Xtr)}")

        model = build_model(len(ts_cols), len(STATIC_COLS))
        model.fit({'ts_input': Xtr, 'static_input': Str}, {'out_emg': Ytr},
                  epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

        Xva, Sva, Yva, _ = make_windows(va_dfs, ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                        None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
        pred = model.predict({'ts_input': Xva, 'static_input': Sva}, verbose=0)
        pred = pred[0] if isinstance(pred, list) else pred
        m = full_metrics(Yva, pred)
        print(f"   [zero-shot] main r={m['r'][0]:.3f} nRMSE={m['nrmse'][0]:.3f}"
              f" | synergist r={m['r'][1]:.3f} nRMSE={m['nrmse'][1]:.3f}")

        model.save(os.path.join(MODEL_DIR, f'loso_without_{hold}.keras'))
        joblib.dump(s_ts, os.path.join(MODEL_DIR, f'loso_scaler_ts_{hold}.pkl'))
        joblib.dump(s_st, os.path.join(MODEL_DIR, f'loso_scaler_static_{hold}.pkl'))
        zs_rows.append({'Subject': hold,
                        'r_main': m['r'][0], 'r_syn': m['r'][1],
                        'mae_main': m['mae'][0], 'mae_syn': m['mae'][1],
                        'nrmse_main': m['nrmse'][0], 'nrmse_syn': m['nrmse'][1]})

        # ---- personalized fine-tuning ----
        tr_parts, va_parts = [], []
        for d in va_dfs:
            a, b = temporal_split_df(d, VAL_TAIL_FRAC)
            tr_parts.append(a); va_parts.append(b)

        Xf, Sf, Yf, _ = make_windows(tr_parts, ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                     None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
        Xv, Sv, Yv, _ = make_windows(va_parts, ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                     None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
        if len(Xv) == 0:
            print("   ⚠️ 0 validation windows, skipping fine-tuning")
            continue

        pb = model.predict({'ts_input': Xv, 'static_input': Sv}, verbose=0)
        pb = pb[0] if isinstance(pb, list) else pb
        mb = full_metrics(Yv, pb)

        for layer in model.layers:
            layer.trainable = ('feature' not in layer.name)
        model.compile(optimizer=tf.keras.optimizers.Adam(FT_LR),
                      loss={'out_emg': 'mse'}, loss_weights={'out_emg': 3.0},
                      metrics={'out_emg': 'mae'})
        model.fit({'ts_input': Xf, 'static_input': Sf}, {'out_emg': Yf},
                  epochs=FT_EPOCHS, batch_size=FT_BATCH,
                  validation_data=({'ts_input': Xv, 'static_input': Sv}, {'out_emg': Yv}),
                  callbacks=[tf.keras.callbacks.EarlyStopping(
                      monitor='val_loss', patience=5, restore_best_weights=True)],
                  verbose=0)

        pa = model.predict({'ts_input': Xv, 'static_input': Sv}, verbose=0)
        pa = pa[0] if isinstance(pa, list) else pa
        ma = full_metrics(Yv, pa)
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
    zs.to_csv(os.path.join(RESULTS_DIR, 'loso_9subj_metrics.csv'), index=False, encoding='utf-8-sig')
    ft.to_csv(os.path.join(RESULTS_DIR, 'finetune_9subj_results.csv'), index=False, encoding='utf-8-sig')
    joblib.dump({'ts_cols': ts_cols, 'static_cols': STATIC_COLS,
                 'feature_mode': FEATURE_MODE, 'excluded': sorted(EXCLUDE_SUBJECTS)},
                os.path.join(MODEL_DIR, 'feature_spec.pkl'))

    print(f"\n{'='*62}\n📈 9-subject results (fully excluding {sorted(EXCLUDE_SUBJECTS)})")
    print(f"   LOSO       main r={zs['r_main'].mean():.3f} nRMSE={zs['nrmse_main'].mean():.3f}"
          f" | synergist r={zs['r_syn'].mean():.3f} nRMSE={zs['nrmse_syn'].mean():.3f}")
    print(f"   fine-tuned main r={ft['r_main_after'].mean():.3f}"
          f" | synergist r={ft['r_syn_after'].mean():.3f} nRMSE={ft['nrmse_syn_after'].mean():.3f}")
    print("✅ Saved loso_9subj_metrics.csv / finetune_9subj_results.csv")


if __name__ == '__main__':
    main()
