"""
loso_train_and_save.py — LOSO training that saves every fold's model (the basis of the personalization experiments)
=========================================================================
Why this script exists:
-------------------------------------------------------------------------
The original phase2_finetune used `global_fitness_model.keras` as the fine-tuning starting point,
but that global model's training set included 6 of the 8 subjects. In other words, for those 6 people the
"before fine-tuning" baseline was the score of a model that **had already seen this person** (just a different
time period), so the numbers were systematically overestimated by about +0.15 (S01 even +0.28).

Evidence: the only two subjects not in the training set, S03 and S06, had "before fine-tuning" numbers almost equal to
          LOSO (S03 0.680 vs LOSO 0.694; S06 0.778 vs LOSO 0.734), while the other 6 were clearly higher.

This version:
  For each subject S, train a model loso_without_{S}.keras whose training set excludes S entirely,
  and load it when phase2 fine-tunes S. Then "before fine-tuning" is a true zero-shot baseline, and
  "after − before" is a clean personalization gain that can go into the paper.

★ Scalers must be saved per fold too: each fold's scaler is fit only on that fold's training subjects;
  using the wrong scaler is another form of leakage.

Output (saved under models/ in this folder):
  loso_without_{S}.keras
  loso_scaler_ts_{S}.pkl / loso_scaler_static_{S}.pkl
  loso_zeroshot_metrics.csv   ← each subject's zero-shot r / MAE / nRMSE
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
                        make_windows, full_metrics)

# ==========================================
# settings (kept identical to Global_Training_FIXED.py to ensure comparability)
# ==========================================
DATA_DIR = COMBINED_DIR
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = LOSO_MODEL_DIR          # per-fold LOSO models + scalers
FEATURE_MODE = "spatial"
WINDOW_SIZE = 40
STEP_SIZE = 5
EPOCHS = 30
BATCH_SIZE = 32
RANDOM_STATE = 42

STATIC_COLS = ['Subj_Age', 'Subj_Height_m', 'Subj_Weight_norm', 'Subj_Gender']
LABEL_EMG_COLS = ['EMG_Main_MVC', 'EMG_Compass_MVC']


def load_one(fp):
    return pd.read_csv(fp)


def build_model(n_ts_feat, n_static):
    """Same architecture as Global_Training_FIXED.build_model (without the compensation branch)."""
    from tensorflow.keras.layers import (Input, Dense, Dropout, Conv1D,
                                         BatchNormalization, GlobalAveragePooling1D,
                                         Concatenate)
    input_ts = Input(shape=(WINDOW_SIZE, n_ts_feat), name='ts_input')
    input_static = Input(shape=(n_static,), name='static_input')

    x = Conv1D(64, 3, padding='causal', activation='relu', name='feature_conv1')(input_ts)
    x = BatchNormalization(name='feature_bn1')(x)
    x = Conv1D(64, 3, padding='causal', dilation_rate=2, activation='relu', name='feature_conv2')(x)
    ts_feat = GlobalAveragePooling1D(name='feature_pool')(x)

    fused = Concatenate(name='fusion_concat')([ts_feat, input_static])
    fused = Dense(64, activation='relu', name='fusion_dense')(fused)
    fused = Dropout(0.2, name='fusion_dropout')(fused)
    emg_branch = Dense(32, activation='relu', name='emg_dense')(fused)
    out_emg = Dense(2, activation='sigmoid', name='out_emg')(emg_branch)

    model = tf.keras.Model(inputs=[input_ts, input_static], outputs=[out_emg])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss={'out_emg': 'mse'}, loss_weights={'out_emg': 3.0},
                  metrics={'out_emg': 'mae'})
    return model


def main():
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    os.makedirs(MODEL_DIR, exist_ok=True)

    file_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*_Combined_Features.csv")))
    groups = [subject_id_from_path(p) for p in file_paths]
    subjects = sorted(set(groups))
    print(f"📂 files={len(file_paths)}  subjects={subjects}")

    ts_cols = build_ts_cols(load_one(file_paths[0]).columns.tolist(), FEATURE_MODE, False)
    print(f"🧩 FEATURE_MODE={FEATURE_MODE} | time-series features={len(ts_cols)}")

    rows = []
    for hold in subjects:
        tr_files = [p for p, g in zip(file_paths, groups) if g != hold]
        va_files = [p for p, g in zip(file_paths, groups) if g == hold]
        print(f"\n{'='*60}\n🔁 LOSO fold — holding out {hold}"
              f" (train {len(tr_files)} files / holdout {len(va_files)} files)\n{'='*60}")

        tr_dfs = [load_one(p) for p in tr_files]
        va_dfs = [load_one(p) for p in va_files]

        # ★ the scaler is fit only on this fold's training subjects
        big = pd.concat(tr_dfs, ignore_index=True)
        s_ts = StandardScaler().fit(big[ts_cols].values)
        s_st = StandardScaler().fit(big[STATIC_COLS].values)

        Xtr, Str, Ytr, _ = make_windows(tr_dfs, ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                        None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
        Xva, Sva, Yva, _ = make_windows(va_dfs, ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                        None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
        print(f"   training windows={len(Xtr)}  holdout windows={len(Xva)}")

        model = build_model(len(ts_cols), len(STATIC_COLS))
        model.fit({'ts_input': Xtr, 'static_input': Str}, {'out_emg': Ytr},
                  epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

        pred = model.predict({'ts_input': Xva, 'static_input': Sva}, verbose=0)
        pred = pred[0] if isinstance(pred, list) else pred
        m = full_metrics(Yva, pred)
        print(f"   [zero-shot] main r={m['r'][0]:.3f} MAE={m['mae'][0]:.3f} nRMSE={m['nrmse'][0]:.3f}")
        print(f"               synergist r={m['r'][1]:.3f} MAE={m['mae'][1]:.3f} nRMSE={m['nrmse'][1]:.3f}")

        model.save(os.path.join(MODEL_DIR, f'loso_without_{hold}.keras'))
        joblib.dump(s_ts, os.path.join(MODEL_DIR, f'loso_scaler_ts_{hold}.pkl'))
        joblib.dump(s_st, os.path.join(MODEL_DIR, f'loso_scaler_static_{hold}.pkl'))

        rows.append({
            'Subject': hold,
            'r_main': m['r'][0], 'r_syn': m['r'][1],
            'mae_main': m['mae'][0], 'mae_syn': m['mae'][1],
            'rmse_main': m['rmse'][0], 'rmse_syn': m['rmse'][1],
            'nrmse_main': m['nrmse'][0], 'nrmse_syn': m['nrmse'][1],
        })

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, 'loso_zeroshot_metrics.csv')
    df.to_csv(out, index=False, encoding='utf-8-sig')

    joblib.dump({'ts_cols': ts_cols, 'static_cols': STATIC_COLS,
                 'feature_mode': FEATURE_MODE}, os.path.join(MODEL_DIR, 'feature_spec.pkl'))

    print(f"\n{'='*60}\n📈 LOSO zero-shot mean")
    print(f"   main r={df['r_main'].mean():.3f}  MAE={df['mae_main'].mean():.3f}  nRMSE={df['nrmse_main'].mean():.3f}")
    print(f"   synergist r={df['r_syn'].mean():.3f}  MAE={df['mae_syn'].mean():.3f}  nRMSE={df['nrmse_syn'].mean():.3f}")
    print(f"✅ Models and scalers saved to {MODEL_DIR}")
    print(f"✅ Metrics saved to {out}")


if __name__ == '__main__':
    main()
