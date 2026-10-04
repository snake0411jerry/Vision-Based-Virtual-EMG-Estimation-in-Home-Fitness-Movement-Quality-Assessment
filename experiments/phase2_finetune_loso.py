"""
phase2_finetune_loso.py — leakage-free personalized fine-tuning
=========================================================================
Difference from the old phase2_finetune_FIXED.py (the only, but crucial, difference):
  old:  loads global_fitness_model.keras — its training set includes 6 of the 8 subjects,
        so for those 6 the "before fine-tuning" score comes from a model that has already seen them; the baseline is inflated by about +0.15.
  this: loads loso_without_{S}.keras — trained with subject S fully excluded, so
        "before fine-tuning" is a true zero-shot, and "after − before" is a clean personalization gain.

The rest keeps the old version's leakage-prevention design:
  each segment is split in time into the first 80% / last 20%; split first, then window; windows never cross the boundary;
  the validation part is a future the model has truly not seen.

Output: finetune_results.csv (r / MAE / nRMSE, before and after fine-tuning)
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
STEP_SIZE = 5
FT_EPOCHS = 20
FT_BATCH = 16
FT_LR = 1e-5

LABEL_EMG_COLS = ['EMG_Main_MVC', 'EMG_Compass_MVC']


def load_one(fp):
    return pd.read_csv(fp)


def finetune_one(subject, ts_cols, static_cols, seed=None, save_model=True, verbose=True):
    """Fine-tune a single subject.

    seed        : when set, fixes the dropout/shuffling draws (note: still not bit-reproducible on this machine's GPU,
                  see the <set_random_seed> section of 專案現況_交接用.md)
    save_model  : set False for multi-seed sweeps, to avoid overwriting the existing {subject}_personalized.keras
    """
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)
    files = sorted(glob.glob(os.path.join(DATA_DIR, f"{subject}_Seg_*_Combined_Features.csv")))
    if not files:
        print(f"⚠️ No data found for {subject}, skipping")
        return None

    base_path = os.path.join(MODEL_DIR, f'loso_without_{subject}.keras')
    if not os.path.exists(base_path):
        print(f"⚠️ {base_path} not found; run loso_train_and_save.py first")
        return None

    s_ts = joblib.load(os.path.join(MODEL_DIR, f'loso_scaler_ts_{subject}.pkl'))
    s_st = joblib.load(os.path.join(MODEL_DIR, f'loso_scaler_static_{subject}.pkl'))

    dfs = [load_one(f) for f in files]
    train_parts, val_parts = [], []
    for df in dfs:
        tr, va = temporal_split_df(df, VAL_TAIL_FRAC)
        train_parts.append(tr)
        val_parts.append(va)

    Xtr, Str, Ytr, _ = make_windows(train_parts, ts_cols, static_cols, LABEL_EMG_COLS,
                                    None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
    Xva, Sva, Yva, _ = make_windows(val_parts, ts_cols, static_cols, LABEL_EMG_COLS,
                                    None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
    if len(Xva) == 0:
        print(f"⚠️ {subject}: 0 validation windows, skipping")
        return None

    if verbose:
        print(f"\n===== {subject} ({len(files)} files | fine-tune windows {len(Xtr)} / validation windows {len(Xva)}) =====")

    model = tf.keras.models.load_model(base_path)

    # ---- before fine-tuning: true zero-shot baseline (the model has never seen this person) ----
    pred = model.predict({'ts_input': Xva, 'static_input': Sva}, verbose=0)
    pred = pred[0] if isinstance(pred, list) else pred
    before = full_metrics(Yva, pred)
    if verbose:
        print(f"  before (zero-shot)   main r={before['r'][0]:.3f} MAE={before['mae'][0]:.3f} nRMSE={before['nrmse'][0]:.3f}"
              f" | synergist r={before['r'][1]:.3f} MAE={before['mae'][1]:.3f} nRMSE={before['nrmse'][1]:.3f}")

    # ---- freeze the feature layers and fine-tune only the fusion/output layers ----
    for layer in model.layers:
        layer.trainable = ('feature' not in layer.name)
    model.compile(optimizer=tf.keras.optimizers.Adam(FT_LR),
                  loss={'out_emg': 'mse'}, loss_weights={'out_emg': 3.0},
                  metrics={'out_emg': 'mae'})

    model.fit({'ts_input': Xtr, 'static_input': Str}, {'out_emg': Ytr},
              epochs=FT_EPOCHS, batch_size=FT_BATCH,
              validation_data=({'ts_input': Xva, 'static_input': Sva}, {'out_emg': Yva}),
              callbacks=[tf.keras.callbacks.EarlyStopping(
                  monitor='val_loss', patience=5, restore_best_weights=True)],
              verbose=0)

    pred = model.predict({'ts_input': Xva, 'static_input': Sva}, verbose=0)
    pred = pred[0] if isinstance(pred, list) else pred
    after = full_metrics(Yva, pred)
    if verbose:
        print(f"  after (personalized) main r={after['r'][0]:.3f} MAE={after['mae'][0]:.3f} nRMSE={after['nrmse'][0]:.3f}"
              f" | synergist r={after['r'][1]:.3f} MAE={after['mae'][1]:.3f} nRMSE={after['nrmse'][1]:.3f}")
    if verbose:
        print(f"  ▶ gain  main Δr={after['r'][0]-before['r'][0]:+.3f}"
              f"  synergist Δr={after['r'][1]-before['r'][1]:+.3f}")

    if save_model:
        model.save(os.path.join(PERSONALIZED_FROM_LOSO_DIR, f'{subject}_personalized.keras'))

    row = {'Subject': subject, 'n_files': len(files),
           'n_ft_windows': len(Xtr), 'n_val_windows': len(Xva)}
    for tag, m in (('before', before), ('after', after)):
        for i, mu in enumerate(['main', 'syn']):
            row[f'r_{mu}_{tag}'] = m['r'][i]
            row[f'mae_{mu}_{tag}'] = m['mae'][i]
            row[f'nrmse_{mu}_{tag}'] = m['nrmse'][i]
    row['d_r_main'] = after['r'][0] - before['r'][0]
    row['d_r_syn'] = after['r'][1] - before['r'][1]
    row['d_nrmse_main'] = after['nrmse'][0] - before['nrmse'][0]
    row['d_nrmse_syn'] = after['nrmse'][1] - before['nrmse'][1]
    return row


def main():
    spec = joblib.load(os.path.join(MODEL_DIR, 'feature_spec.pkl'))
    ts_cols, static_cols = spec['ts_cols'], spec['static_cols']

    all_files = glob.glob(os.path.join(DATA_DIR, "*_Combined_Features.csv"))
    subjects = sorted(set(subject_id_from_path(p) for p in all_files))
    print(f"👥 Fine-tuning {len(subjects)} subjects in turn: {subjects}")

    rows = [r for s in subjects if (r := finetune_one(s, ts_cols, static_cols))]
    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, 'finetune_results.csv')
    df.to_csv(out, index=False, encoding='utf-8-sig')

    print(f"\n{'='*66}\n📈 Means (leakage-free baseline)")
    print(f"  main r: {df['r_main_before'].mean():.3f} → {df['r_main_after'].mean():.3f}"
          f"  (Δ{df['d_r_main'].mean():+.3f})")
    print(f"  synergist r: {df['r_syn_before'].mean():.3f} → {df['r_syn_after'].mean():.3f}"
          f"  (Δ{df['d_r_syn'].mean():+.3f})")
    print(f"  main nRMSE: {df['nrmse_main_before'].mean():.3f} → {df['nrmse_main_after'].mean():.3f}")
    print(f"  synergist nRMSE: {df['nrmse_syn_before'].mean():.3f} → {df['nrmse_syn_after'].mean():.3f}")
    print(f"✅ Saved {out}")


if __name__ == '__main__':
    main()
