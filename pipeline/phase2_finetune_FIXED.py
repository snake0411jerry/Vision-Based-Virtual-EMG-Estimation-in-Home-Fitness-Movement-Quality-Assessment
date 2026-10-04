"""
phase2_finetune_FIXED.py  —  corrected personalized fine-tuning
=========================================================================
Fixes relative to the original `phase2_finetune.py`:

[Leak-2] ★ Most important: the original first built all overlapping windows (step5/win40, 87.5% overlap)
         and then split them randomly with train_test_split(random_state) → nearly identical windows landed in both
         train and val, so val metrics were badly inflated (your paper's single-subject r≈0.70 very likely comes from this).
         This version "first splits each segment by time into the first 80% / last 20%, then windows each part";
         windows never cross the boundary, and val is a truly unseen future time period.

[Leak-1] Removed Load_1RM_Ratio (the ts_cols brought in from global_feature_spec.pkl already exclude it).
[Eval-8] Per-muscle Pearson r reported on the temporal held-out set (not just MAE any more).
[Consistency] ts_cols / static_cols are loaded from the spec saved by global training, guaranteeing dimensions match the global model.

[Batching] ★ The original fine-tuned only a single hard-coded SUBJECT_PREFIX="S01". This version by default scans all subject IDs
         present under DATA_DIR, fine-tuning and saving each one, so there is no need to edit the name and rerun each time.
         To fine-tune only specific subjects, set a list such as SUBJECT_PREFIXES = ["S01"].

Usage: run Global_Training_FIXED.py first (it produces the scaler and feature spec), then this file.
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
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from eval_utils import (subject_id_from_path, build_ts_cols, pearson_per_column,
                        make_windows, temporal_split_df, add_comp_labels)

# ==========================================
# 0. settings
# ==========================================
DATA_DIR = COMBINED_DIR
SUBJECT_PREFIXES = None          # None = automatically scan all subjects in DATA_DIR; can also be specified, e.g. ["S01", "S07"]
VAL_TAIL_FRAC = 0.2              # the last 20% of each segment's time is the held-out validation
WINDOW_SIZE = 40
STEP_SIZE = 5

STATIC_COLS_FALLBACK = ['Subj_Age', 'Subj_Height_m', 'Subj_Weight_norm', 'Subj_Gender']
LABEL_EMG_COLS = ['EMG_Main_MVC', 'EMG_Compass_MVC']
# compensation columns are for post-hoc diagnostics only and do not enter training (multi-task removed on 2026-08-04, see Global_Training_FIXED.py)


def load_one(fp):
    df = pd.read_csv(fp)
    # ★ thresholds are centralized in eval_utils.COMP_THRESHOLDS; do not hard-code them here (see the history lesson in the comment there)
    return add_comp_labels(df)


def discover_subjects(data_dir):
    all_files = glob.glob(os.path.join(data_dir, "*_Combined_Features.csv"))
    return sorted(set(subject_id_from_path(p) for p in all_files))


def finetune_subject(subject_prefix, scaler_ts, scaler_static, ts_cols, static_cols):
    files = sorted(glob.glob(os.path.join(DATA_DIR, f"{subject_prefix}_Seg_*_Combined_Features.csv")))
    if not files:
        print(f"⚠️ No personal data files found for {subject_prefix}; skipping.")
        return
    print(f"\n===== Personalized fine-tuning: {subject_prefix} (files={len(files)}) =====")

    dfs = [load_one(fp) for fp in files]

    # ---------------------------------------------------------------
    # ★ temporal split: split into earlier/later parts first, then window each (replacing the original random window split)
    # ---------------------------------------------------------------
    train_parts, val_parts = [], []
    for df in dfs:
        tr, va = temporal_split_df(df, VAL_TAIL_FRAC)
        train_parts.append(tr)
        val_parts.append(va)

    Xtr_ts, Xtr_st, Ytr_emg, _ = make_windows(
        train_parts, ts_cols, static_cols, LABEL_EMG_COLS, None,
        WINDOW_SIZE, STEP_SIZE, scaler_ts, scaler_static)
    Xva_ts, Xva_st, Yva_emg, _ = make_windows(
        val_parts, ts_cols, static_cols, LABEL_EMG_COLS, None,
        WINDOW_SIZE, STEP_SIZE, scaler_ts, scaler_static)
    print(f"✅ fine-tuning train windows={len(Xtr_ts)}  temporal held-out val windows={len(Xva_ts)}")
    if len(Xva_ts) == 0:
        print(f"⚠️ {subject_prefix}: 0 validation windows (segments too short); skipping.")
        return

    # ---- load the global model and freeze the feature layers ----
    model = tf.keras.models.load_model(os.path.join(GLOBAL_MODEL_DIR, 'global_fitness_model.keras'))
    for layer in model.layers:
        layer.trainable = ('feature' not in layer.name)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss={'out_emg': 'mse'}, loss_weights={'out_emg': 3.0},
                  metrics={'out_emg': 'mae'})

    # ---- baseline before fine-tuning (frozen global model inferring directly on held-out) ----
    print("📊 Before fine-tuning (global model, has not seen this person):")
    _report(model, Xva_ts, Xva_st, Yva_emg)

    model.fit({'ts_input': Xtr_ts, 'static_input': Xtr_st},
              {'out_emg': Ytr_emg},
              epochs=20, batch_size=16,
              validation_data=({'ts_input': Xva_ts, 'static_input': Xva_st},
                               {'out_emg': Yva_emg}),
              callbacks=[tf.keras.callbacks.EarlyStopping(
                  monitor='val_loss', patience=5, restore_best_weights=True)],
              verbose=1)

    print("📊 After fine-tuning (personalized, on temporal held-out):")
    _report(model, Xva_ts, Xva_st, Yva_emg)

    model.save(os.path.join(PERSONALIZED_FROM_GLOBAL_DIR, f'{subject_prefix}_personalized_fitness_model.keras'))
    print(f"✅ Saved {subject_prefix}_personalized_fitness_model.keras")
    print("💡 Comparing r “before vs after fine-tuning” gives the true gain of personalized calibration (leak-free).")


def main():
    # ---- load the global scaler and feature spec (guarantees consistent dimensions/columns, shared by all subjects) ----
    scaler_ts = joblib.load(os.path.join(GLOBAL_MODEL_DIR, 'global_scaler_ts.pkl'))
    scaler_static = joblib.load(os.path.join(GLOBAL_MODEL_DIR, 'global_scaler_static.pkl'))
    try:
        spec = joblib.load(os.path.join(GLOBAL_MODEL_DIR, 'global_feature_spec.pkl'))
        ts_cols = spec['ts_cols']
        static_cols = spec['static_cols']
        print(f"✅ Loaded feature spec: {len(ts_cols)} time-series features, mode={spec['feature_mode']}")
    except FileNotFoundError:
        print("⚠️ global_feature_spec.pkl not found; falling back to the default spatial setting")
        sample_fp = glob.glob(os.path.join(DATA_DIR, "*_Combined_Features.csv"))[0]
        ts_cols = build_ts_cols(load_one(sample_fp).columns.tolist(), "spatial", False)
        static_cols = STATIC_COLS_FALLBACK

    subjects = SUBJECT_PREFIXES if SUBJECT_PREFIXES else discover_subjects(DATA_DIR)
    print(f"👥 Fine-tuning {len(subjects)} subjects in turn: {subjects}")

    for subject_prefix in subjects:
        finetune_subject(subject_prefix, scaler_ts, scaler_static, ts_cols, static_cols)


def _report(model, Xts, Xst, Yemg):
    pred = model.predict({'ts_input': Xts, 'static_input': Xst}, verbose=0)
    pred_emg = pred[0] if isinstance(pred, list) else pred
    r = pearson_per_column(Yemg, pred_emg)
    mae = np.mean(np.abs(Yemg - pred_emg), axis=0)
    print(f"   Pearson r  main={r[0]:.3f}  synergist={r[1]:.3f} | "
          f"MAE main={mae[0]:.3f} synergist={mae[1]:.3f}")


if __name__ == '__main__':
    main()
