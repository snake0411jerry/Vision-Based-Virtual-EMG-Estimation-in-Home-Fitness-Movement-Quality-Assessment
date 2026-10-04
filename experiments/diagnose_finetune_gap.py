"""
diagnose_finetune_gap.py — explain the gap between the fine-tuning numbers 0.832 and 0.814
=========================================================================
Background:
  pipeline_variance.py (5 full runs, base model fine-tuned directly in memory) → 0.832±0.004
  phase2_finetune_multiseed.py (loads the base saved under models/)          → 0.814±0.000
  The difference is about 5 sd, cause unknown.

This script measures three arms **within the same job**, ruling out cross-session interference:

  arm=inmem     after training each fold, fine-tune the in-memory model directly
  arm=reload    save that fold's weights, reload them, then fine-tune
                (identical starting weights to inmem — saved before any fine-tuning)
  arm=existing  load models/loso_without_*.keras (output of loso_train_and_save.py)

Reading:
  inmem ≠ reload            → the problem is in the save/load path
  inmem ≈ reload ≠ existing → the problem is the base model itself,
                              i.e. the training loop in loso_train_and_save.py differs materially from this one
                              (known candidate: that script does not call clear_session() between folds)
  all three similar         → the earlier gap was a sampling coincidence; re-evaluate with more seeds

Output: finetune_gap_raw.csv / finetune_gap_summary.csv
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


import argparse
import glob
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
FIXED_DIR = PIPELINE_DIR
sys.path.insert(0, FIXED_DIR)
sys.path.insert(0, HERE)

from eval_utils import (subject_id_from_path, build_ts_cols, make_windows,   # noqa: E402
                        temporal_split_df, full_metrics)
from loso_train_and_save import (build_model, FEATURE_MODE, WINDOW_SIZE,     # noqa: E402
                                 STEP_SIZE, EPOCHS, BATCH_SIZE,
                                 STATIC_COLS, LABEL_EMG_COLS, DATA_DIR)
from phase2_finetune_loso import (VAL_TAIL_FRAC, FT_EPOCHS, FT_BATCH,        # noqa: E402
                                  FT_LR, MODEL_DIR)


def predict(model, X, S):
    p = model.predict({'ts_input': X, 'static_input': S}, verbose=0)
    return p[0] if isinstance(p, list) else p


def finetune(model, Xf, Sf, Yf, Xv, Sv, Yv):
    """Use the phase2 fine-tuning protocol and return (before, after)."""
    before = full_metrics(Yv, predict(model, Xv, Sv))
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
    after = full_metrics(Yv, predict(model, Xv, Sv))
    return before, after


def row_of(arm, seed, subject, before, after):
    return {'arm': arm, 'seed': seed, 'Subject': subject,
            'r_main_before': before['r'][0], 'r_main_after': after['r'][0],
            'r_syn_before': before['r'][1], 'r_syn_after': after['r'][1],
            'd_r_main': after['r'][0] - before['r'][0],
            'd_r_syn': after['r'][1] - before['r'][1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 1, 2])
    ap.add_argument("--prefix", default="finetune_gap")
    args = ap.parse_args()

    tmpdir = tempfile.mkdtemp(prefix='ftgap_')
    print(f"🗂  Base models cached in {tmpdir}")

    file_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*_Combined_Features.csv")))
    groups = [subject_id_from_path(p) for p in file_paths]
    subjects = sorted(set(groups))
    dfs = {p: pd.read_csv(p) for p in file_paths}
    ts_cols = build_ts_cols(list(dfs[file_paths[0]].columns), FEATURE_MODE, False)

    rows = []
    t0 = time.time()

    # ---------- arm=existing: use the existing base models under models/ (run once, seed-independent) ----------
    spec = joblib.load(os.path.join(MODEL_DIR, 'feature_spec.pkl'))
    ex_ts_cols = spec['ts_cols']
    print(f"\n▶ arm=existing (load the existing base from models/, ts_cols={len(ex_ts_cols)})")
    for hold in subjects:
        base = os.path.join(MODEL_DIR, f'loso_without_{hold}.keras')
        if not os.path.exists(base):
            print(f"   ⚠️ missing {base}, skipped")
            continue
        s_ts = joblib.load(os.path.join(MODEL_DIR, f'loso_scaler_ts_{hold}.pkl'))
        s_st = joblib.load(os.path.join(MODEL_DIR, f'loso_scaler_static_{hold}.pkl'))
        va_dfs = [dfs[p] for p, g in zip(file_paths, groups) if g == hold]
        tr_p, va_p = zip(*(temporal_split_df(d, VAL_TAIL_FRAC) for d in va_dfs))
        Xf, Sf, Yf, _ = make_windows(list(tr_p), ex_ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                     None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
        Xv, Sv, Yv, _ = make_windows(list(va_p), ex_ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                     None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
        tf.keras.utils.set_random_seed(42)
        b, a = finetune(tf.keras.models.load_model(base), Xf, Sf, Yf, Xv, Sv, Yv)
        rows.append(row_of('existing', 42, hold, b, a))
        tf.keras.backend.clear_session()
    print(f"   done ({(time.time()-t0)/60:.1f} min)")

    # ---------- arm=inmem / reload ----------
    for seed in args.seeds:
        tf.keras.utils.set_random_seed(seed)
        print(f"\n▶ seed={seed}: train 10 folds, compare inmem vs reload in each fold")
        for hold in subjects:
            tr_dfs = [dfs[p] for p, g in zip(file_paths, groups) if g != hold]
            va_dfs = [dfs[p] for p, g in zip(file_paths, groups) if g == hold]
            big = pd.concat(tr_dfs, ignore_index=True)
            s_ts = StandardScaler().fit(big[ts_cols].values)
            s_st = StandardScaler().fit(big[STATIC_COLS].values)

            Xtr, Str, Ytr, _ = make_windows(tr_dfs, ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                            None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
            model = build_model(len(ts_cols), len(STATIC_COLS))
            model.fit({'ts_input': Xtr, 'static_input': Str}, {'out_emg': Ytr},
                      epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

            # ★ save before any fine-tuning so both arms start from exactly the same weights
            base_path = os.path.join(tmpdir, f'base_{seed}_{hold}.keras')
            model.save(base_path)

            tr_p, va_p = zip(*(temporal_split_df(d, VAL_TAIL_FRAC) for d in va_dfs))
            Xf, Sf, Yf, _ = make_windows(list(tr_p), ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                         None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
            Xv, Sv, Yv, _ = make_windows(list(va_p), ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                         None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)

            b1, a1 = finetune(model, Xf, Sf, Yf, Xv, Sv, Yv)
            rows.append(row_of('inmem', seed, hold, b1, a1))

            b2, a2 = finetune(tf.keras.models.load_model(base_path),
                              Xf, Sf, Yf, Xv, Sv, Yv)
            rows.append(row_of('reload', seed, hold, b2, a2))

            os.remove(base_path)
            tf.keras.backend.clear_session()

        pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, f'{args.prefix}_raw.csv'),
                                  index=False, encoding='utf-8-sig')
        print(f"   done (cumulative {(time.time()-t0)/60:.1f} min)")

    shutil.rmtree(tmpdir, ignore_errors=True)

    # ---------- summary ----------
    df = pd.DataFrame(rows)
    per = df.groupby(['arm', 'seed'])[['r_main_before', 'r_main_after',
                                       'r_syn_after', 'd_r_main']].mean().reset_index()
    per.to_csv(os.path.join(RESULTS_DIR, f'{args.prefix}_summary.csv'),
               index=False, encoding='utf-8-sig')

    print("\n" + "=" * 84)
    print("Comparison of the three arms (mean over 10 subjects)")
    print("=" * 84)
    for arm in ['existing', 'inmem', 'reload']:
        g = per[per.arm == arm]
        if not len(g):
            continue
        sd = f"±{g.r_main_after.std(ddof=1):.4f}" if len(g) > 1 else ""
        print(f"  {arm:<9} n={len(g)}  before {g.r_main_before.mean():.4f}  "
              f"after {g.r_main_after.mean():.4f}{sd}  "
              f"synergist after {g.r_syn_after.mean():.4f}  Δ {g.d_r_main.mean():+.4f}")

    print("\n" + "-" * 84)
    print("Paired comparison inmem vs reload (same fold, same starting weights)")
    print("-" * 84)
    a = df[df.arm == 'inmem'].set_index(['seed', 'Subject'])
    b = df[df.arm == 'reload'].set_index(['seed', 'Subject'])
    idx = a.index.intersection(b.index)
    if len(idx):
        d_before = (a.loc[idx, 'r_main_before'] - b.loc[idx, 'r_main_before'])
        d_after = (a.loc[idx, 'r_main_after'] - b.loc[idx, 'r_main_after'])
        print(f"  difference before fine-tuning (should be 0, otherwise save/load already changed the model): "
              f"mean {d_before.mean():+.6f}  max|d| {d_before.abs().max():.6f}")
        print(f"  difference after fine-tuning: mean {d_after.mean():+.4f}  sd {d_after.std(ddof=1):.4f}  "
              f"max|d| {d_after.abs().max():.4f}")

    print("\nReading:")
    print("  inmem ≠ reload            -> save/load path is broken")
    print("  inmem ≈ reload ≠ existing -> the base model itself differs (training loop in loso_train_and_save differs)")
    print("  all three similar         -> the earlier gap was a sampling coincidence; re-evaluate with more seeds")
    print(f"\n✅ {args.prefix}_raw.csv / {args.prefix}_summary.csv")


if __name__ == '__main__':
    main()
