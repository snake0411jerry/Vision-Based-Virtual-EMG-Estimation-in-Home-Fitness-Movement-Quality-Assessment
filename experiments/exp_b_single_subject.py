"""
exp_b_single_subject.py — Experiment B: single-subject modelling (theoretical upper bound)
=========================================================================
Corresponds to Step 3, Experiment B of paper 035S_IC3MT2026:
  "Restricts training to a single subject to eliminate physiological
    confounders ... determine the theoretical upper bound of waveform tracking"

Method: for each subject, **train from scratch on that subject's own data only**, never looking at others.
Physiological confounders (subcutaneous fat, electrode placement, muscle recruitment strategy) are removed,
so the result is the upper bound "if we could train a separate model for every person".

★ The evaluation set is deliberately identical to `phase2_finetune_loso.py`
  (first 80% of each segment for training, last 20% for validation; split first, then window, windows never
  cross the boundary), so all three can be compared side by side:

    zero-shot LOSO     model has never seen this person              ← lower bound
    personalized FT    LOSO model + fine-tuning on a little of this person's data
    single subject     trained from scratch on this person's data    ← upper bound (this script)

⚠️ This is **not** a deployment scenario: in practice it is impossible to collect a full dataset for every new
   user and train on it. Its purpose is to bound "how much room personalization has"; the paper must not present
   it as achievable performance.

⚠️ Each person has very little data (7~11 segments), so training from scratch overfits easily.
   This script keeps the same architecture and epochs as LOSO for comparability, and additionally outputs
   the training-set score itself so the degree of overfitting can be judged.

Output: exp_b_single_subject_raw.csv / _summary.csv
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
import sys
import time

import numpy as np
import pandas as pd
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
from phase2_finetune_loso import VAL_TAIL_FRAC                               # noqa: E402


def predict(model, X, S):
    p = model.predict({'ts_input': X, 'static_input': S}, verbose=0)
    return p[0] if isinstance(p, list) else p


def main():
    ap = argparse.ArgumentParser(description="Experiment B: single-subject modelling")
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 1, 2, 3, 4])
    ap.add_argument("--feature-mode", default=FEATURE_MODE,
                    choices=['spatial', 'spatial+kinematic'])
    ap.add_argument("--prefix", default="exp_b_single_subject")
    args = ap.parse_args()

    file_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*_Combined_Features.csv")))
    groups = [subject_id_from_path(p) for p in file_paths]
    subjects = sorted(set(groups))
    dfs = {p: pd.read_csv(p) for p in file_paths}
    ts_cols = build_ts_cols(list(dfs[file_paths[0]].columns), args.feature_mode, False)
    print(f"📂 {len(subjects)} subjects / features {len(ts_cols)} dims ({args.feature_mode})")
    print(f"🔁 {len(args.seeds)} seeds; each person trained from scratch on their own")

    rows = []
    t0 = time.time()
    for si, seed in enumerate(args.seeds, 1):
        tf.keras.utils.set_random_seed(seed)
        for subj in subjects:
            own = [dfs[p] for p, g in zip(file_paths, groups) if g == subj]
            tr_parts, va_parts = [], []
            for d in own:
                a, b = temporal_split_df(d, VAL_TAIL_FRAC)
                tr_parts.append(a)
                va_parts.append(b)

            # ★ the scaler is fit only on this person's training segments, to avoid using validation-segment statistics
            big = pd.concat(tr_parts, ignore_index=True)
            s_ts = StandardScaler().fit(big[ts_cols].values)
            s_st = StandardScaler().fit(big[STATIC_COLS].values)

            Xtr, Str, Ytr, _ = make_windows(tr_parts, ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                            None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
            Xva, Sva, Yva, _ = make_windows(va_parts, ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                            None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
            if len(Xtr) == 0 or len(Xva) == 0:
                print(f"   ⚠️ {subj} has too few windows (train {len(Xtr)} / val {len(Xva)}), skipping")
                continue

            model = build_model(len(ts_cols), len(STATIC_COLS))
            model.fit({'ts_input': Xtr, 'static_input': Str}, {'out_emg': Ytr},
                      epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

            m_va = full_metrics(Yva, predict(model, Xva, Sva))
            m_tr = full_metrics(Ytr, predict(model, Xtr, Str))   # for overfitting diagnosis

            rows.append({
                'seed': seed, 'Subject': subj,
                'n_files': len(own), 'n_train_win': len(Xtr), 'n_val_win': len(Xva),
                'r_main': m_va['r'][0], 'r_syn': m_va['r'][1],
                'mae_main': m_va['mae'][0], 'mae_syn': m_va['mae'][1],
                'nrmse_main': m_va['nrmse'][0], 'nrmse_syn': m_va['nrmse'][1],
                'r_main_train': m_tr['r'][0], 'r_syn_train': m_tr['r'][1],
                'overfit_gap_main': m_tr['r'][0] - m_va['r'][0],
            })
            tf.keras.backend.clear_session()

        pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, f'{args.prefix}_raw.csv'),
                                  index=False, encoding='utf-8-sig')
        d = pd.DataFrame(rows)
        d = d[d.seed == seed]
        el = (time.time() - t0) / si
        print(f"[{si}/{len(args.seeds)}] seed={seed}  "
              f"main r={d.r_main.mean():.3f}  synergist r={d.r_syn.mean():.3f}  "
              f"(overfit gap {d.overfit_gap_main.mean():+.3f})  "
              f"({el/60:.1f} min/seed, about {el*(len(args.seeds)-si)/60:.0f} min left)", flush=True)

    df = pd.DataFrame(rows)
    per = df.groupby('seed')[['r_main', 'r_syn', 'nrmse_main', 'nrmse_syn',
                              'overfit_gap_main']].mean()
    per_ex = df[df.Subject != 'S09'].groupby('seed')[['r_main', 'r_syn']].mean()
    per.to_csv(os.path.join(RESULTS_DIR, f'{args.prefix}_summary.csv'), encoding='utf-8-sig')

    print("\n" + "=" * 80)
    print(f"Experiment B results (mean ± sd over {len(args.seeds)} seeds)")
    print("=" * 80)
    for c in per.columns:
        print(f"  {c:<18} {per[c].mean():.4f} ± {per[c].std(ddof=1):.4f}")
    print(f"  {'r_main (excl. S09)':<18} {per_ex.r_main.mean():.4f} ± {per_ex.r_main.std(ddof=1):.4f}")
    print(f"  {'r_syn  (excl. S09)':<18} {per_ex.r_syn.mean():.4f} ± {per_ex.r_syn.std(ddof=1):.4f}")

    print("\n  Per subject (mean across seeds):")
    g = df.groupby('Subject').agg(
        n_segments=('n_files', 'first'), train_windows=('n_train_win', 'first'), val_windows=('n_val_win', 'first'),
        r_main=('r_main', 'mean'), r_syn=('r_syn', 'mean'),
        train_r_main=('r_main_train', 'mean'), overfit_gap=('overfit_gap_main', 'mean'))
    print(g.round(3).to_string())

    print("\n  ⚠️ overfit gap = training-set r − validation-set r. With so little data per person a large value is expected;")
    print("     when the paper cites single-subject scores, it must also state the data volume and the overfitting risk.")
    print(f"\n✅ {args.prefix}_raw.csv / {args.prefix}_summary.csv")


if __name__ == '__main__':
    main()
