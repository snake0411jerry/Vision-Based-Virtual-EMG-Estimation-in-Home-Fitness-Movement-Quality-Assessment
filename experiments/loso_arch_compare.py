"""
loso_arch_compare.py — architecture comparison: current Conv1D vs the paper's TCN-Transformer
=========================================================================
Paper 035S_IC3MT2026 claims to use a Multi-Task TCN-Transformer, but all formal results in
exp_personalization so far come from `loso_train_and_save.build_model`
(Conv1D×2 + GAP + Dense, no Transformer). This script compares the two **on the same data within the same job**,
so the paper can decide honestly which architecture to report.

Parameters: baseline 22,306 / tcn_transformer 107,554 (about 4.8×).
At the scale of only 10 subjects and ~34k training windows, more capacity is not necessarily better.

Usage:
  python loso_arch_compare.py --archs baseline tcn_transformer --seeds 42 1 2

Output: arch_compare_raw.csv / arch_compare_summary.csv
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

from eval_utils import (subject_id_from_path, build_ts_cols,                 # noqa: E402
                        make_windows, full_metrics)
from loso_train_and_save import (FEATURE_MODE, WINDOW_SIZE, STEP_SIZE,       # noqa: E402
                                 EPOCHS, BATCH_SIZE, STATIC_COLS,
                                 LABEL_EMG_COLS, DATA_DIR)
from models_tcn_transformer import ARCHS                                     # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="LOSO architecture comparison")
    ap.add_argument("--archs", nargs="*", default=['baseline', 'tcn_transformer'],
                    choices=list(ARCHS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 1, 2])
    ap.add_argument("--feature-mode", default=FEATURE_MODE,
                    choices=['spatial', 'spatial+kinematic'])
    ap.add_argument("--prefix", default="arch_compare")
    args = ap.parse_args()

    file_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*_Combined_Features.csv")))
    groups = [subject_id_from_path(p) for p in file_paths]
    subjects = sorted(set(groups))
    dfs = {p: pd.read_csv(p) for p in file_paths}
    ts_cols = build_ts_cols(list(dfs[file_paths[0]].columns), args.feature_mode, False)
    print(f"📂 {len(file_paths)} files / {len(subjects)} subjects / features {len(ts_cols)} dims")
    print(f"🏗  architectures {args.archs} × {len(args.seeds)} seeds")
    for a in args.archs:
        print(f"    {a:<18} params {ARCHS[a](len(ts_cols), len(STATIC_COLS)).count_params():,}")
    tf.keras.backend.clear_session()

    rows = []
    t0 = time.time()
    total = len(args.archs) * len(args.seeds)
    done = 0
    for arch in args.archs:
        for seed in args.seeds:
            tf.keras.utils.set_random_seed(seed)
            for hold in subjects:
                tr_dfs = [dfs[p] for p, g in zip(file_paths, groups) if g != hold]
                va_dfs = [dfs[p] for p, g in zip(file_paths, groups) if g == hold]
                big = pd.concat(tr_dfs, ignore_index=True)
                s_ts = StandardScaler().fit(big[ts_cols].values)
                s_st = StandardScaler().fit(big[STATIC_COLS].values)

                Xtr, Str, Ytr, _ = make_windows(tr_dfs, ts_cols, STATIC_COLS,
                                                LABEL_EMG_COLS, None, WINDOW_SIZE,
                                                STEP_SIZE, s_ts, s_st)
                Xva, Sva, Yva, _ = make_windows(va_dfs, ts_cols, STATIC_COLS,
                                                LABEL_EMG_COLS, None, WINDOW_SIZE,
                                                STEP_SIZE, s_ts, s_st)

                model = ARCHS[arch](len(ts_cols), len(STATIC_COLS))
                model.fit({'ts_input': Xtr, 'static_input': Str}, {'out_emg': Ytr},
                          epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
                pred = model.predict({'ts_input': Xva, 'static_input': Sva}, verbose=0)
                pred = pred[0] if isinstance(pred, list) else pred
                m = full_metrics(Yva, pred)
                rows.append({'arch': arch, 'seed': seed, 'Subject': hold,
                             'r_main': m['r'][0], 'r_syn': m['r'][1],
                             'nrmse_main': m['nrmse'][0], 'nrmse_syn': m['nrmse'][1]})
                tf.keras.backend.clear_session()

            pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, f'{args.prefix}_raw.csv'),
                                      index=False, encoding='utf-8-sig')
            done += 1
            d = pd.DataFrame(rows)
            d = d[(d.arch == arch) & (d.seed == seed)]
            el = (time.time() - t0) / done
            print(f"[{done}/{total}] {arch} seed={seed}  "
                  f"main r={d.r_main.mean():.3f}  synergist r={d.r_syn.mean():.3f}  "
                  f"({el/60:.1f} min/round, about {el*(total-done)/60:.0f} min left)", flush=True)

    df = pd.DataFrame(rows)
    per = df.groupby(['arch', 'seed'])[['r_main', 'r_syn', 'nrmse_main',
                                        'nrmse_syn']].mean().reset_index()
    per.to_csv(os.path.join(RESULTS_DIR, f'{args.prefix}_summary.csv'),
               index=False, encoding='utf-8-sig')

    print("\n" + "=" * 82)
    print(f"Architecture comparison (mean ± sd over {len(args.seeds)} seeds)")
    print("=" * 82)
    for c in ['r_main', 'r_syn', 'nrmse_main', 'nrmse_syn']:
        line = f"  {c:<12}"
        for a in args.archs:
            g = per[per.arch == a][c]
            line += f"  {a}: {g.mean():.4f}±{g.std(ddof=1):.4f}"
        print(line)

    if len(args.archs) == 2:
        from scipy import stats
        print("\n  Paired t-tests (latter − former; same job, both arms share the same seeds):")
        a0 = per[per.arch == args.archs[0]].set_index('seed')
        b0 = per[per.arch == args.archs[1]].set_index('seed')
        shared = a0.index.intersection(b0.index)
        for c in ['r_main', 'r_syn', 'nrmse_main', 'nrmse_syn']:
            a, b = a0.loc[shared, c], b0.loc[shared, c]
            t, p = stats.ttest_rel(b, a)
            print(f"    {c:<12} diff {b.mean()-a.mean():+.4f}   "
                  f"t={t:+.2f}  p={p:.3f}  (n={len(shared)} seed)")
        print(f"\n  ⚠️ n={len(args.seeds)} seeds, limited power; when p is not significant "
              f"we can only say “no evidence of a difference”, not that the two are equivalent.")

    print(f"\n✅ {args.prefix}_raw.csv / {args.prefix}_summary.csv")


if __name__ == '__main__':
    main()
