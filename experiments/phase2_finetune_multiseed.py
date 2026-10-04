"""
phase2_finetune_multiseed.py — multi-seed stability of personalized fine-tuning
=========================================================================
Why:
-------------------------------------------------------------------------
`finetune_results.csv` is the result of a **single run**. Even with a fixed seed this machine's GPU is known not to be
bit-reproducible (see <set_random_seed> in 專案現況_交接用.md),
so a claim like "personalization gain Δr = +0.033" likewise needs multiple runs to stand.

This script fine-tunes the same set of LOSO base models with N seeds, quantifying the stability of Δ.

⚠️ **This experiment only covers noise from the "fine-tuning stage".**
   The base models `loso_without_{S}.keras` are fixed (not retrained),
   so the sd here **underestimates** the total variation of the whole pipeline.
   Measuring the total requires retraining LOSO N times as well, which costs much more.

⚠️ Does not overwrite `{subject}_personalized.keras` — those models correspond to
   the single run in `finetune_results.csv` and are kept consistent.

Output: phase2_multiseed_raw.csv / phase2_multiseed_summary.csv
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
import os
import sys
import time

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase2_finetune_loso import (finetune_one, MODEL_DIR, DATA_DIR)  # noqa: E402

sys.path.insert(0, PIPELINE_DIR)
from eval_utils import subject_id_from_path  # noqa: E402

import glob  # noqa: E402

METRICS = ['r_main_before', 'r_main_after', 'r_syn_before', 'r_syn_after',
           'nrmse_main_before', 'nrmse_main_after',
           'nrmse_syn_before', 'nrmse_syn_after',
           'd_r_main', 'd_r_syn', 'd_nrmse_main', 'd_nrmse_syn']


def main():
    ap = argparse.ArgumentParser(description="Multi-seed stability of personalized fine-tuning")
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 1, 2, 3, 4])
    ap.add_argument("--prefix", default="phase2_multiseed")
    args = ap.parse_args()

    spec = joblib.load(os.path.join(MODEL_DIR, 'feature_spec.pkl'))
    ts_cols, static_cols = spec['ts_cols'], spec['static_cols']
    subjects = sorted(set(subject_id_from_path(p) for p in
                          glob.glob(os.path.join(DATA_DIR, "*_Combined_Features.csv"))))
    print(f"👥 {len(subjects)} subjects × {len(args.seeds)} seeds (models not saved)")

    raw, summ = [], []
    t0 = time.time()
    for si, seed in enumerate(args.seeds, 1):
        rows = []
        for s in subjects:
            r = finetune_one(s, ts_cols, static_cols, seed=seed,
                             save_model=False, verbose=False)
            if r:
                rows.append(r)
            tf.keras.backend.clear_session()
        df = pd.DataFrame(rows)
        df.insert(0, 'seed', seed)
        raw.append(df)

        rec = {'seed': seed}
        for c in METRICS:
            rec[c] = df[c].mean()
            rec[c + '_ex_s09'] = df[df.Subject != 'S09'][c].mean()
        summ.append(rec)

        pd.concat(raw, ignore_index=True).to_csv(
            os.path.join(RESULTS_DIR, f'{args.prefix}_raw.csv'), index=False, encoding='utf-8-sig')
        pd.DataFrame(summ).to_csv(
            os.path.join(RESULTS_DIR, f'{args.prefix}_summary.csv'), index=False, encoding='utf-8-sig')

        el = (time.time() - t0) / si
        print(f"[{si}/{len(args.seeds)}] seed={seed}  "
              f"main {rec['r_main_before']:.3f}→{rec['r_main_after']:.3f} (Δ{rec['d_r_main']:+.3f})  "
              f"synergist {rec['r_syn_before']:.3f}→{rec['r_syn_after']:.3f} (Δ{rec['d_r_syn']:+.3f})  "
              f"({el/60:.1f} min/seed, about {el*(len(args.seeds)-si)/60:.0f} min left)", flush=True)

    sm = pd.DataFrame(summ)
    print("\n" + "=" * 88)
    print(f"Summary across seeds (mean ± sd, n={len(args.seeds)})")
    print("=" * 88)
    for c in METRICS:
        print(f"  {c:<20} all 10 {sm[c].mean():.4f}±{sm[c].std(ddof=1):.4f}"
              f"    excl. S09 {sm[c+'_ex_s09'].mean():.4f}±{sm[c+'_ex_s09'].std(ddof=1):.4f}")

    print("\n  Note: *_before is the base model's deterministic prediction and in theory does not vary with the seed;")
    print("      if its sd is clearly >0, GPU nondeterminism is affecting the inference stage too.")
    print(f"\n✅ {args.prefix}_raw.csv / {args.prefix}_summary.csv")


if __name__ == '__main__':
    main()
