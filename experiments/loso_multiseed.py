"""
loso_multiseed.py — multi-seed LOSO, quantifying fold-to-fold noise and comparing before/after the axis fix
=========================================================================
Why this script exists:
-------------------------------------------------------------------------
After the coordinate-axis fix on 2026-07-30, a single LOSO run gave main r 0.706 → 0.689 and
synergist 0.405 → 0.380, which looked like a regression. But the per-subject swings (up to ±0.156) are far
larger than the mean difference (0.02), and the paired t-tests gave p=0.45/0.47 — i.e. **a single run with one
seed simply cannot resolve differences of order 0.02**.

This script runs a full 10-fold LOSO with N seeds on both the "before" and "after" datasets and uses the
between-seed variation as the noise scale, answering two questions:
  1. Does the axis fix have any effect at all? (comparison of cross-seed means)
  2. How large is the fold/seed noise? (the yardstick for every future "changing X improved it by Y" claim)

★ No models are saved (metrics only); results are written after each seed, so an interruption still leaves partial results.

⚠️ The "after" set contains two changes at once:
   (a) the coordinate-axis fix (Trunk_Lean_Angle, Knee_Angle) — affects all 89 segments
   (b) a 1.383 s head trim of S09 b.trc (the EMG device was pressed twice by mistake) — affects 1 segment
   They cannot be separated in this experiment and must be mentioned together when cited.

Output:
  loso_multiseed_raw.csv       full metrics for each (arm, seed, subject)
  loso_multiseed_summary.csv   10-subject mean for each (arm, seed)
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
import glob
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
from eval_utils import (subject_id_from_path, build_ts_cols,      # noqa: E402
                        make_windows, full_metrics)
from loso_train_and_save import (build_model, FEATURE_MODE, WINDOW_SIZE,   # noqa: E402
                                 STEP_SIZE, EPOCHS, BATCH_SIZE,
                                 STATIC_COLS, LABEL_EMG_COLS)

DATASET = DATASET_DIR
# default: before vs after the coordinate-axis fix (the 2026-07-30 experiment)
DEFAULT_ARMS = {
    'before': os.path.join(DATASET, 'Combined_before_axisfix'),
    'after': os.path.join(DATASET, 'Combined'),
}
DEFAULT_SEEDS = [42, 1, 2, 3, 4]

ap = argparse.ArgumentParser(description="Multi-seed LOSO comparison")
ap.add_argument("--arms", nargs="*", default=None,
                help="Arms, format name=folder_path[@feature_mode][~col1,col2,...] (multiple allowed). "
                     "List the feature columns to keep after ~, for feature-subset ablation. "
                     "If omitted, the default before/after are used. Relative paths are appended to Dataset/. "
                     "feature_mode can be spatial (default) or spatial+kinematic, "
                     "e.g. kinematic=Combined@spatial+kinematic")
ap.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
ap.add_argument("--prefix", default="loso_multiseed",
                help="Output file-name prefix, e.g. --prefix loso_shoulderx")
ap.add_argument("--order", choices=["seed", "arm"], default="seed",
                help="Execution order. seed (default) = finish every arm for one seed before moving to the next seed; "
                     "arm = finish every seed for one arm before moving to the next arm. "
                     "**It only affects what you have if you stop midway; it does not affect any number** "
                     "(every fold-set calls set_random_seed at its start). "
                     "seed is the default because long jobs like this are often interrupted, and seed-outer guarantees "
                     "every arm has the same number of seeds at any interruption point — asymmetric partial results "
                     "cannot be used for paired tests and would be wasted.")
ap.add_argument("--resume", action="store_true",
                help="Resume from the existing <prefix>_raw.csv: completed (arm, seed) pairs are skipped. "
                     "Used to split one big job over several runs so all arms end up in the same CSV. "
                     "⚠️ GPU nondeterminism still differs between runs; see the comments in main() below.")
args = ap.parse_args()

if args.arms:
    ARMS = {}
    for spec in args.arms:
        name, _, rest = spec.partition('=')
        if not rest:
            ap.error(f"--arms format should be name=path[@feature_mode], got: {spec}")
        rest, _, keep = rest.partition('~')
        path, _, mode = rest.partition('@')
        mode = mode or FEATURE_MODE
        if mode not in ('spatial', 'spatial+kinematic'):
            ap.error(f"Unknown feature_mode: {mode}")
        keep = [c.strip() for c in keep.split(',') if c.strip()] or None
        ARMS[name] = (path if os.path.isabs(path) else os.path.join(DATASET, path), mode, keep)
else:
    ARMS = {k: (v, FEATURE_MODE, None) for k, v in DEFAULT_ARMS.items()}
SEEDS = args.seeds

RAW_OUT = os.path.join(RESULTS_DIR, f'{args.prefix}_raw.csv')
SUM_OUT = os.path.join(RESULTS_DIR, f'{args.prefix}_summary.csv')


def run_fold_set(data_dir, seed, feature_mode=FEATURE_MODE, keep=None):
    """Run one full 10-fold LOSO on the given folder and return per-subject metrics."""
    tf.keras.utils.set_random_seed(seed)

    file_paths = sorted(glob.glob(os.path.join(data_dir, "*_Combined_Features.csv")))
    groups = [subject_id_from_path(p) for p in file_paths]
    subjects = sorted(set(groups))
    dfs = {p: pd.read_csv(p) for p in file_paths}
    ts_cols = build_ts_cols(list(dfs[file_paths[0]].columns), feature_mode, False)
    if keep:
        # keep only the specified columns (in their original order) for feature-subset ablation; a non-existent column is an immediate error
        miss = [c for c in keep if c not in ts_cols]
        if miss:
            raise SystemExit(f"Columns given after ~ are not among the features: {miss}")
        ts_cols = [c for c in ts_cols if c in keep]

    rows = []
    for hold in subjects:
        tr_dfs = [dfs[p] for p, g in zip(file_paths, groups) if g != hold]
        va_dfs = [dfs[p] for p, g in zip(file_paths, groups) if g == hold]

        big = pd.concat(tr_dfs, ignore_index=True)
        s_ts = StandardScaler().fit(big[ts_cols].values)
        s_st = StandardScaler().fit(big[STATIC_COLS].values)

        Xtr, Str, Ytr, _ = make_windows(tr_dfs, ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                        None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
        Xva, Sva, Yva, _ = make_windows(va_dfs, ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                        None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)

        model = build_model(len(ts_cols), len(STATIC_COLS))
        model.fit({'ts_input': Xtr, 'static_input': Str}, {'out_emg': Ytr},
                  epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        pred = model.predict({'ts_input': Xva, 'static_input': Sva}, verbose=0)
        pred = pred[0] if isinstance(pred, list) else pred
        m = full_metrics(Yva, pred)

        rows.append({'Subject': hold,
                     'r_main': m['r'][0], 'r_syn': m['r'][1],
                     'mae_main': m['mae'][0], 'mae_syn': m['mae'][1],
                     'rmse_main': m['rmse'][0], 'rmse_syn': m['rmse'][1],
                     'nrmse_main': m['nrmse'][0], 'nrmse_syn': m['nrmse'][1]})
        tf.keras.backend.clear_session()
    out = pd.DataFrame(rows)
    out.attrs['n_ts_feat'] = len(ts_cols)
    return out


def main():
    # ---- resume (--resume) ----
    # Why: the full 8 arms x 3 seeds = 24 fold-sets take more than 8 hours in one go,
    # and the machine often has to do other things midway. Running in two parts but writing to the same CSV
    # at least lets "arms within the same file" be handled by one analysis script.
    #
    # 🔴 Honest limitation: a resume is **a new process**, so arms in different chunks still carry the
    #    "cross-run dispersion" recorded in the handover doc (≈0.025, larger than the within-run between-seed sd of 0.013–0.017).
    #    Resuming only solves "scattered files"; it **does not solve "not subtractable"**.
    #    Arms that really need to be subtracted should run together within one chunk.
    #    analyze_mediapipe_experiment.py prints each arm's chunk number as a reminder of this.
    all_raw, all_sum = [], []
    done_keys = set()
    if args.resume and os.path.exists(RAW_OUT):
        prev = pd.read_csv(RAW_OUT)
        if 'chunk' not in prev.columns:
            prev['chunk'] = 1
        all_raw.append(prev)
        done_keys = {(str(a), int(sd)) for a, sd in
                     prev[['arm', 'seed']].drop_duplicates().values}
        if os.path.exists(SUM_OUT):
            psum = pd.read_csv(SUM_OUT)
            if 'chunk' not in psum.columns:
                psum['chunk'] = 1
            all_sum = psum.to_dict('records')
        print(f"--resume: {RAW_OUT}")
        print(f"  {len(done_keys)} (arm, seed) pairs already done, will be skipped: "
              f"{sorted(done_keys)}")
    chunk = (max((int(r.get('chunk', 1)) for r in all_sum), default=0) + 1
             if all_sum else 1)

    # the execution order only affects "what you have if you stop midway", not any number
    # (run_fold_set() calls set_random_seed(seed) at its start; every fold-set is seeded independently).
    if args.order == 'seed':
        pending = [(a, sd) for sd in SEEDS for a in ARMS]
    else:
        pending = [(a, sd) for a in ARMS for sd in SEEDS]
    pending = [(a, sd) for a, sd in pending if (a, int(sd)) not in done_keys]
    skipped = [a for a, (d, _, _) in ARMS.items() if not os.path.isdir(d)]
    for a in skipped:
        print(f"⚠️ {ARMS[a][0]} not found, skipping arm={a}")
    pending = [(a, sd) for a, sd in pending if a not in skipped]
    total = len(pending)
    if total == 0:
        print("No (arm, seed) left to run — everything is already in the existing CSV.")
        return
    print(f"{total} fold-sets to run (this is chunk {chunk})")
    print()
    done = 0
    t_start = time.time()

    for arm, seed in pending:
        data_dir, mode, keep = ARMS[arm]
        t0 = time.time()
        df = run_fold_set(data_dir, seed, mode, keep)
        df.insert(0, 'seed', seed)
        df.insert(0, 'arm', arm)
        df['chunk'] = chunk
        all_raw.append(df)

        s = {'arm': arm, 'seed': seed, 'feature_mode': mode,
             'n_ts_feat': df.attrs.get('n_ts_feat'), 'chunk': chunk}
        for c in ['r_main', 'r_syn', 'nrmse_main', 'nrmse_syn']:
            s[c] = df[c].mean()
            s[c + '_ex_s09'] = df[df['Subject'] != 'S09'][c].mean()
        all_sum.append(s)

        pd.concat(all_raw, ignore_index=True).to_csv(RAW_OUT, index=False,
                                                     encoding='utf-8-sig')
        pd.DataFrame(all_sum).to_csv(SUM_OUT, index=False, encoding='utf-8-sig')

        done += 1
        el = time.time() - t0
        eta = (time.time() - t_start) / done * (total - done) / 60
        print(f"[{done}/{total}] arm={arm}({mode}, {s['n_ts_feat']} dims) seed={seed}  "
              f"main r={s['r_main']:.3f}  synergist r={s['r_syn']:.3f}  "
              f"({el/60:.1f} min, about {eta:.0f} min left)", flush=True)

    print("\n" + "=" * 74)
    print("Summary across seeds (mean ± sd, n =", len(SEEDS), "seeds)")
    print("=" * 74)
    sm = pd.DataFrame(all_sum)
    names = [a for a in ARMS if (sm['arm'] == a).any()]
    for c in ['r_main', 'r_syn', 'nrmse_main', 'nrmse_syn']:
        line = f"  {c:<12}"
        for arm in names:
            g = sm[sm['arm'] == arm][c]
            line += f"  {arm}: {g.mean():.3f}±{g.std(ddof=1):.3f}"
        if len(names) == 2:   # with two arms, report the difference directly (latter − former)
            d = sm[sm['arm'] == names[1]][c].mean() - sm[sm['arm'] == names[0]][c].mean()
            line += f"   |  diff {d:+.3f}"
        print(line)

    print(f"\n✅ {RAW_OUT}")
    print(f"✅ {SUM_OUT}")


if __name__ == '__main__':
    main()
