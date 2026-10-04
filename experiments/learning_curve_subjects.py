"""
learning_curve_subjects.py — subject-count learning curve: is data the bottleneck?
=========================================================================
[Why run this]
-------------------------------------------------------------------------
Three independent lines of evidence point to "limited by data scale, not architecture":
  ① GBT ties or even beats the deep models (nRMSE)
  ② all three distillation designs are ineffective
  ③ the feature-dimension ablation shows no difference (p=0.989)

But "not enough data" has so far been an **inference**, not a **measurement**. This script treats
"how many people are in the training set" as the independent variable and plots performance against it, answering:

  - Is the curve still rising at N=9? → still rising = recruiting more people helps
  - Roughly where would it reach at N=30 by extrapolation? → decides whether to invest in more recruitment
  - Has the curve already flattened? → stop recruiting; spend the money on the monocular gap and the task ceiling

⚠️ This is **not** `calibration_curve.py`. That one asks "how many sets should one person record"
   (amount of personal calibration data), with k sets as the independent variable; this one asks "how many people",
   with N people as the independent variable. They answer completely different questions.

[Definition of the estimator (important)]
-------------------------------------------------------------------------
For each subject s (all 10 take turns as the test subject):
    randomly draw N people from the other 9 → train → evaluate on s
then average over the 10 subjects.

★ The benefit of this definition: **at N=9 it reduces to standard LOSO**, directly comparable with the existing
  0.72~0.74. The script performs this anchor check automatically when it finishes — if it does not match, the
  implementation differs and the curve cannot be trusted.

★ N=9 has only one possible draw (all other 9), so repeats automatically drops to 1.

[Methodology (follows the project rules; do not change)]
-------------------------------------------------------------------------
1. Folds are always split by **subject**; sliding windows never cross segment boundaries.
2. The scaler is fit only on the training folds' **frames**, then applied to the windows.
   (Windows are pre-built raw windows that are affine-scaled afterwards — equivalent to scaling first and windowing
    afterwards, but the windows only need to be built once, saving per-fold recomputation.)
3. r and nRMSE are reported together; the synergist is additionally reported excluding S09.
4. All N run within **the same job**, because cross-run dispersion (≈0.025) is larger than the within-run
   between-seed sd (0.013~0.017), and cross-run numbers must not be subtracted.

[Usage]
-------------------------------------------------------------------------
    # first verify the implementation (about 10 minutes, only the N=9 anchor)
    python experiments/learning_curve_subjects.py --smoke

    # full run (interruptible; rerunning resumes automatically)
    python experiments/learning_curve_subjects.py

    # faster: fewer N values, one fewer repeat
    python experiments/learning_curve_subjects.py --n-list 2 4 6 8 9 --repeats 2

Output:
  results/learning_curve_subjects_raw.csv      metrics for each (seed, N, repeat, test subject)
  results/learning_curve_subjects_summary.csv  10-subject mean for each (seed, N)
"""
# --- Paths: always managed centrally by paths.py at the repo root; do not revert to absolute paths ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import (  # noqa: E402,F401
    REPO_ROOT, PIPELINE_DIR, RESULTS_DIR, DATASET_DIR, COMBINED_DIR,
    ensure_results_dir, require_dataset,
)
# --- End of path setup ---

import argparse
import glob
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)
sys.path.insert(0, HERE)
from eval_utils import (subject_id_from_path, build_ts_cols,      # noqa: E402
                        make_windows, full_metrics)
from loso_train_and_save import (build_model, FEATURE_MODE, WINDOW_SIZE,  # noqa: E402
                                 STEP_SIZE, EPOCHS, BATCH_SIZE,
                                 STATIC_COLS, LABEL_EMG_COLS)

ap = argparse.ArgumentParser(description="Subject-count learning curve")
ap.add_argument("--data-dir", default=None,
                help="Feature folder, default Dataset/Combined")
ap.add_argument("--n-list", nargs="*", type=int, default=[2, 3, 4, 5, 6, 7, 8, 9],
                help="Training-set sizes, default 2..9")
ap.add_argument("--repeats", type=int, default=0,
                help="Fixed number of subject combinations to draw per N. 0 (default) = allocate automatically by --repeat-budget")
ap.add_argument("--repeat-budget", type=int, default=8,
                help="Budget for automatic allocation: reps(N)=ceil(budget/N), capped at 4. "
                     "Small N varies more between combinations so it gets more draws, large N fewer — compute goes where it matters")
ap.add_argument("--seeds", nargs="*", type=int, default=[42],
                help="Weight-initialization seeds. Add more if time allows; one is enough to see the trend")
ap.add_argument("--epochs", type=int, default=EPOCHS,
                help=f"Defaults to the current setting {EPOCHS}, comparable with existing numbers")
ap.add_argument("--prefix", default="learning_curve_subjects")
ap.add_argument("--smoke", action="store_true",
                help="Only run the N=9 anchor, to confirm the implementation matches standard LOSO")
args = ap.parse_args()

DATA_DIR = args.data_dir or COMBINED_DIR
if not os.path.isabs(DATA_DIR):
    DATA_DIR = os.path.join(DATASET_DIR, DATA_DIR)
N_LIST = [9] if args.smoke else sorted(set(args.n_list))
RAW_OUT = os.path.join(RESULTS_DIR, f'{args.prefix}_raw.csv')
SUM_OUT = os.path.join(RESULTS_DIR, f'{args.prefix}_summary.csv')


# =====================================================================
# preload the data: windows are built only once
# =====================================================================
def load_all():
    """Return (subjects, per_subject).

    per_subject[s] = {
        'frames_ts' : ts features of all of this subject's frames (for fitting the scaler)
        'frames_st' : same, static columns
        'X', 'S', 'Y': unscaled windows (scaling is deferred until after the fold split)
    }
    """
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_Combined_Features.csv")))
    if not files:
        sys.exit(f"Feature files not found: {DATA_DIR}")
    dfs = {p: pd.read_csv(p) for p in files}
    ts_cols = build_ts_cols(list(dfs[files[0]].columns), FEATURE_MODE, False)

    per = {}
    for p in files:
        s = subject_id_from_path(p)
        per.setdefault(s, {'dfs': []})['dfs'].append(dfs[p])

    for s, d in per.items():
        # each segment is windowed separately and windows never cross segments — the precondition for preventing temporal leakage
        X, St, Y, _ = make_windows(d['dfs'], ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                   None, WINDOW_SIZE, STEP_SIZE, None, None)
        big = pd.concat(d['dfs'], ignore_index=True)
        d['frames_ts'] = big[ts_cols].values
        d['frames_st'] = big[STATIC_COLS].values
        d['X'], d['S'], d['Y'] = X, St, Y
        del d['dfs']
    subjects = sorted(per)
    print(f"Loaded {len(files)} feature files / {len(subjects)} subjects / "
          f"{sum(len(d['X']) for d in per.values())} windows / {len(ts_cols)} feature dims")
    return subjects, per, len(ts_cols)


def scale(X, sc):
    """Apply a StandardScaler to pre-built windows (an affine transform, equivalent to scaling first and windowing afterwards)."""
    return (X - sc.mean_) / sc.scale_


def run_one(per, train_ids, test_id, n_ts, seed):
    """Train one fold and return the test subject's metrics."""
    tf.keras.utils.set_random_seed(seed)

    s_ts = StandardScaler().fit(np.concatenate([per[s]['frames_ts'] for s in train_ids]))
    s_st = StandardScaler().fit(np.concatenate([per[s]['frames_st'] for s in train_ids]))

    Xtr = np.concatenate([scale(per[s]['X'], s_ts) for s in train_ids])
    Str = np.concatenate([scale(per[s]['S'], s_st) for s in train_ids])
    Ytr = np.concatenate([per[s]['Y'] for s in train_ids])

    Xva = scale(per[test_id]['X'], s_ts)
    Sva = scale(per[test_id]['S'], s_st)
    Yva = per[test_id]['Y']

    model = build_model(n_ts, len(STATIC_COLS))
    model.fit({'ts_input': Xtr, 'static_input': Str}, {'out_emg': Ytr},
              epochs=args.epochs, batch_size=BATCH_SIZE, verbose=0)
    pred = model.predict({'ts_input': Xva, 'static_input': Sva}, verbose=0)
    pred = pred[0] if isinstance(pred, list) else pred
    m = full_metrics(Yva, pred)
    tf.keras.backend.clear_session()
    return {'r_main': m['r'][0], 'r_syn': m['r'][1],
            'nrmse_main': m['nrmse'][0], 'nrmse_syn': m['nrmse'][1],
            'n_train_windows': int(len(Xtr))}


def subset_for(subjects, test_id, n, rep, seed):
    """Decide which N people to draw for repeat rep.

    Uses (seed, test_id, n, rep) as the RNG seed — the same arguments always give the same people,
    so an interrupted run resumes without changing combinations and the old rows in raw.csv stay valid.
    """
    pool = [s for s in subjects if s != test_id]
    if n >= len(pool):
        return list(pool)
    # use crc32 rather than the built-in hash: Python's string hash differs per process
    # (PYTHONHASHSEED randomization), so a resumed run would draw different people and invalidate the old data.
    key = ('%d|%s|%d|%d' % (seed, test_id, n, rep)).encode('utf-8')
    rng = np.random.default_rng(zlib.crc32(key))
    return sorted(rng.choice(pool, size=n, replace=False).tolist())


def main():
    ensure_results_dir()
    subjects, per, n_ts = load_all()

    done = set()
    rows = []
    if os.path.exists(RAW_OUT):
        old = pd.read_csv(RAW_OUT)
        rows = old.to_dict('records')
        done = {(int(r['seed']), int(r['n_train']), int(r['repeat']), r['test_subject'])
                for r in rows}
        print(f"Resuming previous results: {len(done)} rows already present, skipped.")

    def n_reps(n):
        if n >= len(subjects) - 1:
            return 1            # only one possible draw: take everyone else
        if args.repeats > 0:
            return args.repeats
        return max(1, min(4, -(-args.repeat_budget // n)))   # ceil, capped at 4

    jobs = []
    for seed in args.seeds:
        for n in N_LIST:
            reps = n_reps(n)
            for rep in range(reps):
                for test_id in subjects:
                    key = (seed, n, rep, test_id)
                    if key not in done:
                        jobs.append(key)

    print(f"{len(jobs)} folds to run ({len(args.seeds)} seeds × {len(N_LIST)} values of N)\n")
    t_start = time.time()
    for i, (seed, n, rep, test_id) in enumerate(jobs, 1):
        t0 = time.time()
        train_ids = subset_for(subjects, test_id, n, rep, seed)
        m = run_one(per, train_ids, test_id, n_ts, seed)
        rows.append({'seed': seed, 'n_train': n, 'repeat': rep,
                     'test_subject': test_id, 'train_subjects': '+'.join(train_ids),
                     **m})
        pd.DataFrame(rows).to_csv(RAW_OUT, index=False, encoding='utf-8-sig')

        el = time.time() - t0
        eta = (time.time() - t_start) / i * (len(jobs) - i) / 60
        print(f"[{i}/{len(jobs)}] N={n} rep={rep} test={test_id}  "
              f"r_main={m['r_main']:.3f}  nrmse={m['nrmse_main']:.3f}  "
              f"({el:.0f}s, about {eta:.0f} min left)", flush=True)

    summarise(rows, subjects)


def summarise(rows, subjects):
    df = pd.DataFrame(rows)
    if df.empty:
        return
    # average over repeats first (different subject combinations at the same N), then over subjects
    g = (df.groupby(['seed', 'n_train', 'test_subject'], as_index=False)
           .mean(numeric_only=True))
    ex = g[g['test_subject'] != 'S09']
    sm = g.groupby(['seed', 'n_train'], as_index=False).mean(numeric_only=True)
    sm2 = (ex.groupby(['seed', 'n_train'], as_index=False)
             .mean(numeric_only=True)[['seed', 'n_train', 'r_syn', 'nrmse_syn']]
             .rename(columns={'r_syn': 'r_syn_ex_s09',
                              'nrmse_syn': 'nrmse_syn_ex_s09'}))
    sm = sm.merge(sm2, on=['seed', 'n_train'])
    sm = sm.drop(columns=[c for c in ('repeat',) if c in sm])
    sm.to_csv(SUM_OUT, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 66)
    print("Subject-count learning curve (mean over 10 test subjects)")
    print("=" * 66)
    print(f"{'N':>3}  {'main r':>9}  {'nRMSE':>8}  {'syn r':>9}  {'train win':>9}")
    agg = sm.groupby('n_train', as_index=False).mean(numeric_only=True)
    for _, r in agg.iterrows():
        print(f"{int(r['n_train']):>3}  {r['r_main']:>9.4f}  {r['nrmse_main']:>8.4f}  "
              f"{r['r_syn']:>9.4f}  {int(r['n_train_windows']):>9,}")

    anchor = agg[agg['n_train'] == 9]
    if len(anchor):
        v = float(anchor['r_main'].iloc[0])
        ok = 0.70 <= v <= 0.77
        print(f"\nAnchor check: N=9 should equal standard LOSO (past runs 0.723~0.740)")
        print(f"  this run N=9 main r = {v:.4f}  {'✅ within the range' if ok else '🔴 off — the implementation may differ'}")

    print(f"\n✅ {RAW_OUT}")
    print(f"✅ {SUM_OUT}")
    print("\nNext: python experiments/plot_learning_curve.py  (plot and extrapolate)")


if __name__ == '__main__':
    require_dataset()
    main()
