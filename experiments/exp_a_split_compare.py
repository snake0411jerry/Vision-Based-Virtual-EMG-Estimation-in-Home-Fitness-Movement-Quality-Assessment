"""
exp_a_split_compare.py — Experiment A: split-strategy comparison (LOSO vs random splits)
=========================================================================
Corresponds to Step 3, Experiment A of paper 035S_IC3MT2026:
  "Contrasts Leave-One-Subject-Out (LOSO) cross-validation with random
    data-splitting ... thereby justifying the need for a personalized
    fine-tuning mechanism"

The repo originally had only the LOSO half; the random-split half had never been run properly
(`專案現況_交接用.md:174` only records a rule of thumb, "the global model inflates by about +0.15", with no
multi-seed comparison). This script fills the gap and splits the "inflation" into two distinguishable sources.

Three arms (run within the same job to rule out cross-session GPU nondeterminism):

  loso            fold = subject. The model has never seen anyone in the test fold     ← honest baseline
  random_segment  fold = randomly assigned segments. Windows still never cross segment
                  boundaries, but other segments of the same person appear in training ← leak: has seen this person
  random_window   fold = randomly assigned sliding windows. Adjacent overlapping windows
                  fall in both training and test sets                                  ← leak: has seen this person + neighbouring frames

★ All three arms share **the same pre-built windows**; the only difference is how folds are assigned.
  Architecture, EPOCHS, WINDOW_SIZE/STEP_SIZE and fold count (10) are identical, and the training set
  is always 9/10, so compute and comparability are matched.

★ The metric definition is identical for all three: collect **the out-of-fold predictions for every window**,
  compute r/MAE/nRMSE per subject, then average over the 10 subjects.
  In this framework LOSO is just the special case where "folds happen to equal subjects", so the arms can be
  shown side by side.
  ⚠️ Do not switch to computing a single r on the pooled test set — that gets inflated by between-subject
     variance, and the three numbers would no longer be the same quantity.

⚠️ Scaler handling (honest disclosure):
   loso / random_segment fit only on the training-fold segment frames (exactly as the current pipeline).
   For random_window the training windows span every segment, so the scaler effectively sees all the data;
   it is therefore fit on all frames. That arm is the "everything leaks" arm anyway, so its role is unchanged,
   but if the paper cites random_window numbers, this point must be stated too.

Output: exp_a_split_{raw,summary}.csv
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

from eval_utils import (subject_id_from_path, build_ts_cols,          # noqa: E402
                        make_windows, full_metrics)
from loso_train_and_save import (build_model, FEATURE_MODE, WINDOW_SIZE,   # noqa: E402
                                 STEP_SIZE, EPOCHS, BATCH_SIZE,
                                 STATIC_COLS, LABEL_EMG_COLS, DATA_DIR)

N_FOLDS = 10
ARMS_ALL = ['loso', 'random_segment', 'random_window']


# ------------------------------------------------------------------ data loading
def load_all_windows(ts_cols):
    """Build all windows at once (unstandardized), recording which subject and segment each window belongs to.

    Window generation is delegated entirely to eval_utils.make_windows (scaler passed as None),
    guaranteeing window-for-window identical semantics with loso_train_and_save / loso_multiseed.
    """
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*_Combined_Features.csv")))
    Xs, Ss, Ys, w_subj, w_seg = [], [], [], [], []
    seg_frames_ts, seg_frames_st, seg_subj = [], [], []

    for seg_i, p in enumerate(paths):
        df = pd.read_csv(p)
        subj = subject_id_from_path(p)
        X, S, Y, _ = make_windows([df], ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                  None, WINDOW_SIZE, STEP_SIZE, None, None)
        if len(X) == 0:
            print(f"⚠️ {os.path.basename(p)} has fewer frames than one window, skipping")
            continue
        Xs.append(X.astype(np.float32))
        Ss.append(S.astype(np.float32))
        Ys.append(Y.astype(np.float32))
        w_subj.append(np.full(len(X), subj))
        w_seg.append(np.full(len(X), seg_i))
        seg_frames_ts.append(df[ts_cols].values.astype(np.float32))
        seg_frames_st.append(df[STATIC_COLS].values.astype(np.float32))
        seg_subj.append(subj)

    return {
        'X': np.concatenate(Xs), 'S': np.concatenate(Ss), 'Y': np.concatenate(Ys),
        'w_subj': np.concatenate(w_subj), 'w_seg': np.concatenate(w_seg),
        'seg_ts': seg_frames_ts, 'seg_st': seg_frames_st,
        'seg_subj': np.array(seg_subj), 'n_seg': len(seg_subj),
    }


# ------------------------------------------------------------------ fold assignment
def assign_folds(arm, d, subjects, rng):
    """Return (window_fold, seg_fold). seg_fold is None for random_window."""
    n_seg, N = d['n_seg'], len(d['X'])

    if arm == 'loso':
        idx = {s: i for i, s in enumerate(subjects)}
        seg_fold = np.array([idx[s] for s in d['seg_subj']])
        return seg_fold[d['w_seg']], seg_fold

    if arm == 'random_segment':
        seg_fold = np.empty(n_seg, dtype=int)
        for f, part in enumerate(np.array_split(rng.permutation(n_seg), N_FOLDS)):
            seg_fold[part] = f
        return seg_fold[d['w_seg']], seg_fold

    if arm == 'random_window':
        fold = np.empty(N, dtype=int)
        for f, part in enumerate(np.array_split(rng.permutation(N), N_FOLDS)):
            fold[part] = f
        return fold, None

    raise ValueError(f"Unknown arm: {arm}")


def fit_scalers(d, arm, seg_fold, f):
    """The scaler may only see training data — see the header note on random_window."""
    if arm == 'random_window':
        keep = range(d['n_seg'])
    else:
        keep = [i for i in range(d['n_seg']) if seg_fold[i] != f]
    ts = np.concatenate([d['seg_ts'][i] for i in keep])
    st = np.concatenate([d['seg_st'][i] for i in keep])
    return StandardScaler().fit(ts), StandardScaler().fit(st)


# ------------------------------------------------------------------ single arm+seed
def run_arm_seed(d, arm, seed, subjects, n_ts):
    """Run all 10 folds and return per-subject metrics (using out-of-fold predictions for every window)."""
    tf.keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)
    w_fold, seg_fold = assign_folds(arm, d, subjects, rng)

    oof = np.full_like(d['Y'], np.nan)
    for f in range(N_FOLDS):
        te = w_fold == f
        tr = ~te
        if te.sum() == 0 or tr.sum() == 0:
            print(f"⚠️ arm={arm} seed={seed} fold={f} is empty, skipping")
            continue

        s_ts, s_st = fit_scalers(d, arm, seg_fold, f)
        mu, sd = s_ts.mean_.astype(np.float32), s_ts.scale_.astype(np.float32)

        Xtr = (d['X'][tr] - mu) / sd
        Xte = (d['X'][te] - mu) / sd
        Str = s_st.transform(d['S'][tr]).astype(np.float32)
        Ste = s_st.transform(d['S'][te]).astype(np.float32)

        model = build_model(n_ts, len(STATIC_COLS))
        model.fit({'ts_input': Xtr, 'static_input': Str}, {'out_emg': d['Y'][tr]},
                  epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        pred = model.predict({'ts_input': Xte, 'static_input': Ste}, verbose=0)
        oof[te] = pred[0] if isinstance(pred, list) else pred
        tf.keras.backend.clear_session()

    assert not np.isnan(oof).any(), "Some windows did not get an out-of-fold prediction"

    rows = []
    for s in subjects:
        m = full_metrics(d['Y'][d['w_subj'] == s], oof[d['w_subj'] == s])
        rows.append({'arm': arm, 'seed': seed, 'Subject': s,
                     'r_main': m['r'][0], 'r_syn': m['r'][1],
                     'mae_main': m['mae'][0], 'mae_syn': m['mae'][1],
                     'nrmse_main': m['nrmse'][0], 'nrmse_syn': m['nrmse'][1]})
    return rows


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Experiment A: split-strategy comparison")
    ap.add_argument("--arms", nargs="*", default=ARMS_ALL)
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 1, 2, 3, 4])
    ap.add_argument("--prefix", default="exp_a_split")
    args = ap.parse_args()

    probe = sorted(glob.glob(os.path.join(DATA_DIR, "*_Combined_Features.csv")))
    ts_cols = build_ts_cols(list(pd.read_csv(probe[0], nrows=1).columns),
                            FEATURE_MODE, False)
    d = load_all_windows(ts_cols)
    subjects = sorted(set(d['seg_subj']))

    print(f"📂 {d['n_seg']} segments / {len(subjects)} subjects / {len(d['X']):,} windows / {len(ts_cols)} dims")
    print(f"🔁 arms={args.arms} × {len(args.seeds)} seeds × {N_FOLDS} folds"
          f"  (window={WINDOW_SIZE}, step={STEP_SIZE}, epochs={EPOCHS})")

    raw_out = os.path.join(RESULTS_DIR, f'{args.prefix}_raw.csv')
    sum_out = os.path.join(RESULTS_DIR, f'{args.prefix}_summary.csv')
    rows = []
    total, done, t0 = len(args.arms) * len(args.seeds), 0, time.time()

    for arm in args.arms:
        for seed in args.seeds:
            rows += run_arm_seed(d, arm, seed, subjects, len(ts_cols))
            pd.DataFrame(rows).to_csv(raw_out, index=False, encoding='utf-8-sig')
            done += 1
            cur = pd.DataFrame(rows)
            cur = cur[(cur.arm == arm) & (cur.seed == seed)]
            el = (time.time() - t0) / done
            print(f"[{done}/{total}] {arm:<15} seed={seed}  "
                  f"main r={cur.r_main.mean():.3f}  synergist r={cur.r_syn.mean():.3f}  "
                  f"({el/60:.1f} min/round, about {el*(total-done)/60:.0f} min left)", flush=True)

    df = pd.DataFrame(rows)
    per = df.groupby(['arm', 'seed'])[['r_main', 'r_syn', 'nrmse_main',
                                       'nrmse_syn']].mean().reset_index()
    per.to_csv(sum_out, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 88)
    print(f"Experiment A: split-strategy comparison (mean ± sd over {len(args.seeds)} seeds)")
    print("=" * 88)
    for c in ['r_main', 'r_syn', 'nrmse_main', 'nrmse_syn']:
        line = f"  {c:<12}"
        for a in args.arms:
            g = per[per.arm == a][c]
            line += f"  {a}: {g.mean():.4f}±{g.std(ddof=1):.4f}"
        print(line)

    if 'loso' in args.arms:
        from scipy import stats
        base = per[per.arm == 'loso'].set_index('seed')
        print("\n  Paired t-tests against loso (same job; every arm shares the same seeds):")
        for a in args.arms:
            if a == 'loso':
                continue
            cur = per[per.arm == a].set_index('seed')
            shared = base.index.intersection(cur.index)
            for c in ['r_main', 'r_syn']:
                x, y = base.loc[shared, c], cur.loc[shared, c]
                t, p = stats.ttest_rel(y, x)
                print(f"    {a:<15} {c:<8} inflation {y.mean()-x.mean():+.4f}   "
                      f"t={t:+.2f}  p={p:.4f}  (n={len(shared)} seed)")
        print("\n  ⚠️ “Inflation” here is the **overestimate** caused by the split strategy, not a performance gain.")

    print(f"\n✅ {raw_out}\n✅ {sum_out}")


if __name__ == '__main__':
    main()
