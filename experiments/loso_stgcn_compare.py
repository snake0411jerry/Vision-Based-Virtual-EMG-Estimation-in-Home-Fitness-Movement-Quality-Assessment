"""
loso_stgcn_compare.py — ST-GCN student network vs the current Conv1D (multi-seed LOSO)
=========================================================================
Corresponds to "step 2: student network" and "expected outcome 1" of the NSTC proposal.

Three arms (run within the same job to rule out cross-session GPU nondeterminism):

  baseline      Conv1D ×2, input = the 16 **hand-crafted features** in Dataset/Combined
  stgcn_pos     ST-GCN, input = 20 joints × 3 coordinates from Dataset/Skeleton
  stgcn_posva   ST-GCN, input = 20 joints × 9 channels (position + velocity + acceleration)

⚠️ **This is not a pure architecture swap**: the baseline consumes hand-crafted features (angles, ratios, heel height…),
   while ST-GCN consumes the raw skeleton. What is compared is "hand-crafted features + temporal convolution" vs
   "raw skeleton + graph convolution". That is exactly what the proposal asks, but the manuscript must say so clearly
   and not describe it as merely swapping the architecture.

⚠️ The **evaluation windows are fully aligned** on both sides: same LOSO folds, same WINDOW_SIZE/STEP_SIZE,
   never crossing segment boundaries, and the scaler is fit only on that fold's training subjects.

Output: stgcn_compare_{raw,summary}.csv
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
from loso_train_and_save import (build_model as build_conv1d, FEATURE_MODE,  # noqa: E402
                                 WINDOW_SIZE, STEP_SIZE, EPOCHS, BATCH_SIZE,
                                 STATIC_COLS, LABEL_EMG_COLS, DATA_DIR)
from models_stgcn import build_stgcn, NUM_JOINTS                             # noqa: E402

SKEL_DIR = SKELETON_DIR
# ---------------------------------------------------------------- skeleton data
def load_skeleton(channels):
    """Return {subject: [(X(T,V,C), emg(T,2), static(4))...]}"""
    out = {}
    for f in sorted(glob.glob(os.path.join(SKEL_DIR, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        parts = [z['coords']]
        if channels >= 6:
            parts.append(z['vel'])
        if channels >= 9:
            parts.append(z['acc'])
        X = np.concatenate(parts, axis=2).astype(np.float32)   # (T, V, C)
        subj = str(z['subject'])
        out.setdefault(subj, []).append((X, z['emg'].astype(np.float32),
                                         z['static'].astype(np.float32)))
    return out


def skeleton_windows(segments, scaler_x, scaler_s):
    """Window each segment separately; windows never cross a segment boundary (same semantics as make_windows)."""
    Xs, Ss, Ys = [], [], []
    for X, emg, static in segments:
        T = min(len(X), len(emg))
        if T < WINDOW_SIZE:
            continue
        V, C = X.shape[1], X.shape[2]
        flat = scaler_x.transform(X[:T].reshape(T, V * C)).reshape(T, V, C)
        st = scaler_s.transform(static.reshape(1, -1))[0]
        for s in range(0, T - WINDOW_SIZE + 1, STEP_SIZE):
            Xs.append(flat[s:s + WINDOW_SIZE])
            Ss.append(st)
            Ys.append(emg[s + WINDOW_SIZE - 1])      # the label is taken at the window end, consistent with make_windows
    return (np.asarray(Xs, np.float32), np.asarray(Ss, np.float32),
            np.asarray(Ys, np.float32))


def run_stgcn_fold(data, subjects, hold, channels, filters):
    tr = [s for k in subjects if k != hold for s in data[k]]
    va = data[hold]
    V = tr[0][0].shape[1]
    big = np.concatenate([X.reshape(len(X), V * channels) for X, _, _ in tr])
    sx = StandardScaler().fit(big)
    ss = StandardScaler().fit(np.stack([st for _, _, st in tr]))

    Xtr, Str, Ytr = skeleton_windows(tr, sx, ss)
    Xva, Sva, Yva = skeleton_windows(va, sx, ss)

    model = build_stgcn(WINDOW_SIZE, channels, len(STATIC_COLS), filters=filters)
    model.fit({'ts_input': Xtr, 'static_input': Str}, {'out_emg': Ytr},
              epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    pred = model.predict({'ts_input': Xva, 'static_input': Sva}, verbose=0)
    pred = pred[0] if isinstance(pred, list) else pred
    n_params = model.count_params()
    return full_metrics(Yva, pred), n_params


def run_baseline_fold(dfs, file_paths, groups, ts_cols, hold):
    tr = [dfs[p] for p, g in zip(file_paths, groups) if g != hold]
    va = [dfs[p] for p, g in zip(file_paths, groups) if g == hold]
    big = pd.concat(tr, ignore_index=True)
    s_ts = StandardScaler().fit(big[ts_cols].values)
    s_st = StandardScaler().fit(big[STATIC_COLS].values)
    Xtr, Str, Ytr, _ = make_windows(tr, ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                    None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
    Xva, Sva, Yva, _ = make_windows(va, ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                    None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
    model = build_conv1d(len(ts_cols), len(STATIC_COLS))
    model.fit({'ts_input': Xtr, 'static_input': Str}, {'out_emg': Ytr},
              epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    pred = model.predict({'ts_input': Xva, 'static_input': Sva}, verbose=0)
    pred = pred[0] if isinstance(pred, list) else pred
    return full_metrics(Yva, pred), model.count_params()


def main():
    ap = argparse.ArgumentParser(description="ST-GCN vs Conv1D")
    ap.add_argument("--arms", nargs="*",
                    default=['baseline', 'stgcn_pos', 'stgcn_posva'])
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 1, 2])
    ap.add_argument("--filters", nargs="*", type=int, default=[16, 32, 32])
    ap.add_argument("--prefix", default="stgcn_compare")
    args = ap.parse_args()
    filters = tuple(args.filters)

    file_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*_Combined_Features.csv")))
    groups = [subject_id_from_path(p) for p in file_paths]
    subjects = sorted(set(groups))
    dfs = {p: pd.read_csv(p) for p in file_paths}
    ts_cols = build_ts_cols(list(dfs[file_paths[0]].columns), FEATURE_MODE, False)

    skel = {c: load_skeleton(c) for c in
            ({3} if 'stgcn_pos' in args.arms else set()) |
            ({9} if 'stgcn_posva' in args.arms else set())}
    print(f"📂 Combined {len(file_paths)} files / hand-crafted features {len(ts_cols)} dims")
    if skel:
        any_c = next(iter(skel))
        print(f"🦴 Skeleton {sum(len(v) for v in skel[any_c].values())} segments / "
              f"{NUM_JOINTS} joints / filters={filters}")
    print(f"🔁 arms={args.arms} × {len(args.seeds)} seeds")

    rows = []
    total = len(args.arms) * len(args.seeds)
    done = 0
    t0 = time.time()
    for arm in args.arms:
        for seed in args.seeds:
            tf.keras.utils.set_random_seed(seed)
            npar = None
            for hold in subjects:
                if arm == 'baseline':
                    m, npar = run_baseline_fold(dfs, file_paths, groups, ts_cols, hold)
                else:
                    c = 3 if arm == 'stgcn_pos' else 9
                    m, npar = run_stgcn_fold(skel[c], subjects, hold, c, filters)
                rows.append({'arm': arm, 'seed': seed, 'Subject': hold,
                             'r_main': m['r'][0], 'r_syn': m['r'][1],
                             'nrmse_main': m['nrmse'][0], 'nrmse_syn': m['nrmse'][1],
                             'n_params': npar})
                tf.keras.backend.clear_session()

            pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, f'{args.prefix}_raw.csv'),
                                      index=False, encoding='utf-8-sig')
            done += 1
            d = pd.DataFrame(rows)
            d = d[(d.arm == arm) & (d.seed == seed)]
            el = (time.time() - t0) / done
            print(f"[{done}/{total}] {arm} seed={seed}  main r={d.r_main.mean():.3f}  "
                  f"synergist r={d.r_syn.mean():.3f}  params {npar:,}  "
                  f"({el/60:.1f} min/round, about {el*(total-done)/60:.0f} min left)", flush=True)

    df = pd.DataFrame(rows)
    per = df.groupby(['arm', 'seed'])[['r_main', 'r_syn', 'nrmse_main',
                                       'nrmse_syn']].mean().reset_index()
    per.to_csv(os.path.join(RESULTS_DIR, f'{args.prefix}_summary.csv'),
               index=False, encoding='utf-8-sig')

    print("\n" + "=" * 92)
    print(f"ST-GCN comparison (mean ± sd over {len(args.seeds)} seeds)")
    print("=" * 92)
    par = df.groupby('arm')['n_params'].first()
    for c in ['r_main', 'r_syn', 'nrmse_main', 'nrmse_syn']:
        line = f"  {c:<12}"
        for a in args.arms:
            g = per[per.arm == a][c]
            line += f"  {a}: {g.mean():.4f}±{g.std(ddof=1):.4f}"
        print(line)
    print("\n  Parameters: " + "   ".join(f"{a} {par[a]:,}" for a in args.arms if a in par))

    if 'baseline' in args.arms:
        from scipy import stats
        base = per[per.arm == 'baseline'].set_index('seed')
        print("\n  Paired t-tests against baseline (same job; every arm shares the same seeds):")
        for a in args.arms:
            if a == 'baseline':
                continue
            cur = per[per.arm == a].set_index('seed')
            shared = base.index.intersection(cur.index)
            for c in ['r_main', 'r_syn']:
                x, y = base.loc[shared, c], cur.loc[shared, c]
                t, p = stats.ttest_rel(y, x)
                print(f"    {a:<14} {c:<8} diff {y.mean()-x.mean():+.4f}   "
                      f"t={t:+.2f}  p={p:.3f}  (n={len(shared)} seed)")
        print(f"\n  ⚠️ n={len(args.seeds)} seeds, limited power; "
              f"a non-significant p only means “no evidence of a difference”, not equivalence.")

    print(f"\n✅ {args.prefix}_raw.csv / {args.prefix}_summary.csv")


if __name__ == '__main__':
    main()
