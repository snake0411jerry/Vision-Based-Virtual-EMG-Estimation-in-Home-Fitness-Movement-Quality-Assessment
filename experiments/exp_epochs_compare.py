"""
exp_epochs_compare.py — training-epoch comparison: is LOSO overtrained?
=========================================================================
Origin (2026-08-05): three independent lines of evidence all point to "the model is limited by data, not architecture" —

  ① External audit report `MODEL_REVIEW.md` (outside the repo): an off-the-shelf gradient-boosted tree under the same
     LOSO reaches r=0.776 / nRMSE=0.745, better than Conv1D 16-dim (0.727 / 0.815).
  ② exp_distill_compress.py: cutting ST-GCN parameters from 29,734 to 10,070 (2.95×) does not lower r at all
     (−0.0004, p=0.92). If capacity were the bottleneck, cutting to 1/3 should hurt.
  ③ 專案現況_交接用.md recorded long ago: "the training curve shows val bottoming out around epochs 3–5 and then
     getting worse; the current 30 epochs is too many." **But it was never formally measured.**

This experiment fills in ③. It is currently the highest-return experiment: for the cost of a single full LOSO run,
it yields the results for every epoch setting at once.

★ Why one training run is enough
  Training is sequential, so "the model at epoch N" is exactly "the model you would get with EPOCHS=N".
  Evaluating once at each chosen epoch via a callback is equivalent to rerunning with each setting, and the
  comparisons are **perfectly paired** (same training trajectory), at 1/6 of the cost.

★ Training-set performance is recorded too — that is what separates overfitting from underfitting
  training far above validation → overfitting: shrink the model / add regularization / early stopping
  both low                       → underfitting or an optimization problem: tune the learning rate / train more
  (corresponds to "step 2" of MODEL_REVIEW.md)

🔴 **This curve must not be used to choose the epoch.**
  The evaluation set is the LOSO holdout subject = the test set. Picking the best epoch by looking at it
  means tuning on the test set — leakage — and the reported number would be inflated.
  This experiment has exactly one purpose: **determine whether the "overtraining" phenomenon exists at all**.
  If it does, the correct fix is an inner validation (nested LOSO, or carving a part out of the training folds)
  that decides the epoch; **the best value in this table must not be adopted directly**.

Usage
-----
    # full run (about 2 hours, the cost of one current LOSO run)
    python exp_epochs_compare.py --seeds 42 1 2 3 4

    # smoke test
    python exp_epochs_compare.py --seeds 42 --subjects S01 S02 S03 \
        --checkpoints 1 2 --max-epochs 2 --prefix smoke_epochs

Output: results/{prefix}_{raw,summary}.csv (**written after every fold**, so an interruption still leaves partial results)
"""
# --- Paths: always managed centrally by paths.py at the repo root; do not revert to absolute paths ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import (  # noqa: E402,F401
    REPO_ROOT, PIPELINE_DIR, RESULTS_DIR, ensure_results_dir,
)
# --- End of path setup ---

import argparse
import gc
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

for _gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)
sys.path.insert(0, HERE)

from eval_utils import (subject_id_from_path, build_ts_cols,      # noqa: E402
                        make_windows, full_metrics)
from loso_train_and_save import (build_model, FEATURE_MODE, WINDOW_SIZE,  # noqa: E402
                                 STEP_SIZE, BATCH_SIZE, STATIC_COLS,
                                 LABEL_EMG_COLS, DATA_DIR)


class EpochEval(tf.keras.callbacks.Callback):
    """Record holdout and training-set metrics at the specified epochs.

    ⚠️ Records only; makes no selection — see the header note on leakage.
    """

    def __init__(self, checkpoints, va, tr_sample):
        super().__init__()
        self.checkpoints = set(checkpoints)
        self.va = va
        self.tr = tr_sample
        self.records = {}

    def on_epoch_end(self, epoch, logs=None):
        e = epoch + 1
        if e not in self.checkpoints:
            return
        Xva, Sva, Yva = self.va
        Xtr, Str, Ytr = self.tr
        pv = self.model.predict({'ts_input': Xva, 'static_input': Sva},
                                batch_size=256, verbose=0)
        pt = self.model.predict({'ts_input': Xtr, 'static_input': Str},
                                batch_size=256, verbose=0)
        self.records[e] = (full_metrics(Yva, pv), full_metrics(Ytr, pt))


def run_fold(dfs, file_paths, groups, ts_cols, hold, args, rng):
    tr_dfs = [dfs[p] for p, g in zip(file_paths, groups) if g != hold]
    va_dfs = [dfs[p] for p, g in zip(file_paths, groups) if g == hold]

    # ★ the scaler is fit only on this fold's training subjects (exactly as the current pipeline)
    big = pd.concat(tr_dfs, ignore_index=True)
    s_ts = StandardScaler().fit(big[ts_cols].values)
    s_st = StandardScaler().fit(big[STATIC_COLS].values)

    Xtr, Str, Ytr, _ = make_windows(tr_dfs, ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                    window_size=WINDOW_SIZE, step_size=STEP_SIZE,
                                    scaler_ts=s_ts, scaler_static=s_st)
    Xva, Sva, Yva, _ = make_windows(va_dfs, ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                    window_size=WINDOW_SIZE, step_size=STEP_SIZE,
                                    scaler_ts=s_ts, scaler_static=s_st)

    # sampling is enough for training-set metrics (we only need the overfitting gap, not full predictions)
    n = min(args.train_sample, len(Xtr))
    idx = rng.choice(len(Xtr), n, replace=False)
    tr_sample = (Xtr[idx], Str[idx], Ytr[idx])

    cb = EpochEval(args.checkpoints, (Xva, Sva, Yva), tr_sample)
    model = build_model(len(ts_cols), len(STATIC_COLS))
    model.fit({'ts_input': Xtr, 'static_input': Str}, {'out_emg': Ytr},
              epochs=args.max_epochs, batch_size=BATCH_SIZE,
              callbacks=[cb], verbose=0)
    n_params = model.count_params()
    del model, Xtr, Str, Ytr, Xva, Sva, Yva, tr_sample
    return cb.records, n_params


def main():
    ap = argparse.ArgumentParser(description="Training-epoch comparison: is LOSO overtrained")
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 1, 2, 3, 4])
    ap.add_argument("--checkpoints", nargs="*", type=int,
                    default=[3, 5, 10, 15, 20, 30],
                    help="Record metrics at these epochs")
    ap.add_argument("--max-epochs", type=int, default=30,
                    help="Actual number of training epochs; must be ≥ the largest checkpoint")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--holdout", nargs="*", default=None)
    ap.add_argument("--train-sample", type=int, default=8000,
                    help="Number of windows sampled to estimate training-set metrics (affects only diagnostic precision, not training)")
    ap.add_argument("--prefix", default="exp_epochs")
    ap.add_argument("--resume", action="store_true",
                    help="Reuse the existing raw.csv and only fill in (seed, subject) pairs not yet run")
    args = ap.parse_args()

    if max(args.checkpoints) > args.max_epochs:
        ap.error(f"largest checkpoint {max(args.checkpoints)} "
                 f"exceeds --max-epochs {args.max_epochs}")

    ensure_results_dir()
    file_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*_Combined_Features.csv")))
    groups = [subject_id_from_path(p) for p in file_paths]
    subjects = sorted(set(groups))
    if args.subjects:
        keep = set(args.subjects)
        file_paths = [p for p, g in zip(file_paths, groups) if g in keep]
        groups = [g for g in groups if g in keep]
        subjects = sorted(keep)
    holdouts = subjects if args.holdout is None else [s for s in subjects if s in args.holdout]

    dfs = {p: pd.read_csv(p) for p in file_paths}
    ts_cols = build_ts_cols(dfs[file_paths[0]].columns.tolist(), FEATURE_MODE, False)

    print(f"📄 {len(file_paths)} feature files / {len(subjects)} subjects")
    print(f"🧮 {len(ts_cols)} time-series features ({FEATURE_MODE}) + {len(STATIC_COLS)} static")
    print(f"🔁 {len(args.seeds)} seeds × {len(holdouts)} folds, each fold trained for "
          f"{args.max_epochs} epochs, recorded at {args.checkpoints}\n")

    raw_out = os.path.join(RESULTS_DIR, f'{args.prefix}_raw.csv')
    rows, skip = [], set()
    if args.resume and os.path.exists(raw_out):
        prev = pd.read_csv(raw_out)
        rows = prev.to_dict('records')
        skip = {(int(r['seed']), str(r['Subject'])) for r in rows}
        print(f"↻ resuming: reusing {len(skip)} existing folds\n")

    todo = [(s, h) for s in args.seeds for h in holdouts if (s, h) not in skip]
    total, done, t0 = len(todo), 0, time.time()

    for seed, hold in todo:
        tf.keras.utils.set_random_seed(seed)
        rng = np.random.default_rng(seed)
        recs, n_params = run_fold(dfs, file_paths, groups, ts_cols, hold, args, rng)
        for e, (mv, mt) in sorted(recs.items()):
            rows.append({'epochs': e, 'seed': seed, 'Subject': hold,
                         'r_main': mv['r'][0], 'r_syn': mv['r'][1],
                         'nrmse_main': mv['nrmse'][0], 'nrmse_syn': mv['nrmse'][1],
                         'train_r_main': mt['r'][0],
                         'train_nrmse_main': mt['nrmse'][0],
                         'n_params': n_params})
        pd.DataFrame(rows).to_csv(raw_out, index=False, encoding='utf-8-sig')

        done += 1
        el = (time.time() - t0) / done
        curve = "  ".join(f"e{e}={recs[e][0]['r'][0]:.3f}" for e in sorted(recs))
        print(f"[{done}/{total}] seed={seed} hold={hold}  {curve}   "
              f"({el/60:.1f} min/fold, about {el*(total-done)/60:.0f} min left)", flush=True)

        del recs
        tf.keras.backend.clear_session()
        gc.collect()

    # ---------------- summary ----------------
    df = pd.DataFrame(rows)
    cols = ['r_main', 'r_syn', 'nrmse_main', 'nrmse_syn',
            'train_r_main', 'train_nrmse_main']
    per = df.groupby(['epochs', 'seed'])[cols].mean().reset_index()
    ex = df[df['Subject'] != 'S09'].groupby(['epochs', 'seed'])[
        ['r_main', 'r_syn']].mean().reset_index()
    per = per.merge(ex, on=['epochs', 'seed'], suffixes=('', '_ex_s09'))
    per.to_csv(os.path.join(RESULTS_DIR, f'{args.prefix}_summary.csv'),
               index=False, encoding='utf-8-sig')

    eps = sorted(df.epochs.unique())
    print("\n" + "=" * 96)
    print(f"Training-epoch comparison (mean ± sd over {per.seed.nunique()} seeds)")
    print("=" * 96)
    print(f"  {'epochs':>7} {'val r_main':>16} {'val nRMSE':>16} "
          f"{'train r_main':>14} {'overfit gap':>12}")
    for e in eps:
        d = per[per.epochs == e]
        gap = d['train_r_main'].mean() - d['r_main'].mean()
        print(f"  {e:>7} {d.r_main.mean():>9.4f}±{d.r_main.std(ddof=1):.4f} "
              f"{d.nrmse_main.mean():>9.4f}±{d.nrmse_main.std(ddof=1):.4f} "
              f"{d.train_r_main.mean():>13.4f} {gap:>+12.4f}")

    best = per.groupby('epochs')['r_main'].mean().idxmax()
    print(f"\n  The curve peaks at epoch {best}"
          f" (🔴 **do not adopt directly**, see header: it was picked on the test set)")

    # ---------------- paired tests: each setting vs the current 30 ----------------
    from scipy import stats
    ref_e = args.max_epochs if args.max_epochs in eps else eps[-1]
    ref = per[per.epochs == ref_e].set_index('seed')
    print("\n" + "-" * 96)
    print(f"★ Paired comparison with the current setting (epochs={ref_e})")
    print("-" * 96)
    for e in eps:
        if e == ref_e:
            continue
        cur = per[per.epochs == e].set_index('seed')
        sh = ref.index.intersection(cur.index)
        if len(sh) < 2:
            continue
        line = f"  epochs={e:<3}"
        for c in ['r_main', 'nrmse_main']:
            x, y = ref.loc[sh, c], cur.loc[sh, c]
            diff = y - x
            t, p = stats.ttest_rel(y, x)
            sd = diff.std(ddof=1)
            dz = diff.mean() / sd if sd > 0 else np.nan
            line += f"  {c} diff {diff.mean():+.4f} (p={p:.3f}, dz={dz:+.2f})"
        d = cur.loc[sh, 'r_main'] - ref.loc[sh, 'r_main']
        line += f"  {'same sign✅' if (d > 0).all() or (d < 0).all() else 'mixed signs⚠️'}"
        print(line)

    print(f"\n  ⚠️ n={per.seed.nunique()} seeds. Between-seed sd in this project is about 0.015~0.018,")
    print(f"     so detection needs Δ > 0.03; a non-significant p only means “no evidence of a difference”.")
    print(f"  🔴 If the curve shows overtraining, the correct fix is an **inner validation** to decide the epoch;")
    print(f"     do not write this table's best value into the paper — it was picked on the test set.")
    print(f"\n✅ {args.prefix}_raw.csv / {args.prefix}_summary.csv")


if __name__ == '__main__':
    main()
