"""
loso_dualdomain.py — train on both skeleton sources together to see whether a single camera can catch up with multi-camera
=========================================================================
[Idea]
-------------------------------------------------------------------------
We have two **frame-by-frame paired** skeletons (OpenCap multi-camera, MediaPipe monocular) and **the same
EMG labels**. So both skeletons of the same motion can be used as training samples:

  training folds: 89 OpenCap segments + 89 MediaPipe segments (identical labels)
  test fold:      evaluate the same model on OpenCap and on MediaPipe separately

This does two things at once:
  1. doubles the training data (but not as independent samples; see the limitation below)
  2. forces the model to be invariant to "skeleton source" — seeing the same EMG paired with two slightly
     different inputs, it cannot rely on features that disagree between the two sources

This is the standard **domain generalization** approach, stronger than feature selection on a single source:
feature selection is a person manually deciding "which ones to drop", whereas here the model learns by itself
"not to rely on the unstable ones".

[Why this experiment is meaningful rather than cheating]
-------------------------------------------------------------------------
Using OpenCap during training is legitimate — it is a **training-stage** resource you already have.
Deployment only needs a phone (MediaPipe). So "train with OpenCap's help, deploy with a phone"
is a genuinely viable product route, not peeking at test data.

⚠️ But there is a real limitation: **the two datasets are not independent samples.** The same motion appears
   twice, so "twice the data" does not mean "twice the people recruited". The N of the learning curve does not
   grow; only the input diversity at each N does. The paper must not say "equivalent to 20 people".

[Control design]
-------------------------------------------------------------------------
  oc_only    train on OpenCap only     -> test on OpenCap / test on MediaPipe
  mp_only    train on MediaPipe only   -> test on OpenCap / test on MediaPipe
  both       train on both             -> test on OpenCap / test on MediaPipe

The cell to look at is **both -> test on MediaPipe** against **mp_only -> test on MediaPipe**:
only if the former is better does "borrowing OpenCap for training" help.
Also, **oc_only -> test on MediaPipe** is the retrained version of `eval_crossdomain_mediapipe.py`
(same scaler, same architecture) and serves as an upper bound on domain shift.

The scaler is always fit on that arm's own training folds (both: fit after merging the two domains),
never touching the test subject. Windows cross neither segments nor domains (`make_windows` windows each df separately).

Usage:
    python experiments/loso_dualdomain.py --seeds 42 1 2
    python experiments/loso_dualdomain.py --seeds 42 --smoke        # only 2 folds to test the waters
    python experiments/loso_dualdomain.py --oc Combined_trunkfix \
        --mp Combined_mediapipe_trunkfix --prefix loso_dual_trunkfix
Output:
    results/<prefix>_raw.csv      each (arm, seed, test domain, subject)
    results/<prefix>_summary.csv  10-subject mean for each (arm, seed, test domain)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import (PIPELINE_DIR, RESULTS_DIR, DATASET_DIR,  # noqa: E402
                   ensure_results_dir, require_dataset)

import argparse  # noqa: E402
import glob  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, PIPELINE_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_utils import (subject_id_from_path, build_ts_cols,  # noqa: E402
                        make_windows, full_metrics)
from loso_train_and_save import (build_model, FEATURE_MODE, WINDOW_SIZE,  # noqa: E402
                                 STEP_SIZE, EPOCHS, BATCH_SIZE,
                                 STATIC_COLS, LABEL_EMG_COLS)

ap = argparse.ArgumentParser()
ap.add_argument('--oc', default='Combined', help='OpenCap feature folder (relative to Dataset/)')
ap.add_argument('--mp', default='Combined_mediapipe', help='MediaPipe feature folder')
ap.add_argument('--seeds', nargs='*', type=int, default=[42, 1, 2])
ap.add_argument('--arms', nargs='*', default=['oc_only', 'mp_only', 'both'],
                choices=['oc_only', 'mp_only', 'both'])
ap.add_argument('--keep', default=None,
                help='Keep only these feature columns, comma-separated (for feature-subset ablation)')
ap.add_argument('--prefix', default='loso_dualdomain')
ap.add_argument('--smoke', action='store_true', help='Run only the first 2 folds to verify the script runs')
ap.add_argument('--resume', action='store_true',
                help='Resume from the existing <prefix>_raw.csv: completed (arm, seed) pairs are skipped. '
                     'Used to survive the occasional CUDA crashes of this machine\'s sm_120.')
args = ap.parse_args()

DIRS = {'opencap': os.path.join(DATASET_DIR, args.oc),
        'mediapipe': os.path.join(DATASET_DIR, args.mp)}
TRAIN_SRC = {'oc_only': ['opencap'], 'mp_only': ['mediapipe'],
             'both': ['opencap', 'mediapipe']}

RAW_OUT = os.path.join(RESULTS_DIR, f'{args.prefix}_raw.csv')
SUM_OUT = os.path.join(RESULTS_DIR, f'{args.prefix}_summary.csv')


def load(data_dir):
    """{subject: [df, ...]}, also remembering file names to check the two domains match file by file."""
    by, names = {}, {}
    for p in sorted(glob.glob(os.path.join(data_dir, '*_Combined_Features.csv'))):
        s = subject_id_from_path(p)
        by.setdefault(s, []).append(pd.read_csv(p))
        names.setdefault(s, []).append(os.path.basename(p))
    return by, names


def main():
    ensure_results_dir()
    for k, d in DIRS.items():
        if not os.path.isdir(d):
            sys.exit(f'Not found: {d} ({k})')

    data, names = {}, {}
    for k, d in DIRS.items():
        data[k], names[k] = load(d)

    # the two domains must match file by file, otherwise the premise "two skeletons of the same motion" does not hold
    subjects = sorted(set(data['opencap']) & set(data['mediapipe']))
    for s in subjects:
        a, b = names['opencap'][s], names['mediapipe'][s]
        if a != b:
            sys.exit(f'{s}: the two domains have different file lists:\n  OpenCap  {a}\n  MediaPipe {b}')
    print(f'Two domains matched file by file: {len(subjects)} subjects, '
          f'{sum(len(names["opencap"][s]) for s in subjects)} segments')

    ts_cols = build_ts_cols(list(data['opencap'][subjects[0]][0].columns),
                            FEATURE_MODE, False)
    if args.keep:
        keep = [c.strip() for c in args.keep.split(',') if c.strip()]
        miss = [c for c in keep if c not in ts_cols]
        if miss:
            sys.exit(f'Columns given to --keep do not exist: {miss}')
        ts_cols = [c for c in ts_cols if c in keep]
    print(f'Features {len(ts_cols)} dims: {ts_cols}\n')

    folds = subjects[:2] if args.smoke else subjects

    # ---- resume (--resume) ----
    # 5 seeds x 3 arms takes more than 6.5 hours, and this machine's sm_120 + TF 2.10 combination occasionally hits
    # CUDA_ERROR_UNKNOWN, killing the process outright (happened once on 2026-10-02).
    # Without resume, one crash destroys everything. The granularity is "one (arm, seed)".
    #
    # 🔴 Honest limitation: a resume is **a new process**, so arms in different chunks still carry the
    #    cross-run dispersion (≈0.025) recorded in the handover doc. Arms to be subtracted should run within one execution.
    #    The order is seed-outer, so stopping midway leaves every arm with the same number of seeds.
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
        print(f'--resume: {RAW_OUT}')
        print(f'  {len(done_keys)} (arm, seed) pairs already done, will be skipped: {sorted(done_keys)}')
    chunk = (max((int(r.get('chunk', 1)) for r in all_sum), default=0) + 1
             if all_sum else 1)

    # seed-outer: at any interruption point every arm has the same number of seeds, so partial results can still be paired
    pending = [(a, sd) for sd in args.seeds for a in args.arms
               if (a, int(sd)) not in done_keys]
    total = len(pending)
    if total == 0:
        print('No (arm, seed) left to run — everything is already in the existing CSV.')
        return
    print(f'{total} fold-sets to run (this is chunk {chunk})')
    print()
    done, t_start = 0, time.time()

    for arm, seed in pending:
        t0 = time.time()
        tf.keras.utils.set_random_seed(seed)
        rows = []
        for hold in folds:
            tr_dfs = [df for src in TRAIN_SRC[arm]
                      for s in data[src] if s != hold
                      for df in data[src][s]]
            big = pd.concat(tr_dfs, ignore_index=True)
            s_ts = StandardScaler().fit(big[ts_cols].values)
            s_st = StandardScaler().fit(big[STATIC_COLS].values)

            Xtr, Str, Ytr, _ = make_windows(tr_dfs, ts_cols, STATIC_COLS,
                                            LABEL_EMG_COLS, None,
                                            WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
            model = build_model(len(ts_cols), len(STATIC_COLS))
            model.fit({'ts_input': Xtr, 'static_input': Str}, {'out_emg': Ytr},
                      epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

            # evaluate the same model on each domain's test data
            for test_dom in ('opencap', 'mediapipe'):
                Xv, Sv, Yv, _ = make_windows(data[test_dom][hold], ts_cols,
                                             STATIC_COLS, LABEL_EMG_COLS, None,
                                             WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
                p = model.predict({'ts_input': Xv, 'static_input': Sv},
                                  verbose=0, batch_size=512)
                p = p[0] if isinstance(p, list) else p
                m = full_metrics(Yv, p)
                rows.append({'arm': arm, 'seed': seed, 'test_domain': test_dom,
                             'Subject': hold,
                             'r_main': m['r'][0], 'r_syn': m['r'][1],
                             'nrmse_main': m['nrmse'][0], 'nrmse_syn': m['nrmse'][1]})
            tf.keras.backend.clear_session()

        df = pd.DataFrame(rows)
        df['chunk'] = chunk
        all_raw.append(df)
        for dom, g in df.groupby('test_domain'):
            all_sum.append({'arm': arm, 'seed': seed, 'test_domain': dom,
                            'chunk': chunk,
                            'n_ts_feat': len(ts_cols), 'n_fold': len(g),
                            **{c: g[c].mean() for c in
                               ('r_main', 'r_syn', 'nrmse_main', 'nrmse_syn')}})
        pd.concat(all_raw, ignore_index=True).to_csv(RAW_OUT, index=False,
                                                     encoding='utf-8-sig')
        pd.DataFrame(all_sum).to_csv(SUM_OUT, index=False, encoding='utf-8-sig')

        done += 1
        eta = (time.time() - t_start) / done * (total - done) / 60
        oc = df[df.test_domain == 'opencap'].r_main.mean()
        mp = df[df.test_domain == 'mediapipe'].r_main.mean()
        print(f'[{done}/{total}] train={arm:8} seed={seed}  '
              f'test OpenCap r={oc:.3f}  test MediaPipe r={mp:.3f}  '
              f'({(time.time()-t0)/60:.1f} min, about {eta:.0f} min left)', flush=True)

    sm = pd.DataFrame(all_sum)
    print('\n' + '=' * 78)
    print(f'Mean across seeds (n = {len(args.seeds)} seeds)  main-muscle r')
    print('=' * 78)
    print(f"{'Train skeleton':<14}{'test OpenCap':>16}{'test MediaPipe':>16}{'domain gap':>12}")
    for arm in args.arms:
        g = sm[sm.arm == arm]
        a = g[g.test_domain == 'opencap'].r_main
        b = g[g.test_domain == 'mediapipe'].r_main
        print(f'{arm:<14}{a.mean():>9.3f}±{a.std(ddof=1):<6.3f}'
              f'{b.mean():>9.3f}±{b.std(ddof=1):<6.3f}{b.mean()-a.mean():>+12.3f}')

    if {'mp_only', 'both'} <= set(args.arms):
        a = sm[(sm.arm == 'mp_only') & (sm.test_domain == 'mediapipe')].r_main.mean()
        b = sm[(sm.arm == 'both') & (sm.test_domain == 'mediapipe')].r_main.mean()
        print(f'\nKey cell: testing on MediaPipe, both − mp_only = {b - a:+.4f}')
        print('  > 0 means “training together with OpenCap” helps single-camera deployment.')
        print(f'  Decision threshold: between-seed sd ≈ 0.013–0.017; |diff| < 0.02 counts as indistinguishable.')

    print(f'\n[done] {RAW_OUT}')
    print(f'[done] {SUM_OUT}')
    print('\n⚠️ The two domains are not independent samples (the same motion appears twice); '
          'do not write “equivalent to 20 people”.')


if __name__ == '__main__':
    require_dataset()
    main()
