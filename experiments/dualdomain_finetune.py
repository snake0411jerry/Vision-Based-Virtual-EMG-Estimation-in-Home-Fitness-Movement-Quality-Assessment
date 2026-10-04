"""
dualdomain_finetune.py — dual-domain joint training + personalized fine-tuning: how good can phone deployment get?
=========================================================================
[The question this script answers]
-------------------------------------------------------------------------
The real product flow is: factory model (cross-subject) → the user does one short calibration → personalized model.
The earlier experiments only measured the first step. This one measures the second:

  **The user films one calibration with a phone and wears electrodes once, then never again — how accurate can it get?**

[Procedure (per fold)]
-------------------------------------------------------------------------
  1. Train the base model on "the other 9 subjects"
       arm=both     both OpenCap and MediaPipe skeletons used as training samples (domain generalization)
       arm=mp_only  MediaPipe only (control: does domain generalization still help after personalization?)
  2. For test subject S: split each segment in time into the first 80% / last 20%
  3. Zero-shot evaluation: test the base model directly on S's **last 20%**
  4. Fine-tune: fine-tune on S's **first 80%** (feature layers frozen, only fusion/output layers trained)
  5. Post-fine-tune evaluation: test again on the same last 20%

MediaPipe and OpenCap are each fine-tuned once (branching from the same base model),
so we can answer "after personalization, how far behind is the single camera".

⚠️ The evaluation basis is "last 20% of each segment", **different** from the "full window" basis of
   loso_multiseed / loso_dualdomain; the two must not be subtracted. This script outputs both zero-shot
   and fine-tuned numbers itself, and differences are only meaningful within the same basis.

[🔴 Key difference from phase2_finetune_loso.py: no early stopping on the test set]
-------------------------------------------------------------------------
`phase2_finetune_loso.py` fine-tunes with
    EarlyStopping(monitor='val_loss', restore_best_weights=True)
and that `val` **is exactly the 20% used to report the score**. That amounts to picking weights on the
test set — so the project's existing "fine-tuned 0.830" carries an optimistic bias.

This script **always runs the full FT_EPOCHS with no early stopping** and reports the last epoch (clean).
It also uses a callback to record the test-tail r at every epoch and additionally outputs
`r_main_oracle` (= the best epoch picked after the fact).
`r_main_oracle − r_main` is how much that leakage can inflate — a number worth knowing in itself.
**Cite `r_main` in the paper, not `r_main_oracle`.**

Usage:
    python experiments/dualdomain_finetune.py --seeds 42 1 2
    python experiments/dualdomain_finetune.py --seeds 42 --arms both --smoke
Output:
    results/dualdomain_finetune_raw.csv
    results/dualdomain_finetune_summary.csv
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
from eval_utils import (subject_id_from_path, build_ts_cols, make_windows,  # noqa: E402
                        full_metrics, temporal_split_df)
from loso_train_and_save import (build_model, FEATURE_MODE, WINDOW_SIZE,  # noqa: E402
                                 STEP_SIZE, EPOCHS, BATCH_SIZE,
                                 STATIC_COLS, LABEL_EMG_COLS)

# fine-tuning hyperparameters follow phase2_finetune_loso.py; the only difference is no early stopping (see header)
# 🔴 FT_EPOCHS=20 was inherited from phase2_finetune_loso.py, but on 2026-10-03 we measured that
#    **almost every fold is still improving at epoch 20** (best_epoch=20 in 93–100% of folds),
#    i.e. it never converged. Every "fine-tuned r" obtained with 20 epochs is a lower bound.
#    Use --ft-epochs to run until the curve flattens, and --save-curve to save the per-epoch curve.
FT_EPOCHS, FT_BATCH, FT_LR = 20, 16, 1e-5
VAL_TAIL_FRAC = 0.2

RELIABLE_8 = ('Shoulder_Y_norm,Knee_Y_norm,Ankle_Y_norm,Toe_Y_norm,'
              'Knee_X_norm,Knee_Angle_norm,Trunk_Lean_Angle_norm,Knee_Ankle_Ratio_norm')

ap = argparse.ArgumentParser()
ap.add_argument('--oc', default='Combined')
ap.add_argument('--mp', default='Combined_mediapipe')
ap.add_argument('--seeds', nargs='*', type=int, default=[42, 1, 2])
ap.add_argument('--arms', nargs='*', default=['both', 'mp_only'],
                choices=['oc_only', 'mp_only', 'both'])
ap.add_argument('--keep', default=RELIABLE_8,
                help='Feature columns to keep, comma-separated. Defaults to the high-SNR 8 dims (recommended deployment set)')
ap.add_argument('--ft-src', nargs='*', default=['mediapipe'],
                choices=['mediapipe', 'opencap', 'both'],
                help='Which domain of the test subject\'s own first 80% to use for fine-tuning. '
                     'mediapipe = deployment scenario (the user only has a phone at calibration); '
                     'both = academic upper bound (assumes the person also has multi-camera data, unavailable in practice). '
                     'Every fine-tuning source is evaluated once on the last 20% of each of the two domains.')
ap.add_argument('--ft-epochs', type=int, default=FT_EPOCHS,
                help='Number of fine-tuning epochs. Default 20 (inherited from phase2), but measured to be unconverged; '
                     'increase it for convergence experiments and combine with --save-curve.')
ap.add_argument('--save-curve', action='store_true',
                help='Save the per-epoch test-tail r to <prefix>_curve.csv to see where it converges. '
                     '⚠️ Diagnostic use only: **do not use this curve to pick the epoch**, '
                     'that is tuning on the test set. The correct use is to see where it flattens, '
                     'then rerun with that epoch count as a constant fixed in advance.')
ap.add_argument('--prefix', default='dualdomain_finetune')
ap.add_argument('--resume', action='store_true')
ap.add_argument('--smoke', action='store_true', help='Run only the first 2 folds')
args = ap.parse_args()

DIRS = {'opencap': os.path.join(DATASET_DIR, args.oc),
        'mediapipe': os.path.join(DATASET_DIR, args.mp)}
TRAIN_SRC = {'oc_only': ['opencap'], 'mp_only': ['mediapipe'],
             'both': ['opencap', 'mediapipe']}
RAW_OUT = os.path.join(RESULTS_DIR, f'{args.prefix}_raw.csv')
SUM_OUT = os.path.join(RESULTS_DIR, f'{args.prefix}_summary.csv')


class TailProbe(tf.keras.callbacks.Callback):
    """Measure the test-tail main-muscle r at the end of every epoch, to compute the oracle upper bound.

    ⚠️ This record is **only for quantifying the size of the early-stopping leak** and must never be used to pick weights.
    """

    def __init__(self, Xv, Sv, Yv):
        super().__init__()
        self.Xv, self.Sv, self.Yv, self.hist = Xv, Sv, Yv, []

    def on_epoch_end(self, epoch, logs=None):
        p = self.model.predict({'ts_input': self.Xv, 'static_input': self.Sv},
                               verbose=0, batch_size=512)
        p = p[0] if isinstance(p, list) else p
        self.hist.append(full_metrics(self.Yv, p)['r'][0])


def load(d):
    by, names = {}, {}
    for p in sorted(glob.glob(os.path.join(d, '*_Combined_Features.csv'))):
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
    subjects = sorted(set(data['opencap']) & set(data['mediapipe']))
    for s in subjects:
        if names['opencap'][s] != names['mediapipe'][s]:
            sys.exit(f'{s}: the two domains have different file lists — the premise "two skeletons of the same motion" does not hold')
    print(f'Two domains matched file by file: {len(subjects)} subjects, '
          f'{sum(len(names["opencap"][s]) for s in subjects)} segments')

    ts_cols = build_ts_cols(list(data['opencap'][subjects[0]][0].columns),
                            FEATURE_MODE, False)
    keep = [c.strip() for c in args.keep.split(',') if c.strip()]
    miss = [c for c in keep if c not in ts_cols]
    if miss:
        sys.exit(f'Columns given to --keep do not exist: {miss}')
    ts_cols = [c for c in ts_cols if c in keep]
    print(f'Features {len(ts_cols)} dims: {ts_cols}')
    print(f'Fine-tuning: {args.ft_epochs} epochs, batch {FT_BATCH}, lr {FT_LR}, **no early stopping**'
          + ('  (saving per-epoch curve)' if args.save_curve else '') + '\n')

    folds = subjects[:2] if args.smoke else subjects

    all_raw, done_keys = [], set()
    if args.resume and os.path.exists(RAW_OUT):
        prev = pd.read_csv(RAW_OUT)
        all_raw.append(prev)
        done_keys = {(str(a), int(s)) for a, s in
                     prev[['arm', 'seed']].drop_duplicates().values}
        print(f'--resume: already done {sorted(done_keys)}')
    pending = [(a, sd) for sd in args.seeds for a in args.arms
               if (a, int(sd)) not in done_keys]
    if not pending:
        print('No (arm, seed) left to run.')
        return
    print(f'{len(pending)} fold-sets to run\n')

    curves = []
    t_start, done = time.time(), 0
    for arm, seed in pending:
        t0 = time.time()
        tf.keras.utils.set_random_seed(seed)
        rows = []
        for hold in folds:
            # ---- base model: full data of the other 9 subjects ----
            tr_dfs = [df for src in TRAIN_SRC[arm]
                      for s in data[src] if s != hold for df in data[src][s]]
            big = pd.concat(tr_dfs, ignore_index=True)
            s_ts = StandardScaler().fit(big[ts_cols].values)
            s_st = StandardScaler().fit(big[STATIC_COLS].values)
            Xtr, Str, Ytr, _ = make_windows(tr_dfs, ts_cols, STATIC_COLS,
                                            LABEL_EMG_COLS, None,
                                            WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
            base = build_model(len(ts_cols), len(STATIC_COLS))
            base.fit({'ts_input': Xtr, 'static_input': Str}, {'out_emg': Ytr},
                     epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
            w0 = base.get_weights()

            # ---- 80/20 temporal split of the test subject's own data (one per domain) ----
            split, val = {}, {}
            for dm in ('mediapipe', 'opencap'):
                tr_p, va_p = [], []
                for df in data[dm][hold]:
                    a, b = temporal_split_df(df, VAL_TAIL_FRAC)
                    tr_p.append(a)
                    va_p.append(b)
                split[dm] = tr_p
                # last-20% evaluation windows: all fine-tuning sources share the same test set, so they are comparable
                val[dm] = make_windows(va_p, ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                       None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)

            # ---- every fine-tuning source branches from the same base model w0 ----
            for ft_src in args.ft_src:
                srcs = ['mediapipe', 'opencap'] if ft_src == 'both' else [ft_src]
                tr_parts = [df for dm in srcs for df in split[dm]]
                Xf, Sf, Yf, _ = make_windows(tr_parts, ts_cols, STATIC_COLS,
                                             LABEL_EMG_COLS, None,
                                             WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
                if len(Xf) == 0:
                    print(f'  ⚠️ {hold} ft_src={ft_src} has 0 fine-tuning windows, skipping')
                    continue

                base.set_weights(w0)

                # zero-shot before fine-tuning (base model, independent of ft_src, but recorded for every ft_src
                # so each row is self-contained)
                zs = {}
                for dm in ('mediapipe', 'opencap'):
                    Xv, Sv, Yv, _ = val[dm]
                    if len(Xv) == 0:
                        continue
                    q = base.predict({'ts_input': Xv, 'static_input': Sv},
                                     verbose=0, batch_size=512)
                    zs[dm] = full_metrics(Yv, q[0] if isinstance(q, list) else q)

                # freeze the feature layers and fine-tune only the fusion/output layers (same as phase2)
                for layer in base.layers:
                    layer.trainable = ('feature' not in layer.name)
                base.compile(optimizer=tf.keras.optimizers.Adam(FT_LR),
                             loss={'out_emg': 'mse'}, loss_weights={'out_emg': 3.0},
                             metrics={'out_emg': 'mae'})

                # the oracle probe is attached to "the domain visible at deployment"
                probe_dom = 'opencap' if ft_src == 'opencap' else 'mediapipe'
                Xp, Sp, Yp, _ = val[probe_dom]
                probe = TailProbe(Xp, Sp, Yp)
                base.fit({'ts_input': Xf, 'static_input': Sf}, {'out_emg': Yf},
                         epochs=args.ft_epochs, batch_size=FT_BATCH,
                         callbacks=[probe], verbose=0)   # ← EarlyStopping deliberately omitted

                if args.save_curve:
                    for ep, rv in enumerate(probe.hist, start=1):
                        curves.append({'arm': arm, 'seed': seed, 'ft_src': ft_src,
                                       'probe_domain': probe_dom, 'Subject': hold,
                                       'epoch': ep, 'r_main': rv})

                for dm in ('mediapipe', 'opencap'):
                    Xv, Sv, Yv, _ = val[dm]
                    if len(Xv) == 0 or dm not in zs:
                        continue
                    q = base.predict({'ts_input': Xv, 'static_input': Sv},
                                     verbose=0, batch_size=512)
                    after = full_metrics(Yv, q[0] if isinstance(q, list) else q)
                    before = zs[dm]
                    rows.append({
                        'arm': arm, 'seed': seed, 'ft_src': ft_src,
                        'domain': dm, 'Subject': hold,
                        'n_ft_win': len(Xf), 'n_val_win': len(Xv),
                        'r_main_zs': before['r'][0], 'r_main': after['r'][0],
                        'r_main_oracle': (float(np.max(probe.hist))
                                          if dm == probe_dom else np.nan),
                        'best_epoch': (int(np.argmax(probe.hist)) + 1
                                       if dm == probe_dom else -1),
                        'r_syn_zs': before['r'][1], 'r_syn': after['r'][1],
                        'nrmse_main_zs': before['nrmse'][0], 'nrmse_main': after['nrmse'][0],
                        'nrmse_syn_zs': before['nrmse'][1], 'nrmse_syn': after['nrmse'][1],
                    })
                # the next ft_src must start from a clean base model, with trainable restored
                for layer in base.layers:
                    layer.trainable = True
            tf.keras.backend.clear_session()

        df = pd.DataFrame(rows)
        all_raw.append(df)
        out = pd.concat(all_raw, ignore_index=True)
        out.to_csv(RAW_OUT, index=False, encoding='utf-8-sig')
        (out.groupby(['arm', 'seed', 'ft_src', 'domain'], as_index=False)
            .mean(numeric_only=True)).to_csv(SUM_OUT, index=False, encoding='utf-8-sig')
        if curves:
            pd.DataFrame(curves).to_csv(
                os.path.join(RESULTS_DIR, f'{args.prefix}_curve.csv'),
                index=False, encoding='utf-8-sig')

        done += 1
        eta = (time.time() - t_start) / done * (len(pending) - done) / 60
        bits = []
        for fs in args.ft_src:
            g = df[(df.domain == 'mediapipe') & (df.ft_src == fs)]
            if len(g):
                bits.append(f'ft={fs}:MP {g.r_main_zs.mean():.3f}->{g.r_main.mean():.3f}')
        print(f'[{done}/{len(pending)}] arm={arm:8} seed={seed}  ' + '  '.join(bits) +
              f'  ({(time.time()-t0)/60:.1f} min, about {eta:.0f} min left)', flush=True)

    out = pd.concat(all_raw, ignore_index=True)
    print('\n' + '=' * 88)
    print(f'Mean across seeds (basis: last 20% of each segment; n = {len(args.seeds)} seeds)')
    print('=' * 88)
    print(f"{'Base':<8}{'FT source':<11}{'Eval dom':<11}{'zero-shot r':>10}{'FT r':>10}"
          f"{'gain':>9}{'oracle r':>10}{'leak':>10}")
    ps = out.groupby(['arm', 'ft_src', 'domain', 'seed'],
                     as_index=False).mean(numeric_only=True)
    for arm in args.arms:
      for fs in args.ft_src:
        for dom in ('mediapipe', 'opencap'):
            q = ps[(ps.arm == arm) & (ps.ft_src == fs) & (ps.domain == dom)]
            if not len(q):
                continue
            print(f"{arm:<8}{fs:<11}{dom:<11}{q.r_main_zs.mean():>10.4f}{q.r_main.mean():>10.4f}"
                  f"{q.r_main.mean()-q.r_main_zs.mean():>+9.4f}"
                  f"{q.r_main_oracle.mean():>10.4f}"
                  f"{q.r_main_oracle.mean()-q.r_main.mean():>+10.4f}")
    print('\n“leak” = best epoch picked after the fact − full 20 epochs.')
    print('  This column is the spurious gain that phase2_finetune_loso.py\'s EarlyStopping(restore_best_weights)')
    print('  obtains by picking weights on the test set. **Cite the “FT r” column in the paper.**')
    print(f'\n[done] {RAW_OUT}')
    print(f'[done] {SUM_OUT}')


if __name__ == '__main__':
    require_dataset()
    main()
