"""
pipeline_variance_converge.py — converged rerun of slide 22's "fine-tuned 0.830"
=========================================================================
[Background]
-------------------------------------------------------------------------
The "fine-tuned 0.83" on slide 22 of the IC3MT talk comes from `pv_mvcfix_finetune.csv`
(`pipeline_variance.py`: every seed retrains 10 LOSO folds → fine-tunes 10 subjects),
5 seeds: before fine-tuning 0.7885 ± 0.0070 → after **0.8305 ± 0.0048**.

⚠️ It is not `phase2_multiseed_summary.csv` (that one is 0.814, using the fixed
   `models/loso_zeroshot` weights, a different basis). `REPRODUCE_IC3MT2026.md`
   originally cited the wrong source; corrected on 2026-10-04.

That protocol has two problems (found in the MediaPipe experiments on 2026-10-03/04, see
`MEDIAPIPE_EXPERIMENT.md` §9.1d / §9.1e):

  1. FT_EPOCHS = 20 is **not converged**. In comparable settings 93–100% of folds are still improving at epoch 20;
     running to 100 epochs raises r by about 0.05.
  2. The val of `EarlyStopping(monitor='val_loss', restore_best_weights=True)` is exactly the test tail used to
     report the score — weights are picked on the test set. At 20 epochs the curve rises monotonically so it has no
     effect, but **with longer runs, once the curve flattens or turns down, it starts inflating the score**.

[This script]
-------------------------------------------------------------------------
Each fold trains one LOSO base model, saves its weights w0, then branches two paths from w0:

  (a) original  exactly the original protocol: 20 epochs + early stopping on the test tail
                → should reproduce ≈0.830 (the in-run baseline)
  (b) converge  runs --ft-epochs (default 150) with **no early stopping**,
                recording the test-tail r at every epoch

Both paths share the same base model and the same test set, so (b) − (a) is a clean paired difference,
unaffected by cross-run dispersion (≈0.025).

🔴 The per-epoch curve is **diagnostic**: it shows where the curve flattens. It must not be used to pick the epoch
   (that is tuning on the test set). Cite the last epoch of (b).

Usage:
    python experiments/pipeline_variance_converge.py --seeds 42 1 2
    python experiments/pipeline_variance_converge.py --seeds 42 --smoke
Output:
    results/<prefix>_raw.csv     (a)/(b) results for each (seed, subject)
    results/<prefix>_curve.csv   per-epoch curve of (b)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import PIPELINE_DIR, RESULTS_DIR, ensure_results_dir, require_dataset  # noqa: E402

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
                        temporal_split_df, full_metrics)
from loso_train_and_save import (build_model, FEATURE_MODE, WINDOW_SIZE,  # noqa: E402
                                 STEP_SIZE, EPOCHS, BATCH_SIZE,
                                 STATIC_COLS, LABEL_EMG_COLS, DATA_DIR)
from phase2_finetune_loso import VAL_TAIL_FRAC, FT_EPOCHS, FT_BATCH, FT_LR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--seeds', nargs='*', type=int, default=[42, 1, 2])
ap.add_argument('--ft-epochs', type=int, default=150,
                help='Number of epochs for the converge branch (the original protocol uses 20)')
ap.add_argument('--prefix', default='pv_converge')
ap.add_argument('--resume', action='store_true')
ap.add_argument('--smoke', action='store_true', help='Run only the first 2 folds')
args = ap.parse_args()

RAW_OUT = os.path.join(RESULTS_DIR, f'{args.prefix}_raw.csv')
CURVE_OUT = os.path.join(RESULTS_DIR, f'{args.prefix}_curve.csv')


def predict(model, X, S):
    p = model.predict({'ts_input': X, 'static_input': S}, verbose=0, batch_size=512)
    return p[0] if isinstance(p, list) else p


class TailProbe(tf.keras.callbacks.Callback):
    """Measure the test-tail main-muscle r every epoch. ⚠️ Diagnostic only; never used to pick weights."""

    def __init__(self, X, S, Y):
        super().__init__()
        self.X, self.S, self.Y, self.hist = X, S, Y, []

    def on_epoch_end(self, epoch, logs=None):
        self.hist.append(full_metrics(self.Y, predict(self.model, self.X, self.S))['r'][0])


def prep_finetune(model):
    """Freeze the feature layers, train only the fusion/output layers, and switch to the fine-tuning optimizer (same as phase2)."""
    for layer in model.layers:
        layer.trainable = ('feature' not in layer.name)
    model.compile(optimizer=tf.keras.optimizers.Adam(FT_LR),
                  loss={'out_emg': 'mse'}, loss_weights={'out_emg': 3.0},
                  metrics={'out_emg': 'mae'})


def run_seed(seed, file_paths, groups, subjects, dfs, ts_cols):
    tf.keras.utils.set_random_seed(seed)
    rows, curves = [], []
    for hold in subjects:
        tr_dfs = [dfs[p] for p, g in zip(file_paths, groups) if g != hold]
        va_dfs = [dfs[p] for p, g in zip(file_paths, groups) if g == hold]

        big = pd.concat(tr_dfs, ignore_index=True)
        s_ts = StandardScaler().fit(big[ts_cols].values)
        s_st = StandardScaler().fit(big[STATIC_COLS].values)
        Xtr, Str, Ytr, _ = make_windows(tr_dfs, ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                        None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
        model = build_model(len(ts_cols), len(STATIC_COLS))
        model.fit({'ts_input': Xtr, 'static_input': Str}, {'out_emg': Ytr},
                  epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        w0 = model.get_weights()

        # the test subject: first 80% of each segment for fine-tuning, last 20% for evaluation (same as the original protocol)
        tr_p, va_p = [], []
        for df in va_dfs:
            a, b = temporal_split_df(df, VAL_TAIL_FRAC)
            tr_p.append(a)
            va_p.append(b)
        Xf, Sf, Yf, _ = make_windows(tr_p, ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                     None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
        Xv, Sv, Yv, _ = make_windows(va_p, ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                     None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
        if len(Xv) == 0 or len(Xf) == 0:
            print(f'  ⚠️ {hold} has 0 windows, skipping')
            tf.keras.backend.clear_session()
            continue
        before = full_metrics(Yv, predict(model, Xv, Sv))

        # ---- (a) original protocol: 20 epochs + early stopping on the test tail ----
        model.set_weights(w0)
        prep_finetune(model)
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                              restore_best_weights=True)
        model.fit({'ts_input': Xf, 'static_input': Sf}, {'out_emg': Yf},
                  epochs=FT_EPOCHS, batch_size=FT_BATCH,
                  validation_data=({'ts_input': Xv, 'static_input': Sv}, {'out_emg': Yv}),
                  callbacks=[es], verbose=0)
        orig = full_metrics(Yv, predict(model, Xv, Sv))
        orig_stopped = int(es.stopped_epoch) if es.stopped_epoch else FT_EPOCHS

        # ---- (b) converge: run the full ft_epochs, no early stopping, record every epoch ----
        for layer in model.layers:
            layer.trainable = True
        model.set_weights(w0)
        prep_finetune(model)
        probe = TailProbe(Xv, Sv, Yv)
        model.fit({'ts_input': Xf, 'static_input': Sf}, {'out_emg': Yf},
                  epochs=args.ft_epochs, batch_size=FT_BATCH,
                  callbacks=[probe], verbose=0)      # ← EarlyStopping deliberately omitted
        conv = full_metrics(Yv, predict(model, Xv, Sv))

        h = probe.hist
        rows.append({
            'seed': seed, 'Subject': hold,
            'n_ft_win': len(Xf), 'n_val_win': len(Xv),
            'r_main_before': before['r'][0],
            'r_main_orig': orig['r'][0], 'orig_stopped_epoch': orig_stopped,
            'r_main_ep20': h[19] if len(h) >= 20 else np.nan,
            'r_main_ep50': h[49] if len(h) >= 50 else np.nan,
            'r_main_ep100': h[99] if len(h) >= 100 else np.nan,
            'r_main_conv': conv['r'][0],
            'best_epoch': int(np.argmax(h)) + 1, 'r_main_oracle': float(np.max(h)),
            'r_syn_before': before['r'][1], 'r_syn_orig': orig['r'][1],
            'r_syn_conv': conv['r'][1],
            'nrmse_main_before': before['nrmse'][0], 'nrmse_main_orig': orig['nrmse'][0],
            'nrmse_main_conv': conv['nrmse'][0],
            'nrmse_syn_before': before['nrmse'][1], 'nrmse_syn_orig': orig['nrmse'][1],
            'nrmse_syn_conv': conv['nrmse'][1],
        })
        for ep, rv in enumerate(h, start=1):
            curves.append({'seed': seed, 'Subject': hold, 'epoch': ep, 'r_main': rv})
        tf.keras.backend.clear_session()
    return rows, curves


def main():
    ensure_results_dir()
    file_paths = sorted(glob.glob(os.path.join(DATA_DIR, '*_Combined_Features.csv')))
    groups = [subject_id_from_path(p) for p in file_paths]
    subjects = sorted(set(groups))
    if args.smoke:
        subjects = subjects[:2]
    dfs = {p: pd.read_csv(p) for p in file_paths}
    ts_cols = build_ts_cols(list(dfs[file_paths[0]].columns), FEATURE_MODE, False)
    print(f'Data {DATA_DIR}: {len(file_paths)} files / {len(set(groups))} subjects / features {len(ts_cols)} dims')
    print(f'(a) original {FT_EPOCHS} epochs + early stopping on the test tail  (b) {args.ft_epochs} epochs without early stopping\n')

    all_rows, all_curves, done_seeds = [], [], set()
    if args.resume and os.path.exists(RAW_OUT):
        prev = pd.read_csv(RAW_OUT)
        all_rows = prev.to_dict('records')
        done_seeds = set(int(s) for s in prev.seed.unique())
        if os.path.exists(CURVE_OUT):
            all_curves = pd.read_csv(CURVE_OUT).to_dict('records')
        print(f'--resume: seeds already done {sorted(done_seeds)}')
    pending = [s for s in args.seeds if s not in done_seeds]
    if not pending:
        print('No seed left to run.')
        return
    print(f'{len(pending)} seeds to run\n')

    t0 = time.time()
    for i, seed in enumerate(pending, 1):
        rows, curves = run_seed(seed, file_paths, groups, subjects, dfs, ts_cols)
        all_rows += rows
        all_curves += curves
        pd.DataFrame(all_rows).to_csv(RAW_OUT, index=False, encoding='utf-8-sig')
        pd.DataFrame(all_curves).to_csv(CURVE_OUT, index=False, encoding='utf-8-sig')
        d = pd.DataFrame(rows)
        eta = (time.time() - t0) / i * (len(pending) - i) / 60
        print(f'[{i}/{len(pending)}] seed={seed}  before {d.r_main_before.mean():.3f}  '
              f'(a)original {d.r_main_orig.mean():.3f}  '
              f'(b)ep20 {d.r_main_ep20.mean():.3f} → ep{args.ft_epochs} {d.r_main_conv.mean():.3f}  '
              f'({(time.time()-t0)/i/60:.1f} min/seed, about {eta:.0f} min left)', flush=True)

    out = pd.DataFrame(all_rows)
    ps = out.groupby('seed').mean(numeric_only=True)
    print('\n' + '=' * 78)
    print(f'Across seeds (n = {len(ps)}) main-muscle r, basis: last 20% of each segment')
    print('=' * 78)
    for c, lab in (('r_main_before', 'before fine-tuning (zero-shot)'),
                   ('r_main_orig', '(a) original 20ep + early stop'),
                   ('r_main_ep20', '(b) epoch 20'),
                   ('r_main_ep50', '(b) epoch 50'),
                   ('r_main_ep100', '(b) epoch 100'),
                   ('r_main_conv', f'(b) epoch {args.ft_epochs} (cite this)')):
        if c in ps and ps[c].notna().any():
            print(f'  {lab:<26} {ps[c].mean():.4f} ± {ps[c].std(ddof=1):.4f}')
    print(f'\n  share with best_epoch = {args.ft_epochs}: '
          f'{100*(out.best_epoch == args.ft_epochs).mean():.0f}% (high = still not converged)')
    print('  Reference: slide 22 cites 0.8305 ± 0.0048 (pv_mvcfix_finetune.csv, 5 seeds)')
    print(f'\n[done] {RAW_OUT}\n[done] {CURVE_OUT}')


if __name__ == '__main__':
    require_dataset()
    main()
