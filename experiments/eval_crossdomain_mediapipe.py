"""
eval_crossdomain_mediapipe.py — feed MediaPipe input straight into the model trained on OpenCap
=========================================================================
[The question this script answers]
-------------------------------------------------------------------------
"The model is already trained (on OpenCap multi-camera skeletons). If I now switch to a single phone camera
  and **retrain nothing**, how much does r drop?"

This is the laziest option in deployment, and the one most likely to be tried first. So measure it first.

[Method]
-------------------------------------------------------------------------
Use the 10 LOSO models in `models/loso_zeroshot/`. For each fold (the subject the model has never seen) feed:
  opencap    `Dataset/Combined/`           that subject's features (the original evaluation)
  mediapipe  `Dataset/Combined_mediapipe/` same subject, same segments, same EMG labels,
                                           only the skeleton source differs
The scaler is always the one saved when that fold was trained (**not refit**) — at deployment you only
have the training-time scaler; refitting would peek at the test data.

[This number is conservative, and the reason must be stated]
-------------------------------------------------------------------------
MediaPipe features have systematic offsets (`Knee_Angle` standing upright is 151–170° vs OpenCap's
176–180°; `compare_mediapipe_features.py` measured bias/sd up to 3.5).
After standardizing with the OpenCap scaler, these offsets push inputs into regions the model has never seen.
**This part of the loss is "domain shift", not "insufficient single-camera information"** — it vanishes
with retraining. Separating the two requires running `loso_multiseed.py --arms ... Combined_mediapipe`.

This script also offers `--refit-scaler` as a control: refit the scaler on the MediaPipe training folds
(model weights unchanged). This is **feasible** at deployment (calibrate once, no EMG needed), so its number
has practical meaning, sitting between "change nothing" and "retrain everything".

⚠️ `models/loso_zeroshot` is a low draw (see the reproducibility docs), but this script only compares
   "the same model fed two inputs", so a low baseline does not affect the difference.

Usage:
    python experiments/eval_crossdomain_mediapipe.py
    python experiments/eval_crossdomain_mediapipe.py --refit-scaler
Output:
    results/mediapipe_crossdomain_raw.csv
    results/mediapipe_crossdomain_summary.csv
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import (PIPELINE_DIR, RESULTS_DIR, DATASET_DIR, LOSO_MODEL_DIR,  # noqa: E402
                   ensure_results_dir, require_dataset)

import argparse  # noqa: E402
import glob  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, PIPELINE_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_utils import subject_id_from_path, make_windows, full_metrics  # noqa: E402
from loso_train_and_save import WINDOW_SIZE, STEP_SIZE, LABEL_EMG_COLS  # noqa: E402

ARMS = {
    'opencap': os.path.join(DATASET_DIR, 'Combined'),
    'mediapipe': os.path.join(DATASET_DIR, 'Combined_mediapipe'),
}

RELIABLE_8 = ['Shoulder_Y_norm', 'Knee_Y_norm', 'Ankle_Y_norm', 'Toe_Y_norm',
              'Knee_X_norm', 'Knee_Angle_norm', 'Trunk_Lean_Angle_norm',
              'Knee_Ankle_Ratio_norm']


def load_by_subject(data_dir):
    by = {}
    for p in sorted(glob.glob(os.path.join(data_dir, '*_Combined_Features.csv'))):
        by.setdefault(subject_id_from_path(p), []).append((p, pd.read_csv(p)))
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refit-scaler', action='store_true',
                    help='Refit the scaler on the MediaPipe training folds (model weights unchanged)')
    args = ap.parse_args()

    ensure_results_dir()
    for name, d in ARMS.items():
        if not os.path.isdir(d):
            sys.exit(f'Not found: {d} (arm={name}). '
                     f'For the MediaPipe set, run pipeline/align_mediapipe_trc.py first, '
                     f'then pipeline/build_combined_mediapipe.py.')

    spec = joblib.load(os.path.join(LOSO_MODEL_DIR, 'feature_spec.pkl'))
    ts_cols, st_cols = spec['ts_cols'], spec['static_cols']
    print(f'Model input: {len(ts_cols)} dims')

    data = {k: load_by_subject(v) for k, v in ARMS.items()}
    subjects = sorted(set(data['opencap']) & set(data['mediapipe']))
    only = (set(data['opencap']) ^ set(data['mediapipe']))
    if only:
        print(f'[warning] subjects present on only one side (excluded): {sorted(only)}')

    rows = []
    for s in subjects:
        mpath = os.path.join(LOSO_MODEL_DIR, f'loso_without_{s}.keras')
        if not os.path.exists(mpath):
            print(f'[warning] {mpath} not found, skipping {s}')
            continue
        model = tf.keras.models.load_model(mpath, compile=False)
        s_ts = joblib.load(os.path.join(LOSO_MODEL_DIR, f'loso_scaler_ts_{s}.pkl'))
        s_st = joblib.load(os.path.join(LOSO_MODEL_DIR, f'loso_scaler_static_{s}.pkl'))

        line = f'{s}  '
        for arm in ARMS:
            dfs = [d for _, d in data[arm][s]]
            use_ts, use_st = s_ts, s_st
            if args.refit_scaler and arm == 'mediapipe':
                # refit on MediaPipe using "the training subjects of the same fold"; never touch the test subject
                tr = pd.concat([d for k in data[arm] if k != s
                                for _, d in data[arm][k]], ignore_index=True)
                use_ts = StandardScaler().fit(tr[ts_cols].values)
                use_st = StandardScaler().fit(tr[st_cols].values)

            X, S, Y, _ = make_windows(dfs, ts_cols, st_cols, LABEL_EMG_COLS,
                                      None, WINDOW_SIZE, STEP_SIZE, use_ts, use_st)
            p = model.predict({'ts_input': X, 'static_input': S},
                              verbose=0, batch_size=512)
            p = p[0] if isinstance(p, list) else p
            m = full_metrics(Y, p)
            rows.append({'Subject': s, 'arm': arm, 'n_win': len(X),
                         'r_main': m['r'][0], 'r_syn': m['r'][1],
                         'nrmse_main': m['nrmse'][0], 'nrmse_syn': m['nrmse'][1]})
            line += f'{arm} r={m["r"][0]:.3f}  '
        print(line, flush=True)
        tf.keras.backend.clear_session()

    raw = pd.DataFrame(rows)
    tag = '_refit' if args.refit_scaler else ''
    raw.to_csv(os.path.join(RESULTS_DIR, f'mediapipe_crossdomain{tag}_raw.csv'),
               index=False, encoding='utf-8-sig')

    piv = raw.pivot(index='Subject', columns='arm',
                    values=['r_main', 'r_syn', 'nrmse_main', 'nrmse_syn'])
    out = []
    for met in ('r_main', 'r_syn', 'nrmse_main', 'nrmse_syn'):
        a, b = piv[(met, 'opencap')], piv[(met, 'mediapipe')]
        d = b - a
        out.append({'metric': met, 'opencap': a.mean(), 'mediapipe': b.mean(),
                    'delta': d.mean(), 'delta_sd': d.std(ddof=1),
                    'delta_worst': d.min() if met.startswith('r_') else d.max(),
                    'worst_subject': d.idxmin() if met.startswith('r_') else d.idxmax()})
    sm = pd.DataFrame(out)
    sm.to_csv(os.path.join(RESULTS_DIR, f'mediapipe_crossdomain{tag}_summary.csv'),
              index=False, encoding='utf-8-sig')

    mode = ('MediaPipe scaler refit' if args.refit_scaler
            else 'OpenCap scaler kept (nothing changed)')
    print('\n' + '=' * 78)
    print(f'OpenCap model applied directly  ({mode})  n = {len(subjects)} subjects')
    print('=' * 78)
    print(f"{'Metric':<12}{'OpenCap':>10}{'MediaPipe':>11}{'Diff':>9}{'Diff sd':>10}"
          f"{'Worst':>9}  ")
    for _, r in sm.iterrows():
        print(f"{r['metric']:<12}{r['opencap']:>10.4f}{r['mediapipe']:>11.4f}"
              f"{r['delta']:>+9.4f}{r['delta_sd']:>10.4f}"
              f"{r['delta_worst']:>+9.4f}  {r['worst_subject']}")
    print('\nPer-subject main-muscle r:')
    for s in subjects:
        a = piv[('r_main', 'opencap')][s]
        b = piv[('r_main', 'mediapipe')][s]
        print(f'   {s}  {a:.3f} -> {b:.3f}  ({b - a:+.3f})')
    print('\nReading: this difference contains both “domain shift” and “insufficient single-camera information”, which this experiment cannot separate.')
    print('      To separate them, run loso_multiseed.py with Combined_mediapipe as an independent arm and retrain.')


if __name__ == '__main__':
    require_dataset()
    main()
