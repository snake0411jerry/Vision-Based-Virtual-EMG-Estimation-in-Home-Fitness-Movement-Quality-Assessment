"""
probe_camera_features.py — are the features a single camera tracks poorly actually used by the model?
=========================================================================
[Background]
-------------------------------------------------------------------------
A teammate compared single-camera MediaPipe against OpenCap multi-camera on 8 subjects / 32 sets / 324 reps:
  - tracked well: vertical (Y), knee anterior-posterior (Knee_X), knee angle r=0.974, trunk lean, knee-ankle ratio
  - tracked poorly: left-right (*_Z, r≈0.25), foot anterior-posterior (Ankle_X, Toe_X), heel rise
The conclusion was "single-camera accuracy is roughly fixed at a few centimetres: large displacements are tracked,
small ones are not".

So the real question is not "how to restore depth" but:
**does the EMG model need those poorly tracked features?** If not, depth reconstruction is unnecessary.

[Method: use the existing 10 LOSO models, no retraining at all]
-------------------------------------------------------------------------
For each fold (the subject the model has never seen), remove the information of one feature group and see how much r drops:

  zero      set the group's features to 0 in standardized space (= training-set mean), simulating "the camera sees nothing".
            ⚠️ the model has never seen constant inputs, so this may **overestimate** importance.
  shuffle   shuffle the group's whole windows across windows, preserving the value distribution but breaking the correspondence.
            ⚠️ when features are correlated, the model can recover from other columns, so this may **underestimate** importance.

Both drop little → strongly suggests depth reconstruction is unnecessary.
Both drop a lot  → no conclusion yet; "retrain without the features" is needed to see whether the model can compensate
(that is a loso_multiseed.py-level experiment taking hours).

⚠️ models/loso_zeroshot is a low draw (see the handover doc); this only compares "the difference before and after removal",
   so a low baseline does not affect the interpretation of the difference.

Usage:
    python experiments/probe_camera_features.py
Output:
    results/probe_camera_features_raw.csv      per fold × per group × per method
    results/probe_camera_features_summary.csv  mean over 10 subjects
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import (PIPELINE_DIR, RESULTS_DIR, COMBINED_DIR, LOSO_MODEL_DIR,  # noqa: E402
                   ensure_results_dir, require_dataset)

import glob
import os
import sys

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, PIPELINE_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_utils import subject_id_from_path, make_windows, full_metrics  # noqa: E402
from loso_train_and_save import WINDOW_SIZE, STEP_SIZE, LABEL_EMG_COLS  # noqa: E402

# groups follow the teammate's single-camera comparison (per-axis correlation / compensation scatter plots)
GROUPS = {
    # --- tracked poorly by a single camera ---
    'Left-right Z (r≈0.25)':      ['Shoulder_Z_norm', 'Knee_Z_norm', 'Ankle_Z_norm', 'Toe_Z_norm'],
    'Foot anterior-posterior X (r≈0.48)':  ['Ankle_X_norm', 'Toe_X_norm'],
    'Heel rise (r=0.73)':    ['L_Heel_Rise_norm', 'R_Heel_Rise_norm'],
    '★ All of the above (8 dims)':    ['Shoulder_Z_norm', 'Knee_Z_norm', 'Ankle_Z_norm', 'Toe_Z_norm',
                             'Ankle_X_norm', 'Toe_X_norm',
                             'L_Heel_Rise_norm', 'R_Heel_Rise_norm'],
    # --- tracked well by a single camera (control: removing these should hurt a lot) ---
    'Vertical Y':                ['Shoulder_Y_norm', 'Knee_Y_norm', 'Ankle_Y_norm', 'Toe_Y_norm'],
    'Knee anterior-posterior Knee_X':         ['Knee_X_norm'],
    'Knee angle':                  ['Knee_Angle_norm'],
    'Trunk lean + knee-ankle ratio':     ['Trunk_Lean_Angle_norm', 'Knee_Ankle_Ratio_norm'],
}
UNRELIABLE = '★ All of the above (8 dims)'
RELIABLE_ALL = [c for k in ('Vertical Y', 'Knee anterior-posterior Knee_X', 'Knee angle', 'Trunk lean + knee-ankle ratio')
                for c in GROUPS[k]]
GROUPS['Control: all well-tracked (8 dims)'] = RELIABLE_ALL


def main():
    ensure_results_dir()
    spec = joblib.load(os.path.join(LOSO_MODEL_DIR, 'feature_spec.pkl'))
    ts_cols, st_cols = spec['ts_cols'], spec['static_cols']
    for g, cols in GROUPS.items():
        miss = [c for c in cols if c not in ts_cols]
        if miss:
            sys.exit(f'Feature group “{g}” has columns not among the model inputs: {miss}')
    print(f'Model input {len(ts_cols)} dims: {ts_cols}\n')

    files = sorted(glob.glob(os.path.join(COMBINED_DIR, '*_Combined_Features.csv')))
    by_subj = {}
    for p in files:
        by_subj.setdefault(subject_id_from_path(p), []).append(pd.read_csv(p))

    rng = np.random.default_rng(42)
    rows = []
    for s in sorted(by_subj):
        mpath = os.path.join(LOSO_MODEL_DIR, f'loso_without_{s}.keras')
        if not os.path.exists(mpath):
            print(f'⚠️ {mpath} not found, skipping'); continue
        model = tf.keras.models.load_model(mpath, compile=False)
        s_ts = joblib.load(os.path.join(LOSO_MODEL_DIR, f'loso_scaler_ts_{s}.pkl'))
        s_st = joblib.load(os.path.join(LOSO_MODEL_DIR, f'loso_scaler_static_{s}.pkl'))
        X, S, Y, _ = make_windows(by_subj[s], ts_cols, st_cols, LABEL_EMG_COLS,
                                  None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)

        def score(Xin):
            p = model.predict({'ts_input': Xin, 'static_input': S}, verbose=0, batch_size=512)
            p = p[0] if isinstance(p, list) else p
            m = full_metrics(Y, p)
            return m['r'][0], m['nrmse'][0], m['r'][1]

        base = score(X)
        rows.append({'Subject': s, 'group': '(full 16 dims)', 'method': 'base',
                     'r_main': base[0], 'nrmse_main': base[1], 'r_syn': base[2]})
        for g, cols in GROUPS.items():
            idx = [ts_cols.index(c) for c in cols]
            Xz = X.copy(); Xz[:, :, idx] = 0.0
            Xs = X.copy(); perm = rng.permutation(len(X))
            Xs[:, :, idx] = X[perm][:, :, idx]
            for method, Xm in (('zero', Xz), ('shuffle', Xs)):
                r, n, rs = score(Xm)
                rows.append({'Subject': s, 'group': g, 'method': method,
                             'r_main': r, 'nrmse_main': n, 'r_syn': rs})
        print(f'{s}  baseline r_main={base[0]:.3f}')
        tf.keras.backend.clear_session()

    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(RESULTS_DIR, 'probe_camera_features_raw.csv'),
               index=False, encoding='utf-8-sig')

    base = raw[raw.method == 'base'].set_index('Subject')
    out = []
    for (g, m), d in raw[raw.method != 'base'].groupby(['group', 'method'], sort=False):
        d = d.set_index('Subject')
        dr = d['r_main'] - base.loc[d.index, 'r_main']
        dn = d['nrmse_main'] - base.loc[d.index, 'nrmse_main']
        out.append({'group': g, 'method': m, 'n_feat': len(GROUPS[g]),
                    'd_r_main': dr.mean(), 'd_r_main_worst': dr.min(),
                    'd_nrmse_main': dn.mean()})
    sm = pd.DataFrame(out)
    sm.to_csv(os.path.join(RESULTS_DIR, 'probe_camera_features_summary.csv'),
              index=False, encoding='utf-8-sig')

    print('\n' + '=' * 78)
    print(f'Baseline (full 16 dims) main r = {base.r_main.mean():.4f}  nRMSE = {base.nrmse_main.mean():.4f}')
    print('=' * 78)
    print(f"{'Feature group':<24}{'dim':>3}  {'Method':<8}{'Δr mean':>9}{'Δr worst':>11}{'ΔnRMSE':>9}")
    for _, r in sm.iterrows():
        print(f"{r['group']:<24}{r['n_feat']:>3}  {r['method']:<8}"
              f"{r['d_r_main']:>+9.4f}{r['d_r_main_worst']:>+11.4f}{r['d_nrmse_main']:>+9.4f}")
    print('\nDecision threshold: between-seed sd ≈ 0.015; |Δr| < 0.02 counts as “barely used”.')


if __name__ == '__main__':
    require_dataset()
    main()
