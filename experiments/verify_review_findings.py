"""
verify_review_findings.py — re-verify issues 2 and 3 of the external audit report on the MVC-corrected data
=========================================================================
Corresponds to the external audit report `MODEL_REVIEW.md` (outside the repo):

  Issue 2: the model cannot tell people apart (the auditor measured a rank correlation of −0.05)
  Issue 3: the eccentric/concentric asymmetry argument does not hold (the auditor measured direction correct 5/10, correlation +0.05)

⚠️ The auditor's numbers come from the **0804 dataset** (before the MVC fix). This script reruns them on the corrected data
   with the `models/loso_zeroshot/` retrained on 8/7, to decide whether the conclusions still hold.

The auditor's definitions are reused (see the MODEL_REVIEW appendix) to ensure comparability:

  **Asymmetry** — split the knee angle into 10 bins, and in each bin compare the mean of "concentric (knee angle increasing = rising)"
                with "eccentric (knee angle decreasing = descending)", taking the difference divided by the signal's standard deviation.
                Computed on the knee angle itself it must be 0 (a purely geometric quantity gives the same values up and down).

  **Partial correlation** — after removing the linear effect of the knee angle, how much correlation remains between the model
                prediction and the true EMG. Method: regress each on the knee angle, take residuals, correlate the residuals.

  **Knee angle + velocity baseline** — fit a linear [knee angle, knee-angle velocity] -> EMG model on **that subject's own** data.
                This is an unfair advantage (the auditor noted it too), but the point is: a three-parameter linear model cannot
                conjure something absent from the data, so if it reproduces the phenomenon, the phenomenon essentially lives in kinematics.

Output: verify_review_findings.csv (per-subject detail)
"""

import glob
import os
import sys

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'pipeline'))
sys.path.insert(0, HERE)

from paths import COMBINED_DIR, MODELS_DIR                                   # noqa: E402
from eval_utils import subject_id_from_path, make_windows                    # noqa: E402
from loso_train_and_save import (WINDOW_SIZE, STEP_SIZE, STATIC_COLS,        # noqa: E402
                                 LABEL_EMG_COLS)

LOSO_DIR = os.path.join(MODELS_DIR, 'loso_zeroshot')
N_BINS = 10


def asymmetry(signal, knee, knee_vel, n_bins=N_BINS):
    """The audit report's asymmetry definition: compare concentric vs eccentric per knee-angle bin, difference divided by the signal's standard deviation."""
    sd = np.std(signal)
    if sd < 1e-9:
        return 0.0
    edges = np.quantile(knee, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    diffs = []
    for i in range(n_bins):
        m = (knee >= edges[i]) & (knee < edges[i + 1])
        con = m & (knee_vel > 0)      # knee angle increasing = rising = concentric
        ecc = m & (knee_vel < 0)      # knee angle decreasing = descending = eccentric
        if con.sum() < 5 or ecc.sum() < 5:
            continue
        diffs.append(signal[con].mean() - signal[ecc].mean())
    return float(np.mean(diffs) / sd) if diffs else 0.0


def partial_corr(a, b, ctrl):
    """Correlation between a and b after removing the linear effect of ctrl."""
    def resid(y):
        A = np.column_stack([ctrl, np.ones(len(ctrl))])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        return y - A @ coef
    ra, rb = resid(a), resid(b)
    if ra.std() < 1e-9 or rb.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    spec = joblib.load(os.path.join(LOSO_DIR, 'feature_spec.pkl'))
    ts_cols = spec['ts_cols']

    files = sorted(glob.glob(os.path.join(COMBINED_DIR, '*_Combined_Features.csv')))
    groups = [subject_id_from_path(p) for p in files]
    subjects = sorted(set(groups))
    dfs = {p: pd.read_csv(p) for p in files}

    rows = []
    for subj in subjects:
        mp = os.path.join(LOSO_DIR, f'loso_without_{subj}.keras')
        if not os.path.exists(mp):
            print(f'⚠️ missing {mp}, skipping')
            continue
        s_ts = joblib.load(os.path.join(LOSO_DIR, f'loso_scaler_ts_{subj}.pkl'))
        s_st = joblib.load(os.path.join(LOSO_DIR, f'loso_scaler_static_{subj}.pkl'))
        model = tf.keras.models.load_model(mp)

        va = [dfs[p] for p, g in zip(files, groups) if g == subj]
        X, S, Y, _ = make_windows(va, ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                  None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
        pred = model.predict({'ts_input': X, 'static_input': S}, verbose=0)
        pred = (pred[0] if isinstance(pred, list) else pred)[:, 0]
        true = Y[:, 0]

        # knee angle for each window (the window-end frame, synchronized with the label)
        knee, kvel = [], []
        for d in va:
            ka = d['Knee_Angle_norm'].values * 180.0
            kv = np.concatenate([[0.0], np.diff(ka)])
            n = len(d)
            for s in range(0, n - WINDOW_SIZE + 1, STEP_SIZE):
                knee.append(ka[s + WINDOW_SIZE - 1])
                kvel.append(kv[s + WINDOW_SIZE - 1])
        knee, kvel = np.asarray(knee), np.asarray(kvel)
        n = min(len(knee), len(true))
        knee, kvel, true, pred = knee[:n], kvel[:n], true[:n], pred[:n]

        # knee angle + velocity baseline (fit on that subject's own data)
        A = np.column_stack([knee, kvel, np.ones(n)])
        coef, *_ = np.linalg.lstsq(A, true, rcond=None)
        kv_fit = A @ coef

        rows.append({
            'Subject': subj, 'n_win': n,
            'true_level': float(true.mean()), 'model_level': float(pred.mean()),
            'r_main': float(np.corrcoef(true, pred)[0, 1]),
            'asym_true': asymmetry(true, knee, kvel),
            'asym_model': asymmetry(pred, knee, kvel),
            'asym_knee': asymmetry(knee, knee, kvel),
            'asym_knee_plus_vel': asymmetry(kv_fit, knee, kvel),
            'partial_r_model': partial_corr(pred, true, knee),
            'partial_r_knee_plus_vel': partial_corr(kv_fit, true, knee),
        })
        tf.keras.backend.clear_session()
        print(f'  {subj} done', flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(HERE, 'verify_review_findings.csv'), index=False, encoding='utf-8-sig')
    pd.set_option('display.width', 200)
    pd.set_option('display.unicode.east_asian_width', True)

    # ---------------- issue 2 ----------------
    print('\n' + '=' * 88)
    print('Issue 2: the model cannot tell people apart')
    print('=' * 88)
    d2 = d.sort_values('true_level', ascending=False).reset_index(drop=True)
    d2['true_rank'] = range(1, len(d2) + 1)
    d2['model_rank'] = d2['model_level'].rank(ascending=False).astype(int)
    print(d2[['Subject', 'true_level', 'model_level', 'true_rank', 'model_rank']].round(4).to_string(index=False))
    pear = np.corrcoef(d['true_level'], d['model_level'])[0, 1]
    spear = stats.spearmanr(d['true_level'], d['model_level']).statistic
    print(f'\n  true level vs model level: Pearson {pear:+.3f}   Spearman {spear:+.3f}')
    print(f'  (the audit report measured −0.05 on the old data)')
    print(f'  spread: true std {d["true_level"].std():.4f} / model std {d["model_level"].std():.4f}'
          f'  = {100*d["model_level"].std()/d["true_level"].std():.0f}%')

    # association with body measurements
    info = []
    for _, r in d.iterrows():
        p = sorted(glob.glob(os.path.join(COMBINED_DIR, f'{r.Subject}_*.csv')))[0]
        c = pd.read_csv(p, nrows=1)
        info.append({'Subject': r.Subject, 'weight': c['Subj_Weight_norm'][0],
                     'height': c['Subj_Height_m'][0], 'sex': c['Subj_Gender'][0]})
    m = d.merge(pd.DataFrame(info), on='Subject')
    print('\n  Association with body measurements:')
    for c in ['weight', 'height', 'sex']:
        print(f'    {c:<4} vs true level {np.corrcoef(m[c], m["true_level"])[0,1]:+.3f}   '
              f'vs model level {np.corrcoef(m[c], m["model_level"])[0,1]:+.3f}')

    # ---------------- issue 3 ----------------
    print('\n' + '=' * 88)
    print('Issue 3: eccentric/concentric asymmetry')
    print('=' * 88)
    print(d[['Subject', 'asym_true', 'asym_model', 'asym_knee', 'asym_knee_plus_vel',
             'partial_r_model', 'partial_r_knee_plus_vel']].round(3).to_string(index=False))
    pos = (d['asym_true'] > 0).sum()
    print(f'\n  ① real data: rising harder {pos} subjects / descending harder {len(d)-pos} subjects'
          f'   mean {d["asym_true"].mean():+.3f}')
    same = (np.sign(d['asym_model']) == np.sign(d['asym_true'])).sum()
    sp = ((d['asym_true'] > 0) & (d['asym_model'] > 0)).sum()
    sn = ((d['asym_true'] < 0) & (d['asym_model'] < 0)).sum()
    print(f'  ② model direction correct {same}/{len(d)}'
          f' (true positive {sp}/{pos}, true negative {sn}/{len(d)-pos})')
    print(f'     model asymmetry vs true asymmetry correlation '
          f'{np.corrcoef(d["asym_model"], d["asym_true"])[0,1]:+.3f}   (audit report, old data +0.05)')
    print(f'  ③ knee angle\'s own asymmetry mean {d["asym_knee"].mean():+.4f} (should be 0)')
    kv_ok = (np.sign(d['asym_knee_plus_vel']) == np.sign(d['asym_true'])).sum()
    print(f'     knee angle + velocity direction correct {kv_ok}/{len(d)}   correlation '
          f'{np.corrcoef(d["asym_knee_plus_vel"], d["asym_true"])[0,1]:+.3f}   (audit report, old data +0.99)')
    print(f'  ④ partial correlation (explanatory power after removing the knee angle): model {d["partial_r_model"].mean():.3f}   '
          f'knee angle + velocity {d["partial_r_knee_plus_vel"].mean():.3f}   '
          f'model higher in {(d["partial_r_model"] > d["partial_r_knee_plus_vel"]).sum()}/{len(d)} subjects')
    print(f'\n✅ verify_review_findings.csv')


if __name__ == '__main__':
    main()
