"""
compare_mediapipe_features.py — per-feature comparison of OpenCap vs MediaPipe (no model training)
=========================================================================
Teammates already compared the marker level (per-axis coordinate correlation). But **the model consumes
features, not coordinates**: Knee_Angle uses 3D vectors, Trunk_Lean uses hypot(X,Z), Knee_Ankle_Ratio is a
ratio of two distances — these nonlinear combinations can amplify or cancel coordinate errors, so they
cannot be inferred from marker-level r.

This script computes, segment by segment over all 89 segments, three quantities for every feature that
enters the model:

  r         Pearson correlation. Does the "shape" match? The model consumes changes within a window,
            so r is the most relevant metric.
  bias      Mean difference (MediaPipe − OpenCap), in the feature's (normalized) units.
            🔴 Fatal for "feeding MediaPipe into a model trained on OpenCap" (the scaler is off),
               almost harmless for "retraining on MediaPipe" (the scaler absorbs it).
  sd_ratio  Ratio of standard deviations. <1 means MediaPipe's dynamic range is compressed
            (over-smoothed); >1 means extra noise.

The `_vel` (velocity) columns are computed too: spatial mode does not use them, but if spatial+kinematic
is run later, differentiation amplifies noise far more than position does, so we measure it now.

⚠️ This script only answers "do the features look alike", not "how much will r drop".
   Features can differ a lot without affecting prediction (the model may not use that feature), and
   vice versa. Answering the latter requires running LOSO (see the mediapipe section of the README).

Usage:
    python experiments/compare_mediapipe_features.py
Output:
    results/mediapipe_feature_agreement.csv         per segment x per feature
    results/mediapipe_feature_agreement_summary.csv summary over 89 segments
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import DATASET_DIR, RESULTS_DIR, ensure_results_dir  # noqa: E402

import os  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

OC_DIR = os.path.join(DATASET_DIR, 'Combined')
MP_DIR = os.path.join(DATASET_DIR, 'Combined_mediapipe')

# The 16 dims that enter the model (spatial). The first 8 are the set a single camera tracks “reliably”
# (see RELIABLE_ALL in experiments/probe_camera_features.py)
IN_MODEL_16 = ['L_Heel_Rise_norm', 'R_Heel_Rise_norm', 'Knee_Ankle_Ratio_norm',
               'Trunk_Lean_Angle_norm', 'Shoulder_Y_norm', 'Shoulder_Z_norm',
               'Knee_X_norm', 'Knee_Y_norm', 'Knee_Z_norm', 'Ankle_X_norm',
               'Ankle_Y_norm', 'Ankle_Z_norm', 'Toe_X_norm', 'Toe_Y_norm',
               'Toe_Z_norm', 'Knee_Angle_norm']
# Knee_Toe_Diff_* and Subj_* are blocked by eval_utils.EXCLUDE_ALWAYS and never enter the model, so they are not listed.

RELIABLE_8 = ['Shoulder_Y_norm', 'Knee_Y_norm', 'Ankle_Y_norm', 'Toe_Y_norm',
              'Knee_X_norm', 'Knee_Angle_norm', 'Trunk_Lean_Angle_norm',
              'Knee_Ankle_Ratio_norm']


def pearson(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30:
        return np.nan
    a, b = a[m], b[m]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ensure_results_dir()
    files = sorted(f for f in os.listdir(OC_DIR) if f.endswith('_Combined_Features.csv'))
    rows = []
    for fn in files:
        p_oc, p_mp = os.path.join(OC_DIR, fn), os.path.join(MP_DIR, fn)
        if not os.path.exists(p_mp):
            print(f'[warning] MediaPipe is missing {fn}, skipping')
            continue
        a, b = pd.read_csv(p_oc), pd.read_csv(p_mp)
        n = min(len(a), len(b))
        if abs(len(a) - len(b)) > 2:
            print(f'[warning] {fn} row count differs by {abs(len(a)-len(b))} ({len(a)} vs {len(b)})')
        a, b = a.iloc[:n], b.iloc[:n]

        # EMG labels must be identical — otherwise the two sets are not measuring the same thing
        for lab in ('EMG_Main_MVC', 'EMG_Compass_MVC'):
            d = float(np.max(np.abs(a[lab].values - b[lab].values)))
            if d > 1e-9:
                print(f'🚨 {lab} differs between the two sets in {fn} (max diff {d:.2e}) — labels should be identical!')

        subj, seg = fn.split('_Seg_')[0], fn.split('_Seg_')[1].split('_')[0]
        for c in a.columns:
            if not c.endswith(('_norm', '_vel')):
                continue
            if c.startswith('Subj_') or c.startswith('Knee_Toe_Diff'):
                continue      # not in the model (EXCLUDE_ALWAYS)
            x, y = a[c].values.astype(float), b[c].values.astype(float)
            rows.append({
                'Subject': subj, 'segment': int(seg), 'feature': c,
                'kind': 'norm' if c.endswith('_norm') else 'vel',
                'r': pearson(x, y),
                'bias': float(np.nanmean(y - x)),
                'oc_sd': float(np.nanstd(x)), 'mp_sd': float(np.nanstd(y)),
                'sd_ratio': float(np.nanstd(y) / (np.nanstd(x) + 1e-12)),
            })

    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(RESULTS_DIR, 'mediapipe_feature_agreement.csv'),
               index=False, encoding='utf-8-sig')

    sm = (raw.groupby(['feature', 'kind'], as_index=False)
             .agg(r_median=('r', 'median'), r_min=('r', 'min'),
                  r_q10=('r', lambda s: s.quantile(0.10)),
                  bias_mean=('bias', 'mean'),
                  bias_over_oc_sd=('bias', 'mean'),
                  sd_ratio_median=('sd_ratio', 'median'),
                  n_seg=('r', 'size')))
    # the raw magnitude of bias is hard to read; it only means something divided by OpenCap's typical sd for that feature
    oc_sd = raw.groupby('feature')['oc_sd'].median()
    sm['bias_over_oc_sd'] = sm.apply(
        lambda r: r['bias_mean'] / (oc_sd[r['feature']] + 1e-12), axis=1)
    sm = sm.sort_values(['kind', 'r_median'], ascending=[True, False])
    sm.to_csv(os.path.join(RESULTS_DIR, 'mediapipe_feature_agreement_summary.csv'),
              index=False, encoding='utf-8-sig')

    for kind, title in (('norm', 'The 16 dims entering the model (spatial)'),
                        ('vel', 'Velocity columns (unused in spatial mode, for future reference)')):
        d = sm[sm['kind'] == kind]
        print('\n' + '=' * 92)
        print(f'{title}    n = {int(d.n_seg.iloc[0])} segments')
        print('=' * 92)
        print(f"{'Feature':<26}{'r median':>8}{'r 10th pct':>13}{'r min':>8}"
              f"{'bias/sd':>10}{'sd ratio':>8}  reliable set")
        for _, x in d.iterrows():
            tag = '★' if x['feature'] in RELIABLE_8 else ''
            print(f"{x['feature']:<26}{x['r_median']:>8.3f}{x['r_q10']:>13.3f}"
                  f"{x['r_min']:>8.3f}{x['bias_over_oc_sd']:>10.2f}"
                  f"{x['sd_ratio_median']:>8.2f}  {tag}")

    d = sm[sm['kind'] == 'norm']
    rel = d[d.feature.isin(RELIABLE_8)]
    unrel = d[~d.feature.isin(RELIABLE_8)]
    print('\n' + '-' * 92)
    print(f'★ Reliable 8 dims  median of r medians = {rel.r_median.median():.3f}'
          f'  (median |bias/sd| {rel.bias_over_oc_sd.abs().median():.2f})')
    print(f'  Other 8 dims      median of r medians = {unrel.r_median.median():.3f}'
          f'  (median |bias/sd| {unrel.bias_over_oc_sd.abs().median():.2f})')
    print('\nReading: features with large bias/sd lose a lot under “apply the OpenCap model directly”,'
          'but not under “retrain on MediaPipe” — the two experiments must be run separately to tell them apart.')
    print(f'\n[done] {RESULTS_DIR}/mediapipe_feature_agreement_summary.csv')


if __name__ == '__main__':
    main()
