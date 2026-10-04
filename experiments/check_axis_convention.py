"""
check_axis_convention.py — coordinate-axis convention diagnosis (CHANGES_REQUIRED.md step 0)
=========================================================================
Before touching any feature-extraction code, confirm that the TRC axis convention is consistent
across all subjects.

[Why do this first]
  OpenCap's world frame is set by how the calibration board was placed at the capture site, which
  can differ between sessions. This project's data were recorded in two batches (0625 / 0707), so we
  cannot assume all 10 subjects are consistent. If someone faced a different direction, the premise
  "X = anterior-posterior, Z = left-right" no longer holds and the fix must be handled per subject.

[Criteria (two logically independent pieces of evidence)]
  1. Hip width RHip - LHip        → axis of the largest component = left-right axis; length should be 0.2~0.3 m
  2. Foot direction BigToe - Heel → axis of the largest component = anterior-posterior axis; length should be 0.15~0.25 m
  (Also check the vertical axis: Neck_Y > midHip_Y > Knee_Y > Ankle_Y and Ankle_Y ≈ 0)

  Criterion 1 can only tell "which axis is left-right"; the foot is directional and can independently
  tell "which axis is anterior-posterior". If both paths reach the same conclusion, confidence is high.

[TRC column mapping] (per columns_to_keep in Features_insert_FIXED.py)
  X1/Y1/Z1    = Neck          X9/Y9/Z9    = RHip
  X8/Y8/Z8    = midHip        X12/Y12/Z12 = LHip  (called Hip in the code)
  X13/../Z13  = LKnee (Knee)  X14/../Z14  = LAnkle (Ankle)
  X15/../Z15  = LBigToe       Y17         = LHeel_Y  ← only Y is kept; this script reads X17/Z17 itself

Output: axis_diagnosis.csv (per-TRC detail) + per-subject summary
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


import os
import sys
import glob

import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

FIXED_DIR = PIPELINE_DIR
sys.path.insert(0, FIXED_DIR)
import Features_insert_FIXED as F   # noqa: E402  provides SUBJECT_TABLE / SUBJECTS

HERE = os.path.dirname(os.path.abspath(__file__))
AXES = ['X', 'Y', 'Z']


def read_trc(path):
    """Read a TRC (skipping the 4 header lines) and return a DataFrame."""
    df = pd.read_csv(path, sep='\t', skiprows=4)
    df = df.rename(columns={'Unnamed: 0': 'Frame', 'Unnamed: 1': 'Time'})
    return df.dropna(subset=['Frame'])


def vec(df, a_idx, b_idx):
    """Return the mean vector (X, Y, Z) of marker a minus marker b. Returns None if a column is missing."""
    out = []
    for ax in AXES:
        ca, cb = f'{ax}{a_idx}', f'{ax}{b_idx}'
        if ca not in df.columns or cb not in df.columns:
            return None
        va = pd.to_numeric(df[ca], errors='coerce')
        vb = pd.to_numeric(df[cb], errors='coerce')
        out.append(float((va - vb).mean()))
    return np.array(out)


def dominant(v):
    """Axis name of the largest absolute component and its signed length."""
    i = int(np.argmax(np.abs(v)))
    return AXES[i], v[i], float(np.linalg.norm(v))


def main():
    rows = []
    for subj in F.SUBJECTS:
        key = subj['key']
        entry = F.SUBJECT_TABLE.get(key)
        if entry is None:
            print(f"⚠️ {key} not in subjects.csv, skipping")
            continue
        trc_dir = entry['trc_dir']

        for trc_name in subj['trc_order']:
            p = os.path.join(trc_dir, trc_name)
            if not os.path.exists(p):
                continue
            try:
                df = read_trc(p)
            except Exception as e:
                print(f"⚠️ Failed to read {key}/{trc_name}: {e}")
                continue

            hip = vec(df, 9, 12)      # RHip - LHip  → left-right
            foot = vec(df, 15, 17)    # LBigToe - LHeel → anterior-posterior
            if hip is None or foot is None:
                print(f"⚠️ {key}/{trc_name} is missing required marker columns")
                continue

            h_ax, h_val, h_len = dominant(hip)
            f_ax, f_val, f_len = dominant(foot)

            # Vertical-axis check: values should decrease from top to bottom
            ys = {}
            for nm, idx in [('Neck', 1), ('midHip', 8), ('Knee', 13), ('Ankle', 14)]:
                col = f'Y{idx}'
                ys[nm] = float(pd.to_numeric(df[col], errors='coerce').mean()) \
                    if col in df.columns else np.nan
            y_ok = (ys['Neck'] > ys['midHip'] > ys['Knee'] > ys['Ankle'])

            rows.append({
                'Subject': key, 'TRC': trc_name,
                'lat_axis': h_ax, 'hip_width_m': round(h_len, 3), 'hip_signed': round(h_val, 3),
                'ap_axis': f_ax, 'foot_len_m': round(f_len, 3), 'foot_signed': round(f_val, 3),
                'y_monotonic': y_ok, 'ankle_y': round(ys['Ankle'], 3),
                'axes_distinct': (h_ax != f_ax) and 'Y' not in (h_ax, f_ax),
            })

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, 'axis_diagnosis.csv')
    df.to_csv(out, index=False, encoding='utf-8-sig')

    print(f"\n{'='*76}\n📐 Per-subject summary ({len(df)} TRCs in total)\n{'='*76}")
    print(f"{'Subject':<7}{'n':>3}  {'LR axis':<8}{'Hip w(m)':>10}  {'AP axis':<8}{'Foot l(m)':>10}"
          f"  {'Vert OK':>7}{'Distinct':>7}")
    print('-' * 76)
    for key, g in df.groupby('Subject', sort=True):
        lat = g['lat_axis'].unique()
        ap = g['ap_axis'].unique()
        lat_s = lat[0] if len(lat) == 1 else 'mixed:' + '/'.join(lat)
        ap_s = ap[0] if len(ap) == 1 else 'mixed:' + '/'.join(ap)
        flag = '' if (len(lat) == 1 and len(ap) == 1 and g['axes_distinct'].all()
                      and g['y_monotonic'].all()) else '   ⚠️'
        print(f"{key:<7}{len(g):>3}  {lat_s:<8}{g['hip_width_m'].mean():>10.3f}"
              f"  {ap_s:<8}{g['foot_len_m'].mean():>10.3f}"
              f"  {str(g['y_monotonic'].all()):>7}{str(g['axes_distinct'].all()):>7}{flag}")

    print(f"\n{'='*76}\n🔎 Global consistency check\n{'='*76}")
    lat_all = sorted(df['lat_axis'].unique())
    ap_all = sorted(df['ap_axis'].unique())
    print(f"  Left-right axis (main component of hip width): {lat_all}")
    print(f"  Anterior-posterior axis (main component of foot direction): {ap_all}")
    print(f"  Hip width range: {df['hip_width_m'].min():.3f} ~ {df['hip_width_m'].max():.3f} m"
          f"   (anatomically expected 0.20~0.30)")
    print(f"  Foot length range: {df['foot_len_m'].min():.3f} ~ {df['foot_len_m'].max():.3f} m"
          f"   (anatomically expected 0.15~0.25)")
    print(f"  Vertical ordering holds: {df['y_monotonic'].sum()}/{len(df)}")
    print(f"  Mean Ankle_Y {df['ankle_y'].mean():.3f} m (should be close to 0 = ground)")

    consistent = (len(lat_all) == 1 and len(ap_all) == 1
                  and df['axes_distinct'].all() and df['y_monotonic'].all())
    print()
    if consistent:
        print(f"✅ All {len(df)} TRCs consistent: left-right axis = {lat_all[0]}, anterior-posterior axis = {ap_all[0]}, vertical = Y")
        if ap_all[0] == 'X':
            print("   → Matches the premise of the internal audit report; the fix can be applied safely.")
        else:
            print(f"   ⚠️ But the anterior-posterior axis is {ap_all[0]}, not the X assumed by that document — the fix must be adjusted!")
    else:
        print("⚠️ Inconsistent! The following TRCs need individual review:")
        bad = df[~(df['axes_distinct'] & df['y_monotonic'])]
        if len(bad):
            print(bad[['Subject', 'TRC', 'lat_axis', 'ap_axis',
                       'y_monotonic', 'axes_distinct']].to_string(index=False))
        for k, v in [('Left-right axis', 'lat_axis'), ('Anterior-posterior axis', 'ap_axis')]:
            if df[v].nunique() > 1:
                print(f"\n  {k} disagreement details:")
                print(df.groupby([v, 'Subject']).size().to_string())

    # Sign consistency: determines whether +X is the facing direction or the opposite
    print(f"\n{'='*76}\n🧭 Facing sign (sign along the same axis; decides whether the person faces +X or −X)\n{'='*76}")
    piv = df.groupby('Subject').agg(
        hip_sign=('hip_signed', lambda s: '+' if s.mean() > 0 else '−'),
        foot_sign=('foot_signed', lambda s: '+' if s.mean() > 0 else '−'))
    print(piv.to_string())
    if piv['foot_sign'].nunique() > 1:
        print("\n  ⚠️ Foot-direction signs disagree → some subjects face the opposite direction.")
        print("     This does not affect this fix (hypot and the 3D angle are both sign-independent),")
        print("     but if a signed anterior-posterior displacement is used as a feature later, orientation must be aligned first.")
    else:
        print(f"\n  ✅ Facing direction consistent for everyone (foot points {piv['foot_sign'].iloc[0]}{ap_all[0] if len(ap_all)==1 else '?'})")

    print(f"\n✅ Details saved to {out}")


if __name__ == '__main__':
    main()
