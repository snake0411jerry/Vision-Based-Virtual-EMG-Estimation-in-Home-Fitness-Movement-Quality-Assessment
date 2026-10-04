"""
diagnose_mediapipe_features.py — why do some features break under a single camera? Can a different definition rescue them?
=========================================================================
`compare_mediapipe_features.py` tells us **which** features agree poorly.
This script asks **why**, and **whether a different formula can rescue them**.

Background (established facts, see the handover doc and probe_camera_features.py):
  The axes are X = anterior-posterior, Y = vertical, Z = left-right. The camera is in front, so
  **the image plane contains Y and Z; depth is X.**
  Yet measured Knee_X r=0.960 (good) and Knee_Z r=0.546 (poor) — **the opposite of "the depth axis is worse".**

The explanation this script tests: **it is not an axis problem, it is a signal-to-noise problem.**
In a squat the knee moves a dozen or more centimetres anterior-posteriorly (large X signal) but only a few
centimetres laterally (small Z signal), while MediaPipe's error is roughly fixed at a few centimetres. So
although Z lies in the image plane, its SNR is worse.

[Why the distinction matters]
-------------------------------------------------------------------------
If "the depth axis is worse" → the fix is to add depth (monocular depth estimation, QR-code distance markers).
If "low-amplitude axes are worse" → **adding depth does not help**; either give up those quantities,
                                  or change the camera angle so low-amplitude quantities become high-amplitude.
The former has already been ruled out by probe_camera_features.py (removing the 8 poorly-tracked dims
costs only 0.03–0.07). This script provides quantitative evidence for the latter.

[Three things]
-------------------------------------------------------------------------
1. Per-axis SNR = OpenCap signal sd ÷ residual sd of (MediaPipe − OpenCap).
   This splits "low correlation" into "signal too small" vs "noise too large".
2. Attribute the failure of Trunk_Lean: is it the Z term inside hypot, or the definition of the Neck point?
   (MediaPipe's Neck is approximated as the shoulder midpoint; OpenCap's is an actual marker)
3. Try several alternative definitions for Trunk_Lean and Knee_Ankle_Ratio and see which agrees best.
   ⚠️ Definitions may only be chosen by "agreement", **never by EMG prediction r** —
      the latter would mean selecting the model on the test metric. The final verdict needs an independent LOSO.

Usage:
    python experiments/diagnose_mediapipe_features.py
Output:
    results/mediapipe_axis_snr.csv
    results/mediapipe_feature_variants.csv
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import DATASET_DIR, RESULTS_DIR, PIPELINE_DIR, ensure_results_dir  # noqa: E402

import os  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, PIPELINE_DIR)
import Features_insert_FIXED as fi  # noqa: E402

MP_DIR = os.path.join(DATASET_DIR, 'mediapipe_aligned')

# TRC marker number -> name (consistent with Features_insert_FIXED.columns_to_keep)
M = {'Neck': 1, 'LShoulder': 5, 'midHip': 8, 'RHip': 9, 'RKnee': 10, 'RAnkle': 11,
     'LHip': 12, 'LKnee': 13, 'LAnkle': 14, 'LBigToe': 15, 'LSmallToe': 16,
     'LHeel': 17, 'RBigToe': 18, 'RHeel': 20}


def load_markers(path):
    """Read a TRC and return {('LKnee','Y'): array, ...}."""
    df = pd.read_csv(path, sep='\t', skiprows=4)
    df = df.rename(columns={'Unnamed: 0': 'Frame', 'Unnamed: 1': 'Time'})
    df = df.dropna(subset=['Frame'])
    out = {}
    for name, idx in M.items():
        for ax in ('X', 'Y', 'Z'):
            col = f'{ax}{idx}'
            if col in df.columns:
                out[(name, ax)] = pd.to_numeric(df[col], errors='coerce').values
    return out


def ref_height(m):
    """Same as Features_insert_FIXED: the 95th percentile of (LShoulder_Y − LAnkle_Y) for the segment."""
    st = m[('LShoulder', 'Y')] - m[('LAnkle', 'Y')]
    h = float(np.nanpercentile(st, 95))
    return h if h > 1e-6 else float(np.nanmax(st))


def ang(a, b):
    """Angle (degrees) between two sets of 3D vectors. a, b shape = (n, 3)."""
    dot = np.sum(a * b, axis=1)
    n1 = np.linalg.norm(a, axis=1)
    n2 = np.linalg.norm(b, axis=1)
    return np.degrees(np.arccos(np.clip(dot / (n1 * n2 + 1e-12), -1, 1)))


def variants(m):
    """Return {variant name: time series}. Each is normalized by that source's own ref_height."""
    h = ref_height(m)
    g = lambda k, a: m[(k, a)]  # noqa: E731
    v = {}

    # ---- Trunk lean: current definition vs several alternatives ----
    tx = g('Neck', 'X') - g('midHip', 'X')
    ty = g('Neck', 'Y') - g('midHip', 'Y')
    tz = g('Neck', 'Z') - g('midHip', 'Z')
    v['Trunk_current_hypotXZ'] = np.degrees(np.arctan2(np.hypot(tx, tz), ty)) / 180.0
    v['Trunk_Xonly_signed'] = np.degrees(np.arctan2(tx, ty)) / 180.0
    v['Trunk_Xonly_abs'] = np.degrees(np.arctan2(np.abs(tx), ty)) / 180.0
    # use the shoulder midpoint instead of Neck (to see whether the Neck approximation is the culprit)
    sx = (g('LShoulder', 'X') + g('Neck', 'X')) / 2.0 - g('midHip', 'X')
    sy = (g('LShoulder', 'Y') + g('Neck', 'Y')) / 2.0 - g('midHip', 'Y')
    v['Trunk_shoulderHip_Xonly'] = np.degrees(np.arctan2(sx, sy)) / 180.0
    # use the hip->shoulder vector against vertical (sidesteps midHip's Z)
    v['Trunk_hipShoulder_Xonly'] = np.degrees(np.arctan2(
        g('LShoulder', 'X') - g('LHip', 'X'),
        g('LShoulder', 'Y') - g('LHip', 'Y'))) / 180.0

    # ---- Knee valgus: current definition (XZ-plane distance ratio) vs Z only / X only ----
    def ratio(axes):
        kd = np.sqrt(sum((g('LKnee', a) - g('RKnee', a)) ** 2 for a in axes))
        ad = np.sqrt(sum((g('LAnkle', a) - g('RAnkle', a)) ** 2 for a in axes))
        return kd / (ad + 1e-5)
    v['KneeAnkle_current_XZ'] = ratio(('X', 'Z'))
    v['KneeAnkle_Zonly'] = ratio(('Z',))
    v['KneeAnkle_Xonly'] = ratio(('X',))
    # knee distance divided by height (not by ankle distance, avoiding noise in the denominator)
    v['KneeAnkle_kneeDistOverHeight'] = np.sqrt(
        sum((g('LKnee', a) - g('RKnee', a)) ** 2 for a in ('X', 'Z'))) / h

    # ---- Knee angle: current 3D vs XY plane only (sidesteps Z) ----
    th = np.stack([g('LHip', a) - g('LKnee', a) for a in ('X', 'Y', 'Z')], 1)
    sh = np.stack([g('LAnkle', a) - g('LKnee', a) for a in ('X', 'Y', 'Z')], 1)
    v['KneeAngle_current_3D'] = ang(th, sh) / 180.0
    v['KneeAngle_XYonly'] = ang(th[:, :2], sh[:, :2]) / 180.0

    # ---- Shoulder height: current (relative to LHip) vs relative to midHip ----
    v['ShoulderY_current_relLHip'] = (g('LShoulder', 'Y') - g('LHip', 'Y')) / h
    v['ShoulderY_relMidHip'] = (g('LShoulder', 'Y') - g('midHip', 'Y')) / h
    v['NeckY_relMidHip'] = (g('Neck', 'Y') - g('midHip', 'Y')) / h
    return v


def axis_quantities(m):
    """Per-axis joint-relative-to-hip displacement, used to compute SNR."""
    h = ref_height(m)
    out = {}
    for j in ('LShoulder', 'LKnee', 'LAnkle'):
        for a in ('X', 'Y', 'Z'):
            out[f'{j}_{a}'] = (m[(j, a)] - m[('LHip', a)]) / h
    return out


def pear(a, b):
    k = np.isfinite(a) & np.isfinite(b)
    if k.sum() < 30:
        return np.nan
    a, b = a[k], b[k]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ensure_results_dir()
    if not fi.SUBJECT_TABLE:
        sys.exit('subjects.csv not found; cannot continue.')

    snr_rows, var_rows = [], []
    for subj in fi.SUBJECT_TABLE and fi.SUBJECTS:
        key = subj['key']
        entry = fi.SUBJECT_TABLE.get(key)
        if entry is None:
            continue
        for trc, seg in zip(subj['trc_order'], subj['segment_ids']):
            oc_p = os.path.join(entry['trc_dir'], trc)
            mp_p = os.path.join(MP_DIR, f'{key}_{os.path.splitext(trc)[0]}_mediapipe.trc')
            if not (os.path.exists(oc_p) and os.path.exists(mp_p)):
                continue
            mo, mm = load_markers(oc_p), load_markers(mp_p)
            n = min(len(mo[('LKnee', 'Y')]), len(mm[('LKnee', 'Y')]))
            mo = {k: v[:n] for k, v in mo.items()}
            mm = {k: v[:n] for k, v in mm.items()}

            qo, qm = axis_quantities(mo), axis_quantities(mm)
            for k in qo:
                a, b = qo[k], qm[k]
                res = b - a
                snr_rows.append({'Subject': key, 'segment': seg, 'quantity': k,
                                 'axis': k.split('_')[-1],
                                 'signal_sd': float(np.nanstd(a)),
                                 'noise_sd': float(np.nanstd(res)),
                                 'r': pear(a, b)})
            vo, vm = variants(mo), variants(mm)
            for k in vo:
                var_rows.append({'Subject': key, 'segment': seg, 'variant': k,
                                 'r': pear(vo[k], vm[k]),
                                 'signal_sd': float(np.nanstd(vo[k])),
                                 'noise_sd': float(np.nanstd(vm[k] - vo[k]))})

    snr = pd.DataFrame(snr_rows)
    snr.to_csv(os.path.join(RESULTS_DIR, 'mediapipe_axis_snr.csv'),
               index=False, encoding='utf-8-sig')
    var = pd.DataFrame(var_rows)
    var.to_csv(os.path.join(RESULTS_DIR, 'mediapipe_feature_variants.csv'),
               index=False, encoding='utf-8-sig')

    print('=' * 88)
    print('1. Per-axis signal-to-noise  signal = OpenCap sd; noise = sd of (MediaPipe − OpenCap)')
    print('   (all divided by ref_height, so units are “fraction of body height” and comparable across axes)')
    print('=' * 88)
    g = snr.groupby('quantity').agg(signal=('signal_sd', 'median'),
                                    noise=('noise_sd', 'median'),
                                    r=('r', 'median')).reset_index()
    g['SNR'] = g['signal'] / g['noise']
    g = g.sort_values('SNR', ascending=False)
    print(f"{'Quantity':<18}{'signal sd':>10}{'noise sd':>10}{'SNR':>8}{'r median':>9}")
    for _, r in g.iterrows():
        print(f"{r['quantity']:<18}{r['signal']:>10.4f}{r['noise']:>10.4f}"
              f"{r['SNR']:>8.2f}{r['r']:>9.3f}")

    print('\nSummary by axis:')
    ga = snr.groupby('axis').agg(signal=('signal_sd', 'median'),
                                 noise=('noise_sd', 'median')).reset_index()
    ga['SNR'] = ga['signal'] / ga['noise']
    for _, r in ga.iterrows():
        label = {'X': 'X anterior-posterior (depth axis)', 'Y': 'Y vertical (image plane)',
                 'Z': 'Z left-right (image plane)'}[r['axis']]
        print(f"  {label:<22} signal {r['signal']:.4f}  noise {r['noise']:.4f}  "
              f"SNR {r['SNR']:.2f}")
    print('\n  If Z\'s noise is similar to X\'s but its signal is clearly smaller → “low amplitude”, not “depth”, is the main cause.')

    print('\n' + '=' * 88)
    print('2. Agreement of alternative definitions (higher r = a single camera reproduces OpenCap\'s quantity better)')
    print('=' * 88)
    gv = var.groupby('variant').agg(r_median=('r', 'median'),
                                    r_q10=('r', lambda s: s.quantile(0.10)),
                                    r_min=('r', 'min'),
                                    signal=('signal_sd', 'median'),
                                    noise=('noise_sd', 'median')).reset_index()
    gv['SNR'] = gv['signal'] / gv['noise']
    groups = {}
    for _, r in gv.iterrows():
        groups.setdefault(r['variant'].split('_')[0], []).append(r)
    for fam, lst in groups.items():
        print(f'\n[{fam}]')
        print(f"{'  Definition':<30}{'r median':>9}{'r 10%':>10}{'r min':>9}{'SNR':>8}")
        for r in sorted(lst, key=lambda x: -x['r_median']):
            sig = '  <<< best' if r is max(lst, key=lambda x: x['r_median']) else ''
            print(f"  {r['variant']:<28}{r['r_median']:>9.3f}{r['r_q10']:>10.3f}"
                  f"{r['r_min']:>9.3f}{r['SNR']:>8.2f}{sig}")

    print('\n⚠️ “Best” here only means “reproduces OpenCap\'s quantity best”.')
    print('   Whether it improves EMG prediction requires a separate LOSO run — after choosing a definition by agreement,')
    print('   going back to EMG r to keep picking definitions is tuning on the test metric.')
    print(f'\n[done] {RESULTS_DIR}/mediapipe_axis_snr.csv')
    print(f'[done] {RESULTS_DIR}/mediapipe_feature_variants.csv')


if __name__ == '__main__':
    main()
