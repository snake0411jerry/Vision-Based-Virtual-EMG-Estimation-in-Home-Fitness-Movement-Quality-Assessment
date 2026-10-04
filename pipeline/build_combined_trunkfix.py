"""
build_combined_trunkfix.py — rebuild both feature sets with a signed Trunk_Lean
=========================================================================
`diagnose_mediapipe_features.py` found a problem with the current
`Trunk_Lean = atan2(hypot(ΔX, ΔZ), ΔY)`: **hypot is always positive**,
so backward trunk lean is folded onto the positive side. Computed for the same movement from the two skeleton sources, agreement is only r=0.354
(10th percentile −0.578); switching to `atan2(ΔX, ΔY)` (signed, Z term dropped) gives r=0.896.
Taking the absolute value is even worse (0.221) — so the problem is sign folding, not Z noise.

This program rebuilds **both** datasets with `trunk_lean_mode='signed_x'`:

    Dataset/Combined_trunkfix/            (OpenCap skeleton)
    Dataset/Combined_mediapipe_trunkfix/  (MediaPipe skeleton)

🔴 **Both must be rebuilt.** This change affects both OpenCap and MediaPipe; changing only
   the MediaPipe side would mix two variables, "changed definition" and "changed skeleton source", into later comparisons.

🔴 **`Dataset/Combined/` is not overwritten.** It is tied to the git tag `ic3mt-2026` and the SHA-256 list in
   `reproducibility/DATA_MANIFEST_ic3mt-2026.csv`;
   overwriting it would destroy the paper's reproducibility point. So output goes to new folders.

[The cost of signed_x, to be stated in the paper]
-------------------------------------------------------------------------
The hypot version is independent of the subject's facing direction (forward lean is measured whichever way the person faces).
The signed_x version assumes "the person roughly faces the camera, and positive X is forward".
This holds for this study's single-camera deployment scenario, but it is an **extra assumption**, not a pure improvement.
If arbitrary orientations need to be handled later, estimate the facing direction from the feet first and project, rather than going back to hypot.

Usage:
    python pipeline/build_combined_trunkfix.py
    python pipeline/build_combined_trunkfix.py --only opencap
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import DATASET_DIR, require_dataset  # noqa: E402

import argparse  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Features_insert_FIXED as fi  # noqa: E402
from build_combined_mediapipe import mp_name, MP_DIR  # noqa: E402

TARGETS = {
    'opencap': (None, os.path.join(DATASET_DIR, 'Combined_trunkfix')),
    'mediapipe': (MP_DIR, os.path.join(DATASET_DIR, 'Combined_mediapipe_trunkfix')),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=list(TARGETS), default=None,
                    help='rebuild only one of the sets (default: both)')
    args = ap.parse_args()

    require_dataset()
    if not fi.SUBJECT_TABLE:
        sys.exit('subjects.csv not found; cannot continue.')

    names = [args.only] if args.only else list(TARGETS)
    for name in names:
        src_dir, out_dir = TARGETS[name]
        if name == 'mediapipe' and not os.path.isdir(src_dir):
            sys.exit(f'{src_dir} not found; run pipeline/align_mediapipe_trc.py first.')
        os.makedirs(out_dir, exist_ok=True)
        print('\n' + '=' * 70)
        print(f'rebuilding {name} -> {out_dir}  (trunk_lean_mode=signed_x)')
        print('=' * 70)

        for subj in fi.SUBJECTS:
            key = subj['key']
            entry = fi.SUBJECT_TABLE.get(key)
            if entry is None:
                print(f'[warning] subjects.csv has no {key}; skipping')
                continue
            if name == 'opencap':
                trc_dir, order = entry['trc_dir'], subj['trc_order']
            else:
                trc_dir = src_dir
                order = [mp_name(key, t) for t in subj['trc_order']]
            fi.process_subject(
                subject_key=key,
                raw_csv_path=entry['raw_csv'],
                emg_env_csv_path=entry['emg_env_csv'],
                trc_dir=trc_dir,
                trc_order=order,
                segment_ids=subj['segment_ids'],
                subject_info=entry['info'],
                output_dir=out_dir,
                trunk_lean_mode='signed_x',
            )

        n = len([f for f in os.listdir(out_dir) if f.endswith('_Combined_Features.csv')])
        print(f'\n{name}: {n} files')

    # ---- health check: old and new versions should be cell-for-cell identical except for Trunk_Lean ----
    import numpy as np
    import pandas as pd
    print('\n' + '=' * 70)
    print('Health check: differences between the signed_x version and the original should appear only in Trunk_Lean_Angle_*')
    print('=' * 70)
    pairs = [('Combined', 'Combined_trunkfix'),
             ('Combined_mediapipe', 'Combined_mediapipe_trunkfix')]
    for a_dir, b_dir in pairs:
        A, B = os.path.join(DATASET_DIR, a_dir), os.path.join(DATASET_DIR, b_dir)
        if not (os.path.isdir(A) and os.path.isdir(B)):
            continue
        files = sorted(f for f in os.listdir(A) if f.endswith('_Combined_Features.csv'))
        changed, trunk_r = set(), []
        for f in files:
            if not os.path.exists(os.path.join(B, f)):
                continue
            x, y = pd.read_csv(os.path.join(A, f)), pd.read_csv(os.path.join(B, f))
            n = min(len(x), len(y))
            for c in x.columns:
                if c not in y.columns:
                    continue
                d = float(np.nanmax(np.abs(x[c].values[:n] - y[c].values[:n])))
                if d > 1e-9:
                    changed.add(c)
            trunk_r.append(float(np.nanstd(y['Trunk_Lean_Angle_norm'].values[:n])))
        unexpected = {c for c in changed if not c.startswith('Trunk_Lean')}
        print(f'  {a_dir} vs {b_dir}: changed columns = {sorted(changed)}')
        if unexpected:
            print(f'  🚨 changed but should not have: {sorted(unexpected)} — find out why before continuing.')
        else:
            print(f'  OK, only Trunk_Lean changed.'
                  f'median sd of the new Trunk_Lean_Angle_norm = {np.median(trunk_r):.4f}')


if __name__ == '__main__':
    main()
