"""
build_combined_mediapipe.py — build features from the MediaPipe monocular skeleton with the same spec as Combined/
=========================================================================
[What this program answers]
-------------------------------------------------------------------------
"If the skeleton source changes from multi-camera OpenCap to a single phone camera, how much does the EMG prediction r drop?"

For this question to be meaningful, **everything except the skeleton source must be identical**:
the same EMG labels, the same MVC denominator, the same segment pairing, the same feature formulas,
the same subject static data. So this program writes no feature logic of its own; it calls
`Features_insert_FIXED.process_subject()` directly and only swaps `trc_dir` / `trc_order`
for the corresponding files under `Dataset/mediapipe_for_model/`.

Output: `Dataset/Combined_mediapipe/` (89 files, one-to-one with `Dataset/Combined/`)

🔴 [Prerequisite: run align_mediapipe_trc.py first]
-------------------------------------------------------------------------
This program reads `Dataset/mediapipe_aligned/`, **not** `mediapipe_for_model/`.

In `mediapipe_for_model/`, 75 of the 89 trials have a time range that does not match OpenCap (3–10 seconds longer):
the teammate's export script trusted the NumFrames in the OpenCap TRC headers, and those files were cropped by hand
without updating the header. Using them directly misaligns skeleton and EMG by several seconds; r collapses for reasons unrelated to single-camera accuracy.
`align_mediapipe_trc.py` crops MediaPipe to OpenCap's range by Frame#
(Time differs by 0.000000s per frame) and verifies every trial with the left-knee-height correlation (median 0.968).

[Why swapping trc_dir is enough]
-------------------------------------------------------------------------
The aligned files are in OpenCap TRC format, with the same axes, frame count and time axis,
and the first 20 markers are numbered exactly as in OpenCap (X13 = LKnee, etc.).
The feature script only uses some of X1–Z20 (see `columns_to_keep`);
MediaPipe's extra markers 21–26 (fingers) are filtered out.
For axes and known error sources see `Dataset/mediapipe_for_model/README.md`.

[How the scale difference is handled: it isn't; normalization cancels it]
-------------------------------------------------------------------------
MediaPipe's estimated absolute scale is about 5–8% smaller than OpenCap's. But all spatial features are divided by
`ref_height` (the 95th percentile of Shoulder_Y − Ankle_Y for the segment), and ref_height is
**computed from the same skeleton**, so a constant ratio cancels automatically.
The same goes for origin translation (the teammate already aligned it, but even unaligned it would not matter; all features are relative).
**This is deliberate**: applying OpenCap's ref_height instead would sneak in multi-camera information
that is unavailable at deployment.

[Known degradations (mention them whenever citing results)]
-------------------------------------------------------------------------
1. `*_Z` (left-right) is noisy — a monocular camera cannot resolve left-right shifts of a few centimetres; knee Z vs OpenCap
   has a median correlation of only 0.61, negative for some segments. Affects `Knee_Z_norm` and `Knee_Ankle_Ratio_norm`.
2. `Knee_Angle_norm` has a systematic offset — standing knee angle is 151–170° in MediaPipe,
   176–180° in OpenCap. **Same shape, whole distribution shifted**.
   ⚠️ This matters most for "feeding MediaPipe input directly into a model trained on OpenCap"
   (see `experiments/eval_crossdomain_mediapipe.py`), but little for "retraining on MediaPipe",
   because the scaler absorbs the shift.
3. Heel points are the least accurate — the SNR of `L/R_Heel_Rise_norm` is about 1/3 of OpenCap's.

[Minor flaw in the load log]
-------------------------------------------------------------------------
`process_subject()` infers the load from the leading number of the TRC file name to print in the log (`derive_load_kg`).
MediaPipe file names start with `S01_` rather than `20`, so the log always prints 0kg.
**Since the [Leak-1] fix the load is not a model feature**; it only appears in the log and affects no output values.
This program additionally prints the correct load on the same line for manual checking.

Usage:
    python pipeline/build_combined_mediapipe.py
    python pipeline/build_combined_mediapipe.py --dry-run   # only check pairing, no files written
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

MP_DIR = os.path.join(DATASET_DIR, 'mediapipe_aligned')
OUT_DIR = os.path.join(DATASET_DIR, 'Combined_mediapipe')


def mp_name(subject_key, opencap_trc):
    """`S01` + `20.trc` -> `S01_20_mediapipe.trc` (trial name consistent with OpenCap)."""
    return f"{subject_key}_{os.path.splitext(opencap_trc)[0]}_mediapipe.trc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true', help='only check file pairing and durations; no files written')
    args = ap.parse_args()

    require_dataset('mediapipe_aligned')
    if not fi.SUBJECT_TABLE:
        sys.exit('❌ subjects.csv not read; cannot continue (contains personal data, not under version control).')

    # ---- check all files first; stop if any is missing. Never produce half a dataset and then run experiments ----
    plan, missing = [], []
    for subj in fi.SUBJECTS:
        key = subj['key']
        entry = fi.SUBJECT_TABLE.get(key)
        if entry is None:
            print(f'⚠️ subjects.csv has no {key}; skipping')
            continue
        order = []
        for trc in subj['trc_order']:
            name = mp_name(key, trc)
            if not os.path.exists(os.path.join(MP_DIR, name)):
                missing.append((key, trc, name))
            order.append(name)
        plan.append((subj, entry, order))

    if missing:
        print(f'\n❌ {len(missing)} MediaPipe TRCs missing:')
        for k, t, n in missing:
            print(f'   {k}  {t}  ->  {n}')
        sys.exit('Run pipeline/align_mediapipe_trc.py first to produce the aligned TRCs.')
    print(f'✅ Pairing check passed: all {sum(len(o) for _, _, o in plan)} trials have a matching MediaPipe TRC')

    # ---- duration health check: MediaPipe was interpolated to OpenCap's Time column, so both should be nearly identical ----
    print('\nduration comparison (MediaPipe vs OpenCap, should be < 0.02s):')
    worst = 0.0
    for subj, entry, order in plan:
        for trc, name in zip(subj['trc_order'], order):
            a = fi.trc_duration_s(os.path.join(entry['trc_dir'], trc))
            b = fi.trc_duration_s(os.path.join(MP_DIR, name))
            if a is None or b is None:
                print(f'   ⚠️ cannot read duration of {name}'); continue
            d = abs(a - b)
            worst = max(worst, d)
            if d > 0.02:
                print(f'   🚨 {subj["key"]} {trc}: OpenCap {a:.3f}s / MediaPipe {b:.3f}s (diff {d:.3f}s)')
    print(f'   max diff {worst:.4f}s')

    if args.dry_run:
        print('\n--dry-run: no files written, exiting.')
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    for subj, entry, order in plan:
        # print the correct load first (process_subject prints 0kg in the log, see the file header)
        loads = [fi.derive_load_kg(t) for t in subj['trc_order']]
        print(f"\n[{subj['key']}] actual load sequence = {loads} (ignore the 0kg in the log)")
        fi.process_subject(
            subject_key=subj['key'],
            raw_csv_path=entry['raw_csv'],        # MVC denominator: identical to the OpenCap version
            emg_env_csv_path=entry['emg_env_csv'],  # EMG labels: identical to the OpenCap version
            trc_dir=MP_DIR,
            trc_order=order,                      # ← the only difference
            segment_ids=subj['segment_ids'],
            subject_info=entry['info'],
            output_dir=OUT_DIR,
        )

    n = len([f for f in os.listdir(OUT_DIR) if f.endswith('_Combined_Features.csv')])
    ref = os.path.join(DATASET_DIR, 'Combined')
    n_ref = len([f for f in os.listdir(ref) if f.endswith('_Combined_Features.csv')]) \
        if os.path.isdir(ref) else -1
    print(f'\n🚀 Done: {OUT_DIR}  {n} files in total (Combined/ has {n_ref} files)')
    if n != n_ref:
        print('🚨 File counts differ — the two sets cannot be compared directly; find the cause first.')


if __name__ == '__main__':
    main()
