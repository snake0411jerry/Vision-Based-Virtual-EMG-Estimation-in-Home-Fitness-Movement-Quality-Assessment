"""
check_event_alignment.py — "event alignment" health check for EMG/TRC pairing (whole dataset)
=========================================================================
Why this script exists:
-------------------------------------------------------------------------
The built-in health check in `Features_insert_FIXED.py` only compares **total duration** (0.5 s tolerance).
It has two blind spots, and both have actually happened:

  1. Total duration matches but both ends are slightly off — completely undetected.
  2. When total duration does not match, you cannot tell whether the problem is at the head or the
     tail of the file, which determines how to fix it.
     (S09 b.trc had 1.05 s extra at the head; trimming the tail by intuition would turn the health
      check green while the actual misalignment worsened from 51 frames to 82 frames.)

This script compares **motion events** instead:
  TRC side = squat lowest point in hip height (Y12)
  EMG side = burst peak of the main-muscle (Raw0) envelope
The counts should match and the timings should align.

⚠️ Reading notes:
  - Peak detection double-counts the "bounce at the bottom", so adjacent peaks closer than 1.5 s are filtered out.
  - If the two sides have **different counts**, item-by-item matching shifts by one beat and yields a
    fake misalignment of 2~4 s. Such rows are marked `Count mismatch` and **must not be used as
    evidence of misalignment** — inspect them manually.
  - The EMG burst peak and the lowest knee point are not the same instant to begin with; a residual of
    0.1~0.4 s is normal. What matters is the **relative size across segments**, not the absolute value.

Output: event_alignment_report.csv
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

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
FIXED_DIR = PIPELINE_DIR
sys.path.insert(0, FIXED_DIR)
import Features_insert_FIXED as F  # noqa: E402

MIN_REP_GAP_S = 1.5      # adjacent peaks closer than this are treated as repeated detections of the same rep
FPS = 60.0
WINDOW_SIZE = 40         # training time window; a misalignment larger than this means the whole window is off


def dedup(idx, t, depth):
    """Filter out repeated detections closer than MIN_REP_GAP_S, keeping the deeper one."""
    if len(idx) == 0:
        return idx
    keep = [idx[0]]
    for i in idx[1:]:
        if t[i] - t[keep[-1]] < MIN_REP_GAP_S:
            if depth[i] > depth[keep[-1]]:
                keep[-1] = i
        else:
            keep.append(i)
    return np.array(keep)


def trc_events(path):
    """Return (time axis, relative times of the squat lowest points)."""
    df = pd.read_csv(path, sep='\t', skiprows=4)
    df = df.rename(columns={'Unnamed: 0': 'Frame', 'Unnamed: 1': 'Time'}).dropna(subset=['Frame'])
    t = pd.to_numeric(df['Time'], errors='coerce').values
    if 'Y12' not in df.columns:
        return t, None
    hy = pd.to_numeric(df['Y12'], errors='coerce').interpolate().values
    depth = np.nanmax(hy) - hy
    pk, _ = find_peaks(depth, height=np.nanmax(depth) * 0.5, distance=int(FPS))
    pk = dedup(pk, t, depth)
    return t, t[pk] - t[0]


def emg_events(env, sid):
    s = env[env['Segment'] == sid]
    r0 = pd.to_numeric(s['Raw0'], errors='coerce').fillna(0).values
    tt = s['Time_ms'].values / 1000.0
    pk, _ = find_peaks(r0, height=np.percentile(r0, 85), distance=int(FPS))
    pk = dedup(pk, tt, r0)
    return tt, tt[pk] - tt[0]


def main():
    rows = []
    for subj in F.SUBJECTS:
        key = subj['key']
        entry = F.SUBJECT_TABLE.get(key)
        if entry is None:
            continue
        try:
            env = pd.read_csv(entry['emg_env_csv'], encoding='utf-8-sig')
        except UnicodeDecodeError:
            env = pd.read_csv(entry['emg_env_csv'], encoding='big5')

        for trc, sid in zip(subj['trc_order'], subj['segment_ids']):
            p = os.path.join(entry['trc_dir'], trc)
            rec = {'subject': key, 'trc': trc, 'seg': sid}
            if not os.path.exists(p) or sid not in set(env['Segment']):
                rec['note'] = 'File or segment does not exist'
                rows.append(rec)
                continue

            t, tev = trc_events(p)
            tt, eev = emg_events(env, sid)
            trc_dur, emg_dur = t[-1] - t[0], tt[-1] - tt[0]
            rec.update({'trc_dur': trc_dur, 'emg_dur': emg_dur,
                        'dur_diff': abs(trc_dur - emg_dur),
                        'n_trc': len(tev) if tev is not None else np.nan,
                        'n_emg': len(eev)})

            if tev is None or len(tev) == 0 or len(eev) == 0:
                rec['note'] = 'No events detected'
            elif len(tev) != len(eev):
                rec['note'] = 'Count mismatch (needs manual review; do not treat as misalignment evidence)'
                rec['head_gap_diff'] = tev[0] - eev[0]
            else:
                # compare after simulating process_subject's linear resampling
                f = emg_dur / trc_dur
                d = tev * f - eev
                rec.update({'head_gap_diff': tev[0] - eev[0],
                            'mean_err': d.mean(), 'max_abs_err': np.abs(d).max(),
                            'max_err_frames': np.abs(d).max() * FPS,
                            'note': ''})
            rows.append(rec)

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, 'event_alignment_report.csv')
    df.to_csv(out, index=False, encoding='utf-8-sig')

    pd.set_option('display.width', 220)
    ok = df[df['note'] == '']
    print("=" * 104)
    print(f"Event alignment check: {len(df)} segments, of which {len(ok)} have matching event counts on both sides and quantifiable misalignment")
    print("=" * 104)

    print("\n[A] Counts match but misalignment is large (sorted by max_abs_err, top 12)")
    top = ok.sort_values('max_abs_err', ascending=False).head(12)
    print(top[['subject', 'trc', 'seg', 'n_trc', 'dur_diff', 'head_gap_diff',
               'mean_err', 'max_abs_err', 'max_err_frames']].round(3).to_string(index=False))

    print(f"\n  Reference: WINDOW_SIZE = {WINDOW_SIZE} frames")
    over = ok[ok['max_err_frames'] > WINDOW_SIZE]
    print(f"  Segments whose maximum misalignment exceeds one time window: {len(over)}/{len(ok)}")
    if len(over):
        print(over[['subject', 'trc', 'seg', 'max_err_frames']].round(1).to_string(index=False))

    print("\n[B] Total duration differs by more than 0.5 s (flagged by the current health check)")
    bad = df[df['dur_diff'] > 0.5]
    print(bad[['subject', 'trc', 'seg', 'trc_dur', 'emg_dur', 'dur_diff',
               'n_trc', 'n_emg', 'head_gap_diff', 'note']].round(3).to_string(index=False)
          if len(bad) else "  (none)")

    print("\n[C] Event counts mismatch; needs manual review")
    mism = df[df['note'].astype(str).str.startswith('Count mismatch')]
    print(mism[['subject', 'trc', 'seg', 'dur_diff', 'n_trc', 'n_emg',
                'head_gap_diff']].round(3).to_string(index=False)
          if len(mism) else "  (none)")

    print(f"\n✅ Details written to: {out}")


if __name__ == '__main__':
    main()
