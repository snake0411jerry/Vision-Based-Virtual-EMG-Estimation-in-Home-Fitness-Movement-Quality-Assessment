"""
align_mediapipe_trc.py — crop MediaPipe TRCs to align frame-by-frame with the OpenCap TRCs
=========================================================================
[Why this step is needed]
-------------------------------------------------------------------------
`Dataset/mediapipe_for_model/README.md` says "frame count, Frame# and Time are identical to the corresponding
OpenCap TRC". **This holds for 14 of the 89 trials and not for the other 75.**

Reason: the OpenCap TRCs in `Dataset/Open/` were cropped by hand (both head and tail), but **the header's
NumFrames was not updated** (this was already noted in the comments of `Features_insert_FIXED.trc_duration_s()`).
The teammate's export script used **the header's NumFrames**, so the resulting
MediaPipe files cover the whole original recording and are 3–10 seconds longer than the cropped OpenCap.

If fed into `process_subject()` untreated, resampling would linearly squeeze "34 seconds of skeleton" onto
"28.6 seconds of EMG" — movement and EMG misaligned by several seconds, r collapses, and the cause has nothing to do with
"single-camera accuracy". The whole experiment would be wasted.

[Good news: it can be fixed exactly, not approximately]
-------------------------------------------------------------------------
For 74 trials, the OpenCap `Time` and MediaPipe `Time` were measured to **differ by 0.000000 s per frame**
(both have Time = (Frame-1)/60). So simply cutting MediaPipe to OpenCap's range by Frame#
gives **exact frame-to-frame alignment**, with no interpolation or guessing.

[Three cases]
-------------------------------------------------------------------------
  crop   74 trials (S01-S08) MediaPipe 1..N_header contains OpenCap [fa..fb]
         -> take MediaPipe rows fa..fb.
  asis   14 trials (S09/S10) MediaPipe row count == OpenCap row count, already the cropped range
         -> use the whole file (only Frame#/Time renumbered from 1/0).
  tail    1 trial (S09 b.trc) MediaPipe has 83 extra rows = 1.3833s
         -> exactly the handover doc's note "b.trc additionally had 1.383s trimmed from its start"; the teammate used the pre-crop version.
         -> take the last N_oc rows. This trial's offset is independently confirmed by the signal-correlation search below, not assumed.

[Every trial gets a signal check, not just an arithmetic check]
-------------------------------------------------------------------------
After alignment, the Pearson r between OpenCap and MediaPipe is computed on "left knee height relative to left hip"
(Y13-Y12, the most reliable single-camera quantity). Based on the teammate's comparison it should be 0.86-0.99.
If a trial has r < 0.5 the alignment is still wrong; this program lists it and exits with a non-zero code —
**better to produce no files than data that looks normal but is actually misaligned.**

In addition, a +-120 frame shift search is run for each trial, printing "the shift with the highest correlation".
With correct alignment the best shift should be 0. This is a cross-check independent of the classification logic above.

Output: `Dataset/mediapipe_aligned/` (TRC format, Frame#/Time rewritten to match OpenCap)
        `Code/results/mediapipe_alignment_report.csv`

Usage:
    python pipeline/align_mediapipe_trc.py
    python pipeline/align_mediapipe_trc.py --shift-search 0   # skip the shift search (faster)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import DATASET_DIR, RESULTS_DIR, ensure_results_dir, require_dataset  # noqa: E402

import argparse  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Features_insert_FIXED as fi  # noqa: E402

MP_DIR = os.path.join(DATASET_DIR, 'mediapipe_for_model')
OUT_DIR = os.path.join(DATASET_DIR, 'mediapipe_aligned')
FPS = 60.0
MIN_R = 0.5          # below this value the alignment counts as failed
EXPECT_R = 0.80      # below this value only a warning (the teammate measured a lower bound of about 0.86)


def read_trc(path):
    """Return (first 6 header lines, data rows as list[str], Frame# array, Time array).

    pandas is not used because the data-row text must be preserved verbatim (avoids float re-formatting).
    """
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    head, data, frames, times = lines[:6], [], [], []
    for ln in lines[6:]:
        cells = ln.rstrip('\n').split('\t')
        if len(cells) < 2 or not cells[1].strip():
            continue
        try:
            fr = int(float(cells[0]))
            tm = float(cells[1])
        except ValueError:
            continue
        data.append(ln.rstrip('\n'))
        frames.append(fr)
        times.append(tm)
    return head, data, np.array(frames), np.array(times)


def col_series(data_lines, col_idx):
    """Get the values of one column (col_idx 0-based, 0=Frame#, 1=Time, 2=X1 ...)."""
    out = np.empty(len(data_lines))
    for i, ln in enumerate(data_lines):
        c = ln.split('\t')
        try:
            out[i] = float(c[col_idx])
        except (ValueError, IndexError):
            out[i] = np.nan
    return out


def knee_height(data_lines):
    """Left knee height relative to left hip = Y13 - Y12. Column position: Ym is at 2 + 3*(m-1) + 1."""
    y13 = col_series(data_lines, 2 + 3 * (13 - 1) + 1)
    y12 = col_series(data_lines, 2 + 3 * (12 - 1) + 1)
    return y13 - y12


def pearson(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30:
        return np.nan
    a, b = a[m], b[m]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def best_shift(ref, mp_full, start, max_shift):
    """Find the shift with the highest correlation within +-max_shift frames. Positive means MediaPipe is taken later."""
    n = len(ref)
    best_r, best_s = np.nan, 0
    for s in range(-max_shift, max_shift + 1):
        lo = start + s
        if lo < 0 or lo + n > len(mp_full):
            continue
        r = pearson(ref, mp_full[lo:lo + n])
        if np.isfinite(r) and (not np.isfinite(best_r) or r > best_r):
            best_r, best_s = r, s
    return best_r, best_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shift-search', type=int, default=120,
                    help='shift search range (frames); 0 = no search')
    ap.add_argument('--out-dir', default=OUT_DIR)
    args = ap.parse_args()

    require_dataset('mediapipe_for_model')
    ensure_results_dir()
    if not fi.SUBJECT_TABLE:
        sys.exit('subjects.csv not read (contains personal data, not under version control); cannot continue.')
    os.makedirs(args.out_dir, exist_ok=True)

    rows, bad = [], []
    for subj in fi.SUBJECTS:
        key = subj['key']
        entry = fi.SUBJECT_TABLE.get(key)
        if entry is None:
            print(f'[warning] subjects.csv has no {key}; skipping')
            continue
        for trc, seg in zip(subj['trc_order'], subj['segment_ids']):
            oc_path = os.path.join(entry['trc_dir'], trc)
            mp_name = f"{key}_{os.path.splitext(trc)[0]}_mediapipe.trc"
            mp_path = os.path.join(MP_DIR, mp_name)
            if not (os.path.exists(oc_path) and os.path.exists(mp_path)):
                bad.append((key, trc, 'file not found'))
                continue

            oc_head, oc_data, oc_fr, oc_t = read_trc(oc_path)
            mp_head, mp_data, mp_fr, mp_t = read_trc(mp_path)
            n_oc, n_mp = len(oc_data), len(mp_data)
            fa, fb = int(oc_fr[0]), int(oc_fr[-1])

            # ---- decide which rows to take ----
            if n_mp == n_oc:
                mode, lo = 'asis', 0
            elif fb <= n_mp:
                mode, lo = 'crop', fa - 1
                dt = float(np.max(np.abs(oc_t - mp_t[lo:lo + n_oc])))
                if dt > 1e-6:
                    bad.append((key, trc, f'crop but Time mismatch (diff {dt:.4f}s)'))
                    continue
            elif n_mp > n_oc:
                mode, lo = 'tail', n_mp - n_oc      # presumed to be "OpenCap additionally cropped the head"
            else:
                bad.append((key, trc, f'MediaPipe too short ({n_mp} < {n_oc})'))
                continue

            sel = mp_data[lo:lo + n_oc]

            # ---- signal check ----
            ref = knee_height(oc_data)
            r = pearson(ref, knee_height(sel))
            sh_r, sh = np.nan, 0
            if args.shift_search:
                sh_r, sh = best_shift(ref, knee_height(mp_data), lo, args.shift_search)

            rows.append({'Subject': key, 'trc': trc, 'segment': seg, 'mode': mode,
                         'n_oc': n_oc, 'n_mp': n_mp, 'oc_frame_lo': fa, 'oc_frame_hi': fb,
                         'take_from': lo, 'r_knee_height': r,
                         'best_shift_frames': sh, 'r_at_best_shift': sh_r})
            if not np.isfinite(r) or r < MIN_R:
                bad.append((key, trc, f'correlation too low after alignment r={r:.3f} (best shift {sh:+d})'))
                continue

            # ---- write out. The header is MediaPipe's, but NumFrames is corrected and
            #      Frame#/Time rewritten to OpenCap's, so the later duration health check actually works ----
            h = list(mp_head)
            c = h[2].rstrip('\n').split('\t')
            if len(c) >= 8:
                c[2] = str(n_oc)        # NumFrames
                c[7] = str(n_oc)        # OrigNumFrames
                h[2] = '\t'.join(c) + '\n'
            with open(os.path.join(args.out_dir, mp_name), 'w',
                      encoding='utf-8', newline='') as f:
                f.writelines(h)
                for i, ln in enumerate(sel):
                    cells = ln.split('\t')
                    cells[0] = str(int(oc_fr[i]))
                    cells[1] = f'{oc_t[i]:.7f}'
                    f.write('\t'.join(cells) + '\n')

    rep = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, 'mediapipe_alignment_report.csv')
    rep.to_csv(out_csv, index=False, encoding='utf-8-sig')

    print('=' * 74)
    print(f'aligned {len(rep)} / 89 trials')
    print('=' * 74)
    print('mode distribution:', rep['mode'].value_counts().to_dict())
    print('\nleft knee height correlation (OpenCap vs aligned MediaPipe)')
    print(f'  median {rep.r_knee_height.median():.3f}   '
          f'min {rep.r_knee_height.min():.3f}   max {rep.r_knee_height.max():.3f}')
    low = rep[rep.r_knee_height < EXPECT_R]
    if len(low):
        print(f'\n[note] {len(low)} trials have correlation below {EXPECT_R} (not necessarily wrong, but worth a look):')
        for _, x in low.iterrows():
            print(f'   {x.Subject} {x.trc:12} r={x.r_knee_height:.3f} '
                  f'best_shift={x.best_shift_frames:+d} (r={x.r_at_best_shift:.3f})')
    if args.shift_search:
        nz = rep[rep.best_shift_frames != 0]
        print(f'\nshift search (+-{args.shift_search} frames): best shift is 0 for '
              f'{len(rep) - len(nz)} / {len(rep)} trials')
        if len(nz):
            print('   best shift not 0:')
            for _, x in nz.iterrows():
                gain = x.r_at_best_shift - x.r_knee_height
                print(f'   {x.Subject} {x.trc:12} shift={x.best_shift_frames:+4d} '
                      f'r {x.r_knee_height:.3f} -> {x.r_at_best_shift:.3f} (+{gain:.3f})')
            print('   a shift of a few frames is usually MediaPipe\'s smoothing lag and does not affect conclusions;')
            print('   only if it exceeds +-10 frames and the correlation clearly improves should the alignment be rechecked.')

    print(f'\n[done] {out_csv}')
    print(f'[done] {args.out_dir} ({len(os.listdir(args.out_dir))} files)')

    if bad:
        print(f'\n[failed] {len(bad)} trials failed alignment, no files written:')
        for k, t, why in bad:
            print(f'   {k} {t:12} {why}')
        sys.exit('Not all alignments passed — do not run experiments with this data.')


if __name__ == '__main__':
    main()
