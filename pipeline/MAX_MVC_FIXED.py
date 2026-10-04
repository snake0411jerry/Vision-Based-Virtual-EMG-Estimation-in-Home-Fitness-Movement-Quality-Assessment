"""
MAX_MVC_FIXED.py  —  corrected MVC reference computation
=========================================================================
Fixes relative to the original `MAX_MVC.py`:

[Normalization-5] ★ The original used the 'single instantaneous peak' np.max(envelope) as the MVC denominator,
           which is extremely sensitive to noise (one spike raises the denominator and systematically lowers %MVC).
           This version instead takes 'the maximum after a moving-window (default 500ms) average', i.e. the stable plateau peak,
           which is standard practice in sEMG normalization.

[Normalization-7] 🔴 **The approach of [Normalization-5] was withdrawn on 2026-08-05** — it created a numerator/denominator mismatch.

           The denominator was the maximum after 500ms smoothing, but the **numerator** (EMG_*_MVC in
           Features_insert_FIXED.py:366) uses the **unsmoothed** instantaneous envelope value.
           The maximum of a moving average is necessarily <= the instantaneous peak, so %MVC structurally exceeds 1.0.

           Measured (10 subjects): main muscle inflated by 1.56x on average, synergist by 1.95x.
           For the four subjects not truncated by clip(0,1.5), the inflation factor matches the measured %MVC maximum to three decimals
           (S05 1.396/1.396, S06 1.305/1.301, S10 1.485/1.483) — mechanism confirmed.

           The spike problem that [Normalization-5] worried about is **already blocked by the 5Hz low-pass** —
           the last step of apply_full_emg_pipeline is a 5Hz low-pass envelope,
           with a time scale of about 200ms; adding 500ms smoothing on top is redundant double smoothing.

           → Now uses mvc_denominator(), on the same scale as the numerator (neither is smoothed).
           windowed_peak() is kept for diagnostic comparison and is **no longer used for normalization**.

[Normalization-8] 🔴 **2026-08-05: the denominator is now the 99.9th percentile of all segments pooled, not the max.**

           Taking the max lets artifacts set the scale. On S02 the denominator came from a single
           75ms spike (18.40) in a bodyweight segment, while the peak of their actual heaviest 40kg set was only 7.65;
           the envelope dropping to near zero on both sides of the spike = filter ringing on a step discontinuity, not muscle activity.

           Pooling and taking p99.9 preserves both consistency and robustness; see the mvc_denominator() docstring for the reasoning.

           ⚠️ **This project has no real MVC recordings** — the segment counts in Dataset/Raw map one-to-one to
           Dataset/Combined (S01 7/7, S02 11/11 …), and all of them are squat segments.
           So EMG_*_MVC is **not %MVC**, but "normalized to the maximum activation observed for that
           subject". The paper must not say %MVC, nor compare directly with %MVC values in the literature.
           (The column name EMG_*_MVC is kept only for compatibility with existing code; the meaning defined here takes precedence.)

[Normalization-6] The file name carries a reminder: MVC and movement data must come from the same subject and the same electrode placement.
           If the electrodes are re-applied midway, gain/position change and %MVC normalization becomes invalid.

Enter the output MVC values into MVC_MAIN/COMP of Features_insert(_FIXED).py.
"""

import os
import sys

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

# --- Paths: always managed centrally by paths.py at the repo root; do not revert to absolute paths ---
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import RAW_DIR  # noqa: E402

# raw file to compute MVC from. Can be overridden on the command line: python MAX_MVC_FIXED.py <file path>
MVC_FILE_PATH = (sys.argv[1] if len(sys.argv) > 1
                 else os.path.join(RAW_DIR, "S01_Raw_DATA.csv"))
FS = 1000.0
MVC_WINDOW_MS = 500          # for diagnostic comparison only, **no longer used for normalization** (see [Normalization-7])
MVC_PERCENTILE = 99.9        # ★ percentile used for the normalization denominator (see [Normalization-8])


def apply_full_emg_pipeline(data, fs):
    """remove DC -> 60Hz notch -> 20Hz high-pass -> full-wave rectification -> 5Hz low-pass envelope"""
    x = data - np.mean(data)
    b_n, a_n = iirnotch(60.0, 30.0, fs)
    x = filtfilt(b_n, a_n, x)
    nyq = 0.5 * fs
    x = sosfiltfilt(butter(4, 20.0 / nyq, btype='high', output='sos'), x)
    rect = np.abs(x)
    env = sosfiltfilt(butter(4, 5.0 / nyq, btype='low', output='sos'), rect)
    return x, env


def windowed_peak(envelope, fs, window_ms):
    """
    Return 'the maximum after a moving-window average'.

    ⚠️ **Do not use it as the %MVC denominator** — the numerator is the unsmoothed envelope, the two scales differ,
       and %MVC gets structurally inflated by about 1.5~2× (see [Normalization-7] in the file header).
       This function is kept only for diagnostics / comparison with the old version. Use mvc_denominator() for normalization.
    """
    win = max(1, int(round(fs * window_ms / 1000.0)))
    if len(envelope) < win:
        return float(np.max(envelope))         # fall back to a single point when the data are too short
    kernel = np.ones(win) / win
    smoothed = np.convolve(envelope, kernel, mode='valid')
    return float(np.max(smoothed))


def mvc_denominator(envelopes, pct=MVC_PERCENTILE):
    """
    ★ Normalization denominator: pool the envelopes of **all segments** of the subject into one and take the pct-th percentile.

    Two conditions must hold at the same time:

    ① **Same scale as the numerator** — the numerator is EMG_*_MVC written by Features_insert_FIXED.py,
       i.e. the per-frame envelope value (not smoothed again). So the denominator must not be smoothed again either
       (that was the bug fixed by [Normalization-7]).

    ② **Robust to artifacts** — [Normalization-8]. Simply taking the max lets an electrode bounce set the scale:
       on S02 the denominator came from a 75ms spike in a bodyweight segment (peak 18.40,
       while their actual heaviest 40kg set was only 7.65); the envelope drops to near zero on both sides of the spike,
       which is filter ringing on a step discontinuity, not muscle activity.

    Pooling and taking a high percentile solves both:
      · a 75ms artifact is only 0.02% of the subject's total recording time (about 300~400 s),
        below (100−99.9)%, so it is naturally diluted away.
      · a true maximal effort is a plateau lasting 200~600ms, more than 0.1%, so it is not diluted.

    Diagnostic ratio max/this function: measured S02 = 1.97, S01 = 1.40, the other seven 1.07~1.25
    — an especially large ratio means that person's max is propped up by a handful of points.

    ⚠️ By definition about (100−pct)% of frames are >1.0; this is normal, not a bug.

    Parameters
    ----
    envelopes : a single envelope array, or a list/tuple of envelope arrays from several segments
    """
    if isinstance(envelopes, (list, tuple)):
        pool = np.concatenate([np.asarray(e).ravel() for e in envelopes])
    else:
        pool = np.asarray(envelopes).ravel()
    return float(np.percentile(pool, pct))


def calculate_max_mvc_per_segment():
    print(f"📂 Reading MVC raw file: {MVC_FILE_PATH}")
    try:
        df = pd.read_csv(MVC_FILE_PATH, encoding='big5', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(MVC_FILE_PATH, encoding='cp950', low_memory=False)

    start_indices = df[df['Time_ms'] == 1].index.tolist()
    if not start_indices:
        start_indices = [0]
    start_indices.append(len(df))

    envs_main, envs_comp = [], []
    best_seg_main, best_seg_comp = 0, 0
    peak_main, peak_comp = 0.0, 0.0

    print("\n" + "=" * 52)
    print(" 📊 per-set peaks (diagnostic only)")
    print(f"    normalization denominator = {MVC_PERCENTILE}th percentile of all segments pooled")
    print("=" * 52)

    for i in range(len(start_indices) - 1):
        seg = df.iloc[start_indices[i]:start_indices[i + 1]].copy()
        if len(seg) < FS:
            continue
        raw0 = pd.to_numeric(seg['Raw0'], errors='coerce').fillna(0).values
        raw1 = pd.to_numeric(seg['Raw1'], errors='coerce').fillna(0).values
        _, env0 = apply_full_emg_pipeline(raw0, FS)
        _, env1 = apply_full_emg_pipeline(raw1, FS)

        # ★ the denominator is now a percentile of all segments pooled, so here we only collect; no per-segment comparison
        envs_main.append(env0)
        envs_comp.append(env1)

        # per-segment peaks are diagnostic only (which segment is highest, and whether it is an outlier)
        seg_main, seg_comp = float(np.max(env0)), float(np.max(env1))
        if seg_main > peak_main:
            peak_main, best_seg_main = seg_main, i + 1
        if seg_comp > peak_comp:
            peak_comp, best_seg_comp = seg_comp, i + 1

        print(f"🎬 set {i+1:02d} | main peak={seg_main:6.1f} "
              f"| synergist peak={seg_comp:6.1f}")

    global_max_main = mvc_denominator(envs_main)
    global_max_comp = mvc_denominator(envs_comp)

    print("\n" + "🔥" * 22)
    print(f"▶ main (Raw0) denominator = {global_max_main:.2f}   "
          f"highest segment peak {peak_main:.2f} (set {best_seg_main})   "
          f"ratio {peak_main / global_max_main:.2f}")
    print(f"▶ synergist (Raw1) denominator = {global_max_comp:.2f}   "
          f"highest segment peak {peak_comp:.2f} (set {best_seg_comp})   "
          f"ratio {peak_comp / global_max_comp:.2f}")
    print("-" * 44)
    print("💡 The denominator is computed automatically by Features_insert_FIXED.compute_subject_mvc(); "
          "no manual entry needed.")
    print("⚠️ A ratio >1.6 means that person's highest segment peak may be an artifact (measured S02=1.97, an electrode bounce);")
    print("   the pooled percentile has diluted it, but it is still worth checking that segment's waveform.")
    print("=" * 44)


if __name__ == '__main__':
    calculate_max_mvc_per_segment()
