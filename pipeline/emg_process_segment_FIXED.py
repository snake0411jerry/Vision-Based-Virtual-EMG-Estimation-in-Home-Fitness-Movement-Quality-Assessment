"""
emg_process_segment_FIXED.py — corrected dual-track EMG preprocessing (60FPS AI / 1000Hz FFT)
=========================================================================
Fixes relative to the original `emg_process_segment.py`:

[Generalize-1] ★ The original hard-coded the input file and output file names to a single subject (S01), and the output file name
           even reused another subject's code, not matching the subject actually processed. This version scans the whole
           Raw folder, infers the subject from the file name, processes each one and writes separate outputs;
           file names are always taken from the input file itself, so outputs can no longer be attributed to the wrong person.

[Robustness-1] ★ Encoding detection when reading files used a bare except, which also swallowed unrelated errors;
           it now explicitly catches only UnicodeDecodeError.

[Performance-1] ★ The notch/high-pass/low-pass filter coefficients depend only on the sampling rate fs, yet the original
           redesigned the filters for every channel inside apply_dual_emg_pipeline()
           (3 butter/iirnotch calls). This version moves filter design
           out of the loop, designing once per file and reusing it for Raw0/Raw1/Raw2.

The rest of the signal-processing algorithm (remove DC → 60Hz notch → 20Hz high-pass → full-wave rectification →
5Hz low-pass envelope) and the segment resampling logic are identical to the original version.
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


import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# --- settings ---
TARGET_FPS = 60.0
EMG_FS = 1000.0

# 1. raw data path (input) — a folder; all *_Raw_DATA.csv in it are scanned
RAW_INPUT_DIR = RAW_DIR

# 2. output paths (output) — a separate output directory for each track
# CLEAN_DIR comes from paths.py
ENV_OUTPUT_DIR = os.path.join(CLEAN_DIR, "60FPS")    # 60FPS envelope for AI training
FFT_OUTPUT_DIR = os.path.join(CLEAN_DIR, "1000HZ")   # 1000Hz high-passed for fatigue analysis

SUBJECT_NAME_RE = re.compile(r'^(.+?)(?:_\d+)?_Raw_DATA\.csv$', re.IGNORECASE)


def design_emg_filters(fs):
    """Design the notch / high-pass / low-pass filter coefficients (only needs to be called once per file)."""
    nyq = 0.5 * fs
    b_notch, a_notch = iirnotch(60.0, 30.0, fs)
    sos_high = butter(4, 20.0 / nyq, btype='high', output='sos')
    sos_low = butter(4, 5.0 / nyq, btype='low', output='sos')
    return b_notch, a_notch, sos_high, sos_low


def apply_dual_emg_pipeline(data, filters):
    """Dual-track EMG processing: outputs data for FFT (high frequencies kept) and for AI (low-pass envelope) at the same time"""
    b_notch, a_notch, sos_high, sos_low = filters

    # 1. remove the DC offset
    signal_centered = data - np.mean(data)

    # 2. 60Hz notch filter
    signal_notched = filtfilt(b_notch, a_notch, signal_centered)

    # 3. 20Hz high-pass filter
    signal_high = sosfiltfilt(sos_high, signal_notched)

    # branch point 1: keep high-frequency oscillation above 20Hz; this is for the FFT fatigue analysis
    signal_for_fft = signal_high

    # 4. full-wave rectification
    signal_rectified = np.abs(signal_high)

    # 5. 5Hz low-pass filter
    signal_envelope = sosfiltfilt(sos_low, signal_rectified)

    # branch point 2: the smooth muscle-output trend; this is for the AI model
    return signal_for_fft, signal_envelope


def process_and_sync_emg(df, fps, fs):
    print("⏳ Running dual-track EMG processing (60Hz for AI training + 1000Hz for fatigue analysis)...")
    data_cols = ['Raw0', 'Raw1', 'Raw2']
    filters = design_emg_filters(fs)

    # filter and build the envelope at 1000Hz first
    for col in data_cols:
        if col in df.columns:
            raw_data = pd.to_numeric(df[col], errors='coerce').fillna(0).values
            fft_data, env_data = apply_dual_emg_pipeline(raw_data, filters)
            df[f'{col}_FFT'] = fft_data
            df[f'{col}_Env'] = env_data

    # find the sync-signal split points (this logic is kept to ensure each set's time axis is aligned correctly)
    start_indices = df[df['Time_ms'] == 1].index.tolist()
    start_indices.append(len(df))

    all_segments_env = []
    all_segments_fft = []

    for i in range(len(start_indices) - 1):
        start_idx = start_indices[i]
        end_idx = start_indices[i + 1]

        segment_df = df.iloc[start_idx:end_idx].copy()
        max_ms = segment_df['Time_ms'].max()

        if max_ms <= 0 or pd.isna(max_ms):
            continue

        num_frames = int(round((max_ms / 1000.0) * fps))
        if num_frames == 0:
            num_frames = 1

        print(f"🎬 set {i + 1}: total duration {max_ms} ms -> AI downsampled to {num_frames} frames / FFT keeps {len(segment_df)} samples")

        # build the ideal theoretical time axis
        num_raw_samples = len(segment_df)
        perfect_valid_x = np.linspace(0, max_ms, num_raw_samples)

        # ==========================================
        # build track 1: the 60FPS downsampled envelope data for AI
        # ==========================================
        new_times = np.linspace(0, max_ms, num_frames)
        resampled_env = pd.DataFrame({
            'Segment': i + 1,
            'Frame': np.arange(1, num_frames + 1),
            'Time_ms': new_times
        })

        for col in data_cols:
            env_col = f'{col}_Env'
            if env_col in segment_df.columns:
                valid_y = segment_df[env_col].values
                resampled_env[col] = np.clip(np.interp(new_times, perfect_valid_x, valid_y), 0, None)

        if 'Marker' in segment_df.columns:
            valid_y_marker = pd.to_numeric(segment_df['Marker'], errors='coerce').fillna(0).values
            resampled_env['Marker'] = np.round(np.interp(new_times, perfect_valid_x, valid_y_marker)).astype(int)

        all_segments_env.append(resampled_env)

        # ==========================================
        # build track 2: the clean 1000Hz high-passed data for FFT
        # ==========================================
        fft_df = pd.DataFrame({
            'Segment': i + 1,
            'Time_ms': perfect_valid_x  # use the repaired smooth time axis
        })
        for col in data_cols:
            fft_col = f'{col}_FFT'
            if fft_col in segment_df.columns:
                fft_df[col] = segment_df[fft_col].values

        if 'Marker' in segment_df.columns:
            fft_df['Marker'] = pd.to_numeric(segment_df['Marker'], errors='coerce').fillna(0).values

        all_segments_fft.append(fft_df)

    return all_segments_env, all_segments_fft


def extract_subject_name(input_file):
    """Infer the subject name from the file name, e.g. S01_0622_Raw_DATA.csv -> S01."""
    base = os.path.basename(input_file)
    m = SUBJECT_NAME_RE.match(base)
    return m.group(1) if m else os.path.splitext(base)[0]


def read_raw_csv(input_file):
    try:
        return pd.read_csv(input_file, encoding='big5', low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(input_file, encoding='cp950', low_memory=False)


def process_file(input_file, env_output_dir=ENV_OUTPUT_DIR, fft_output_dir=FFT_OUTPUT_DIR,
                 fps=TARGET_FPS, fs=EMG_FS):
    """Process a single subject's Raw CSV and write two combined files, 60FPS and 1000Hz."""
    subject = extract_subject_name(input_file)
    print(f"\n===== Subject: {subject} ({os.path.basename(input_file)}) =====")

    df = read_raw_csv(input_file)
    segmented_dfs_env, segmented_dfs_fft = process_and_sync_emg(df, fps, fs)

    if not (segmented_dfs_env and segmented_dfs_fft):
        print(f"⚠️ {subject}: no valid segments; skipping output.")
        return None

    os.makedirs(env_output_dir, exist_ok=True)
    os.makedirs(fft_output_dir, exist_ok=True)

    print("\n📦 Generating the combined files...")

    combined_env_df = pd.concat(segmented_dfs_env, ignore_index=True)
    env_out_path = os.path.join(env_output_dir, f'{subject}_CLEAN_SYNCED_Env_60FPS.csv')
    combined_env_df.to_csv(env_out_path, index=False, encoding='utf-8-sig')

    combined_fft_df = pd.concat(segmented_dfs_fft, ignore_index=True)
    fft_out_path = os.path.join(fft_output_dir, f'{subject}_CLEAN_SYNCED_FFT_1000Hz.csv')
    combined_fft_df.to_csv(fft_out_path, index=False, encoding='utf-8-sig')

    print(f"🎉 AI combined file saved to: {env_out_path}")
    print(f"🎉 FFT combined file saved to: {fft_out_path}")

    return env_out_path, fft_out_path


if __name__ == '__main__':
    raw_files = sorted(glob.glob(os.path.join(RAW_INPUT_DIR, '*_Raw_DATA.csv')))

    if not raw_files:
        print(f"⚠️ No *_Raw_DATA.csv files found in {RAW_INPUT_DIR}.")

    for raw_file in raw_files:
        process_file(raw_file)

    print("\n🚀 Dual-track combined files for all subjects processed and written successfully!")
