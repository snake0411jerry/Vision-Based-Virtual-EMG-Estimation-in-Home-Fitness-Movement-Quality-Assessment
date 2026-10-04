import pandas as pd
import numpy as np
import scipy.signal as signal
import os
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

# --- settings ---
TARGET_FPS = 60.0  
EMG_FS = 1000.0

# 1. raw data path (input)
INPUT_FILE = r"<DATASET_ROOT>\Raw\S01_Raw_DATA.csv"

# 2. output path - only the combined-file output directory is kept
COMBINED_OUTPUT_DIR = r"<CLEAN_DIR>"


def apply_dual_emg_pipeline(data, fs):
    """Dual-track EMG processing: outputs data for FFT (high frequencies kept) and for AI (low-pass envelope) at the same time"""
    # 1. remove the DC offset
    signal_centered = data - np.mean(data)
    
    # 2. 60Hz notch filter
    b_notch, a_notch = iirnotch(60.0, 30.0, fs)
    signal_notched = filtfilt(b_notch, a_notch, signal_centered)
    
    # 3. 20Hz high-pass filter
    nyq = 0.5 * fs
    sos_high = butter(4, 20.0 / nyq, btype='high', output='sos')
    signal_high = sosfiltfilt(sos_high, signal_notched)
    
    # 🌟 branch point 1: keep oscillations above 20Hz; this is for the FFT fatigue analysis
    signal_for_fft = signal_high 
    
    # 4. full-wave rectification
    signal_rectified = np.abs(signal_high)
    
    # 5. 5Hz low-pass filter
    sos_low = butter(4, 5.0 / nyq, btype='low', output='sos')
    signal_envelope = sosfiltfilt(sos_low, signal_rectified)
    
    # 🌟 branch point 2: the smooth muscle-output trend; this is for the AI model
    return signal_for_fft, signal_envelope


def process_and_sync_emg(df, fps, fs):
    print("⏳ Running dual-track EMG processing (60Hz for AI training + 1000Hz for fatigue analysis)...")
    data_cols = ['Raw0', 'Raw1', 'Raw2']
    
    # filter and build the envelope at 1000Hz first
    for col in data_cols:
        if col in df.columns:
            raw_data = pd.to_numeric(df[col], errors='coerce').fillna(0).values
            fft_data, env_data = apply_dual_emg_pipeline(raw_data, fs)
            df[f'{col}_FFT'] = fft_data
            df[f'{col}_Env'] = env_data

    # find the sync-signal split points (this logic is kept to ensure each set's time axis is aligned correctly)
    start_indices = df[df['Time_ms'] == 1].index.tolist()
    start_indices.append(len(df)) 
    
    all_segments_env = []
    all_segments_fft = []
    
    for i in range(len(start_indices) - 1):
        start_idx = start_indices[i]
        end_idx = start_indices[i+1]
        
        segment_df = df.iloc[start_idx:end_idx].copy()
        max_ms = segment_df['Time_ms'].max()
        
        if max_ms <= 0 or pd.isna(max_ms): continue
            
        num_frames = int(round((max_ms / 1000.0) * fps))
        if num_frames == 0: num_frames = 1
            
        print(f"🎬 set {i+1}: total duration {max_ms} ms -> AI downsampled to {num_frames} frames / FFT keeps {len(segment_df)} samples")

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
            'Time_ms': perfect_valid_x # use the repaired smooth time axis
        })
        for col in data_cols:
            fft_col = f'{col}_FFT'
            if fft_col in segment_df.columns:
                fft_df[col] = segment_df[fft_col].values
                
        if 'Marker' in segment_df.columns:
            fft_df['Marker'] = pd.to_numeric(segment_df['Marker'], errors='coerce').fillna(0).values
            
        all_segments_fft.append(fft_df)

    return all_segments_env, all_segments_fft


if __name__ == '__main__':
    try:
        df = pd.read_csv(INPUT_FILE, encoding='big5', low_memory=False)
    except:
        df = pd.read_csv(INPUT_FILE, encoding='cp950', low_memory=False)

    # get the two different data lists
    segmented_dfs_env, segmented_dfs_fft = process_and_sync_emg(df, TARGET_FPS, EMG_FS)

    if segmented_dfs_env and segmented_dfs_fft:
        os.makedirs(COMBINED_OUTPUT_DIR, exist_ok=True)

        # 📦 task: write the combined files (Env_60FPS and FFT_1000Hz added to the file names to tell them apart)
        print("\n📦 Generating the combined files...")
        
        # write the 60Hz AI training combined file
        combined_env_df = pd.concat(segmented_dfs_env, ignore_index=True)
        env_out_path = os.path.join(COMBINED_OUTPUT_DIR, 'S06_CLEAN_SYNCED_Env_60FPS.csv')
        combined_env_df.to_csv(env_out_path, index=False, encoding='utf-8-sig')
        
        # write the 1000Hz FFT fatigue-analysis combined file
        combined_fft_df = pd.concat(segmented_dfs_fft, ignore_index=True)
        fft_out_path = os.path.join(COMBINED_OUTPUT_DIR, 'S06_CLEAN_SYNCED_FFT_1000Hz.csv')
        combined_fft_df.to_csv(fft_out_path, index=False, encoding='utf-8-sig')
        
        print(f"🎉 AI combined file saved to: {env_out_path}")
        print(f"🎉 FFT combined file saved to: {fft_out_path}")
            
        print("\n🚀 Both combined files processed and written!")