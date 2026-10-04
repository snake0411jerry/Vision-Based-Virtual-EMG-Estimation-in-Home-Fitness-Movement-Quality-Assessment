import pandas as pd
import numpy as np
import os

# --- settings ---
target_fps = 60  
file_path = r"<DATASET_ROOT>\0601\emg\TEST1_0601_RAW_DATA.csv"
output_dir = r"<DATASET_ROOT>\0601\emg"
# ----------------

def process_per_frame(df, fps):
    # 🚨 bug fix: your data's Time_ms starts at 0, not 1!
    # this is needed to catch the sync signal at the first row where Marker = 1
    start_indices = df[df['Time_ms'] == 1].index.tolist()
    start_indices.append(len(df)) # append the end index
    
    all_segments = []
    data_cols = ['Raw0', 'Raw1', 'Raw2']
    
    for i in range(len(start_indices) - 1):
        start_idx = start_indices[i]
        end_idx = start_indices[i+1]
        
        # extract this set's data
        segment_df = df.iloc[start_idx:end_idx].copy()
        
        # automatically compute this set's duration (ms)
        max_ms = segment_df['Time_ms'].max()
        
        if max_ms <= 0 or pd.isna(max_ms):
            print(f"Set {i+1} has an abnormal duration; skipping.")
            continue
            
        # automatically compute the total number of frames from duration and FPS
        num_frames = int(round((max_ms / 1000.0) * fps))
        if num_frames == 0:
            num_frames = 1
            
        print(f"Set {i+1}: detected total duration {max_ms} ms -> computed as {num_frames} frames (FPS: {fps})")

        # build a new time axis (from 0 to max_ms)
        new_times = np.linspace(0, max_ms, num_frames)
        
        resampled = pd.DataFrame({
            'Segment': i + 1,
            'Frame': np.arange(1, num_frames + 1),
            'Time_ms': new_times
        })
        
        # linearly interpolate the EMG signal
        for col in data_cols:
            if col in segment_df.columns:
                valid_x = segment_df['Time_ms'].values
                valid_y = pd.to_numeric(segment_df[col], errors='coerce').fillna(0).values
                resampled[col] = np.interp(new_times, valid_x, valid_y)
            
        # interpolate the Marker (making sure the sync signal is not lost)
        if 'Marker' in segment_df.columns:
            valid_x = segment_df['Time_ms'].values
            valid_y_marker = pd.to_numeric(segment_df['Marker'], errors='coerce').fillna(0).values
            # max-preserving approach: if a 1 occurs anywhere in the interval, keep 1, so interpolation never dilutes the sync signal
            resampled['Marker'] = np.round(np.interp(new_times, valid_x, valid_y_marker)).astype(int)
            
        all_segments.append(resampled)

    if not all_segments:
        return pd.DataFrame()
    return pd.concat(all_segments, ignore_index=True)

# read the file
try:
    df = pd.read_csv(file_path, encoding='big5', low_memory=False)
except:
    df = pd.read_csv(file_path, encoding='cp950', low_memory=False)

# run the conversion
result_df = process_per_frame(df, target_fps)

# save the result
if not result_df.empty:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    full_output_path = os.path.join(output_dir, 'TEST1_0601_RAW_DATA_auto_calculated.csv')
    result_df.to_csv(full_output_path, index=False, encoding='utf-8-sig')
    print(f"\n🎉 Conversion done! File saved to: {full_output_path}")
else:
    print("\n❌ Failed: could not extract valid data from the file.")