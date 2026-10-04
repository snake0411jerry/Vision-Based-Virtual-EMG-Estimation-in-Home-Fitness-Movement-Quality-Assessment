import pandas as pd
import numpy as np
import scipy.signal as signal
import os

# ==========================================
# parameter area (adjust to your actual paths)
# ==========================================
# the already converted EMG file (contains Segments 1~5, already resampled per frame)
emg_file_path = r"<DATASET_ROOT>\0513\emg\S01_RAW_DATA_auto_calculated.csv"

# list of TRC files (corresponding to Segments 1~5)
trc_files = [
    r"<DATASET_ROOT>\0513\opoencap\S01\21-50\OpenCapData_6dd32c04-9aaa-446f-b58b-f56163bf7867\MarkerData\general1-10.trc",
    r"<DATASET_ROOT>\0513\opoencap\S01\21-50\OpenCapData_6dd32c04-9aaa-446f-b58b-f56163bf7867\MarkerData\general11-20.trc", 
    r"<DATASET_ROOT>\0513\opoencap\S01\21-50\OpenCapData_6dd32c04-9aaa-446f-b58b-f56163bf7867\MarkerData\general21-30.trc",
    r"<DATASET_ROOT>\0513\opoencap\S01\21-50\OpenCapData_6dd32c04-9aaa-446f-b58b-f56163bf7867\MarkerData\general31-40.trc" ,
    r"<DATASET_ROOT>\0513\opoencap\S01\21-50\OpenCapData_6dd32c04-9aaa-446f-b58b-f56163bf7867\MarkerData\general41-50.trc"
]

# output path
output_dir = r"<DATASET_ROOT>\0602\Combined"

# ==========================================
# core algorithm parameters
# ==========================================
MVC_MAIN_RAW = 1019
MVC_COMP_RAW = 1018
BASELINE_FIXED = 462.0  # 🔥 fixed reference value

FPS = 60.0              # video frame rate
dt = 1.0 / FPS

# 🔥 key change: because your EMG file has already been converted to "one row per frame",
# the data's "sampling rate" now equals the video's "frame rate (FPS)"!
EMG_FS = FPS           

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==========================================
# helper: compute the knee angle
# ==========================================
def calculate_angle(v1, v2):
    dot_product = np.sum(v1 * v2, axis=1)
    norm_v1 = np.linalg.norm(v1, axis=1)
    norm_v2 = np.linalg.norm(v2, axis=1)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

# ==========================================
# step 1: read the EMG file and get each set's target frame count
# ==========================================
try:
    df_emg_all = pd.read_csv(emg_file_path, encoding='utf-8-sig')
except UnicodeDecodeError:
    df_emg_all = pd.read_csv(emg_file_path, encoding='big5')

segment_target_frames = df_emg_all.groupby('Segment')['Frame'].max().to_dict()
print("🎯 Target frame count per Segment:", segment_target_frames)

# ==========================================
# step 2: process each Segment
# ==========================================
for segment_idx, trc_path in enumerate(trc_files, start=1):
    print(f"\n⏳ Processing Segment {segment_idx} ...")
    
    if segment_idx not in segment_target_frames:
        print(f"⚠️ No EMG data found for Segment {segment_idx}; skipping.")
        continue
        
    target_frames = int(segment_target_frames[segment_idx])
    
    # ----------------------------------------------------
    # 2.1 read the TRC skeleton data and align it by interpolation
    # ----------------------------------------------------
    try:
        df_gen = pd.read_csv(trc_path, sep='\t', skiprows=4)
    except FileNotFoundError:
        print(f"⚠️ TRC file not found: {trc_path}; skipping this set.")
        continue

    df_gen = df_gen.rename(columns={'Unnamed: 0': 'Frame', 'Unnamed: 1': 'Time'})
    df_gen = df_gen.dropna(subset=['Frame'])
    df_gen['Frame'] = df_gen['Frame'].astype(int)
    
    start_frame = df_gen['Frame'].min()
    df_gen['Frame'] = df_gen['Frame'] - start_frame + 1
    
    original_frames = df_gen['Frame'].values
    target_frame_grid = np.linspace(1, target_frames, target_frames)
    resampled_data = {'Frame': np.arange(1, target_frames + 1)}
    
    for col in df_gen.columns:
        if col not in ['Frame', 'Time']:
            valid_y = pd.to_numeric(df_gen[col], errors='coerce').fillna(0).values
            resampled_data[col] = np.interp(target_frame_grid, original_frames, valid_y)
            
    df_resampled_skel = pd.DataFrame(resampled_data)
    
    # ----------------------------------------------------
    # 2.2 feature engineering: extract the needed skeleton points and compute angles/velocity/acceleration
    # ----------------------------------------------------
    columns_to_keep = {
        'Y5': 'Shoulder_Y', 'Z5': 'Shoulder_Z',         # drop Shoulder_X to reduce useless noise
        'X12': 'Hip_X', 'Y12': 'Hip_Y', 'Z12': 'Hip_Z', # Hip_X must be kept to compute the 3D thigh vector
        'X13': 'Knee_X', 'Y13': 'Knee_Y', 'Z13': 'Knee_Z',       # corresponds to LKnee in the screenshot
        'X14': 'Ankle_X', 'Y14': 'Ankle_Y', 'Z14': 'Ankle_Z',    # corresponds to LAnkle in the screenshot
        'X15': 'BigToe_X', 'Y15': 'BigToe_Y', 'Z15': 'BigToe_Z', # corresponds to LBigToe in the screenshot
        'X16': 'SmallToe_X', 'Y16': 'SmallToe_Y', 'Z16': 'SmallToe_Z' # corresponds to LSmallToe in the screenshot
    }
    df_clean = df_resampled_skel[list(columns_to_keep.keys())].rename(columns=columns_to_keep)

    # 🔥 improvement: use the midpoint of BigToe and SmallToe as the forefoot, giving a more stable foot vector
    df_clean['Toe_X'] = (df_clean['BigToe_X'] + df_clean['SmallToe_X']) / 2.0
    df_clean['Toe_Y'] = (df_clean['BigToe_Y'] + df_clean['SmallToe_Y']) / 2.0
    df_clean['Toe_Z'] = (df_clean['BigToe_Z'] + df_clean['SmallToe_Z']) / 2.0
    
    # original thigh–shank angle (computed from the Y, Z projections)
    v_thigh = np.array([df_clean['Hip_Z'] - df_clean['Knee_Z'], df_clean['Hip_Y'] - df_clean['Knee_Y']]).T
    v_shank = np.array([df_clean['Ankle_Z'] - df_clean['Knee_Z'], df_clean['Ankle_Y'] - df_clean['Knee_Y']]).T
    df_clean['Knee_Angle'] = calculate_angle(v_thigh, v_shank)

    # compute the 3D direction difference between knee and toes (detects knee-valgus compensation)
    # vector 1: 3D thigh direction (Hip towards Knee)
    v_knee_dir = np.array([df_clean['Knee_X'] - df_clean['Hip_X'],
                           df_clean['Knee_Y'] - df_clean['Hip_Y'],
                           df_clean['Knee_Z'] - df_clean['Hip_Z']]).T
    
    # vector 2: 3D foot direction (Ankle towards the stable Toe midpoint)
    v_toe_dir = np.array([df_clean['Toe_X'] - df_clean['Ankle_X'],
                          df_clean['Toe_Y'] - df_clean['Ankle_Y'],
                          df_clean['Toe_Z'] - df_clean['Ankle_Z']]).T
    
    # compute the angle between the two vectors 
    df_clean['Knee_Toe_Angle_Diff'] = calculate_angle(v_knee_dir, v_toe_dir)
    
    ref_height = (df_clean['Shoulder_Y'] - df_clean['Ankle_Y']).max()
    
    # for the calculus loop below, the toes can be merged into a single virtual joint 'Toe'
    for joint in ['Shoulder', 'Knee', 'Ankle', 'Toe']:
        # Shoulder has no X axis, so add a check to avoid a KeyError
        axes = ['Y', 'Z'] if joint == 'Shoulder' else ['X', 'Y', 'Z']
        for axis in axes: 
            col_name = f"{joint}_{axis}"
            root_col = f"Hip_{axis}" if axis in ['X', 'Y', 'Z'] else f"Hip_Y"
            
            # only subtract when Hip_X exists (Shoulder has no X and never gets here)
            centered_col = f"{col_name}_centered"
            df_clean[centered_col] = df_clean[col_name] - df_clean[root_col]
            
            norm_col = f"{col_name}_norm"
            df_clean[norm_col] = df_clean[centered_col] / ref_height
            
            vel_col = f"{col_name}_vel"
            df_clean[vel_col] = df_clean[norm_col].diff() / dt
            df_clean[vel_col] = df_clean[vel_col].fillna(0)
            
            acc_col = f"{col_name}_acc"
            df_clean[acc_col] = df_clean[vel_col].diff() / dt
            df_clean[acc_col] = df_clean[acc_col].fillna(0)

    # angle feature normalization and velocity
    df_clean['Knee_Angle_norm'] = df_clean['Knee_Angle'] / 180.0
    df_clean['Knee_Angle_vel'] = df_clean['Knee_Angle_norm'].diff() / dt
    df_clean['Knee_Angle_vel'] = df_clean['Knee_Angle_vel'].fillna(0)

    # normalization and rate of change of the angle difference
    df_clean['Knee_Toe_Diff_norm'] = df_clean['Knee_Toe_Angle_Diff'] / 180.0
    df_clean['Knee_Toe_Diff_vel'] = df_clean['Knee_Toe_Diff_norm'].diff() / dt
    df_clean['Knee_Toe_Diff_vel'] = df_clean['Knee_Toe_Diff_vel'].fillna(0)

    # select the final features sent to the model
    final_columns = [col for col in df_clean.columns if col.endswith(('_norm', '_vel', '_acc'))]
    df_final_features = df_clean[final_columns].copy()
    
    # ----------------------------------------------------
    # 2.3 extract the matching EMG data and build the smoothed MVC envelope
    # ----------------------------------------------------
    df_emg_segment = df_emg_all[df_emg_all['Segment'] == segment_idx].copy().reset_index(drop=True)
    
    # compute amplitude (fixed reference value)
    mvc_main_amp = MVC_MAIN_RAW - BASELINE_FIXED
    mvc_comp_amp = MVC_COMP_RAW - BASELINE_FIXED
    
    # 🔥 key change: 'Main_raw' becomes 'Raw0' and 'Compasste1_raw' becomes 'Raw1'
    # (adjust Raw0 or Raw1 to match your actual electrode channels)
    emg_main_rectified = np.abs(df_emg_segment['Raw0'].values - BASELINE_FIXED) / mvc_main_amp
    emg_comp_rectified = np.abs(df_emg_segment['Raw1'].values - BASELINE_FIXED) / mvc_comp_amp
    
    # low-pass filter (cutoff 5Hz, sampling rate set to the FPS)
    b, a = signal.butter(4, 5.0 / (EMG_FS / 2.0), btype='low')
    emg_main_mvc_env = signal.filtfilt(b, a, emg_main_rectified)
    emg_comp_mvc_env = signal.filtfilt(b, a, emg_comp_rectified)
    
    # merge features
    df_final_features['EMG_Main_MVC'] = emg_main_mvc_env
    df_final_features['EMG_Compass_MVC'] = emg_comp_mvc_env
    
    # ----------------------------------------------------
    # 2.4 write out this Segment's result
    # ----------------------------------------------------
    output_filename = os.path.join(output_dir, f"S01_0513_Segment_{segment_idx}_Combined_Features.csv")
    df_final_features.to_csv(output_filename, index=False)
    print(f"✅ Segment {segment_idx} done, output length: {len(df_final_features)} frames")
    print(f"📁 Saved to: {output_filename}")

print("\n🎉 All Segments combined!")