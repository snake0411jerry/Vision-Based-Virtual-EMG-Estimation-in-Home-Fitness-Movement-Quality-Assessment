import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import glob

# ==========================================
# 0. redefine the custom loss function (needed to load the model)
# ==========================================
def peak_weighted_mse(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    squared_error = tf.square(y_true - y_pred)
    peak_weight = tf.exp(4.0 * y_true)  # make sure the prediction script uses the same loss function as training
    return tf.reduce_mean(squared_error * peak_weight)

# ==========================================
# 1. load the model and the scaler
# ==========================================
print("🧠 Loading the model and normalizer...")
model = tf.keras.models.load_model(
    r"<DATASET_ROOT>\Code\teacher_tcn_transformer_model.keras", 
    custom_objects={'peak_weighted_mse': peak_weighted_mse}
)
scaler_x = joblib.load(r"<DATASET_ROOT>\Code\scaler_x.pkl")
print("✅ Loaded! Ready for batch prediction.\n")

# ==========================================
# 2. set the per-subject MVC dictionary
# ==========================================
mvc_dict = {
    'S01':     {'MVC_Main': 920, 'MVC_Compass': 750},
    'PILOT_A': {'MVC_Main': 975, 'MVC_Compass': 774}, 
    'PILOT_B':   {'MVC_Main': 827, 'MVC_Compass': 608},
    'TEST1':    {'MVC_Main': 1018, 'MVC_Compass': 1019}  
}

search_pattern = r"<DATASET_ROOT>\Code\0519Combined\Origin" + r"/*_Segment_*_Combined_Features.csv"
file_list = glob.glob(search_pattern, recursive=True)

if not file_list:
    print("❌ No matching CSV files found; check the working directory or file-name format.")
else:
    print(f"🔍 Found {len(file_list)} files; starting...\n")

# ==========================================
# 3. process each file in batch
# ==========================================
# 🚨 fix: keep the prediction window consistent with the newly trained model (40)
window_size = 40

for file_path in file_list:
    print("-" * 50)
    print(f"📂 Processing: {file_path}")
    
    # read the CSV
    df = pd.read_csv(file_path)
    filename = os.path.basename(file_path).upper()
    
    # automatically identify the subject and assign the MVC
    matched_subject = "DEFAULT"
    for subject in mvc_dict.keys():
        if subject in filename and subject != "DEFAULT":
            matched_subject = subject
            break
            
    subj_mvc = mvc_dict[matched_subject]
    df['Subject_MVC_Main'] = subj_mvc["MVC_Main"]
    df['Subject_MVC_Compass'] = subj_mvc["MVC_Compass"]
    print(f"👤 Subject identified as: {matched_subject} (Main: {subj_mvc['MVC_Main']}, Compass: {subj_mvc['MVC_Compass']})")

    # ==========================================
    # force the 22 feature names and their order
    # ==========================================
    feature_cols = [
        'Shoulder_Y_norm', 'Shoulder_Y_vel', 'Shoulder_Y_acc',
        'Shoulder_Z_norm', 'Shoulder_Z_vel', 'Shoulder_Z_acc',
        'Knee_Y_norm', 'Knee_Y_vel', 'Knee_Y_acc',
        'Knee_Z_norm', 'Knee_Z_vel', 'Knee_Z_acc',
        'Ankle_Y_norm', 'Ankle_Y_vel', 'Ankle_Y_acc',
        'Ankle_Z_norm', 'Ankle_Z_vel', 'Ankle_Z_acc',
        'Knee_Angle_norm', 'Knee_Angle_vel',
        'Subject_MVC_Main', 'Subject_MVC_Compass' 
    ]
    
    # check that every feature exists in the table
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️ Fatal error: the new data lacks these required features: {missing_cols}")
        continue

    print(f"👉 Got {len(feature_cols)} features; about to normalize...")
    
    # normalize
    X_new_for_scaling = df[feature_cols].values
    X_new_scaled = scaler_x.transform(X_new_for_scaling)
    print(f"📊 scaled features - mean: {np.mean(X_new_scaled):.2f}, max: {np.max(X_new_scaled):.2f}, min: {np.min(X_new_scaled):.2f}")

    # ==========================================
    # sliding-window segmentation
    # ==========================================
    X_windows = []
    for i in range(len(X_new_scaled) - window_size + 1):
        X_windows.append(X_new_scaled[i : i + window_size])
        
    if not X_windows:
        print(f"⚠️ The file has fewer rows than the window length {window_size}; skipping this file.")
        continue
        
    X_tensor_new = np.array(X_windows)
    
    # 1. raw model prediction
    predictions = model.predict(X_tensor_new, verbose=0)
    pred_main = predictions[:, 0]
    pred_compass = predictions[:, 1]
    
    # ==========================================
    # 🌟 smoothing added here (prediction stage only)
    # ==========================================
    # window=5 averages the 5 surrounding frames; change to 3 or 7 as needed
    pred_main = pd.Series(pred_main).rolling(window=5, min_periods=1).mean().values
    pred_compass = pd.Series(pred_compass).rolling(window=5, min_periods=1).mean().values
    
    # ==========================================
    # 4. pad the window offset (Padding) - 🚨 centre-aligned fix 🚨
    # ==========================================
    # because during training the model predicts the "exact middle (frame 20)" of the window,
    # the 39 missing frames must be split: 20 NaNs in front and 19 NaNs at the end
    front_pad_length = window_size // 2
    back_pad_length = window_size - 1 - front_pad_length
    
    front_pad = np.full(front_pad_length, np.nan)
    back_pad = np.full(back_pad_length, np.nan)
    
    # concatenating this way keeps the prediction curve 100% aligned with the real motion on the time axis
    pred_main_padded = np.concatenate([front_pad, pred_main, back_pad])
    pred_compass_padded = np.concatenate([front_pad, pred_compass, back_pad])
    
    df['Predicted_EMG_Main'] = pred_main_padded
    df['Predicted_EMG_Compass'] = pred_compass_padded
    
    # save as a new file
    output_path = file_path.replace(".csv", "_Predicted.csv")
    df.to_csv(output_path, index=False)
    
    print(f"✅ Prediction done and merged! New file saved to: {output_path}")

print("-" * 50)
print("🎉 All matching files have been processed!")