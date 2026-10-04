import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib

# ==========================================
# 🌟 these two lines fix Chinese text rendering in Matplotlib
# ==========================================
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # Windows uses Microsoft JhengHei
plt.rcParams['axes.unicode_minus'] = False                # make sure minus signs render correctly in the charts

# ==========================================
# 1. set the file path and load the model/normalizer
# ==========================================
# ⚠️ replace this with the path of the single CSV file you want to test
test_csv_path = r"<DATASET_ROOT>\0529\Combined\S01_0529_Segment_4_Combined_Features.csv"

# load the model and scaler just trained (experiment B version)
model_path = "teacher_multitask_random_noVA.keras"
scaler_x_path = "scaler_x_multitask_random_noVA.pkl"
scaler_y_path = "scaler_y_emg_multitask_random.pkl"

print("⏳ Loading the model and normalizer...")
model = tf.keras.models.load_model(model_path)
scaler_x = joblib.load(scaler_x_path)
scaler_y_emg = joblib.load(scaler_y_path)
print("✅ Loaded!")

# ==========================================
# 2. read the test data and preprocess
# ==========================================
print(f"📂 Reading test file: {os.path.basename(test_csv_path)}")
df = pd.read_csv(test_csv_path)

# the exclusion list must match training exactly
exclude_keywords = ['EMG_', 'Knee_Toe_Diff', 'vel', 'acc', '_V', '_A', 'Velocity', 'Acceleration', 'Comp_Prob_Target', 'Unnamed: 0']
feature_cols = [col for col in df.columns if not any(k in col for k in exclude_keywords)]

print(f"✅ Number of features in the test set: {len(feature_cols)}") 

# extract the ground-truth labels used as plot references
true_valgus_angle = df['Knee_Toe_Diff_norm'].values * 180.0 

# normalize the input features
X_data = scaler_x.transform(df[feature_cols].values)

# ==========================================
# 3. sliding-window segmentation (inference only: step = 1)
# ==========================================
window_size = 40
X_test_list = []
valid_indices = [] 

for j in range(0, len(X_data) - window_size + 1, 1):
    X_test_list.append(X_data[j : j + window_size, :])
    valid_indices.append(j + window_size - 1) 

X_test = np.array(X_test_list)
print(f"✅ Test data converted; {len(X_test)} consecutive window prediction points produced.")

# ==========================================
# 4. run the model prediction
# ==========================================
print("🧠 Predicting...")
predictions = model.predict(X_test)
pred_emg_scaled = predictions[0]  
pred_comp_prob = predictions[1].flatten() # flatten dimensions for the computations below

# inverse-transform the EMG back to actual %MVC values
pred_emg_real = scaler_y_emg.inverse_transform(pred_emg_scaled)
pred_emg_main = pred_emg_real[:, 0]
pred_emg_comp = pred_emg_real[:, 1]

# ==========================================
# 4.5 🔥 new: ±100-frame smoothing logic
# ==========================================
print("🔧 Smoothing the predicted signal (±100 frames)...")
smoothed_comp_binary = np.zeros_like(pred_comp_prob)
half_window = 100

for i in range(len(pred_comp_prob)):
    # decide the window range (without going past the array bounds)
    start_idx = max(0, i - half_window)
    end_idx = min(len(pred_comp_prob), i + half_window + 1)
    
    # take the predicted probabilities in this range
    window_data = pred_comp_prob[start_idx:end_idx]
    
    # count frames in this range whose "probability < 0.5"
    count_less_than_half = np.sum(window_data < 0.5)
    
    # core logic: if more than 20 frames are below 0.5, output 0 (no compensation), otherwise output 1 (compensation)
    if count_less_than_half > 20:
        smoothed_comp_binary[i] = 0
    else:
        smoothed_comp_binary[i] = 1

# ==========================================
# 5. plot the prediction results
# ==========================================
print("📈 Generating charts...")

time_axis = valid_indices 

plt.figure(figsize=(14, 12))

# --- chart 1: predicted activation of the two muscles (%MVC) ---
plt.subplot(3, 1, 1)
plt.plot(time_axis, pred_emg_main, label='Predicted EMG Main (main muscle)', color='red', linewidth=2)
plt.plot(time_axis, pred_emg_comp, label='Predicted EMG Compass (synergist)', color='orange', linewidth=2)
plt.title(f'Predicted Muscle Forces (%MVC) - {os.path.basename(test_csv_path)}')
plt.ylabel('EMG Force (%MVC)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')

# --- chart 2: 🔥 revised knee-valgus compensation decision (binary output) ---
plt.subplot(3, 1, 2)
# drawstyle='steps-mid' makes the 0/1 changes look like a square wave
plt.plot(time_axis, smoothed_comp_binary, label='Smoothed Valgus Detection (after smoothing)', 
         color='purple', linewidth=2, drawstyle='steps-mid')

# fill the compensation regions (1) with colour for a better visual
plt.fill_between(time_axis, 0, smoothed_comp_binary, step='mid', color='purple', alpha=0.3)

plt.title('Smoothed Knee Valgus Detection (0: Normal, 1: Compensation)')
plt.ylabel('Status (0 or 1)')
plt.yticks([0, 1], ['0 (normal)', '1 (compensation)']) # force the Y axis to show only 0 and 1
plt.ylim(-0.2, 1.2)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')

# --- chart 3: the true knee-valgus angle (for reference) ---
plt.subplot(3, 1, 3)
plt.plot(true_valgus_angle, label='True Valgus Angle (actual valgus angle)', color='gray', alpha=0.8, linewidth=2)
plt.axhline(y=30.0, color='red', linestyle='--', label='Compensation Threshold (30°)')

plt.axvline(x=valid_indices[0], color='black', linestyle=':', label='Prediction Start')

plt.title('True Knee Valgus Angle (Reference)')
plt.xlabel('Time (Frames)')
plt.ylabel('Angle (Degrees)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

print("🎉 Finished!")