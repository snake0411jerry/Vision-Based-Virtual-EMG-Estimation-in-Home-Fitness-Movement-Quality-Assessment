import cv2
import numpy as np
import tensorflow as tf
import keras
import joblib
from collections import deque
from tensorflow.keras.layers import LSTM, Bidirectional, Attention
import mediapipe as mp
import matplotlib.pyplot as plt
import pandas as pd
# ==========================================
# 1. parameters and model loading
# ==========================================
MODEL_PATH = r"<DATASET_ROOT>\Code\Model\lightweight_emg_model.h5"
SCALER_PATH = r"<DATASET_ROOT>\Code\Model\scaler_x.pkl"
VIDEO_PATH = r"<DATASET_ROOT>\Code\Test_video\video\S01_TEST_0513_Segment11-20.mp4"
@keras.saving.register_keras_serializable()
class SafeLSTM(LSTM):
    def __init__(self, **kwargs):
        kwargs.pop('time_major', None)
        super().__init__(**kwargs)

custom_objs = {"LSTM": SafeLSTM, "Bidirectional": Bidirectional, "Attention": Attention}
model = tf.keras.models.load_model(MODEL_PATH, compile=False, custom_objects=custom_objs) 
scaler_x = joblib.load(SCALER_PATH)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

FAKE_MVC_MAIN = 900
FAKE_MVC_COMPASS = 700

# ==========================================
# 2. skeleton feature extraction (fixed to match the TRC algorithm)
# ==========================================
def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def extract_spatial_vars_aligned(landmarks):
    # MediaPipe coordinates: x = horizontal, y = vertical (downwards), z = depth
    # convert directly to a NumPy array here
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    
    L_sh, R_sh = 11, 12
    L_hip, R_hip = 23, 24
    L_kn, R_kn = 25, 26
    L_an, R_an = 27, 28
    
    shoulder = (pts[L_sh] + pts[R_sh]) / 2.0
    hip      = (pts[L_hip] + pts[R_hip]) / 2.0
    knee     = (pts[L_kn] + pts[R_kn]) / 2.0
    ankle    = (pts[L_an] + pts[R_an]) / 2.0
    
    # 1. knee angle divided by 180.0 (matches TRC's Knee_Angle_norm)
    knee_angle_norm = float(calculate_angle(hip, knee, ankle)) / 180.0
    
    # 2. compute ref_height (shoulder Y − ankle Y distance, to estimate body-height scale)
    axis_Y, axis_Z = 1, 2
    ref_height = abs(shoulder[axis_Y] - ankle[axis_Y])
    if ref_height == 0: ref_height = 1e-5 # avoid division by 0
    
    # 3. centre the coordinates on the Hip and divide by ref_height (Norm)
    shoulder_norm_Y = (shoulder[axis_Y] - hip[axis_Y]) / ref_height
    shoulder_norm_Z = (shoulder[axis_Z] - hip[axis_Z]) / ref_height
    
    knee_norm_Y = (knee[axis_Y] - hip[axis_Y]) / ref_height
    knee_norm_Z = (knee[axis_Z] - hip[axis_Z]) / ref_height
    
    ankle_norm_Y = (ankle[axis_Y] - hip[axis_Y]) / ref_height
    ankle_norm_Z = (ankle[axis_Z] - hip[axis_Z]) / ref_height

    # return the 7 purely normalized variables
    return np.array([
        float(shoulder_norm_Y), float(shoulder_norm_Z),
        float(knee_norm_Y),     float(knee_norm_Z),
        float(ankle_norm_Y),    float(ankle_norm_Z),
        float(knee_angle_norm)
    ])

# ==========================================
# 3. quickly extract features for the whole video (fixed EMG and derivatives)
# ==========================================
# change the fake EMG values to a sensible %MVC range (e.g. 0.0)
FAKE_MVC_MAIN_NORM = 0.0 
FAKE_MVC_COMPASS_NORM = 0.0

print("⏳ Quickly parsing the video's skeleton features...")
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps): fps = 30.0
dt = 1.0 / fps

pos_history = deque(maxlen=3)
all_frame_features = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_world_landmarks:
        # use the newly fixed function to get Normalized coordinates
        current_norm_pos = extract_spatial_vars_aligned(results.pose_world_landmarks)
        pos_history.append(current_norm_pos)
        
        if len(pos_history) == 3:
            p0, p1, p2 = pos_history
            
            # differentiate the Normalized values so v and a have exactly the same scale as TRC
            v1 = (p1 - p0) / dt
            v2 = (p2 - p1) / dt
            a2 = (v2 - v1) / dt
            
            frame_features = np.array([
                p2[0], v2[0], a2[0],  # Shoulder Y norm, vel, acc
                p2[1], v2[1], a2[1],  # Shoulder Z norm, vel, acc
                p2[2], v2[2], a2[2],  # Knee Y norm, vel, acc
                p2[3], v2[3], a2[3],  # Knee Z norm, vel, acc
                p2[4], v2[4], a2[4],  # Ankle Y norm, vel, acc
                p2[5], v2[5], a2[5],  # Ankle Z norm, vel, acc
                p2[6], v2[6],         # Knee Angle norm, vel (note: no a2)
                FAKE_MVC_MAIN_NORM, FAKE_MVC_COMPASS_NORM  # put 0.0, not 900
            ])
            all_frame_features.append(frame_features)

cap.release()
print(f"✅ Video parsed; {len(all_frame_features)} frames of features extracted!")

# ==========================================
# 4. batch data processing and neural-network prediction
# ==========================================
print("🧠 Running batch model prediction...")
# scale all features first (exactly the same logic as your TrainModel.py)
all_features_np = np.array(all_frame_features)
all_scaled = scaler_x.transform(all_features_np)

window_size = 100
X_windows = []

# cut into 100-frame sliding windows
for i in range(len(all_scaled) - window_size + 1):
    X_windows.append(all_scaled[i : i + window_size])

X_windows = np.array(X_windows)

predictions = model.predict(X_windows)
# take column 0 (Main) and column 1 (Compass) precisely
pred_main = predictions[:, 0]
pred_compass = predictions[:, 1]

print("✅ Prediction done! Generating charts...")

# ==========================================
# 5. draw a nice EMG trend chart
# ==========================================
# build the time axis (X axis): seconds corresponding to frames from frame 100 onwards
time_axis = np.arange(len(pred_main)) * dt + (window_size * dt)

plt.figure(figsize=(12, 6))
plt.plot(time_axis, pred_main * 100, label='Predicted EMG Main (%MVC)', color='red', linewidth=2)
plt.plot(time_axis, pred_compass * 100, label='Predicted EMG Compass (%MVC)', color='green', linewidth=2)

plt.title('AI Predicted sEMG Muscle Activation over Squat Video', fontsize=16, fontweight='bold')
plt.xlabel('Time (Seconds)', fontsize=12)
plt.ylabel('Muscle Activation (% MVC)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()

# show the chart
plt.show()

# ==========================================
# 6. write the predictions to a CSV file (matching the attachment's format)
# ==========================================
print("📝 Generating and writing the CSV file...")

# set the 22 column names exactly as in the attachment
columns = [
    'Shoulder_Y_norm', 'Shoulder_Y_vel', 'Shoulder_Y_acc', 
    'Shoulder_Z_norm', 'Shoulder_Z_vel', 'Shoulder_Z_acc', 
    'Knee_Y_norm', 'Knee_Y_vel', 'Knee_Y_acc', 
    'Knee_Z_norm', 'Knee_Z_vel', 'Knee_Z_acc', 
    'Ankle_Y_norm', 'Ankle_Y_vel', 'Ankle_Y_acc', 
    'Ankle_Z_norm', 'Ankle_Z_vel', 'Ankle_Z_acc', 
    'Knee_Angle_norm', 'Knee_Angle_vel', 
    'EMG_Main_Predicted', 'EMG_Compass_Predicted'
]

# convert all extracted skeleton features to a DataFrame
df_output = pd.DataFrame(all_frame_features, columns=columns)

# because window_size = 100, the first 99 frames have no prediction; pad the length with NaN
pad_length = window_size - 1
padded_pred_main = np.pad(pred_main, (pad_length, 0), constant_values=np.nan)
padded_pred_compass = np.pad(pred_compass, (pad_length, 0), constant_values=np.nan)

# overwrite the fake FAKE_MVC with the AI's actual predictions
df_output['EMG_Main_Predicted'] = padded_pred_main
df_output['EMG_Compass_Predicted'] = padded_pred_compass

# set your output path
OUTPUT_CSV_PATH = r"<DATASET_ROOT>\Code\Test_video\CSV\S01_TEST_0513_Segment11-20.csv"

# write as a CSV file (without the index)
df_output.to_csv(OUTPUT_CSV_PATH, index=False)

print(f"✅ CSV file written to: {OUTPUT_CSV_PATH}")