import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# parameters
# ==========================================
# replace with the path of your merged CSV file
CSV_PATH = r"<DATASET_ROOT>\Code\Test_video\CSV\S01_TEST_0513_Segment11-20.csv"
# set the image output folder
OUTPUT_DIR = r"<DATASET_ROOT>\Code\Test_video\img"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"📂 Reading data: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

# ==========================================
# define the list of features to compare
# ==========================================
# basic feature names listed here (without the .1 suffix)
feature_names = [
    'Shoulder_Y_norm', 'Shoulder_Y_vel', 'Shoulder_Y_acc',
    'Shoulder_Z_norm', 'Shoulder_Z_vel', 'Shoulder_Z_acc',
    'Knee_Y_norm', 'Knee_Y_vel', 'Knee_Y_acc',
    'Knee_Z_norm', 'Knee_Z_vel', 'Knee_Z_acc',
    'Ankle_Y_norm', 'Ankle_Y_vel', 'Ankle_Y_acc',
    'Ankle_Z_norm', 'Ankle_Z_vel', 'Ankle_Z_acc',
    'Knee_Angle_norm', 'Knee_Angle_vel'
]

print(f"📊 Preparing {len(feature_names)} feature comparison plots...")

# ==========================================
# plot and save in batch
# ==========================================
for feature in feature_names:
    col_opencv = feature        # column name of the original OpenCV (TRC) feature
    col_mediapipe = feature + '.1' # column name of the appended MediaPipe feature
    
    # check that the columns exist
    if col_opencv not in df.columns or col_mediapipe not in df.columns:
        print(f"⚠️ Feature {feature} not found; skipping its plot.")
        continue

    # create the figure
    plt.figure(figsize=(10, 5))
    
    # plot the OpenCV data (blue)
    plt.plot(df[col_opencv], label='OpenCV (Ground Truth)', color='blue', alpha=0.7, linewidth=1.5)
    
    # plot the MediaPipe data (orange)
    plt.plot(df[col_mediapipe], label='Mediapipe (Predicted)', color='orange', alpha=0.8, linestyle='--', linewidth=2)
    
    plt.title(f'Comparison: {feature}')
    plt.xlabel('Frame')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    # save the image
    save_path = os.path.join(OUTPUT_DIR, f"{feature}_comparison.png")
    plt.savefig(save_path, dpi=150)
    plt.close() # close the figure to free memory

print(f"✅ All feature comparison plots saved to the folder {OUTPUT_DIR}!")