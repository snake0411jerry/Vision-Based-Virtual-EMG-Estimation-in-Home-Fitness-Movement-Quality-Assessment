import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. set the file search path and output path
# ==========================================
# this is the folder where the previous stage wrote _Predicted.csv
data_dir = r"<DATASET_ROOT>\Code\0519Combined\Predict"
search_pattern = os.path.join(data_dir, "*_Predicted.csv")
file_list = glob.glob(search_pattern)

# automatically create a new folder for these 20 charts
output_dir = os.path.join(data_dir, "Prediction_Plots")
os.makedirs(output_dir, exist_ok=True)

if not file_list:
    print("❌ No _Predicted.csv files found; check the path!")
else:
    print(f"🔍 Found {len(file_list)} prediction files; starting to plot...\n")

# ==========================================
# 2. loop over each file and plot
# ==========================================
for file_path in file_list:
    filename = os.path.basename(file_path)
    print(f"📊 Plotting: {filename}")
    
    # read the CSV containing the predictions
    df = pd.read_csv(file_path)
    
    # safeguard: check that the required columns exist
    required_cols = ['EMG_Main_MVC', 'Predicted_EMG_Main', 'EMG_Compass_MVC', 'Predicted_EMG_Compass']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"⚠️ {filename} is missing required columns {missing}; skipping this file.")
        continue
        
    # create the figure (two stacked subplots, size 15x8)
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    # ------------------------------------------
    # top: plot the Main muscle
    # ------------------------------------------
    axes[0].plot(df['EMG_Main_MVC'], label='True EMG Main (Real)', color='blue', alpha=0.7, linewidth=2)
    axes[0].plot(df['Predicted_EMG_Main'], label='Predicted EMG Main (AI)', color='red', linestyle='--', linewidth=2)
    axes[0].set_title(f'EMG Main Comparison: {filename.replace(".csv", "")}', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Muscle Activation (%MVC)', fontsize=12)
    axes[0].legend(loc='upper right', fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # ------------------------------------------
    # bottom: plot the Compass muscle
    # ------------------------------------------
    axes[1].plot(df['EMG_Compass_MVC'], label='True EMG Compass (Real)', color='green', alpha=0.7, linewidth=2)
    axes[1].plot(df['Predicted_EMG_Compass'], label='Predicted EMG Compass (AI)', color='orange', linestyle='--', linewidth=2)
    axes[1].set_title(f'EMG Compass Comparison: {filename.replace(".csv", "")}', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Time (Frames)', fontsize=12)
    axes[1].set_ylabel('Muscle Activation (%MVC)', fontsize=12)
    axes[1].legend(loc='upper right', fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    # auto-adjust the layout to avoid overlapping text
    plt.tight_layout()
    
    # ==========================================
    # 3. save the image and close the canvas to free memory
    # ==========================================
    # name the image after the original file name plus .png
    save_path = os.path.join(output_dir, filename.replace('.csv', '.png'))
    plt.savefig(save_path, dpi=150) # dpi=150 ensures a high enough resolution
    plt.close() # must close, otherwise memory blows up after 20 charts

print("-" * 50)
print(f"🎉 Done! All charts have been plotted!")
print(f"📂 Find your charts in this folder: {output_dir}")