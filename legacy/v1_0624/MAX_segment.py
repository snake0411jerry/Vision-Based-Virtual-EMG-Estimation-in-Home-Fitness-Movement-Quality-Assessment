import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# ==========================================
# parameter area
# ==========================================
# 🚨 set the folder path of the Segment files just written
DATA_DIR = r"<DATASET_ROOT>\0622\Clean\Segmented\S01_0622_CLEAN_SEGMENTS"
# find all Segment files matching the naming pattern
FILE_PATTERN = os.path.join(DATA_DIR, "S01_0622_CLEAN_Segment_*.csv")

# ==========================================
# main program: batch computation and visualization
# ==========================================
def batch_calculate_mvc():
    # get the list of all Segment files
    segment_files = glob.glob(FILE_PATTERN)
    
    if not segment_files:
        print(f"❌ No Segment files found in {DATA_DIR}; check the path and file names!")
        return

    # sort files by Segment number (avoiding string-sort problems like 1, 10, 2)
    segment_files.sort(key=lambda x: int(x.split('_Segment_')[-1].replace('.csv', '')))

    print(f"🔍 Found {len(segment_files)} Segment files in total; starting the analysis...\n")

    for file_path in segment_files:
        segment_num = file_path.split('_Segment_')[-1].replace('.csv', '')
        print(f"📂 Analysing: set {segment_num} ({os.path.basename(file_path)})")
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig', low_memory=False)
        except Exception as e:
            print(f"Failed to read file {file_path}: {e}")
            continue

        # build the time axis (using the Time_ms in the file)
        time_sec = df['Time_ms'].values / 1000.0

        # --- process the main muscle (Raw0) ---
        # note: Raw0 here is already the "envelope" processed in the previous step
        raw0_env = pd.to_numeric(df['Raw0'], errors='coerce').fillna(0).values
        max_mvc_main = np.max(raw0_env)

        # --- process the synergist (Raw1) ---
        raw1_env = pd.to_numeric(df['Raw1'], errors='coerce').fillna(0).values
        max_mvc_comp = np.max(raw1_env)

        # --- output the result ---
        print("-" * 40)
        print(f"  [set {segment_num}] MVC result")
        print(f"▶ main muscle (Raw0) maximum envelope value: {max_mvc_main:.2f}")
        print(f"▶ synergist (Raw1) maximum envelope value: {max_mvc_comp:.2f}")
        print("-" * 40 + "\n")

        # --- plot to check signal quality ---
        plt.figure(figsize=(14, 8))
        
        # plot the main muscle
        plt.subplot(2, 1, 1)
        plt.plot(time_sec, raw0_env, color='red', linewidth=2, label='Smoothed Envelope (60 FPS)')
        plt.axhline(y=max_mvc_main, color='green', linestyle='--', linewidth=2, label=f'Max MVC: {max_mvc_main:.1f}')
        plt.title(f'Segment {segment_num}: Main Muscle (Raw0) - MVC Test', fontsize=14, fontweight='bold')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

        # plot the synergist
        plt.subplot(2, 1, 2)
        plt.plot(time_sec, raw1_env, color='orange', linewidth=2, label='Smoothed Envelope (60 FPS)')
        plt.axhline(y=max_mvc_comp, color='green', linestyle='--', linewidth=2, label=f'Max MVC: {max_mvc_comp:.1f}')
        plt.title(f'Segment {segment_num}: Compass Muscle (Raw1) - MVC Test', fontsize=14, fontweight='bold')
        plt.xlabel('Time (Seconds)')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        
        # save the image for later review, without blocking the program
        img_out_path = file_path.replace('.csv', '_MVC_Plot.png')
        plt.savefig(img_out_path)
        plt.close() # close the canvas to free memory
        print(f"📸 Image saved to: {img_out_path}\n")

    print("🎉 All Segment files analysed!")

if __name__ == '__main__':
    batch_calculate_mvc()