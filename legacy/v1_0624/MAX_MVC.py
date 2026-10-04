import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

# ==========================================
# parameter area
# ==========================================
# ✅ make sure this is the path of your original 1000Hz raw data file!
MVC_FILE_PATH = r"<DOWNLOADS>\0625data\data\OTHER\S05_0625_Raw_DATA.csv"
FS = 1000.0  # hardware raw sampling rate

# ==========================================
# core filter function (SOS floating-point precision issue fixed)
# ==========================================
def apply_full_emg_pipeline(data, fs):
    """Standard EMG processing pipeline: remove DC -> 60Hz notch -> 20Hz high-pass -> full-wave rectification -> 5Hz low-pass envelope"""
    # 1. remove the DC offset
    signal_centered = data - np.mean(data)
    
    # 2. 60Hz notch filter
    b_notch, a_notch = iirnotch(60.0, 30.0, fs)
    signal_notched = filtfilt(b_notch, a_notch, signal_centered)
    
    # 3. 20Hz high-pass filter
    nyq = 0.5 * fs
    sos_high = butter(4, 20.0 / nyq, btype='high', output='sos')
    signal_high = sosfiltfilt(sos_high, signal_notched)
    
    # 4. full-wave rectification
    signal_rectified = np.abs(signal_high)
    
    # 5. 5Hz low-pass filter
    sos_low = butter(4, 5.0 / nyq, btype='low', output='sos')
    signal_envelope = sosfiltfilt(sos_low, signal_rectified)
    
    return signal_high, signal_envelope

# ==========================================
# main program: dynamic segmentation and visualization
# ==========================================
def calculate_max_mvc_per_segment():
    print(f"📂 Reading the raw MVC test file: {MVC_FILE_PATH}")
    try:
        df = pd.read_csv(MVC_FILE_PATH, encoding='big5', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(MVC_FILE_PATH, encoding='cp950', low_memory=False)
    except FileNotFoundError:
        print(f"❌ File not found: {MVC_FILE_PATH}; check that the path is correct!")
        return

    # 💡 dynamic segmentation logic: find the points where the hardware restarts timing (Time_ms == 1)
    start_indices = df[df['Time_ms'] == 1].index.tolist()
    
    # if Time_ms == 1 is never found, the file is probably continuous, so treat the whole file as one set
    if not start_indices:
        start_indices = [0]
        
    start_indices.append(len(df)) 

    global_max_main = 0
    global_max_comp = 0
    best_seg_main = 0
    best_seg_comp = 0

    print("\n" + "="*40)
    print(" 📊 Dynamic segmentation and per-set MVC report")
    print("="*40)

    # compute set by set
    for i in range(len(start_indices) - 1):
        seg_num = i + 1
        start_idx = start_indices[i]
        end_idx = start_indices[i+1]
        
        # cut out this set's data
        seg_df = df.iloc[start_idx:end_idx].copy()
        
        # skip sets that are too short (e.g. noise breakpoints)
        if len(seg_df) < FS: # at least one second of data is required
            continue
            
        time_sec = seg_df['Time_ms'].values / 1000.0

        # --- process the main muscle (Raw0) ---
        raw0 = pd.to_numeric(seg_df['Raw0'], errors='coerce').fillna(0).values
        raw0_clean, raw0_env = apply_full_emg_pipeline(raw0, FS)
        seg_max_main = np.max(raw0_env)

        # --- process the synergist (Raw1) ---
        raw1 = pd.to_numeric(seg_df['Raw1'], errors='coerce').fillna(0).values
        raw1_clean, raw1_env = apply_full_emg_pipeline(raw1, FS)
        seg_max_comp = np.max(raw1_env)

        # update the global maxima
        if seg_max_main > global_max_main:
            global_max_main = seg_max_main
            best_seg_main = seg_num
            
        if seg_max_comp > global_max_comp:
            global_max_comp = seg_max_comp
            best_seg_comp = seg_num

        max_ms = seg_df['Time_ms'].max()
        print(f"🎬 set {seg_num:02d} (duration {max_ms/1000.0:.1f}s) | main MVC: {seg_max_main:6.1f} | synergist MVC: {seg_max_comp:6.1f}")

        # --- plot this set ---
        plt.figure(figsize=(12, 6))
        
        # main-muscle subplot
        plt.subplot(2, 1, 1)
        plt.plot(time_sec, raw0_clean, color='blue', alpha=0.3, label='Filtered AC Signal (Raw0)')
        plt.plot(time_sec, raw0_env, color='red', linewidth=2, label='Smoothed Envelope')
        plt.axhline(y=seg_max_main, color='green', linestyle='--', linewidth=2, label=f'Max MVC: {seg_max_main:.1f}')
        plt.title(f'Segment {seg_num}: Main Muscle (Raw0)', fontsize=12, fontweight='bold')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

        # synergist subplot
        plt.subplot(2, 1, 2)
        plt.plot(time_sec, raw1_clean, color='blue', alpha=0.3, label='Filtered AC Signal (Raw1)')
        plt.plot(time_sec, raw1_env, color='orange', linewidth=2, label='Smoothed Envelope')
        plt.axhline(y=seg_max_comp, color='green', linestyle='--', linewidth=2, label=f'Max MVC: {seg_max_comp:.1f}')
        plt.title(f'Segment {seg_num}: Compass Muscle (Raw1)', fontsize=12, fontweight='bold')
        plt.xlabel('Time (Seconds)')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show(block=False) 

    # --- output the overall result ---
    print("\n" + "🔥"*20)
    print(" 🏆 Final maximal voluntary contraction (Global MVC) conclusion")
    print("🔥"*20)
    print(f"▶ main muscle (Raw0) overall maximum: {global_max_main:.2f} (from set {best_seg_main})")
    print(f"▶ synergist (Raw1) overall maximum: {global_max_comp:.2f} (from set {best_seg_comp})")
    print("-" * 40)
    print("💡 Enter the two values above into your `emg_skelton_combined.py` script:")
    print(f"   MVC_MAIN_CLEAN_MAX = {global_max_main:.1f}")
    print(f"   MVC_COMP_CLEAN_MAX = {global_max_comp:.1f}")
    print("=" * 40 + "\n")
    
    # keep the program paused until the user closes every chart window
    print("👀 Waveforms for all sets are open; close all chart windows to end the program...")
    plt.show()

if __name__ == '__main__':
    calculate_max_mvc_per_segment()