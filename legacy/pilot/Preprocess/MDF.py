import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- parameters ---
FILE_PATH = r"<CLEAN_DIR>\S06_CLEAN_SYNCED_FFT_1000Hz.csv"
FS = 1000                             
CHANNEL = 'Raw1'                      

# --- interval settings ---
ACTIVE_WINDOW = (0, 226) # the contraction interval you set originally

# --- spectral analysis and sliding-window parameters ---
FREQ_MIN = 20  # lowest valid frequency
FREQ_MAX = 150 # highest valid frequency
WINDOW_SIZE_SEC = 1  # size of each window (1 s = 1000 samples)
STEP_SIZE_SEC = 1    # step per slide (0.1 s = 100 samples, i.e. 80% overlap)

def calculate_mdf_trend(file_path, channel, fs):
    # 1. read and clean the data (keeps the robust cleaning logic of the previous version)
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    df['Time_ms'] = pd.to_numeric(df['Time_ms'], errors='coerce')
    df[channel] = pd.to_numeric(df[channel], errors='coerce')
    df = df.dropna(subset=['Time_ms', channel])
    
    time_sec = df['Time_ms'].values / 1000.0  
    raw_signal = df[channel].values

    # 2. remove the DC offset & extract the contraction interval
    signal_centered = raw_signal - np.mean(raw_signal)
    active_mask = (time_sec >= ACTIVE_WINDOW[0]) & (time_sec <= ACTIVE_WINDOW[1])
    
    active_signal = signal_centered[active_mask]
    active_time = time_sec[active_mask]
    
    if len(active_signal) < int(WINDOW_SIZE_SEC * fs):
        print("❌ Error: your contraction interval is shorter than the window size; sliding-window analysis is impossible!")
        return

    # ==========================================
    # 🌟 core: compute the MDF with a sliding window
    # ==========================================
    window_pts = int(WINDOW_SIZE_SEC * fs) # 500 points
    step_pts = int(STEP_SIZE_SEC * fs)     # 100 points
    
    mdf_values = []
    time_points = []
    
    # build Hanning window weights to reduce spectral edge errors
    hanning_win = np.hanning(window_pts)

    print("\n🔍 Starting sliding-window FFT and MDF computation...")
    for start_idx in range(0, len(active_signal) - window_pts + 1, step_pts):
        end_idx = start_idx + window_pts
        
        # take this small window's signal and record the window's "centre time"
        segment = active_signal[start_idx:end_idx]
        t_center = active_time[start_idx + window_pts // 2]
        
        # apply the Hanning window to smooth the edges
        segment_windowed = segment * hanning_win
        
        # run the FFT
        freqs = np.fft.rfftfreq(window_pts, d=1/fs)
        mag = np.abs(np.fft.rfft(segment_windowed)) / window_pts
        mag[1:] = mag[1:] * 2 
        
        # take the valid 20~150Hz band
        valid_idx = np.where((freqs >= FREQ_MIN) & (freqs <= FREQ_MAX))[0]
        valid_freqs = freqs[valid_idx]
        valid_mag = mag[valid_idx]
        
        if len(valid_mag) > 0:
            power = valid_mag ** 2
            total_power = np.sum(power)
            
            # guard against division by 0
            if total_power > 0:
                cumulative_power = np.cumsum(power)
                mdf_idx = np.where(cumulative_power >= total_power / 2)[0][0]
                mdf_values.append(valid_freqs[mdf_idx])
                time_points.append(t_center)

    print(f"✅ Done! {len(mdf_values)} MDF data points produced.")

    # ==========================================
    # 5. plotting
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # --- top subplot: raw muscle signal in the contraction interval ---
    ax1.plot(active_time, active_signal, color='blue', linewidth=0.8)
    ax1.set_title(f'Active EMG Signal ({ACTIVE_WINDOW[0]}s - {ACTIVE_WINDOW[1]}s)', fontsize=14)
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)

    # --- bottom subplot: MDF decreasing over time ---
    ax2.plot(time_points, mdf_values, marker='o', markersize=4, color='purple', 
             linestyle='-', linewidth=1.5, alpha=0.7, label='MDF Value (per window)')
    
    # 📈 compute a linear regression (trend line)
    if len(time_points) > 1:
        z = np.polyfit(time_points, mdf_values, 1) # first-order (straight-line) regression
        p = np.poly1d(z)
        slope = z[0] # slope
        
        # draw the trend line
        ax2.plot(time_points, p(time_points), color='red', linestyle='--', 
                 linewidth=2.5, label=f'Trendline (Slope: {slope:.3f} Hz/s)')
        
        # annotate the slope on the chart
        if slope < 0:
            conclusion = "📉 Fatigue Detected (MDF Decreasing)"
            color = 'red'
        else:
            conclusion = "⚖️ No Fatigue (MDF Stable/Increasing)"
            color = 'green'
            
        ax2.text(0.02, 0.05, f"{conclusion}\nRate of Change: {slope:.3f} Hz/sec", 
                 transform=ax2.transAxes, fontsize=12, fontweight='bold', 
                 color=color, bbox=dict(facecolor='S08', alpha=0.8))

    ax2.set_title('Median Frequency (MDF) Trend Over Time', fontsize=14)
    ax2.set_xlabel('Time (Seconds)', fontsize=12)
    ax2.set_ylabel('MDF (Hz)', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    calculate_mdf_trend(FILE_PATH, CHANNEL, FS)