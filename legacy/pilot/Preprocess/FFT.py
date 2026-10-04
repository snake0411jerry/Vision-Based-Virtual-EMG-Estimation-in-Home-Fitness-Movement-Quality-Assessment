import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- parameters ---
FILE_PATH = r"<CLEAN_DIR>\S06_CLEAN_SYNCED_FFT_1000Hz.csv"
FS = 1000                             
CHANNEL = 'Raw1'                      

# --- SNR intervals (in seconds) ---
REST_WINDOW = (30, 35)   # rest interval
ACTIVE_WINDOW = (120, 150) # contraction interval

# --- spectral analysis parameters ---
FREQ_MIN = 20  # ignore low-frequency motion noise below 20Hz
FREQ_MAX = 150 # ignore high-frequency instrument noise above 150Hz

def calculate_rms(signal_array):
    if len(signal_array) == 0: return 1e-6 
    return np.sqrt(np.mean(signal_array**2))

def analyze_emg_with_snr(file_path, channel, fs):
    # 1. read and clean the data
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    df['Time_ms'] = pd.to_numeric(df['Time_ms'], errors='coerce')
    df[channel] = pd.to_numeric(df[channel], errors='coerce')
    df = df.dropna(subset=['Time_ms', channel])
    
    if len(df) == 0:
        print("❌ Error: no data left after dropping missing values!")
        return

    time_sec = df['Time_ms'].values / 1000.0  
    raw_signal = df[channel].values

    # 2. remove the DC offset
    signal_centered = raw_signal - np.mean(raw_signal)

    # 3. extract intervals and compute SNR
    rest_mask = (time_sec >= REST_WINDOW[0]) & (time_sec <= REST_WINDOW[1])
    active_mask = (time_sec >= ACTIVE_WINDOW[0]) & (time_sec <= ACTIVE_WINDOW[1])

    rest_signal = signal_centered[rest_mask]
    active_signal = signal_centered[active_mask]

    rms_rest = calculate_rms(rest_signal)
    rms_active = calculate_rms(active_signal)
    snr_db = 20 * np.log10(rms_active / rms_rest)

    print(f"[{channel} signal quality report]")
    print(f"rest-interval RMS: {rms_rest:.2f} (ADC units)")
    print(f"contraction-interval RMS: {rms_active:.2f} (ADC units)")
    print(f"SNR (signal-to-noise ratio): {snr_db:.2f} dB")

    # 4. compute the FFT
    n_active = len(active_signal)
    freqs = np.fft.rfftfreq(n_active, d=1/FS)
    fft_magnitude = np.abs(np.fft.rfft(active_signal)) / n_active
    fft_magnitude[1:] = fft_magnitude[1:] * 2 

    # ==========================================
    # 🌟 new: muscle fatigue index (median frequency, MDF)
    # ==========================================
    # find the frequency indices between 20Hz and 150Hz
    valid_idx = np.where((freqs >= FREQ_MIN) & (freqs <= FREQ_MAX))[0]
    valid_freqs = freqs[valid_idx]
    valid_mag = fft_magnitude[valid_idx]
    
    # compute power (magnitude squared)
    power = valid_mag ** 2
    total_power = np.sum(power)
    
    # compute cumulative power and find the frequency where 50% of the total power is reached (that is the MDF)
    cumulative_power = np.cumsum(power)
    mdf_idx = np.where(cumulative_power >= total_power / 2)[0][0]
    mdf_value = valid_freqs[mdf_idx]
    
    print(f"Muscle fatigue index (MDF, {FREQ_MIN}-{FREQ_MAX}Hz): {mdf_value:.2f} Hz")
    # ==========================================

    # 5. plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # --- top subplot: time domain ---
    ax1.plot(time_sec, raw_signal, color='blue', linewidth=0.5)
    ax1.axhline(y=np.mean(raw_signal), color='red', linestyle='--', label='Baseline')
    ax1.axvspan(REST_WINDOW[0], REST_WINDOW[1], color='green', alpha=0.2, label='Resting Window (Noise)')
    ax1.axvspan(ACTIVE_WINDOW[0], ACTIVE_WINDOW[1], color='orange', alpha=0.2, label='Active Window (Signal)')
    
    ax1.text(0.02, 0.9, f"SNR: {snr_db:.1f} dB", transform=ax1.transAxes, 
             fontsize=14, fontweight='bold', bbox=dict(facecolor='S08', alpha=0.8))

    ax1.set_title(f'EMG Time Domain Signal ({channel})')
    ax1.set_xlabel('Time (Seconds)')
    ax1.set_ylabel('ADC Value')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- bottom subplot: frequency domain (focused on the valid band) ---
    ax2.plot(freqs, fft_magnitude, color='purple', linewidth=1, alpha=0.5, label='Raw FFT')
    
    # highlight the valid band used for the MDF (20~150Hz)
    ax2.plot(valid_freqs, valid_mag, color='indigo', linewidth=1.5, label=f'Valid Range ({FREQ_MIN}-{FREQ_MAX}Hz)')
    
    # draw a vertical line at the MDF
    ax2.axvline(x=mdf_value, color='red', linestyle='-', linewidth=2, label=f'MDF: {mdf_value:.1f} Hz')

    ax2.set_title('EMG Frequency Spectrum & Median Frequency (MDF)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.set_xlim(0, 200) # 💡 limit the X axis to 0~200Hz to zoom in on the region of interest
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    analyze_emg_with_snr(FILE_PATH, CHANNEL, FS)