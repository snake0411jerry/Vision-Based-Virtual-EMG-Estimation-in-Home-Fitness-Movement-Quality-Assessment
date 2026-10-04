import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, BatchNormalization, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Add
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split 
import joblib

# ==========================================
# 1. read data and preprocess labels (🔥 reverted to: random split at the file level)
# ==========================================
data_dir = r"<DATASET_ROOT>\0529\Combined" 
file_pattern = os.path.join(data_dir, "*_Combined_Features.csv")
file_paths = glob.glob(file_pattern)

if not file_paths:
    raise ValueError(f"No files found in {data_dir}; check the path!")

# 🔥 key change: shuffle the "list of file paths" and split it (80% train, 20% validation)
# random_state=42 makes every run use the same file split, which helps debugging
train_files, val_files = train_test_split(file_paths, test_size=0.2, random_state=42)

print(f"📂 Found {len(file_paths)} files in total!")
print(f"   - 🏋️ randomly assigned {len(train_files)} files to the [training set]")
print(f"   - 🧪 randomly assigned {len(val_files)} files to the [validation set]")

def load_and_preprocess(file_list):
    dfs = []
    for fp in file_list:
        df = pd.read_csv(fp)
        
        # compute the compensation-probability labels (threshold > 30 degrees)
        COMPENSATE_THRESHOLD = 30.0 / 180.0
        df['Comp_Prob_Target'] = (df['Knee_Toe_Diff_norm'] > COMPENSATE_THRESHOLD).astype(float)
        dfs.append(df)
    return dfs

train_dfs = load_and_preprocess(train_files)
val_dfs = load_and_preprocess(val_files)

combined_train_df = pd.concat(train_dfs, ignore_index=True)

# fix: correct feature and label names
exclude_keywords = ['EMG_', 'Knee_Toe_Diff']
feature_cols = [col for col in combined_train_df.columns if not any(k in col for k in exclude_keywords)]
label_emg_cols = ['EMG_Main_MVC', 'EMG_Compass_MVC'] # predict the absolute activation value
label_comp_col = ['Comp_Prob_Target']

# the scaler may only be fit on "training-set" data
scaler_x = StandardScaler()
scaler_x.fit(combined_train_df[feature_cols].values)

scaler_y_emg = MinMaxScaler() # used together with sigmoid
scaler_y_emg.fit(combined_train_df[label_emg_cols].values)

# ==========================================
# 2. sliding-window segmentation 
# ==========================================
window_size = 40  
step_size = 5     

def create_windows(df_list):
    X_list, Y_emg_list, Y_comp_list = [], [], []
    segments_lengths = []
    
    for df in df_list:
        X_data = scaler_x.transform(df[feature_cols].values)
        Y_emg_data = scaler_y_emg.transform(df[label_emg_cols].values)
        Y_comp_data = df[label_comp_col].values 
        
        count = 0
        for j in range(0, len(X_data) - window_size, step_size):
            X_list.append(X_data[j : j + window_size, :])
            target_idx = j + window_size - 1 
            Y_emg_list.append(Y_emg_data[target_idx, :])
            Y_comp_list.append(Y_comp_data[target_idx, :])
            count += 1
        segments_lengths.append(count) 
        
    return np.array(X_list), np.array(Y_emg_list), np.array(Y_comp_list), segments_lengths

print("\n⏳ Segmenting into time windows...")
X_train, Y_train_emg, Y_train_comp, _ = create_windows(train_dfs)
X_val, Y_val_emg, Y_val_comp, val_segments_lengths = create_windows(val_dfs)

print(f"✅ training samples (windows): {len(X_train)}")
print(f"✅ validation samples (windows): {len(X_val)}")

# ==========================================
# 3. model building and training (multi-task outputs)
# ==========================================
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    res = Add()([x, inputs]) 
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x) 
    return Add()([x, res]) 

print("\n🧠 Building the multi-task TCN + Transformer model...")
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

# --- shared feature-extraction layers (shared representation) ---
x = Conv1D(filters=64, kernel_size=3, padding='causal', dilation_rate=1, activation='relu')(inputs)
x = BatchNormalization()(x)
shared_features = Dropout(0.1)(x)

# 🔥 dedicated head for task 1 (EMG branch)
emg_x = Conv1D(filters=64, kernel_size=3, padding='causal', dilation_rate=2, activation='relu')(shared_features)
emg_x = transformer_encoder(emg_x, head_size=64, num_heads=2, ff_dim=128, dropout=0.1)
emg_x = GlobalAveragePooling1D()(emg_x)
emg_branch = Dense(units=32, activation='relu')(emg_x)
out_emg = Dense(units=2, activation='sigmoid', name='out_emg')(emg_branch) # sigmoid keeps it within 0~1

# 🔥 dedicated head for task 2 (compensation branch)
comp_x = GlobalAveragePooling1D()(shared_features)
comp_branch = Dense(units=16, activation='relu')(comp_x)
out_comp = Dense(units=1, activation='sigmoid', name='out_comp')(comp_branch)

model = tf.keras.Model(inputs=inputs, outputs=[out_emg, out_comp])

custom_adam = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    optimizer=custom_adam, 
    loss={
        'out_emg': 'mse',                  
        'out_comp': 'binary_crossentropy'  
    },
    # force the model to emphasize EMG by scaling its weight up 10×
    loss_weights={'out_emg': 10.0, 'out_comp': 0.1}, 
    metrics={'out_emg': 'mae', 'out_comp': 'accuracy'}
)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history = model.fit(
    X_train, 
    {'out_emg': Y_train_emg, 'out_comp': Y_train_comp},
    epochs=100,
    batch_size=32,
    validation_data=(X_val, {'out_emg': Y_val_emg, 'out_comp': Y_val_comp}),
    callbacks=[early_stop], 
    verbose=2 # one line per epoch
)

# ==========================================
# 4. prediction and visualization (including absolute MVC and post-processed slope)
# ==========================================
predictions = model.predict(X_val)
pred_emg = predictions[0]  
pred_comp = predictions[1] 

num_plots = min(3, len(val_segments_lengths))
start_idx = 0

print(f"\n📈 Preparing to plot {num_plots} consecutive motion clips from the validation set...")

for i in range(num_plots):
    seg_len = val_segments_lengths[i]
    end_idx = start_idx + seg_len
    
    # 1. take the raw prediction clip (absolute MVC)
    y_true_emg_seg = Y_val_emg[start_idx:end_idx]
    y_pred_emg_seg = pred_emg[start_idx:end_idx]
    
    y_true_comp_seg = Y_val_comp[start_idx:end_idx]
    y_pred_comp_seg = pred_comp[start_idx:end_idx]
    
    # 2. post-processing: compute the slope (rate of change) 
    true_emg_main_slope = np.diff(y_true_emg_seg[:, 0])
    pred_emg_main_slope = np.diff(y_pred_emg_seg[:, 0])
    
    true_emg_comp_slope = np.diff(y_true_emg_seg[:, 1])
    pred_emg_comp_slope = np.diff(y_pred_emg_seg[:, 1])

    # ----------------------------------------------------
    # 📊 chart A: raw predictions (MVC and compensation probability)
    # ----------------------------------------------------
    plt.figure(figsize=(12, 10))
    
    # --- main-muscle MVC ---
    plt.subplot(3, 1, 1)
    plt.plot(y_true_emg_seg[:, 0], label='True EMG_Main (%MVC)', color='blue', alpha=0.7)
    plt.plot(y_pred_emg_seg[:, 0], label='Predicted EMG_Main (sigmoid)', color='red', linestyle='--')
    plt.title(f'Validation Segment {i+1} - EMG Main Muscle Force (Absolute Amplitude)')
    plt.ylabel('EMG Amplitude [0, 1]')
    plt.legend()

    # --- synergist MVC ---
    plt.subplot(3, 1, 2)
    plt.plot(y_true_emg_seg[:, 1], label='True EMG_Compass (%MVC)', color='green', alpha=0.7)
    plt.plot(y_pred_emg_seg[:, 1], label='Predicted EMG_Compass (sigmoid)', color='orange', linestyle='--')
    plt.title(f'Validation Segment {i+1} - EMG Compass Muscle Force (Absolute Amplitude)')
    plt.ylabel('EMG Amplitude [0, 1]')
    plt.legend()
    
    # --- compensation probability ---
    plt.subplot(3, 1, 3)
    plt.plot(y_true_comp_seg, label='True Compensation (1=Yes, 0=No)', color='gray', alpha=0.5, drawstyle='steps-pre')
    plt.plot(y_pred_comp_seg, label='Predicted Probability (sigmoid)', color='purple')
    plt.axhline(y=0.5, color='r', linestyle=':', label='Threshold (0.5)')
    plt.title(f'Validation Segment {i+1} - Knee Valgus Compensation Probability')
    plt.xlabel('Time Windows')
    plt.ylabel('Probability [0, 1]')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ----------------------------------------------------
    # 📊 chart B: 🔥 post-processed slope comparison 
    # ----------------------------------------------------
    plt.figure(figsize=(12, 6))
    
    # --- main-muscle slope ---
    plt.subplot(2, 1, 1)
    plt.plot(true_emg_main_slope, label='True EMG_Main Slope (Post-processed)', color='blue', alpha=0.6)
    plt.plot(pred_emg_main_slope, label='Predicted EMG_Main Slope (Post-processed)', color='red', linestyle='-', alpha=0.8)
    plt.title(f'Validation Segment {i+1} - Post-processed Rate of Change (EMG Main Slope)')
    plt.ylabel('Rate of Change (Δ Amplitude)')
    plt.axhline(y=0, color='black', linewidth=0.8, linestyle='--') 
    plt.legend()

    # --- synergist slope ---
    plt.subplot(2, 1, 2)
    plt.plot(true_emg_comp_slope, label='True EMG_Compass Slope (Post-processed)', color='green', alpha=0.6)
    plt.plot(pred_emg_comp_slope, label='Predicted EMG_Compass Slope (Post-processed)', color='orange', linestyle='-', alpha=0.8)
    plt.title(f'Validation Segment {i+1} - Post-processed Rate of Change (EMG Compass Slope)')
    plt.xlabel('Time Windows (n-1)')
    plt.ylabel('Rate of Change (Δ Amplitude)')
    plt.axhline(y=0, color='black', linewidth=0.8, linestyle='--') 
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    start_idx = end_idx


# ==========================================
# 4.5 final overall evaluation report on the validation set
# ==========================================
print("\n" + "="*50)
print("📊 Random validation set: overall evaluation report (Validation Metrics)")
print("="*50)

all_true_slopes_main, all_pred_slopes_main = [], []
all_true_slopes_comp, all_pred_slopes_comp = [], []

start_idx = 0
for seg_len in val_segments_lengths:
    end_idx = start_idx + seg_len
    
    y_true_seg = Y_val_emg[start_idx:end_idx]
    y_pred_seg = pred_emg[start_idx:end_idx]
    
    all_true_slopes_main.extend(np.diff(y_true_seg[:, 0]))
    all_pred_slopes_main.extend(np.diff(y_pred_seg[:, 0]))
    all_true_slopes_comp.extend(np.diff(y_true_seg[:, 1]))
    all_pred_slopes_comp.extend(np.diff(y_pred_seg[:, 1]))
    
    start_idx = end_idx

# --- 1. evaluation of absolute EMG (MVC) ---
mae_main = mean_absolute_error(Y_val_emg[:, 0], pred_emg[:, 0])
mae_comp = mean_absolute_error(Y_val_emg[:, 1], pred_emg[:, 1])

print("💪 [Absolute EMG activation (Absolute MVC)]")
print(f"  - main muscle (Main) mean absolute error (MAE): {mae_main:.4f} (about {mae_main*100:.2f}%)")
print(f"  - synergist (Comp) mean absolute error (MAE): {mae_comp:.4f} (about {mae_comp*100:.2f}%)")

# 🔥 this is what was just added for you: Pearson correlation of the absolute envelope (check here for your > 0.85 target!)
mvc_corr_main, _ = pearsonr(Y_val_emg[:, 0], pred_emg[:, 0])
mvc_corr_comp, _ = pearsonr(Y_val_emg[:, 1], pred_emg[:, 1])

print("\n🎯 [EMG envelope waveform similarity (Envelope Correlation)]")
print(f"  - main muscle (Main) envelope Pearson correlation: {mvc_corr_main:.4f}")
print(f"  - synergist (Comp) envelope Pearson correlation: {mvc_corr_comp:.4f}")

# --- 2. 🔥 waveform similarity of the EMG slope (rate of change) ---
slope_corr_main, _ = pearsonr(all_true_slopes_main, all_pred_slopes_main)
slope_corr_comp, _ = pearsonr(all_true_slopes_comp, all_pred_slopes_comp)

print("\n📈 [EMG slope / dynamic-change similarity (Post-processed Slope Correlation)]")
print(f"  - main muscle (Main) waveform similarity (Pearson r): {slope_corr_main:.4f}")
print(f"  - synergist (Comp) waveform similarity (Pearson r): {slope_corr_comp:.4f}")

# --- 3. evaluation of the compensation probability ---
pred_comp_binary = (pred_comp > 0.5).astype(float)
comp_acc = accuracy_score(Y_val_comp, pred_comp_binary)

print("\n⚠️ [Compensation movement recognition (Compensation Detection)]")
print(f"  - knee-valgus binary classification accuracy: {comp_acc*100:.2f}%")
print("="*50 + "\n")

# ==========================================
# 5. save the model and normalizer 
# ==========================================
print("\n💾 Saving the multi-task model (Random Split)...")
model.save("teacher_multitask_random.keras")
joblib.dump(scaler_x, "scaler_x_multitask_random.pkl")
joblib.dump(scaler_y_emg, "scaler_y_emg_multitask_random.pkl")
print("✅ Saved! Good luck with the paper!")