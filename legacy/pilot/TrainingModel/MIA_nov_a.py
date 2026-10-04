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
data_dir = r"<DATASET_ROOT>\0602\Combined"
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

# 🔥 fix data leakage: the label column 'Comp_Prob_Target' and the index column 'Unnamed: 0' must be excluded
exclude_keywords = ['EMG_', 'Knee_Toe_Diff', 'vel', 'acc', '_V', '_A', 'Velocity', 'Acceleration', 'Comp_Prob_Target', 'Unnamed: 0']
feature_cols = [col for col in combined_train_df.columns if not any(k in col for k in exclude_keywords)]

print(f"✅ After the fix - final number of features used: {len(feature_cols)}")
# ⚠️ confirm the printed count is 11 and that Comp_Prob_Target is not among them
print(f"🔍 feature list sample: {feature_cols[:5]}...")
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
# 3.5 training-history visualization (learning curve) - in the style of Screenshot 2026-06-12 100106.jpg
# ==========================================
print("\n📊 Plotting the model's training history...")

# get the history dictionary
hist = history.history
epochs_range = range(1, len(hist['loss']) + 1)

# define the metrics to plot (Keras multi-task generates these keys automatically)
# structure: (key in history, chart title)
metrics_to_plot = [
    ('loss', 'Total Loss'),                         # total loss
    ('out_emg_loss', 'EMG Loss (MSE)'),             # task 1: EMG loss
    ('out_comp_loss', 'Compensation Loss (BCE)'),   # task 2: compensation classification loss
    ('out_emg_mae', 'EMG MAE'),                     # task 1: EMG mean absolute error
    ('out_comp_accuracy', 'Compensation Accuracy')  # task 2: compensation classification accuracy
]

# define an exponential moving average (EMA) smoothing function for the smooth dashed lines in the chart
def smooth_curve(points, factor=0.7):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

# create a 2x3 grid of charts (2 rows, 3 columns)
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Model Training & Validation History', fontsize=16, fontweight='bold')
axes = axes.flatten()

for i, (metric, title) in enumerate(metrics_to_plot):
    ax = axes[i]
    
    # read the training and validation data
    train_data = hist[metric]
    val_data = hist[f'val_{metric}']
    
    # 1. plot the raw data (solid line with markers, slightly transparent)
    ax.plot(epochs_range, train_data, marker='.', linestyle='-', color='tab:blue', alpha=0.3, label='train (raw)')
    ax.plot(epochs_range, val_data, marker='.', linestyle='-', color='tab:orange', alpha=0.3, label='val (raw)')
    
    # 2. plot the smoothed data (thicker dashed line, for the long-term trend)
    ax.plot(epochs_range, smooth_curve(train_data), linestyle='--', color='tab:blue', linewidth=2, label='train (smooth)')
    ax.plot(epochs_range, smooth_curve(val_data), linestyle='--', color='tab:orange', linewidth=2, label='val (smooth)')
    
    # set chart labels and details
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Value')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=9)

# we only have 5 metrics, so the 6th cell is empty; hide it
axes[5].axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.92) # leave room for the main title
plt.show()
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
print("\n💾 Saving the multi-task model (experiment B: no kinetic inputs)...")
model.save("teacher_multitask_random_noVA.keras")
joblib.dump(scaler_x, "scaler_x_multitask_random_noVA.pkl")
joblib.dump(scaler_y_emg, "scaler_y_emg_multitask_random.pkl") # the Y scaler is unaffected, but saving a new file is safer
print("✅ Experiment B model saved!")