import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, BatchNormalization, GlobalAveragePooling1D, Concatenate
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. read data and preprocess
# ==========================================
data_dir = r"<DATASET_ROOT>\0625\Combined"
file_paths = glob.glob(os.path.join(data_dir, "*_Combined_Features.csv"))
train_files, val_files = train_test_split(file_paths, test_size=0.2, random_state=42)

# define feature column names
static_cols = ['Subj_Age', 'Subj_Height_m', 'Subj_Weight_norm', 'Subj_Gender', 'Load_1RM_Ratio']
label_emg_cols = ['EMG_Main_MVC', 'EMG_Compass_MVC']
# 💡 update: define the column names of the three compensation labels
label_comp_cols = ['Comp_Heel_Raise', 'Comp_Knee_Valgus', 'Comp_Trunk_Lean']
def plot_predictions_vs_actuals(y_true_emg, y_pred_emg, y_true_comp, y_pred_comp, num_samples=200):
    """
    Plot true vs predicted values.
    - y_true_emg, y_pred_emg: true and predicted muscle activation (continuous values)
    - y_true_comp, y_pred_comp: true and predicted compensation (0 or 1 vs predicted probability)
    - num_samples: how many time-slice samples to plot (default 200, to avoid overcrowded lines)
    """
    # make sure not to exceed the data length
    num_samples = min(num_samples, len(y_true_emg))
    
    # create a canvas with 5 rows and 1 column (2 EMG + 3 compensation)
    fig, axes = plt.subplots(5, 1, figsize=(15, 18))
    plt.subplots_adjust(hspace=0.4)
    
    # --- task 1: EMG regression (continuous values) ---
    emg_names = ['EMG_Main_MVC', 'EMG_Compass_MVC']
    for i in range(2):
        axes[i].plot(y_true_emg[:num_samples, i], label='Actual (True)', color='#1f77b4', linestyle='-', linewidth=2)
        axes[i].plot(y_pred_emg[:num_samples, i], label='Predicted', color='#ff7f0e', linestyle='--', linewidth=2)
        axes[i].set_title(f'Regression Task: {emg_names[i]} (Predicted vs Actual)', fontsize=14, fontweight='bold')
        axes[i].set_ylabel('% MVC', fontsize=12)
        axes[i].legend(loc='upper right')
        axes[i].grid(True, linestyle=':', alpha=0.6)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        
    # --- task 2: compensation classification (probability vs actual label) ---
    comp_names = ['Comp_Heel_Raise (Heel)', 'Comp_Knee_Valgus (Knee)', 'Comp_Trunk_Lean (Trunk)']
    for i in range(3):
        ax_idx = i + 2
        
        # the true label is 0 or 1 (a step line, drawstyle='steps-mid', fits the digital logic better)
        axes[ax_idx].plot(y_true_comp[:num_samples, i], label='Actual Class (0 or 1)', 
                          color='#1f77b4', drawstyle='steps-mid', linewidth=2)
        
        # the prediction is a probability in 0~1 (sigmoid output)
        axes[ax_idx].plot(y_pred_comp[:num_samples, i], label='Predicted Probability', 
                          color='#ff7f0e', linestyle='--', linewidth=2)
        
        # draw a 0.5 threshold reference line for easy visual judgement
        axes[ax_idx].axhline(y=0.5, color='gray', linestyle=':', label='0.5 Threshold')
        
        axes[ax_idx].set_title(f'Classification Task: {comp_names[i]} (Probability vs Actual)', fontsize=14, fontweight='bold')
        axes[ax_idx].set_ylabel('Probability / Class', fontsize=12)
        axes[ax_idx].set_ylim(-0.1, 1.1) # fix the classification range at slightly more than 0~1
        axes[ax_idx].legend(loc='upper right')
        axes[ax_idx].grid(True, linestyle=':', alpha=0.6)
        axes[ax_idx].spines['top'].set_visible(False)
        axes[ax_idx].spines['right'].set_visible(False)
        
    plt.xlabel('Sample Index (Time Windows)', fontsize=12)
    plt.tight_layout()
    save_name = 'prediction_vs_actual.png'
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"\n🎯 Prediction vs actual plot saved to: {save_name}")
    plt.show()
def plot_yolo_style_results(history, save_path='results.png'):
    h = history.history
    epochs = range(1, len(h['loss']) + 1)
    
    # create a 2x5 grid canvas
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    # define the data and titles for the first row (Train)
    row1_metrics = [
        ('train/total_loss', 'loss'),
        ('train/emg_loss', 'out_emg_loss'),
        ('train/comp_loss', 'out_comp_loss'),
        ('metrics/train_emg_mae', 'out_emg_mae'),
        ('metrics/train_comp_acc', 'out_comp_accuracy')
    ]
    
    # define the data and titles for the second row (Val)
    row2_metrics = [
        ('val/total_loss', 'val_loss'),
        ('val/emg_loss', 'val_out_emg_loss'),
        ('val/comp_loss', 'val_out_comp_loss'),
        ('metrics/val_emg_mae', 'val_out_emg_mae'),
        ('metrics/val_comp_acc', 'val_out_comp_accuracy')
    ]
    
    # helper: draw a single chart panel
    def plot_panel(ax, title, data):
        # 1. raw data (blue solid line with points)
        ax.plot(epochs, data, marker='.', markersize=6, linestyle='-', 
                linewidth=1.5, color='#1f77b4', label='results')
        
        # 2. smoothed data (orange dashed line, moving average via pandas rolling)
        smooth_data = pd.Series(data).rolling(window=3, min_periods=1).mean()
        ax.plot(epochs, smooth_data, linestyle=':', linewidth=2, 
                color='#ff7f0e', alpha=0.8, label='smooth')
        
        ax.set_title(title, fontsize=12)
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        
        # remove the top and right spines for a cleaner chart (YOLO style)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # show the legend only on the first chart
        if title == 'train/total_loss':
            ax.legend(loc='upper right', frameon=True)

    # draw the first row (Train)
    for i, (title, key) in enumerate(row1_metrics):
        if key in h:
            plot_panel(axes[0, i], title, h[key])
        
    # draw the second row (Val)
    for i, (title, key) in enumerate(row2_metrics):
        if key in h:
            plot_panel(axes[1, i], title, h[key])
        
    # auto-layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📈 Training result chart saved in the current directory as: {save_path}")
    plt.show()

def load_data(files):
    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        
        # 🌟 label engineering: set thresholds on the geometric features from step 2 (thresholds can be adjusted to the actual experimental standard)
        # 1. tiptoeing: left or right heel raised above some fraction of body height (set to 0.02 as an example)
        df['Comp_Heel_Raise'] = ((df['L_Heel_Rise_norm'] > 0.02) | (df['R_Heel_Rise_norm'] > 0.02)).astype(float)
        
        # 2. knee valgus: knee distance / ankle distance below 0.92 (the knees are not spread enough and cave inwards)
        df['Comp_Knee_Valgus'] = (df['Knee_Ankle_Ratio_norm'] < 0.92).astype(float)
        
        # 3. trunk lean: lean angle greater than 40 degrees (40.0 / 180.0)
        df['Comp_Trunk_Lean'] = (df['Trunk_Lean_Angle_norm'] > (40.0 / 180.0)).astype(float)
        
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True), dfs

train_df_all, train_dfs = load_data(train_files)
val_df_all, val_dfs = load_data(val_files)

# dynamic features (excluding static, label and irrelevant columns)
# 💡 update: add 'Comp_' to exclude so the newly created label columns do not become training features
exclude = ['EMG_', 'Knee_Toe_Diff', 'vel', 'acc', 'Target', 'Unnamed', 'Subj_', 'Comp_']
ts_cols = [col for col in train_df_all.columns if not any(k in col for k in exclude)]

print(f"Number of dynamic features used for training: {len(ts_cols)}")
print(f"Dynamic feature columns: {ts_cols}")

# create the scaler
scaler_ts = StandardScaler().fit(train_df_all[ts_cols].values)
scaler_static = StandardScaler().fit(train_df_all[static_cols].values)

# ==========================================
# 2. cut sliding windows
# ==========================================
WINDOW_SIZE = 40
STEP_SIZE = 5

def create_multi_modal_windows(df_list):
    X_ts, X_static, Y_emg, Y_comp = [], [], [], []
    for df in df_list:
        ts_data = scaler_ts.transform(df[ts_cols].values)
        static_data = scaler_static.transform(df[static_cols].values)
        emg_data = df[label_emg_cols].values 
        comp_data = df[label_comp_cols].values # 💡 update: here all three label values are taken at once (shape: N x 3)
        
        for j in range(0, len(ts_data) - WINDOW_SIZE, STEP_SIZE):
            X_ts.append(ts_data[j : j + WINDOW_SIZE, :])
            target_idx = j + WINDOW_SIZE - 1 
            X_static.append(static_data[target_idx, :]) 
            Y_emg.append(emg_data[target_idx, :])
            Y_comp.append(comp_data[target_idx, :]) # 💡 what gets stored is a one-hot-like array of size 3 (e.g. [0, 1, 0])
            
    return np.array(X_ts), np.array(X_static), np.array(Y_emg), np.array(Y_comp)

X_train_ts, X_train_stat, Y_train_emg, Y_train_comp = create_multi_modal_windows(train_dfs)
X_val_ts, X_val_stat, Y_val_emg, Y_val_comp = create_multi_modal_windows(val_dfs)

print(f"training samples: {len(X_train_ts)} | validation samples: {len(X_val_ts)}")
print(f"compensation label training-set shape: {Y_train_comp.shape}") # expected shape is (samples, 3)

# ==========================================
# 3. 🧠 build the multimodal multi-task model (upgraded to multi-label classification)
# ==========================================
# input 1: continuous skeleton signal
input_ts = Input(shape=(WINDOW_SIZE, len(ts_cols)), name='ts_input')
# input 2: subject static features
input_static = Input(shape=(len(static_cols),), name='static_input')

# --- time-series feature extraction (1D-CNN / TCN) ---
x_ts = Conv1D(64, 3, padding='causal', activation='relu')(input_ts)
x_ts = BatchNormalization()(x_ts)
x_ts = Conv1D(64, 3, padding='causal', dilation_rate=2, activation='relu')(x_ts)
ts_features = GlobalAveragePooling1D()(x_ts)

# --- feature fusion layer (late fusion) ---
fused_features = Concatenate()([ts_features, input_static])
fused_features = Dense(64, activation='relu')(fused_features)
fused_features = Dropout(0.2)(fused_features)

# --- task 1: EMG %MVC prediction (regression) ---
emg_branch = Dense(32, activation='relu')(fused_features)
# 💡 make sure the activation here has been changed to 'linear'
out_emg = Dense(2, activation='linear', name='out_emg')(emg_branch) 

# ==========================================
# 👇👇👇 paste your new code here 👇👇👇
# ==========================================
# --- task 2: recognition of the three main compensation movements (multi-label classification) ---
comp_branch = Dense(64, activation='relu')(fused_features) # more neurons to strengthen classification learning
comp_branch = Dropout(0.3)(comp_branch) # add Dropout to stabilize validation performance and prevent a late collapse
out_comp = Dense(3, activation='sigmoid', name='out_comp')(comp_branch)
# ==========================================
# 👆👆👆 paste your new code here 👆👆👆
# ==========================================

# bind both outputs to the model
model = tf.keras.Model(inputs=[input_ts, input_static], outputs=[out_emg, out_comp])

# ==========================================
# 4. model compilation and training
# ==========================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001), 
    # 💡 in multi-label classification the total loss of three independent binary classifications is still binary_crossentropy
    loss={'out_emg': 'mse', 'out_comp': 'binary_crossentropy'},
    loss_weights={'out_emg': 3.0, 'out_comp': 1.0}, # the weights can be adjusted according to how regression and classification converge after training
    metrics={'out_emg': 'mae', 'out_comp': 'accuracy'}
)

model.summary()

print("\n🚀 Starting multimodal multi-label training (Multi-Modal Multi-Label Training)...")

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=5, 
    min_lr=1e-5, 
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=10, 
    restore_best_weights=True
)

history = model.fit(
    {'ts_input': X_train_ts, 'static_input': X_train_stat}, 
    {'out_emg': Y_train_emg, 'out_comp': Y_train_comp},
    epochs=50, 
    batch_size=32,
    validation_data=(
        {'ts_input': X_val_ts, 'static_input': X_val_stat}, 
        {'out_emg': Y_val_emg, 'out_comp': Y_val_comp}
    ),
    callbacks=[early_stopping, reduce_lr], 
    verbose=1
)

# call the plotting function
plot_yolo_style_results(history, save_path='my_model_results.png')
# ==========================================
# 5. prediction vs actual visualization
# ==========================================

print("\n🔍 Predicting on the validation set...")
# model.predict returns a list containing the predictions of both outputs
predictions = model.predict({'ts_input': X_val_ts, 'static_input': X_val_stat})

# predictions[0] is the EMG prediction (shape: N x 2)
# predictions[1] is the compensation prediction (shape: N x 3)
pred_emg = predictions[0]
pred_comp = predictions[1]

# call the plotting function (change num_samples to choose how long a time span to view)
plot_predictions_vs_actuals(Y_val_emg, pred_emg, Y_val_comp, pred_comp, num_samples=150)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

print("\n📊 Quality of the EMG regression predictions:")
emg_names = ['EMG_Main_MVC', 'EMG_Compass_MVC']

for i in range(2):
    # extract true and predicted values (for a single muscle channel)
    y_true = Y_val_emg[:, i]
    y_pred = pred_emg[:, i]
    
    # compute the metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"--- {emg_names[i]} ---")
    print(f"  📌 MAE  (mean absolute error):    {mae:.4f} %MVC")
    print(f"  📌 RMSE (root mean square error): {rmse:.4f} %MVC")
    print(f"  🎯 R^2  (coefficient of determination): {r2:.4f} (closer to 1 = better waveform match)")