import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# ==========================================
# 1. batch-read the personal data and do feature engineering
# ==========================================
data_dir = r"<DATASET_ROOT>\0625\Combined"

file_pattern = os.path.join(data_dir, "S01_Seg_*_Combined_Features.csv")

file_paths = glob.glob(file_pattern)

print(f"📂 Found {len(file_paths)} personal data files; processing...")

static_cols = ['Subj_Age', 'Subj_Height_m', 'Subj_Weight_norm', 'Subj_Gender', 'Load_1RM_Ratio']
label_emg_cols = ['EMG_Main_MVC', 'EMG_Compass_MVC']
label_comp_cols = ['Comp_Heel_Raise', 'Comp_Knee_Valgus', 'Comp_Trunk_Lean']

df_list = []
df_all_concat = pd.DataFrame() # temporary big table used to build/adapt the scaler

for fp in file_paths:
    df_temp = pd.read_csv(fp)
    
    # label engineering
    df_temp['Comp_Heel_Raise'] = ((df_temp['L_Heel_Rise_norm'] > 0.02) | (df_temp['R_Heel_Rise_norm'] > 0.02)).astype(float)
    df_temp['Comp_Knee_Valgus'] = (df_temp['Knee_Ankle_Ratio_norm'] < 0.92).astype(float)
    df_temp['Comp_Trunk_Lean'] = (df_temp['Trunk_Lean_Angle_norm'] > (40.0 / 180.0)).astype(float)
    
    df_list.append(df_temp)

# merge into one big table, only to find the dynamic column names and to have it ready for the scaler
df_all_concat = pd.concat(df_list, ignore_index=True)

# select the dynamic feature columns
exclude = ['EMG_', 'Knee_Toe_Diff', 'vel', 'acc', 'Target', 'Unnamed', 'Subj_', 'Comp_']
ts_cols = [col for col in df_all_concat.columns if not any(k in col for k in exclude)]

# handle the scaler (try loading the global one; otherwise build it from all current S01 data)
try:
    scaler_ts = joblib.load('global_scaler_ts.pkl')
    scaler_static = joblib.load('global_scaler_static.pkl')
    print("✅ Global scaler loaded")
except FileNotFoundError:
    print("⚠️ Global scaler not found; building a new one from the current personal data")
    scaler_ts = StandardScaler().fit(df_all_concat[ts_cols].values)
    scaler_static = StandardScaler().fit(df_all_concat[static_cols].values)

# ==========================================
# 2. cut sliding windows safely (avoid producing wrong windows across file boundaries)
# ==========================================
WINDOW_SIZE = 40
STEP_SIZE = 5

def create_windows_safely(dfs):
    X_ts, X_static, Y_emg, Y_comp = [], [], [], []
    
    # 💡 key: transform and window "each file" independently
    for df in dfs:
        ts_data = scaler_ts.transform(df[ts_cols].values)
        static_data = scaler_static.transform(df[static_cols].values)
        emg_data = df[label_emg_cols].values
        comp_data = df[label_comp_cols].values
        
        for j in range(0, len(ts_data) - WINDOW_SIZE, STEP_SIZE):
            X_ts.append(ts_data[j : j + WINDOW_SIZE, :])
            target_idx = j + WINDOW_SIZE - 1 
            X_static.append(static_data[target_idx, :]) 
            Y_emg.append(emg_data[target_idx, :])
            Y_comp.append(comp_data[target_idx, :])
            
    return np.array(X_ts), np.array(X_static), np.array(Y_emg), np.array(Y_comp)

# call the function to process all files at once
X_S01_ts, X_S01_stat, Y_S01_emg, Y_S01_comp = create_windows_safely(df_list)

# split into fine-tuning training and validation sets
X_train_ts, X_val_ts, X_train_stat, X_val_stat, Y_train_emg, Y_val_emg, Y_train_comp, Y_val_comp = train_test_split(
    X_S01_ts, X_S01_stat, Y_S01_emg, Y_S01_comp, test_size=0.2, random_state=42
)

print(f"✅ total fine-tuning training samples: {len(X_train_ts)} | total fine-tuning validation samples: {len(X_val_ts)}")

# ==========================================
# 3. load the global model and freeze the feature layers
# ==========================================
model_path = 'global_fitness_model.keras'
print(f"\n🔄 Loading the global model: {model_path}")
personalized_model = tf.keras.models.load_model(model_path)

print("\n❄️ Freezing the convolution and feature-extraction layers (keeping the skeleton-dynamics recognition ability)...")
for layer in personalized_model.layers:
    if 'feature' in layer.name:
        layer.trainable = False
        print(f"  🔒 frozen: {layer.name}")
    else:
        layer.trainable = True 
        print(f"  🔓 trainable: {layer.name}")

# ==========================================
# 4. recompile the model (very low learning rate)
# ==========================================
# fine-tune with 1e-5 to avoid heavily destroying the globally pre-trained weights
personalized_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
    loss={'out_emg': 'mse', 'out_comp': 'binary_crossentropy'},
    loss_weights={'out_emg': 3.0, 'out_comp': 1.0},
    metrics={'out_emg': 'mae', 'out_comp': 'accuracy'}
)

print("\nFine-tuning model architecture overview:")
personalized_model.summary()

# ==========================================
# 5. run personalized training (fine-tuning)
# ==========================================
print("\n🎯 Starting personalized fine-tuning...")

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5, 
    restore_best_weights=True
)

history_finetune = personalized_model.fit(
    {'ts_input': X_train_ts, 'static_input': X_train_stat}, 
    {'out_emg': Y_train_emg, 'out_comp': Y_train_comp},
    epochs=20,     # fine-tuning does not need many epochs
    batch_size=16, # for personal data, lower the batch size to update weights more often
    validation_data=(
        {'ts_input': X_val_ts, 'static_input': X_val_stat}, 
        {'out_emg': Y_val_emg, 'out_comp': Y_val_comp}
    ),
    callbacks=[early_stopping],
    verbose=1
)

# ==========================================
# 6. save the personal model
# ==========================================
save_name = 'S01_personalized_fitness_model.keras'
personalized_model.save(save_name)
print(f"\n✅ Personalized model saved to: {save_name}")

print("\n📊 Generating the training-history chart...")

# set a font that supports Chinese (if needed; Matplotlib's built-in default is used here)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False # fix minus-sign rendering

# create a 2x5 grid of subplots
fig, axs = plt.subplots(2, 5, figsize=(20, 10))
fig.suptitle('S01 personalized fine-tuning training history', fontsize=16)

# define the metrics to plot and their titles
metrics = [
    ('loss', 'total_loss'),
    ('out_emg_loss', 'emg_loss'),
    ('out_comp_loss', 'comp_loss'),
    ('out_emg_mae', 'emg_mae'),
    ('out_comp_accuracy', 'comp_acc')
]

# plot training-set metrics (first row)
for i, (metric_key, title_name) in enumerate(metrics):
    ax = axs[0, i]
    if metric_key in history_finetune.history:
        ax.plot(history_finetune.history[metric_key], label='results', marker='.', linestyle='-')
        
        # compute the smoothed curve (moving average)
        data = history_finetune.history[metric_key]
        if len(data) > 5:
            smooth_data = pd.Series(data).rolling(window=5, min_periods=1).mean()
            ax.plot(smooth_data, label='smooth', linestyle='--')
            
        ax.set_title(f'train/{title_name}')
        ax.grid(True, linestyle=':', alpha=0.6)
        if i == 0:
            ax.legend() # show the legend only on the first chart
    else:
        ax.set_title(f'train/{title_name} (No Data)')
        ax.axis('off') # turn the subplot off if there is no data

# plot validation-set metrics (second row)
for i, (metric_key, title_name) in enumerate(metrics):
    ax = axs[1, i]
    val_metric_key = f'val_{metric_key}'
    if val_metric_key in history_finetune.history:
        ax.plot(history_finetune.history[val_metric_key], label='results', marker='.', linestyle='-', color='orange')
        
        # compute the smoothed curve (moving average)
        data = history_finetune.history[val_metric_key]
        if len(data) > 5:
            smooth_data = pd.Series(data).rolling(window=5, min_periods=1).mean()
            ax.plot(smooth_data, label='smooth', linestyle='--', color='red')
            
        ax.set_title(f'val/{title_name}')
        ax.grid(True, linestyle=':', alpha=0.6)
    else:
        ax.set_title(f'val/{title_name} (No Data)')
        ax.axis('off')
        
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # set to Microsoft JhengHei
plt.rcParams['axes.unicode_minus'] = False
# adjust the layout
plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# show the chart
print("✅ Chart generated; displaying...")
plt.show()

# optional: save the chart
# plt.savefig('S01_finetune_history.png', dpi=300)
# print("✅ Chart saved as S01_finetune_history.png")