import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from tensorflow.keras.layers import BatchNormalization, Attention, Input, Concatenate, Flatten
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import joblib

# ==========================================
# 1. read data and isolate subjects (Train: PILOT_A, PILOT_B, TEST1 / Val: S01)
# ==========================================
# 🔥 updated to the new data path
data_dir = r"<DATASET_ROOT>\Code\0519Combined\Origin"
file_pattern = os.path.join(data_dir, "*_Combined_Features.csv")
file_paths = glob.glob(file_pattern)

if not file_paths:
    raise ValueError(f"No files found in {data_dir}; check the path!")

# 🔥 added TEST1's MVC values
subject_mvc_map = {
    'S01':     {'MVC_Main': 920, 'MVC_Compass': 750},
    'PILOT_A': {'MVC_Main': 975, 'MVC_Compass': 774}, 
    'PILOT_B':   {'MVC_Main': 827, 'MVC_Compass': 608},
    'TEST1':    {'MVC_Main': 1018, 'MVC_Compass': 1019}
}

print(f"📂 Found {len(file_paths)} files in total to process!")

train_dfs = []
val_dfs = []

for fp in file_paths:
    df = pd.read_csv(fp)
    filename = os.path.basename(fp).upper()
    
    current_subj = None
    for subj in subject_mvc_map.keys():
        if subj in filename:
            current_subj = subj
            break
            
    if current_subj is None:
        print(f"⚠️ skipping file of unknown subject: {filename}")
        continue
        
    df['Subject_MVC_Main'] = subject_mvc_map[current_subj]['MVC_Main']
    df['Subject_MVC_Compass'] = subject_mvc_map[current_subj]['MVC_Compass']
    
    # S01 goes into the validation set, the rest (PILOT_A, PILOT_B, TEST1) into the training set
    if current_subj == 'TEST1':
        val_dfs.append(df)
    else:
        train_dfs.append(df)

if not train_dfs or not val_dfs:
    raise ValueError("Training or validation set is empty! Check the file names and the split logic.")

# ==========================================
# 1.5 fit the scaler on its own (prevents data leakage)
# ==========================================
train_combined_df = pd.concat(train_dfs, ignore_index=True)

feature_cols = [col for col in train_combined_df.columns if not col.startswith('EMG_')]
label_cols = ['EMG_Main_MVC', 'EMG_Compass_MVC']

scaler_x = StandardScaler()
scaler_x.fit(train_combined_df[feature_cols].values)

scaler_y = MinMaxScaler()
scaler_y.fit(train_combined_df[label_cols].values)

print(f"✅ Loaded! training files: {len(train_dfs)}, validation (S01) files: {len(val_dfs)}")

# ==========================================
# 2. sliding-window segmentation
# ==========================================
window_size = 40  
step_size = 5     

def create_windows(df_list, dataset_name=""):
    X_list, Y_list = [], []
    segments = 0
    for df in df_list:
        X_scaled = scaler_x.transform(df[feature_cols].values)
        Y_scaled = scaler_y.transform(df[label_cols].values)
        
        for j in range(0, len(X_scaled) - window_size, step_size):
            X_list.append(X_scaled[j : j + window_size, :])
            Y_list.append(Y_scaled[j + window_size // 2, :])
        segments += 1
    
    X_arr = np.array(X_list)
    Y_arr = np.array(Y_list)
    print(f"  - {dataset_name}: {len(X_arr)} windows extracted (from {segments} Segments)")
    return X_arr, Y_arr

print("⏳ Segmenting into time windows...")
X_train, Y_train = create_windows(train_dfs, "Training set (Train)")
X_test, Y_test = create_windows(val_dfs, "Validation set (Val/S01)")

# ==========================================
# 3. model building and training
# ==========================================
print("\n🧠 Building the CNN-BiLSTM-Attention model...")
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

# CNN layer 
x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
x = BatchNormalization()(x)

# first Bi-LSTM layer
x = Bidirectional(LSTM(units=64, return_sequences=True))(x)
x = BatchNormalization()(x)
x = Dropout(0.1)(x) 

# second Bi-LSTM layer 
lstm_out = Bidirectional(LSTM(units=32, return_sequences=True, dropout=0.1))(x)

# Attention 
attention_out = Attention()([lstm_out, lstm_out])
x = Flatten()(attention_out)

# fully connected and output layers
x = Dense(units=32, activation='relu')(x)
outputs = Dense(units=2, activation='sigmoid')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# declare the optimizer
custom_adam = tf.keras.optimizers.Adam(learning_rate=0.001)

# use the Huber loss
huber_loss = tf.keras.losses.Huber(delta=0.1) 

model.compile(optimizer=custom_adam, loss=huber_loss, metrics=['mae'])
model.summary()

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.3, patience=5, min_lr=0.00001, verbose=1
)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# train the model
history = model.fit(
    X_train, Y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, Y_test), 
    callbacks=[early_stop, reduce_lr], 
    verbose=1
)

# ==========================================
# 4. prediction and visualization (on the unseen S01 data)
# ==========================================
predictions = model.predict(X_test)

plt.figure(figsize=(15, 8))
plt.subplot(2, 1, 1)
plt.plot(Y_test[:, 0], label='True EMG_Main (%MVC)', color='blue', alpha=0.7)
plt.plot(predictions[:, 0], label='Predicted EMG_Main', color='red', linestyle='--')
plt.title('Performance on Unseen Subject (S01) - EMG Main (Huber Loss)')
plt.ylabel('EMG (0~1)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(Y_test[:, 1], label='True EMG_Compass (%MVC)', color='green', alpha=0.7)
plt.plot(predictions[:, 1], label='Predicted EMG_Compass', color='orange', linestyle='--')
plt.title('Performance on Unseen Subject (S01) - EMG Compass (Huber Loss)')
plt.xlabel('Time Windows')
plt.ylabel('EMG (0~1)')
plt.legend()

plt.tight_layout()
plt.show()

# ==========================================
# 6. compute evaluation metrics
# ==========================================
print("\n📊 Final evaluation report on the validation set (S01 only):")

y_true_main = Y_test[:, 0]
y_pred_main = predictions[:, 0]
y_true_comp = Y_test[:, 1]
y_pred_comp = predictions[:, 1]

mae_main = mean_absolute_error(y_true_main, y_pred_main)
mae_comp = mean_absolute_error(y_true_comp, y_pred_comp)
rmse_main = np.sqrt(mean_squared_error(y_true_main, y_pred_main))
rmse_comp = np.sqrt(mean_squared_error(y_true_comp, y_pred_comp))
r2_main = r2_score(y_true_main, y_pred_main)
r2_comp = r2_score(y_true_comp, y_pred_comp)
corr_main, _ = pearsonr(y_true_main, y_pred_main)
corr_comp, _ = pearsonr(y_true_comp, y_pred_comp)

print("\n💪 [EMG Main (main muscle)]")
print(f"  - waveform similarity (Pearson r) : {corr_main:.4f}")
print(f"  - explained variance (R² score)    : {r2_main:.4f}")
print(f"  - mean absolute error (MAE)        : {mae_main:.4f} (about {mae_main*100:.2f}% MVC)")
print(f"  - root mean square error (RMSE)    : {rmse_main:.4f}")

print("\n🧭 [EMG Compass (secondary muscle)]")
print(f"  - waveform similarity (Pearson r) : {corr_comp:.4f}")
print(f"  - explained variance (R² score)    : {r2_comp:.4f}")
print(f"  - mean absolute error (MAE)        : {mae_comp:.4f} (about {mae_comp*100:.2f}% MVC)")
print(f"  - root mean square error (RMSE)    : {rmse_comp:.4f}")

# ==========================================
# 7. save the model and normalizer 
# ==========================================
print("\n💾 Saving the model and normalizer...")
model.save("lightweight_emg_model_huber.h5")
joblib.dump(scaler_x, "scaler_x_huber.pkl")
print("✅ Saved!")