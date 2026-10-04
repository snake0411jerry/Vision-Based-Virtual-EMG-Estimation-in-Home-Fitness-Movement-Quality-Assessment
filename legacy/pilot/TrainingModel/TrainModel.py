import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from tensorflow.keras.layers import BatchNormalization,Attention, Input, Concatenate, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import joblib

# ==========================================
# 🔥 key change 3: a brand-new loss function (exponential penalty on peaks)
# ==========================================
def peak_weighted_mse(y_true, y_pred):
    # keep data types consistent to avoid TensorFlow errors
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # compute the basic mean squared error (MSE)
    squared_error = tf.square(y_true - y_pred)
    
    # exponential penalty! the higher the true value, the more extreme the cost of missing it
    # tf.exp makes the weight at y_true=1.0 equal to e^4 (about 54×), forcing the model to catch peaks
    peak_weight = tf.exp(4.0 * y_true) 
    
    # remove the protection for low values so the model focuses entirely on peaks
    return tf.reduce_mean(squared_error * peak_weight)

# ==========================================
# 1. read data and prepare normalization (version with personal MVC features)
# ==========================================
data_dir = r"<DATASET_ROOT>\0513\Combined"
file_pattern = os.path.join(data_dir, "*_Combined_Features.csv")
file_paths = glob.glob(file_pattern)

if not file_paths:
    raise ValueError(f"No files found in {data_dir}; check the path!")

subject_mvc_map = {
    'S01':     {'MVC_Main': 920, 'MVC_Compass': 750},
    'PILOT_A': {'MVC_Main': 975, 'MVC_Compass': 774}, 
    'PILOT_B':   {'MVC_Main': 827, 'MVC_Compass': 608}
}

print(f"📂 Found {len(file_paths)} files in total for training!")

all_dfs = []
for fp in file_paths:
    df = pd.read_csv(fp)
    
    filename = os.path.basename(fp).upper()
    current_subj = None
    for subj in subject_mvc_map.keys():
        if subj in filename:
            current_subj = subj
            break
            
    if current_subj is None:
        raise ValueError(f"⚠️ Cannot identify the subject from the file name {filename}!")
        
    df['Subject_MVC_Main'] = subject_mvc_map[current_subj]['MVC_Main']
    df['Subject_MVC_Compass'] = subject_mvc_map[current_subj]['MVC_Compass']
    
    all_dfs.append(df)

combined_df = pd.concat(all_dfs, ignore_index=True)

feature_cols = [col for col in combined_df.columns if not col.startswith('EMG_')]
label_cols = ['EMG_Main_MVC', 'EMG_Compass_MVC']

scaler_x = StandardScaler()
scaler_x.fit(combined_df[feature_cols].values)

scaler_y = MinMaxScaler()
scaler_y.fit(combined_df[label_cols].values)

# ==========================================
# 2. sliding-window segmentation (🚨 done independently within each Segment)
# ==========================================
# 🔥 key change 1: shrink the window to raise feature density so bursts of activation are not diluted
window_size = 40  
step_size = 5     

X_windows_list = []
Y_labels_list = []

print("⏳ Segmenting into time windows...")
for i, df in enumerate(all_dfs):
    X_data = scaler_x.transform(df[feature_cols].values)
    Y_data = scaler_y.transform(df[label_cols].values)
    
    segments_extracted = 0
    for j in range(0, len(X_data) - window_size, step_size):
        window_X = X_data[j : j + window_size, :]
        target_Y = Y_data[j + window_size // 2, :]
        
        X_windows_list.append(window_X)
        Y_labels_list.append(target_Y)
        segments_extracted += 1
        
    print(f"  - Segment {i+1}: {segments_extracted} windows extracted")

X_tensor = np.array(X_windows_list)
Y_tensor = np.array(Y_labels_list)
print(f"✅ {len(X_tensor)} window samples generated in total!")

# ==========================================
# 3. data split, model building and training
# ==========================================
print("\n🔀 Randomly drawing 20% of all windows as the test set...")

X_train, X_test, Y_train, Y_test = train_test_split(
    X_tensor, Y_tensor, test_size=0.2, random_state=42
)

print(f"✅ training samples: {len(X_train)}")
print(f"✅ test samples: {len(X_test)}")

print("\n🧠 Building the CNN-BiLSTM-Attention model...")
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

# 🔥 key change 2: remove MaxPooling, shrink the kernel size, lower dropout
# CNN layer 
x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
x = BatchNormalization()(x)
# (MaxPooling1D removed)

# first Bi-LSTM layer
x = Bidirectional(LSTM(units=64, return_sequences=True))(x)
x = BatchNormalization()(x)
x = Dropout(0.1)(x) # lower dropout

# second Bi-LSTM layer 
lstm_out = Bidirectional(LSTM(units=32, return_sequences=True, dropout=0.1))(x)

# Attention 
attention_out = Attention()([lstm_out, lstm_out])
x = Flatten()(attention_out)

# fully connected and output layers
x = Dense(units=32, activation='relu')(x)
outputs = Dense(units=2, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# compile the model
custom_adam = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=custom_adam, loss=peak_weighted_mse, metrics=['mae'])
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
# 4. prediction and visualization (plotting)
# ==========================================
predictions = model.predict(X_test)

plt.figure(figsize=(15, 8))
plt.subplot(2, 1, 1)
plt.plot(Y_test[:, 0], label='True EMG_Main (%MVC)', color='blue', alpha=0.7)
plt.plot(predictions[:, 0], label='Predicted EMG_Main', color='red', linestyle='--')
plt.title('Optimized CNN-BiLSTM - EMG Main (Real Causality)')
plt.ylabel('EMG (0~1)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(Y_test[:, 1], label='True EMG_Compass (%MVC)', color='green', alpha=0.7)
plt.plot(predictions[:, 1], label='Predicted EMG_Compass', color='orange', linestyle='--')
plt.title('Optimized CNN-BiLSTM - EMG Compass (Real Causality)')
plt.xlabel('Time Windows (Tested on Random 20% Data)')
plt.ylabel('EMG (0~1)')
plt.legend()

plt.tight_layout()
plt.show()

# ==========================================
# 6. compute evaluation metrics
# ==========================================
print("\n📊 Final evaluation report on the test set (20% unseen data):")

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
model.save("lightweight_emg_model.h5")
joblib.dump(scaler_x, "scaler_x.pkl")
print("✅ Saved! You now have the two files lightweight_emg_model.h5 and scaler_x.pkl.")