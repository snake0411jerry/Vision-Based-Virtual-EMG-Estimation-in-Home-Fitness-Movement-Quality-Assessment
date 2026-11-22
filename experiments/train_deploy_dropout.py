# English summary: Trains the deployment load-dropout model on all data (OpenCap S01-S20 + MediaPipe S01-S10) and exports Keras, TFLite, scalers and deploy_config.json to models/deploy_dropout/.
# (Added Nov 2026; the detailed comments below are in Chinese.)
"""
train_deploy_dropout.py — 訓練給 App 用的「負重 dropout」部署模型（全部受試者，不留測試者）
================================================================================================
設定與 phone_load_10.py 的 drop arm 完全相同，只是不留任何人：
  訓練資料：OpenCap 骨架 S01–S20 ＋ MediaPipe 骨架 S01–S10（手機情境）
  輸入：8 維骨架特徵 ＋ 負重兩欄（Load_pct1RM、Load_per_BW）＋ 缺值旗標，共 11 維
  訓練時每個視窗 30% 機率遮住負重（負重兩欄設 0、旗標設 1），所以沒填負重也能用
留一驗證成績（手機骨架，3 種子 × 10 人）：有填 r 0.785、沒填 0.727、只用骨架 0.704。
這顆模型看過所有人，不能再拿來評估；成績請引用上面的 LOSO 數字。

輸出 models/deploy_dropout/：
  dropout_conv8_load.keras   Keras 模型（輸入 ts_input (N,40,11)、static_input (N,4)；輸出 (N,2)）
  dropout_conv8_load.tflite  同一模型的 TFLite 版（若轉換成功）
  scaler_ts.pkl / scaler_static.pkl   sklearn StandardScaler（只作用在前 10 維與 4 個靜態欄）
  deploy_config.json         欄位順序、scaler 平均與標準差、視窗、負重算法 —— 非 Python 的 App 照這份做前處理
  README.md                  給 App 端的使用說明

用法：python experiments/train_deploy_dropout.py [--seed 42]
"""
# --- 路徑設定：一律由 repo 根的 paths.py 集中管理，請勿改回絕對路徑 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import PIPELINE_DIR, MODELS_DIR  # noqa: E402
# --- 路徑設定結束 ---

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)
sys.path.insert(0, HERE)

from phone_load_10 import load_both, ALL20, TEST_SUBJECTS, LOAD_COLS_ALL, P_DROP  # noqa: E402
from load_moment_20 import FEATURES_8, LOAD_COLS                                  # noqa: E402

OUT = os.path.join(MODELS_DIR, 'deploy_dropout')
NAME = 'dropout_conv8_load'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    from eval_utils import make_windows, full_metrics
    from loso_train_and_save import (build_model, WINDOW_SIZE, STEP_SIZE, EPOCHS, BATCH_SIZE,
                                     STATIC_COLS, LABEL_EMG_COLS)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    tf.keras.utils.set_random_seed(args.seed)
    oc, mp = load_both()
    cols = LOAD_COLS_ALL
    n_load = len(cols)
    load_idx = [cols.index(c) for c in LOAD_COLS]

    tr = [d for s in ALL20 for d in oc[s]] + [d for s in TEST_SUBJECTS for d in mp[s]]
    big = pd.concat(tr, ignore_index=True)
    s_ts = StandardScaler().fit(big[cols].values)
    s_st = StandardScaler().fit(big[STATIC_COLS].values)
    Xtr, Str, Ytr, _ = make_windows(tr, cols, STATIC_COLS, LABEL_EMG_COLS, None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
    Xtr = np.concatenate([np.asarray(Xtr, np.float32), np.zeros(np.shape(Xtr)[:2] + (1,), np.float32)], axis=-1)
    Str, Ytr = np.asarray(Str, np.float32), np.asarray(Ytr, np.float32)
    print(f'訓練視窗 {len(Xtr):,}（OpenCap 20 人 ＋ MediaPipe 10 人）', flush=True)

    lm = np.zeros(n_load + 1, np.float32); lm[load_idx] = 1.0
    fv = np.zeros(n_load + 1, np.float32); fv[n_load] = 1.0
    lm, fv = tf.constant(lm), tf.constant(fv)

    def drop(x, y):
        X = x['ts_input']
        m = tf.cast(tf.random.uniform([tf.shape(X)[0], 1, 1]) < P_DROP, X.dtype)
        return {'ts_input': X * (1.0 - m * lm) + m * fv, 'static_input': x['static_input']}, y

    with tf.device('/CPU:0'):
        ds = (tf.data.Dataset.from_tensor_slices(({'ts_input': Xtr, 'static_input': Str}, {'out_emg': Ytr}))
              .shuffle(len(Xtr), seed=args.seed, reshuffle_each_iteration=True)
              .batch(BATCH_SIZE).map(drop, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE))
    model = build_model(n_load + 1, len(STATIC_COLS))
    model.fit(ds, epochs=EPOCHS, verbose=2)
    kpath = os.path.join(OUT, f'{NAME}.keras')
    model.save(kpath)
    pickle.dump(s_ts, open(os.path.join(OUT, 'scaler_ts.pkl'), 'wb'))
    pickle.dump(s_st, open(os.path.join(OUT, 'scaler_static.pkl'), 'wb'))

    # 自我檢查（訓練集內，只確認前處理與兩種輸入方式都接得通，不是泛化成績）
    chk = {}
    m2 = tf.keras.models.load_model(kpath)
    for s in ['S01', 'S15']:
        data = mp[s] if s in mp else oc[s]
        Xv, Sv, Yv, _ = make_windows(data, cols, STATIC_COLS, LABEL_EMG_COLS, None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
        Xv = np.concatenate([np.asarray(Xv, np.float32), np.zeros(np.shape(Xv)[:2] + (1,), np.float32)], axis=-1)
        for sc in ('true', 'missing'):
            X = Xv.copy()
            if sc == 'missing':
                X[..., load_idx] = 0.0; X[..., n_load] = 1.0
            P = m2.predict({'ts_input': X, 'static_input': np.asarray(Sv, np.float32)}, verbose=0, batch_size=1024)
            chk[f'{s}_{sc}'] = round(float(full_metrics(Yv, np.asarray(P, np.float32))['r'][0]), 3)
    print('自我檢查（訓練集內 r_main）：', chk, flush=True)

    # TFLite
    tfl = None
    try:
        conv = tf.lite.TFLiteConverter.from_keras_model(m2)
        open(os.path.join(OUT, f'{NAME}.tflite'), 'wb').write(conv.convert())
        tfl = f'{NAME}.tflite'
        print('TFLite 轉換成功', flush=True)
    except Exception as e:  # noqa: BLE001
        print('TFLite 轉換失敗（Keras 檔仍可用）：', repr(e)[:300], flush=True)

    cfg = {
        'model': 'Conv1D 8 維骨架 ＋ 負重 ＋ 負重 dropout（P=0.3）',
        'files': {'keras': f'{NAME}.keras', 'tflite': tfl,
                  'scaler_ts': 'scaler_ts.pkl', 'scaler_static': 'scaler_static.pkl'},
        'fps': 60, 'window_size': WINDOW_SIZE, 'step_size': STEP_SIZE,
        'inputs': {
            'ts_input': {'shape': [None, WINDOW_SIZE, n_load + 1],
                         'columns': cols + ['Load_missing_flag'],
                         'scaled_columns': cols,
                         'scaler_mean': s_ts.mean_.tolist(), 'scaler_scale': s_ts.scale_.tolist()},
            'static_input': {'shape': [None, len(STATIC_COLS)], 'columns': STATIC_COLS,
                             'scaler_mean': s_st.mean_.tolist(), 'scaler_scale': s_st.scale_.tolist()},
        },
        'static_raw': {'Subj_Age': '年齡（歲）/ 100', 'Subj_Height_m': '身高（cm）/ 100',
                       'Subj_Weight_norm': '體重（kg）/ 100', 'Subj_Gender': '性別，與 pipeline/subjects.csv 的 gender 欄同一套編碼'},
        'load_raw': {'Load_pct1RM': '這組總重量 kg（含 20 kg 槓）/ 1RM（kg）；徒手 = 0',
                     'Load_per_BW': '這組總重量 kg（含 20 kg 槓）/ 體重 kg；徒手 = 0',
                     '1RM 不知道時': '最重一組約 10 下極限 → 1RM ≈ 該重量 / 0.7'},
        'load_missing': '沒填負重：先照常做標準化，再把 Load_pct1RM、Load_per_BW 兩欄（標準化後）設 0，旗標設 1；有填時旗標 = 0',
        'outputs': {'columns': LABEL_EMG_COLS, 'unit': '%MVC / 100 的包絡（0–1 左右），每個視窗一個值'},
        'feature_source': 'FEATURES_8 由 pipeline/Features_insert_FIXED.py 從 TRC 骨架算出（MediaPipe 走 build_combined_mediapipe.py 同一套公式）',
        'train': {'opencap_subjects': ALL20, 'mediapipe_subjects': TEST_SUBJECTS, 'seed': args.seed,
                  'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'n_windows': int(len(Xtr)),
                  'n_params': int(m2.count_params())},
        'loso_performance_phone': {'r_main_load_given': 0.785, 'r_main_load_missing': 0.727,
                                   'r_main_skeleton_only_model': 0.704,
                                   'source': 'results/phone_load/tests.csv（3 種子 × S01–S10）'},
        'self_check_in_sample_r_main': chk,
    }
    json.dump(cfg, open(os.path.join(OUT, 'deploy_config.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'✅ 已存到 {OUT}（{(time.time() - t0) / 60:.1f} 分）', flush=True)


if __name__ == '__main__':
    main()
