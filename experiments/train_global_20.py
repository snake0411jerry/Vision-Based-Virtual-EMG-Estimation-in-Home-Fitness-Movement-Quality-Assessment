# English summary: Trains a 20-subject global ST-GCN (weights not included in this repo).
# (Added Nov 2026; the detailed comments below are in Chinese.)
"""
train_global_20.py — 用 S01–S20 全部 20 人訓練部署用的全域模型
=========================================================================
模型與超參數同 loso_10v20_compare.py 的 stgcn_posva（IC3MT slide 21 的 ST-GCN）：
20 關節 × 9 通道（座標 + 速度 + 加速度）+ 4 個靜態特徵 → 主動肌 / 協同肌活化度。

⚠️ 全域模型看過所有人，**不能拿來報泛化成績**；跨受試者的誠實數字請引用
   loso_10v20_compare.py 的 pool20 結果。這裡印出的訓練集 r 只用來確認訓練有收斂。

輸出 models/global_20/：
    global20_stgcn_posva.h5      Keras 模型
    global20_scaler_x.pkl        骨架 (V×C) 的 StandardScaler
    global20_scaler_static.pkl   靜態特徵的 StandardScaler
    global20_config.json         視窗、通道、關節順序、訓練受試者與 seed
"""
# --- 路徑設定：一律由 repo 根的 paths.py 集中管理，請勿改回絕對路徑 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import MODELS_DIR, PIPELINE_DIR  # noqa: E402
# --- 路徑設定結束 ---

import argparse
import json
import os
import sys

import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)
sys.path.insert(0, HERE)

from eval_utils import full_metrics                                             # noqa: E402
from loso_10v20_compare import load_skeleton, on_cpu, CHANNELS, FILTERS, OLD, NEW      # noqa: E402
from loso_stgcn_compare import skeleton_windows                                 # noqa: E402
from loso_train_and_save import WINDOW_SIZE, STEP_SIZE, EPOCHS, BATCH_SIZE, STATIC_COLS  # noqa: E402
from models_stgcn import build_stgcn                                            # noqa: E402
from build_skeleton_dataset import JOINT_NAMES                                  # noqa: E402

OUT_DIR = os.path.join(MODELS_DIR, 'global_20')


def load_global20(out_dir=OUT_DIR):
    """載入全域模型與 scaler。

    GraphConv 的 get_config 沒有存鄰接矩陣，`load_model` 會失敗 —— 改成用同樣設定重建架構再載權重。
    回傳 (model, scaler_x, scaler_static, config)。
    """
    with open(os.path.join(out_dir, 'global20_config.json'), encoding='utf-8') as f:
        cfg = json.load(f)
    model = build_stgcn(cfg['window_size'], 3 * len(cfg['channels']), len(cfg['static_cols']),
                        filters=tuple(cfg['filters']))
    model.load_weights(os.path.join(out_dir, 'global20_stgcn_posva.h5'))
    return (model, joblib.load(os.path.join(out_dir, 'global20_scaler_x.pkl')),
            joblib.load(os.path.join(out_dir, 'global20_scaler_static.pkl')), cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    os.makedirs(OUT_DIR, exist_ok=True)

    data = load_skeleton()
    subjects = OLD + NEW
    tr = [seg for s in subjects for seg in data[s]]
    V = tr[0][0].shape[1]
    sx = StandardScaler().fit(np.concatenate([X.reshape(len(X), V * CHANNELS) for X, _, _ in tr]))
    ss = StandardScaler().fit(np.stack([st for _, _, st in tr]))
    X, S, Y = skeleton_windows(tr, sx, ss)
    print(f'🦴 {len(subjects)} 人 / {len(tr)} 段 / {len(X):,} 個視窗', flush=True)

    model = build_stgcn(WINDOW_SIZE, CHANNELS, len(STATIC_COLS), filters=FILTERS)
    Xc, Sc, Yc = on_cpu(X, S, Y)
    model.fit({'ts_input': Xc, 'static_input': Sc}, {'out_emg': Yc},
              epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)

    model.save(os.path.join(OUT_DIR, 'global20_stgcn_posva.h5'))
    joblib.dump(sx, os.path.join(OUT_DIR, 'global20_scaler_x.pkl'))
    joblib.dump(ss, os.path.join(OUT_DIR, 'global20_scaler_static.pkl'))
    cfg = dict(model='ST-GCN (stgcn_posva)', window_size=WINDOW_SIZE, step_size=STEP_SIZE,
               channels=['coord_xyz', 'vel_xyz', 'acc_xyz'], joints=JOINT_NAMES,
               coords='midHip 為原點、除以 ref_height（LShoulder_Y−LAnkle_Y 的 95 百分位）',
               static_cols=STATIC_COLS, outputs=['EMG_Main_MVC', 'EMG_Compass_MVC'],
               filters=list(FILTERS), epochs=EPOCHS, batch_size=BATCH_SIZE, seed=args.seed,
               train_subjects=subjects, n_params=int(model.count_params()))
    with open(os.path.join(OUT_DIR, 'global20_config.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # 只確認訓練有收斂（訓練集上的 r，不是泛化成績）
    print('\n訓練集 r（⚠️ 非泛化成績）:')
    for s in subjects:
        Xv, Sv, Yv = skeleton_windows(data[s], sx, ss)
        m = full_metrics(Yv, model.predict({'ts_input': Xv, 'static_input': Sv}, verbose=0))
        print(f'  {s}: 主動肌 r={m["r"][0]:.3f}  協同肌 r={m["r"][1]:.3f}')
    print(f'\n✅ 模型與 scaler 已存到 {OUT_DIR}（參數 {model.count_params():,}）')


if __name__ == '__main__':
    main()
