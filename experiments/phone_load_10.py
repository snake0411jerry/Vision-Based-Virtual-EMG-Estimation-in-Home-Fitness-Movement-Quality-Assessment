# English summary: Single-phone check (MediaPipe skeleton, S01-S10, 3 seeds): skeleton only r 0.704 -> load-dropout model with load 0.785 (10/10 subjects, p = 0.001), without load 0.727.
# (Added Nov 2026; the detailed comments below are in Chinese.)
"""
phone_load_10.py — 手機骨架（MediaPipe 單鏡頭）上，加負重 / 負重 dropout 還有沒有用（S01–S10 LOSO）
=====================================================================================================
動機：0.793（conv8_load）與 dropout 都是在 OpenCap 雙手機骨架上量的；部署用的是一支手機的 MediaPipe 骨架。
前 10 人有逐幀對齊的 MediaPipe 特徵（Dataset/Combined_mediapipe，與 Combined_20 幀數、肌電標籤完全相同）。

訓練資料（部署情境，沿用 loso_dualdomain 的 both 設定）：
  OpenCap 骨架：測試者以外的 19 人 ＋ MediaPipe 骨架：測試者以外的 9 人（S01–S10 中）
測試：測試者的 MediaPipe 骨架（主要），以及同一人的 OpenCap 骨架（對照，免費）。
負重兩欄與肌電標籤只跟「組」有關，兩種骨架共用。

兩個 arm（同 fold 種子）：
  skel   8 維骨架特徵（只用骨架）
  drop   8 維 ＋ 負重兩欄 ＋ 缺值旗標，訓練時每個視窗 30% 機率遮住負重（同 load_dropout_20）
         測試兩種情境：true（有填負重）、missing（沒填）
種子 [42, 1, 2] × S01–S10 × 2 arm = 60 次訓練。fold 種子 = seed × 100 + 測試者在 20 人中的序號。

用法：
  python experiments/phone_load_10.py --max-jobs 2          # 量時間：每個 arm 各跑 1 個 fold（結果保留，可續跑）
  python experiments/phone_load_10.py                         # 全部（建議單一程序；這台筆電多程序反而慢且會當）
  python experiments/phone_load_10.py --summarize-only
輸出 results/phone_load/：raw.csv、summary.csv、tests.csv
"""
# --- 路徑設定：一律由 repo 根的 paths.py 集中管理，請勿改回絕對路徑 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import PIPELINE_DIR, RESULTS_DIR  # noqa: E402
# --- 路徑設定結束 ---

import argparse
import glob
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)
sys.path.insert(0, HERE)

from load_moment_20 import load_data, FEATURES_8, LOAD_COLS, COMBINED_20  # noqa: E402
from load_sensitivity import keep_awake                                   # noqa: E402

MP_DIR = os.path.join(os.path.dirname(COMBINED_20), 'Combined_mediapipe')
SEEDS = [42, 1, 2]
TEST_SUBJECTS = [f'S{i:02d}' for i in range(1, 11)]
ALL20 = [f'S{i:02d}' for i in range(1, 21)]
ARMS = ['skel', 'drop']
LOAD_COLS_ALL = FEATURES_8 + LOAD_COLS
P_DROP = 0.3
OUT = os.path.join(RESULTS_DIR, 'phone_load')
RAW = os.path.join(OUT, 'raw.csv')
METRICS = ['r_main', 'r_syn', 'nrmse_main', 'nrmse_syn']


def load_both():
    """回傳 oc[s]、mp[s]（各為該受試者各組 DataFrame 的 list，順序一致）。mp 只把 8 維特徵換成 MediaPipe 的。"""
    oc, _ = load_data()
    paths = sorted(glob.glob(os.path.join(COMBINED_20, '*_Combined_Features.csv')))
    mp = {}
    for s in TEST_SUBJECTS:
        names = [os.path.basename(p) for p in paths if re.match(rf'^{s}_Seg_\d+_', os.path.basename(p))]
        assert len(names) == len(oc[s])
        out = []
        for name, df in zip(names, oc[s]):
            m = pd.read_csv(os.path.join(MP_DIR, name))
            assert len(m) == len(df), f'{name} 幀數不符'
            d = df.copy()
            d[FEATURES_8] = m[FEATURES_8].values
            out.append(d)
        mp[s] = out
    return oc, mp


def done_jobs():
    if not os.path.exists(RAW):
        return set()
    d = pd.read_csv(RAW)
    return set(map(tuple, d[['arm', 'seed', 'Subject']].drop_duplicates().values))


def run(args):
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    from eval_utils import make_windows, full_metrics
    from loso_train_and_save import (build_model as build_conv1d, WINDOW_SIZE, STEP_SIZE,
                                     EPOCHS, BATCH_SIZE, STATIC_COLS, LABEL_EMG_COLS)
    keep_awake()
    os.makedirs(OUT, exist_ok=True)
    oc, mp = load_both()
    have = done_jobs()
    # 依 (seed, 受試者) 排，同一 fold 的兩個 arm 相鄰 → --max-jobs 2 剛好各量一個 arm
    jobs = [(arm, seed, s) for seed in SEEDS for s in TEST_SUBJECTS for arm in ARMS if (arm, seed, s) not in have]
    if args.max_jobs:
        jobs = jobs[:args.max_jobs]
    print(f'🔁 待跑 {len(jobs)} 次訓練（已完成 {len(have)} / {len(SEEDS) * len(TEST_SUBJECTS) * len(ARMS)}）', flush=True)

    n_load = len(LOAD_COLS_ALL)
    load_idx = [LOAD_COLS_ALL.index(c) for c in LOAD_COLS]
    load_mask = np.zeros(n_load + 1, np.float32); load_mask[load_idx] = 1.0
    flag_vec = np.zeros(n_load + 1, np.float32); flag_vec[n_load] = 1.0
    load_mask, flag_vec = tf.constant(load_mask), tf.constant(flag_vec)

    def drop(x, y):
        X = x['ts_input']
        m = tf.cast(tf.random.uniform([tf.shape(X)[0], 1, 1]) < P_DROP, X.dtype)
        X = X * (1.0 - m * load_mask) + m * flag_vec
        return {'ts_input': X, 'static_input': x['static_input']}, y

    def with_flag(X, missing):
        X = np.concatenate([np.asarray(X, np.float32), np.zeros(X.shape[:2] + (1,), np.float32)], axis=-1)
        if missing:
            X[..., load_idx] = 0.0
            X[..., n_load] = 1.0
        return X

    t0 = time.time()
    for i, (arm, seed, hold) in enumerate(jobs, 1):
        ft = time.time()
        fold_seed = seed * 100 + ALL20.index(hold)
        tf.keras.utils.set_random_seed(fold_seed)
        cols = FEATURES_8 if arm == 'skel' else LOAD_COLS_ALL
        tr = [d for s in ALL20 if s != hold for d in oc[s]] + [d for s in TEST_SUBJECTS if s != hold for d in mp[s]]
        big = pd.concat(tr, ignore_index=True)
        s_ts = StandardScaler().fit(big[cols].values)
        s_st = StandardScaler().fit(big[STATIC_COLS].values)
        Xtr, Str, Ytr, _ = make_windows(tr, cols, STATIC_COLS, LABEL_EMG_COLS, None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
        Str, Ytr = np.asarray(Str, np.float32), np.asarray(Ytr, np.float32)
        if arm == 'skel':
            Xtr = np.asarray(Xtr, np.float32)
            with tf.device('/CPU:0'):
                ds = (tf.data.Dataset.from_tensor_slices(({'ts_input': Xtr, 'static_input': Str}, {'out_emg': Ytr}))
                      .shuffle(len(Xtr), seed=fold_seed, reshuffle_each_iteration=True)
                      .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
            model = build_conv1d(len(cols), len(STATIC_COLS))
        else:
            Xtr = with_flag(Xtr, missing=False)
            with tf.device('/CPU:0'):
                ds = (tf.data.Dataset.from_tensor_slices(({'ts_input': Xtr, 'static_input': Str}, {'out_emg': Ytr}))
                      .shuffle(len(Xtr), seed=fold_seed, reshuffle_each_iteration=True)
                      .batch(BATCH_SIZE).map(drop, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE))
            model = build_conv1d(len(cols) + 1, len(STATIC_COLS))
        model.fit(ds, epochs=EPOCHS, verbose=0)
        rows = []
        for domain, data in [('phone', mp[hold]), ('opencap', oc[hold])]:
            Xv, Sv, Yv, _ = make_windows(data, cols, STATIC_COLS, LABEL_EMG_COLS, None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
            for sc in (['skel'] if arm == 'skel' else ['true', 'missing']):
                X = np.asarray(Xv, np.float32) if arm == 'skel' else with_flag(Xv, missing=(sc == 'missing'))
                P = model.predict({'ts_input': X, 'static_input': Sv}, verbose=0, batch_size=1024)
                m = full_metrics(Yv, np.asarray(P[0] if isinstance(P, list) else P, np.float32))
                rows.append({'arm': arm, 'scenario': sc, 'domain': domain, 'seed': seed, 'Subject': hold,
                             'r_main': m['r'][0], 'r_syn': m['r'][1], 'nrmse_main': m['nrmse'][0], 'nrmse_syn': m['nrmse'][1],
                             'minutes': (time.time() - ft) / 60})
        tf.keras.backend.clear_session()
        pd.DataFrame(rows).to_csv(RAW, mode='a', header=not os.path.exists(RAW), index=False, encoding='utf-8-sig')
        ph = {r['scenario']: r['r_main'] for r in rows if r['domain'] == 'phone'}
        print(f'[{i}/{len(jobs)}] {arm} seed={seed} test={hold} 手機 r_main ' +
              ' '.join(f'{k}={v:.3f}' for k, v in ph.items()) +
              f'  ({(time.time() - ft) / 60:.1f} 分，累計 {(time.time() - t0) / 60:.0f} 分)', flush=True)
    print('✅ 完成', flush=True)


def summarize():
    if not os.path.exists(RAW):
        print('尚無結果')
        return
    d = pd.read_csv(RAW).drop_duplicates(['arm', 'scenario', 'domain', 'seed', 'Subject'], keep='last')
    d['cond'] = d.arm + '_' + d.scenario.where(d.arm == 'drop', '')
    d['cond'] = d.cond.str.rstrip('_')
    folds = d.groupby(['seed', 'Subject']).arm.nunique()
    keep = set(folds[folds == len(ARMS)].index)
    d = d[[(a, b) in keep for a, b in d[['seed', 'Subject']].values]]
    print(f'完成 fold（兩個 arm 都跑完）：{len(keep)} / {len(SEEDS) * len(TEST_SUBJECTS)}；'
          f'平均每次訓練 {d.groupby(["arm", "seed", "Subject"]).minutes.max().mean():.1f} 分')
    subj = d.groupby(['domain', 'cond', 'Subject'])[METRICS].mean()
    summ = subj.groupby(['domain', 'cond'])[METRICS].mean()
    summ['worst_subject_r'] = subj.r_main.groupby(['domain', 'cond']).min()
    summ.to_csv(os.path.join(OUT, 'summary.csv'), encoding='utf-8-sig')
    pairs = [('phone', 'drop_true', 'phone', 'skel', '手機：加負重（dropout 有填）vs 只用骨架'),
             ('phone', 'drop_missing', 'phone', 'skel', '手機：dropout 沒填 vs 只用骨架'),
             ('opencap', 'drop_true', 'opencap', 'skel', 'OpenCap：加負重 vs 只用骨架（同 10 人對照）'),
             ('phone', 'skel', 'opencap', 'skel', '只用骨架：手機 − OpenCap'),
             ('phone', 'drop_true', 'opencap', 'drop_true', '加負重：手機 − OpenCap')]
    tests = []
    for da, a, db, b, lab in pairs:
        try:
            x, y = subj.loc[(da, a)].r_main, subj.loc[(db, b)].r_main
        except KeyError:
            continue
        y = y.reindex(x.index)
        dd = x - y
        tests.append({'比較': lab, 'r_a': x.mean(), 'r_b': y.mean(), 'diff': dd.mean(),
                      'p': stats.ttest_rel(x, y).pvalue if len(x) > 1 else np.nan,
                      'n_better': int((dd > 0).sum()), 'n': len(x)})
    tests = pd.DataFrame(tests)
    tests.to_csv(os.path.join(OUT, 'tests.csv'), index=False, encoding='utf-8-sig')
    pd.set_option('display.width', 250)
    print('\n' + summ.round(4).to_string())
    print('\n配對比較（每人先取種子平均，n = 人數）\n' + tests.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-jobs', type=int, default=0)
    ap.add_argument('--summarize-only', action='store_true')
    a = ap.parse_args()
    if a.summarize_only:
        summarize()
    else:
        run(a)
        summarize()


if __name__ == '__main__':
    main()
