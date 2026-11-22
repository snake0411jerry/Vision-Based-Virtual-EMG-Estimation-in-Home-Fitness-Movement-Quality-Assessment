# English summary: Load dropout: during training the load inputs are masked with probability 0.3 and a missing flag is set, so one model works with or without load (r 0.795 with, 0.762 without).
# (Added Nov 2026; the detailed comments below are in Chinese.)
"""
load_dropout_20.py — 負重 dropout：讓同一個模型在「有填 / 沒填負重」都能用（20 人 LOSO）
==========================================================================================
動機（load_sensitivity.py）：conv8_load 沒填負重時，補 0 = 0.712、補平均 = 0.716，都比只用骨架的 conv8（0.751）差，
因為模型訓練時負重永遠存在，學會依賴它。

做法：輸入 = 8 維骨架特徵 + 負重兩欄 + 「沒填」旗標（第 11 欄）。
  訓練時每個 batch 的每個視窗以機率 P_DROP 遮掉負重：負重兩欄設成 0（標準化後 = 訓練者平均）、旗標設成 1；
  每個 epoch 重新抽（tf.data map），所以同一個視窗有時看得到負重、有時看不到。
測試情境（同一個模型）：
  true      正確負重、旗標 0           → 對照 conv8_load（load_sens 的 true，同種子同 fold）
  missing   沒填：負重 0、旗標 1        → 對照只用骨架的 conv8（load20，同種子）
  no_bar    忘了算 20 kg 槓鈴、旗標 0   → 對照 load_sens 的 no_bar
種子 [42, 1, 2] × 20 人 = 60 個 fold；fold 種子 = seed × 100 + 測試者序號（同 load_moment_20 / load_sensitivity）。

用法：
  python experiments/load_dropout_20.py --smoke
  python experiments/load_dropout_20.py --worker k --n-workers 5
  python experiments/load_dropout_20.py --summarize-only
輸出 results/load_dropout/：raw_w*.csv、summary.csv、tests.csv
"""
# --- 路徑設定：一律由 repo 根的 paths.py 集中管理，請勿改回絕對路徑 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import PIPELINE_DIR, RESULTS_DIR  # noqa: E402
# --- 路徑設定結束 ---

import argparse
import glob
import os
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

from load_moment_20 import load_data, FEATURES_8, LOAD_COLS  # noqa: E402
from load_sensitivity import perturb, keep_awake             # noqa: E402

SEEDS = [42, 1, 2]
SUBJECTS = [f'S{i:02d}' for i in range(1, 21)]
COLS = FEATURES_8 + LOAD_COLS
LOAD_IDX = [COLS.index(c) for c in LOAD_COLS]
FLAG_IDX = len(COLS)                       # 旗標放在最後一欄
P_DROP = 0.3
SCENARIOS = ['true', 'missing', 'no_bar']
OUT = os.path.join(RESULTS_DIR, 'load_dropout')
METRICS = ['r_main', 'r_syn', 'nrmse_main', 'nrmse_syn']


def add_flag(X, missing):
    """X：(n, T, 10) 已標準化；回傳 (n, T, 11)。missing=True 時負重設 0、旗標 1。"""
    X = np.concatenate([X, np.zeros(X.shape[:2] + (1,), X.dtype)], axis=-1)
    if missing:
        X[..., LOAD_IDX] = 0.0
        X[..., FLAG_IDX] = 1.0
    return X


def done_folds(paths):
    fs = [p for p in paths if os.path.exists(p)]
    if not fs:
        return set()
    d = pd.concat([pd.read_csv(f) for f in fs])
    n = d.groupby(['seed', 'Subject']).scenario.nunique()
    return set(n[n == len(SCENARIOS)].index)


def run(args):
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    from eval_utils import make_windows, full_metrics
    from loso_train_and_save import (build_model as build_conv1d, WINDOW_SIZE, STEP_SIZE,
                                     EPOCHS, BATCH_SIZE, STATIC_COLS, LABEL_EMG_COLS)
    keep_awake()
    os.makedirs(OUT, exist_ok=True)
    by_subj, info = load_data()
    subs = sorted(by_subj)
    if args.smoke:
        jobs, raw_path = [(42, 'S11')], os.path.join(OUT, 'smoke_raw.csv')
        if os.path.exists(raw_path):
            os.remove(raw_path)
    else:
        allj = [(seed, s) for seed in SEEDS for s in SUBJECTS]
        done = done_folds(glob.glob(os.path.join(OUT, 'raw_w*.csv')))
        if args.split_remaining:      # 收尾用：把所有還沒跑的 fold 重新平分（每個 fold 的種子固定，結果不受分法影響）
            jobs = [j for j in allj if j not in done]
            jobs = jobs[args.worker::args.n_workers]
            raw_path = os.path.join(OUT, f'raw_w{args.worker}r.csv')
        else:
            jobs = [j for i, j in enumerate(allj) if i % args.n_workers == args.worker]
            jobs = [j for j in jobs if j not in done]
            raw_path = os.path.join(OUT, f'raw_w{args.worker}.csv')

    load_mask = np.zeros(len(COLS) + 1, np.float32)
    load_mask[LOAD_IDX] = 1.0
    flag_vec = np.zeros(len(COLS) + 1, np.float32)
    flag_vec[FLAG_IDX] = 1.0
    load_mask, flag_vec = tf.constant(load_mask), tf.constant(flag_vec)

    def drop(x, y):
        X = x['ts_input']
        m = tf.cast(tf.random.uniform([tf.shape(X)[0], 1, 1]) < P_DROP, X.dtype)   # 每個視窗各自抽
        X = X * (1.0 - m * load_mask) + m * flag_vec
        return {'ts_input': X, 'static_input': x['static_input']}, y

    print(f'🔁 worker {args.worker}/{args.n_workers}：待跑 {len(jobs)} 個 fold（P_DROP = {P_DROP}）', flush=True)
    t0 = time.time()
    for i, (seed, hold) in enumerate(jobs, 1):
        ft = time.time()
        tf.keras.utils.set_random_seed(seed * 100 + subs.index(hold))
        tr = [d for s in subs if s != hold for d in by_subj[s]]
        big = pd.concat(tr, ignore_index=True)
        s_ts = StandardScaler().fit(big[COLS].values)
        s_st = StandardScaler().fit(big[STATIC_COLS].values)
        Xtr, Str, Ytr, _ = make_windows(tr, COLS, STATIC_COLS, LABEL_EMG_COLS, None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
        Xtr = add_flag(np.asarray(Xtr, np.float32), missing=False)
        with tf.device('/CPU:0'):
            ds = (tf.data.Dataset.from_tensor_slices(({'ts_input': Xtr, 'static_input': np.asarray(Str, np.float32)},
                                                      {'out_emg': np.asarray(Ytr, np.float32)}))
                  .shuffle(len(Xtr), seed=seed * 100 + subs.index(hold), reshuffle_each_iteration=True)
                  .batch(BATCH_SIZE).map(drop, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE))
        model = build_conv1d(len(COLS) + 1, len(STATIC_COLS))
        model.fit(ds, epochs=1 if args.smoke else EPOCHS, verbose=0)
        kgs = info[info.Subject == hold].kg.tolist()
        rows = []
        for sc in SCENARIOS:
            test = perturb(by_subj[hold], kgs, 'no_bar' if sc == 'no_bar' else 'true', None, None)
            Xv, Sv, Yv, _ = make_windows(test, COLS, STATIC_COLS, LABEL_EMG_COLS, None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
            Xv = add_flag(np.asarray(Xv, np.float32), missing=(sc == 'missing'))
            P = model.predict({'ts_input': Xv, 'static_input': Sv}, verbose=0, batch_size=1024)
            m = full_metrics(Yv, np.asarray(P[0] if isinstance(P, list) else P, np.float32))
            rows.append({'scenario': sc, 'seed': seed, 'Subject': hold, 'r_main': m['r'][0], 'r_syn': m['r'][1],
                         'nrmse_main': m['nrmse'][0], 'nrmse_syn': m['nrmse'][1]})
        tf.keras.backend.clear_session()
        pd.DataFrame(rows).to_csv(raw_path, mode='a', header=not os.path.exists(raw_path),
                                  index=False, encoding='utf-8-sig')
        r = {x['scenario']: x['r_main'] for x in rows}
        print(f'[{i}/{len(jobs)}] seed={seed} test={hold} r_main 有填={r["true"]:.3f} 沒填={r["missing"]:.3f} '
              f'忘了槓鈴={r["no_bar"]:.3f}  ({(time.time() - ft) / 60:.1f} 分，累計 {(time.time() - t0) / 60:.0f} 分)', flush=True)
    print('✅ worker 完成', flush=True)


def summarize():
    fs = sorted(glob.glob(os.path.join(OUT, 'raw_w*.csv')))
    if not fs:
        print('尚無結果')
        return
    d = pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates(['scenario', 'seed', 'Subject'], keep='last')
    d['scenario'] = 'drop_' + d.scenario
    folds = set(map(tuple, d[['seed', 'Subject']].values))
    sens = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RESULTS_DIR, 'load_sens', 'raw_w*.csv'))])
    sens = sens[sens.scenario.isin(['true', 'no_bar', 'mean_load', 'no_load'])]
    sens['scenario'] = 'load_' + sens.scenario
    ref = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RESULTS_DIR, 'load20', 'raw_w*.csv'))])
    ref = ref[(ref.arm == 'conv8') & ref.seed.isin(SEEDS)].drop_duplicates(['seed', 'Subject'], keep='last')
    ref = ref.assign(scenario='conv8_skeleton_only')
    allr = pd.concat([d, sens[['scenario', 'seed', 'Subject'] + METRICS], ref[['scenario', 'seed', 'Subject'] + METRICS]],
                     ignore_index=True)
    allr = allr[[(a, b) in folds for a, b in allr[['seed', 'Subject']].values]]
    print(f'完成 fold：{len(folds)} / {len(SEEDS) * len(SUBJECTS)}')
    subj = allr.groupby(['scenario', 'Subject'])[METRICS].mean()
    summ = subj.groupby('scenario')[METRICS].mean()
    summ['worst_subject_r'] = subj.r_main.groupby('scenario').min()
    summ.to_csv(os.path.join(OUT, 'summary.csv'), encoding='utf-8-sig')
    pairs = [('drop_true', 'load_true', '有填：dropout 模型 vs 原本加負重模型'),
             ('drop_missing', 'conv8_skeleton_only', '沒填：dropout 模型 vs 只用骨架模型'),
             ('drop_missing', 'load_mean_load', '沒填：dropout 模型 vs 原模型補平均'),
             ('drop_true', 'conv8_skeleton_only', '有填：dropout 模型 vs 只用骨架模型'),
             ('drop_no_bar', 'load_no_bar', '忘了槓鈴：dropout 模型 vs 原模型')]
    tests = []
    for a, b, lab in pairs:
        if a not in subj.index.get_level_values(0) or b not in subj.index.get_level_values(0):
            continue
        x, y = subj.loc[a].r_main, subj.loc[b].r_main.reindex(subj.loc[a].index)
        dd = x - y
        tests.append({'比較': lab, 'a': a, 'b': b, 'r_a': x.mean(), 'r_b': y.mean(), 'diff': dd.mean(),
                      'p': stats.ttest_rel(x, y).pvalue if len(x) > 1 else np.nan,
                      'n_better': int((dd > 0).sum()), 'n': len(x), 'min_diff': dd.min()})
    tests = pd.DataFrame(tests)
    tests.to_csv(os.path.join(OUT, 'tests.csv'), index=False, encoding='utf-8-sig')
    pd.set_option('display.width', 250)
    print('\n' + summ.round(4).to_string())
    print('\n配對比較（每人先取種子平均，n = 人數）\n' + tests.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--worker', type=int, default=0)
    ap.add_argument('--n-workers', type=int, default=1)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--split-remaining', action='store_true')
    ap.add_argument('--summarize-only', action='store_true')
    a = ap.parse_args()
    if a.summarize_only:
        summarize()
    else:
        run(a)
        if not a.smoke:
            summarize()


if __name__ == '__main__':
    main()
