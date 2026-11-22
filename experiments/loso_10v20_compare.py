# English summary: Does growing the dataset from 10 to 20 subjects help? LOSO on the old 10, the new 10 and the pooled 20 (Conv1D and ST-GCN).
# (Added Nov 2026; the detailed comments below are in Chinese.)
"""
loso_10v20_compare.py — 受試者人數 10 → 20 對同一模型的影響（LOSO 多 seed）
=========================================================================
沿用 IC3MT 2026 slide 21 的兩個模型與全部超參數（loso_stgcn_compare.py）：

  baseline      Conv1D ×2，16 維手工特徵（Dataset/Combined_20）
  stgcn_posva   ST-GCN，20 關節 × 9 通道（Dataset/Skeleton_20）

三種訓練/測試條件（同一個 job、同一組 seed，可配對比較）：

  pool10        S01–S10 做 LOSO，每折用另外 9 人訓練       ← 原 10 人實驗
  pool20        S01–S20 做 LOSO，每折用另外 19 人訓練      ← 20 人實驗
  old10_to_new  用 S01–S10 全部 10 人訓練一次，測 S11–S20 每一位（零樣本）

主要問題與檢定：
  Q1 同一批被留出的 S01–S10：訓練池 9 人 → 19 人，r / nRMSE 變多少？
     （pool20 vs pool10，限 S01–S10，逐 seed 配對 t 檢定）
  Q2 新收的 S11–S20：訓練池 10 人 → 19 人？（pool20 vs old10_to_new，限 S11–S20）
  Q3 20 人 LOSO 下 ST-GCN 是否仍勝過 Conv1D？（slide 21 的 20 人版）

S01–S10 在 Combined_20 / Skeleton_20 裡是原檔複製，與原 10 人實驗同一份資料。
⚠️ 跨執行的數字不可相減（GPU 非決定性）—— 比較一律取本 job 內的配對。

可中斷續跑：raw CSV 每完成一個 (arm, condition, seed) 就寫檔，重跑時自動跳過已完成的組合。

輸出：results/{prefix}_raw.csv、{prefix}_summary.csv、{prefix}_tests.csv
"""
# --- 路徑設定：一律由 repo 根的 paths.py 集中管理，請勿改回絕對路徑 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import DATASET_DIR, PIPELINE_DIR, RESULTS_DIR  # noqa: E402
# --- 路徑設定結束 ---

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)
sys.path.insert(0, HERE)

from eval_utils import subject_id_from_path, build_ts_cols, make_windows, full_metrics  # noqa: E402
from loso_train_and_save import (build_model as build_conv1d, FEATURE_MODE,           # noqa: E402
                                 WINDOW_SIZE, STEP_SIZE, EPOCHS, BATCH_SIZE, STATIC_COLS, LABEL_EMG_COLS)
from loso_stgcn_compare import skeleton_windows                                        # noqa: E402
from models_stgcn import build_stgcn                                                   # noqa: E402

COMBINED_20 = os.path.join(DATASET_DIR, 'Combined_20')
SKELETON_20 = os.path.join(DATASET_DIR, 'Skeleton_20')
OLD = [f'S{i:02d}' for i in range(1, 11)]
NEW = [f'S{i:02d}' for i in range(11, 21)]
CHANNELS = 9
FILTERS = (16, 32, 32)


# ---------------------------------------------------------------- 資料
def load_combined():
    paths = sorted(glob.glob(os.path.join(COMBINED_20, '*_Combined_Features.csv')))
    by_subj = {}
    for p in paths:
        by_subj.setdefault(subject_id_from_path(p), []).append(pd.read_csv(p))
    ts_cols = build_ts_cols(list(next(iter(by_subj.values()))[0].columns), FEATURE_MODE, False)
    return by_subj, ts_cols


def load_skeleton():
    out = {}
    for f in sorted(glob.glob(os.path.join(SKELETON_20, 'S*.npz'))):
        z = np.load(f, allow_pickle=True)
        X = np.concatenate([z['coords'], z['vel'], z['acc']], axis=2).astype(np.float32)
        out.setdefault(str(z['subject']), []).append((X, z['emg'].astype(np.float32),
                                                       z['static'].astype(np.float32)))
    return out


# ---------------------------------------------------------------- 一次訓練、多位受試者測試
def on_cpu(*arrays):
    """訓練資料先放在 CPU 端的 tensor 再交給 fit。

    直接給 numpy 時 Keras 會把整份陣列一次複製到 GPU；20 人的 ST-GCN 窗口約 2 GB，
    幾個 worker 平行時會 OOM（"Dst tensor is not initialized"）。
    改成 CPU tensor 後每個 batch 才搬上 GPU；切批與 shuffle 的方式與 numpy 輸入相同，運算仍在 GPU。
    """
    with tf.device('/CPU:0'):
        return [tf.constant(a) for a in arrays]


def fit_predict_baseline(data, ts_cols, train, tests):
    tr = [d for s in train for d in data[s]]
    big = pd.concat(tr, ignore_index=True)
    s_ts = StandardScaler().fit(big[ts_cols].values)
    s_st = StandardScaler().fit(big[STATIC_COLS].values)
    Xtr, Str, Ytr, _ = make_windows(tr, ts_cols, STATIC_COLS, LABEL_EMG_COLS, None,
                                    WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
    model = build_conv1d(len(ts_cols), len(STATIC_COLS))
    Xtr, Str, Ytr = on_cpu(Xtr, Str, Ytr)
    model.fit({'ts_input': Xtr, 'static_input': Str}, {'out_emg': Ytr},
              epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    res = {}
    for s in tests:
        Xv, Sv, Yv, _ = make_windows(data[s], ts_cols, STATIC_COLS, LABEL_EMG_COLS, None,
                                     WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
        res[s] = full_metrics(Yv, model.predict({'ts_input': Xv, 'static_input': Sv}, verbose=0))
    return res, model.count_params()


def fit_predict_stgcn(data, train, tests):
    tr = [seg for s in train for seg in data[s]]
    V = tr[0][0].shape[1]
    sx = StandardScaler().fit(np.concatenate([X.reshape(len(X), V * CHANNELS) for X, _, _ in tr]))
    ss = StandardScaler().fit(np.stack([st for _, _, st in tr]))
    Xtr, Str, Ytr = skeleton_windows(tr, sx, ss)
    model = build_stgcn(WINDOW_SIZE, CHANNELS, len(STATIC_COLS), filters=FILTERS)
    Xtr, Str, Ytr = on_cpu(Xtr, Str, Ytr)
    model.fit({'ts_input': Xtr, 'static_input': Str}, {'out_emg': Ytr},
              epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    res = {}
    for s in tests:
        Xv, Sv, Yv = skeleton_windows(data[s], sx, ss)
        pred = model.predict({'ts_input': Xv, 'static_input': Sv}, verbose=0)
        res[s] = full_metrics(Yv, pred[0] if isinstance(pred, list) else pred)
    return res, model.count_params()


def folds(condition):
    """回傳 [(train 受試者, test 受試者清單), ...]"""
    if condition == 'pool10':
        return [([s for s in OLD if s != h], [h]) for h in OLD]
    if condition == 'pool20':
        return [([s for s in OLD + NEW if s != h], [h]) for h in OLD + NEW]
    if condition == 'old10_to_new':
        return [(OLD, NEW)]
    raise ValueError(condition)


# ---------------------------------------------------------------- 統計
def paired(a, b, label):
    """a、b：以 seed 為 index 的 Series。回傳 b − a 的配對 t 檢定。"""
    idx = a.index.intersection(b.index)
    x, y = a.loc[idx], b.loc[idx]
    t, p = stats.ttest_rel(y, x) if len(idx) > 1 else (np.nan, np.nan)
    return dict(comparison=label, n_seeds=len(idx), mean_a=x.mean(), mean_b=y.mean(),
                diff=(y - x).mean(), sd_diff=(y - x).std(ddof=1) if len(idx) > 1 else np.nan, t=t, p=p)


def summarize(df, prefix):
    seed_mean = (df.groupby(['arm', 'condition', 'group', 'seed'])[['r_main', 'r_syn', 'nrmse_main', 'nrmse_syn']]
                 .mean().reset_index())
    summ = (seed_mean.groupby(['arm', 'condition', 'group'])[['r_main', 'r_syn', 'nrmse_main', 'nrmse_syn']]
            .agg(['mean', 'std']).reset_index())
    summ.columns = ['_'.join(c).strip('_') for c in summ.columns]
    summ.to_csv(os.path.join(RESULTS_DIR, f'{prefix}_summary.csv'), index=False, encoding='utf-8-sig')

    def sm(arm, cond, group, metric):
        d = seed_mean[(seed_mean.arm == arm) & (seed_mean.condition == cond) & (seed_mean.group == group)]
        return d.set_index('seed')[metric]

    tests = []
    for metric in ['r_main', 'r_syn', 'nrmse_main', 'nrmse_syn']:
        for arm in df.arm.unique():
            tests.append(dict(paired(sm(arm, 'pool10', 'old', metric), sm(arm, 'pool20', 'old', metric),
                                     'Q1 S01–S10：pool10(9人) → pool20(19人)'), arm=arm, metric=metric))
            tests.append(dict(paired(sm(arm, 'old10_to_new', 'new', metric), sm(arm, 'pool20', 'new', metric),
                                     'Q2 S11–S20：10人訓練 → 19人訓練'), arm=arm, metric=metric))
        for cond, group in [('pool10', 'old'), ('pool20', 'all')]:
            tests.append(dict(paired(sm('baseline', cond, group, metric), sm('stgcn_posva', cond, group, metric),
                                     f'Q3 {cond}（{group}）：Conv1D → ST-GCN'), arm='stgcn−baseline', metric=metric))
    tests = pd.DataFrame(tests)
    tests.to_csv(os.path.join(RESULTS_DIR, f'{prefix}_tests.csv'), index=False, encoding='utf-8-sig')
    return summ, tests


def add_groups(df):
    df = df.copy()
    df['group'] = np.where(df.Subject.isin(OLD), 'old', 'new')
    allg = df[df.condition == 'pool20'].copy()
    allg['group'] = 'all'
    return pd.concat([df, allg], ignore_index=True)


# ---------------------------------------------------------------- 主程式
def main():
    ap = argparse.ArgumentParser(description='10 人 vs 20 人 LOSO')
    ap.add_argument('--arms', nargs='*', default=['baseline', 'stgcn_posva'])
    ap.add_argument('--conditions', nargs='*', default=['pool10', 'pool20', 'old10_to_new'])
    ap.add_argument('--seeds', nargs='*', type=int, default=[42, 1, 2, 3, 4])
    ap.add_argument('--prefix', default='loso_10v20')
    ap.add_argument('--tag', default='',
                    help='平行執行時每個 worker 給不同 tag，各寫 {prefix}_raw_{tag}.csv；彙整時讀全部 {prefix}_raw*.csv')
    ap.add_argument('--summarize-only', action='store_true')
    args = ap.parse_args()

    def read_all_raw():
        fs = sorted(glob.glob(os.path.join(RESULTS_DIR, f'{args.prefix}_raw*.csv')))
        if not fs:
            return pd.DataFrame()
        return (pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
                .drop_duplicates(['arm', 'condition', 'seed', 'Subject'], keep='last'))

    prev = read_all_raw()
    done = set(map(tuple, prev[['arm', 'condition', 'seed']].values)) if len(prev) else set()
    raw_path = os.path.join(RESULTS_DIR, f'{args.prefix}_raw{"_" + args.tag if args.tag else ""}.csv')
    rows = pd.read_csv(raw_path).to_dict('records') if os.path.exists(raw_path) else []

    if not args.summarize_only:
        comb, ts_cols = load_combined()
        skel = load_skeleton()
        assert sorted(comb) == sorted(skel) == OLD + NEW, (sorted(comb), sorted(skel))
        print(f'📂 Combined {sum(map(len, comb.values()))} 段 / Skeleton {sum(map(len, skel.values()))} 段 / '
              f'{len(comb)} 人 / 手工特徵 {len(ts_cols)} 維', flush=True)
        jobs = [(a, c, s) for s in args.seeds for c in args.conditions for a in args.arms if (a, c, s) not in done]
        print(f'🔁 待跑 {len(jobs)} 組（已完成 {len(done)} 組會跳過）', flush=True)
        t0 = time.time()
        for n, (arm, cond, seed) in enumerate(jobs, 1):
            tf.keras.utils.set_random_seed(seed)
            new_rows = []
            for train, tests in folds(cond):
                ft = time.time()
                if arm == 'baseline':
                    res, npar = fit_predict_baseline(comb, ts_cols, train, tests)
                else:
                    res, npar = fit_predict_stgcn(skel, train, tests)
                tf.keras.backend.clear_session()
                for s, m in res.items():
                    new_rows.append({'arm': arm, 'condition': cond, 'seed': seed, 'Subject': s,
                                     'n_train_subjects': len(train),
                                     'r_main': m['r'][0], 'r_syn': m['r'][1],
                                     'nrmse_main': m['nrmse'][0], 'nrmse_syn': m['nrmse'][1], 'n_params': npar})
                print(f'    {arm} {cond} seed={seed} test={",".join(tests) if len(tests) < 3 else "S11–S20"} '
                      f'r_main={np.mean([res[s]["r"][0] for s in tests]):.3f}  ({(time.time()-ft)/60:.1f} 分)', flush=True)
            rows += new_rows
            pd.DataFrame(rows).to_csv(raw_path, index=False, encoding='utf-8-sig')
            d = pd.DataFrame(new_rows)
            el = (time.time() - t0) / n
            print(f'[{n}/{len(jobs)}] {arm} {cond} seed={seed}  主動肌 r={d.r_main.mean():.3f}  '
                  f'協同肌 r={d.r_syn.mean():.3f}  （平均 {el/60:.1f} 分/組，剩約 {el*(len(jobs)-n)/60:.0f} 分）', flush=True)

    df = add_groups(read_all_raw())
    summ, tests = summarize(df, args.prefix)
    pd.set_option('display.width', 250)
    print('\n' + summ.to_string(index=False))
    print('\n' + tests[tests.metric.isin(['r_main', 'nrmse_main'])][
        ['comparison', 'arm', 'metric', 'n_seeds', 'mean_a', 'mean_b', 'diff', 't', 'p']].round(4).to_string(index=False))
    print(f'\n✅ {args.prefix}_raw.csv / _summary.csv / _tests.csv')


if __name__ == '__main__':
    main()
