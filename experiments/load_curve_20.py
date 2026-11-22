# English summary: Learning curve of the load model, same splits and seeds as fair_and_curve_20.py. Beats the skeleton-only model at every N; N = 8 already exceeds its ceiling.
# (Added Nov 2026; the detailed comments below are in Chinese.)
"""
load_curve_20.py — 加負重後的學習曲線（規劃三步驟 3：上限是否提高）
=====================================================================
與 fair_and_curve_20.py 的學習曲線完全相同的切法：10 個 repeat，每個 repeat 固定留 4 人測試（舊、新各 2），
其餘 16 人隨機排序取前 N 人訓練（N = 2 / 4 / 8 / 12 / 16，巢狀子集），隨機種子 100 + repeat。
只換模型輸入：conv8_load = 8 維骨架特徵 + 負重兩欄（%1RM、負重 / 體重，同 load_moment_20.py）。
只用骨架的 conv8 直接沿用 results/curve20_raw_w*.csv（同一切法、同一種子），逐 (repeat, N) 配對比較。

用法：
  python experiments/load_curve_20.py --smoke                    # repeat 0、N = 2、1 epoch
  python experiments/load_curve_20.py --worker k --n-workers 5
  python experiments/load_curve_20.py --summarize-only
輸出 results/load_curve/：raw_w*.csv、summary.csv、fit.csv、paired.csv
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

CURVE_REPEATS = list(range(10))
CURVE_N = [2, 4, 8, 12, 16]
COLS = FEATURES_8 + LOAD_COLS
ARM = 'conv8_load'
OUT = os.path.join(RESULTS_DIR, 'load_curve')
METRICS = ['r_main', 'r_syn', 'nrmse_main', 'nrmse_syn']


def keep_awake():
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass


def read_raw(pattern):
    fs = sorted(glob.glob(pattern))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def run(args):
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    from eval_utils import make_windows, full_metrics
    from loso_train_and_save import (build_model as build_conv1d, WINDOW_SIZE, STEP_SIZE,
                                     EPOCHS, BATCH_SIZE, STATIC_COLS, LABEL_EMG_COLS)
    from loso_10v20_compare import on_cpu
    from fair_and_curve_20 import curve_split
    keep_awake()
    os.makedirs(OUT, exist_ok=True)
    by_subj, _ = load_data()
    if args.smoke:
        jobs, raw_path = [(0, 2)], os.path.join(OUT, 'smoke_raw.csv')
        if os.path.exists(raw_path):
            os.remove(raw_path)
    else:
        allj = [(rep, n) for rep in CURVE_REPEATS for n in CURVE_N]
        jobs = [j for j in allj if j[0] % args.n_workers == args.worker]     # 與 curve20 相同：依 repeat 分
        done = read_raw(os.path.join(OUT, 'raw_w*.csv'))
        if len(done):
            have = set(map(tuple, done[['repeat', 'N']].drop_duplicates().values))
            jobs = [j for j in jobs if j not in have]
        raw_path = os.path.join(OUT, f'raw_w{args.worker}.csv')
    print(f'🔁 worker {args.worker}/{args.n_workers}：待跑 {len(jobs)} 個工作', flush=True)
    t0 = time.time()
    for i, (rep, n) in enumerate(jobs, 1):
        ft = time.time()
        test, pool = curve_split(rep)
        train = pool[:n]
        tf.keras.utils.set_random_seed(100 + rep)
        tr = [d for s in train for d in by_subj[s]]
        big = pd.concat(tr, ignore_index=True)
        s_ts = StandardScaler().fit(big[COLS].values)
        s_st = StandardScaler().fit(big[STATIC_COLS].values)
        Xtr, Str, Ytr, _ = make_windows(tr, COLS, STATIC_COLS, LABEL_EMG_COLS, None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
        model = build_conv1d(len(COLS), len(STATIC_COLS))
        Xtr, Str, Ytr = on_cpu(Xtr, Str, Ytr)
        model.fit({'ts_input': Xtr, 'static_input': Str}, {'out_emg': Ytr},
                  epochs=1 if args.smoke else EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        rows = []
        for s in test:
            Xv, Sv, Yv, _ = make_windows(by_subj[s], COLS, STATIC_COLS, LABEL_EMG_COLS, None,
                                         WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
            P = model.predict({'ts_input': Xv, 'static_input': Sv}, verbose=0, batch_size=1024)
            m = full_metrics(Yv, np.asarray(P[0] if isinstance(P, list) else P, np.float32))
            rows.append({'arm': ARM, 'repeat': rep, 'N': n, 'Subject': s, 'train': ' '.join(train),
                         'r_main': m['r'][0], 'r_syn': m['r'][1], 'nrmse_main': m['nrmse'][0], 'nrmse_syn': m['nrmse'][1]})
        tf.keras.backend.clear_session()
        pd.DataFrame(rows).to_csv(raw_path, mode='a', header=not os.path.exists(raw_path),
                                  index=False, encoding='utf-8-sig')
        r = np.mean([x['r_main'] for x in rows])
        print(f'[{i}/{len(jobs)}] repeat={rep} N={n} r_main={r:.3f}'
              f'  ({(time.time() - ft) / 60:.1f} 分，累計 {(time.time() - t0) / 60:.0f} 分)', flush=True)
    print('✅ worker 完成', flush=True)


def summarize():
    from scipy.optimize import curve_fit
    d = read_raw(os.path.join(OUT, 'raw_w*.csv'))
    if not len(d):
        print('尚無結果')
        return
    ref = read_raw(os.path.join(RESULTS_DIR, 'curve20_raw_w*.csv'))
    ref = ref[ref.arm == 'conv8']
    d = pd.concat([ref[d.columns.intersection(ref.columns)], d], ignore_index=True)
    rep = d.groupby(['arm', 'N', 'repeat'])[METRICS].mean().reset_index()
    print(f'完成 (repeat, N)：{len(rep[rep.arm == ARM])} / {len(CURVE_REPEATS) * len(CURVE_N)}')
    summ = rep.groupby(['arm', 'N'])[METRICS].agg(['mean', 'std', 'count'])
    summ.columns = ['_'.join(c) for c in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(os.path.join(OUT, 'summary.csv'), index=False, encoding='utf-8-sig')

    f = lambda n, rinf, b, c: rinf - b * np.power(n, -c)          # 與 curve20 相同的飽和冪次曲線
    fits = []
    for arm, g in rep.groupby('arm'):
        try:
            p, cov = curve_fit(f, g.N.values, g.r_main.values, p0=[0.8, 0.3, 0.5],
                               bounds=([0, 0, 0.05], [1, 5, 3]), maxfev=20000)
            se = np.sqrt(np.diag(cov))
            fits.append(dict(arm=arm, r_inf=p[0], r_inf_se=se[0], b=p[1], c=p[2],
                             **{f'r_at_{n}': f(n, *p) for n in [19, 30, 50, 100]}))
        except Exception as e:
            fits.append(dict(arm=arm, error=str(e)))
    fits = pd.DataFrame(fits)
    fits.to_csv(os.path.join(OUT, 'fit.csv'), index=False, encoding='utf-8-sig')

    piv = rep.pivot_table(index=['N', 'repeat'], columns='arm', values='r_main').dropna()
    paired = []
    for n, g in piv.groupby(level='N'):
        dd = g[ARM] - g['conv8']
        paired.append({'N': n, 'conv8': g['conv8'].mean(), ARM: g[ARM].mean(), 'diff': dd.mean(),
                       'p': stats.ttest_rel(g[ARM], g['conv8']).pvalue if len(g) > 1 else np.nan,
                       'n_better': int((dd > 0).sum()), 'n_repeats': len(g)})
    paired = pd.DataFrame(paired)
    paired.to_csv(os.path.join(OUT, 'paired.csv'), index=False, encoding='utf-8-sig')
    pd.set_option('display.width', 250)
    print('\n' + summ[['arm', 'N', 'r_main_mean', 'r_main_std', 'r_main_count', 'nrmse_main_mean']].round(4).to_string(index=False))
    print('\n飽和曲線 r(N) = r_inf − b·N^(−c)\n' + fits.round(4).to_string(index=False))
    print('\n同一 (repeat, N) 配對：conv8_load − conv8（n = repeat 數）\n' + paired.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--worker', type=int, default=0)
    ap.add_argument('--n-workers', type=int, default=1)
    ap.add_argument('--smoke', action='store_true')
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
