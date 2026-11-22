# English summary: Fair 20-subject comparison of Conv1D on 8 reliable features vs ST-GCN (tie: r 0.753 vs 0.755, p = 0.81), plus a learning curve over N = 2-16 training subjects (saturates near r 0.77).
# (Added Nov 2026; the detailed comments below are in Chinese.)
"""
fair_and_curve_20.py — 20 人資料的兩個補強實驗
=========================================================================
接續 loso_10v20_compare.py。模型與超參數完全相同（IC3MT slide 21）。

A. 模型對照（pool20 LOSO，seed 42 1 2 3 4，三個 arm 在同一次執行內跑，可逐 seed 配對）
   loso_10v20 的 Q3 裡 ST-GCN 多看了速度/加速度，架構與輸入資訊混在一起。

     conv8        Conv1D，8 維可靠特徵（MEDIAPIPE_EXPERIMENT.md §2 的 ★ 組；
                  §8：8 維不降平均，且把單鏡頭的逐人變異砍成 1/3，是部署用的特徵組）
     stgcn_pos    ST-GCN，20 關節 × 3 通道（只有座標）
     stgcn_posva  ST-GCN，20 關節 × 9 通道（座標＋速度＋加速度）

   stgcn_pos − conv8：架構（兩者都只有位置資訊）
   stgcn_posva − stgcn_pos：速度/加速度對 ST-GCN 的貢獻

B. 學習曲線（訓練人數 N = 2 / 4 / 8 / 12 / 16）
   每個 repeat（10 次，seed 100–109）：隨機留 4 人測試（舊、新各 2 人），
   其餘 16 人隨機排序，取前 N 人訓練（巢狀子集：2 ⊂ 4 ⊂ … ⊂ 16，降低變異）。
   模型：conv8 與 stgcn_posva。

平行執行：--worker k --n-workers 5，各 worker 依固定規則分配工作，各寫自己的 raw 檔。
可中斷續跑：每完成一個工作就寫檔，重跑時跳過已完成的。

輸出 results/：fair20_raw_*.csv、curve20_raw_*.csv（--summarize-only 產生 fair20_tests.csv、
curve20_summary.csv、curve20_fit.csv、curve20.png）
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
import tensorflow as tf
from scipy import stats
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)
sys.path.insert(0, HERE)

from eval_utils import subject_id_from_path, build_ts_cols, make_windows, full_metrics  # noqa: E402
from loso_train_and_save import (build_model as build_conv1d, WINDOW_SIZE, STEP_SIZE,   # noqa: E402
                                 EPOCHS, BATCH_SIZE, STATIC_COLS, LABEL_EMG_COLS)
from loso_stgcn_compare import skeleton_windows                                        # noqa: E402
from models_stgcn import build_stgcn                                                   # noqa: E402
from loso_10v20_compare import (COMBINED_20, OLD, NEW, FILTERS, load_skeleton, on_cpu,  # noqa: E402
                                paired)

FAIR_SEEDS = [42, 1, 2, 3, 4]
CURVE_REPEATS = list(range(10))
CURVE_N = [2, 4, 8, 12, 16]
# 8 維可靠組（MEDIAPIPE_EXPERIMENT.md §2 ★；與 loso_dualdomain.py --keep 用的同一組）
FEATURES_8 = ['Knee_Angle_norm', 'Ankle_Y_norm', 'Knee_Y_norm', 'Toe_Y_norm', 'Knee_X_norm',
              'Shoulder_Y_norm', 'Knee_Ankle_Ratio_norm', 'Trunk_Lean_Angle_norm']
ARM_FEATURE = {'conv8': FEATURES_8}
ARM_CHANNELS = {'stgcn_pos': 3, 'stgcn_posva': 9}
FAIR_ARMS = ['conv8', 'stgcn_pos', 'stgcn_posva']
CURVE_ARMS = ['conv8', 'stgcn_posva']
METRICS = ['r_main', 'r_syn', 'nrmse_main', 'nrmse_syn']


# ---------------------------------------------------------------- 資料與訓練
def load_combined():
    by_subj = {}
    for p in sorted(glob.glob(os.path.join(COMBINED_20, '*_Combined_Features.csv'))):
        by_subj.setdefault(subject_id_from_path(p), []).append(pd.read_csv(p))
    spatial = build_ts_cols(list(next(iter(by_subj.values()))[0].columns), 'spatial', False)
    miss = [c for c in FEATURES_8 if c not in spatial]
    assert not miss, f'Combined_20 缺少 8 維特徵：{miss}'
    return by_subj, {arm: [c for c in spatial if c in keep] for arm, keep in ARM_FEATURE.items()}


def fit_predict(arm, comb, ts_cols, skel, train, tests, epochs):
    if arm in ARM_FEATURE:
        cols = ts_cols[arm]
        tr = [d for s in train for d in comb[s]]
        big = pd.concat(tr, ignore_index=True)
        s_ts = StandardScaler().fit(big[cols].values)
        s_st = StandardScaler().fit(big[STATIC_COLS].values)
        Xtr, Str, Ytr, _ = make_windows(tr, cols, STATIC_COLS, LABEL_EMG_COLS, None,
                                        WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
        model = build_conv1d(len(cols), len(STATIC_COLS))
        test_windows = {s: make_windows(comb[s], cols, STATIC_COLS, LABEL_EMG_COLS, None,
                                        WINDOW_SIZE, STEP_SIZE, s_ts, s_st)[:3] for s in tests}
    else:
        C = ARM_CHANNELS[arm]
        cut = {s: [(X[..., :C], emg, st) for X, emg, st in skel[s]] for s in set(train) | set(tests)}
        tr = [seg for s in train for seg in cut[s]]
        V = tr[0][0].shape[1]
        sx = StandardScaler().fit(np.concatenate([X.reshape(len(X), V * C) for X, _, _ in tr]))
        ss = StandardScaler().fit(np.stack([st for _, _, st in tr]))
        Xtr, Str, Ytr = skeleton_windows(tr, sx, ss)
        model = build_stgcn(WINDOW_SIZE, C, len(STATIC_COLS), filters=FILTERS)
        test_windows = {s: skeleton_windows(cut[s], sx, ss) for s in tests}
    Xtr, Str, Ytr = on_cpu(Xtr, Str, Ytr)
    model.fit({'ts_input': Xtr, 'static_input': Str}, {'out_emg': Ytr},
              epochs=epochs, batch_size=BATCH_SIZE, verbose=0)
    res = {}
    for s, (Xv, Sv, Yv) in test_windows.items():
        pred = model.predict({'ts_input': Xv, 'static_input': Sv}, verbose=0)
        res[s] = full_metrics(Yv, pred[0] if isinstance(pred, list) else pred)
    npar = model.count_params()
    tf.keras.backend.clear_session()
    return res, npar


# ---------------------------------------------------------------- 工作清單
def curve_split(rep):
    """回傳 (test 4 人, 依序排列的 16 人訓練池)。舊、新各留 2 人測試。"""
    rng = np.random.default_rng(1000 + rep)
    test = list(rng.choice(OLD, 2, replace=False)) + list(rng.choice(NEW, 2, replace=False))
    pool = [s for s in OLD + NEW if s not in test]
    return sorted(test), list(rng.permutation(pool))


def all_jobs():
    """(實驗, arm, key) —— fair 的 key 是 seed；curve 的 key 是 (repeat, N)。"""
    jobs = []
    for rep in CURVE_REPEATS:
        for n in CURVE_N:
            for arm in CURVE_ARMS:
                jobs.append(('curve', arm, (rep, n)))
    for arm in FAIR_ARMS:
        for seed in FAIR_SEEDS:
            jobs.append(('fair', arm, seed))
    return jobs


def worker_jobs(k, n_workers):
    """curve 依 repeat 分、fair 依 seed 分；每個 worker 先跑 curve（快），再依 FAIR_ARMS 順序跑 fair。"""
    mine = []
    for exp, arm, key in all_jobs():
        idx = key[0] if exp == 'curve' else FAIR_SEEDS.index(key)
        if idx % n_workers == k:
            mine.append((exp, arm, key))
    return sorted(mine, key=lambda j: 0 if j[0] == 'curve' else 1 + FAIR_ARMS.index(j[1]))


def read_raw(prefix):
    fs = sorted(glob.glob(os.path.join(RESULTS_DIR, f'{prefix}_raw*.csv')))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def done_keys():
    fair, curve = read_raw('fair20'), read_raw('curve20')
    out = set()
    if len(fair):
        out |= {('fair', a, int(s)) for a, s in fair[['arm', 'seed']].drop_duplicates().values}
    if len(curve):
        out |= {('curve', a, (int(r), int(n))) for a, r, n in curve[['arm', 'repeat', 'N']].drop_duplicates().values}
    return out


# ---------------------------------------------------------------- 執行
def run(args):
    comb, ts_cols = load_combined()
    skel = load_skeleton()
    print(f'📂 {len(comb)} 人 / Conv1D 特徵 {len(ts_cols["conv8"])} 維：{ts_cols["conv8"]}', flush=True)
    done = done_keys()
    jobs = [j for j in worker_jobs(args.worker, args.n_workers) if j not in done]
    if args.smoke:
        jobs = [('curve', 'conv8', (0, 2)), ('curve', 'stgcn_pos', (0, 2))]
    print(f'🔁 worker {args.worker}/{args.n_workers}：待跑 {len(jobs)} 個工作', flush=True)
    # smoke 的檔名不以 fair20_raw / curve20_raw 開頭，不會混進續跑判斷與彙整
    paths = {e: os.path.join(RESULTS_DIR, f'smoke_{e}20.csv' if args.smoke else f'{e}20_raw_w{args.worker}.csv')
             for e in ['fair', 'curve']}
    rows = {e: (pd.read_csv(p).to_dict('records') if os.path.exists(p) else []) for e, p in paths.items()}
    epochs = 1 if args.smoke else EPOCHS
    t0 = time.time()

    for i, (exp, arm, key) in enumerate(jobs, 1):
        new = []
        if exp == 'curve':
            rep, n = key
            test, pool = curve_split(rep)
            tf.keras.utils.set_random_seed(100 + rep)
            res, npar = fit_predict(arm, comb, ts_cols, skel, pool[:n], test, epochs)
            for s, m in res.items():
                new.append({'arm': arm, 'repeat': rep, 'N': n, 'Subject': s, 'train': ' '.join(pool[:n]),
                            'r_main': m['r'][0], 'r_syn': m['r'][1],
                            'nrmse_main': m['nrmse'][0], 'nrmse_syn': m['nrmse'][1], 'n_params': npar})
        else:
            seed = key
            tf.keras.utils.set_random_seed(seed)
            for hold in OLD + NEW:
                ft = time.time()
                train = [s for s in OLD + NEW if s != hold]
                res, npar = fit_predict(arm, comb, ts_cols, skel, train, [hold], epochs)
                m = res[hold]
                new.append({'arm': arm, 'condition': 'pool20', 'seed': seed, 'Subject': hold,
                            'n_train_subjects': len(train), 'r_main': m['r'][0], 'r_syn': m['r'][1],
                            'nrmse_main': m['nrmse'][0], 'nrmse_syn': m['nrmse'][1], 'n_params': npar})
                print(f'    {arm} seed={seed} test={hold} r_main={m["r"][0]:.3f}  ({(time.time()-ft)/60:.1f} 分)', flush=True)
        rows[exp] += new
        pd.DataFrame(rows[exp]).to_csv(paths[exp], index=False, encoding='utf-8-sig')
        d = pd.DataFrame(new)
        el = (time.time() - t0) / 60
        print(f'[{i}/{len(jobs)}] {exp} {arm} {key}  主動肌 r={d.r_main.mean():.3f}  協同肌 r={d.r_syn.mean():.3f}'
              f'  （累計 {el:.0f} 分）', flush=True)
    print('✅ worker 完成', flush=True)


# ---------------------------------------------------------------- 彙整
def summarize_fair():
    df = read_raw('fair20').drop_duplicates(['arm', 'seed', 'Subject'], keep='last')
    df['group'] = np.where(df.Subject.isin(OLD), 'old', 'new')
    df = pd.concat([df, df.assign(group='all')], ignore_index=True)
    sm = df.groupby(['arm', 'group', 'seed'])[METRICS].mean()
    summ = sm.groupby(['arm', 'group']).agg(['mean', 'std'])
    summ.columns = ['_'.join(c) for c in summ.columns]
    summ.reset_index().to_csv(os.path.join(RESULTS_DIR, 'fair20_summary.csv'), index=False, encoding='utf-8-sig')

    comps = [('conv8', 'stgcn_pos', '架構（只有位置）：Conv1D 8 維 → ST-GCN 座標'),
             ('stgcn_pos', 'stgcn_posva', '輸入（ST-GCN）：3 → 9 通道'),
             ('conv8', 'stgcn_posva', '部署組 Conv1D 8 維 → ST-GCN 9 通道')]
    tests = []
    for group in ['all', 'old', 'new']:
        for metric in METRICS:
            for a, b, label in comps:
                if a in sm.index.get_level_values(0) and b in sm.index.get_level_values(0):
                    tests.append(dict(paired(sm.loc[(a, group)][metric], sm.loc[(b, group)][metric], label),
                                      group=group, metric=metric))
    tests = pd.DataFrame(tests)
    tests.to_csv(os.path.join(RESULTS_DIR, 'fair20_tests.csv'), index=False, encoding='utf-8-sig')
    return summ, tests


def summarize_curve():
    from scipy.optimize import curve_fit
    df = read_raw('curve20')
    rep_mean = df.groupby(['arm', 'N', 'repeat'])[METRICS].mean().reset_index()
    summ = rep_mean.groupby(['arm', 'N'])[METRICS].agg(['mean', 'std', 'count'])
    summ.columns = ['_'.join(c) for c in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(os.path.join(RESULTS_DIR, 'curve20_summary.csv'), index=False, encoding='utf-8-sig')

    # 飽和冪次曲線 r(N) = r_inf − b·N^(−c)，用逐 repeat 的點擬合
    fits = []
    for arm, d in rep_mean.groupby('arm'):
        f = lambda n, rinf, b, c: rinf - b * np.power(n, -c)
        try:
            p, cov = curve_fit(f, d.N.values, d.r_main.values, p0=[0.8, 0.3, 0.5],
                               bounds=([0, 0, 0.05], [1, 5, 3]), maxfev=20000)
            se = np.sqrt(np.diag(cov))
            fits.append(dict(arm=arm, r_inf=p[0], r_inf_se=se[0], b=p[1], c=p[2],
                             **{f'r_at_{n}': f(n, *p) for n in [19, 30, 50, 100]}))
        except Exception as e:  # 點太少或不收斂
            fits.append(dict(arm=arm, error=str(e)))
    fits = pd.DataFrame(fits)
    fits.to_csv(os.path.join(RESULTS_DIR, 'curve20_fit.csv'), index=False, encoding='utf-8-sig')

    # 相鄰 N 的配對（同一 repeat 的測試者與巢狀訓練集相同）
    steps = []
    for arm, d in rep_mean.groupby('arm'):
        piv = d.pivot(index='repeat', columns='N', values='r_main')
        ns = sorted(piv.columns)
        for a, b in zip(ns[:-1], ns[1:]):
            steps.append(dict(paired(piv[a].dropna(), piv[b].dropna(), f'N {a} → {b}'), arm=arm, metric='r_main'))
    steps = pd.DataFrame(steps)
    steps.to_csv(os.path.join(RESULTS_DIR, 'curve20_steps.csv'), index=False, encoding='utf-8-sig')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
        for (arm, d), color in zip(summ.groupby('arm'), ['#2B5F8A', '#C8692F']):
            ax.errorbar(d.N, d.r_main_mean, yerr=d.r_main_std, marker='o', capsize=3, color=color, label=arm)
            fr = fits[fits.arm == arm]
            if len(fr) and 'r_inf' in fr and pd.notna(fr.r_inf.iloc[0]):
                nn = np.linspace(2, 50, 200)
                ax.plot(nn, fr.r_inf.iloc[0] - fr.b.iloc[0] * nn ** (-fr.c.iloc[0]), '--', color=color, alpha=0.6)
        ax.set_xlabel('Training subjects (N)')
        ax.set_ylabel('Agonist r (held-out subjects)')
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, 'curve20.png'))
    except ImportError:
        pass
    return summ, fits, steps


def main():
    ap = argparse.ArgumentParser(description='20 人：公平對照 + 學習曲線')
    ap.add_argument('--worker', type=int, default=0)
    ap.add_argument('--n-workers', type=int, default=1)
    ap.add_argument('--smoke', action='store_true', help='1 epoch、3 個小工作，只檢查程式能跑')
    ap.add_argument('--summarize-only', action='store_true')
    args = ap.parse_args()
    if not args.summarize_only:
        run(args)
        if args.smoke:
            return
    pd.set_option('display.width', 250)
    if len(read_raw('fair20')):
        summ, tests = summarize_fair()
        print('\n' + summ.round(4).to_string())
        if len(tests): print('\n' + tests[tests.metric.isin(['r_main', 'nrmse_main'])][
            ['comparison', 'group', 'metric', 'n_seeds', 'mean_a', 'mean_b', 'diff', 't', 'p']].round(4).to_string(index=False))
    if len(read_raw('curve20')):
        summ, fits, steps = summarize_curve()
        print('\n' + summ.round(4).to_string(index=False))
        print('\n' + fits.round(4).to_string(index=False))
        print('\n' + steps[['comparison', 'arm', 'n_seeds', 'mean_a', 'mean_b', 'diff', 'p']].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
