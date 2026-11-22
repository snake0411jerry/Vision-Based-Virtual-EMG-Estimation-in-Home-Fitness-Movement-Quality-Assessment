# English summary: Equal-weight ensemble of Conv1D-8 and ST-GCN, 3 seeds x 20 LOSO folds (r 0.774, beats both models).
# (Added Nov 2026; the detailed comments below are in Chinese.)
"""
ensemble_20.py — 規劃一：Conv1D 8 維 × ST-GCN 9 通道 互補集成（20 人 LOSO）
=========================================================================
fair20 發現兩模型平均打平（0.753 vs 0.755），但失準的受試者不同。
這裡重跑兩模型的 LOSO，存下每位測試者的逐窗預測，再做融合：

  ens   = 0.5 · Conv1D + 0.5 · ST-GCN（主分析，等權，不需調參、不會洩漏）
  w 掃描 0.0 … 1.0（探索性；w 若用測試者挑選會偏樂觀，只報告不當主結果）

工作單位 = (arm, seed, 測試者)，共 2 × 3 × 20 = 120 個；每個工作做完立即存 npz，
中斷後重跑會跳過已存在的檔案（最多損失一個 fold）。
工作依成本（ST-GCN ≈ 2.5 × Conv1D）以 LPT 分給各 worker，讓大家差不多同時結束。
每個 fold 的隨機種子 = seed × 100 + 測試者序號，與執行順序、worker 數無關。

用法：
  python experiments/ensemble_20.py --smoke                 # 1 epoch，檢查兩模型的視窗與標籤是否逐窗一致
  python experiments/ensemble_20.py --worker k --n-workers 5
  python experiments/ensemble_20.py --summarize-only

輸出 results/ens20/：preds/*.npz、raw_w*.csv、ens20_summary.csv、ens20_tests.csv、ens20_per_subject.csv、ens20_wsweep.csv
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

from eval_utils import make_windows, full_metrics  # noqa: E402
from loso_train_and_save import (build_model as build_conv1d, WINDOW_SIZE, STEP_SIZE,   # noqa: E402
                                 EPOCHS, BATCH_SIZE, STATIC_COLS, LABEL_EMG_COLS)
from loso_stgcn_compare import skeleton_windows                                        # noqa: E402
from models_stgcn import build_stgcn                                                   # noqa: E402
from loso_10v20_compare import OLD, NEW, FILTERS, load_skeleton, on_cpu                 # noqa: E402
from fair_and_curve_20 import load_combined                                            # noqa: E402

SEEDS = [42, 1, 2]
ARMS = ['conv8', 'stgcn_posva']
SUBJECTS = OLD + NEW
COST = {'conv8': 1.0, 'stgcn_posva': 2.5}
OUT = os.path.join(RESULTS_DIR, 'ens20')
PRED = os.path.join(OUT, 'preds')


# ---------------------------------------------------------------- 訓練一個 fold，回傳測試者的 (Y, 預測)
def train_predict(arm, comb, ts_cols, skel, hold, epochs):
    train = [s for s in SUBJECTS if s != hold]
    if arm == 'conv8':
        cols = ts_cols['conv8']
        tr = [d for s in train for d in comb[s]]
        big = pd.concat(tr, ignore_index=True)
        s_ts = StandardScaler().fit(big[cols].values)
        s_st = StandardScaler().fit(big[STATIC_COLS].values)
        Xtr, Str, Ytr, _ = make_windows(tr, cols, STATIC_COLS, LABEL_EMG_COLS, None,
                                        WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
        Xv, Sv, Yv, _ = make_windows(comb[hold], cols, STATIC_COLS, LABEL_EMG_COLS, None,
                                     WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
        model = build_conv1d(len(cols), len(STATIC_COLS))
    else:
        C = 9
        tr = [seg for s in train for seg in skel[s]]
        V = tr[0][0].shape[1]
        sx = StandardScaler().fit(np.concatenate([X.reshape(len(X), V * C) for X, _, _ in tr]))
        ss = StandardScaler().fit(np.stack([st for _, _, st in tr]))
        Xtr, Str, Ytr = skeleton_windows(tr, sx, ss)
        Xv, Sv, Yv = skeleton_windows(skel[hold], sx, ss)
        model = build_stgcn(WINDOW_SIZE, C, len(STATIC_COLS), filters=FILTERS)
    Xtr, Str, Ytr = on_cpu(Xtr, Str, Ytr)
    model.fit({'ts_input': Xtr, 'static_input': Str}, {'out_emg': Ytr},
              epochs=epochs, batch_size=BATCH_SIZE, verbose=0)
    pred = model.predict({'ts_input': Xv, 'static_input': Sv}, verbose=0)
    pred = pred[0] if isinstance(pred, list) else pred
    tf.keras.backend.clear_session()
    return np.asarray(Yv, np.float32), np.asarray(pred, np.float32)


# ---------------------------------------------------------------- 工作分配
def all_jobs():
    return [(arm, seed, s) for seed in SEEDS for s in SUBJECTS for arm in ARMS]


def worker_jobs(k, n):
    """LPT：成本高的先排，每次給目前負擔最輕的 worker；所有 worker 算出同一份分配。"""
    load, assign = [0.0] * n, [[] for _ in range(n)]
    for j in sorted(all_jobs(), key=lambda j: (-COST[j[0]], SEEDS.index(j[1]), SUBJECTS.index(j[2]))):
        w = int(np.argmin(load))
        assign[w].append(j)
        load[w] += COST[j[0]]
    return assign[k]


def pred_path(arm, seed, subj, root=PRED):
    return os.path.join(root, f'{arm}_s{seed}_{subj}.npz')


# ---------------------------------------------------------------- 執行
def run(args):
    comb, ts_cols = load_combined()
    skel = load_skeleton()
    root = os.path.join(OUT, 'smoke') if args.smoke else PRED
    os.makedirs(root, exist_ok=True)
    if args.smoke:
        jobs = [(arm, 42, 'S11') for arm in ARMS]
    else:
        jobs = [j for j in worker_jobs(args.worker, args.n_workers) if not os.path.exists(pred_path(*j, root))]
    print(f'🔁 worker {args.worker}/{args.n_workers}：待跑 {len(jobs)} 個 fold', flush=True)
    raw_path = os.path.join(OUT, 'smoke_raw.csv' if args.smoke else f'raw_w{args.worker}.csv')
    t0 = time.time()
    for i, (arm, seed, hold) in enumerate(jobs, 1):
        ft = time.time()
        tf.keras.utils.set_random_seed(seed * 100 + SUBJECTS.index(hold))
        Y, P = train_predict(arm, comb, ts_cols, skel, hold, 1 if args.smoke else EPOCHS)
        p = pred_path(arm, seed, hold, root)
        tmp = p[:-4] + '.tmp.npz'
        np.savez_compressed(tmp, y=Y, pred=P)
        os.replace(tmp, p)
        m = full_metrics(Y, P)
        row = pd.DataFrame([{'arm': arm, 'seed': seed, 'Subject': hold, 'n_windows': len(Y),
                             'r_main': m['r'][0], 'r_syn': m['r'][1],
                             'nrmse_main': m['nrmse'][0], 'nrmse_syn': m['nrmse'][1],
                             'minutes': (time.time() - ft) / 60}])
        row.to_csv(raw_path, mode='a', header=not os.path.exists(raw_path), index=False, encoding='utf-8-sig')
        print(f'[{i}/{len(jobs)}] {arm} seed={seed} test={hold} 窗數={len(Y)} r_main={m["r"][0]:.3f}'
              f'  ({(time.time() - ft) / 60:.1f} 分，累計 {(time.time() - t0) / 60:.0f} 分)', flush=True)
    if args.smoke:
        a, b = [np.load(pred_path(arm, 42, 'S11', root)) for arm in ARMS]
        same_shape = a['y'].shape == b['y'].shape
        print(f'🔎 視窗數 conv8={a["y"].shape}  stgcn={b["y"].shape}  形狀一致={same_shape}')
        if same_shape:
            print(f'🔎 標籤最大差 = {np.abs(a["y"] - b["y"]).max():.6g}（需 ≈ 0 才能逐窗平均）')
    print('✅ worker 完成', flush=True)


# ---------------------------------------------------------------- 彙整
def metrics_row(y, p):
    m = full_metrics(y, p)
    return {'r_main': m['r'][0], 'r_syn': m['r'][1], 'nrmse_main': m['nrmse'][0], 'nrmse_syn': m['nrmse'][1]}


def align_windows(yc, yg):
    """回傳 yg 的索引，使 yg[idx] 與 yc 逐窗相同（標籤為連續值，幾乎唯一；重複者依序取用）"""
    from collections import defaultdict
    pool = defaultdict(list)
    for j, r in enumerate(yg):
        pool[tuple(np.round(r, 6))].append(j)
    idx = []
    for r in yc:
        k = tuple(np.round(r, 6))
        assert pool.get(k), '找不到對應視窗'
        idx.append(pool[k].pop(0))
    return np.array(idx)


def summarize():
    rows, sweep = [], []
    n_align = [0]
    for seed in SEEDS:
        for s in SUBJECTS:
            ps = [pred_path(a, seed, s) for a in ARMS]
            if not all(os.path.exists(p) for p in ps):
                continue
            c, g = [dict(np.load(p)) for p in ps]
            if c['y'].shape != g['y'].shape or not np.allclose(c['y'], g['y'], atol=1e-5):
                # 兩邊視窗數差 1–3 窗（段尾多一窗）且少數受試者段落順序不同 → 以標籤逐窗配對到 conv8 的視窗集合
                idx = align_windows(c['y'], g['y'])
                g = {'y': g['y'][idx], 'pred': g['pred'][idx]}
                n_align[0] += 1
            assert np.allclose(c['y'], g['y'], atol=1e-5), f'{seed} {s} 標籤不一致'
            y = c['y']
            for name, p in [('conv8', c['pred']), ('stgcn_posva', g['pred']),
                            ('ensemble', 0.5 * c['pred'] + 0.5 * g['pred'])]:
                rows.append({'model': name, 'seed': seed, 'Subject': s, **metrics_row(y, p)})
            for w in np.round(np.linspace(0, 1, 11), 1):
                sweep.append({'w_conv8': w, 'seed': seed, 'Subject': s,
                              **metrics_row(y, w * c['pred'] + (1 - w) * g['pred'])})
    df = pd.DataFrame(rows)
    if df.empty:
        print('尚無完成的配對')
        return
    n_pairs = df[df.model == 'ensemble'].shape[0]
    print(f'完成配對：{n_pairs} / {len(SEEDS) * len(SUBJECTS)}（其中 {n_align[0]} 組以標籤重新對齊視窗）')
    df.to_csv(os.path.join(OUT, 'ens20_per_subject.csv'), index=False, encoding='utf-8-sig')
    per_seed = df.groupby(['model', 'seed'])[['r_main', 'r_syn', 'nrmse_main', 'nrmse_syn']].mean().reset_index()
    summ = per_seed.groupby('model')[['r_main', 'r_syn', 'nrmse_main', 'nrmse_syn']].agg(['mean', 'std'])
    worst = df.groupby(['model', 'Subject']).r_main.mean().groupby('model').agg(['min', lambda x: x.nsmallest(3).mean()])
    worst.columns = ['worst_subject_r', 'worst3_mean_r']
    summ.columns = ['_'.join(c) for c in summ.columns]
    summ = summ.join(worst)
    summ.to_csv(os.path.join(OUT, 'ens20_summary.csv'), encoding='utf-8-sig')
    tests = []
    for other in ['conv8', 'stgcn_posva']:
        for metric in ['r_main', 'nrmse_main', 'r_syn']:
            a = df[df.model == other].set_index(['seed', 'Subject'])[metric]
            b = df[df.model == 'ensemble'].set_index(['seed', 'Subject'])[metric].loc[a.index]
            sa = per_seed[per_seed.model == other].set_index('seed')[metric]
            sb = per_seed[per_seed.model == 'ensemble'].set_index('seed')[metric].loc[sa.index]
            tests.append({'comparison': f'{other} → ensemble', 'metric': metric,
                          'mean_a': a.mean(), 'mean_b': b.mean(), 'diff': (b - a).mean(),
                          'p_subject_pairs': stats.ttest_rel(b, a).pvalue, 'n_pairs': len(a),
                          'p_seed_means': stats.ttest_rel(sb, sa).pvalue if len(sa) > 1 else np.nan,
                          'n_seeds': len(sa)})
    tests = pd.DataFrame(tests)
    tests.to_csv(os.path.join(OUT, 'ens20_tests.csv'), index=False, encoding='utf-8-sig')
    sw = pd.DataFrame(sweep).groupby('w_conv8')[['r_main', 'nrmse_main', 'r_syn']].mean()
    sw.to_csv(os.path.join(OUT, 'ens20_wsweep.csv'), encoding='utf-8-sig')
    pd.set_option('display.width', 250)
    print('\n' + summ.round(4).to_string())
    print('\n' + tests.round(4).to_string(index=False))
    print('\nw 掃描（探索性，w = Conv1D 權重）\n' + sw.round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--worker', type=int, default=0)
    ap.add_argument('--n-workers', type=int, default=1)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--summarize-only', action='store_true')
    args = ap.parse_args()
    os.makedirs(PRED, exist_ok=True)
    if args.summarize_only:
        summarize()
    else:
        run(args)


if __name__ == '__main__':
    main()
