# English summary: Ensemble of the load model with ST-GCN, using saved predictions (+0.007, not significant).
# (Added Nov 2026; the detailed comments below are in Chinese.)
"""
ensemble_load.py — 規劃一 × 規劃三：加負重的 Conv1D 與 ST-GCN 集成（只用 CPU，不需訓練）
======================================================================================
直接讀已存好的逐窗預測：
  results/load20/preds/{conv8, conv8_load, conv8_moment}_s{seed}_{S}.npz（5 種子）
  results/ens20/preds/stgcn_posva_s{seed}_{S}.npz（3 種子）
兩邊共有的種子 42、1、2 × 20 人。ST-GCN 視窗以標籤逐窗對齊到 Conv1D（同 ensemble_20.align_windows）。

比較（全部用同一組種子、同一組視窗）：
  conv8、conv8_load、stgcn、ens_conv8_stgcn（重現規劃一）、ens_load_stgcn（主分析，等權）、ens_load_moment
  w 掃描（探索性）：w · conv8_load + (1 − w) · stgcn

用法：python experiments/ensemble_load.py
輸出 results/load20/：ensload_per_subject.csv、ensload_summary.csv、ensload_tests.csv、ensload_wsweep.csv
"""
# --- 路徑設定：一律由 repo 根的 paths.py 集中管理，請勿改回絕對路徑 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import PIPELINE_DIR, RESULTS_DIR  # noqa: E402
# --- 路徑設定結束 ---

import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, PIPELINE_DIR)
from eval_utils import full_metrics  # noqa: E402

SEEDS = [42, 1, 2]
SUBJECTS = [f'S{i:02d}' for i in range(1, 21)]
LOAD = os.path.join(RESULTS_DIR, 'load20')
ENS = os.path.join(RESULTS_DIR, 'ens20')
METRICS = ['r_main', 'r_syn', 'nrmse_main', 'nrmse_syn']


def metrics_row(y, p):
    m = full_metrics(y, p)
    return {'r_main': m['r'][0], 'r_syn': m['r'][1], 'nrmse_main': m['nrmse'][0], 'nrmse_syn': m['nrmse'][1]}


def align_windows(yc, yg):
    """回傳 yg 的索引，使 yg[idx] 與 yc 逐窗相同（標籤連續值幾乎唯一；重複者依序取用）"""
    pool = defaultdict(list)
    for j, r in enumerate(yg):
        pool[tuple(np.round(r, 6))].append(j)
    idx = []
    for r in yc:
        k = tuple(np.round(r, 6))
        assert pool.get(k), '找不到對應視窗'
        idx.append(pool[k].pop(0))
    return np.array(idx)


def load_pred(path):
    z = np.load(path)
    return z['y'], z['pred']


def main():
    rows, sweep, n_align = [], [], 0
    for seed in SEEDS:
        for s in SUBJECTS:
            y, p_c8 = load_pred(os.path.join(LOAD, 'preds', f'conv8_s{seed}_{s}.npz'))
            y2, p_ld = load_pred(os.path.join(LOAD, 'preds', f'conv8_load_s{seed}_{s}.npz'))
            y3, p_mo = load_pred(os.path.join(LOAD, 'preds', f'conv8_moment_s{seed}_{s}.npz'))
            assert np.allclose(y, y2, atol=1e-5) and np.allclose(y, y3, atol=1e-5), f'{seed} {s} load20 標籤不一致'
            yg, p_g = load_pred(os.path.join(ENS, 'preds', f'stgcn_posva_s{seed}_{s}.npz'))
            if yg.shape != y.shape or not np.allclose(yg, y, atol=1e-5):
                idx = align_windows(y, yg)
                yg, p_g = yg[idx], p_g[idx]
                n_align += 1
            assert np.allclose(yg, y, atol=1e-5), f'{seed} {s} ST-GCN 對齊失敗'
            for name, p in [('conv8', p_c8), ('conv8_load', p_ld), ('stgcn', p_g),
                            ('ens_conv8_stgcn', 0.5 * p_c8 + 0.5 * p_g),
                            ('ens_load_stgcn', 0.5 * p_ld + 0.5 * p_g),
                            ('ens_load_moment', 0.5 * p_ld + 0.5 * p_mo)]:
                rows.append({'model': name, 'seed': seed, 'Subject': s, **metrics_row(y, p)})
            for w in np.round(np.linspace(0, 1, 11), 1):
                sweep.append({'w_load': w, 'seed': seed, 'Subject': s, **metrics_row(y, w * p_ld + (1 - w) * p_g)})
    df = pd.DataFrame(rows)
    print(f'配對：{len(df) // 6} / {len(SEEDS) * len(SUBJECTS)}（其中 {n_align} 組 ST-GCN 以標籤重新對齊）')
    df.to_csv(os.path.join(LOAD, 'ensload_per_subject.csv'), index=False, encoding='utf-8-sig')

    per_seed = df.groupby(['model', 'seed'])[METRICS].mean().reset_index()
    summ = per_seed.groupby('model')[METRICS].agg(['mean', 'std'])
    summ.columns = ['_'.join(c) for c in summ.columns]
    subj = df.groupby(['model', 'Subject'])[METRICS].mean()          # 每人先取種子平均
    w = subj.r_main.groupby('model')
    summ['worst_subject_r'] = w.min()
    summ['worst_subject'] = w.idxmin().str[1]
    order = ['conv8', 'stgcn', 'ens_conv8_stgcn', 'conv8_load', 'ens_load_moment', 'ens_load_stgcn']
    summ = summ.loc[order]
    summ.to_csv(os.path.join(LOAD, 'ensload_summary.csv'), encoding='utf-8-sig')

    tests = []
    for a_name, b_name in [('conv8_load', 'ens_load_stgcn'), ('stgcn', 'ens_load_stgcn'),
                           ('ens_conv8_stgcn', 'ens_load_stgcn'), ('conv8', 'ens_load_stgcn'),
                           ('conv8_load', 'ens_load_moment')]:
        for metric in ['r_main', 'nrmse_main', 'r_syn']:
            a, b = subj.loc[a_name][metric], subj.loc[b_name][metric].loc[subj.loc[a_name].index]
            better = (b < a) if metric.startswith('nrmse') else (b > a)
            tests.append({'comparison': f'{a_name} → {b_name}', 'metric': metric,
                          'mean_a': a.mean(), 'mean_b': b.mean(), 'diff': (b - a).mean(),
                          'p_subjects': stats.ttest_rel(b, a).pvalue, 'n_subjects': len(a),
                          'n_better': int(better.sum())})
    tests = pd.DataFrame(tests)
    tests.to_csv(os.path.join(LOAD, 'ensload_tests.csv'), index=False, encoding='utf-8-sig')
    sw = pd.DataFrame(sweep).groupby('w_load')[['r_main', 'nrmse_main', 'r_syn']].mean()
    sw.to_csv(os.path.join(LOAD, 'ensload_wsweep.csv'), encoding='utf-8-sig')

    pd.set_option('display.width', 250)
    print('\n' + summ.round(4).to_string())
    print('\n' + tests.round(4).to_string(index=False))
    print('\nw 掃描（探索性，w = 加負重 Conv1D 的權重）\n' + sw.round(4).to_string())
    print('\n最差 5 人（主動肌 r，種子平均）')
    print(subj.r_main.unstack(0)[['conv8', 'conv8_load', 'stgcn', 'ens_load_stgcn']].nsmallest(5, 'ens_load_stgcn').round(3).to_string())


if __name__ == '__main__':
    main()
