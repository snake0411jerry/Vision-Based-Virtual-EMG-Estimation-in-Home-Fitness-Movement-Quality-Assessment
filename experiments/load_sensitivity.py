# English summary: What if the load is entered wrong or not at all? 1RM error of +-30% barely matters; a missing load drops the load model to r 0.716, below the skeleton-only model.
# (Added Nov 2026; the detailed comments below are in Chinese.)
"""
load_sensitivity.py — 負重輸錯時的敏感度（20 人 LOSO，conv8_load）
=================================================================
模型照常用「正確負重」訓練；測試時只把留出受試者輸入的負重改錯，骨架與肌電標籤完全不變，
看主動肌 r / nRMSE 掉多少。這就是部署時使用者輸錯數字的情況。

負重輸入（同 load_moment_20.py）：Load_pct1RM = kg / 1RM，Load_per_BW = kg / 體重。
情境：
  true            正確負重（應重現 load20 的 conv8_load）
  1rm_x{k}        使用者估的 1RM = 真值 × k（k = 0.7 … 1.3）→ 只有 %1RM 變成 ÷ k
  set_noise{a}    每一組的公斤數各自 × U(1 − a, 1 + a)（a = 10 %、20 %；5 次抽樣取平均）
  no_bar          忘了算 20 kg 槓鈴：有負重的組 kg − 20（下限 0）
  no_load         完全沒填、程式補 0：兩個負重欄位都 = 0（等同告訴模型「每組都是徒手」，是錯的資訊）
  mean_load       完全沒填、程式補訓練者平均：每組都一樣、不帶任何組間差異（「沒有資訊」，理論上應退回只用骨架）
參考：只用骨架的 conv8（load20 已有，種子 42 / 1 / 2）

工作 = (seed, 測試者)，3 × 20 = 60 個；每個 fold 約 11 分鐘（5 個 worker 同時跑）。中斷後重跑會跳過已完成的 fold。
每個 fold 的隨機種子 = seed × 100 + 測試者序號，與 load_moment_20.py 相同。

用法：
  python experiments/load_sensitivity.py --smoke                   # S11、1 epoch，檢查流程
  python experiments/load_sensitivity.py --worker k --n-workers 5
  python experiments/load_sensitivity.py --summarize-only
輸出 results/load_sens/：raw_w*.csv、sens_summary.csv、sens_tests.csv
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

SEEDS = [42, 1, 2]
SUBJECTS = [f'S{i:02d}' for i in range(1, 21)]
COLS = FEATURES_8 + LOAD_COLS
OUT = os.path.join(RESULTS_DIR, 'load_sens')
BAR_KG = 20.0
N_DRAWS = 5
SCENARIOS = (['true'] + [f'1rm_x{k}' for k in ['0.7', '0.8', '0.9', '1.1', '1.2', '1.3']]
             + ['set_noise10', 'set_noise20', 'no_bar', 'no_load', 'mean_load'])
METRICS = ['r_main', 'r_syn', 'nrmse_main', 'nrmse_syn']


def keep_awake():
    """只要這個程序還在跑，就請 Windows 不要睡眠（程序結束自動解除）。"""
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass


def perturb(dfs, kgs, scenario, rng, train_mean):
    """dfs：測試者每段的 DataFrame；kgs：每段真實公斤數；train_mean：訓練者的負重欄位平均。回傳改過負重欄位的副本。"""
    out = []
    for df, kg in zip(dfs, kgs):
        d = df.copy()
        if scenario.startswith('1rm_x'):
            d['Load_pct1RM'] = d['Load_pct1RM'] / float(scenario[5:])
        elif scenario.startswith('set_noise'):
            f = rng.uniform(1 - int(scenario[9:]) / 100, 1 + int(scenario[9:]) / 100)
            d[LOAD_COLS] = d[LOAD_COLS] * f
        elif scenario == 'no_bar':
            f = max(kg - BAR_KG, 0.0) / kg if kg > 0 else 1.0
            d[LOAD_COLS] = d[LOAD_COLS] * f
        elif scenario == 'no_load':
            d[LOAD_COLS] = 0.0
        elif scenario == 'mean_load':
            for c in LOAD_COLS:
                d[c] = train_mean[c]
        out.append(d)
    return out


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
    from loso_10v20_compare import on_cpu
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
        jobs = [j for i, j in enumerate(allj) if i % args.n_workers == args.worker]
        done = done_folds(glob.glob(os.path.join(OUT, 'raw_w*.csv')))
        jobs = [j for j in jobs if j not in done]
        raw_path = os.path.join(OUT, f'raw_w{args.worker}.csv')
    print(f'🔁 worker {args.worker}/{args.n_workers}：待跑 {len(jobs)} 個 fold', flush=True)
    t0 = time.time()
    for i, (seed, hold) in enumerate(jobs, 1):
        ft = time.time()
        tf.keras.utils.set_random_seed(seed * 100 + subs.index(hold))
        tr = [d for s in subs if s != hold for d in by_subj[s]]
        big = pd.concat(tr, ignore_index=True)
        s_ts = StandardScaler().fit(big[COLS].values)
        s_st = StandardScaler().fit(big[STATIC_COLS].values)
        Xtr, Str, Ytr, _ = make_windows(tr, COLS, STATIC_COLS, LABEL_EMG_COLS, None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
        model = build_conv1d(len(COLS), len(STATIC_COLS))
        Xtr, Str, Ytr = on_cpu(Xtr, Str, Ytr)
        model.fit({'ts_input': Xtr, 'static_input': Str}, {'out_emg': Ytr},
                  epochs=1 if args.smoke else EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        kgs = info[info.Subject == hold].kg.tolist()
        assert len(kgs) == len(by_subj[hold])
        rows = []
        for sc in SCENARIOS:
            draws = N_DRAWS if sc.startswith('set_noise') else 1
            rng = np.random.default_rng(seed * 1000 + subs.index(hold))
            ms = []
            for _ in range(draws):
                test = perturb(by_subj[hold], kgs, sc, rng, big[LOAD_COLS].mean())
                Xv, Sv, Yv, _ = make_windows(test, COLS, STATIC_COLS, LABEL_EMG_COLS, None,
                                             WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
                P = model.predict({'ts_input': Xv, 'static_input': Sv}, verbose=0, batch_size=1024)
                P = np.asarray(P[0] if isinstance(P, list) else P, np.float32)
                m = full_metrics(Yv, P)
                ms.append([m['r'][0], m['r'][1], m['nrmse'][0], m['nrmse'][1]])
            ms = np.mean(ms, axis=0)
            rows.append({'scenario': sc, 'seed': seed, 'Subject': hold, **dict(zip(METRICS, ms))})
        tf.keras.backend.clear_session()
        pd.DataFrame(rows).to_csv(raw_path, mode='a', header=not os.path.exists(raw_path),
                                  index=False, encoding='utf-8-sig')
        r = {x['scenario']: x['r_main'] for x in rows}
        print(f'[{i}/{len(jobs)}] seed={seed} test={hold} r_main 正確={r["true"]:.3f} '
              f'1RM×0.8={r["1rm_x0.8"]:.3f} 1RM×1.2={r["1rm_x1.2"]:.3f} 補0={r["no_load"]:.3f} 補平均={r["mean_load"]:.3f}'
              f'  ({(time.time() - ft) / 60:.1f} 分，累計 {(time.time() - t0) / 60:.0f} 分)', flush=True)
    print('✅ worker 完成', flush=True)


def summarize():
    fs = sorted(glob.glob(os.path.join(OUT, 'raw_w*.csv')))
    if not fs:
        print('尚無結果')
        return
    d = pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates(['scenario', 'seed', 'Subject'], keep='last')
    ref = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RESULTS_DIR, 'load20', 'raw_w*.csv'))])
    ref = ref[(ref.arm == 'conv8') & ref.seed.isin(SEEDS)].drop_duplicates(['seed', 'Subject'], keep='last')
    ref = ref.assign(scenario='ref_conv8_no_load_model')[['scenario', 'seed', 'Subject'] + METRICS]
    folds = set(map(tuple, d[['seed', 'Subject']].values))
    ref = ref[[(a, b) in folds for a, b in ref[['seed', 'Subject']].values]]
    d = pd.concat([d, ref], ignore_index=True)
    print(f'完成 fold：{len(folds)} / {len(SEEDS) * len(SUBJECTS)}')
    subj = d.groupby(['scenario', 'Subject'])[METRICS].mean()
    order = SCENARIOS + ['ref_conv8_no_load_model']
    summ = subj.groupby('scenario')[METRICS].mean().reindex(order)
    summ['worst_subject_r'] = subj.r_main.groupby('scenario').min().reindex(order)
    base = subj.loc['true']
    tests = []
    for sc in order[1:]:
        if sc not in subj.index.get_level_values(0):
            continue
        x = subj.loc[sc].loc[base.index]
        dr = x.r_main - base.r_main
        tests.append({'scenario': sc, 'r_main': x.r_main.mean(), 'diff_vs_true': dr.mean(),
                      'p_vs_true': stats.ttest_rel(x.r_main, base.r_main).pvalue if len(x) > 1 else np.nan,
                      'n_worse': int((dr < 0).sum()), 'n': len(x), 'max_drop': dr.min(),
                      'nrmse_main': x.nrmse_main.mean(), 'nrmse_diff': (x.nrmse_main - base.nrmse_main).mean()})
    tests = pd.DataFrame(tests)
    summ.to_csv(os.path.join(OUT, 'sens_summary.csv'), encoding='utf-8-sig')
    tests.to_csv(os.path.join(OUT, 'sens_tests.csv'), index=False, encoding='utf-8-sig')
    pd.set_option('display.width', 250)
    print('\n' + summ.round(4).to_string())
    print('\n與「正確負重」比較（每人先取種子平均，n = 人數）\n' + tests.round(4).to_string(index=False))


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
