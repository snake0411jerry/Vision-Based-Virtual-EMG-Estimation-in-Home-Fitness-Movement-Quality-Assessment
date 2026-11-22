# English summary: Adds the known barbell load (%1RM, load / body weight) or quasi-static joint moments to Conv1D-8. Load: r 0.750 -> 0.793 (p = 0.004); moments: 0.778. Also defines FEATURES_8, LOAD_COLS and load_data().
# (Added Nov 2026; the detailed comments below are in Chinese.)
"""
load_moment_20.py — 規劃三：加入已知負重與關節力矩（20 人 LOSO 消融）
=========================================================================
學習曲線顯示多收人上限約 r ≈ 0.77；肌肉要出多少力取決於負重，影片看不到，但使用者知道槓鈴多重。

負重：每組 TRC 檔名開頭的數字（公斤），0 = 無負重，字母 a/b/c = 代償（無負重）。器材為史密斯機。
  1RM 推估：兩批收案都把最重一組（≈ 可連續做 10 下的極限）視為 70 % 1RM → 1RM = 最重 ÷ 0.7
  （不使用 subjects.csv 的 squat_1rm_kg，因舊 10 人那欄與此規則不一致）

關節力矩（準靜態、矢狀面、雙腳平均分擔，單位 N·m / kg 體重；沒有力板，所以不是 OpenSim 逆向動力學）：
  前後方向 = 每幀腳跟 → 大拇趾的水平方向（不依賴攝影機座標）
  節段質量 de Leva (1996)：頭頸軀幹手臂 HAT 0.6028、大腿 0.1416、小腿 0.0433（× 體重）
  HAT 質心 = midHip + 0.55·(Neck − midHip)；大腿質心距髖 0.4095；小腿質心距膝 0.4459
  槓鈴位置 = 左右肩中點
  M_hip  = g/2 · [m_HAT·(x_HAT − x_hip) + L·(x_bar − x_hip)]
  M_knee = −[上半身項改以膝為支點 + g·m_thigh·(x_thighCOM − x_knee)]（取負號：三個關節皆以伸肌需求為正）
  M_ankle 同理再加小腿
  （忽略 I·α 慣性項；深蹲速度慢，先以準靜態近似）

Arms（同一次執行、同 seed 配對）：
  conv8          Conv1D 8 維（基準，重跑以避免跨次執行波動）
  conv8_load     ＋ 負重 2 通道（% 1RM、負重 / 體重）
  conv8_moment   ＋ 負重 2 通道 ＋ 力矩 3 通道（髖、膝、踝）

用法：
  python experiments/load_moment_20.py --check                # 只用 CPU：力矩與肌電包絡的相關
  python experiments/load_moment_20.py --smoke                # 1 epoch
  python experiments/load_moment_20.py --worker k --n-workers 5
  python experiments/load_moment_20.py --summarize-only
輸出 results/load20/
"""
# --- 路徑設定：一律由 repo 根的 paths.py 集中管理，請勿改回絕對路徑 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import PIPELINE_DIR, RESULTS_DIR, DATASET_DIR  # noqa: E402
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

G = 9.81
J = {n: i for i, n in enumerate(['Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist',
                                 'midHip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'LBigToe',
                                 'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe', 'RHeel'])}
M_HAT, M_THIGH, M_SHANK = 0.6028, 0.1416, 0.0433
SEEDS = [42, 1, 2, 3, 4]
FEATURES_8 = ['Knee_Angle_norm', 'Ankle_Y_norm', 'Knee_Y_norm', 'Toe_Y_norm', 'Knee_X_norm',
              'Shoulder_Y_norm', 'Knee_Ankle_Ratio_norm', 'Trunk_Lean_Angle_norm']
LOAD_COLS = ['Load_pct1RM', 'Load_per_BW']
MOMENT_COLS = ['M_hip', 'M_knee', 'M_ankle']
ARMS = {'conv8': FEATURES_8,
        'conv8_load': FEATURES_8 + LOAD_COLS,
        'conv8_moment': FEATURES_8 + LOAD_COLS + MOMENT_COLS}
OUT = os.path.join(RESULTS_DIR, 'load20')
COMBINED_20 = os.path.join(DATASET_DIR, 'Combined_20')
SKEL_20 = os.path.join(DATASET_DIR, 'Skeleton_20')


# ---------------------------------------------------------------- 負重與力矩
def trc_kg(trc):
    m = re.match(r'^(\d+)', trc)
    return float(m.group(1)) if m else 0.0


def moments(coords, ref_height, body_kg, load_kg):
    """coords：(T, 20, 3)，以 midHip 為原點、除以 ref_height。回傳 (T, 3) 髖 / 膝 / 踝力矩（N·m / kg），左右平均。"""
    P = coords.astype(np.float64) * ref_height                       # 還原成公尺
    fwd = np.zeros((len(P), 3))
    for side in 'LR':
        fwd += P[:, J[f'{side}BigToe']] - P[:, J[f'{side}Heel']]
    fwd[:, 1] = 0.0                                                   # 只取水平分量（Y 為垂直）
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True) + 1e-9

    def x(p):                                                         # 前後座標
        return np.sum(p * fwd, axis=1)

    hat = P[:, J['midHip']] + 0.55 * (P[:, J['Neck']] - P[:, J['midHip']])
    bar = 0.5 * (P[:, J['LShoulder']] + P[:, J['RShoulder']])
    m_hat, m_th, m_sh = M_HAT * body_kg, M_THIGH * body_kg, M_SHANK * body_kg
    out = []
    for side in 'LR':
        hip, knee, ank = P[:, J[f'{side}Hip']], P[:, J[f'{side}Knee']], P[:, J[f'{side}Ankle']]
        thigh = hip + 0.4095 * (knee - hip)
        shank = knee + 0.4459 * (ank - knee)

        def upper(pivot):
            return 0.5 * G * (m_hat * (x(hat) - x(pivot)) + load_kg * (x(bar) - x(pivot)))

        m_hip = upper(hip)
        # 膝：重量線落在膝後方時需要伸膝力矩 → 取負號，讓三個關節都是「伸肌需求為正」
        m_knee = -(upper(knee) + G * m_th * (x(thigh) - x(knee)))
        m_ank = upper(ank) + G * m_th * (x(thigh) - x(ank)) + G * m_sh * (x(shank) - x(ank))
        out.append(np.stack([m_hip, m_knee, m_ank], 1))
    return (0.5 * (out[0] + out[1]) / body_kg).astype(np.float32)


def load_data():
    """與 fair_and_curve_20.load_combined 相同的檔案順序（排序後的路徑），每段再補上負重與力矩欄位。"""
    paths = sorted(glob.glob(os.path.join(COMBINED_20, '*_Combined_Features.csv')))
    meta = {}
    for f in glob.glob(os.path.join(SKEL_20, '*.npz')):
        z = np.load(f, allow_pickle=True)
        meta[(str(z['subject']), int(z['seg']))] = z
    top = {}
    for (s, _), z in meta.items():
        top[s] = max(top.get(s, 0.0), trc_kg(str(z['trc'])))
    by_subj, info = {}, []
    for p in paths:
        m = re.match(r'^(S\d+)_Seg_(\d+)_', os.path.basename(p))
        s, seg = m.group(1), int(m.group(2))
        df = pd.read_csv(p)
        z = meta[(s, seg)]
        body = float(z['static'][2]) * 100.0
        kg = trc_kg(str(z['trc']))
        assert len(z['coords']) == len(df), f'{s} Seg{seg} 幀數不符 {len(z["coords"])} vs {len(df)}'
        df['Load_pct1RM'] = kg / (top[s] / 0.7)
        df['Load_per_BW'] = kg / body
        mo = moments(z['coords'], float(z['ref_height']), body, kg)
        for i, c in enumerate(MOMENT_COLS):
            df[c] = mo[:, i]
        by_subj.setdefault(s, []).append(df)
        info.append({'Subject': s, 'seg': seg, 'trc': str(z['trc']), 'kg': kg, 'body_kg': body,
                     'pct1RM': kg / (top[s] / 0.7)})
    return by_subj, pd.DataFrame(info)


# ---------------------------------------------------------------- CPU 檢查：力矩與肌電包絡
def check():
    os.makedirs(OUT, exist_ok=True)
    by_subj, info = load_data()
    rows = []
    for s, dfs in by_subj.items():
        segs = info[info.Subject == s].reset_index(drop=True)
        for df, (_, r) in zip(dfs, segs.iterrows()):
            row = {'Subject': s, 'trc': r.trc, 'kg': r.kg, 'pct1RM': r.pct1RM,
                   'emg_main_mean': df.EMG_Main_MVC.mean(), 'emg_main_p90': df.EMG_Main_MVC.quantile(0.9)}
            for c in MOMENT_COLS:
                row[f'{c}_p90'] = df[c].quantile(0.9)
                row[f'r_{c}_vs_emg'] = np.corrcoef(df[c], df.EMG_Main_MVC)[0, 1]
            row['r_kneeangle_vs_emg'] = np.corrcoef(df['Knee_Angle_norm'], df.EMG_Main_MVC)[0, 1]
            rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, 'check_per_segment.csv'), index=False, encoding='utf-8-sig')
    pd.set_option('display.width', 250)
    print('▶ 組內（逐幀）與主動肌包絡的相關，20 人所有組平均：')
    print(d[[f'r_{c}_vs_emg' for c in MOMENT_COLS] + ['r_kneeangle_vs_emg']].mean().round(3).to_string())
    print('\n▶ 組間：同一人不同組，「組的負重 / 力矩高點」能否解釋「組的肌電高點」？（逐人 Spearman，再平均）')
    res = []
    for s, g in d.groupby('Subject'):
        if g.kg.nunique() < 3:
            continue
        res.append({'Subject': s,
                    'rho_kg': stats.spearmanr(g.kg, g.emg_main_p90)[0],
                    'rho_Mknee_p90': stats.spearmanr(g.M_knee_p90, g.emg_main_p90)[0],
                    'rho_Mhip_p90': stats.spearmanr(g.M_hip_p90, g.emg_main_p90)[0]})
    res = pd.DataFrame(res)
    print(res.round(2).to_string(index=False))
    print('平均：', res.drop(columns='Subject').mean().round(3).to_dict())
    loaded = d[d.kg > 0]
    print('\n▶ 力矩範圍（N·m/kg，各組 90 百分位）：')
    print(d.groupby(pd.cut(d.pct1RM, [-0.01, 0.01, 0.3, 0.5, 0.75]))[[f'{c}_p90' for c in MOMENT_COLS]]
          .median().round(2).to_string())
    print(f'\n有負重的組數 {len(loaded)}；負重範圍 {loaded.kg.min():.0f}–{loaded.kg.max():.0f} kg')


# ---------------------------------------------------------------- 訓練
def all_jobs():
    subs = [f'S{i:02d}' for i in range(1, 21)]
    return [(arm, seed, s) for seed in SEEDS for s in subs for arm in ARMS]


def worker_jobs(k, n):
    return [j for i, j in enumerate(all_jobs()) if i % n == k]


def run(args):
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    from eval_utils import make_windows, full_metrics
    from loso_train_and_save import (build_model as build_conv1d, WINDOW_SIZE, STEP_SIZE,
                                     EPOCHS, BATCH_SIZE, STATIC_COLS, LABEL_EMG_COLS)
    from loso_10v20_compare import on_cpu
    by_subj, _ = load_data()
    subs = sorted(by_subj)
    root = os.path.join(OUT, 'smoke' if args.smoke else 'preds')
    os.makedirs(root, exist_ok=True)
    jobs = [(a, 42, 'S11') for a in ARMS] if args.smoke else worker_jobs(args.worker, args.n_workers)
    jobs = [j for j in jobs if args.smoke or not os.path.exists(os.path.join(root, f'{j[0]}_s{j[1]}_{j[2]}.npz'))]
    print(f'🔁 worker {args.worker}/{args.n_workers}：待跑 {len(jobs)} 個 fold', flush=True)
    raw_path = os.path.join(OUT, 'smoke_raw.csv' if args.smoke else f'raw_w{args.worker}.csv')
    t0 = time.time()
    for i, (arm, seed, hold) in enumerate(jobs, 1):
        ft = time.time()
        tf.keras.utils.set_random_seed(seed * 100 + subs.index(hold))
        cols = ARMS[arm]
        tr = [d for s in subs if s != hold for d in by_subj[s]]
        big = pd.concat(tr, ignore_index=True)
        s_ts = StandardScaler().fit(big[cols].values)
        s_st = StandardScaler().fit(big[STATIC_COLS].values)
        Xtr, Str, Ytr, _ = make_windows(tr, cols, STATIC_COLS, LABEL_EMG_COLS, None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
        Xv, Sv, Yv, _ = make_windows(by_subj[hold], cols, STATIC_COLS, LABEL_EMG_COLS, None,
                                     WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
        model = build_conv1d(len(cols), len(STATIC_COLS))
        Xtr, Str, Ytr = on_cpu(Xtr, Str, Ytr)
        model.fit({'ts_input': Xtr, 'static_input': Str}, {'out_emg': Ytr},
                  epochs=1 if args.smoke else EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        P = model.predict({'ts_input': Xv, 'static_input': Sv}, verbose=0)
        P = np.asarray(P[0] if isinstance(P, list) else P, np.float32)
        npar = model.count_params()
        tf.keras.backend.clear_session()
        p = os.path.join(root, f'{arm}_s{seed}_{hold}.npz')
        np.savez_compressed(p[:-4] + '.tmp.npz', y=np.asarray(Yv, np.float32), pred=P)
        os.replace(p[:-4] + '.tmp.npz', p)
        m = full_metrics(Yv, P)
        pd.DataFrame([{'arm': arm, 'seed': seed, 'Subject': hold, 'r_main': m['r'][0], 'r_syn': m['r'][1],
                       'nrmse_main': m['nrmse'][0], 'nrmse_syn': m['nrmse'][1], 'n_params': npar,
                       'minutes': (time.time() - ft) / 60}]).to_csv(
            raw_path, mode='a', header=not os.path.exists(raw_path), index=False, encoding='utf-8-sig')
        print(f'[{i}/{len(jobs)}] {arm} seed={seed} test={hold} r_main={m["r"][0]:.3f}'
              f'  ({(time.time() - ft) / 60:.1f} 分，累計 {(time.time() - t0) / 60:.0f} 分)', flush=True)
    print('✅ worker 完成', flush=True)


def summarize():
    fs = sorted(glob.glob(os.path.join(OUT, 'raw_w*.csv')))
    if not fs:
        print('尚無結果')
        return
    d = pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates(['arm', 'seed', 'Subject'], keep='last')
    print(d.groupby('arm').size().to_string())
    per_seed = d.groupby(['arm', 'seed'])[['r_main', 'r_syn', 'nrmse_main', 'nrmse_syn']].mean().reset_index()
    summ = per_seed.groupby('arm')[['r_main', 'r_syn', 'nrmse_main', 'nrmse_syn']].agg(['mean', 'std'])
    # 每位受試者先取種子平均（只用各 arm 都有的 seed），作為 n = 20 的配對檢定與最差受試者
    common = set.intersection(*[set(g.seed) for _, g in d.groupby('arm')])
    per_subj = d[d.seed.isin(common)].groupby(['arm', 'Subject'])[['r_main', 'nrmse_main', 'r_syn']].mean()
    worst = per_subj.r_main.groupby('arm').agg(['min', 'idxmin'])
    summ['worst_subject_r'] = worst['min']
    summ['worst_subject'] = worst['idxmin'].map(lambda t: t[1])
    tests = []
    for a, b in [('conv8', 'conv8_load'), ('conv8_load', 'conv8_moment'), ('conv8', 'conv8_moment')]:
        for metric in ['r_main', 'nrmse_main', 'r_syn']:
            x = per_seed[per_seed.arm == a].set_index('seed')[metric]
            y = per_seed[per_seed.arm == b].set_index('seed')[metric]
            k = x.index.intersection(y.index)
            if len(k) < 2:
                continue
            row = {'comparison': f'{a} → {b}', 'metric': metric, 'n_seeds': len(k),
                   'mean_a': x[k].mean(), 'mean_b': y[k].mean(), 'diff': (y[k] - x[k]).mean(),
                   'p': stats.ttest_rel(y[k], x[k]).pvalue}
            if a in per_subj.index.get_level_values(0) and b in per_subj.index.get_level_values(0):
                xs, ys = per_subj.loc[a, metric], per_subj.loc[b, metric]
                ks = xs.index.intersection(ys.index)
                better = (ys[ks] < xs[ks]) if metric.startswith('nrmse') else (ys[ks] > xs[ks])
                row.update({'n_subjects': len(ks), 'p_subjects': stats.ttest_rel(ys[ks], xs[ks]).pvalue,
                            'n_better': int(better.sum())})
            tests.append(row)
    tests = pd.DataFrame(tests)
    summ.to_csv(os.path.join(OUT, 'load20_summary.csv'), encoding='utf-8-sig')
    tests.to_csv(os.path.join(OUT, 'load20_tests.csv'), index=False, encoding='utf-8-sig')
    pd.set_option('display.width', 250)
    print('\n' + summ.round(4).to_string())
    print('\n' + tests.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--worker', type=int, default=0)
    ap.add_argument('--n-workers', type=int, default=1)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--summarize-only', action='store_true')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.check:
        check()
    elif args.summarize_only:
        summarize()
    else:
        run(args)


if __name__ == '__main__':
    main()
