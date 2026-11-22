# English summary: Fatigue feasibility: bar velocity loss from the skeleton is only ~4% in the heaviest sets (no set went to failure).
# (Added Nov 2026; the detailed comments below are in Chinese.)
"""
velocity_loss_check.py — 規劃二改版的可行性檢查：組內「速度損失」能否當疲勞標籤？
=========================================================================
阻力訓練文獻（Sánchez-Medina & González-Badillo 2011）以組內向心速度下降（velocity loss）作為
神經肌肉疲勞指標；它直接來自動作本身，影片就量得到，不需要肌電。

槓鈴高度 ≈ 左右肩中點相對左右踝中點的垂直距離（踝在地上不動；史密斯機槓鈴架在上背）。
每一下：高度最低點 → 下一個最高點為向心期，取平均向心速度（m/s）。
速度損失 = 1 − 最後一下 / 前兩下中較快者。
輸出 results/fatigue/velocity_per_segment.csv，並與 RMS 斜率（mdf_per_segment.csv）合併看相關。
"""
# --- 路徑設定：一律由 repo 根的 paths.py 集中管理，請勿改回絕對路徑 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import RESULTS_DIR, DATASET_DIR  # noqa: E402
# --- 路徑設定結束 ---

import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, sosfiltfilt, find_peaks

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

FPS = 60.0
SOS = butter(2, 3.0 / (FPS / 2), output='sos')
LSH, RSH, LANK, RANK = 4, 1, 13, 10
OUT = os.path.join(RESULTS_DIR, 'fatigue')


def rep_velocities(z):
    c = z['coords'].astype(float) * float(z['ref_height'])
    h = sosfiltfilt(SOS, 0.5 * (c[:, LSH, 1] + c[:, RSH, 1]) - 0.5 * (c[:, LANK, 1] + c[:, RANK, 1]))
    rng = np.percentile(h, 95) - np.percentile(h, 5)
    lows, _ = find_peaks(-h, distance=int(1.2 * FPS), prominence=0.4 * rng)
    highs, _ = find_peaks(h, distance=int(1.2 * FPS), prominence=0.4 * rng)
    v = []
    for lo in lows:
        nxt = highs[highs > lo]
        if len(nxt) == 0:
            continue
        hi = nxt[0]
        if hi - lo < 10:
            continue
        v.append((h[hi] - h[lo]) / ((hi - lo) / FPS))
    return np.array(v), rng


def main():
    rows = []
    files = glob.glob(os.path.join(DATASET_DIR, 'Skeleton_20', '*.npz'))
    meta = [np.load(f, allow_pickle=True) for f in files]
    top = {}
    for z in meta:
        m = re.match(r'^(\d+)', str(z['trc']))
        top[str(z['subject'])] = max(top.get(str(z['subject']), 0), float(m.group(1)) if m else 0)
    for z in meta:
        s, trc = str(z['subject']), str(z['trc'])
        m = re.match(r'^(\d+)', trc)
        kg = float(m.group(1)) if m else None
        cls = ('compensation' if kg is None else '0 kg' if kg == 0 else 'top (~10RM)' if kg == top[s]
               else '20 kg bar' if kg == 20 else 'middle')
        v, rng = rep_velocities(z)
        row = {'Subject': s, 'seg': int(z['seg']), 'trc': trc, 'kg': kg, 'class': cls, 'n_reps': len(v),
               'depth_m': rng}
        if len(v) >= 4:
            best = v[:2].max()
            row.update({'v_first_best': best, 'v_last': v[-1], 'velocity_loss_pct': 100 * (1 - v[-1] / best),
                        'v_slope_pct_per_rep': stats.linregress(np.arange(len(v)), 100 * v / best).slope})
        rows.append(row)
    d = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    d.to_csv(os.path.join(OUT, 'velocity_per_segment.csv'), index=False, encoding='utf-8-sig')
    ok = d.dropna(subset=['velocity_loss_pct'])
    order = ['0 kg', '20 kg bar', 'middle', 'top (~10RM)', 'compensation']
    summ = ok.groupby('class').agg(n=('Subject', 'size'), reps=('n_reps', 'median'),
                                   v_first=('v_first_best', 'median'),
                                   vl_mean=('velocity_loss_pct', 'mean'), vl_median=('velocity_loss_pct', 'median'),
                                   vl_q25=('velocity_loss_pct', lambda x: x.quantile(.25)),
                                   vl_q75=('velocity_loss_pct', lambda x: x.quantile(.75))).reindex(order)
    pd.set_option('display.width', 250)
    print(f'有效組數 {len(ok)} / {len(d)}')
    print(summ.round(2).to_string())
    p = ok.groupby(['Subject', 'class']).velocity_loss_pct.mean().unstack()
    x = p[['top (~10RM)', '0 kg']].dropna()
    print(f'\n最重組 vs 徒手（同一人配對 n={len(x)}）：速度損失差 {(x.iloc[:, 0] - x.iloc[:, 1]).mean():.1f} 個百分點，'
          f'p = {stats.ttest_rel(x.iloc[:, 0], x.iloc[:, 1]).pvalue:.4g}')
    print('Spearman：速度損失 vs 負重 % 最重 =',
          stats.spearmanr(ok.dropna(subset=['kg']).kg / ok.dropna(subset=['kg']).Subject.map(top),
                          ok.dropna(subset=['kg']).velocity_loss_pct))
    mdf = os.path.join(OUT, 'mdf_per_segment.csv')
    if os.path.exists(mdf):
        e = pd.read_csv(mdf)[['Subject', 'trc', 'slope_rms_main_pct_per_rep', 'slope_main_pct_per_rep']]
        j = ok.merge(e, on=['Subject', 'trc']).dropna()
        print(f'\n與肌電合併 {len(j)} 組：速度損失 vs RMS 上升斜率 Spearman =',
              stats.spearmanr(j.velocity_loss_pct, j.slope_rms_main_pct_per_rep),
              '\n                         速度損失 vs MDF 斜率 Spearman =',
              stats.spearmanr(j.velocity_loss_pct, j.slope_main_pct_per_rep))


if __name__ == '__main__':
    main()
