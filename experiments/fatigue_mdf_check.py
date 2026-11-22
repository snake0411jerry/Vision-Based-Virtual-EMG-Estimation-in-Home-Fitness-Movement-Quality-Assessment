# English summary: Fatigue feasibility: EMG median frequency does not drop within a set, even at ~10RM, so the current data cannot support fatigue modelling.
# (Added Nov 2026; the detailed comments below are in Chinese.)
"""
fatigue_mdf_check.py — 規劃二 可行性檢查：每組深蹲內，肌電中位頻率（MDF）是否隨次數下降？
=========================================================================
若 MDF 在一組內不會隨次數穩定下降，就無法拿它當「影片估計疲勞」的標籤，規劃二要改方向。

流程（只用 CPU）：
  1. 原始 1000 Hz 肌電 → 與 emg_process_segment_FIXED 相同的濾波（去直流、60 Hz notch、20 Hz 高通），
     依同步訊號切成各組（Segment k ＝ Skeleton_20 的 Seg_k）。
  2. 主動肌（Raw0）整流後 1.5 Hz 低通得到包絡線，找出每一下的出力峰（相隔 ≥ 1.5 s）。
  3. 每一下取包絡線 ≥ 峰值 50 % 的連續區段，Welch 頻譜算 MDF（20–450 Hz）。
  4. 每組：MDF 相對前 3 下平均的變化 %，對次數做線性迴歸 → 斜率（%/下）；以及末 3 下 vs 前 3 下。
  5. 依負重分類（徒手 0 kg / 空槓 20 kg / 中間 / 最重 ≈ 10 下極限 / 代償動作）彙整與檢定。

負重來自每組 TRC 檔名（Skeleton_20 npz 的 trc 欄位）：數字 = 公斤，0 = 無負重，字母 a/b/c = 代償（無負重）。
輸出 results/fatigue/：mdf_per_rep.csv、mdf_per_segment.csv、mdf_summary.csv（只含 S 編號，不含姓名）
"""
# --- 路徑設定：一律由 repo 根的 paths.py 集中管理，請勿改回絕對路徑 ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import PIPELINE_DIR, RESULTS_DIR, DATASET_DIR, require_subjects_csv  # noqa: E402
# --- 路徑設定結束 ---

import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, sosfiltfilt, find_peaks, welch

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, PIPELINE_DIR)
from emg_process_segment_FIXED import design_emg_filters, read_raw_csv  # noqa: E402

FS = 1000.0
OUT = os.path.join(RESULTS_DIR, 'fatigue')
SOS_ENV = butter(2, 1.5 / (FS / 2), btype='low', output='sos')


def raw_files():
    """{S編號: 原始 csv 路徑}。S01–S10 由 subjects.csv（含個資、已 gitignore）對應，S11–S20 直接依檔名。"""
    m = {}
    sub = pd.read_csv(require_subjects_csv())
    for _, r in sub.iterrows():
        m[r['subject_key']] = os.path.join(DATASET_DIR, 'Raw', os.path.basename(str(r['raw_csv']).replace('\\', '/')))
    for p in sorted(glob.glob(os.path.join(DATASET_DIR, 'Raw_20', 'S*_Raw_DATA.csv'))):
        m[os.path.basename(p)[:3]] = p
    return m


def seg_loads():
    """{(S, seg): (trc 名, 公斤或 None, 類別)}"""
    rows = []
    for f in glob.glob(os.path.join(DATASET_DIR, 'Skeleton_20', '*.npz')):
        z = np.load(f, allow_pickle=True)
        trc = str(z['trc'])
        m = re.match(r'^(\d+)', trc)
        rows.append((str(z['subject']), int(z['seg']), trc, float(m.group(1)) if m else None, len(z['coords']) / 60.0))
    df = pd.DataFrame(rows, columns=['Subject', 'seg', 'trc', 'kg', 'dur_s'])
    out = {}
    for s, g in df.groupby('Subject'):
        top = g.kg.max()
        for _, r in g.iterrows():
            if r.kg is None or np.isnan(r.kg):
                cls = 'compensation'
            elif r.kg == 0:
                cls = '0 kg'
            elif r.kg == top:
                cls = 'top (~10RM)'
            elif r.kg == 20:
                cls = '20 kg bar'
            else:
                cls = 'middle'
            out[(s, int(r.seg))] = (r.trc, r.kg, cls, r.dur_s)
    return out


def match_segments(raw_durs, skel):
    """原始檔的段數可能多於骨架段（如 S11–S20 有多錄的段）。依時間順序、以時長（誤差 < 1 s）配對。
    skel：[(seg, dur_s)] 依 seg 排序；回傳 {seg: 原始段索引}"""
    out, j = {}, 0
    for seg, d in skel:
        while j < len(raw_durs) and abs(raw_durs[j] - d) >= 1.0:
            j += 1
        if j < len(raw_durs):
            out[seg] = j
            j += 1
    return out


def segments_fft(path):
    """與 emg_process_segment_FIXED 相同：先整檔濾波，再依 Time_ms == 1 切段。回傳 [Raw0, Raw1] 高通訊號列表。"""
    df = read_raw_csv(path)
    b_n, a_n, sos_h, _ = design_emg_filters(FS)
    from scipy.signal import filtfilt
    sig = {}
    for c in ['Raw0', 'Raw1']:
        x = pd.to_numeric(df[c], errors='coerce').fillna(0).values.astype(float)
        sig[c] = sosfiltfilt(sos_h, filtfilt(b_n, a_n, x - x.mean()))
    starts = df.index[df['Time_ms'] == 1].tolist() + [len(df)]
    return [(sig['Raw0'][a:b], sig['Raw1'][a:b]) for a, b in zip(starts[:-1], starts[1:])]


def mdf(x):
    f, p = welch(x, fs=FS, nperseg=min(256, len(x)))
    band = (f >= 20) & (f <= 450)
    f, p = f[band], p[band]
    c = np.cumsum(p)
    return float(f[np.searchsorted(c, c[-1] / 2)])


def reps(main):
    env = sosfiltfilt(SOS_ENV, np.abs(main))
    lo, hi = np.percentile(env, 5), np.percentile(env, 95)
    peaks, _ = find_peaks(env, distance=int(1.5 * FS), prominence=0.3 * (hi - lo))
    out = []
    for pk in peaks:
        thr = 0.5 * env[pk]
        a = pk
        while a > 0 and env[a] > thr:
            a -= 1
        b = pk
        while b < len(env) - 1 and env[b] > thr:
            b += 1
        if b - a >= 300:
            out.append((a, b))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    loads = seg_loads()
    per_rep, per_seg = [], []
    for s, path in sorted(raw_files().items()):
        if not os.path.exists(path):
            print(f'⚠️ {s} 找不到原始檔，略過')
            continue
        segs = segments_fft(path)
        skel = sorted((k, v[3]) for (ss, k), v in loads.items() if ss == s)
        pair = match_segments([len(a) / FS for a, _ in segs], skel)
        miss = [k for k, _ in skel if k not in pair]
        print(f'{s}: 原始 {len(segs)} 段，骨架 {len(skel)} 段，配對 {len(pair)}' + (f'，未配對 seg {miss}' if miss else ''),
              flush=True)
        for k, idx in sorted(pair.items()):
            m0, m1 = segs[idx]
            trc, kg, cls, _ = loads[(s, k)]
            rr = reps(m0)
            if len(rr) < 4:
                per_seg.append({'Subject': s, 'seg': k, 'trc': trc, 'kg': kg, 'class': cls, 'n_reps': len(rr)})
                continue
            f0 = np.array([mdf(m0[a:b]) for a, b in rr])
            f1 = np.array([mdf(m1[a:b]) for a, b in rr])
            rms0 = np.array([np.sqrt(np.mean(m0[a:b] ** 2)) for a, b in rr])
            rel = 100 * (rms0 / rms0[:3].mean() - 1)
            rms_slope = stats.linregress(np.arange(len(rms0)), rel).slope
            for i, ((a, b), x0, x1) in enumerate(zip(rr, f0, f1), 1):
                per_rep.append({'Subject': s, 'seg': k, 'class': cls, 'kg': kg, 'rep': i,
                                'dur_s': (b - a) / FS, 'mdf_main': x0, 'mdf_syn': x1})
            row = {'Subject': s, 'seg': k, 'trc': trc, 'kg': kg, 'class': cls, 'n_reps': len(rr),
                   'slope_rms_main_pct_per_rep': rms_slope, 'dur_s': len(m0) / FS}
            for name, f in [('main', f0), ('syn', f1)]:
                rel = 100 * (f / f[:3].mean() - 1)
                row[f'slope_{name}_pct_per_rep'] = stats.linregress(np.arange(len(f)), rel).slope
                row[f'last3_vs_first3_{name}_pct'] = 100 * (f[-3:].mean() / f[:3].mean() - 1)
                row[f'mdf_{name}_first3'] = f[:3].mean()
            per_seg.append(row)
    pr, ps = pd.DataFrame(per_rep), pd.DataFrame(per_seg)
    pr.to_csv(os.path.join(OUT, 'mdf_per_rep.csv'), index=False, encoding='utf-8-sig')
    ps.to_csv(os.path.join(OUT, 'mdf_per_segment.csv'), index=False, encoding='utf-8-sig')
    ok = ps.dropna(subset=['slope_main_pct_per_rep'])
    rows = []
    order = ['0 kg', '20 kg bar', 'middle', 'top (~10RM)', 'compensation']
    for cls in order:
        g = ok[ok['class'] == cls]
        if g.empty:
            continue
        r = {'class': cls, 'n_segments': len(g), 'n_subjects': g.Subject.nunique(), 'median_reps': g.n_reps.median()}
        for name in ['main', 'syn']:
            sl = g[f'slope_{name}_pct_per_rep']
            d = g[f'last3_vs_first3_{name}_pct']
            r[f'{name}_slope_mean'] = sl.mean()
            r[f'{name}_slope_p'] = stats.ttest_1samp(sl, 0).pvalue if len(sl) > 2 else np.nan
            r[f'{name}_frac_negative'] = (sl < 0).mean()
            r[f'{name}_last3_vs_first3_mean'] = d.mean()
        sl = g['slope_rms_main_pct_per_rep']
        r['rms_main_slope_mean'] = sl.mean()
        r['rms_main_slope_p'] = stats.ttest_1samp(sl, 0).pvalue if len(sl) > 2 else np.nan
        r['rms_main_frac_positive'] = (sl > 0).mean()
        rows.append(r)
    summ = pd.DataFrame(rows)
    top, bw = ok[ok['class'] == 'top (~10RM)'], ok[ok['class'] == '0 kg']
    both = top.merge(bw, on='Subject', suffixes=('_top', '_0'))
    if len(both) > 2:
        p = stats.ttest_rel(both.slope_main_pct_per_rep_top, both.slope_main_pct_per_rep_0).pvalue
        print(f'\n最重組 vs 徒手（同一人配對，n={both.Subject.nunique()}）主動肌斜率差 p = {p:.4f}')
    summ.to_csv(os.path.join(OUT, 'mdf_summary.csv'), index=False, encoding='utf-8-sig')
    pd.set_option('display.width', 250)
    print(f'\n有效組數 {len(ok)} / {len(ps)}（少於 4 下的組不計）')
    print(summ.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
