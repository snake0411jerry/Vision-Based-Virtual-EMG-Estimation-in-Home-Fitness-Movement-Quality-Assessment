# English summary: Aligns newly collected OpenCap TRC files to EMG segments using the waist-pack LED visible in the raw videos (recording order enforced, review PDF, --apply to trim).
# (Added Nov 2026; the detailed comments below are in Chinese.)
"""
align_trc_by_led.py — 用影片中的 LED 把 OpenCap TRC 裁到與肌電段落逐幀對齊
=========================================================================
【做什麼】
-------------------------------------------------------------------------
收案時，Teensy 開始錄肌電的瞬間腰包上的 LED 會亮起、停止時熄滅
（firmware/0505/V0/V0.ino 的 STATE_RECORDING）。OpenCap 的影片頭尾各多錄了
幾秒，肌電檔裡也常有誤按產生的多餘短段落。本程式：

  1. 在兩台相機的原始 .mov（720×1280）上，以 OpenCap 偵測到的受試者關鍵點框出腰包，逐幀找 LED
  2. 把 .mov 的幀號換算成 sync 影片（＝TRC）的幀號
  3. 依「錄製順序」把每個 trial 和每段肌電配對
  4. 每台相機各自量出 LED 亮起與熄滅的幀，選亮滅間隔最吻合肌電長度的那台決定起點
  5. 輸出對照表、LED 訊號檢查圖、以及每個 trial 開頭／結尾畫面的 PDF
  6. 加 --apply 才實際裁切 MarkerData/*.trc

**只裁 MarkerData，不動影片，也不動肌電原始檔**（多錄的段落在對照表中標出，
下游 Features_insert_FIXED.py 用 segment_ids 跳過即可）。

【為什麼這樣做（都是實際踩過的）】
-------------------------------------------------------------------------
* 整張畫面找紅點會被場景裡的紅白標示牌、紅色鞋底騙（2026-10-14 收案）→ 只看腰包。
* 用 MediaPipe 追腰包，在有鏡子的場地會追到鏡中倒影 → 改用 OpenCap 自己的 2D 關鍵點，
  每幀挑「身高最大、髖部最靠近畫面中央」的人（倒影較小且偏一側）。
* sync 影片只有 360×640，LED 常只剩 2–5 個像素 → 改在原始 .mov 上找。
* 有些 trial 的兩支影片被放在對調的相機資料夾（2026-10-08 收案有一個 trial 因此讓 OpenCap 處理失敗）
  → 關鍵點檔依「幀數相同」對到影片，sync 影片依「畫面相同」對到影片，不信資料夾名稱。
* 兩台相機常有一台在開頭被誤判為亮 → 不直接合併兩台，每台各自量邊界，
  取亮滅間隔與肌電長度最吻合的那台。
* 好幾段肌電長度幾乎一樣（差 0.2 秒）→ 只靠長度分不出來，配對時強制遵守錄製順序
  （從 .mov 的 mvhd 讀拍攝時間）。

【裁切慣例（與手動整理的 0707 一致）】
-------------------------------------------------------------------------
* 保留的第一幀 Frame# = LED 第一次亮起的 sync 影片幀索引（cv2 從 0 起算）。
  已在 0707 的手動裁切上核對過：手動保留的第一幀正是 LED 第一次亮起的那幀。
* 保留列數 = round(肌電段落秒數 × 60) + 1，裁完的 TRC 時長即等於肌電時長。
* Frame# 與 Time 保留原值（不重新編號），與 sync 影片幀仍可一一對照。
* 🔴 與 0707 不同：**表頭 NumFrames 會改成實際保留的列數**。0707 沒改，
  曾讓組員的 MediaPipe 匯出照表頭多抓了 3–10 秒（見 align_mediapipe_trc.py）。
* 第一次 --apply 會把整個 MarkerData 備份到 MarkerData_original/；之後一律從
  備份裁切，可重複執行。若某個 TRC 換成了新的未裁切檔（例如 OpenCap 重跑後），
  會自動以新檔為準並更新備份。
* 第一行含 "LOCAL RECONSTRUCTION" 的 TRC（本機兩相機重建的 trial）一律不動。

【一定要人工看的】
-------------------------------------------------------------------------
* alignment_frames.pdf：每個 trial 一頁，兩台相機各四格 ——
  開頭前一格 LED 應未亮、開頭那格應剛亮；結尾那格應還亮、後一格應已熄滅。
* 狀態不是 "aligned" 的列（例如影片比肌電先停）需要人工決定。

【資料夾結構】
-------------------------------------------------------------------------
    <收案日>/Open/<OpenCap 資料夾>/MarkerData/*.trc
    <收案日>/Open/<OpenCap 資料夾>/Videos/Cam0|Cam1/InputMedia/<trial>/<trial>.mov 與 <trial>_sync.mp4
    <收案日>/Open/<OpenCap 資料夾>/Videos/Cam0|Cam1/OutputPkl/<trial>_keypoints.pkl
    <收案日>/Raw/*_Raw_DATA.csv
（Open/Raw 不分大小寫。）OpenCap 資料夾與肌電檔用名稱自動對應（去掉 "OpenCapData_"
與日期後，不分大小寫比對肌電檔名開頭）。對不起來時用 --map 指定。

用法：
    python pipeline/align_trc_by_led.py <收案日資料夾>             # 只分析，不改檔
    python pipeline/align_trc_by_led.py <收案日資料夾> --apply     # 看過 PDF 後實際裁切
    python pipeline/align_trc_by_led.py <收案日資料夾> --exclude <OpenCap 資料夾>:<trial> --apply
    python pipeline/align_trc_by_led.py <收案日資料夾> --map <OpenCap 資料夾>=<肌電檔名>

輸出（寫在收案日資料夾內）：
    alignment_report.csv   每段肌電對應哪個 TRC、保留範圍、各相機 LED 亮滅幀與狀態
    alignment_check.png    每個 trial 的 LED 訊號與保留範圍
    alignment_frames.pdf   每個 trial 開頭與結尾前後的畫面（腰包放大）
    _led_cache/            每支影片的 LED 逐幀數值與 sync 對應（快取，影片變更時自動重算）
"""
import argparse
import datetime
import filecmp
import glob
import json
import os
import pickle
import re
import shutil
import struct
import sys

import cv2
import numpy as np
import pandas as pd

FPS = 60
CAMS = ('Cam0', 'Cam1')
METHOD = 'mov-keypoint-roi-v1'       # 偵測方法改了就換字串，舊快取自動失效
_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# ---------------------------------------------------------------- 關鍵點與 LED
def load_subject_keypoints(path, width=720, height=1280):
    """OpenCap 2D 關鍵點（BODY_25）→ (幀數, 25, 3)。每幀挑身高最大、髖部最靠近畫面中央的人（排除鏡中倒影）。"""
    frames = pickle.load(open(path, 'rb'))
    out = np.full((len(frames), 25, 3), np.nan)
    for i, people in enumerate(frames):
        best, best_key = None, -np.inf
        for p in people:
            k = np.array(p['pose_keypoints_2d'], float).reshape(25, 3)
            good = k[:, 2] > 0.2
            if good.sum() < 8:
                continue
            span = k[good, 1].max() - k[good, 1].min()
            hip = k[8, :2] if k[8, 2] > 0.2 else k[good, :2].mean(axis=0)
            key = span / height - 0.8 * abs(hip[0] - width / 2) / width
            if key > best_key:
                best, best_key = k, key
        if best is not None:
            out[i] = best
    return out


def waist_box(k, W, H):
    """腰包範圍：髖部中心左右各 0.3 個軀幹長、往上 0.45 個軀幹長（軀幹長 = 頸到髖）。"""
    if not (k[8, 2] > 0.2 and k[1, 2] > 0.2):
        return None
    hx, hy, nx, ny = k[8, 0], k[8, 1], k[1, 0], k[1, 1]
    t = max(40.0, float(np.hypot(hx - nx, hy - ny)))
    box = (int(max(0, hx - 0.3 * t)), int(max(0, hy - 0.45 * t)), int(min(W, hx + 0.3 * t)), int(min(H, hy + 0.05 * t)))
    return box if box[2] > box[0] and box[3] > box[1] else None


def led_pixels(roi):
    """飽和紅、且在過曝白色核心 2 px 內的像素數。"""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((h >= 165) | (h <= 8)) & (s >= 100) & (v >= 110)
    core = ((v >= 225) & (s <= 140)).astype(np.uint8)
    return int((red & (cv2.dilate(core, _KERNEL) > 0)).sum())


def _sig(*paths):
    return {'method': METHOD, 'files': [[os.path.basename(p), os.path.getsize(p), int(os.path.getmtime(p))] for p in paths]}


def _cached(cache_path, sig):
    meta = cache_path + '.json'
    if os.path.exists(cache_path) and os.path.exists(meta):
        with open(meta, encoding='utf-8') as f:
            if json.load(f) == sig:
                return True
    return False


def _save_sig(cache_path, sig):
    with open(cache_path + '.json', 'w', encoding='utf-8') as f:
        json.dump(sig, f)


def mov_led(mov, kp, cache_path):
    """原始 .mov 上逐幀的 LED 分數與腰包範圍（以關鍵點檔的幀對應 .mov 的幀）。"""
    sig = _sig(mov)
    if _cached(cache_path, sig):
        z = np.load(cache_path)
        return z['score'], z['box']
    cap = cv2.VideoCapture(mov)
    scores, boxes, i = [], [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        box = waist_box(kp[i], fr.shape[1], fr.shape[0]) if i < len(kp) else None
        scores.append(led_pixels(fr[box[1]:box[3], box[0]:box[2]]) if box else 0)
        boxes.append(box or (0, 0, 0, 0))
        i += 1
    score, box = np.array(scores, float), np.array(boxes, int)
    np.savez(cache_path, score=score, box=box)
    _save_sig(cache_path, sig)
    return score, box


def grey_stack(path, size=(45, 80)):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA).astype(np.float32))
    return np.array(out)


def sync_offset(sync_g, mov_g, step=25):
    """sync 第 k 幀 = .mov 第 k+s 幀 的 s，以縮小灰階畫面的平均絕對差最小決定。"""
    n = len(sync_g)
    ks = np.arange(0, n, step)
    best = (None, np.inf)
    for s in range(-30, len(mov_g) - n + 31):
        kk = ks[(ks + s >= 0) & (ks + s < len(mov_g))]
        if len(kk) < len(ks) * 0.8:
            continue
        d = float(np.mean(np.abs(sync_g[kk] - mov_g[kk + s])))
        if d < best[1]:
            best = (s, d)
    return best


def trial_cameras(sdir, trial, cache):
    """每台（校正上的）相機：對應的 .mov、sync 影片、幀號偏移，以及換算到 sync 時間軸的 LED。

    回傳 {cam: dict(mov, offset, sync, score_sync, lit, box_mov, swapped)}，cam 依關鍵點檔所在資料夾命名
    （OpenCap 的關鍵點與校正是對的；放錯的只會是 .mov / sync 影片的資料夾）。
    """
    vids = os.path.join(sdir, 'Videos')
    movs = {c: os.path.join(vids, c, 'InputMedia', trial, f'{trial}.mov') for c in CAMS}
    syncs = {c: os.path.join(vids, c, 'InputMedia', trial, f'{trial}_sync.mp4') for c in CAMS}
    movs = {c: p for c, p in movs.items() if os.path.exists(p)}
    syncs = {c: p for c, p in syncs.items() if os.path.exists(p) and cv2.VideoCapture(p).get(cv2.CAP_PROP_FRAME_COUNT) > 10 * FPS}
    if not movs or not syncs:
        return {}
    nmov = {c: int(cv2.VideoCapture(p).get(cv2.CAP_PROP_FRAME_COUNT)) for c, p in movs.items()}
    subj = os.path.basename(sdir)

    map_path = os.path.join(cache, f'{subj}__{trial}__syncmap.json')
    sig = _sig(*movs.values(), *syncs.values())
    mapping = None
    if os.path.exists(map_path):
        with open(map_path, encoding='utf-8') as f:
            saved = json.load(f)
        if saved.get('sig') == sig:
            mapping = saved['map']
    if mapping is None:
        mov_g = {c: grey_stack(p) for c, p in movs.items()}
        mapping = {}
        for sc, sp in syncs.items():
            sg = grey_stack(sp)
            best = min(((mc,) + sync_offset(sg, g) for mc, g in mov_g.items()), key=lambda r: r[2])
            mapping[sc] = dict(mov=best[0], offset=int(best[1]), diff=round(best[2], 2), n=len(sg))
        with open(map_path, 'w', encoding='utf-8') as f:
            json.dump({'sig': sig, 'map': mapping}, f)

    out = {}
    for pc in CAMS:
        kpp = os.path.join(vids, pc, 'OutputPkl', f'{trial}_keypoints.pkl')
        if not os.path.exists(kpp):
            continue
        kp = load_subject_keypoints(kpp)
        if pc in mapping:
            # OpenCap 的第 k 台相機：sync 影片、關鍵點、校正都在 Cam<k>/；它來自哪支 .mov 由畫面比對決定
            mc, m, sync_path = mapping[pc]['mov'], mapping[pc], syncs[pc]
        else:
            # 這台相機沒有可用的 sync 影片（例如 OpenCap 處理失敗）：退而以幀數最接近的 .mov 配對
            mc = min(movs, key=lambda c: abs(nmov[c] - len(kp)))
            sync_of = [sc for sc, mm in mapping.items() if mm['mov'] == mc]
            if not sync_of:
                continue
            m, sync_path = mapping[sync_of[0]], syncs[sync_of[0]]
        score, box = mov_led(movs[mc], kp, os.path.join(cache, f'{subj}__{trial}__kp{pc}_mov{mc}.npz'))
        idx = np.arange(m['n']) + m['offset']
        inside = (idx >= 0) & (idx < len(score))
        s_sync = np.where(inside, score[np.clip(idx, 0, len(score) - 1)], 0.0)
        out[pc] = dict(mov=movs[mc], mov_folder=mc, offset=m['offset'], sync=sync_path, score_sync=s_sync,
                       lit=lit_of(s_sync), box_mov=box, swapped=(mc != pc))
    return out


def lit_of(s):
    base, top = np.percentile(s, 10), np.percentile(s, 95)
    return (s > base + max(2.0, 0.2 * (top - base))).astype(float)


def best_window(lit, D):
    """長度 D 的窗：窗內亮比例 − 窗外亮比例 最大的起點。"""
    n = len(lit)
    if D > n:
        return None, -np.inf
    c = np.r_[0.0, np.cumsum(lit)]
    st = np.arange(0, n - D + 1)
    inside = c[st + D] - c[st]
    out_n = n - D
    score = inside / D - ((c[-1] - inside) / out_n if out_n else 0.0)
    k = int(np.argmax(score))
    return int(st[k]), float(score[k])


def lit_span(l, w=30, frac=0.6):
    """與肌電無關、單純從 LED 量出的亮起區間：(第一個亮的幀, 最後一個亮的幀 + 1)。"""
    if len(l) <= w:
        return None, None
    c = np.r_[0.0, np.cumsum(l)]
    idx = np.flatnonzero((c[w:] - c[:-w]) / w >= frac)
    if not len(idx):
        return None, None
    a, b = int(idx[0]), int(idx[-1]) + w - 1
    a = next((t for t in range(max(0, a - 5), min(len(l), a + w)) if l[t]), a)
    b = next((t for t in range(min(len(l) - 1, b + 5), max(-1, b - w), -1) if l[t]), b)
    return a, b + 1


# ---------------------------------------------------------------- 錄製順序
def _boxes(f, start, end):
    pos = start
    while pos + 8 <= end:
        f.seek(pos)
        size, typ = struct.unpack('>I4s', f.read(8))
        hdr = 8
        if size == 1:
            size, hdr = struct.unpack('>Q', f.read(8))[0], 16
        elif size == 0:
            size = end - pos
        if size < hdr:
            return
        yield typ, pos + hdr, pos + size
        pos += size


def mov_creation_time(path):
    """QuickTime moov/mvhd 的建立時間（UTC）；讀不到回傳 None。"""
    try:
        with open(path, 'rb') as f:
            for typ, s, e in _boxes(f, 0, os.path.getsize(path)):
                if typ == b'moov':
                    for t2, s2, _ in _boxes(f, s, e):
                        if t2 == b'mvhd':
                            f.seek(s2)
                            ver = f.read(4)[0]
                            secs = struct.unpack('>Q' if ver == 1 else '>I', f.read(8 if ver == 1 else 4))[0]
                            return datetime.datetime(1904, 1, 1) + datetime.timedelta(seconds=secs)
    except (OSError, struct.error):
        pass
    return None


def ordered_assignment(S, min_score=0.2):
    """trial（依錄製順序）× 段落（依時間順序）分數矩陣 → 順序不交錯、總分最高的配對；多的段落可跳過。"""
    T, N = S.shape
    ok = np.isfinite(S) & (S >= min_score)
    dp = np.zeros((T + 1, N + 1))
    move = np.zeros((T + 1, N + 1), int)          # 0 跳過段落、1 trial 不配、2 配對
    for i in range(T + 1):
        for j in range(N + 1):
            if i == 0 and j == 0:
                continue
            best, mv = -np.inf, -1
            if j > 0 and dp[i, j - 1] > best:
                best, mv = dp[i, j - 1], 0
            if i > 0 and dp[i - 1, j] > best:
                best, mv = dp[i - 1, j], 1
            if i > 0 and j > 0 and ok[i - 1, j - 1] and dp[i - 1, j - 1] + S[i - 1, j - 1] > best:
                best, mv = dp[i - 1, j - 1] + S[i - 1, j - 1], 2
            dp[i, j], move[i, j] = best, mv
    pairs, i, j = {}, T, N
    while i > 0 or j > 0:
        mv = move[i, j]
        if mv == 2:
            pairs[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif mv == 1:
            i -= 1
        else:
            j -= 1
    return pairs


# ---------------------------------------------------------------- 檔案
def emg_durations(path):
    """每段肌電的秒數；段落以 Time_ms 歸零切分。"""
    t = pd.read_csv(path, usecols=['Time_ms'])['Time_ms'].to_numpy()
    starts = np.flatnonzero(np.r_[True, np.diff(t) < 0])
    ends = np.r_[starts[1:], len(t)]
    return np.array([(t[b - 1] - t[a]) / 1000.0 for a, b in zip(starts, ends)])


def read_trc(path):
    with open(path, 'rb') as f:
        lines = f.read().split(b'\n')
    data = [l for l in lines[6:] if l.strip()]
    frames = [int(float(l.split(b'\t', 1)[0])) for l in data]
    hdr = int(float(lines[2].split(b'\t')[2]))
    return lines, data, frames, hdr


def is_uncropped(path):
    _, data, frames, hdr = read_trc(path)
    return bool(frames) and frames[0] == 1 and len(data) == hdr


def crop_trc(src, dst, start, n_rows):
    lines, data, frames, _ = read_trc(src)
    kept = [l for l, fr in zip(data, frames) if start <= fr < start + n_rows]
    if len(kept) != n_rows:
        raise ValueError(f'{src}: 要從 Frame# {start} 取 {n_rows} 列，實際只有 {len(kept)} 列')
    head = lines[:6]
    vals = head[2].split(b'\t')
    vals[2] = str(n_rows).encode()
    head[2] = b'\t'.join(vals)
    eol = b'\r\n' if lines[0].endswith(b'\r') else b'\n'
    body = b'\n'.join(head + kept)
    with open(dst, 'wb') as f:
        f.write(body if body.endswith(eol) else body + eol)


def subdir(parent, name):
    """不分大小寫找子資料夾（例如 Open / open）。"""
    for d in os.listdir(parent):
        if d.lower() == name.lower() and os.path.isdir(os.path.join(parent, d)):
            return os.path.join(parent, d)
    return os.path.join(parent, name)


def subject_key(folder):
    k = re.sub(r'^OpenCapData_?', '', folder, flags=re.I)
    k = re.sub(r'(^|_)\d{3,8}(?=_|$)', '', k).strip('_')
    return k.lower()


def map_raw(raw_dir, subjects, manual):
    raws = sorted(os.path.basename(p) for p in glob.glob(os.path.join(raw_dir, '*_Raw_DATA.csv')))
    out = {}
    for s in subjects:
        if s in manual:
            out[s] = manual[s]
            continue
        key = subject_key(s)
        hits = [r for r in raws if r.lower().split('_')[0] == key] or [r for r in raws if r.lower().startswith(key)]
        if len(hits) != 1:
            sys.exit(f'❌ {s} 找不到唯一對應的肌電檔（候選：{hits or raws}）。請用 --map {s}=<肌電檔名> 指定。')
        out[s] = hits[0]
    return out


def local_reconstruction(sdir, tr):
    """本機重建的 TRC（第一行含 LOCAL RECONSTRUCTION）→ 回傳 (肌電段號, 秒數)；否則 None。這類檔案本程式一律不動。"""
    cur = os.path.join(sdir, 'MarkerData', f'{tr}.trc')
    if not os.path.exists(cur):
        return None
    with open(cur, 'rb') as f:
        first = f.readline().decode('utf-8', 'replace')
    if 'LOCAL RECONSTRUCTION' not in first:
        return None
    seg = re.search(r'emg_segment=(\d+)', first)
    dur = re.search(r'emg_duration_s=(\d+(?:\.\d+)?)', first)
    return (int(seg.group(1)) if seg else None, float(dur.group(1)) if dur else None)


def trc_source(sdir, tr):
    cur = os.path.join(sdir, 'MarkerData', f'{tr}.trc')
    bak = os.path.join(sdir, 'MarkerData_original', f'{tr}.trc')
    if os.path.exists(cur) and is_uncropped(cur):
        return cur
    if os.path.exists(bak):
        return bak
    return cur if os.path.exists(cur) else None


# ---------------------------------------------------------------- 主流程
def analyze(open_dir, raw_dir, raw_of, cache, min_trial_s, excluded, starts=None):
    starts = starts or {}
    rows, trials_out = [], []
    for subj, raw in raw_of.items():
        sdir = os.path.join(open_dir, subj)
        durs = emg_durations(os.path.join(raw_dir, raw))
        cand = [j for j, d in enumerate(durs) if d >= min_trial_s]

        info = []
        for tdir in sorted(glob.glob(os.path.join(sdir, 'Videos', 'Cam0', 'InputMedia', '*'))):
            tr = os.path.basename(tdir)
            if tr == 'neutral':
                continue
            t0 = None
            for cam in CAMS:
                t0 = t0 or mov_creation_time(os.path.join(sdir, 'Videos', cam, 'InputMedia', tr, f'{tr}.mov'))
            rec = local_reconstruction(sdir, tr)
            cams = {} if rec else trial_cameras(sdir, tr, cache)
            print(f'  {subj} {tr}: ' + ', '.join(f"{c} <- 影片 {v['mov_folder']}/{'（資料夾對調）' if v['swapped'] else ''}"
                                                 for c, v in cams.items()), flush=True)
            src = trc_source(sdir, tr)
            broken = src is None or len(read_trc(src)[1]) < 10 * FPS
            info.append(dict(trial=tr, t0=t0, cams=cams, rec=rec, usable=bool(cams) and not broken and not rec))

        info.sort(key=lambda x: (x['t0'] or datetime.datetime.max, x['trial']))
        use = [x for x in info if x['usable']]
        # 配對分數 = LED 窗比對分數 − LED 實際亮的長度與肌電長度的落差（相對值）
        # 後一項讓 15 秒的中斷段落不會被配給亮了 33 秒的 trial（2026-10-14 收案實例）
        S = np.full((len(use), len(cand)), -np.inf)
        for i, x in enumerate(use):
            n = min(len(v['lit']) for v in x['cams'].values())
            spans = [lit_span(v['lit'][:n]) for v in x['cams'].values()]
            spans = [(a, b) for a, b in spans if a is not None]
            for jj, j in enumerate(cand):
                D = int(round(durs[j] * FPS))
                win = max(best_window(v['lit'], D)[1] for v in x['cams'].values())
                gap = min(abs(b - a - D) for a, b in spans) / D if spans else 0.0
                S[i, jj] = win - 0.5 * gap
        pairs = ordered_assignment(S)

        used = set()
        for x in info:
            base = dict(subject=subj, raw_file=raw, trc=f"{x['trial']}.trc",
                        recorded_at=x['t0'].strftime('%Y-%m-%d %H:%M:%S') if x['t0'] else '',
                        video_folders_swapped=any(v['swapped'] for v in x['cams'].values()))
            if x['rec']:
                base.update(emg_segment=x['rec'][0], emg_duration_s=x['rec'][1],
                            status='aligned（本機重建的 TRC，本程式不裁切也不還原）')
                rows.append(base)
                used.add(x['rec'][0])
                continue
            if not x['usable']:
                base['status'] = 'OpenCap 輸出異常（TRC 列數過少或缺少 sync 影片），未裁切'
                rows.append(base)
                continue
            i = use.index(x)
            if i not in pairs:
                base['status'] = '配不到肌電段落'
                rows.append(base)
                continue
            j = cand[pairs[i]]
            D = int(round(durs[j] * FPS))
            r = dict(base, emg_segment=j + 1, emg_duration_s=round(durs[j], 3), match_score=round(S[i, pairs[i]], 3))
            n = min(len(v['lit']) for v in x['cams'].values())
            best = None
            for cam in CAMS:
                v = x['cams'].get(cam)
                a, b = lit_span(v['lit'][:n]) if v else (None, None)
                r[f'led_on_{cam}'], r[f'led_off_{cam}'] = a, b
                err = (b - a - D) if (a is not None and b is not None) else None
                r[f'duration_error_frames_{cam}'] = err
                if err is not None and (best is None or abs(err) < abs(best[1])):
                    best = (cam, err, a, b)
            fallback = max(x['cams'], key=lambda c: best_window(x['cams'][c]['lit'][:n], D)[1])
            if best and abs(best[1]) <= 6:
                on, status = best[2], 'aligned'
            elif best and abs(best[1]) <= 30:
                on, status = best[2], f'aligned（LED 間隔與肌電差 {best[1]:+d} 幀，請看 PDF）'
            else:
                on = best_window(x['cams'][fallback]['lit'][:n], D)[0]
                status = 'aligned（LED 邊界不清楚，起點由整段比對決定，請看 PDF）'
            if (subj, x['trial']) in starts:
                on = starts[(subj, x['trial'])]
                status = 'aligned（起點經人工逐幀確認）'
            ends_lit = best is not None and best[3] >= n - 2
            manual = (subj, x['trial']) in starts
            if on + D > n or (not manual and ends_lit and best[2] + D > n):
                status = '影片比肌電先停（LED 到最後一幀仍亮），未裁切，需人工決定'
            r.update(trc_first_frame=on, trc_last_frame=on + D, decided_by=best[0] if best else fallback)
            if (subj, x['trial']) in excluded:
                status = '排除（--exclude），未裁切'
            r['status'] = status
            rows.append(r)
            used.add(j + 1)
            trials_out.append(dict(subject=subj, trial=x['trial'], cams=x['cams'], on=on, off=on + D,
                                   seg=j + 1, dur=durs[j], status=status, n=n))

        for j, d in enumerate(durs):
            if j + 1 not in used:
                rows.append(dict(subject=subj, raw_file=raw, emg_segment=j + 1, emg_duration_s=round(d, 3),
                                 status='多錄的短段落（不是 trial）' if d < min_trial_s else '沒有對應的 TRC'))
    rep = pd.DataFrame(rows)
    rep['_seg'] = rep['emg_segment'].fillna(1e9) if 'emg_segment' in rep else 1e9
    rep = rep.sort_values(['subject', '_seg', 'trc']).drop(columns='_seg')
    return rep, trials_out


def _cjk_font():
    import matplotlib
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for n in ('Microsoft JhengHei', 'Microsoft YaHei', 'Noto Sans CJK TC', 'PingFang TC', 'SimHei'):
        if n in names:
            matplotlib.rcParams['font.sans-serif'] = [n, 'DejaVu Sans']
            break
    matplotlib.rcParams['axes.unicode_minus'] = False


def draw_signals(trials, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _cjk_font()
    if not trials:
        return
    fig, axes = plt.subplots(len(trials), 1, figsize=(15, 1.6 * len(trials)), squeeze=False)
    for ax, t in zip(axes[:, 0], trials):
        for k, (cam, c) in enumerate(zip(CAMS, ('tab:blue', 'tab:orange'))):
            if cam in t['cams']:
                ax.plot(t['cams'][cam]['lit'][:t['n']] * (1 - 0.08 * k), lw=0.5, color=c, label=cam)
        ax.axvspan(t['on'] - 1, t['off'] - 1, color='green', alpha=0.12)
        ax.axvline(t['on'] - 1, color='green', lw=1)
        ax.axvline(t['off'] - 1, color='red', lw=1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['暗', '亮'], fontsize=7)
        ax.tick_params(labelsize=7)
        ax.set_title(f"{t['subject']} {t['trial']}  ←  肌電第 {t['seg']} 段（{t['dur']:.2f} s）   "
                     f"保留 TRC Frame# {t['on']}–{t['off']}   {t['status']}", fontsize=8.5, loc='left')
    axes[0, 0].legend(fontsize=7, loc='upper right')
    axes[-1, 0].set_xlabel('sync 影片幀（各相機偵測到的 LED；綠色 = 保留範圍）')
    plt.tight_layout()
    plt.savefig(path, dpi=70)
    plt.close(fig)


def _mov_frame(path, idx):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    cap.release()
    return fr if ok else None


def draw_frames_pdf(trials, path):
    """每個 trial 一頁：兩台相機 × [開頭前一幀、開頭、結尾、結尾後一幀]，腰包放大（取自原始 .mov）。

    TRC Frame# k 對應 sync 影片第 k-1 幀（cv2 從 0 起算）。保留 Frame# on..off 時：
      sync 第 on-1 幀 → LED 應該還沒亮；第 on 幀 → LED 第一次亮起
      sync 第 off-1 幀 → LED 應該還亮；第 off 幀 → LED 應該已熄滅
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    _cjk_font()
    with PdfPages(path) as pdf:
        for t in trials:
            fig, axes = plt.subplots(2, 4, figsize=(11.7, 8.3))
            fig.suptitle(f"{t['subject']}  {t['trial']}  ←  肌電第 {t['seg']} 段（{t['dur']:.3f} 秒）\n"
                         f"保留 TRC Frame# {t['on']}–{t['off']}　｜　{t['status']}", fontsize=11)
            cells = [('開頭前一幀：應該還沒亮', t['on'] - 1), ('開頭：應該剛亮', t['on']),
                     ('結尾：應該還亮', t['off'] - 1), ('結尾後一幀：應該已熄滅', t['off'])]
            for row, cam in enumerate(CAMS):
                v = t['cams'].get(cam)
                for col, (label, sidx) in enumerate(cells):
                    ax = axes[row, col]
                    ax.axis('off')
                    if v is None:
                        ax.set_title(f'{cam}：無影片', fontsize=8)
                        continue
                    midx = sidx + v['offset']
                    fr = _mov_frame(v['mov'], midx)
                    if fr is None:
                        ax.set_title(f'{cam} {label}\nsync 第 {sidx} 幀：超出影片', fontsize=8)
                        continue
                    box = v['box_mov'][midx] if 0 <= midx < len(v['box_mov']) else (0, 0, 0, 0)
                    if box[2] > box[0]:
                        x0, y0, x1, y1 = box
                        pw, ph = int((x1 - x0) * 0.5), int((y1 - y0) * 0.6)
                        fr = fr[max(0, y0 - ph):min(fr.shape[0], y1 + ph), max(0, x0 - pw):min(fr.shape[1], x1 + pw)]
                    lit = v['lit']
                    det = '' if not 0 <= sidx < len(lit) else ('　偵測：亮' if lit[sidx] else '　偵測：暗')
                    ax.imshow(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                    ax.set_title(f'{cam} {label}\nsync 第 {sidx} 幀{det}', fontsize=8)
            plt.tight_layout(rect=(0, 0, 1, 0.92))
            pdf.savefig(fig)
            plt.close(fig)


def apply_crop(open_dir, rep, excluded):
    for subj in rep['subject'].unique():
        sdir = os.path.join(open_dir, subj)
        md, bak = os.path.join(sdir, 'MarkerData'), os.path.join(sdir, 'MarkerData_original')
        if not os.path.exists(bak):
            shutil.copytree(md, bak)
            print(f'💾 已備份 {md} -> {bak}')
    for _, r in rep.iterrows():
        if not isinstance(r.get('trc'), str) or not r['trc']:
            continue
        sdir = os.path.join(open_dir, r['subject'])
        tr = r['trc'][:-4]
        if local_reconstruction(sdir, tr):
            continue                                   # 本機重建檔：不裁切、不還原
        cur = os.path.join(sdir, 'MarkerData', r['trc'])
        bak = os.path.join(sdir, 'MarkerData_original', r['trc'])
        if os.path.exists(cur) and is_uncropped(cur) and os.path.exists(bak) and not filecmp.cmp(cur, bak, shallow=False):
            shutil.copy2(cur, bak)                     # 新的未裁切檔（例如 OpenCap 重跑）→ 更新備份
            print(f'💾 {r["subject"]} {r["trc"]} 是新的未裁切檔，已更新備份')
        aligned = str(r['status']).startswith('aligned')
        if aligned and (r['subject'], tr) not in excluded:
            n = int(round(r['emg_duration_s'] * FPS)) + 1
            crop_trc(bak, cur, int(r['trc_first_frame']), n)
            print(f'✂️  {r["subject"]:24} {r["trc"]:10} <- 肌電第 {int(r["emg_segment"]):2} 段  '
                  f'Frame# {int(r["trc_first_frame"])}..{int(r["trc_first_frame"]) + n - 1}'
                  f'（{(n - 1) / FPS:.3f} s / 肌電 {r["emg_duration_s"]:.3f} s）')
        elif os.path.exists(bak) and os.path.exists(cur) and not is_uncropped(cur):
            shutil.copy2(bak, cur)                     # 先前裁過、現在不該裁 → 還原
            print(f'↩️  {r["subject"]} {r["trc"]} 還原為未裁切的原檔')


def main():
    ap = argparse.ArgumentParser(description='用影片中的 LED 把 OpenCap TRC 裁到與肌電段落對齊')
    ap.add_argument('day', help='收案日資料夾（底下有 Open/ 與 Raw/）')
    ap.add_argument('--apply', action='store_true', help='實際裁切 MarkerData（預設只分析）')
    ap.add_argument('--exclude', nargs='*', default=[], metavar='資料夾:trial', help='不裁切的 trial')
    ap.add_argument('--map', nargs='*', default=[], metavar='資料夾=肌電檔', help='手動指定 OpenCap 資料夾對應的肌電檔')
    ap.add_argument('--min-trial-s', type=float, default=15.0, help='短於此秒數的肌電段落視為誤按（預設 15）')
    ap.add_argument('--no-pdf', action='store_true', help='不產生 alignment_frames.pdf')
    ap.add_argument('--start', nargs='*', default=[], metavar='資料夾:trial=幀',
                    help='人工逐幀確認過的起點（LED 第一次亮起的 sync 影片幀），覆寫自動偵測；配對仍照自動結果')
    args = ap.parse_args()
    starts = {}
    for s in args.start:
        k, v = s.rsplit('=', 1)
        starts[tuple(k.split(':', 1))] = int(v)

    day = os.path.abspath(args.day)
    open_dir, raw_dir = subdir(day, 'Open'), subdir(day, 'Raw')
    subjects = sorted(os.path.basename(p) for p in glob.glob(os.path.join(open_dir, '*')) if os.path.isdir(p))
    if not subjects:
        sys.exit(f'❌ {open_dir} 底下沒有 OpenCap 資料夾')
    manual = dict(m.split('=', 1) for m in args.map)
    excluded = {tuple(e.split(':', 1)) for e in args.exclude}
    raw_of = map_raw(raw_dir, subjects, manual)
    for s, r in raw_of.items():
        print(f'{s}  <->  {r}')
    cache = os.path.join(day, '_led_cache')
    os.makedirs(cache, exist_ok=True)

    rep, trials = analyze(open_dir, raw_dir, raw_of, cache, args.min_trial_s, excluded, starts)
    rep.to_csv(os.path.join(day, 'alignment_report.csv'), index=False, encoding='utf-8-sig')
    draw_signals(trials, os.path.join(day, 'alignment_check.png'))
    if not args.no_pdf:
        draw_frames_pdf(trials, os.path.join(day, 'alignment_frames.pdf'))

    pd.set_option('display.width', 240)
    cols = [c for c in ['subject', 'recorded_at', 'emg_segment', 'emg_duration_s', 'trc', 'trc_first_frame',
                        'duration_error_frames_Cam0', 'duration_error_frames_Cam1', 'status'] if c in rep]
    print('\n' + rep[cols].to_string(index=False))
    print('\n給 Features_insert_FIXED.py 的 SUBJECTS 設定：')
    for s in rep['subject'].unique():
        a = rep[(rep.subject == s) & rep['status'].astype(str).str.startswith('aligned')]
        print(f"  {s}: segment_ids={[int(x) for x in a['emg_segment']]}  trc_order={a['trc'].tolist()}")
    print(f'\n📄 {os.path.join(day, "alignment_report.csv")}\n🖼️  {os.path.join(day, "alignment_check.png")}')
    if not args.no_pdf:
        print(f'📑 {os.path.join(day, "alignment_frames.pdf")}')

    if args.apply:
        print()
        apply_crop(open_dir, rep, excluded)
    else:
        print('\n（只分析，沒有改任何檔案。確認 PDF 無誤後加 --apply 實際裁切。）')


if __name__ == '__main__':
    main()
