# English summary: One-click supervised runner for load_moment_20.py (preflight, smoke test, workers with restart and stall watchdog, --stop-at HH:MM).
# (Added Nov 2026; the detailed comments below are in Chinese.)
"""
run_load20.py — 規劃三（load_moment_20.py）一鍵啟動與看管
==========================================================
換到另一台電腦時用：檢查環境 → 煙霧測試 → 開 N 個 worker → 自動重啟卡住或當掉的 worker
→ 全部跑完後自動彙整。可以中途關掉再重跑，已完成的 fold（results/load20/preds/*.npz）會自動跳過。

用法（在 Code 目錄）：
  python experiments/run_load20.py                 # 全流程（預設 5 個 worker）
  python experiments/run_load20.py --preflight-only
  python experiments/run_load20.py --workers 3     # GPU 記憶體不夠時減少
  python experiments/run_load20.py --no-smoke      # 已經測過時略過煙霧測試

需要的資料（放在 Code 的上一層 Dataset/，或用環境變數 SEMG_DATASET 指定）：
  Dataset/Combined_20/*.csv   Dataset/Skeleton_20/*.npz
輸出：results/load20/  （run_w*.log、supervisor_events.log、load20_summary.csv、load20_tests.csv）
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)
from paths import DATASET_DIR, RESULTS_DIR  # noqa: E402

SCRIPT = os.path.join(HERE, 'load_moment_20.py')
OUT = os.path.join(RESULTS_DIR, 'load20')
PREDS = os.path.join(OUT, 'preds')
EVENTS = os.path.join(OUT, 'supervisor_events.log')
TOTAL = 300                       # 3 arms × 5 seeds × 20 人
CHILD_ENV = dict(os.environ, TF_FORCE_GPU_ALLOW_GROWTH='true', TF_CPP_MIN_LOG_LEVEL='3',
                 PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')


def now():
    return datetime.now().strftime('%m/%d %H:%M')


def event(msg):
    os.makedirs(OUT, exist_ok=True)
    line = f'{now()} {msg}'
    print(line, flush=True)
    with open(EVENTS, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def n_done():
    return len([p for p in glob.glob(os.path.join(PREDS, '*.npz')) if not p.endswith('.tmp.npz')])


def keep_awake(on=True):
    """Windows：執行期間要求系統不要進入睡眠（程式結束即自動解除，不改電源設定）"""
    if os.name != 'nt':
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


# ---------------------------------------------------------------- 1. 環境檢查
def preflight(py):
    ok = True
    print(f'Python：{py}')
    print(f'資料夾：{DATASET_DIR}')
    n_csv = len(glob.glob(os.path.join(DATASET_DIR, 'Combined_20', '*_Combined_Features.csv')))
    n_npz = len(glob.glob(os.path.join(DATASET_DIR, 'Skeleton_20', '*.npz')))
    print(f'  Combined_20：{n_csv} 個 csv（原機 157）   Skeleton_20：{n_npz} 個 npz（原機 157）')
    if n_csv < 157 or n_npz < 157:
        print('  ❌ 資料不完整：請把 Dataset/Combined_20 與 Dataset/Skeleton_20 整個複製過來，'
              '放在 Code 的上一層，或設定環境變數 SEMG_DATASET')
        ok = False
    free_gb = shutil.disk_usage(CODE).free / 1e9
    print(f'  磁碟剩餘 {free_gb:.0f} GB（結果約需 1 GB）')
    probe = ("import sys, tensorflow as tf, sklearn, scipy, pandas, numpy;"
             "g = tf.config.list_physical_devices('GPU');"
             "print('versions', sys.version.split()[0], tf.__version__, sklearn.__version__, pandas.__version__, numpy.__version__);"
             "print('gpus', len(g));"
             "import time; t = time.time();"
             "x = tf.random.normal((2048, 2048)); y = tf.reduce_sum(tf.matmul(x, x)).numpy();"
             "print('matmul_s', round(time.time() - t, 1), 'device', x.device)")
    print('  測試 TensorFlow 與 GPU（第一次在新顯卡上可能要編譯幾分鐘）…', flush=True)
    try:
        r = subprocess.run([py, '-c', probe], capture_output=True, text=True, encoding='utf-8',
                           errors='replace', env=CHILD_ENV, timeout=3600)
        out = {ln.split()[0]: ln.split()[1:] for ln in r.stdout.splitlines() if ln.strip()}
        if 'versions' not in out:
            print('  ❌ 無法匯入套件：\n' + r.stderr[-1500:])
            return False
        v = out['versions']
        print(f'  Python {v[0]}  TensorFlow {v[1]}  scikit-learn {v[2]}  pandas {v[3]}  numpy {v[4]}'
              '（原機：3.10 / 2.10.0 / 1.7.2 / 2.3.3 / 1.26.4）')
        gpus = int(out.get('gpus', ['0'])[0])
        if gpus == 0:
            print('  ⚠ TensorFlow 看不到 GPU，會用 CPU 跑，時間可能是好幾倍。'
                  'Windows 原生 GPU 只支援到 TensorFlow 2.10（需 CUDA 11.2 + cuDNN 8.1）')
        else:
            print(f'  ✅ GPU {gpus} 張，矩陣測試 {out["matmul_s"][0]} 秒（{out["matmul_s"][-1]}）')
    except subprocess.TimeoutExpired:
        print('  ❌ GPU 測試超過 60 分鐘沒有回應')
        return False
    try:
        smi = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=30)
        if smi.returncode == 0:
            print(f'  顯卡：{smi.stdout.strip()}（5 個 worker 約各用 1–2 GB）')
    except (OSError, subprocess.TimeoutExpired):
        pass
    print(f'  已完成 fold：{n_done()} / {TOTAL}')
    return ok


# ---------------------------------------------------------------- 2. 煙霧測試
def smoke(py):
    print('\n▶ 煙霧測試：S11 三種輸入各訓練 1 epoch（約 5–15 分鐘）', flush=True)
    log = os.path.join(OUT, 'smoke.log')
    with open(log, 'w', encoding='utf-8') as f:
        r = subprocess.run([py, SCRIPT, '--smoke'], cwd=CODE, env=CHILD_ENV, stdout=f, stderr=subprocess.STDOUT)
    tail = open(log, encoding='utf-8', errors='replace').read()[-1200:]
    print(tail)
    if r.returncode != 0 or 'worker 完成' not in tail:
        print(f'❌ 煙霧測試失敗，完整訊息在 {log}')
        return False
    return True


# ---------------------------------------------------------------- 3. 看管 worker
class Worker:
    def __init__(self, k, n, py):
        self.k, self.n, self.py = k, n, py
        self.log = os.path.join(OUT, f'run_w{k}.log')
        self.proc, self.restarts, self.done, self.gave_up = None, 0, False, False

    def start(self):
        f = open(self.log, 'a', encoding='utf-8')
        f.write(f'\n===== {now()} 啟動（第 {self.restarts} 次重啟）=====\n')
        f.flush()
        self.proc = subprocess.Popen([self.py, SCRIPT, '--worker', str(self.k), '--n-workers', str(self.n)],
                                     cwd=CODE, env=CHILD_ENV, stdout=f, stderr=subprocess.STDOUT)
        self.started = time.time()

    def idle_min(self):
        return (time.time() - max(os.path.getmtime(self.log), self.started)) / 60

    def tail(self, n=400):
        with open(self.log, encoding='utf-8', errors='replace') as f:
            return f.read()[-n:]


def parse_stop_at(s):
    """'HH:MM' → 下一個該時刻的 epoch 秒（今天已過就算明天）"""
    from datetime import timedelta
    hh, mm = map(int, s.split(':'))
    t = datetime.now().replace(hour=hh, minute=mm, second=0, microsecond=0)
    if t <= datetime.now():
        t += timedelta(days=1)
    return t.timestamp(), t.strftime('%m/%d %H:%M')


def stop_all(ws):
    for w in ws:
        if w.proc and w.proc.poll() is None:
            w.proc.kill()
            w.proc.wait()


def supervise(py, n, stall_min, max_restarts, deadline=None):
    """回傳 True = 全部完成；False = 有 worker 放棄；None = 到達預定停止時間"""
    ws = [Worker(k, n, py) for k in range(n)]
    for w in ws:
        w.start()
        time.sleep(20)                       # 錯開載入資料，避免同時搶記憶體
    event(f'啟動 {n} 個 worker（已完成 {n_done()} / {TOTAL}）')
    t0, done0, last_report = time.time(), n_done(), 0
    try:
        while not all(w.done or w.gave_up for w in ws):
            time.sleep(60)
            if deadline and time.time() >= deadline:
                stop_all(ws)
                event(f'到達預定停止時間，已停止所有 worker（已完成 {n_done()} / {TOTAL}；'
                      '進行中的 fold 下次重跑，其餘從斷點接續）')
                return None
            for w in ws:
                if w.done or w.gave_up:
                    continue
                code = w.proc.poll()
                reason = None
                idle = w.idle_min()
                if code is None and idle >= stall_min:
                    w.proc.kill()
                    w.proc.wait()
                    reason = f'log 已 {idle:.0f} 分鐘沒有更新，判定卡住'
                elif code == 0 and 'worker 完成' in w.tail():
                    w.done = True
                    event(f'w{w.k} 完成')
                    continue
                elif code is not None:
                    reason = f'結束碼 {code}：…{w.tail(300).strip()[-300:]}'
                if reason:
                    if w.restarts >= max_restarts:
                        w.gave_up = True
                        event(f'w{w.k} 放棄（已重啟 {max_restarts} 次）。{reason}')
                    else:
                        w.restarts += 1
                        event(f'w{w.k} 重啟 #{w.restarts}：{reason}')
                        w.start()
            if time.time() - last_report >= 600:
                last_report = time.time()
                d = n_done()
                rate = (d - done0) / max((time.time() - t0) / 3600, 1e-6)
                eta = f'，預估還要 {(TOTAL - d) / rate:.1f} 小時' if d > done0 else ''
                print(f'{now()} 進度 {d} / {TOTAL}{eta}', flush=True)
    except KeyboardInterrupt:
        event('收到 Ctrl+C，停止所有 worker（已完成的 fold 會保留，重跑會接續）')
        stop_all(ws)
        raise
    event(f'全部結束：{n_done()} / {TOTAL}')
    return all(w.done for w in ws)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--python', default=sys.executable, help='跑訓練用的 Python（預設：目前這個）')
    ap.add_argument('--preflight-only', action='store_true')
    ap.add_argument('--no-smoke', action='store_true')
    ap.add_argument('--stall-min', type=float, default=80, help='log 幾分鐘沒更新就判定卡住')
    ap.add_argument('--max-restarts', type=int, default=6)
    ap.add_argument('--stop-at', help='HH:MM，到這個時刻自動停止（例如要換地方前），之後重跑會接續')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    deadline = None
    if args.stop_at:
        deadline, label = parse_stop_at(args.stop_at)
        event(f'預定 {label} 自動停止')

    print('▶ 環境檢查')
    if not preflight(args.python):
        sys.exit(1)
    if args.preflight_only:
        return
    keep_awake(True)
    try:
        if n_done() < TOTAL:
            if not args.no_smoke and n_done() == 0 and not smoke(args.python):
                sys.exit(1)
            print(f'\n▶ 開始訓練：{args.workers} 個 worker，log 在 {OUT}\\run_w*.log；'
                  '可以關掉這個視窗中斷，之後重跑會從斷點接續', flush=True)
            if supervise(args.python, args.workers, args.stall_min, args.max_restarts, deadline) is None:
                print(f'⏸ 已在預定時間停止（{n_done()} / {TOTAL}）。換好地方後再執行一次即可接續。')
                return
        print('\n▶ 彙整結果', flush=True)
        subprocess.run([args.python, SCRIPT, '--summarize-only'], cwd=CODE, env=CHILD_ENV)
        if n_done() < TOTAL:
            print(f'⚠ 只完成 {n_done()} / {TOTAL}，以上是部分結果；看 {EVENTS} 找原因後重跑即可接續')
        else:
            print(f'✅ 全部完成。結果：{OUT}\\load20_summary.csv、load20_tests.csv')
    finally:
        keep_awake(False)


if __name__ == '__main__':
    main()
