"""
finetune_predict_two.py — after personalized fine-tuning, predict the "_two re-recorded sets"
=========================================================================
[Design] For every subject S with _two data (S02 / S04 / S06 / S08):

  start     loso_without_S.keras — trained with S fully excluded (not even _two)
  fine-tune only S's "non-_two" segments
  test      S's _two segments — never seen by the starting model or the fine-tuning data

  ⟹ Both the zero-shot and fine-tuned curves are evaluated on completely unseen _two data and are directly comparable.

[What question this answers]
  "If a new user records a few sets for calibration and comes back later, does the model predict well?"
  This is the real deployment scenario: calibrate once, then use it every session.

[Difference from existing experiments]
  train_holdout_two.py uses a **global model** trained on "all non-_two" data,
  with no personalization step, and that model has seen S's non-_two data. This script's starting point
  has never seen S at all, so the "zero-shot → fine-tuned" gain is clean.

Output:
  two_finetuned_waveform_main.png / _syn.png   per-segment waveforms
  two_finetuned_metrics.csv                    per-segment metrics
"""
# --- Paths: always managed centrally by paths.py at the repo root; do not revert to absolute paths ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import (  # noqa: E402,F401
    REPO_ROOT, PIPELINE_DIR, EXPERIMENTS_DIR, RESULTS_DIR,
    DATASET_DIR, RAW_DIR, CLEAN_DIR, CLEAN_60FPS_DIR, CLEAN_1000HZ_DIR,
    COMBINED_DIR, COMBINED_BEFORE_AXISFIX_DIR, OPEN_DIR, SKELETON_DIR,
    GLOBAL_MODEL_DIR, LOSO_MODEL_DIR,
    PERSONALIZED_FROM_LOSO_DIR, PERSONALIZED_NONTWO_DIR, PERSONALIZED_FROM_GLOBAL_DIR,
    ABL_BEFORE_AXISFIX_DIR, ABL_MASKED_DIR, ABL_EXCLUDE_S09_DIR, ABL_HOLDOUT_TWO_DIR,
    SUBJECTS_CSV, SUBJECTS_EXAMPLE_CSV,
    ensure_results_dir, require_dataset, require_subjects_csv,
)
# --- End of path setup ---


import os
import re
import sys
import glob

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

FIXED_DIR = PIPELINE_DIR
sys.path.insert(0, FIXED_DIR)
import Features_insert_FIXED as F                                  # noqa: E402
from eval_utils import make_windows, full_metrics                  # noqa: E402

DATA_DIR = COMBINED_DIR
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = LOSO_MODEL_DIR          # per-fold LOSO models + scalers
WINDOW_SIZE, STEP_TRAIN = 40, 5
FT_EPOCHS, FT_BATCH, FT_LR = 20, 16, 1e-5
SEED = 42
FPS = 60.0
MAX_SEC = 45          # at most 45 seconds per panel

STATIC_COLS = ['Subj_Age', 'Subj_Height_m', 'Subj_Weight_norm', 'Subj_Gender']
LABEL_EMG_COLS = ['EMG_Main_MVC', 'EMG_Compass_MVC']
MUSCLE = ['Main muscle (Main)', 'Synergist (Synergist)']
C_TRUE, C_ZS, C_FT = '#15181a', '#e0623c', '#0f766e'


def two_map():
    """{subject: [(segment_id, trc_name), ...]} — _two files only."""
    out = {}
    for s in F.SUBJECTS:
        for trc, seg in zip(s['trc_order'], s['segment_ids']):
            if '_two' in trc.lower():
                out.setdefault(s['key'], []).append((seg, trc))
    return out


def seg_of(p):
    m = re.search(r'_Seg_(\d+)_', os.path.basename(p))
    return int(m.group(1)) if m else -1


def dense_predict(model, df, ts_cols, s_ts, s_st):
    ts = s_ts.transform(df[ts_cols].values)
    st = s_st.transform(df[STATIC_COLS].values)
    y = df[LABEL_EMG_COLS].values
    X, S, idx = [], [], []
    for j in range(0, len(ts) - WINDOW_SIZE):
        X.append(ts[j:j + WINDOW_SIZE, :])
        t = j + WINDOW_SIZE - 1
        S.append(st[t, :]); idx.append(t)
    if not X:
        return None
    idx = np.array(idx)
    p = model.predict({'ts_input': np.array(X), 'static_input': np.array(S)},
                      verbose=0, batch_size=512)
    p = p[0] if isinstance(p, list) else p
    return idx, y[idx, :], p


def main():
    tf.keras.utils.set_random_seed(SEED)
    for f in ['Microsoft JhengHei', 'SimHei']:
        matplotlib.rcParams['font.sans-serif'] = [f]
        matplotlib.rcParams['axes.unicode_minus'] = False
        break

    spec = joblib.load(os.path.join(MODEL_DIR, 'feature_spec.pkl'))
    ts_cols = spec['ts_cols']
    tmap = two_map()
    print(f"👥 subjects with _two data: {sorted(tmap)}")
    for k, v in sorted(tmap.items()):
        print(f"   {k}: {[f'Seg{s}({t})' for s, t in v]}")

    results, rows = [], []
    for subj in sorted(tmap):
        two_segs = {s for s, _ in tmap[subj]}
        trc_of = {s: t for s, t in tmap[subj]}
        files = sorted(glob.glob(os.path.join(DATA_DIR, f"{subj}_Seg_*.csv")), key=seg_of)
        ft_files = [p for p in files if seg_of(p) not in two_segs]     # for fine-tuning
        te_files = [p for p in files if seg_of(p) in two_segs]         # for testing

        base = os.path.join(MODEL_DIR, f'loso_without_{subj}.keras')
        s_ts = joblib.load(os.path.join(MODEL_DIR, f'loso_scaler_ts_{subj}.pkl'))
        s_st = joblib.load(os.path.join(MODEL_DIR, f'loso_scaler_static_{subj}.pkl'))

        print(f"\n{'='*66}\n{subj} | fine-tuning {len(ft_files)} segments (non-_two) / test {len(te_files)} segments (_two)\n{'='*66}")

        ft_dfs = [pd.read_csv(p) for p in ft_files]
        te_dfs = [pd.read_csv(p) for p in te_files]

        Xf, Sf, Yf, _ = make_windows(ft_dfs, ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                     None, WINDOW_SIZE, STEP_TRAIN, s_ts, s_st)

        # ---- zero-shot prediction (starting model, has never seen subj) ----
        model = tf.keras.models.load_model(base)
        zs = [dense_predict(model, d, ts_cols, s_ts, s_st) for d in te_dfs]

        # ---- personalized fine-tuning: non-_two data only ----
        for layer in model.layers:
            layer.trainable = ('feature' not in layer.name)
        model.compile(optimizer=tf.keras.optimizers.Adam(FT_LR),
                      loss={'out_emg': 'mse'}, loss_weights={'out_emg': 3.0},
                      metrics={'out_emg': 'mae'})
        model.fit({'ts_input': Xf, 'static_input': Sf}, {'out_emg': Yf},
                  epochs=FT_EPOCHS, batch_size=FT_BATCH, verbose=0)
        model.save(os.path.join(PERSONALIZED_NONTWO_DIR, f'{subj}_personalized_nontwo.keras'))

        ft = [dense_predict(model, d, ts_cols, s_ts, s_st) for d in te_dfs]
        tf.keras.backend.clear_session()

        for p, a, b in zip(te_files, zs, ft):
            if a is None or b is None:
                continue
            sg = seg_of(p)
            idx, yt, ypz = a
            _, _, ypf = b
            mz, mf = full_metrics(yt, ypz), full_metrics(yt, ypf)
            results.append({'subj': subj, 'seg': sg, 'trc': trc_of[sg],
                            't': idx / FPS, 'y': yt, 'zs': ypz, 'ft': ypf})
            row = {'Subject': subj, 'Segment': sg, 'TRC': trc_of[sg], 'n_frames': len(idx)}
            for i, mu in enumerate(['main', 'syn']):
                row[f'r_{mu}_zs'] = mz['r'][i];      row[f'r_{mu}_ft'] = mf['r'][i]
                row[f'nrmse_{mu}_zs'] = mz['nrmse'][i]; row[f'nrmse_{mu}_ft'] = mf['nrmse'][i]
            rows.append(row)
            print(f"   Seg{sg:<3}{trc_of[sg]:<12} main r {mz['r'][0]:.3f}→{mf['r'][0]:.3f}"
                  f"  nRMSE {mz['nrmse'][0]:.3f}→{mf['nrmse'][0]:.3f}"
                  f" | synergist r {mz['r'][1]:.3f}→{mf['r'][1]:.3f}"
                  f"  nRMSE {mz['nrmse'][1]:.3f}→{mf['nrmse'][1]:.3f}")

    mdf = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, 'two_finetuned_metrics.csv')
    mdf.to_csv(out_csv, index=False, encoding='utf-8-sig')

    # ---------- waveform plots: per segment ----------
    subjects = sorted({r['subj'] for r in results})
    for mi, mname in enumerate(MUSCLE):
        fig, axes = plt.subplots(len(subjects), 3,
                                 figsize=(17, 2.9 * len(subjects)), squeeze=False)
        for si, s in enumerate(subjects):
            rs = sorted([r for r in results if r['subj'] == s], key=lambda r: r['seg'])
            for ci in range(3):
                ax = axes[si][ci]
                if ci >= len(rs):
                    ax.axis('off'); continue
                r = rs[ci]
                n = min(len(r['y']), int(MAX_SEC * FPS))
                t = r['t'][:n] - r['t'][0]
                ax.plot(t, r['y'][:n, mi] * 100, lw=1.2, color=C_TRUE, label='Measured EMG', zorder=3)
                ax.plot(t, r['zs'][:n, mi] * 100, lw=1.2, color=C_ZS, alpha=.85,
                        label='Zero-shot (never seen this person)', zorder=2)
                ax.plot(t, r['ft'][:n, mi] * 100, lw=1.2, color=C_FT, alpha=.85,
                        label='Fine-tuned (calibrated on non-_two)', zorder=2)
                rz = np.corrcoef(r['y'][:, mi], r['zs'][:, mi])[0, 1]
                rf = np.corrcoef(r['y'][:, mi], r['ft'][:, mi])[0, 1]
                ax.set_title(f"{s} · Seg{r['seg']} ({r['trc']})   r {rz:.3f} → {rf:.3f}",
                             fontsize=9.5)
                ax.set_ylabel('%MVC', fontsize=8.5)
                ax.tick_params(labelsize=7.5)
                ax.grid(alpha=.22)
                if si == len(subjects) - 1:
                    ax.set_xlabel('Time (s)', fontsize=8.5)
                if si == 0 and ci == 0:
                    ax.legend(fontsize=8, loc='upper right')
        fig.suptitle(f'{mname}: predicted vs measured on the _two re-recorded sets\n'
                     f'The starting model has never seen this subject; fine-tuning uses only their “non-_two” data — _two is brand-new unseen data',
                     fontsize=13, fontweight='bold')
        fig.tight_layout()
        tag = 'main' if mi == 0 else 'syn'
        p = os.path.join(RESULTS_DIR, f'two_finetuned_waveform_{tag}.png')
        fig.savefig(p, dpi=140); plt.close(fig)
        print(f"\n📊 Saved {p}")

    # ---------- summary ----------
    allt = np.concatenate([r['y'] for r in results])
    allz = np.concatenate([r['zs'] for r in results])
    allf = np.concatenate([r['ft'] for r in results])
    mz, mf = full_metrics(allt, allz), full_metrics(allt, allf)
    print(f"\n{'='*66}\n📈 Summary over all _two ({len(allt)} frames)")
    for i, nm in enumerate(MUSCLE):
        print(f"   {nm}: r {mz['r'][i]:.3f} → {mf['r'][i]:.3f}"
              f"   nRMSE {mz['nrmse'][i]:.3f} → {mf['nrmse'][i]:.3f}")
    print(f"\n   per-segment mean  main r {mdf['r_main_zs'].mean():.3f} → {mdf['r_main_ft'].mean():.3f}"
          f" | synergist r {mdf['r_syn_zs'].mean():.3f} → {mdf['r_syn_ft'].mean():.3f}")
    print(f"✅ Metrics saved to {out_csv}")


if __name__ == '__main__':
    main()
