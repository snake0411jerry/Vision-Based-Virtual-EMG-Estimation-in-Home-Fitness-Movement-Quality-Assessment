"""
train_holdout_two.py — use the "_two re-recorded sets" as the validation set and plot measured vs predicted EMG waveforms
=========================================================================
[Setup]
  training set:   62 non-_two segments
  validation set: 12 _two segments (3 each from S02 / S04 / S06 / S08)

[⚠️ What this measures — must be stated clearly in the paper]
  S01 / S03 / S05 / S07 have no _two data and go entirely into the training set,
  and for the 4 subjects that do have _two, **their other segments are also in the training set**.
  So this split tests "**the same person, an unseen re-recorded set**"
  (within-subject, cross-trial), **not** cross-subject generalization.

  The question it answers is: "can the model reproduce the EMG waveform on a new attempt by the same user?"
  That is meaningful for the "same person tracking repeated training" use case, but must not be used to claim
  cross-person generalization — always cite LOSO for cross-person numbers (main 0.733 / synergist 0.524).

[Output]
  holdout_two_metrics.csv        r / MAE / nRMSE for each _two segment
  holdout_two_waveform_main.png  main muscle: measured vs predicted time-series waveforms (12 sets)
  holdout_two_waveform_syn.png   synergist: same
  holdout_two_scatter.png        predicted vs measured scatter + error distribution
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
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

FIXED_DIR = PIPELINE_DIR
sys.path.insert(0, FIXED_DIR)
import Features_insert_FIXED as F                              # noqa: E402
from eval_utils import build_ts_cols, make_windows, full_metrics  # noqa: E402

DATA_DIR = COMBINED_DIR
HERE = os.path.dirname(os.path.abspath(__file__))

FEATURE_MODE = "spatial"
WINDOW_SIZE = 40
STEP_TRAIN = 5
EPOCHS = 40
BATCH = 32
SEED = 42
FPS = 60.0

STATIC_COLS = ['Subj_Age', 'Subj_Height_m', 'Subj_Weight_norm', 'Subj_Gender']
LABEL_EMG_COLS = ['EMG_Main_MVC', 'EMG_Compass_MVC']
MUSCLE_NAMES = ['Main muscle (Main)', 'Synergist (Synergist)']


def two_segment_map():
    """Take the list of _two segments from the authoritative settings in Features_insert_FIXED."""
    out = {}
    for s in F.SUBJECTS:
        for trc, seg in zip(s['trc_order'], s['segment_ids']):
            if '_two' in trc.lower():
                out.setdefault(s['key'], []).append((seg, trc))
    return out


def seg_of(path):
    m = re.search(r'_Seg_(\d+)_', os.path.basename(path))
    return int(m.group(1)) if m else -1


def subj_of(path):
    return os.path.basename(path).split('_')[0]


def build_model(n_ts, n_static):
    from tensorflow.keras.layers import (Input, Dense, Dropout, Conv1D,
                                         BatchNormalization, GlobalAveragePooling1D,
                                         Concatenate)
    i_ts = Input(shape=(WINDOW_SIZE, n_ts), name='ts_input')
    i_st = Input(shape=(n_static,), name='static_input')
    x = Conv1D(64, 3, padding='causal', activation='relu', name='feature_conv1')(i_ts)
    x = BatchNormalization(name='feature_bn1')(x)
    x = Conv1D(64, 3, padding='causal', dilation_rate=2, activation='relu', name='feature_conv2')(x)
    f = GlobalAveragePooling1D(name='feature_pool')(x)
    z = Concatenate(name='fusion_concat')([f, i_st])
    z = Dense(64, activation='relu', name='fusion_dense')(z)
    z = Dropout(0.2, name='fusion_dropout')(z)
    z = Dense(32, activation='relu', name='emg_dense')(z)
    o = Dense(2, activation='sigmoid', name='out_emg')(z)
    m = tf.keras.Model([i_ts, i_st], [o])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss={'out_emg': 'mse'}, loss_weights={'out_emg': 3.0},
              metrics={'out_emg': 'mae'})
    return m


def dense_predict(model, df, ts_cols, s_ts, s_st):
    """Predict a single segment frame by frame with step=1 and return (frame_idx, y_true, y_pred)."""
    ts = s_ts.transform(df[ts_cols].values)
    st = s_st.transform(df[STATIC_COLS].values)
    y = df[LABEL_EMG_COLS].values

    X, S, idx = [], [], []
    for j in range(0, len(ts) - WINDOW_SIZE):
        X.append(ts[j:j + WINDOW_SIZE, :])
        t = j + WINDOW_SIZE - 1
        S.append(st[t, :])
        idx.append(t)
    if not X:
        return None
    X, S, idx = np.array(X), np.array(S), np.array(idx)
    p = model.predict({'ts_input': X, 'static_input': S}, verbose=0, batch_size=512)
    p = p[0] if isinstance(p, list) else p
    return idx, y[idx, :], p


def main():
    tf.keras.utils.set_random_seed(SEED)
    for f in ['Microsoft JhengHei', 'SimHei']:
        matplotlib.rcParams['font.sans-serif'] = [f]
        matplotlib.rcParams['axes.unicode_minus'] = False
        break

    tmap = two_segment_map()
    two_keys = {(s, seg) for s, lst in tmap.items() for seg, _ in lst}
    trc_of = {(s, seg): trc for s, lst in tmap.items() for seg, trc in lst}

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_Combined_Features.csv")))
    val_files = [p for p in files if (subj_of(p), seg_of(p)) in two_keys]
    tr_files = [p for p in files if (subj_of(p), seg_of(p)) not in two_keys]
    print(f"📂 training {len(tr_files)} files (non-_two) / validation {len(val_files)} files (_two)")
    print(f"   validation sets: {[(subj_of(p), seg_of(p)) for p in val_files]}")

    tr_dfs = [pd.read_csv(p) for p in tr_files]
    ts_cols = build_ts_cols(tr_dfs[0].columns.tolist(), FEATURE_MODE, False)

    big = pd.concat(tr_dfs, ignore_index=True)
    s_ts = StandardScaler().fit(big[ts_cols].values)
    s_st = StandardScaler().fit(big[STATIC_COLS].values)

    Xtr, Str, Ytr, _ = make_windows(tr_dfs, ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                    None, WINDOW_SIZE, STEP_TRAIN, s_ts, s_st)
    print(f"🧩 time-series features={len(ts_cols)}  training windows={len(Xtr)}")

    model = build_model(len(ts_cols), len(STATIC_COLS))
    model.fit({'ts_input': Xtr, 'static_input': Str}, {'out_emg': Ytr},
              epochs=EPOCHS, batch_size=BATCH, verbose=0)
    model.save(os.path.join(ABL_HOLDOUT_TWO_DIR, 'model_holdout_two.keras'))
    joblib.dump(s_ts, os.path.join(ABL_HOLDOUT_TWO_DIR, 'holdout_two_scaler_ts.pkl'))
    joblib.dump(s_st, os.path.join(ABL_HOLDOUT_TWO_DIR, 'holdout_two_scaler_static.pkl'))
    print("✅ Model training finished")

    # ---------- dense per-segment prediction ----------
    results, rows = [], []
    for p in sorted(val_files, key=lambda q: (subj_of(q), seg_of(q))):
        df = pd.read_csv(p)
        out = dense_predict(model, df, ts_cols, s_ts, s_st)
        if out is None:
            continue
        idx, yt, yp = out
        m = full_metrics(yt, yp)
        s, sg = subj_of(p), seg_of(p)
        results.append({'subj': s, 'seg': sg, 'trc': trc_of[(s, sg)],
                        't': idx / FPS, 'yt': yt, 'yp': yp})
        rows.append({'Subject': s, 'Segment': sg, 'TRC': trc_of[(s, sg)], 'n_frames': len(idx),
                     'r_main': m['r'][0], 'mae_main': m['mae'][0], 'nrmse_main': m['nrmse'][0],
                     'r_syn': m['r'][1], 'mae_syn': m['mae'][1], 'nrmse_syn': m['nrmse'][1]})
        print(f"   {s:<8} Seg{sg:<3} {trc_of[(s,sg)]:<12} "
              f"main r={m['r'][0]:.3f} nRMSE={m['nrmse'][0]:.3f} | "
              f"synergist r={m['r'][1]:.3f} nRMSE={m['nrmse'][1]:.3f}")

    mdf = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, 'holdout_two_metrics.csv')
    mdf.to_csv(out_csv, index=False, encoding='utf-8-sig')

    allt = np.concatenate([r['yt'] for r in results])
    allp = np.concatenate([r['yp'] for r in results])
    ov = full_metrics(allt, allp)
    print(f"\n{'='*64}\n📈 Summary over all _two validation sets ({len(allt)} frames)")
    for i, nm in enumerate(MUSCLE_NAMES):
        print(f"   {nm}: r={ov['r'][i]:.3f}  MAE={ov['mae'][i]:.3f}  nRMSE={ov['nrmse'][i]:.3f}")
    print(f"   per-segment mean  main r={mdf['r_main'].mean():.3f} / synergist r={mdf['r_syn'].mean():.3f}")

    # ---------- waveform plots ----------
    subjects = sorted({r['subj'] for r in results})
    for mi, mname in enumerate(MUSCLE_NAMES):
        fig, axes = plt.subplots(len(subjects), 3, figsize=(16, 3.0 * len(subjects)),
                                 squeeze=False)
        for si, s in enumerate(subjects):
            rs = [r for r in results if r['subj'] == s]
            rs.sort(key=lambda r: r['seg'])
            for ci in range(3):
                ax = axes[si][ci]
                if ci >= len(rs):
                    ax.axis('off'); continue
                r = rs[ci]
                ax.plot(r['t'], r['yt'][:, mi], lw=1.1, color='#1b1e1c', label='Measured EMG')
                ax.plot(r['t'], r['yp'][:, mi], lw=1.1, color='#e0623c', alpha=0.9, label='Model prediction')
                rr = np.corrcoef(r['yt'][:, mi], r['yp'][:, mi])[0, 1]
                ax.set_title(f"{s}  Seg{r['seg']} ({r['trc']})   r={rr:.3f}", fontsize=9)
                ax.set_xlabel('Time (s)', fontsize=8)
                ax.set_ylabel('Muscle activation', fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(alpha=0.2)
                if si == 0 and ci == 0:
                    ax.legend(fontsize=8, loc='upper right')
        fig.suptitle(f'{mname}: measured vs model prediction (validation set = _two re-recorded sets, never seen during training)',
                     fontsize=13)
        fig.tight_layout()
        tag = 'main' if mi == 0 else 'syn'
        pth = os.path.join(RESULTS_DIR, f'holdout_two_waveform_{tag}.png')
        fig.savefig(pth, dpi=140); plt.close(fig)
        print(f"📊 Saved {pth}")

    # ---------- scatter plot + error distribution ----------
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for mi, mname in enumerate(MUSCLE_NAMES):
        ax = axes[0][mi]
        ax.scatter(allt[:, mi], allp[:, mi], s=2, alpha=0.06, color='#0f766e')
        lim = [0, max(allt[:, mi].max(), allp[:, mi].max()) * 1.02]
        ax.plot(lim, lim, 'k--', lw=1, label='Ideal y=x')
        ax.set_xlabel('Measured muscle activation'); ax.set_ylabel('Predicted muscle activation')
        ax.set_title(f"{mname}  r={ov['r'][mi]:.3f}  nRMSE={ov['nrmse'][mi]:.3f}", fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.2)

        ax = axes[1][mi]
        err = allp[:, mi] - allt[:, mi]
        ax.hist(err, bins=80, color='#0f766e', alpha=0.75)
        ax.axvline(0, color='k', ls='--', lw=1)
        ax.axvline(err.mean(), color='crimson', lw=1.5, label=f'mean bias {err.mean():+.3f}')
        ax.set_xlabel('Predicted − measured (muscle activation)'); ax.set_ylabel('Frames')
        ax.set_title(f'{mname} error distribution', fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.2)
    fig.suptitle('Overall prediction quality on the _two validation sets', fontsize=13)
    fig.tight_layout()
    pth = os.path.join(RESULTS_DIR, 'holdout_two_scatter.png')
    fig.savefig(pth, dpi=140); plt.close(fig)
    print(f"📊 Saved {pth}")
    print(f"✅ Metrics saved to {out_csv}")


if __name__ == '__main__':
    main()
