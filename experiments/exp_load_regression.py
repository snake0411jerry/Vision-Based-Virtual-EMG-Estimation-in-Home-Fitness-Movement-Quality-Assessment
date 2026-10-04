"""
exp_load_regression.py — load recognition (%1RM regression)
=========================================================================
Corresponds to "research question 2.1" and "experiment B" of the NSTC proposal:
  "If spatio-temporal features such as velocity and acceleration are introduced into a lightweight
    architecture, can they improve discrimination of 'similar-looking trajectories under different loads'"

★ Why we cannot simply reuse the conclusion of Exp C:
  Exp C tested "does adding v/a help **EMG regression**" → the answer was not at all (p=0.99).
  But the proposal claims "adding v/a helps **load recognition**" (citing Zhu 2025's load classification).
  **These are two different tasks** and must be tested separately.

-------------------------------------------------------------------------
Data design (confirmed with the user on 2026-08-02)
-------------------------------------------------------------------------
• Exclude the a/b/c compensation segments (35 segments).
  They are bodyweight movements with **deliberately altered trajectories**, unrelated to load;
  mixing them in would teach the model "odd trajectory = light load".
  → 54 usable segments (14 bodyweight 0kg + 40 loaded)

• The target is **%1RM** (load kg ÷ that subject's 1RM), not absolute kilograms.
  Absolute weight is not comparable across subjects (S02 max 40kg vs S10 max 120kg),
  whereas %1RM is the physiological quantity that determines muscle activation.

• Use **continuous regression** rather than light/moderate/heavy classification, because:
    1. The proposal itself defines no light/moderate/heavy boundaries, so thresholds would be arbitrary
    2. Each person's heaviest load measured at 56%~75% 1RM (median 65%);
       a fixed 70% cut would slice right through "each person's heaviest set"
    3. Some subjects lack certain levels (S01/S03/S07 have no segment >60%),
       so classification would leave those LOSO test folds missing classes
  → Classification accuracy is instead derived by "binning the predictions after the fact", satisfying both needs at once.

-------------------------------------------------------------------------
Extra diagnosis: predictions on the compensation segments
-------------------------------------------------------------------------
After training, predict on the **35 excluded compensation segments** (all 0kg bodyweight).
If the model predicts them as heavy loads, it has learned "movement variation" rather than "load" —
direct counter-evidence to the proposal's "visual ambiguity" claim. The cost is almost zero.

Output: exp_load_{raw,segment,comp,summary}.csv
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


import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
FIXED_DIR = PIPELINE_DIR
sys.path.insert(0, FIXED_DIR)
sys.path.insert(0, HERE)

from eval_utils import (subject_id_from_path, build_ts_cols, make_windows)   # noqa: E402
from loso_train_and_save import (FEATURE_MODE, WINDOW_SIZE, STEP_SIZE,       # noqa: E402
                                 EPOCHS, BATCH_SIZE, STATIC_COLS, DATA_DIR)
import Features_insert_FIXED as F                                            # noqa: E402

# four-level bins (only for deriving classification metrics afterwards; training is still continuous regression)
BINS = [-0.001, 0.0001, 0.35, 0.60, 10.0]
BIN_NAMES = ['Bodyweight (0%)', 'Light (<35%)', 'Moderate (35-60%)', 'Heavy (>60%)']


def build_load_map():
    """(subject, seg_id) -> (%1RM, whether it is a compensation segment)."""
    m = {}
    for s in F.SUBJECTS:
        e = F.SUBJECT_TABLE.get(s['key'])
        if not e:
            continue
        rm = e['info']['Squat_1RM_kg']
        for trc, sid in zip(s['trc_order'], s['segment_ids']):
            base = os.path.splitext(trc)[0]
            is_comp = base[0].lower() in 'abc'
            m[(s['key'], sid)] = (F.derive_load_kg(trc) / rm, is_comp, trc)
    return m


def build_model(n_ts_feat, n_static, lr=1e-3):
    """Same architecture as loso_train_and_save.build_model, except the output is a single %1RM."""
    from tensorflow.keras.layers import (Input, Dense, Dropout, Conv1D,
                                         BatchNormalization,
                                         GlobalAveragePooling1D, Concatenate)
    input_ts = Input(shape=(WINDOW_SIZE, n_ts_feat), name='ts_input')
    input_static = Input(shape=(n_static,), name='static_input')
    x = Conv1D(64, 3, padding='causal', activation='relu', name='feature_conv1')(input_ts)
    x = BatchNormalization(name='feature_bn1')(x)
    x = Conv1D(64, 3, padding='causal', dilation_rate=2, activation='relu',
               name='feature_conv2')(x)
    ts_feat = GlobalAveragePooling1D(name='feature_pool')(x)
    fused = Concatenate(name='fusion_concat')([ts_feat, input_static])
    fused = Dense(64, activation='relu', name='fusion_dense')(fused)
    fused = Dropout(0.2, name='fusion_dropout')(fused)
    h = Dense(32, activation='relu', name='load_dense')(fused)
    out = Dense(1, activation='sigmoid', name='out_load')(h)
    model = tf.keras.Model(inputs=[input_ts, input_static], outputs=[out])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse', metrics=['mae'])
    return model


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-8 or b.std() < 1e-8:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser(description="Load recognition (%1RM regression)")
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 1, 2, 3, 4])
    ap.add_argument("--feature-mode", default=FEATURE_MODE,
                    choices=['spatial', 'spatial+kinematic'])
    ap.add_argument("--prefix", default="exp_load")
    args = ap.parse_args()

    lm = build_load_map()
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_Combined_Features.csv")))

    load_files, comp_files = [], []
    for p in all_files:
        subj = subject_id_from_path(p)
        sid = int(os.path.basename(p).split('_')[2])
        info = lm.get((subj, sid))
        if info is None:
            print(f"⚠️ No load information for {subj} Seg{sid}, skipping")
            continue
        (comp_files if info[1] else load_files).append((p, subj, sid, info[0], info[2]))

    dfs = {t[0]: pd.read_csv(t[0]) for t in load_files + comp_files}
    ts_cols = build_ts_cols(list(dfs[load_files[0][0]].columns), args.feature_mode, False)
    subjects = sorted({t[1] for t in load_files})

    print(f"📂 {len(load_files)} loaded segments / {len(comp_files)} compensation segments (diagnosis only, not trained on)")
    print(f"🧩 features {len(ts_cols)} dims ({args.feature_mode}) / {len(subjects)} subjects")
    y = np.array([t[3] for t in load_files])
    print(f"🎯 %1RM target: min {y.min():.2f} median {np.median(y):.2f} max {y.max():.2f}")

    win_rows, seg_rows, comp_rows = [], [], []
    t0 = time.time()
    for si, seed in enumerate(args.seeds, 1):
        tf.keras.utils.set_random_seed(seed)
        for hold in subjects:
            tr = [t for t in load_files if t[1] != hold]
            va = [t for t in load_files if t[1] == hold]

            big = pd.concat([dfs[t[0]] for t in tr], ignore_index=True)
            s_ts = StandardScaler().fit(big[ts_cols].values)
            s_st = StandardScaler().fit(big[STATIC_COLS].values)

            def windows(items):
                X, S, Y, seg = [], [], [], []
                for p, subj, sid, pct, trc in items:
                    d = dfs[p]
                    xs, ss, _, _ = make_windows([d], ts_cols, STATIC_COLS,
                                                ['EMG_Main_MVC', 'EMG_Compass_MVC'],
                                                None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
                    if len(xs) == 0:
                        continue
                    X.append(xs); S.append(ss)
                    Y.append(np.full(len(xs), pct))
                    seg += [(subj, sid, trc, pct)] * len(xs)
                return (np.concatenate(X), np.concatenate(S), np.concatenate(Y), seg)

            Xtr, Str, Ytr, _ = windows(tr)
            Xva, Sva, Yva, seg_va = windows(va)

            model = build_model(len(ts_cols), len(STATIC_COLS))
            model.fit({'ts_input': Xtr, 'static_input': Str}, Ytr,
                      epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
            pv = model.predict({'ts_input': Xva, 'static_input': Sva}, verbose=0).ravel()

            win_rows.append({'seed': seed, 'Subject': hold, 'n_win': len(Yva),
                             'r': pearson(Yva, pv),
                             'mae': float(np.mean(np.abs(pv - Yva))),
                             'rmse': float(np.sqrt(np.mean((pv - Yva) ** 2)))})

            # per-segment summary (median of window predictions represents the segment) — deployment judges a whole set, not single frames
            tmp = pd.DataFrame(seg_va, columns=['Subject', 'seg', 'trc', 'true_pct'])
            tmp['pred'] = pv
            g = tmp.groupby(['Subject', 'seg', 'trc', 'true_pct'], as_index=False)['pred'].median()
            g['seed'] = seed
            seg_rows.append(g)

            # compensation-segment diagnosis (this subject's compensation segments, predicted by the same fold's model)
            cva = [t for t in comp_files if t[1] == hold]
            if cva:
                Xc, Sc, _, seg_c = windows(cva)
                pc = model.predict({'ts_input': Xc, 'static_input': Sc}, verbose=0).ravel()
                tc = pd.DataFrame(seg_c, columns=['Subject', 'seg', 'trc', 'true_pct'])
                tc['pred'] = pc
                gc = tc.groupby(['Subject', 'seg', 'trc'], as_index=False)['pred'].median()
                gc['seed'] = seed
                comp_rows.append(gc)

            tf.keras.backend.clear_session()

        pd.DataFrame(win_rows).to_csv(os.path.join(RESULTS_DIR, f'{args.prefix}_raw.csv'),
                                      index=False, encoding='utf-8-sig')
        pd.concat(seg_rows, ignore_index=True).to_csv(
            os.path.join(RESULTS_DIR, f'{args.prefix}_segment.csv'), index=False, encoding='utf-8-sig')
        if comp_rows:
            pd.concat(comp_rows, ignore_index=True).to_csv(
                os.path.join(RESULTS_DIR, f'{args.prefix}_comp.csv'), index=False, encoding='utf-8-sig')
        d = pd.DataFrame(win_rows)
        d = d[d.seed == seed]
        el = (time.time() - t0) / si
        print(f"[{si}/{len(args.seeds)}] seed={seed}  per-window r={d.r.mean():.3f}  "
              f"MAE={d.mae.mean():.3f} (%1RM)  "
              f"({el/60:.1f} min/seed, about {el*(len(args.seeds)-si)/60:.0f} min left)", flush=True)

    # ---------------- summary ----------------
    W = pd.DataFrame(win_rows)
    S = pd.concat(seg_rows, ignore_index=True)
    per = W.groupby('seed')[['r', 'mae', 'rmse']].mean()

    print("\n" + "=" * 84)
    print(f"Load recognition results ({len(args.seeds)} seeds, LOSO)")
    print("=" * 84)
    print("\n[Per window]")
    for c in per.columns:
        print(f"  {c:<6} {per[c].mean():.4f} ± {per[c].std(ddof=1):.4f}")

    print("\n[Per segment] (median of window predictions represents the whole segment; closer to real deployment)")
    seg_r, seg_mae, seg_acc, within_r = [], [], [], []
    for sd, g in S.groupby('seed'):
        seg_r.append(pearson(g.true_pct, g.pred))
        seg_mae.append(float(np.mean(np.abs(g.pred - g.true_pct))))
        tb = pd.cut(g.true_pct, BINS, labels=BIN_NAMES)
        pb = pd.cut(g.pred, BINS, labels=BIN_NAMES)
        seg_acc.append(float((tb == pb).mean()))
        # ★ within-subject r — this is what the proposal asks: can the same person's different loads be discriminated
        rs = [pearson(gg.true_pct, gg.pred) for _, gg in g.groupby('Subject') if len(gg) >= 3]
        within_r.append(float(np.nanmean(rs)) if rs else np.nan)
    base = pd.cut(S[S.seed == S.seed.iloc[0]].true_pct, BINS,
                  labels=BIN_NAMES).value_counts(normalize=True).max() * 100
    print(f"  r (pooled, between+within subjects)  {np.mean(seg_r):.4f} ± {np.std(seg_r, ddof=1):.4f}")
    print(f"  ★ r (mean within subject)           {np.nanmean(within_r):.4f} ± {np.nanstd(within_r, ddof=1):.4f}")
    print(f"     ↑ this is the metric the proposal asks about: can the same person's different loads be discriminated")
    print(f"  MAE(%1RM)   {np.mean(seg_mae):.4f} ± {np.std(seg_mae, ddof=1):.4f}")
    print(f"  four-level classification accuracy {np.mean(seg_acc)*100:.1f}% ± {np.std(seg_acc, ddof=1)*100:.1f}%"
          f"   (chance baseline = largest-class share {base:.1f}%)")

    print("\n  Within-subject r per subject (mean across seeds):")
    wr = {}
    for subj, g in S.groupby('Subject'):
        rs = [pearson(gg.true_pct, gg.pred) for _, gg in g.groupby('seed') if len(gg) >= 3]
        wr[subj] = np.nanmean(rs) if rs else np.nan
    print('   ' + '  '.join(f'{k}:{v:+.2f}' for k, v in sorted(wr.items())))

    print("\n[Per-segment detail] (mean across seeds)")
    m = S.groupby(['Subject', 'seg', 'trc', 'true_pct'], as_index=False)['pred'].mean()
    m['err'] = m.pred - m.true_pct
    print(m.sort_values(['Subject', 'true_pct']).round(3).to_string(index=False))

    if comp_rows:
        C = pd.concat(comp_rows, ignore_index=True)
        cm = C.groupby(['Subject', 'seg', 'trc'], as_index=False)['pred'].mean()
        print("\n" + "=" * 84)
        print("[Diagnosis] Predicted load on compensation segments (all actually 0% 1RM bodyweight)")
        print("=" * 84)
        print(f"  predictions mean {cm.pred.mean():.3f}  median {cm.pred.median():.3f}  max {cm.pred.max():.3f}")
        bw = m[m.true_pct == 0]
        print(f"  reference: predictions on true bodyweight segments mean {bw.pred.mean():.3f}  median {bw.pred.median():.3f}")
        print(f"  → if compensation segments are clearly higher than bodyweight ones, the model mistakes “movement variation” for “load”")
        print("\n  Per segment:")
        print(cm.sort_values('pred', ascending=False).round(3).to_string(index=False))

    pd.DataFrame({'seed': per.index, 'win_r': per.r.values, 'win_mae': per.mae.values,
                  'seg_r': seg_r, 'seg_r_within': within_r,
                  'seg_mae': seg_mae, 'seg_acc': seg_acc}).to_csv(
        os.path.join(RESULTS_DIR, f'{args.prefix}_summary.csv'), index=False, encoding='utf-8-sig')
    print(f"\n✅ {args.prefix}_{{raw,segment,comp,summary}}.csv")


if __name__ == '__main__':
    main()
