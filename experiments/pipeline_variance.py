"""
pipeline_variance.py — total variation of the whole pipeline + per-segment residuals
=========================================================================
Two purposes:

[A] Total variation of the whole pipeline
     The earlier multi-seed experiments each covered only one stage:
       - loso_multiseed.py             only retrains LOSO (no fine-tuning)
       - phase2_finetune_multiseed.py  only reruns fine-tuning (base models fixed)
     This script **runs the full pipeline from scratch for every seed** (10 LOSO folds → fine-tune 10 subjects),
     so only the sd of the fine-tuned absolute values here is a truly citable uncertainty.

[B] Per-segment residuals — use model performance to cross-check the suspicious segments flagged by `check_event_alignment.py`
     If a segment really has a skeleton/EMG timing misalignment, the model should perform systematically worse on it.
     This is more objective than eyeballing waveforms and needs no extra training.
     ★ Per-segment r depends on the segment's own variability (segments that barely move have inherently low r),
       so a relative value "segment r − median over the same subject's segments" is also output, comparable across people.

Output:
  pipeline_variance_loso.csv       (seed, subject) zero-shot metrics
  pipeline_variance_finetune.csv   (seed, subject) before and after fine-tuning
  pipeline_variance_segments.csv   (seed, subject, segment) per-segment residuals
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
import os
import sys
import glob
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

from eval_utils import (subject_id_from_path, build_ts_cols, make_windows,   # noqa: E402
                        temporal_split_df, full_metrics)
from loso_train_and_save import (build_model, FEATURE_MODE, WINDOW_SIZE,     # noqa: E402
                                 STEP_SIZE, EPOCHS, BATCH_SIZE,
                                 STATIC_COLS, LABEL_EMG_COLS, DATA_DIR)
from phase2_finetune_loso import (VAL_TAIL_FRAC, FT_EPOCHS, FT_BATCH, FT_LR)  # noqa: E402

import Features_insert_FIXED as F  # noqa: E402

# segment file name -> original TRC name, for cross-referencing with event_alignment_report.csv
SEG2TRC = {}
for _s in F.SUBJECTS:
    for _trc, _sid in zip(_s["trc_order"], _s["segment_ids"]):
        SEG2TRC[(_s["key"], _sid)] = _trc


def seg_id_of(path):
    # S01_Seg_3_Combined_Features.csv -> 3
    return int(os.path.basename(path).split('_')[2])


def predict(model, X, S):
    p = model.predict({'ts_input': X, 'static_input': S}, verbose=0)
    return p[0] if isinstance(p, list) else p


def run_seed(seed, file_paths, groups, subjects, dfs, ts_cols):
    tf.keras.utils.set_random_seed(seed)
    loso_rows, ft_rows, seg_rows = [], [], []

    for hold in subjects:
        tr_files = [p for p, g in zip(file_paths, groups) if g != hold]
        va_files = [p for p, g in zip(file_paths, groups) if g == hold]
        tr_dfs = [dfs[p] for p in tr_files]
        va_dfs = [dfs[p] for p in va_files]

        big = pd.concat(tr_dfs, ignore_index=True)
        s_ts = StandardScaler().fit(big[ts_cols].values)
        s_st = StandardScaler().fit(big[STATIC_COLS].values)

        Xtr, Str, Ytr, _ = make_windows(tr_dfs, ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                        None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
        model = build_model(len(ts_cols), len(STATIC_COLS))
        model.fit({'ts_input': Xtr, 'static_input': Str}, {'out_emg': Ytr},
                  epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

        # ---- zero-shot: the whole subject ----
        Xva, Sva, Yva, _ = make_windows(va_dfs, ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                        None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
        m = full_metrics(Yva, predict(model, Xva, Sva))
        loso_rows.append({'seed': seed, 'Subject': hold,
                          'r_main': m['r'][0], 'r_syn': m['r'][1],
                          'nrmse_main': m['nrmse'][0], 'nrmse_syn': m['nrmse'][1]})

        # ---- [B] per-segment residuals (zero-shot basis) ----
        for p in va_files:
            Xs, Ss, Ys, _ = make_windows([dfs[p]], ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                         None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
            if len(Xs) == 0:
                continue
            pr = predict(model, Xs, Ss)
            ms = full_metrics(Ys, pr)
            sid = seg_id_of(p)
            seg_rows.append({
                'seed': seed, 'Subject': hold, 'seg': sid,
                'trc': SEG2TRC.get((hold, sid), '?'), 'n_win': len(Xs),
                'r_main': ms['r'][0], 'r_syn': ms['r'][1],
                'nrmse_main': ms['nrmse'][0], 'nrmse_syn': ms['nrmse'][1],
                'bias_main': float(np.mean(pr[:, 0] - Ys[:, 0])),
            })

        # ---- [A] then fine-tune (phase2 protocol: first 80% of each segment for fine-tuning, last 20% for validation) ----
        train_parts, val_parts = [], []
        for df in va_dfs:
            a, b = temporal_split_df(df, VAL_TAIL_FRAC)
            train_parts.append(a)
            val_parts.append(b)
        Xf, Sf, Yf, _ = make_windows(train_parts, ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                     None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
        Xv, Sv, Yv, _ = make_windows(val_parts, ts_cols, STATIC_COLS, LABEL_EMG_COLS,
                                     None, WINDOW_SIZE, STEP_SIZE, s_ts, s_st)
        if len(Xv):
            before = full_metrics(Yv, predict(model, Xv, Sv))
            for layer in model.layers:
                layer.trainable = ('feature' not in layer.name)
            model.compile(optimizer=tf.keras.optimizers.Adam(FT_LR),
                          loss={'out_emg': 'mse'}, loss_weights={'out_emg': 3.0},
                          metrics={'out_emg': 'mae'})
            model.fit({'ts_input': Xf, 'static_input': Sf}, {'out_emg': Yf},
                      epochs=FT_EPOCHS, batch_size=FT_BATCH,
                      validation_data=({'ts_input': Xv, 'static_input': Sv}, {'out_emg': Yv}),
                      callbacks=[tf.keras.callbacks.EarlyStopping(
                          monitor='val_loss', patience=5, restore_best_weights=True)],
                      verbose=0)
            after = full_metrics(Yv, predict(model, Xv, Sv))
            ft_rows.append({
                'seed': seed, 'Subject': hold,
                'r_main_before': before['r'][0], 'r_main_after': after['r'][0],
                'r_syn_before': before['r'][1], 'r_syn_after': after['r'][1],
                'nrmse_main_before': before['nrmse'][0], 'nrmse_main_after': after['nrmse'][0],
                'nrmse_syn_before': before['nrmse'][1], 'nrmse_syn_after': after['nrmse'][1],
                'd_r_main': after['r'][0] - before['r'][0],
                'd_r_syn': after['r'][1] - before['r'][1],
            })

        tf.keras.backend.clear_session()

    return loso_rows, ft_rows, seg_rows


def main():
    ap = argparse.ArgumentParser(description="Total pipeline variation + per-segment residuals")
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 1, 2, 3, 4])
    ap.add_argument("--prefix", default="pipeline_variance")
    args = ap.parse_args()

    file_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*_Combined_Features.csv")))
    groups = [subject_id_from_path(p) for p in file_paths]
    subjects = sorted(set(groups))
    dfs = {p: pd.read_csv(p) for p in file_paths}
    ts_cols = build_ts_cols(list(dfs[file_paths[0]].columns), FEATURE_MODE, False)
    print(f"📂 {len(file_paths)} files / {len(subjects)} subjects / time-series features {len(ts_cols)}")
    print(f"🔁 {len(args.seeds)} seeds; each seed runs the full LOSO + fine-tuning")

    L, Fr, S = [], [], []
    t0 = time.time()
    for i, seed in enumerate(args.seeds, 1):
        a, b, c = run_seed(seed, file_paths, groups, subjects, dfs, ts_cols)
        L += a
        Fr += b
        S += c
        pd.DataFrame(L).to_csv(os.path.join(RESULTS_DIR, f'{args.prefix}_loso.csv'),
                               index=False, encoding='utf-8-sig')
        pd.DataFrame(Fr).to_csv(os.path.join(RESULTS_DIR, f'{args.prefix}_finetune.csv'),
                                index=False, encoding='utf-8-sig')
        pd.DataFrame(S).to_csv(os.path.join(RESULTS_DIR, f'{args.prefix}_segments.csv'),
                               index=False, encoding='utf-8-sig')
        el = (time.time() - t0) / i
        dl = pd.DataFrame(a)
        df_ = pd.DataFrame(b)
        print(f"[{i}/{len(args.seeds)}] seed={seed}  "
              f"zero-shot {dl.r_main.mean():.3f}/{dl.r_syn.mean():.3f}  "
              f"fine-tuned {df_.r_main_after.mean():.3f}/{df_.r_syn_after.mean():.3f}  "
              f"({el/60:.1f} min/seed, about {el*(len(args.seeds)-i)/60:.0f} min left)", flush=True)

    # ---------------- summary ----------------
    dl = pd.DataFrame(L)
    dfn = pd.DataFrame(Fr)
    print("\n" + "=" * 86)
    print(f"[A] Total variation of the whole pipeline (n={len(args.seeds)} full runs)")
    print("=" * 86)
    for name, d, cols in (("Zero-shot LOSO", dl, ['r_main', 'r_syn', 'nrmse_main', 'nrmse_syn']),
                          ("Fine-tuned", dfn, ['r_main_after', 'r_syn_after',
                                           'nrmse_main_after', 'nrmse_syn_after']),
                          ("Personalization gain", dfn, ['d_r_main', 'd_r_syn'])):
        print(f"\n  --- {name} ---")
        for c in cols:
            per = d.groupby('seed')[c].mean()
            ex = d[d.Subject != 'S09'].groupby('seed')[c].mean()
            print(f"    {c:<20} all 10 {per.mean():.4f}±{per.std(ddof=1):.4f}"
                  f"    excl. S09 {ex.mean():.4f}±{ex.std(ddof=1):.4f}")

    print("\n" + "=" * 86)
    print("[B] Per-segment residuals vs the event-alignment suspect list")
    print("=" * 86)
    ds = pd.DataFrame(S)
    seg = ds.groupby(['Subject', 'seg', 'trc'], as_index=False).agg(
        r_main=('r_main', 'mean'), nrmse_main=('nrmse_main', 'mean'),
        bias_main=('bias_main', 'mean'), n_win=('n_win', 'first'))
    med = seg.groupby('Subject')['r_main'].transform('median')
    seg['r_main_rel'] = seg['r_main'] - med          # relative to the same subject's median
    seg.to_csv(os.path.join(RESULTS_DIR, f'{args.prefix}_segments_summary.csv'),
               index=False, encoding='utf-8-sig')

    try:
        rep = pd.read_csv(os.path.join(RESULTS_DIR, 'event_alignment_report.csv'))
        ok = rep[rep['note'].isna() | (rep['note'] == '')]
        flagged = set(zip(ok[ok['max_err_frames'] > 40]['subject'],
                          ok[ok['max_err_frames'] > 40]['seg']))
    except Exception as e:                                    # noqa: BLE001
        print(f"  (could not read event_alignment_report.csv: {e})")
        flagged = set()

    seg['flagged'] = [(r.Subject, r.seg) in flagged for r in seg.itertuples()]
    a = seg[seg.flagged]['r_main_rel']
    b = seg[~seg.flagged]['r_main_rel']
    print(f"\n  flagged segments (n={len(a)}) relative r_main: {a.mean():+.4f} ± {a.std(ddof=1):.4f}")
    print(f"  other segments   (n={len(b)}) relative r_main: {b.mean():+.4f} ± {b.std(ddof=1):.4f}")
    if len(a) > 1 and len(b) > 1:
        from scipy import stats
        t, p = stats.ttest_ind(a, b, equal_var=False)
        print(f"  Welch t-test p={p:.3f}  → "
              f"{'flagged segments really perform worse; worth fixing' if p < 0.05 and a.mean() < b.mean() else 'no evidence that flagged segments are worse (probably detector noise)'}")

    print("\n  Per-segment detail of flagged segments (more negative r_main_rel = relatively worse):")
    print(seg[seg.flagged].sort_values('r_main_rel')[
        ['Subject', 'seg', 'trc', 'n_win', 'r_main', 'r_main_rel', 'nrmse_main', 'bias_main']
    ].round(3).to_string(index=False))

    print("\n  The 10 worst segments in the whole dataset (flagged or not):")
    print(seg.sort_values('r_main_rel').head(10)[
        ['Subject', 'seg', 'trc', 'flagged', 'r_main', 'r_main_rel', 'nrmse_main']
    ].round(3).to_string(index=False))

    print(f"\n✅ {args.prefix}_loso.csv / _finetune.csv / _segments.csv / _segments_summary.csv")


if __name__ == '__main__':
    main()
