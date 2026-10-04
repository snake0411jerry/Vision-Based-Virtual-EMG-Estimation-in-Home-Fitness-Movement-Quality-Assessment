"""
exp_distill_arch.py — experiments A + C: necessity of a heterogeneous teacher & distillation vs vision only
=========================================================================
One job answers both ablations of step 4 of the NSTC project proposal:

  Experiment A (necessity of a heterogeneous teacher)
      Baseline : distillation from a homogeneous architecture (enlarged ST-GCN teacher)
      Ours     : distillation from a heterogeneous architecture (TCN-Transformer teacher)
      → if heterogeneous does not beat homogeneous, we cannot claim "heterogeneous distillation beats plain model compression"

  Experiment C (distillation vs vision only)
      Baseline : the student uses only the task loss
      Ours     : the student adds the distillation and reconstruction losses

Five arms (same job, same seeds, same windows):

  teacher_hetero    TCN-Transformer teacher (sequence features) — diagnoses its teaching ability
  teacher_homo      enlarged ST-GCN teacher (graph features)    — diagnoses its teaching ability
  student_only      ST-GCN, task loss only                      ← baseline shared by both experiments
  student_kd_hetero distilled from the heterogeneous teacher    ← "Ours" of experiment A
  student_kd_homo   distilled from the homogeneous teacher      ← "Baseline" of experiment A

Reading:
  kd_hetero − only           experiment C: does distillation help
  kd_hetero − kd_homo        experiment A: is heterogeneous better than homogeneous
  teacher_* vs student_only  is the teacher actually stronger than the student (if not, distillation has nothing to transfer)

---------------------------------------------------------------- Why not privileged EMG

🔴 **Correction 2026-08-06: this section originally said "priv_frames=39 → teacher r=0.993, degenerates into
   copying the answer", which contradicts the project's own measurements; it has been corrected.**

Measured (`probe_teacher_privilege_summary.csv`, two folds S01/S03):

    priv_frames=25/30/35 → gap only 0.001~0.006; the privileged channel might as well not exist
    priv_frames=39       → gap **+0.071**, r(visible)=0.847 ≤ 0.95, **judged usable, not degenerate**

`exp_distill_compare.py` was run with 39, and the teacher really is legitimately stronger
(`teacher_visible` 0.8397 vs `teacher_masked` 0.7584, consistent over 5 seeds),
**yet distillation is still ineffective** (`kd − only = +0.0054, p=0.60`).

→ So this experiment **does not exist because the privileged route failed**; it answers a separate question:
  **given that "the teacher is stronger but the knowledge does not transfer", would architectural heterogeneity differ?**
  That is the original definition of experiment A in the proposal, and it needs no privileged modality.

⚠️ If both fail the same way (nothing transfers), the paper title's
  `Heterogeneous Knowledge Distillation` must be honestly rewritten or turned into a negative-result statement.

⚠️ The evaluation protocol matches the existing experiments: LOSO split by subject, windows never cross
   segment boundaries, scaler fit only on training folds, fixed epochs with no EarlyStopping.

Usage
-----
    python exp_distill_arch.py --seeds 42 1 2 --batch-size 128
"""
# --- Paths: always managed centrally by paths.py at the repo root; do not revert to absolute paths ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import PIPELINE_DIR, RESULTS_DIR, ensure_results_dir  # noqa: E402
# --- End of path setup ---

import argparse
import gc
import os
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

for _gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)
sys.path.insert(0, HERE)

from eval_utils import full_metrics       # noqa: E402
import distill_utils as D                 # noqa: E402
import models_teacher as T                # noqa: E402

ARMS = ['teacher_hetero', 'teacher_homo',
        'student_only', 'student_kd_hetero', 'student_kd_homo']


def _train_students(tr, va, args, seed, targets, deployed_n, out):
    """targets: {arm_suffix: (soft, latent)}; None means no distillation."""
    for arm, tgt in targets.items():
        tf.keras.utils.set_random_seed(seed)      # identical initialization for every arm
        if tgt is None:
            s = D.build_student(D.WINDOW_SIZE, tr['Xg'].shape[3], tr['St'].shape[1],
                                filters=tuple(args.filters), with_distill_heads=False)
            y = {'out_emg': tr['Yend']}
        else:
            soft, latent = tgt
            s = D.build_student(D.WINDOW_SIZE, tr['Xg'].shape[3], tr['St'].shape[1],
                                filters=tuple(args.filters),
                                latent_dim=latent.shape[1], with_distill_heads=True)
            D.compile_student(s, beta=args.beta, gamma=args.gamma, delta=args.delta)
            y = {'out_emg': tr['Yend'], 'out_soft': soft,
                 'hint_proj': latent, 'emg_recon': tr['Ywin']}

        ds = D.student_dataset(tr['Xg'], tr['St'], y, args.batch_size, seed)
        s.fit(ds, epochs=args.epochs, verbose=0)
        out[arm] = (full_metrics(va['Yend'], D.student_predict(s, va['Xg'], va['St'])),
                    deployed_n, s.count_params())
        del s, ds
        tf.keras.backend.clear_session()
        gc.collect()


def run_fold(tr, va, args, seed):
    n_static = tr['St'].shape[1]
    n_ch = tr['Xg'].shape[3]
    out = {}
    deployed_n = D.deployed_param_count(D.WINDOW_SIZE, n_ch, n_static,
                                        tuple(args.filters))

    # ---------- heterogeneous teacher: TCN-Transformer (sequence features), skeleton only ----------
    tf.keras.utils.set_random_seed(seed)
    th = T.build_teacher(D.WINDOW_SIZE, tr['Xf'].shape[2], n_static,
                         latent_dim=args.latent_dim, priv_frames=1)
    # mask_prob=1.0 → the privileged channel is always off, reducing it to a skeleton-only Transformer teacher
    ds = T.teacher_dataset(tr['Xf'], tr['Ywin'], tr['St'], tr['Yend'],
                           mask_prob=1.0, batch_size=args.batch_size, seed=seed)
    th.fit(ds, epochs=args.epochs, verbose=0)
    out['teacher_hetero'] = (
        full_metrics(va['Yend'], T.teacher_predict(th, va['Xf'], va['Ywin'],
                                                   va['St'], emg_visible=False)),
        th.count_params(), th.count_params())
    soft_h, lat_h = T.teacher_targets(th, T.latent_extractor(th),
                                      tr['Xf'], tr['Ywin'], tr['St'])
    del th, ds
    tf.keras.backend.clear_session()
    gc.collect()

    # ---------- homogeneous teacher: enlarged ST-GCN (graph features) ----------
    tf.keras.utils.set_random_seed(seed)
    tm = T.build_homogeneous_teacher(D.WINDOW_SIZE, n_ch, n_static,
                                     filters=tuple(args.homo_filters),
                                     latent_dim=args.latent_dim)
    ds = D.student_dataset(tr['Xg'], tr['St'], {'out_emg': tr['Yend']},
                           args.batch_size, seed)
    tm.fit(ds, epochs=args.epochs, verbose=0)
    out['teacher_homo'] = (
        full_metrics(va['Yend'], T.homo_predict(tm, va['Xg'], va['St'])),
        tm.count_params(), tm.count_params())
    soft_m, lat_m = T.homo_targets(tm, T.latent_extractor(tm), tr['Xg'], tr['St'])
    del tm, ds
    tf.keras.backend.clear_session()
    gc.collect()

    # ---------- three students ----------
    _train_students(tr, va, args, seed, {
        'student_only': None,
        'student_kd_hetero': (soft_h, lat_h),
        'student_kd_homo': (soft_m, lat_m),
    }, deployed_n, out)

    return out


def main():
    ap = argparse.ArgumentParser(description="Experiments A + C: heterogeneous teacher & distillation")
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 1, 2])
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--holdout", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--filters", nargs="*", type=int, default=[16, 32, 32])
    ap.add_argument("--homo-filters", nargs="*", type=int, default=[32, 64, 64],
                    help="Channel widths of the homogeneous teacher (enlarged ST-GCN)")
    ap.add_argument("--channels", type=int, default=9)
    ap.add_argument("--latent-dim", type=int, default=64)
    ap.add_argument("--beta", type=float, default=1.0, help="hint loss weight")
    ap.add_argument("--gamma", type=float, default=0.5, help="soft-label loss weight")
    ap.add_argument("--delta", type=float, default=0.5, help="reconstruction loss weight")
    ap.add_argument("--prefix", default="exp_distill_arch")
    args = ap.parse_args()

    ensure_results_dir()
    data = D.load_skeleton_segments(args.channels)
    subjects = sorted(data) if args.subjects is None else sorted(args.subjects)
    holdouts = subjects if args.holdout is None else [s for s in subjects if s in args.holdout]

    print(f"🦴 skeleton {sum(len(v) for v in data.values())} segments / {D.NUM_JOINTS} joints / "
          f"{args.channels} channels")
    print(f"👥 {len(subjects)} subjects  |  {len(holdouts)} folds  |  seeds {args.seeds}")
    print(f"⚙️  batch={args.batch_size} epochs={args.epochs} "
          f"student filters={args.filters}  homogeneous teacher filters={args.homo_filters}")
    print(f"    β(hint)={args.beta} γ(soft)={args.gamma} δ(recon)={args.delta}\n")

    raw_out = os.path.join(RESULTS_DIR, f'{args.prefix}_raw.csv')
    rows = []
    total = len(args.seeds) * len(holdouts)
    done, t0 = 0, time.time()

    for seed in args.seeds:
        for hold in holdouts:
            tf.keras.utils.set_random_seed(seed)
            tr, va = D.build_fold(data, subjects, hold, args.channels)
            res = run_fold(tr, va, args, seed)
            for arm, (m, dep, trn) in res.items():
                rows.append({'arm': arm, 'seed': seed, 'Subject': hold,
                             'r_main': m['r'][0], 'r_syn': m['r'][1],
                             'nrmse_main': m['nrmse'][0], 'nrmse_syn': m['nrmse'][1],
                             'deployed_params': dep, 'training_params': trn})
            pd.DataFrame(rows).to_csv(raw_out, index=False, encoding='utf-8-sig')

            done += 1
            el = (time.time() - t0) / done
            msg = "  ".join(f"{a.replace('student_', 'S:').replace('teacher_', 'T:')}"
                            f"={res[a][0]['r'][0]:.3f}" for a in ARMS if a in res)
            print(f"[{done}/{total}] seed={seed} hold={hold}  main r → {msg}   "
                  f"({el/60:.1f} min/fold, about {el*(total-done)/60:.0f} min left)", flush=True)

            del tr, va, res
            tf.keras.backend.clear_session()
            gc.collect()

    df = pd.DataFrame(rows)
    per = df.groupby(['arm', 'seed'])[['r_main', 'r_syn', 'nrmse_main',
                                       'nrmse_syn']].mean().reset_index()
    per.to_csv(os.path.join(RESULTS_DIR, f'{args.prefix}_summary.csv'),
               index=False, encoding='utf-8-sig')
    present = [a for a in ARMS if a in set(df.arm)]

    print("\n" + "=" * 96)
    print(f"Experiments A + C (mean ± sd over {len(args.seeds)} seeds)")
    print("=" * 96)
    for c in ['r_main', 'r_syn', 'nrmse_main', 'nrmse_syn']:
        print(f"  {c:<12}" + "".join(
            f"  {a}: {per[per.arm == a][c].mean():.4f}±{per[per.arm == a][c].std(ddof=1):.4f}"
            for a in present))
    par = df.groupby('arm')[['deployed_params', 'training_params']].first()
    print("\n  Parameter counts (deployed / training-time):")
    for a in present:
        print(f"    {a:<20} {par.loc[a, 'deployed_params']:>9,} / "
              f"{par.loc[a, 'training_params']:>9,}")

    from scipy import stats

    def paired(ref, cur, label):
        if ref not in present or cur not in present:
            return
        a = per[per.arm == ref].set_index('seed')
        b = per[per.arm == cur].set_index('seed')
        sh = a.index.intersection(b.index)
        if len(sh) < 2:
            print(f"\n  {label}: not enough seeds")
            return
        print(f"\n  {label} ({len(sh)} shared seeds)")
        for c in ['r_main', 'r_syn', 'nrmse_main']:
            x, y = a.loc[sh, c], b.loc[sh, c]
            d = y - x
            t, p = stats.ttest_rel(y, x)
            sd = d.std(ddof=1)
            print(f"    {c:<12} diff {d.mean():+.4f}   t={t:+.2f}  p={p:.4f}  "
                  f"dz={d.mean()/sd if sd > 0 else float('nan'):+.2f}")
        dd = b.loc[sh, 'r_main'] - a.loc[sh, 'r_main']
        print(f"    per-seed diff (r_main): {'  '.join(f'{v:+.4f}' for v in dd)}"
              f"   {'same sign ✅' if (dd > 0).all() or (dd < 0).all() else 'mixed signs ⚠️'}")

    print("\n" + "-" * 96)
    print("★ Tests")
    print("-" * 96)
    paired('student_only', 'student_kd_hetero', 'Experiment C: does distillation help (kd_hetero − only)')
    paired('student_kd_homo', 'student_kd_hetero', 'Experiment A: is heterogeneous better than homogeneous (hetero − homo)')
    paired('student_only', 'student_kd_homo', '              contribution of homogeneous distillation (kd_homo − only)')

    # precondition check: is the teacher stronger than the student
    m = per.groupby('arm')['r_main'].mean()
    print("\n  ★ Precondition check (the teacher must be stronger than the student for distillation to have anything to transfer)")
    for t_arm in ('teacher_hetero', 'teacher_homo'):
        if t_arm in m and 'student_only' in m:
            gap = m[t_arm] - m['student_only']
            print(f"    {t_arm:<16} {m[t_arm]:.4f}  vs student baseline {m['student_only']:.4f}"
                  f"   diff {gap:+.4f}  "
                  f"{'✅ teacher stronger' if gap > 0 else '⚠️ teacher weaker than student — distillation is disadvantaged from the start'}")

    print(f"\n  ⚠️ n={len(args.seeds)} seeds. Between-seed sd in this project is about 0.015~0.018, "
          f"so detection needs Δ > 0.03.")
    print(f"\n✅ {args.prefix}_raw.csv / {args.prefix}_summary.csv")


if __name__ == '__main__':
    main()
