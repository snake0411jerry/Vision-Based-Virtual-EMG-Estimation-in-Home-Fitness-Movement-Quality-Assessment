"""
exp_distill_compress.py — distillation compression: shrink the student to 1/10 of the TCN-Transformer
=========================================================================
Corresponds to step 2, "student network", and "expected outcome 1: reduce parameters to 1/10" of the NSTC project proposal.

★ This script and `exp_distill_compare.py` solve **different** problems; do not confuse them:

    exp_distill_compare.py   all four student arms have a deployed parameter count of **29,734**
                             → it compares "which distillation losses help" (loss-function ablation)
                             → 5 seeds done, conclusion negative (kd − only = +0.005, p=0.60)

    exp_distill_compress.py  compares "after shrinking the model, can distillation recover the score"
                             → this is the deliverable behind the "1/10 parameters" claim  ← this file

Parameter baseline (settled 2026-08-02): the denominator is this project's TCN-Transformer at 107,554 → target ≤ 10,755.

  teacher (16,32,32) fusion head 64→32   29,734   1/3.6
  student (8,8,8)    fusion head 64→32   12,070   1/8.9   ← shrinking filters alone is not enough
  student (8,8,8)    fusion head 32→16   10,070   1/10.7  ← ★ the student in this experiment

  ⚠️ 7,200 of the 10,070 (71.5%) are graph-structure constants: every GraphConv carries a fixed
     A(3×20×20) + edge_importance(3×20×20); the three blocks total 7,200, **independent of filters**.
     That is a hard floor — however small the filters, it cannot go below ~7.3k. The paper should disclose this
     composition when reporting parameter counts.

Three arms (same job, same seeds, same windows):

  teacher      (16,32,32) 29,734, task loss only     — in-job reference ceiling
  small_only   (8,8,8)    10,070, task loss only     — ★ pure compression, no distillation
  small_kd     (8,8,8)    10,070, task + soft labels — ★ the deliverable

★ None of the three arms can be dropped:
  · Without `teacher` there is no in-job ceiling — cuDNN nondeterminism makes cross-session numbers
    non-subtractable (this project's cross-run dispersion is about 0.025, larger than the within-run between-seed sd).
  · **Without `small_only` you cannot tell whether a score change is "caused by compression" or "recovered by
    distillation"**, and `small_kd − small_only` is the only distillation benefit the paper can claim.

Distillation form: **response-based**
    L = 3.0·MSE(ŷ, y_true) + γ·MSE(ŷ, teacher_pred)
A regression task has no softmax, so **no temperature**. Soft labels = the teacher's continuous predictions on the **training folds**.

⚠️ Teacher and student are both ST-GCN and both consume skeletons → strictly this is **model compression /
   self-distillation**, **not the Heterogeneous KD of the paper title**. The heterogeneous line is
   `exp_distill_arch.py` (not yet run). The two must not be conflated in the manuscript.

⚠️ The evaluation protocol matches the existing experiments exactly: LOSO split by subject, windows never cross
   segment boundaries, scaler fit only on training folds, fixed 30 epochs with no EarlyStopping
   (in LOSO the val set is the test set; early stopping on it is leakage).

Usage
-----
    # full run (about 5~8 hours)
    python exp_distill_compress.py --seeds 42 1 2 3 4

    # smoke test (3 subjects, 2 epochs, a few minutes) — numbers not citable, only verifies it runs to completion
    python exp_distill_compress.py --seeds 42 --epochs 2 --subjects S01 S02 S03 \
        --prefix smoke_compress

    # adding seeds (★ always add --resume, otherwise existing results are overwritten)
    python exp_distill_compress.py --seeds 5 6 7 --resume

Output: results/{prefix}_{raw,summary}.csv (**written after every fold**, so an interruption still leaves partial results)
"""
# --- Paths: always managed centrally by paths.py at the repo root; do not revert to absolute paths ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import (  # noqa: E402,F401
    REPO_ROOT, PIPELINE_DIR, RESULTS_DIR, SKELETON_DIR, ensure_results_dir,
)
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

# let TF allocate GPU memory incrementally as needed instead of grabbing it all up front.
# together with the CPU-side construction in distill_utils.student_dataset, this keeps the ~1GB training arrays
# from being turned into GPU constants and causing OOM (on an 8GB card it crashed at fold 3 in practice).
for _gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)
sys.path.insert(0, HERE)

from eval_utils import full_metrics                      # noqa: E402
import distill_utils as D                                # noqa: E402

ARMS = ['teacher', 'small_only', 'small_kd']

TCN_T_PARAMS = 107_554      # the denominator of "1/10 parameters" (settled 2026-08-02)
TARGET_PARAMS = TCN_T_PARAMS / 10


def run_fold(tr, va, args, seed):
    """Run all arms of one fold and return {arm: (metrics, deployed_params, training_params)}"""
    n_static = tr['St'].shape[1]
    n_ch = tr['Xg'].shape[3]
    t_fl = tuple(args.teacher_filters)
    s_fl = tuple(args.student_filters)
    s_fd, s_ed = args.student_fusion
    out = {}

    # soft labels need the teacher, so whenever small_kd is present the teacher must be trained (even if not evaluated)
    need_teacher = ('teacher' in args.arms) or ('small_kd' in args.arms)
    soft_tr = None

    # ---------------- teacher ----------------
    if need_teacher:
        tf.keras.utils.set_random_seed(seed)
        teacher = D.build_student(D.WINDOW_SIZE, n_ch, n_static,
                                  filters=t_fl, with_distill_heads=False)
        n_teacher = teacher.count_params()
        ds = D.student_dataset(tr['Xg'], tr['St'], {'out_emg': tr['Yend']},
                               args.batch_size, seed)
        teacher.fit(ds, epochs=args.epochs, verbose=0)
        del ds

        if 'teacher' in args.arms:
            m = full_metrics(va['Yend'], D.student_predict(teacher, va['Xg'], va['St']))
            out['teacher'] = (m, n_teacher, n_teacher)

        if 'small_kd' in args.arms:
            # ★ soft labels are generated on the **training folds** (not the validation fold)
            soft_tr = D.student_predict(teacher, tr['Xg'], tr['St'])

        del teacher
        tf.keras.backend.clear_session()
        gc.collect()

    # ---------------- small students ----------------
    # ★ both arms reset their initialization with the same seed and share the same windows;
    #   the only difference is whether the soft-label loss term is present.
    for arm in ('small_only', 'small_kd'):
        if arm not in args.arms:
            continue
        tf.keras.utils.set_random_seed(seed)

        if arm == 'small_only':
            s = D.build_student(D.WINDOW_SIZE, n_ch, n_static, filters=s_fl,
                                fusion_dense=s_fd, emg_dense=s_ed,
                                with_distill_heads=False)
            y = {'out_emg': tr['Yend']}
        else:
            s = D.build_student(D.WINDOW_SIZE, n_ch, n_static, filters=s_fl,
                                fusion_dense=s_fd, emg_dense=s_ed,
                                with_distill_heads=True, heads='soft')
            # response-based: soft labels only; neither hint nor recon is attached
            D.compile_student(s, beta=0.0, gamma=args.gamma, delta=0.0)
            y = {'out_emg': tr['Yend'], 'out_soft': soft_tr}

        ds = D.student_dataset(tr['Xg'], tr['St'], y, args.batch_size, seed)
        s.fit(ds, epochs=args.epochs, verbose=0)
        del ds
        m = full_metrics(va['Yend'], D.student_predict(s, va['Xg'], va['St']))
        out[arm] = (m, args._deployed_small, s.count_params())
        del s
        tf.keras.backend.clear_session()
        gc.collect()

    return out


def main():
    ap = argparse.ArgumentParser(
        description="Distillation compression: shrink the student to 1/10 of the TCN-Transformer")
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 1, 2, 3, 4])
    ap.add_argument("--arms", nargs="*", default=ARMS, choices=ARMS,
                    help="Arms to run. Note that small_kd always trains the teacher as well (needed to produce soft labels)")
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="Restrict the subjects in the data pool (for smoke tests)")
    ap.add_argument("--holdout", nargs="*", default=None,
                    help="Only run these folds (the data pool is still all subjects). For timing or single-fold debugging")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--channels", type=int, default=9, help="9 = position + velocity + acceleration")
    ap.add_argument("--gamma", type=float, default=0.5, help="soft-label loss weight")
    ap.add_argument("--teacher-filters", nargs="*", type=int, default=[16, 32, 32],
                    help="ST-GCN widths of the teacher (default = stgcn_posva, 29,734 parameters)")
    ap.add_argument("--student-filters", nargs="*", type=int, default=[8, 8, 8])
    ap.add_argument("--student-fusion", nargs=2, type=int, default=[32, 16],
                    metavar=('FUSION_DENSE', 'EMG_DENSE'),
                    help="Width of the student fusion head. The default 32 16 is needed to reach 1/10; "
                         "64 32 gives 12,070 (1/8.9, target not met)")
    ap.add_argument("--prefix", default="exp_distill_compress")
    ap.add_argument("--resume", action="store_true",
                    help="Reuse the existing {prefix}_raw.csv and only fill in (seed, subject) combinations not yet run. "
                         "**Always add this** when adding seeds, otherwise all previous results are overwritten")
    args = ap.parse_args()

    ensure_results_dir()
    t_fl = tuple(args.teacher_filters)
    s_fl = tuple(args.student_filters)
    s_fd, s_ed = args.student_fusion

    data = D.load_skeleton_segments(args.channels)
    subjects = sorted(data) if args.subjects is None else sorted(args.subjects)
    holdouts = subjects if args.holdout is None else [s for s in subjects if s in args.holdout]

    n_static = len(next(iter(data.values()))[0][2])
    n_teacher = D.deployed_param_count(D.WINDOW_SIZE, args.channels, n_static, t_fl)
    n_small = D.deployed_param_count(D.WINDOW_SIZE, args.channels, n_static, s_fl,
                                     fusion_dense=s_fd, emg_dense=s_ed)
    args._deployed_small = n_small

    print(f"🦴 skeleton {sum(len(v) for v in data.values())} segments / "
          f"{D.NUM_JOINTS} joints / {args.channels} channels")
    print(f"👥 {len(subjects)} subjects: {subjects}")
    print(f"🔁 arms={args.arms} × {len(args.seeds)} seeds × {len(holdouts)} folds "
          f"| epochs={args.epochs}  γ(soft)={args.gamma}\n")

    print("  Parameter budget (denominator = TCN-Transformer 107,554, target ≤ 10,755)")
    print(f"    teacher  filters={t_fl} fusion head 64→32   {n_teacher:>7,}  "
          f"1/{TCN_T_PARAMS / n_teacher:.1f}")
    ok = '✅ target met' if n_small <= TARGET_PARAMS else '❌ target not met'
    print(f"    student  filters={s_fl} fusion head {s_fd}→{s_ed}  {n_small:>7,}  "
          f"1/{TCN_T_PARAMS / n_small:.1f}  {ok}")
    print(f"    compression ratio (student vs teacher)  1/{n_teacher / n_small:.2f}\n")
    if n_small > TARGET_PARAMS:
        print(f"  ⚠️ student {n_small:,} exceeds the 1/10 target {TARGET_PARAMS:,.0f}; "
              f"the paper must not claim the target is met.\n")

    raw_out = os.path.join(RESULTS_DIR, f'{args.prefix}_raw.csv')

    # --resume: reuse existing results and only fill in (seed, subject) combinations not yet run.
    rows, skip = [], set()
    if args.resume and os.path.exists(raw_out):
        prev = pd.read_csv(raw_out)
        rows = prev.to_dict('records')
        skip = {(int(r['seed']), str(r['Subject'])) for r in rows}
        print(f"↻ resuming: reusing {len(skip)} existing folds "
              f"(seeds {sorted({s for s, _ in skip})})\n")

    todo = [(s, h) for s in args.seeds for h in holdouts if (s, h) not in skip]
    total = len(todo)
    done = 0
    t0 = time.time()

    if not todo:
        print("✅ Every (seed, subject) combination is already done; going straight to the summary.")

    for seed, hold in todo:
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
        msg = "  ".join(f"{a}={res[a][0]['r'][0]:.3f}" for a in ARMS if a in res)
        print(f"[{done}/{total}] seed={seed} hold={hold}  main r → {msg}   "
              f"({el / 60:.1f} min/fold, about {el * (total - done) / 60:.0f} min left)", flush=True)

        # release only after res has been used (each fold's training arrays are about 2GB)
        del tr, va, res
        tf.keras.backend.clear_session()
        gc.collect()

    # ---------------- summary ----------------
    df = pd.DataFrame(rows)
    per = df.groupby(['arm', 'seed'])[['r_main', 'r_syn', 'nrmse_main',
                                       'nrmse_syn']].mean().reset_index()
    # convention excluding S09 (that subject's synergist channel is suspected of poor electrode contact; done consistently project-wide)
    ex = df[df['Subject'] != 'S09'].groupby(['arm', 'seed'])[
        ['r_main', 'r_syn']].mean().reset_index()
    per = per.merge(ex, on=['arm', 'seed'], suffixes=('', '_ex_s09'))
    per.to_csv(os.path.join(RESULTS_DIR, f'{args.prefix}_summary.csv'),
               index=False, encoding='utf-8-sig')

    present = [a for a in ARMS if a in set(df.arm)]
    print("\n" + "=" * 96)
    print(f"Distillation compression (mean ± sd over {per.seed.nunique()} seeds)")
    print("=" * 96)
    for c in ['r_main', 'r_syn', 'nrmse_main', 'nrmse_syn', 'r_main_ex_s09', 'r_syn_ex_s09']:
        print(f"  {c:<15}" + "".join(
            f"  {a}: {per[per.arm == a][c].mean():.4f}±{per[per.arm == a][c].std(ddof=1):.4f}"
            for a in present))

    par = df.groupby('arm')[['deployed_params', 'training_params']].first()
    print("\n  Parameter counts (deployed / training-time):")
    for a in present:
        d = par.loc[a, 'deployed_params']
        print(f"    {a:<12} {d:>9,} / {par.loc[a, 'training_params']:>9,}"
              f"   1/{TCN_T_PARAMS / d:.1f} of TCN-T")

    # ---------------- paired tests ----------------
    from scipy import stats

    def paired(a_ref, a_cur, label):
        if a_ref not in present or a_cur not in present:
            return
        ref = per[per.arm == a_ref].set_index('seed')
        cur = per[per.arm == a_cur].set_index('seed')
        sh = ref.index.intersection(cur.index)
        if len(sh) < 2:
            print(f"\n  {label}: not enough seeds (n={len(sh)}), skipping the test")
            return
        print(f"\n  {label} ({len(sh)} shared seeds)")
        for c in ['r_main', 'r_syn', 'nrmse_main', 'nrmse_syn']:
            x, y = ref.loc[sh, c], cur.loc[sh, c]
            diff = y - x
            t, p = stats.ttest_rel(y, x)
            sd = diff.std(ddof=1)
            dz = diff.mean() / sd if sd > 0 else np.nan
            print(f"    {c:<12} diff {diff.mean():+.4f}   t={t:+.2f}  p={p:.4f}  dz={dz:+.2f}")
        d = cur.loc[sh, 'r_main'] - ref.loc[sh, 'r_main']
        print(f"    per-seed paired diff (r_main): {'  '.join(f'{v:+.4f}' for v in d)}"
              f"   {'same sign ✅' if (d > 0).all() or (d < 0).all() else 'mixed signs ⚠️'}")

    print("\n" + "-" * 96)
    print("★ How to read")
    print("-" * 96)
    paired('teacher', 'small_only', '① cost of pure compression   small_only − teacher (no distillation)')
    paired('small_only', 'small_kd', '② contribution of distillation small_kd − small_only  ★ the only distillation benefit the paper can claim')
    paired('teacher', 'small_kd', '③ net drop, compression+KD   small_kd − teacher')

    print(f"\n  ⚠️ n={per.seed.nunique()} seeds, limited power. Between-seed sd in this project is about 0.015~0.018,")
    print(f"     so detection needs Δ > 0.03; a non-significant p only means “no evidence of a difference” and cannot claim equivalence.")
    print(f"  ⚠️ Lesson learned: at n=3, exp_distill_compare was +0.0144 with all three seeds the same sign and p=0.056,")
    print(f"     and the 4th seed flipped it. **Do not conclude with fewer than 5 seeds**, and always check whether per-seed signs agree.")
    print(f"  📌 Even if ② is not significant, ① is a publishable result on its own — "
          f"“cutting parameters to 1/{n_teacher / n_small:.1f} only loses X” is a compression argument and does not need distillation to win.")

    print(f"\n✅ {args.prefix}_raw.csv / {args.prefix}_summary.csv")


if __name__ == '__main__':
    main()
