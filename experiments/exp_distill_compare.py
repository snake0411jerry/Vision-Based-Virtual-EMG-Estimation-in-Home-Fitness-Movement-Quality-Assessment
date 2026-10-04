"""
exp_distill_compare.py — experiment C: heterogeneous knowledge distillation vs vision-only training
=========================================================================
Corresponds to step 4, "Ablation · Experiment C", of the NSTC project proposal:

  Baseline : the student network is trained with the task loss only
  Ours     : the student network adds a Distillation Loss (imitating teacher features) and a Reconstruction Loss

Six arms (same job, same seeds, same windows; only the loss function differs):

  teacher_masked    teacher with the privileged channel off — its "vision-only" ability
  teacher_visible   teacher with the privileged channel on — ★ diagnostic only; it can see the answer, **not a fair comparison**
  student_only      ST-GCN, task loss only                       ← the proposal's Baseline
  student_recon     + reconstruction (dense supervision, **no teacher information**)
  student_hint      + hint + soft labels (**only** teacher information) ← ★ what the paper wants to claim
  student_kd        + hint + soft labels + reconstruction        ← the proposal's Ours

★ Why split into four student arms:
  Running only `only` vs `kd`, even a win could not tell whether the gain came from "the teacher" or from
  "the reconstruction loss". The reconstruction loss merely replaces sparse labels with dense supervision and
  has nothing to do with knowledge distillation — the paper claims the former, so the two must be measurable
  separately.

★ The gap between `teacher_visible` and `teacher_masked` is **the precondition for distillation to work**:
  if they are close, the teacher is not using the privileged EMG channel at all, its latent representation holds
  nothing the student cannot learn, and distillation naturally gives no gain. This diagnosis is printed at the
  end of every seed.

★ Structural leakage prevention: the teacher's privileged EMG branch drops the window-end frame **inside the model**
  (`priv_drop_target` in `models_teacher.py`), and the window end is exactly the prediction target.
  Otherwise the teacher would degenerate into copying out the answer, making soft labels and latent representations meaningless.

⚠️ The evaluation protocol matches the existing experiments exactly: LOSO split by subject, windows never cross
   segment boundaries, scaler fit only on training folds, fixed 30 epochs with no EarlyStopping
   (in LOSO the val set is the test set; early stopping on it is leakage).

Usage
-----
    # full run
    python exp_distill_compare.py --seeds 42 1 2 3 4

    # smoke test (3 subjects, 2 epochs, a few minutes)
    python exp_distill_compare.py --seeds 42 --epochs 2 --subjects S01 S02 S03 --prefix smoke_distill

    # single-fold timing (data pool is still all subjects)
    python exp_distill_compare.py --seeds 42 --holdout S01 --prefix timing_probe

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
import models_teacher as T                               # noqa: E402

ARMS = ['teacher_masked', 'teacher_visible',
        'student_only', 'student_recon', 'student_hint', 'student_kd']

# student variants — used to break down "where the gain comes from".
# if we only ran student_only vs student_kd, even a win could not tell whether the credit goes to the teacher
# or to the reconstruction loss (dense supervision, unrelated to the teacher), and the paper claims the former.
#
#   student_only   task loss                        ← the proposal's Baseline
#   student_recon  task + reconstruction            ← dense supervision, **no teacher information at all**
#   student_hint   task + hint + soft labels        ← **only** information coming from the teacher
#   student_kd     task + hint + soft labels + recon ← the proposal's Ours (everything on)
#
# attribution reading:
#   recon − only  = contribution of dense supervision
#   hint  − only  = contribution of the teacher      ★ this is exactly what the paper wants to claim
#   kd    − recon = the teacher's **marginal** contribution on top of dense supervision
STUDENT_SPECS = {
    'student_only':  None,
    'student_recon': dict(hint=0.0, soft=0.0, recon=1.0),
    'student_hint':  dict(hint=1.0, soft=1.0, recon=0.0),
    'student_kd':    dict(hint=1.0, soft=1.0, recon=1.0),
}


def run_fold(tr, va, args, seed):
    """Run the four arms of one fold and return {arm: (metrics, deployed_params, training_params)}"""
    n_static = tr['St'].shape[1]
    n_ch = tr['Xg'].shape[3]
    out = {}
    deployed_n = D.deployed_param_count(D.WINDOW_SIZE, n_ch, n_static,
                                        tuple(args.filters))

    # ---------------- teacher (privileged EMG + random masking) ----------------
    teacher = T.build_teacher(D.WINDOW_SIZE, tr['Xf'].shape[2], n_static,
                              latent_dim=args.latent_dim,
                              priv_frames=args.priv_frames)
    ds = T.teacher_dataset(tr['Xf'], tr['Ywin'], tr['St'], tr['Yend'],
                           mask_prob=args.mask_prob,
                           batch_size=args.batch_size, seed=seed)
    teacher.fit(ds, epochs=args.epochs, verbose=0)

    for vis in (False, True):
        p = T.teacher_predict(teacher, va['Xf'], va['Ywin'], va['St'], emg_visible=vis)
        arm = 'teacher_visible' if vis else 'teacher_masked'
        out[arm] = (full_metrics(va['Yend'], p),
                    teacher.count_params(), teacher.count_params())

    # distillation targets (generated on the **training folds**)
    extractor = T.latent_extractor(teacher)
    soft_tr, latent_tr = T.teacher_targets(teacher, extractor,
                                           tr['Xf'], tr['Ywin'], tr['St'])
    del teacher, extractor

    # ---------------- student variants ----------------
    # ★ every variant resets its initialization with the same seed and shares the same windows and teacher targets,
    #   ensuring the only difference between arms is the loss function.
    for arm in args.arms:
        spec = STUDENT_SPECS[arm]
        tf.keras.utils.set_random_seed(seed)

        if spec is None:
            s = D.build_student(D.WINDOW_SIZE, n_ch, n_static,
                                filters=tuple(args.filters), with_distill_heads=False)
            y = {'out_emg': tr['Yend']}
        else:
            s = D.build_student(D.WINDOW_SIZE, n_ch, n_static,
                                filters=tuple(args.filters),
                                latent_dim=args.latent_dim, with_distill_heads=True)
            D.compile_student(s,
                              beta=args.beta * spec['hint'],
                              gamma=args.gamma * spec['soft'],
                              delta=args.delta * spec['recon'])
            y = {'out_emg': tr['Yend'], 'out_soft': soft_tr,
                 'hint_proj': latent_tr, 'emg_recon': tr['Ywin']}

        ds = D.student_dataset(tr['Xg'], tr['St'], y, args.batch_size, seed)
        s.fit(ds, epochs=args.epochs, verbose=0)
        del ds
        m = full_metrics(va['Yend'], D.student_predict(s, va['Xg'], va['St']))
        out[arm] = (m, deployed_n, s.count_params())
        del s
        tf.keras.backend.clear_session()

    return out


def main():
    ap = argparse.ArgumentParser(description="Experiment C: heterogeneous knowledge distillation vs vision-only training")
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 1, 2, 3, 4])
    ap.add_argument("--arms", nargs="*",
                    default=['student_only', 'student_recon', 'student_hint', 'student_kd'],
                    choices=list(STUDENT_SPECS),
                    help="Student arms (the two teacher arms always run; they are the precondition diagnosis for distillation)")
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="Restrict the subjects in the data pool (for smoke tests)")
    ap.add_argument("--holdout", nargs="*", default=None,
                    help="Only run these folds (the data pool is still all subjects). For timing or single-fold debugging")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--filters", nargs="*", type=int, default=[16, 32, 32])
    ap.add_argument("--channels", type=int, default=9, help="9 = position + velocity + acceleration")
    ap.add_argument("--latent-dim", type=int, default=64)
    ap.add_argument("--mask-prob", type=float, default=0.5, help="Fraction of the time the teacher cannot see EMG")
    ap.add_argument("--priv-frames", type=int, default=20,
                    help="Number of EMG frames the teacher can see (counted from the window start, 40 frames total). "
                         "Setting it too high lets the teacher copy the answer directly — see the notes in models_teacher.py")
    ap.add_argument("--beta", type=float, default=1.0, help="hint loss weight")
    ap.add_argument("--gamma", type=float, default=0.5, help="soft-label loss weight")
    ap.add_argument("--delta", type=float, default=0.5, help="reconstruction loss weight")
    ap.add_argument("--prefix", default="exp_distill")
    ap.add_argument("--resume", action="store_true",
                    help="Reuse the existing {prefix}_raw.csv and only fill in (seed, subject) combinations not yet run. "
                         "**Always add this** when adding seeds, otherwise all previous results are overwritten")
    args = ap.parse_args()

    ensure_results_dir()
    data = D.load_skeleton_segments(args.channels)
    subjects = sorted(data) if args.subjects is None else sorted(args.subjects)
    # the data pool stays as subjects (it defines each fold's training set), but only the specified folds are run
    holdouts = subjects if args.holdout is None else [s for s in subjects if s in args.holdout]

    print(f"🦴 skeleton {sum(len(v) for v in data.values())} segments / "
          f"{D.NUM_JOINTS} joints / {args.channels} channels")
    print(f"👥 {len(subjects)} subjects: {subjects}")
    print(f"🔁 arms={args.arms} × {len(args.seeds)} seeds × {len(holdouts)} folds "
          f"| epochs={args.epochs}")
    print(f"⚙️  mask rate={args.mask_prob}  latent={args.latent_dim}  "
          f"β(hint)={args.beta} γ(soft)={args.gamma} δ(recon)={args.delta}\n")

    raw_out = os.path.join(RESULTS_DIR, f'{args.prefix}_raw.csv')

    # --resume: reuse existing results and only fill in (seed, subject) combinations not yet run.
    # without it, adding seeds under the same prefix would **overwrite all previous results**.
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

    for i, (seed, hold) in enumerate(todo):
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
        msg = "  ".join(
            f"{a.replace('student_', 'S:').replace('teacher_', 'T:')}"
            f"={res[a][0]['r'][0]:.3f}" for a in ARMS if a in res)
        print(f"[{done}/{total}] seed={seed} hold={hold}  main r → {msg}   "
              f"({el/60:.1f} min/fold, about {el*(total-done)/60:.0f} min left)", flush=True)

        # release only after res has been used (each fold's training arrays are about 2GB)
        del tr, va, res
        tf.keras.backend.clear_session()
        gc.collect()

        # when the last fold of this seed finishes, print the precondition diagnosis
        if i + 1 == len(todo) or todo[i + 1][0] != seed:
            d = pd.DataFrame(rows)
            d = d[d.seed == seed].groupby('arm')['r_main'].mean()
            if 'teacher_visible' in d and 'teacher_masked' in d:
                gap = d['teacher_visible'] - d['teacher_masked']
                flag = "✅ privileged channel effective" if gap > 0.05 else "⚠️ privileged channel barely used"
                print(f"    └─ precondition seed={seed}: teacher privileged={d['teacher_visible']:.3f} "
                      f"vs vision-only={d['teacher_masked']:.3f}  diff {gap:+.3f}  {flag}\n",
                      flush=True)

    # ---------------- summary ----------------
    df = pd.DataFrame(rows)
    per = df.groupby(['arm', 'seed'])[['r_main', 'r_syn', 'nrmse_main',
                                       'nrmse_syn']].mean().reset_index()
    per.to_csv(os.path.join(RESULTS_DIR, f'{args.prefix}_summary.csv'),
               index=False, encoding='utf-8-sig')

    print("\n" + "=" * 96)
    print(f"Experiment C: heterogeneous knowledge distillation (mean ± sd over {len(args.seeds)} seeds)")
    print("=" * 96)
    present = [a for a in ARMS if a in set(df.arm)]
    for c in ['r_main', 'r_syn', 'nrmse_main', 'nrmse_syn']:
        print(f"  {c:<12}" + "".join(
            f"  {a}: {per[per.arm == a][c].mean():.4f}±{per[per.arm == a][c].std(ddof=1):.4f}"
            for a in present))

    par = df.groupby('arm')[['deployed_params', 'training_params']].first()
    print("\n  Parameter counts (deployed / training-time):")
    for a in present:
        print(f"    {a:<18} {par.loc[a, 'deployed_params']:>9,} / "
              f"{par.loc[a, 'training_params']:>9,}")

    # ---------------- paired tests and gain attribution ----------------
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
    print("★ Gain attribution (all relative to student_only)")
    print("-" * 96)
    paired('student_only', 'student_recon', '① dense supervision     recon − only (no teacher)')
    paired('student_only', 'student_hint', '② teacher contribution  hint − only  ★ this is what the paper wants to claim')
    paired('student_recon', 'student_kd', '③ teacher marginal      kd − recon (on top of dense supervision)')
    paired('student_only', 'student_kd', '④ total gain, all on    kd − only')

    print(f"\n  ⚠️ n={len(args.seeds)} seeds, limited power. Between-seed sd in this project is about 0.015~0.018,")
    print(f"     so detection needs Δ > 0.03; a non-significant p only means “no evidence of a difference” and cannot claim equivalence.")
    print(f"  ⚠️ teacher_visible can see the real EMG — **not a fair comparison**; it only confirms the privileged channel works.")

    print(f"\n✅ {args.prefix}_raw.csv / {args.prefix}_summary.csv")


if __name__ == '__main__':
    main()
