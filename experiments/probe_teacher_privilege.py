"""
probe_teacher_privilege.py — does the privileged-EMG teacher have any usable setting?
=========================================================================
This is the **preliminary diagnosis** for `exp_distill_compare.py`: it trains only the teacher, not the student, so it is cheap.

Background: the teacher in step 2 of the proposal takes "real sEMG" as a privileged input, but EMG is extremely autocorrelated,
raising a dilemma — too close to the target and it degenerates into copying the answer; too far and the privileged channel
might as well not exist.

This script sweeps the middle ground looking for a setting that is "a big enough gain, yet not degenerate".

Criteria (both must pass):
    gap  = r(visible) − r(masked)  ≥ 0.05   the privileged channel is really used
    r(visible)                     ≤ 0.95   not degenerated into copying the answer

🔴 **Correction 2026-08-06: this header originally said "priv_frames=39 teacher r=0.993 → degenerates into copying the answer";
   that was the expectation before running this script, does not match the measurement, and has been removed.**
   Measured (`probe_teacher_privilege_summary.csv`, two folds S01/S03):

       priv_frames   r_masked   r_visible   gap      verdict
              25       0.763      0.769    0.006    privileged channel unused
              30       0.758      0.759    0.001    privileged channel unused
              35       0.779      0.785    0.006    privileged channel unused
              39       0.776      0.847    0.071    ✅ usable (not degenerate, 0.847 ≤ 0.95)

   **39 is usable, not degenerate.** The `teacher_visible = 0.8397` that `exp_distill_compare.py` actually obtained with 39
   (5 seeds, 10 folds) agrees with the 0.847 here.

   → So the negative distillation result **was not "the teacher is broken"** but
     "the teacher is legitimately stronger (gap +0.071, consistent over 5 seeds), yet the knowledge does not reach the student".
     That conclusion is stronger than the original statement.

⚠️ This probe only runs two folds (S01/S03), so the gap estimate is rough — good enough to choose a setting, not to cite as a conclusion.

Usage:
    python probe_teacher_privilege.py --priv-frames 25 30 35 39 --holdout S01 S03
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

from eval_utils import full_metrics          # noqa: E402
import distill_utils as D                    # noqa: E402
import models_teacher as T                   # noqa: E402

GAP_MIN = 0.05      # the privileged channel must actually be used
VIS_MAX = 0.95      # but must not degenerate into copying the answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--priv-frames", nargs="*", type=int, default=[25, 30, 35, 39])
    ap.add_argument("--holdout", nargs="*", default=['S01', 'S03'])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--mask-prob", type=float, default=0.5)
    ap.add_argument("--latent-dim", type=int, default=64)
    ap.add_argument("--channels", type=int, default=9)
    ap.add_argument("--prefix", default="probe_teacher_privilege")
    args = ap.parse_args()

    ensure_results_dir()
    data = D.load_skeleton_segments(args.channels)
    subjects = sorted(data)

    print(f"🔬 privileged-teacher sweep  priv_frames={args.priv_frames}  folds={args.holdout}")
    print(f"   batch={args.batch_size} epochs={args.epochs} mask rate={args.mask_prob}")
    print(f"   criteria: gap ≥ {GAP_MIN} and r(visible) ≤ {VIS_MAX}\n")

    rows = []
    t0 = time.time()
    total = len(args.priv_frames) * len(args.holdout)
    done = 0

    for pf in args.priv_frames:
        for hold in args.holdout:
            tf.keras.utils.set_random_seed(args.seed)
            tr, va = D.build_fold(data, subjects, hold, args.channels)

            teacher = T.build_teacher(D.WINDOW_SIZE, tr['Xf'].shape[2],
                                      tr['St'].shape[1],
                                      latent_dim=args.latent_dim, priv_frames=pf)
            ds = T.teacher_dataset(tr['Xf'], tr['Ywin'], tr['St'], tr['Yend'],
                                   mask_prob=args.mask_prob,
                                   batch_size=args.batch_size, seed=args.seed)
            teacher.fit(ds, epochs=args.epochs, verbose=0)

            m_off = full_metrics(va['Yend'], T.teacher_predict(
                teacher, va['Xf'], va['Ywin'], va['St'], emg_visible=False))
            m_on = full_metrics(va['Yend'], T.teacher_predict(
                teacher, va['Xf'], va['Ywin'], va['St'], emg_visible=True))

            rows.append({'priv_frames': pf, 'Subject': hold,
                         'r_masked': m_off['r'][0], 'r_visible': m_on['r'][0],
                         'gap': m_on['r'][0] - m_off['r'][0],
                         'nrmse_masked': m_off['nrmse'][0]})
            pd.DataFrame(rows).to_csv(
                os.path.join(RESULTS_DIR, f'{args.prefix}_raw.csv'),
                index=False, encoding='utf-8-sig')

            done += 1
            el = (time.time() - t0) / done
            r = rows[-1]
            print(f"[{done}/{total}] priv={pf:>2} hold={hold}  "
                  f"vision-only={r['r_masked']:.3f}  privileged={r['r_visible']:.3f}  "
                  f"gap={r['gap']:+.3f}   ({el/60:.1f} min/round)", flush=True)

            del teacher, ds, tr, va
            tf.keras.backend.clear_session()
            gc.collect()

    df = pd.DataFrame(rows)
    g = df.groupby('priv_frames').agg(
        r_masked=('r_masked', 'mean'), r_visible=('r_visible', 'mean'),
        gap=('gap', 'mean')).reset_index()
    g['gap_ok'] = g.gap >= GAP_MIN
    g['not_degenerate'] = g.r_visible <= VIS_MAX
    g['usable'] = g['gap_ok'] & g['not_degenerate']
    g.to_csv(os.path.join(RESULTS_DIR, f'{args.prefix}_summary.csv'),
             index=False, encoding='utf-8-sig')

    print("\n" + "=" * 74)
    print(f"Privileged-teacher sweep results (mean over {len(args.holdout)} folds)")
    print("=" * 74)
    print(f"  {'priv_frames':>12} {'vision':>8} {'priv':>8} {'gap':>8}   verdict")
    for _, r in g.iterrows():
        verdict = ("✅ usable" if r['usable'] else
                   ("❌ degenerated into copying the answer" if not r['not_degenerate'] else "❌ privileged channel unused"))
        print(f"  {int(r.priv_frames):>12} {r.r_masked:>8.3f} {r.r_visible:>8.3f} "
              f"{r.gap:>+8.3f}   {verdict}")

    usable = g[g['usable']]
    print()
    if len(usable):
        best = usable.loc[usable.gap.idxmax()]
        print(f"✅ recommended priv_frames={int(best.priv_frames)}"
              f" (gap {best.gap:+.3f}, privileged r {best.r_visible:.3f}, not degenerate)")
        print("   → you can go on to run the privileged-teacher version of exp_distill_compare.py")
    else:
        print("❌ No setting satisfies both criteria.")
        print("   → the privileged-EMG route does not work on this dataset (EMG autocorrelation is too strong:")
        print("      close to the target it is the answer, far from it there is no information, and there is no sweet spot in between).")
        print("   → switch to architectural heterogeneity: the proposal's experiment A, Transformer teacher vs homogeneous ST-GCN teacher.")

    print(f"\n✅ {args.prefix}_raw.csv / {args.prefix}_summary.csv")


if __name__ == '__main__':
    main()
