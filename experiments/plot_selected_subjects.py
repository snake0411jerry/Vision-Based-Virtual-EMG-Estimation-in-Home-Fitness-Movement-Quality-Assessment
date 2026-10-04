"""
plot_selected_subjects.py — measured vs predicted signal plots for selected subjects
=========================================================================
Overlay the zero-shot (model has never seen this person) and fine-tuned predictions on the measured EMG.
The evaluation interval is the last 20% of each segment, unseen by both models.

Plots S01 / S09 / S10 by default:
  S01 — control with normal signal quality
  S09 — synergist suspected of poor electrode contact (flat signal, labels inflated by about 2×)
  S10 — a genuine low-activation case (normal signal quality; the synergist simply works very little)

Output:
  selected_waveform_main.png   main muscle
  selected_waveform_syn.png    synergist
  selected_scatter.png         predicted vs measured scatter (diagnostic)
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
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plot_zeroshot_vs_finetuned as Z   # noqa: E402

PICKS = ['S01', 'S09', 'S10']
LABELS = {
    'S01': 'Normal signal quality (control)',
    'S09': 'Synergist suspected of poor electrode contact',
    'S10': 'Genuinely low activation (normal signal, only 14.2 %MVC)',
}
MUSCLE = ['Main muscle (Main)', 'Synergist (Synergist)']
C_TRUE, C_ZS, C_FT = '#15181a', '#e0623c', '#0f766e'
FPS = 60.0
MAX_SEC = 60          # at most 60 seconds per plot, to avoid over-compression


def main():
    for f in ['Microsoft JhengHei', 'SimHei']:
        matplotlib.rcParams['font.sans-serif'] = [f]
        matplotlib.rcParams['axes.unicode_minus'] = False
        break
    matplotlib.rcParams.update({'font.size': 11, 'axes.titlesize': 12})

    spec = joblib.load(os.path.join(LOSO_MODEL_DIR, 'feature_spec.pkl'))
    ts_cols = spec['ts_cols']

    res = {}
    for s in PICKS:
        print(f"Computing {s} ...")
        res[s] = Z.run_subject(s, ts_cols)

    # ---------- waveform plots (one each for main / synergist) ----------
    for mi, mname in enumerate(MUSCLE):
        fig, axes = plt.subplots(len(PICKS), 1, figsize=(15, 3.4 * len(PICKS)))
        for ax, s in zip(axes, PICKS):
            r = res[s]
            n = min(len(r['y']), int(MAX_SEC * FPS))
            t = np.arange(n) / FPS
            ax.plot(t, r['y'][:n, mi] * 100, lw=1.3, color=C_TRUE, label='Measured EMG', zorder=3)
            ax.plot(t, r['zs'][:n, mi] * 100, lw=1.3, color=C_ZS, alpha=.88,
                    label='Zero-shot (never seen this person)', zorder=2)
            ax.plot(t, r['ft'][:n, mi] * 100, lw=1.3, color=C_FT, alpha=.88,
                    label='Fine-tuned (personalized)', zorder=2)
            for b in r['bounds']:
                if b / FPS <= MAX_SEC:
                    ax.axvline(b / FPS, color='#999', ls=':', lw=0.7, alpha=.6)
            rz, rf = r['m_zs']['r'][mi], r['m_ft']['r'][mi]
            nz, nf = r['m_zs']['nrmse'][mi], r['m_ft']['nrmse'][mi]
            ax.set_title(f"{s} — {LABELS[s]}    "
                         f"r {rz:.3f} → {rf:.3f}    nRMSE {nz:.3f} → {nf:.3f}",
                         fontweight='bold')
            ax.set_ylabel('%MVC')
            ax.grid(alpha=.22)
            ax.set_xlim(0, min(MAX_SEC, len(r['y']) / FPS))
        axes[0].legend(loc='upper right', ncol=3, fontsize=9.5, framealpha=.95)
        axes[-1].set_xlabel('Time (s; segment tails joined; grey dashed lines mark the seams)')
        fig.suptitle(f'{mname}: measured vs model prediction  (evaluation interval unseen by both models)',
                     fontsize=13.5, fontweight='bold')
        fig.tight_layout()
        tag = 'main' if mi == 0 else 'syn'
        p = os.path.join(RESULTS_DIR, f'selected_waveform_{tag}.png')
        fig.savefig(p, dpi=140); plt.close(fig)
        print(f"📊 Saved {p}")

    # ---------- scatter plot: diagnosing systematic bias ----------
    fig, axes = plt.subplots(2, len(PICKS), figsize=(5.0 * len(PICKS), 9))
    for ci, s in enumerate(PICKS):
        r = res[s]
        for mi in range(2):
            ax = axes[mi][ci]
            yt, yp, yf = r['y'][:, mi] * 100, r['zs'][:, mi] * 100, r['ft'][:, mi] * 100
            ax.scatter(yt, yp, s=3, alpha=.10, color=C_ZS, label='Zero-shot')
            ax.scatter(yt, yf, s=3, alpha=.10, color=C_FT, label='Fine-tuned')
            lim = [0, max(yt.max(), yp.max(), yf.max()) * 1.02]
            ax.plot(lim, lim, 'k--', lw=1.1, label='Ideal y=x')
            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_xlabel('Measured %MVC'); ax.set_ylabel('Predicted %MVC')
            bias = yf.mean() - yt.mean()
            ax.set_title(f"{s} · {MUSCLE[mi]}\nmean bias after fine-tuning {bias:+.1f} %MVC", fontsize=10.5)
            ax.grid(alpha=.2)
            if mi == 0 and ci == 0:
                lg = ax.legend(fontsize=8.5, loc='upper left')
                for h in lg.legend_handles:
                    try:
                        h.set_alpha(1)
                    except Exception:
                        pass
    fig.suptitle('Predicted vs measured: points off the diagonal = systematic over/underestimation',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    p = os.path.join(RESULTS_DIR, 'selected_scatter.png')
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"📊 Saved {p}")

    # ---------- numeric summary ----------
    print(f"\n{'='*70}")
    print(f"{'Subject':<6} {'Muscle':<6} {'r before→after':>18} {'nRMSE before→after':>20}")
    print("-"*70)
    for s in PICKS:
        r = res[s]
        for mi, nm in enumerate(['Main muscle', 'Synergist']):
            print(f"{s:<6} {nm:<6} "
                  f"{r['m_zs']['r'][mi]:7.3f} → {r['m_ft']['r'][mi]:7.3f}   "
                  f"{r['m_zs']['nrmse'][mi]:8.3f} → {r['m_ft']['nrmse'][mi]:8.3f}")


if __name__ == '__main__':
    main()
