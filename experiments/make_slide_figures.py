"""Generate simplified figures for projection (large fonts, few panels) for embedding in the HTML slides."""
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
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plot_zeroshot_vs_finetuned as Z   # noqa: E402
import joblib                            # noqa: E402

for f in ['Microsoft JhengHei', 'SimHei']:
    matplotlib.rcParams['font.sans-serif'] = [f]
    matplotlib.rcParams['axes.unicode_minus'] = False
    break
matplotlib.rcParams.update({'font.size': 15, 'axes.titlesize': 17,
                            'axes.labelsize': 14, 'xtick.labelsize': 12,
                            'ytick.labelsize': 12, 'legend.fontsize': 13})

C_TRUE, C_ZS, C_FT = '#15181a', '#e0623c', '#0f766e'
FPS = 60.0

spec = joblib.load(os.path.join(LOSO_MODEL_DIR, 'feature_spec.pkl'))
ts_cols = spec['ts_cols']

# ---------- figure 1: waveform comparison (S08 improved most, S04 barely improved) ----------
picks = ['S08', 'S04']
res = {}
for s in picks:
    print(f"Computing {s} ...")
    res[s] = Z.run_subject(s, ts_cols)

fig, axes = plt.subplots(2, 1, figsize=(16, 8.2))
for ax, s in zip(axes, picks):
    r = res[s]
    n = min(len(r['y']), int(45 * FPS))          # only plot the first 45 seconds so it is legible when projected
    t = np.arange(n) / FPS
    ax.plot(t, r['y'][:n, 1], lw=1.9, color=C_TRUE, label='Measured EMG (ground truth)', zorder=3)
    ax.plot(t, r['zs'][:n, 1], lw=1.9, color=C_ZS, alpha=.9,
            label='Zero-shot: the model has never seen this person', zorder=2)
    ax.plot(t, r['ft'][:n, 1], lw=1.9, color=C_FT, alpha=.9,
            label='Fine-tuned: calibrated with this person\'s data', zorder=2)
    rz, rf = r['m_zs']['r'][1], r['m_ft']['r'][1]
    nz, nf = r['m_zs']['nrmse'][1], r['m_ft']['nrmse'][1]
    tag = 'largest improvement' if s == 'S08' else 'r did not improve, but nRMSE did'
    ax.set_title(f"{s} ({tag})    r {rz:.2f} → {rf:.2f}    nRMSE {nz:.2f} → {nf:.2f}",
                 fontweight='bold')
    ax.set_ylabel('Synergist %MVC')
    ax.grid(alpha=.22)
axes[0].legend(loc='upper right', ncol=3, framealpha=.95)
axes[1].set_xlabel('Time (s)')
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'slide_waveform.png'), dpi=110)
plt.close(fig)
print("✅ slide_waveform.png")

# ---------- figure 2: LOSO before and after the fix ----------
subj = ['S02', 'S04', 'S06', 'S03', 'S07', 'S05', 'S01', 'S08']
before = [0.072, -0.003, 0.111, 0.364, 0.512, 0.631, 0.580, np.nan]
after = [0.657, 0.780, 0.734, 0.694, 0.782, 0.765, 0.627, 0.827]
x = np.arange(len(subj)); w = .38
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - w/2, [0 if np.isnan(v) else v for v in before], w,
       label='Before fix (mispaired)', color=C_ZS, alpha=.9)
ax.bar(x + w/2, after, w, label='After fix', color=C_FT, alpha=.9)
for i, (b, a) in enumerate(zip(before, after)):
    if not np.isnan(b):
        ax.annotate(f'{b:.2f}', (i - w/2, max(b, 0)), ha='center', va='bottom', fontsize=11)
    else:
        ax.annotate('Not included', (i - w/2, 0.02), ha='center', va='bottom', fontsize=10, color='#777')
    ax.annotate(f'{a:.2f}', (i + w/2, a), ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.axhline(0, color='k', lw=.8)
ax.set_xticks(x); ax.set_xticklabels(subj, fontsize=13)
ax.set_ylabel('Main muscle Pearson r')
ax.set_title('After fixing the data pairing: cross-subject (LOSO) performance  mean 0.32 → 0.73', fontweight='bold')
ax.legend(loc='upper left'); ax.grid(alpha=.22, axis='y')
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'slide_loso.png'), dpi=110)
plt.close(fig)
print("✅ slide_loso.png")

# ---------- figure 3: calibration curve (mean line) ----------
cal = pd.read_csv(os.path.join(RESULTS_DIR, 'calibration_curve.csv'))
agg = cal.groupby('k_segments')[['r_main', 'r_syn', 'nrmse_syn']].mean()
fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
ax = axes[0]
ax.plot(agg.index, agg['r_main'], marker='o', ms=9, lw=3, color=C_FT, label='Main muscle')
ax.plot(agg.index, agg['r_syn'], marker='s', ms=9, lw=3, color=C_ZS, label='Synergist')
ax.axvline(3, color='#0f766e', ls=':', lw=2)
ax.axvline(6, color='#e0623c', ls=':', lw=2)
ax.annotate('Main muscle saturates\nat about 2–3 sets', (3, .70), fontsize=12, color='#0f766e', ha='center')
ax.annotate('Synergist only improves\nclearly at 6 sets', (6.2, .53), fontsize=12, color='#e0623c')
ax.set_xlabel('Number of sets a new user must record, k'); ax.set_ylabel('Pearson r')
ax.set_title('How many sets are enough?', fontweight='bold'); ax.legend(); ax.grid(alpha=.22)

ax = axes[1]
ax.plot(agg.index, agg['nrmse_syn'], marker='s', ms=9, lw=3, color=C_ZS)
ax.axhline(1.0, color='crimson', ls='--', lw=2.2)
ax.annotate('nRMSE = 1\n(same as guessing the mean; the model adds no value)', (2.2, 1.03),
            fontsize=12, color='crimson')
ax.set_xlabel('Number of sets a new user must record, k'); ax.set_ylabel('Synergist nRMSE (lower is better)')
ax.set_title('The synergist needs 6 sets before it drops below the “guess the mean” line', fontweight='bold')
ax.grid(alpha=.22)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'slide_calibration.png'), dpi=110)
plt.close(fig)
print("✅ slide_calibration.png")
