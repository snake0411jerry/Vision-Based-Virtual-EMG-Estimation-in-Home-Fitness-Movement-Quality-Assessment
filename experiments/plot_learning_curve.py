"""
plot_learning_curve.py — plot the subject-count learning curve and extrapolate
=========================================================================
Reads the output of `learning_curve_subjects.py` and does three things:

  1. plot the "training-set size N vs performance" curve (main-muscle r and nRMSE)
  2. fit a power-saturation model r(N) = r_inf − a·N^(−b) to estimate the ceiling r_inf
  3. extrapolate to N = 15 / 20 / 30 / 50 with bootstrap uncertainty intervals

🔴 **Always read the extrapolation together with its interval.** The data only go to N=9; extrapolating to N=30 is
   more than three times the span. The power and logarithmic models fit almost identically for N≤9, yet give very
   different ceilings. What this plot can reliably answer is "**is the curve still rising**", not "what exactly is N=30".
   This script fits both models and prints both; their gap is a lower bound on the extrapolation uncertainty.

Usage:
    python experiments/plot_learning_curve.py
    python experiments/plot_learning_curve.py --metric nrmse_main --target 0.70
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import RESULTS_DIR, ensure_results_dir  # noqa: E402

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ACCENT, INK, GREY, RULE = '#D18A72', '#1A1A24', '#555868', '#E3E5EC'
INDIGO = '#2E3A87'

ap = argparse.ArgumentParser()
ap.add_argument('--prefix', default='learning_curve_subjects')
ap.add_argument('--metric', default='r_main',
                choices=['r_main', 'r_syn', 'nrmse_main', 'nrmse_syn'])
ap.add_argument('--target', type=float, default=0.85,
                help='Target level; the script back-solves how many people are needed')
ap.add_argument('--boot', type=int, default=2000)
args = ap.parse_args()

RAW = os.path.join(RESULTS_DIR, f'{args.prefix}_raw.csv')
PNG = os.path.join(RESULTS_DIR, f'{args.prefix}.png')
if not os.path.exists(RAW):
    sys.exit(f'{RAW} not found; run learning_curve_subjects.py first')

df = pd.read_csv(RAW)
M = args.metric
HIGHER_BETTER = M.startswith('r_')


def per_subject(d):
    """Average over repeats first, giving one value per (N, test subject)."""
    return d.groupby(['n_train', 'test_subject'], as_index=False)[M].mean()


ps = per_subject(df)
curve = ps.groupby('n_train', as_index=False)[M].agg(['mean', 'std', 'count'])
curve = curve.reset_index() if 'n_train' not in curve.columns else curve
Ns = curve['n_train'].values.astype(float)
ys = curve['mean'].values


# ---------------------------------------------------------------- models
def power_sat(n, r_inf, a, b):
    """Power saturation: r(N) = r_inf − a·N^(−b). Smaller b means slower decay (later saturation)."""
    return r_inf - a * np.power(n, -b)


def log_growth(n, c, d):
    """Logarithmic growth: r(N) = c + d·ln(N). No ceiling; the "keeps growing" control."""
    return c + d * np.log(n)


def fit_power(x, y):
    p0 = [max(y) + 0.05, 0.3, 0.5]
    lo = [-np.inf, 0, 0.05]
    hi = [np.inf, np.inf, 5.0]
    return curve_fit(power_sat, x, y, p0=p0, bounds=(lo, hi), maxfev=20000)[0]


def fit_log(x, y):
    return curve_fit(log_growth, x, y, p0=[y[0], 0.05], maxfev=20000)[0]


if len(Ns) < 3:
    sys.exit(f'Only {len(Ns)} values of N; at least 3 are needed to fit. Finish the experiment first.')

pw = fit_power(Ns, ys)
lg = fit_log(Ns, ys)

# ---------------------------------------------------------------- bootstrap
# resample by "subject" (subjects are the independent units, not windows or folds)
subjects = sorted(ps['test_subject'].unique())
rng = np.random.default_rng(42)
QUERY = np.array([15, 20, 30, 50], dtype=float)
boot_pw, boot_lg, boot_inf = [], [], []
wide = ps.pivot(index='test_subject', columns='n_train', values=M)
for _ in range(args.boot):
    pick = rng.choice(subjects, size=len(subjects), replace=True)
    yb = wide.loc[pick].mean(axis=0).values
    try:
        p = fit_power(Ns, yb)
        boot_pw.append(power_sat(QUERY, *p))
        boot_inf.append(p[0])
        boot_lg.append(log_growth(QUERY, *fit_log(Ns, yb)))
    except Exception:
        continue
boot_pw = np.array(boot_pw)
boot_lg = np.array(boot_lg)
boot_inf = np.array(boot_inf)


def ci(a, q=(2.5, 97.5)):
    return np.percentile(a, q[0]), np.percentile(a, q[1])


# ---------------------------------------------------------------- report
print('=' * 70)
print(f'Subject-count learning curve  metric = {M}')
print('=' * 70)
print(f"{'N':>3}  {'mean':>8}  {'between-subj sd':>11}  {'n':>3}")
for _, r in curve.iterrows():
    print(f"{int(r['n_train']):>3}  {r['mean']:>8.4f}  {r['std']:>11.4f}  {int(r['count']):>3}")

slope_tail = (ys[-1] - ys[-2]) / (Ns[-1] - Ns[-2])
print(f'\nTail slope (N={int(Ns[-2])}→{int(Ns[-1])}): {slope_tail:+.4f} per subject')
print('  → the curve ' + ('**is still rising**; recruiting more people helps' if (slope_tail > 0) == HIGHER_BETTER
                    else 'has flattened or reversed; the marginal benefit of more recruitment is low'))

print(f'\nPower-saturation fit: r_inf={pw[0]:.4f}  a={pw[1]:.4f}  b={pw[2]:.4f}')
lo, hi = ci(boot_inf)
print(f'  95% bootstrap interval of the ceiling r_inf: [{lo:.3f}, {hi:.3f}]')
print(f'Logarithmic fit: c={lg[0]:.4f}  d={lg[1]:.4f} (control model without a ceiling)')

print('\nExtrapolation (both models side by side; their gap is a lower bound on the extrapolation uncertainty):')
print(f"{'N':>4}  {'power sat.':>10}  {'95% CI':>18}  {'log growth':>10}")
for i, n in enumerate(QUERY):
    a, b = ci(boot_pw[:, i])
    print(f'{int(n):>4}  {power_sat(n, *pw):>10.3f}  [{a:>6.3f}, {b:>6.3f}]  '
          f'{log_growth(n, *lg):>10.3f}')

tgt = args.target
if HIGHER_BETTER and pw[0] > tgt:
    n_need = (pw[1] / (pw[0] - tgt)) ** (1.0 / pw[2])
    print(f'\nReaching {tgt} needs about {n_need:.0f} people (power model; extrapolation is highly unreliable, order-of-magnitude only)')
elif HIGHER_BETTER:
    print(f'\n🔴 The power model\'s ceiling {pw[0]:.3f} < target {tgt} — '
          f'**adding people alone cannot get there**; data quality or the task definition must change.')

# ---------------------------------------------------------------- plotting
fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
xs = np.linspace(Ns.min(), 32, 200)

band_lo = np.percentile([power_sat(xs, *fit_power(Ns, wide.loc[
    rng.choice(subjects, len(subjects), True)].mean(axis=0).values))
    for _ in range(200)], 2.5, axis=0)
band_hi = np.percentile([power_sat(xs, *fit_power(Ns, wide.loc[
    rng.choice(subjects, len(subjects), True)].mean(axis=0).values))
    for _ in range(200)], 97.5, axis=0)
ax.fill_between(xs, band_lo, band_hi, color=ACCENT, alpha=0.13, lw=0)
ax.plot(xs, power_sat(xs, *pw), color=ACCENT, lw=1.8, label='Power-saturation fit')
ax.plot(xs, log_growth(xs, *lg), color=INDIGO, lw=1.4, ls='--', label='Log growth (no ceiling)')
ax.errorbar(Ns, ys, yerr=curve['std'] / np.sqrt(curve['count']),
            fmt='o', color=INK, ms=5, capsize=3, lw=1.2, label='Measured (±SE)')
ax.axvline(9.5, color=RULE, lw=1.2)
ax.text(9.8, ax.get_ylim()[0], ' extrapolation to the right', color=GREY, fontsize=8, va='bottom')
ax.set_xlabel('Number of training subjects N')
ax.set_ylabel({'r_main': 'Main muscle r', 'r_syn': 'Synergist r',
               'nrmse_main': 'Main muscle nRMSE', 'nrmse_syn': 'Synergist nRMSE'}[M])
ax.set_title('Subject-count learning curve (mean over 10 LOSO test subjects)', fontsize=11)
ax.grid(alpha=0.25, lw=0.6)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.legend(frameon=False, fontsize=8.5, loc='lower right')
fig.tight_layout()
fig.savefig(PNG)
print(f'\n✅ {PNG}')
