# -*- coding: utf-8 -*-
"""Step 6: compare LOSO zero-shot results before and after the coordinate-axis fix."""
import os
import sys

import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import RESULTS_DIR  # noqa: E402

old = pd.read_csv(os.path.join(RESULTS_DIR, 'loso_zeroshot_metrics_before_axisfix.csv')).set_index('Subject')
new = pd.read_csv(os.path.join(RESULTS_DIR, 'loso_zeroshot_metrics.csv')).set_index('Subject')

cols = ['r_main', 'r_syn', 'nrmse_main', 'nrmse_syn']
cmp = pd.DataFrame(index=old.index)
for c in cols:
    cmp[f'{c}_old'] = old[c]
    cmp[f'{c}_new'] = new[c]
    cmp[f'{c}_d'] = new[c] - old[c]

pd.set_option('display.width', 250)
print("=" * 100)
print("Per subject (d = new − old; higher r is better, lower nRMSE is better)")
print("=" * 100)
print(cmp[['r_main_old', 'r_main_new', 'r_main_d',
           'r_syn_old', 'r_syn_new', 'r_syn_d']].round(3).to_string())
print()
print(cmp[['nrmse_main_old', 'nrmse_main_new', 'nrmse_main_d',
           'nrmse_syn_old', 'nrmse_syn_new', 'nrmse_syn_d']].round(3).to_string())


def block(title, idx):
    o, n = old.loc[idx], new.loc[idx]
    print(f"\n--- {title} (n={len(idx)}) ---")
    for c in cols:
        d = n[c].mean() - o[c].mean()
        arrow = "↑" if d > 0 else "↓"
        good = (d > 0) if c.startswith('r_') else (d < 0)
        print(f"  {c:<12} {o[c].mean():.3f} → {n[c].mean():.3f}   "
              f"{arrow}{abs(d):.3f}  {'better' if good else 'worse'}")


print("\n" + "=" * 100)
print("Summary")
print("=" * 100)
block("All 10 subjects", old.index.tolist())
block("Synergist convention: excluding S09", [s for s in old.index if s != 'S09'])
print("\nNumber improved: "
      f"main r {(cmp['r_main_d'] > 0).sum()}/{len(cmp)}, "
      f"synergist r {(cmp['r_syn_d'] > 0).sum()}/{len(cmp)}")

out = os.path.join(EXP, 'axisfix_loso_comparison.csv')
cmp.round(4).to_csv(out, encoding='utf-8-sig')
print(f"\nDetails written to: {out}")
