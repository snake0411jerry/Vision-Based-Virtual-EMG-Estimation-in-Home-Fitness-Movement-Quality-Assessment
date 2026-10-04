# -*- coding: utf-8 -*-
"""Step 4: verify that the coordinate-axis fix took effect.
Compares Dataset/Combined_before_axisfix/ (old) with Dataset/Combined/ (new).
Criteria (CHANGES_REQUIRED.md section 6):
  trunk angle p95, compensation sets <9deg -> 35~50deg; normal sets <7deg -> 20~27deg
  knee-angle minimum 100~120deg -> 60~90deg
  Comp_Trunk_Lean trigger rate 0.0% -> compensation sets >0
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
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

FIXED_DIR = PIPELINE_DIR
sys.path.insert(0, FIXED_DIR)

DATASET = DATASET_DIR
OLD = os.path.join(DATASET, "Combined_before_axisfix")
NEW = os.path.join(DATASET, "Combined")

# take SUBJECTS (seg -> trc file-name mapping) from Features_insert_FIXED.py without triggering its main flow
import ast
src = open(os.path.join(FIXED_DIR, "Features_insert_FIXED.py"), encoding='utf-8').read()
tree = ast.parse(src)
SUBJECTS = None
for node in tree.body:
    if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == 'SUBJECTS':
        SUBJECTS = ast.literal_eval(node.value)
assert SUBJECTS is not None

seg2trc = {}
for s in SUBJECTS:
    for trc, sid in zip(s["trc_order"], s["segment_ids"]):
        seg2trc[(s["key"], sid)] = trc


def group_of(trc):
    """Compensation sets = names starting with a/b/c; normal bodyweight = starting with 0; the rest are normal loaded sets."""
    base = trc.replace('.trc', '')
    if base[0] in 'abc':
        return 'comp_' + base[0]
    if base[0] == '0':
        return 'normal_bw'
    return 'normal_load'


rows = []
for fn in sorted(os.listdir(NEW)):
    if not fn.endswith('_Combined_Features.csv'):
        continue
    subj, _, sid, _, _ = fn.split('_', 4)
    sid = int(sid)
    trc = seg2trc.get((subj, sid), '?')
    rec = {'subject': subj, 'seg': sid, 'trc': trc, 'group': group_of(trc)}
    for tag, d in (('old', OLD), ('new', NEW)):
        df = pd.read_csv(os.path.join(d, fn))
        trunk = df['Trunk_Lean_Angle_norm'].values * 180.0
        knee = df['Knee_Angle_norm'].values * 180.0
        rec[f'trunk_p95_{tag}'] = np.percentile(trunk, 95)
        rec[f'knee_min_{tag}'] = knee.min()
        rec[f'trunk_std_{tag}'] = trunk.std()
        # Comp_Trunk_Lean trigger rate (the training script uses norm > 40/180)
        rec[f'comp_trunk_rate_{tag}'] = float((df['Trunk_Lean_Angle_norm'] > 40.0 / 180.0).mean())
    rows.append(rec)

res = pd.DataFrame(rows)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'axisfix_verify.csv')
res.to_csv(out, index=False, encoding='utf-8-sig')

pd.set_option('display.width', 200)

print("=" * 78)
print("Summary by group (angles in degrees)")
print("=" * 78)
g = res.groupby('group').agg(
    n=('trc', 'size'),
    trunk_p95_old=('trunk_p95_old', 'mean'),
    trunk_p95_new=('trunk_p95_new', 'mean'),
    knee_min_old=('knee_min_old', 'mean'),
    knee_min_new=('knee_min_new', 'mean'),
    trunk_std_old=('trunk_std_old', 'mean'),
    trunk_std_new=('trunk_std_new', 'mean'),
    comp_rate_old=('comp_trunk_rate_old', 'mean'),
    comp_rate_new=('comp_trunk_rate_new', 'mean'),
).round(3)
print(g.to_string())

print()
print("=" * 78)
print("Criteria check")
print("=" * 78)
comp = res[res['group'].str.startswith('comp_')]
norm = res[res['group'].str.startswith('normal_')]
c_trunk = res[res['group'] == 'comp_c']

def chk(label, val, lo, hi, unit='deg'):
    ok = "OK " if lo <= val <= hi else "!! "
    print(f"{ok}{label}: {val:.2f}{unit}  (expected {lo}~{hi}{unit})")

chk("Mean trunk angle p95 — all compensation sets (a/b/c)", comp['trunk_p95_new'].mean(), 20, 55)
chk("Mean trunk angle p95 — trunk-lean sets (c_*)", c_trunk['trunk_p95_new'].mean(), 35, 50)
chk("Mean trunk angle p95 — normal sets", norm['trunk_p95_new'].mean(), 15, 30)
chk("Mean knee-angle minimum — all", res['knee_min_new'].mean(), 60, 95)
print(f"   (before the change: compensation sets {comp['trunk_p95_old'].mean():.2f}deg / normal sets "
      f"{norm['trunk_p95_old'].mean():.2f}deg / knee-angle minimum {res['knee_min_old'].mean():.2f}deg)")

print()
print(f"Trunk-angle std amplification factor (mean over all files): "
      f"{(res['trunk_std_new'] / res['trunk_std_old']).mean():.2f}x  (document expects ~6x)")

print()
print(f"Comp_Trunk_Lean trigger rate  old: whole dataset {res['comp_trunk_rate_old'].mean() * 100:.3f}% "
      f"/ files with any trigger {(res['comp_trunk_rate_old'] > 0).sum()}/{len(res)}")
print(f"Comp_Trunk_Lean trigger rate  new: whole dataset {res['comp_trunk_rate_new'].mean() * 100:.3f}% "
      f"/ files with any trigger {(res['comp_trunk_rate_new'] > 0).sum()}/{len(res)}")
print(f"   of which compensation sets triggered: {(comp['comp_trunk_rate_new'] > 0).sum()}/{len(comp)}"
      f", normal sets triggered: {(norm['comp_trunk_rate_new'] > 0).sum()}/{len(norm)}")

print()
print("=" * 78)
print("Were other columns changed by accident (they should be identical)")
print("=" * 78)
sample = sorted(os.listdir(NEW))[0]
a = pd.read_csv(os.path.join(OLD, sample))
b = pd.read_csv(os.path.join(NEW, sample))
assert list(a.columns) == list(b.columns), "Number/names of columns changed!"
print(f"Columns: {len(a.columns)} (same in old and new)")
changed = [c for c in a.columns if not np.allclose(a[c].values, b[c].values, atol=1e-9)]
print(f"{sample} changed columns: {changed}")

print(f"\nDetails written to: {out}")
