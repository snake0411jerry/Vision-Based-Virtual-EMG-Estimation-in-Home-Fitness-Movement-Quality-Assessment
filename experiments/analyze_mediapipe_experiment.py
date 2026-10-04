"""
analyze_mediapipe_experiment.py — paired comparisons for the MediaPipe vs OpenCap retraining experiment
=========================================================================
Reads the output of `loso_multiseed.py --prefix loso_mediapipe` and runs two kinds of paired test:

  Paired across seeds (n = number of seeds)
      For each seed, average over the 10 subjects, then subtract the two arms.
      This is the primary test prescribed by the handover doc's <Methodology rules>, computed
      the same way as the p-values on slides 21/23/26. n is small (usually 3–5), so only large
      differences can be detected.

  Paired across subjects (n = 10)
      First average over seeds to get one value per subject, then subtract the two arms.
      n is somewhat larger, but between-subject variation is large (sd up to 0.12), which is
      exactly why pairing matters.
      ⚠️ The two tests have different n and test different things — do not report only the
         one that looks better.

🔴 **Only compare arms from the same job.** The handover doc records a cross-run dispersion
   (≈0.025) larger than the within-run between-seed sd (0.013–0.017), so do not subtract this
   run's MediaPipe numbers from the 0.766 on the slides. This script only handles arms within
   the same raw CSV precisely to prevent that.

Usage:
    python experiments/analyze_mediapipe_experiment.py
    python experiments/analyze_mediapipe_experiment.py --prefix loso_mediapipe \
        --pairs oc16:mp16 oc8:mp8 mp16:mp8
Output:
    results/<prefix>_pairs.csv
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import RESULTS_DIR, ensure_results_dir  # noqa: E402

import argparse  # noqa: E402
import itertools  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

METRICS = ['r_main', 'r_syn', 'nrmse_main', 'nrmse_syn']


def paired(a, b):
    """Return (mean of differences, sd of differences, t, p, n). Returns nan when n<2."""
    d = np.asarray(b) - np.asarray(a)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return d.mean() if len(d) else np.nan, np.nan, np.nan, np.nan, len(d)
    t, p = stats.ttest_rel(np.asarray(b)[:len(d)], np.asarray(a)[:len(d)])
    return float(d.mean()), float(d.std(ddof=1)), float(t), float(p), len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', default='loso_mediapipe')
    ap.add_argument('--pairs', nargs='*', default=None,
                    help='Pairs to compare, format base:comparison (multiple allowed). If omitted, every pair is compared.')
    args = ap.parse_args()

    ensure_results_dir()
    raw_p = os.path.join(RESULTS_DIR, f'{args.prefix}_raw.csv')
    if not os.path.exists(raw_p):
        sys.exit(f'Not found: {raw_p}\n'
                 f'Run loso_multiseed.py --prefix {args.prefix} --arms ... first')
    raw = pd.read_csv(raw_p)
    arms = list(dict.fromkeys(raw['arm']))
    seeds = sorted(raw['seed'].unique())
    subjects = sorted(raw['Subject'].unique())

    print('=' * 84)
    print(f'MediaPipe vs OpenCap retraining experiment  {raw_p}')
    print('=' * 84)
    print(f'arms: {arms}')
    print(f'seeds: {seeds}  (each arm x seed = {len(subjects)} folds)')

    # ---- chunk: loso_multiseed.py --resume records which execution run produced each arm ----
    chunk_of = {}
    if 'chunk' in raw.columns:
        chunk_of = {a: sorted(set(raw[raw.arm == a]['chunk'])) for a in arms}
        print('chunk (execution batch): ' +
              '  '.join(f'{a}=chunk{c[0] if len(c) == 1 else c}'
                        for a, c in chunk_of.items()))
        if len({tuple(c) for c in chunk_of.values()}) > 1:
            print('🔴 This file contains multiple chunks. **Arms from different chunks must not be subtracted** —')
            print('   the handover doc records a cross-run dispersion of ≈0.025, larger than the within-run between-seed sd'
                  ' (0.013–0.017).')
            print('   Each comparison below is flagged if it crosses chunks.')

    # Completeness check: an arm with missing folds cannot be compared with others
    cnt = raw.groupby(['arm', 'seed']).size().unstack(fill_value=0)
    if (cnt != len(subjects)).any().any():
        print('\n[Note] Some arm x seed combinations do not have '
              f'{len(subjects)} folds (possibly still running or interrupted):')
        print(cnt.to_string())

    # ---------------- per-arm summary ----------------
    print('\n' + '-' * 84)
    print('Per-arm means (average over the 10 subjects first, then mean +- sd across seeds)')
    print('-' * 84)
    print(f"{'arm':<12}" + ''.join(f'{m:>20}' for m in METRICS))
    per_seed = raw.groupby(['arm', 'seed'], as_index=False)[METRICS].mean()
    for a in arms:
        g = per_seed[per_seed.arm == a]
        line = f'{a:<12}'
        for m in METRICS:
            line += f'{g[m].mean():>13.4f}±{g[m].std(ddof=1):<6.4f}'
        print(line)

    # ---------------- paired comparisons ----------------
    if args.pairs:
        combos = []
        for s in args.pairs:
            x, _, y = s.partition(':')
            if x not in arms or y not in arms:
                sys.exit(f'--pairs refers to a non-existent arm: {s} (available: {arms})')
            combos.append((x, y))
    else:
        combos = list(itertools.combinations(arms, 2))

    per_subj = raw.groupby(['arm', 'Subject'], as_index=False)[METRICS].mean()
    rows = []
    for base, comp in combos:
        print('\n' + '=' * 84)
        same_chunk = (chunk_of.get(base) == chunk_of.get(comp)
                      if chunk_of else None)
        warn = ''
        if same_chunk is False:
            warn = (f'  🔴 crosses chunks ({chunk_of[base]} vs {chunk_of[comp]})'
                    f': direction only, not a quantitative conclusion')
        print(f'{comp}  vs  {base}  (diff = {comp} − {base}){warn}')
        print('=' * 84)
        print(f"{'Metric':<12}{'Test':<16}{'Diff':>10}{'sd':>9}{'t':>8}{'p':>9}{'n':>4}")
        for m in METRICS:
            for how, key in (('across seeds', 'seed'), ('across subjects', 'Subject')):
                src = per_seed if key == 'seed' else per_subj
                idx = 'seed' if key == 'seed' else 'Subject'
                a = src[src.arm == base].set_index(idx)[m]
                b = src[src.arm == comp].set_index(idx)[m]
                common = a.index.intersection(b.index)
                dm, ds, t, p, n = paired(a.loc[common].values, b.loc[common].values)
                sig = ''
                if np.isfinite(p):
                    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                print(f"{m:<12}{how:<16}{dm:>+10.4f}{ds:>9.4f}{t:>8.2f}{p:>9.4f}{n:>4} {sig}")
                rows.append({'base': base, 'comp': comp, 'metric': m, 'unit': key,
                             'delta': dm, 'delta_sd': ds, 't': t, 'p': p, 'n': n,
                             'same_chunk': same_chunk})

        # Per-subject detail (main-muscle r only — the paper's primary metric)
        a = per_subj[per_subj.arm == base].set_index('Subject')['r_main']
        b = per_subj[per_subj.arm == comp].set_index('Subject')['r_main']
        print('\n  Per-subject main-muscle r:')
        for s in subjects:
            if s in a.index and s in b.index:
                print(f'    {s}  {a[s]:.3f} -> {b[s]:.3f}  ({b[s] - a[s]:+.3f})')

    out = os.path.join(RESULTS_DIR, f'{args.prefix}_pairs.csv')
    pd.DataFrame(rows).to_csv(out, index=False, encoding='utf-8-sig')
    print(f'\n[done] {out}')
    print('\nDecision threshold (from the handover doc): within-run between-seed sd ≈ 0.013–0.017,')
    print('  |diff| < 0.02 counts as “indistinguishable” — do not write “identical”; that requires an equivalence test (TOST).')


if __name__ == '__main__':
    main()
