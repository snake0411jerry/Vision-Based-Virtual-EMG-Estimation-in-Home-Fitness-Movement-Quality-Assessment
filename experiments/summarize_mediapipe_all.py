"""
summarize_mediapipe_all.py — consolidate every MediaPipe single-camera experiment into one report
=========================================================================
This line of work ran six experiments whose results are spread over seven CSVs. This script pulls them into one coherent report
and **automatically runs the paired tests and prints the warnings that should be there**, so nothing has to be re-derived each time.

Files read (missing ones are skipped with a note, never a crash):
    mediapipe_alignment_report.csv          data alignment validation
    mediapipe_feature_agreement_summary.csv per-feature agreement
    mediapipe_axis_snr.csv                  per-axis signal-to-noise
    mediapipe_crossdomain_raw.csv           no retraining, scaler kept
    mediapipe_crossdomain_refit_raw.csv     no retraining, scaler refit
    loso_mediapipe_raw.csv                  full retraining of eight arms
    loso_dualdomain_raw.csv                 dual-domain joint training
    dualdomain_finetune_raw.csv             dual-domain + personalized fine-tuning

🔴 Basis warnings (restated in the output):
   - `loso_*` uses the **full-window** basis; `dualdomain_finetune` uses the **last 20% of each segment** basis.
     Their absolute values must not be subtracted; only look at "differences within the same basis".
   - Arms from different `chunk`s (different runs) must not be subtracted — cross-run dispersion ≈0.025,
     larger than the within-run between-seed sd (0.013–0.017).
   - Paired tests always report both "across seeds" and "across subjects"; never pick the better-looking one.

Usage:
    python experiments/summarize_mediapipe_all.py
    python experiments/summarize_mediapipe_all.py --out ../report.md
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import RESULTS_DIR  # noqa: E402

import argparse  # noqa: E402
import io  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

OUT = io.StringIO()


def say(*a):
    line = ' '.join(str(x) for x in a)
    print(line)
    OUT.write(line + '\n')


def head(title, ch='='):
    say('\n' + ch * 88)
    say(title)
    say(ch * 88)


def read(name):
    p = os.path.join(RESULTS_DIR, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def paired(a, b):
    """Return (mean difference, sd of differences, p). b − a."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    if len(d) < 2:
        return (d.mean() if len(d) else np.nan), np.nan, np.nan
    return float(d.mean()), float(d.std(ddof=1)), float(stats.ttest_rel(b, a)[1])


def stars(p):
    if not np.isfinite(p):
        return ''
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''


def two_tests(df, base, comp, metric, armcol='arm'):
    """Run the paired test both across seeds and across subjects."""
    res = {}
    for how, keys in (('across seeds', ['seed']), ('across subjects', ['Subject'])):
        g = df.groupby([armcol] + keys, as_index=False)[metric].mean()
        a = g[g[armcol] == base].set_index(keys[0])[metric]
        b = g[g[armcol] == comp].set_index(keys[0])[metric]
        k = a.index.intersection(b.index)
        res[how] = paired(a.loc[k].values, b.loc[k].values)
    return res


def show_pair(df, base, comp, metrics, armcol='arm', note=''):
    say(f'\n  {comp} − {base}{note}')
    say(f"    {'Metric':<12}{'Test':<12}{'Diff':>10}{'sd':>9}{'p':>9}")
    for m in metrics:
        r = two_tests(df, base, comp, m, armcol)
        for how, (dm, ds, p) in r.items():
            say(f'    {m:<12}{how:<12}{dm:>+10.4f}{ds:>9.4f}{p:>9.4f} {stars(p)}')


# ──────────────────────────────────────────────── 1. Data preparation
head('1. Data preparation and validation')
al = read('mediapipe_alignment_report.csv')
if al is None:
    say('  (missing mediapipe_alignment_report.csv)')
else:
    say(f"  aligned {len(al)} / 89 segments  mode distribution {al['mode'].value_counts().to_dict()}")
    say(f"  left knee height correlation (OpenCap vs aligned MediaPipe)"
        f"  median {al.r_knee_height.median():.3f}"
        f"  min {al.r_knee_height.min():.3f}")
    nz = al[al.best_shift_frames != 0]
    say(f'  shift search: best shift is 0 for {len(al)-len(nz)} / {len(al)} segments, '
        f'all others within ±{int(al.best_shift_frames.abs().max())} frames (MediaPipe smoothing delay, deliberately not corrected)')

# ──────────────────────────────────────────────── 2. Feature-level agreement
head('2. Feature agreement and per-axis signal-to-noise')
IN_MODEL_16 = ['L_Heel_Rise_norm', 'R_Heel_Rise_norm', 'Knee_Ankle_Ratio_norm',
               'Trunk_Lean_Angle_norm', 'Shoulder_Y_norm', 'Shoulder_Z_norm',
               'Knee_X_norm', 'Knee_Y_norm', 'Knee_Z_norm', 'Ankle_X_norm',
               'Ankle_Y_norm', 'Ankle_Z_norm', 'Toe_X_norm', 'Toe_Y_norm',
               'Toe_Z_norm', 'Knee_Angle_norm']

fa = read('mediapipe_feature_agreement_summary.csv')
if fa is not None:
    # the old CSV includes Knee_Toe_Diff_* and Subj_*, but those are kept out of the model by eval_utils.EXCLUDE_ALWAYS
    # and listing them would only mislead. Only the 16 dims that really enter the model are kept here.
    d = fa[(fa.kind == 'norm') & (fa.feature.isin(IN_MODEL_16))]
    d = d.sort_values('r_median', ascending=False)
    say(f"  {'Feature':<24}{'r median':>9}{'r 10%':>10}{'bias/sd':>10}{'sdRatio':>8}")
    for _, r in d.iterrows():
        say(f"  {r['feature']:<24}{r['r_median']:>9.3f}{r['r_q10']:>10.3f}"
            f"{r['bias_over_oc_sd']:>10.2f}{r['sd_ratio_median']:>8.2f}")
snr = read('mediapipe_axis_snr.csv')
if snr is not None:
    g = snr.groupby('axis').agg(signal=('signal_sd', 'median'),
                                noise=('noise_sd', 'median')).reset_index()
    g['SNR'] = g.signal / g.noise
    say('\n  Per-axis SNR (signal = OpenCap sd; noise = sd of the difference; units are fractions of body height)')
    lab = {'X': 'X anterior-posterior (depth axis)', 'Y': 'Y vertical (image plane)', 'Z': 'Z left-right (image plane)'}
    for _, r in g.iterrows():
        say(f"    {lab[r['axis']]:<22} signal {r['signal']:.4f}  noise {r['noise']:.4f}  SNR {r['SNR']:.2f}")
    say('    → the noise floor is nearly the same on all three axes; Z fails because its signal is too small (~2cm lateral motion in a squat),')
    say('      not because it is the depth axis. **Adding depth cannot fix this.**')

# ──────────────────────────────────────────────── 3. Deployment without retraining
head('3. No retraining: feed MediaPipe straight into the OpenCap model')
for tag, name in (('scaler unchanged', 'mediapipe_crossdomain_raw.csv'),
                  ('scaler refit only', 'mediapipe_crossdomain_refit_raw.csv')):
    cd = read(name)
    if cd is None:
        say(f'  (missing {name})')
        continue
    p = cd.pivot(index='Subject', columns='arm', values='r_main')
    say(f"  {tag:<18} OpenCap {p['opencap'].mean():.4f}  →  "
        f"MediaPipe {p['mediapipe'].mean():.4f}  "
        f" (Δ {p['mediapipe'].mean()-p['opencap'].mean():+.4f})")
say('  ⚠️ The baseline comes from models/loso_zeroshot (recorded in the handover doc as a low draw); '
    'Δ is comparable, the absolute values are not.')

# ──────────────────────────────────────────────── 4. Full retraining: eight arms
head('4. Full retraining  eight arms × 3 seeds (full-window basis)')
r1 = read('loso_mediapipe_raw.csv')
M4 = ['r_main', 'nrmse_main', 'r_syn', 'nrmse_syn']
if r1 is None:
    say('  (missing loso_mediapipe_raw.csv)')
else:
    ps = r1.groupby(['arm', 'seed'], as_index=False)[M4].mean()
    say(f"  {'arm':<8}{'chunk':<8}" + ''.join(f'{m:>20}' for m in M4))
    for arm in ['oc7', 'oc8tf', 'oc8', 'oc16', 'mp8', 'mp8tf', 'mp7', 'mp16']:
        g = ps[ps.arm == arm]
        if not len(g):
            continue
        ck = sorted(set(r1[r1.arm == arm]['chunk'])) if 'chunk' in r1.columns else []
        line = f'  {arm:<8}{str(ck):<8}'
        for m in M4:
            line += f'{g[m].mean():>13.4f}±{g[m].std(ddof=1):<6.4f}'
        say(line)
    say('\n  Paired differences within the same feature set (mp − oc):')
    for a, b in (('oc8', 'mp8'), ('oc7', 'mp7'), ('oc8tf', 'mp8tf'), ('oc16', 'mp16')):
        if not ((r1.arm == a).any() and (r1.arm == b).any()):
            continue
        ck_a = sorted(set(r1[r1.arm == a]['chunk'])) if 'chunk' in r1.columns else []
        ck_b = sorted(set(r1[r1.arm == b]['chunk'])) if 'chunk' in r1.columns else []
        note = '  🔴 crosses chunks, direction only' if ck_a != ck_b else ''
        show_pair(r1, a, b, ['r_main', 'nrmse_main'], note=note)
    say('\n  The three feature-engineering attempts (all within chunk 3, subtractable):')
    for a, b in (('mp8', 'mp7'), ('mp8', 'mp8tf'), ('oc8', 'oc7'), ('oc8', 'oc8tf')):
        if (r1.arm == a).any() and (r1.arm == b).any():
            show_pair(r1, a, b, ['r_main'])

# ──────────────────────────────────────────────── 5. Dual-domain joint training
head('5. Dual-domain joint training (full-window basis, 8 dims)')
r2 = read('loso_dualdomain_raw.csv')
if r2 is None:
    say('  (missing loso_dualdomain_raw.csv)')
else:
    # 🔴 arms may have different numbers of seeds (still running, or an arm crashed and was rerun).
    #    Putting means over different bases side by side would mislead, so only seeds present in every arm are used.
    by_arm = {a: set(g.seed.unique()) for a, g in r2.groupby('arm')}
    common = sorted(set.intersection(*by_arm.values())) if by_arm else []
    if any(set(v) != set(common) for v in by_arm.values()):
        say('  ⚠️ arms have unequal seeds: ' +
            '  '.join(f'{a}={sorted(v)}' for a, v in by_arm.items()))
        say(f'  → only the common seeds are used below: {common}')
    r2 = r2[r2.seed.isin(common)]
    say(f'  seeds = {common} (n = {len(common)})')
    ps = r2.groupby(['arm', 'test_domain', 'seed'], as_index=False)[M4].mean()
    say(f"  {'Train skeleton':<10}{'Test domain':<12}" + ''.join(f'{m:>20}' for m in M4))
    for arm in ['oc_only', 'mp_only', 'both']:
        for dom in ['opencap', 'mediapipe']:
            g = ps[(ps.arm == arm) & (ps.test_domain == dom)]
            if not len(g):
                continue
            line = f'  {arm:<10}{dom:<12}'
            for m in M4:
                line += f'{g[m].mean():>13.4f}±{g[m].std(ddof=1):<6.4f}'
            say(line)
    mp = r2[r2.test_domain == 'mediapipe']
    say('\n  ★ Key cell (testing on MediaPipe):')
    for a, b in (('mp_only', 'both'), ('oc_only', 'both')):
        if (mp.arm == a).any() and (mp.arm == b).any():
            show_pair(mp, a, b, ['r_main', 'nrmse_main'])
    oc = r2[r2.test_domain == 'opencap']
    if (oc.arm == 'oc_only').any() and (oc.arm == 'both').any():
        say('\n  Control (testing on OpenCap, confirming joint training does not hurt multi-camera):')
        show_pair(oc, 'oc_only', 'both', ['r_main'])
    # domain gap
    pj = r2.groupby(['arm', 'test_domain', 'Subject'])['r_main'].mean()
    try:
        g1 = pj['oc_only', 'opencap'] - pj['mp_only', 'mediapipe']
        g2 = pj['both', 'opencap'] - pj['both', 'mediapipe']
        say(f'\n  domain gap  control {g1.mean():+.4f} (sd {g1.std(ddof=1):.3f})'
            f'  →  joint training {g2.mean():+.4f} (sd {g2.std(ddof=1):.3f})'
            f'  reduced by {100*(1-g2.mean()/g1.mean()):.0f}%')
    except KeyError:
        pass

# ──────────────────────────────────────────────── 6. Personalized fine-tuning
head('6. Dual-domain joint training + personalized fine-tuning (basis: last 20% of each segment; not subtractable from the above)')
r3 = read('dualdomain_finetune_raw.csv')
# the variant experiment (ft_src=both) is stored separately; the old file has no ft_src column, so it is filled as "FT source = eval domain" before merging
r3b = read('dualdomain_finetune_v2_raw.csv')
if r3 is not None and 'ft_src' not in r3.columns:
    r3 = r3.copy()
    r3['ft_src'] = r3['domain']
    r3['run'] = 'v1'
if r3b is not None:
    r3b = r3b.copy()
    r3b['run'] = 'v2'
    r3 = r3b if r3 is None else pd.concat([r3, r3b], ignore_index=True)
if r3 is None:
    say('  (missing dualdomain_finetune_raw.csv — the experiment has not finished)')
else:
    by_arm = {a: set(g.seed.unique()) for a, g in r3.groupby('arm')}
    common = sorted(set.intersection(*by_arm.values())) if by_arm else []
    if any(set(v) != set(common) for v in by_arm.values()):
        say('  ⚠️ arms have unequal seeds: ' +
            '  '.join(f'{a}={sorted(v)}' for a, v in by_arm.items()))
        say(f'  → only the common seeds are used below: {common}')
    r3 = r3[r3.seed.isin(common)]
    say(f'  seeds = {common} (n = {len(common)})  fine-tuning: 20 epochs, **no early stopping**')
    gcols = ['arm', 'ft_src', 'domain', 'seed'] + (['run'] if 'run' in r3.columns else [])
    ps = r3.groupby(gcols, as_index=False).mean(numeric_only=True)
    say(f"\n  {'Base':<9}{'FT source':<11}{'Eval dom':<11}{'run':<5}{'zero-shot r':>11}"
        f"{'FT r':>11}{'gain':>9}{'leak':>10}")
    # 🔴 runs (v1/v2) are different executions; **the same cell must not be averaged across runs** — cross-run dispersion
    #    ≈0.025 is larger than the effect sizes of interest here. So each run is listed separately.
    runs = sorted(r3['run'].unique()) if 'run' in r3.columns else ['']
    for arm in sorted(r3.arm.unique()):
      for fs in sorted(r3.ft_src.unique()):
        for dom in ('mediapipe', 'opencap'):
          for rn in runs:
            g = ps[(ps.arm == arm) & (ps.ft_src == fs) & (ps.domain == dom)]
            if 'run' in ps.columns:
                g = g[g['run'] == rn]
            if not len(g):
                continue
            run = rn
            leak = g.r_main_oracle.mean() - g.r_main.mean()
            leak_s = '  n/a' if not np.isfinite(leak) else f'{leak:>+10.4f}'
            say(f'  {arm:<9}{fs:<11}{dom:<11}{run:<5}{g.r_main_zs.mean():>11.4f}'
                f'{g.r_main.mean():>11.4f}'
                f'{g.r_main.mean()-g.r_main_zs.mean():>+9.4f}{leak_s}')
    say('\n  “leak” = best epoch picked after the fact − full 20 epochs.')
    say('   This is the spurious gain that phase2_finetune_loso.py\'s EarlyStopping(restore_best_weights)')
    say('   obtains by picking weights on the test set. Cite “FT r” in the paper.')

    mp = r3[r3.domain == 'mediapipe']
    say('\n  ★ After personalization, does dual-domain joint training still help (testing on MediaPipe):')
    if (mp.arm == 'mp_only').any() and (mp.arm == 'both').any():
        show_pair(mp, 'mp_only', 'both', ['r_main', 'nrmse_main'])
    say('\n  ★ After personalization, how far behind is the single camera (same arm, the two domains compared):')
    for arm in sorted(r3.arm.unique()):
        q = r3[r3.arm == arm]
        g = q.groupby(['domain', 'Subject'])['r_main'].mean()
        try:
            d = g['mediapipe'] - g['opencap']
            t = stats.ttest_rel(g['mediapipe'].values, g['opencap'].values)[1]
            say(f'    {arm:<10} MediaPipe − OpenCap = {d.mean():+.4f}'
                f' (sd {d.std(ddof=1):.3f}, p={t:.4f} {stars(t)})')
        except KeyError:
            pass
    say('\n  Per subject (testing on MediaPipe, mean across seeds):')
    for arm in sorted(r3.arm.unique()):
        q = mp[mp.arm == arm].groupby('Subject')[['r_main_zs', 'r_main']].mean()
        say(f'    [{arm}]')
        for s in q.index:
            say(f'      {s}  {q.r_main_zs[s]:.3f} → {q.r_main[s]:.3f}  '
                f'({q.r_main[s]-q.r_main_zs[s]:+.3f})')
        dd = q.r_main - q.r_main_zs
        say(f'      mean gain {dd.mean():+.3f}  sd {dd.std(ddof=1):.3f}  '
            f'improved {int((dd>0).sum())}/{len(dd)} subjects')

head('Basis and methodology reminders', '-')
say('  1. Sections 4 and 5 use the full-window basis; section 6 uses the last-20%-of-each-segment basis. Absolute values must not be subtracted.')
say('  2. Arms from different chunks (different runs) must not be subtracted (cross-run dispersion ≈0.025).')
say('  3. |diff| < 0.02 counts as “indistinguishable”; **do not write “identical”** — that requires TOST.')
say('  4. Public material always uses S01–S10; real names or nicknames must never appear.')

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=os.path.join(RESULTS_DIR, 'mediapipe_all_summary.txt'))
a = ap.parse_args()
with open(a.out, 'w', encoding='utf-8') as f:
    f.write(OUT.getvalue())
print(f'\n[done] {a.out}')
