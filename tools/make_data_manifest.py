# -*- coding: utf-8 -*-
"""Build a content fingerprint for Dataset/, so a git tag also freezes the "data state".

Dataset/ is 967MB and contains raw subject signals, so it is not under version control (see .gitignore).
Thus `git tag` can only freeze code, models and result CSVs — if the data is overwritten or regenerated,
the tag can still be checked out, but the numbers it produces won't match, and **nobody will notice**.

This program writes the SHA-256 of every file into a list that is committed alongside. To verify later:

    python tools/make_data_manifest.py --check reproducibility/DATA_MANIFEST_ic3mt-2026.csv

Usage:
    python tools/make_data_manifest.py [--out path] [--check path]
"""
import argparse
import csv
import hashlib
import io
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

# only include the directories actually consumed by training/evaluation; *_before_* are historical backups and need not be frozen
DIRS = ['Raw', 'Clean', 'Open', 'Combined', 'Skeleton', 'Feature']
ROOT = paths.DATASET_DIR


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def nick_map():
    """nickname → S01–S10.

    The 30 file names in `Dataset/Raw/` and `Dataset/Clean/` are still subject nicknames (see ETHICS.md);
    writing them straight into the list would commit personal data. Instead the mapping is taken from `pipeline/subjects.csv`
    (gitignored), and the list written out only ever contains S01–S10.

    Without subjects.csv it fails outright — better to produce no list than a list with real names.
    """
    src = os.path.join(paths.REPO_ROOT, 'pipeline', 'subjects.csv')
    if not os.path.exists(src):
        sys.exit('%s not found — without the mapping table de-identification is impossible; refusing to write the list.' % src)
    m = {}
    for r in csv.DictReader(io.open(src, encoding='utf-8-sig')):
        key = r['subject_key']
        # the same person is written differently in the three kinds of file names (Raw includes the recording date, Clean/TRC do not),
        # all three must be collected, otherwise only some get replaced.
        for tok in (r['raw_csv'].split('_Raw_DATA')[0],
                    r['emg_env_csv'].split('_CLEAN_SYNCED')[0],
                    r['trc_dir'].split('_MarkerData')[0]):
            if tok:
                m[tok] = key
    # replace longer ones first, in case one nickname is a prefix of another (the dated version must come before the undated one)
    return [(re.compile(re.escape(k), re.I), v)
            for k, v in sorted(m.items(), key=lambda kv: -len(kv[0]))]


def deid(rel, mapping):
    # inconsistent case (Clean uses lowercase, TRC uppercase), so matching is always case-insensitive
    for pat, key in mapping:
        rel = pat.sub(key, rel)
    return rel


def walk():
    """Return [(de-identified relative path, size, sha256)], paths using / and sorted, so it is comparable across platforms."""
    mapping = nick_map()
    rows = []
    for d in DIRS:
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            print('  ! %s not found; skipping' % base)
            continue
        for cur, _, files in os.walk(base):
            for fn in sorted(files):
                full = os.path.join(cur, fn)
                rel = os.path.relpath(full, ROOT).replace('\\', '/')
                rows.append((deid(rel, mapping), os.path.getsize(full),
                             sha256(full)))
    rows.sort()
    return rows


def write(rows, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['path', 'bytes', 'sha256'])
        w.writerows(rows)
    total = sum(r[1] for r in rows)
    print('wrote %s' % out)
    print('  %d files, %.1f MB' % (len(rows), total / 1048576.0))


def check(path):
    want = {r['path']: (int(r['bytes']), r['sha256'])
            for r in csv.DictReader(io.open(path, encoding='utf-8'))}
    have = {r[0]: (r[1], r[2]) for r in walk()}
    missing = sorted(set(want) - set(have))
    extra = sorted(set(have) - set(want))
    changed = sorted(k for k in set(want) & set(have) if want[k][1] != have[k][1])
    for label, items in (('missing', missing), ('content changed', changed), ('extra', extra)):
        if items:
            print('%s %d:' % (label, len(items)))
            for k in items[:20]:
                print('   ', k)
            if len(items) > 20:
                print('    ... %d more' % (len(items) - 20))
    if missing or changed:
        print('\n✗ Data state does not match the list — the numbers of this tag are not guaranteed to be reproducible.')
        return 1
    print('\n✓ All %d files match%s.'
          % (len(want), (' (plus %d new files, no effect)' % len(extra)) if extra else ''))
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'reproducibility', 'DATA_MANIFEST_ic3mt-2026.csv'))
    ap.add_argument('--check')
    a = ap.parse_args()
    if a.check:
        sys.exit(check(a.check))
    print('scanning %s' % ROOT)
    write(walk(), a.out)
