# -*- coding: utf-8 -*-
"""
check_no_pii.py —— scan for leaked subject personal data before committing / publishing.

⚠️ This program itself **contains no real names** (otherwise it would defeat its own purpose).
   The list of real names lives in `tools/pii_names.txt`, which is gitignored.
   Format: one string to look for per line; lines starting with `#` are comments.
   See `tools/pii_names.example.txt` for an example.

Besides the custom list, several patterns that must always be blocked are built in:
  * absolute paths of the development machine (leak the user account and directory structure)
  * subject personal-data files (subjects.csv, the subject mapping table) accidentally added to version control
  * suspected national ID numbers / mobile phone numbers

Usage:
    python tools/check_no_pii.py            # scan git-tracked files (closest to what will actually be pushed)
    python tools/check_no_pii.py --all      # scan all files in the working tree (including untracked)

Exit code 0 = clean, 1 = findings. Can be installed as a pre-commit hook.
"""

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
NAMES_FILE = os.path.join(HERE, 'pii_names.txt')

TEXT_EXT = {'.py', '.md', '.txt', '.csv', '.json', '.ino', '.yml', '.yaml',
            '.cfg', '.toml', '.ipynb', '.h', '.cpp', '.c'}

# patterns that must always be blocked (contain no real names, safe to version-control)
BUILTIN = [
    (re.compile(r'[A-Za-z]:\\Users\\[^\\\s"\']+'), 'development-machine absolute path (includes user account)'),
    (re.compile(r'[A-Za-z]:\\(?:Claude|project)\\'), 'development-machine absolute path'),
    (re.compile(r'\b[A-Z]\d{9}\b'), 'suspected national ID number'),
    (re.compile(r'\b09\d{2}-?\d{3}-?\d{3}\b'), 'suspected mobile phone number'),
]

# if any of these file names enters version control, it is a leak
FORBIDDEN_NAMES = [
    (re.compile(r'(^|/)subjects\.csv$'), 'subjects.csv contains subjects\' age/height/weight/1RM'),
    (re.compile(r'對照表'), 'subject mapping table contains real names'),
]

SKIP_DIRS = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.vscode', '.claude'}


def _name_pattern(name):
    """
    English names need letter boundaries, otherwise short names produce many false hits on ordinary variable names
    (e.g. a three- or four-letter nickname may be part of `start` or `index`).
    Use (?<![A-Za-z]) / (?![A-Za-z]) rather than \\b — because \\b would miss `<name>_0625`
    (underscore counts as a word character in regex, so \\b does not hold between a letter and an underscore).
    Chinese characters need no boundaries and are matched directly.
    """
    esc = re.escape(name)
    if re.fullmatch(r'[A-Za-z0-9_\-]+', name):
        return re.compile(rf'(?<![A-Za-z]){esc}(?![A-Za-z])', re.I)
    return re.compile(esc, re.I)


def load_custom_names():
    if not os.path.isfile(NAMES_FILE):
        return [], False
    names = []
    with open(NAMES_FILE, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                names.append(line)
    return names, True


def tracked_files():
    out = subprocess.run(['git', 'ls-files'], cwd=REPO_ROOT,
                         capture_output=True, text=True, encoding='utf-8')
    if out.returncode != 0:
        print('⚠️  git ls-files failed; scanning all files in the working tree instead', file=sys.stderr)
        return None
    return [p for p in out.stdout.splitlines() if p.strip()]


def walk_files():
    rels = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            rels.append(os.path.relpath(os.path.join(dirpath, fn), REPO_ROOT)
                        .replace(os.sep, '/'))
    return rels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', action='store_true',
                    help='scan all files in the working tree (default: only git-tracked files)')
    args = ap.parse_args()

    files = walk_files() if args.all else (tracked_files() or walk_files())
    custom, has_names_file = load_custom_names()

    print(f"scanning {len(files)} files"
          f" ({'whole working tree' if args.all else 'git-tracked'})")
    if has_names_file:
        print(f"  custom list: {len(custom)} entries (from tools/pii_names.txt)")
    else:
        print("  ⚠️  tools/pii_names.txt not found — only the built-in patterns will be applied.")
        print("     Create it following tools/pii_names.example.txt, filling in subjects' real names / nicknames / raw file names.")

    hits = []

    # file-name level
    for rel in files:
        for pat, why in FORBIDDEN_NAMES:
            if pat.search(rel):
                hits.append((rel, 0, why, rel))

    # content level
    custom_pats = [(_name_pattern(n), f'custom list: {n}') for n in custom]
    # the list file itself naturally contains real names; skip it (it is gitignored and never enters version control)
    self_rel = os.path.relpath(NAMES_FILE, REPO_ROOT).replace(os.sep, '/')

    for rel in files:
        if rel == self_rel or os.path.splitext(rel)[1].lower() not in TEXT_EXT:
            continue
        full = os.path.join(REPO_ROOT, rel)
        if not os.path.isfile(full):
            continue
        try:
            with open(full, encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f, 1):
                    if len(line) > 4000:      # long lines such as base64 are not scanned
                        continue
                    for pat, why in BUILTIN + custom_pats:
                        m = pat.search(line)
                        if m:
                            hits.append((rel, i, why, line.strip()[:120]))
                            break
        except OSError:
            continue

    print()
    if not hits:
        print('✅ No subject personal data found')
        return 0

    print(f'❌ Found {len(hits)} suspicious items:\n')
    for rel, ln, why, snippet in hits[:80]:
        loc = f'{rel}:{ln}' if ln else rel
        print(f'  [{why}]')
        print(f'    {loc}')
        print(f'      {snippet}')
    if len(hits) > 80:
        print(f'  … and {len(hits) - 80} more')
    return 1


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(main())
