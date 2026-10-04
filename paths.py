"""
Centralized path management — every script takes its paths from here; never hard-code absolute paths again.

Why this file exists:
    Before 2026-08, 25 scripts each hard-coded absolute paths of the development machine
    (`D:\\Claude\\sEMG deep learning\\...`), so not a single one ran for anyone who cloned the repo.
    Everything is now derived from "this file's location", so moving to another machine requires no code changes.

The Dataset is not under version control (about 762MB, and it contains raw subject signals).
By default it is looked up in the **parent directory** of the repo as `Dataset/`, i.e.:

    <somewhere>/<project-root>/
    ├── Code/          ← this repo (REPO_ROOT)
    └── Dataset/       ← default dataset location

If your dataset lives elsewhere, just set an environment variable; no code changes needed:

    Windows PowerShell:  $env:SEMG_DATASET = "E:\\my\\Dataset"
    Linux / macOS:       export SEMG_DATASET=/mnt/data/Dataset

Usage:

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from paths import LOSO_MODEL_DIR, COMBINED_DIR, add_pipeline_to_syspath
    add_pipeline_to_syspath()          # after this, `import eval_utils` works
"""

import os

# ─────────────────────────────────────────────────────────────
# inside the repo (all under version control, cloned along with it)
# ─────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

PIPELINE_DIR = os.path.join(REPO_ROOT, "pipeline")       # core pipeline (preprocessing → features → training → fine-tuning)
EXPERIMENTS_DIR = os.path.join(REPO_ROOT, "experiments")  # paper experiments and diagnostics
RESULTS_DIR = os.path.join(REPO_ROOT, "results")          # experiment output CSVs / figures
FIRMWARE_DIR = os.path.join(REPO_ROOT, "firmware")        # Teensy 4.1 firmware
TOOLS_DIR = os.path.join(REPO_ROOT, "tools")              # smoke_test / check_no_pii

# ─────────────────────────────────────────────────────────────
# models: grouped by "training method", see models/README.md
# ─────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(REPO_ROOT, "models")

# global training: all subjects trained together (for deployment; but it inflates evaluation, see docs/METHODOLOGY.md)
GLOBAL_MODEL_DIR = os.path.join(MODELS_DIR, "global")

# LOSO zero-shot: each fold fully excludes one subject — the only honest cross-subject baseline
LOSO_MODEL_DIR = os.path.join(MODELS_DIR, "loso_zeroshot")

# personalized fine-tuning
PERSONALIZED_FROM_LOSO_DIR = os.path.join(MODELS_DIR, "personalized", "from_loso")
PERSONALIZED_NONTWO_DIR = os.path.join(MODELS_DIR, "personalized", "from_loso_nontwo")
PERSONALIZED_FROM_GLOBAL_DIR = os.path.join(MODELS_DIR, "personalized", "from_global")

# ablations / controls
ABLATION_DIR = os.path.join(MODELS_DIR, "ablation")
ABL_BEFORE_AXISFIX_DIR = os.path.join(ABLATION_DIR, "before_axisfix")
ABL_MASKED_DIR = os.path.join(ABLATION_DIR, "masked_synergist")
ABL_EXCLUDE_S09_DIR = os.path.join(ABLATION_DIR, "exclude_s09")
ABL_HOLDOUT_TWO_DIR = os.path.join(ABLATION_DIR, "holdout_two")

LEGACY_MODEL_DIR = os.path.join(MODELS_DIR, "legacy_v1")

# ─────────────────────────────────────────────────────────────
# Dataset (not under version control)
# ─────────────────────────────────────────────────────────────
DATASET_DIR = os.environ.get(
    "SEMG_DATASET",
    os.path.join(os.path.dirname(REPO_ROOT), "Dataset"),
)

RAW_DIR = os.path.join(DATASET_DIR, "Raw")                    # raw EMG
CLEAN_DIR = os.path.join(DATASET_DIR, "Clean")
CLEAN_60FPS_DIR = os.path.join(CLEAN_DIR, "60FPS")            # envelope, for AI training
CLEAN_1000HZ_DIR = os.path.join(CLEAN_DIR, "1000HZ")          # high-passed, for fatigue analysis
COMBINED_DIR = os.path.join(DATASET_DIR, "Combined")          # merged skeleton+EMG features (training input)
COMBINED_BEFORE_AXISFIX_DIR = os.path.join(DATASET_DIR, "Combined_before_axisfix")
OPEN_DIR = os.path.join(DATASET_DIR, "Open")                  # OpenCap MarkerData (TRC)
SKELETON_DIR = os.path.join(DATASET_DIR, "Skeleton")          # per-joint npz for ST-GCN
FEATURE_DIR = os.path.join(DATASET_DIR, "Feature")

# subject personal-data mapping (age/height/weight/1RM) — gitignored, must be created yourself
SUBJECTS_CSV = os.path.join(PIPELINE_DIR, "subjects.csv")
SUBJECTS_EXAMPLE_CSV = os.path.join(PIPELINE_DIR, "subjects.example.csv")


# ─────────────────────────────────────────────────────────────
# small utilities
# ─────────────────────────────────────────────────────────────
def add_pipeline_to_syspath():
    """Let scripts `import eval_utils` and other shared modules under pipeline/."""
    import sys
    if PIPELINE_DIR not in sys.path:
        sys.path.insert(0, PIPELINE_DIR)
    return PIPELINE_DIR


def ensure_results_dir():
    """Call before writing CSVs / figures to make sure results/ exists."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def require_dataset(subdir=None):
    """
    Check that the dataset is in place; if not, give an actionable error message (instead of a long FileNotFoundError).
    """
    target = DATASET_DIR if subdir is None else os.path.join(DATASET_DIR, subdir)
    if not os.path.isdir(target):
        raise FileNotFoundError(
            f"Dataset not found: {target}\n"
            f"  Dataset/ is not under version control (about 762MB, and it contains raw subject signals).\n"
            f"  Currently assumed location: {DATASET_DIR}\n"
            f"  If the dataset is elsewhere, set the environment variable SEMG_DATASET to point to it, e.g.:\n"
            f'    PowerShell:  $env:SEMG_DATASET = "E:\\my\\Dataset"\n'
            f"    bash:        export SEMG_DATASET=/mnt/data/Dataset"
        )
    return target


def require_subjects_csv():
    """
    subjects.csv contains subject personal data and is gitignored. Explains how to create it when missing.
    """
    if not os.path.isfile(SUBJECTS_CSV):
        raise FileNotFoundError(
            f"{SUBJECTS_CSV} not found\n"
            f"  This file contains subjects' age/height/weight/sex/1RM; it is personal data and is not version-controlled.\n"
            f"  Create it yourself following the column format of {SUBJECTS_EXAMPLE_CSV}."
        )
    return SUBJECTS_CSV


if __name__ == "__main__":
    print(f"REPO_ROOT   = {REPO_ROOT}")
    print(f"DATASET_DIR = {DATASET_DIR}  (exists: {os.path.isdir(DATASET_DIR)})")
    print(f"MODELS_DIR  = {MODELS_DIR}")
    print()
    for name in ("GLOBAL_MODEL_DIR", "LOSO_MODEL_DIR", "PERSONALIZED_FROM_LOSO_DIR",
                 "PERSONALIZED_FROM_GLOBAL_DIR", "ABL_BEFORE_AXISFIX_DIR",
                 "ABL_MASKED_DIR", "ABL_EXCLUDE_S09_DIR", "ABL_HOLDOUT_TWO_DIR"):
        d = globals()[name]
        n = len(os.listdir(d)) if os.path.isdir(d) else 0
        print(f"  {name:<30} {n:>3} files  {d}")
