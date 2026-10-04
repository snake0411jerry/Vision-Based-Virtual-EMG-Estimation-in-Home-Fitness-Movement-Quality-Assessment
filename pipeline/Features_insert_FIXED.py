"""
Features_insert_FIXED.py  —  corrected feature/label generation (batch version)
=========================================================================
Fixes relative to the original `Features_insert.py`:

[Normalization-7] ★ ref_height used to be the single-point maximum of (Shoulder_Y - Ankle_Y).max() for the segment;
           if a segment had no full standing frame, the scale was wrong and not comparable across segments or people.
           This version uses the 95th percentile (robust), with an option to use a fixed anatomical scale instead.

[Leak-1]   ★ Load_1RM_Ratio is no longer output as a feature (unknown at deployment, and highly correlated with %MVC, so it leaks the answer).
           The load can still be kept in file names/logs for analysis, but it does not enter the model input.

[Normalization-5/6] MVC_MAIN/COMP now directly call the same "moving-window stable peak" logic in MAX_MVC_FIXED.py,
             computed automatically from each subject's own Raw file, with no need to copy numbers by hand one by one.

[Batching] ★ The original hard-coded a single subject's paths, TRC list and SUBJECT_INFO.
         This version extracts the core logic into process_subject(), so multiple subjects can be run in batch, each with
         their own TRC list / matching Segment numbers / personal data;
         the pairing order of EMG segments and TRC files, and the extra segments/TRCs to exclude,
         have all been confirmed with the user (see the comments in each subject's settings).

The remaining skeleton features, angles and v/a computations are identical to the original code (whether v/a enters the model is decided by the training script's FEATURE_MODE).
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
import re
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from MAX_MVC_FIXED import (apply_full_emg_pipeline, windowed_peak,  # noqa: E402,F401
                           mvc_denominator, MVC_WINDOW_MS)

# ---------------- shared paths ----------------
# DATASET_DIR comes from paths.py (can be overridden with the environment variable SEMG_DATASET)
RAW_DIR = os.path.join(DATASET_DIR, "Raw")
CLEAN_DIR = os.path.join(DATASET_DIR, "Clean", "60FPS")   # 60FPS envelope combined file (for AI training)
OPEN_DIR = os.path.join(DATASET_DIR, "Open")
OUTPUT_DIR = os.path.join(DATASET_DIR, "Combined")

FPS = 60.0
dt = 1.0 / FPS

# ref_height setting: 'robust_pct' uses the 95th percentile; 'anatomical' uses a fixed value (recommended long term)
# Trunk_Lean formula: 'hypot_xz' (default, consistent with the IC3MT results) or 'signed_x' (see [Axis-4])
TRUNK_LEAN_MODE = 'hypot_xz'

REF_HEIGHT_MODE = 'robust_pct'
REF_HEIGHT_PCT = 95
ANATOMICAL_REF_HEIGHT = None   # if the standing shoulder–ankle distance is known (same units as the TRC), set it here to use a fixed scale

MVC_FS = 1000.0

# tolerance (seconds) between EMG segment and TRC durations. Correct pairings are usually ≤0.05s;
# exceeding it triggers a pairing-error warning during merging.
DURATION_TOLERANCE_S = 0.5

# ---------------- subject mapping table (private) ----------------
# ⚠️ This repo is public. Subjects' personal data (age/height/weight/sex/1RM) and
#    raw file names (which contain names) are never version-controlled — only the de-identified S01…S08 appear in the code.
#    Create subjects.csv in this folder (columns as in subjects.example.csv); it is already in .gitignore.
#
#    Column descriptions:
#      subject_key   de-identified ID (S01…S08), matching the keys in SUBJECTS below
#      raw_csv       raw EMG file name under Dataset/Raw/
#      emg_env_csv   60FPS envelope file name under Dataset/Clean/60FPS/
#      trc_dir       MarkerData folder name under Dataset/Open/
#      age / height_cm / weight_kg / gender / squat_1rm_kg   personal data
SUBJECTS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'subjects.csv')


def load_subject_table(path=SUBJECTS_CSV):
    """Read subjects.csv and return {subject_key: {info, raw_csv, emg_env_csv, trc_dir}}."""
    if not os.path.exists(path):
        print(f"⚠️ Subject mapping table not found: {path}\n"
              f"   Create subjects.csv following subjects.example.csv in the same folder.\n"
              f"   (This file contains personal data and raw file names; it is excluded by .gitignore and never uploaded to GitHub)")
        return {}

    df = pd.read_csv(path)
    table = {}
    for _, r in df.iterrows():
        table[str(r['subject_key'])] = {
            'info': {
                'Age': float(r['age']),
                'Height_cm': float(r['height_cm']),
                'Weight_kg': float(r['weight_kg']),
                'Gender': int(r['gender']),
                'Squat_1RM_kg': float(r['squat_1rm_kg']),
            },
            'raw_csv': os.path.join(RAW_DIR, str(r['raw_csv'])),
            'emg_env_csv': os.path.join(CLEAN_DIR, str(r['emg_env_csv'])),
            'trc_dir': os.path.join(OPEN_DIR, str(r['trc_dir'])),
        }
    return table


SUBJECT_TABLE = load_subject_table()


def calculate_angle(v1, v2):
    dot = np.sum(v1 * v2, axis=1)
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    cos = np.clip(dot / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def compute_subject_mvc(raw_csv_path, fs=MVC_FS, window_ms=MVC_WINDOW_MS):
    """Compute MVC_MAIN / MVC_COMP from the subject's own Raw file, with no manual input.

    🔴 Two fixes on 2026-08-05, see the header of MAX_MVC_FIXED.py:

      [Normalization-7] windowed_peak() is no longer used. The numerator (EMG_*_MVC below) is not smoothed,
                 so a smoothed denominator inflates the ratio structurally by 1.5~2× (measured 1.56× for the main muscle).
      [Normalization-8] per-segment max is no longer used either; instead **pool all segments and take the 99.9th percentile** —
                 taking the max lets a 75ms electrode bounce set the whole person's scale (measured on S02).

    ⚠️ This is not a true %MVC: the dataset has no MVC recordings, so the denominator comes from the subject's own
       squat data. The correct description is "normalized to the maximum activation observed for that subject".

    The `window_ms` parameter is kept only for compatibility with old calls and no longer affects the result.
    """
    try:
        df = pd.read_csv(raw_csv_path, encoding='big5', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(raw_csv_path, encoding='cp950', low_memory=False)

    start_indices = df[df['Time_ms'] == 1].index.tolist()
    if not start_indices:
        start_indices = [0]
    start_indices.append(len(df))

    # ★ collect the envelopes of all segments and take the percentile of the pooled data (never the per-segment max — see [Normalization-8])
    envs_main, envs_comp = [], []
    for i in range(len(start_indices) - 1):
        seg = df.iloc[start_indices[i]:start_indices[i + 1]]
        if len(seg) < fs:
            continue
        raw0 = pd.to_numeric(seg['Raw0'], errors='coerce').fillna(0).values
        raw1 = pd.to_numeric(seg['Raw1'], errors='coerce').fillna(0).values
        _, env0 = apply_full_emg_pipeline(raw0, fs)
        _, env1 = apply_full_emg_pipeline(raw1, fs)
        envs_main.append(env0)
        envs_comp.append(env1)

    if not envs_main:
        raise ValueError(f"{raw_csv_path} has no segment with length >= {fs}; cannot compute the denominator")

    return mvc_denominator(envs_main), mvc_denominator(envs_comp)


def trc_duration_s(trc_path):
    """Actual TRC duration = last row − first row of the Time column (from line 7).

    ★ Do not use the header's NumFrames instead: the headers were not updated after these files were cropped
      (e.g. one subject's 0.trc header says 2039 frames but the data has only 1717),
      so converting from the header gives a wrong duration — which is exactly how pairing errors went unnoticed in the past.
    """
    with open(trc_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    times = []
    for ln in lines[6:]:
        cells = ln.rstrip('\n').split('\t')
        if len(cells) < 2 or not cells[1].strip():
            continue
        try:
            times.append(float(cells[1]))
        except ValueError:
            continue
    return (times[-1] - times[0]) if times else None


def derive_load_kg(trc_filename):
    """Infer the load (kg) from the leading number of the TRC file name; letter names (a/b/c...) count as 0kg."""
    base = os.path.splitext(trc_filename)[0]
    m = re.match(r'^(\d+)', base)
    return float(m.group(1)) if m else 0.0


def process_subject(subject_key, raw_csv_path, emg_env_csv_path, trc_dir,
                     trc_order, segment_ids, subject_info, output_dir=OUTPUT_DIR,
                     ref_height_mode=REF_HEIGHT_MODE, ref_height_pct=REF_HEIGHT_PCT,
                     anatomical_ref_height=ANATOMICAL_REF_HEIGHT,
                     trunk_lean_mode=TRUNK_LEAN_MODE):
    """Process a single subject: pair (trc_order[i], segment_ids[i]) one by one
    and write a Combined_Features.csv for each segment."""
    assert len(trc_order) == len(segment_ids), \
        f"{subject_key}: trc_order and segment_ids have different lengths ({len(trc_order)} vs {len(segment_ids)})"

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n===== Subject: {subject_key} =====")
    mvc_main_max, mvc_comp_max = compute_subject_mvc(raw_csv_path)
    print(f"   automatically computed MVC: main={mvc_main_max:.2f}, synergist={mvc_comp_max:.2f}")

    try:
        df_emg_all = pd.read_csv(emg_env_csv_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df_emg_all = pd.read_csv(emg_env_csv_path, encoding='big5')

    segment_target_frames = df_emg_all.groupby('Segment')['Frame'].max().to_dict()
    segment_duration_s = (df_emg_all.groupby('Segment')['Time_ms'].max() / 1000.0).to_dict()

    for trc_name, segment_idx in zip(trc_order, segment_ids):
        trc_path = os.path.join(trc_dir, trc_name)

        if segment_idx not in segment_target_frames:
            print(f"⚠️ EMG has no Segment {segment_idx}; skipping {trc_name}.")
            continue
        target_frames = int(segment_target_frames[segment_idx])

        # ★ pairing health check: correctly paired EMG/TRC durations should differ by less than ±0.05s.
        #   A large gap means trc_order and segment_ids are mismatched (this once caused
        #   three subjects' EMG to be paired with other sets' movements; training gave r≈0 and nobody noticed).
        if os.path.exists(trc_path):
            trc_s = trc_duration_s(trc_path)
            emg_s = segment_duration_s.get(segment_idx)
            if trc_s is not None and emg_s is not None:
                diff = abs(trc_s - emg_s)
                if diff > DURATION_TOLERANCE_S:
                    print(f"🚨 Segment {segment_idx} <-> {trc_name} duration mismatch: "
                          f"EMG={emg_s:.2f}s / TRC={trc_s:.2f}s (diff {diff:.2f}s)"
                          f" — probable pairing error, check trc_order!")

        try:
            df_gen = pd.read_csv(trc_path, sep='\t', skiprows=4)
        except FileNotFoundError:
            print(f"⚠️ TRC not found: {trc_path}; skipping.")
            continue

        df_gen = df_gen.rename(columns={'Unnamed: 0': 'Frame', 'Unnamed: 1': 'Time'})
        df_gen = df_gen.dropna(subset=['Frame'])
        df_gen['Frame'] = df_gen['Frame'].astype(int)
        df_gen['Frame'] = df_gen['Frame'] - df_gen['Frame'].min() + 1

        original_frames = df_gen['Frame'].values
        grid = np.linspace(1, target_frames, target_frames)
        resampled = {'Frame': np.arange(1, target_frames + 1)}
        for col in df_gen.columns:
            if col not in ['Frame', 'Time']:
                y = pd.to_numeric(df_gen[col], errors='coerce').fillna(0).values
                resampled[col] = np.interp(grid, original_frames, y)
        df_r = pd.DataFrame(resampled)

        columns_to_keep = {
            'X1': 'Neck_X', 'Y1': 'Neck_Y', 'Z1': 'Neck_Z',
            # [Axis-3] adding 'X5': 'Shoulder_X' was tried (16→17 dims); on 2026-08-01 it was
            # verified with a 5-seed LOSO comparison each: main r 0.735→0.729, synergist 0.418→0.409,
            # all four metrics non-significant (p 0.29~0.79) and inconsistent in direction → **not adopted**; kept at 16 dims.
            # details in exp_personalization/loso_shoulderx_summary.csv.
            'Y5': 'Shoulder_Y', 'Z5': 'Shoulder_Z',
            'X8': 'midHip_X', 'Y8': 'midHip_Y', 'Z8': 'midHip_Z',
            'X9': 'RHip_X', 'Y9': 'RHip_Y', 'Z9': 'RHip_Z',
            'X10': 'RKnee_X', 'Y10': 'RKnee_Y', 'Z10': 'RKnee_Z',
            'X11': 'RAnkle_X', 'Y11': 'RAnkle_Y', 'Z11': 'RAnkle_Z',
            'X12': 'Hip_X', 'Y12': 'Hip_Y', 'Z12': 'Hip_Z',
            'X13': 'Knee_X', 'Y13': 'Knee_Y', 'Z13': 'Knee_Z',
            'X14': 'Ankle_X', 'Y14': 'Ankle_Y', 'Z14': 'Ankle_Z',
            'X15': 'BigToe_X', 'Y15': 'BigToe_Y', 'Z15': 'BigToe_Z',
            'X16': 'SmallToe_X', 'Y16': 'SmallToe_Y', 'Z16': 'SmallToe_Z',
            'Y17': 'LHeel_Y', 'Y18': 'RBigToe_Y', 'Y20': 'RHeel_Y',
        }
        avail = {k: v for k, v in columns_to_keep.items() if k in df_r.columns}
        df_clean = df_r[list(avail.keys())].rename(columns=avail)

        df_clean['Toe_X'] = (df_clean['BigToe_X'] + df_clean['SmallToe_X']) / 2.0
        df_clean['Toe_Y'] = (df_clean['BigToe_Y'] + df_clean['SmallToe_Y']) / 2.0
        df_clean['Toe_Z'] = (df_clean['BigToe_Z'] + df_clean['SmallToe_Z']) / 2.0

        # [Axis-2] the thigh–shank angle uses full 3D vectors.
        # The original used only the Z-Y axes; since anterior-posterior is actually X, that computed the knee angle in the frontal plane,
        # and at the bottom of the squat 43~58% of the thigh length lies along X and was projected away,
        # systematically underestimating flexion by 20~30 degrees (worst for the knee-valgus b.trc).
        v_thigh = np.array([df_clean['Hip_X'] - df_clean['Knee_X'],
                            df_clean['Hip_Y'] - df_clean['Knee_Y'],
                            df_clean['Hip_Z'] - df_clean['Knee_Z']]).T
        v_shank = np.array([df_clean['Ankle_X'] - df_clean['Knee_X'],
                            df_clean['Ankle_Y'] - df_clean['Knee_Y'],
                            df_clean['Ankle_Z'] - df_clean['Knee_Z']]).T
        df_clean['Knee_Angle'] = calculate_angle(v_thigh, v_shank)

        v_knee = np.array([df_clean['Knee_X'] - df_clean['Hip_X'],
                           df_clean['Knee_Y'] - df_clean['Hip_Y'],
                           df_clean['Knee_Z'] - df_clean['Hip_Z']]).T
        v_toe = np.array([df_clean['Toe_X'] - df_clean['Ankle_X'],
                          df_clean['Toe_Y'] - df_clean['Ankle_Y'],
                          df_clean['Toe_Z'] - df_clean['Ankle_Z']]).T
        df_clean['Knee_Toe_Angle_Diff'] = calculate_angle(v_knee, v_toe)

        # ---------------------------------------------------------------
        # ★ robust ref_height: not a single-point max
        # ---------------------------------------------------------------
        stature = (df_clean['Shoulder_Y'] - df_clean['Ankle_Y']).values
        if ref_height_mode == 'anatomical' and anatomical_ref_height:
            ref_height = float(anatomical_ref_height)
        else:
            ref_height = float(np.percentile(stature, ref_height_pct))
        if ref_height <= 1e-6:
            ref_height = float(np.max(stature))    # fallback
        print(f"   Segment {segment_idx} ({trc_name}) ref_height={ref_height:.1f} "
              f"(mode={ref_height_mode})")

        # compensation geometry features
        df_clean['L_Heel_Rise'] = df_clean['LHeel_Y'] - df_clean['BigToe_Y']
        df_clean['R_Heel_Rise'] = df_clean['RHeel_Y'] - df_clean['RBigToe_Y']
        df_clean['L_Heel_Rise_norm'] = df_clean['L_Heel_Rise'] / ref_height
        df_clean['R_Heel_Rise_norm'] = df_clean['R_Heel_Rise'] / ref_height
        df_clean['L_Heel_Rise_vel'] = (df_clean['L_Heel_Rise_norm'].diff() / dt).fillna(0)
        df_clean['R_Heel_Rise_vel'] = (df_clean['R_Heel_Rise_norm'].diff() / dt).fillna(0)

        knee_d = np.sqrt((df_clean['Knee_X'] - df_clean['RKnee_X'])**2 + (df_clean['Knee_Z'] - df_clean['RKnee_Z'])**2)
        ankle_d = np.sqrt((df_clean['Ankle_X'] - df_clean['RAnkle_X'])**2 + (df_clean['Ankle_Z'] - df_clean['RAnkle_Z'])**2)
        df_clean['Knee_Ankle_Ratio_norm'] = knee_d / (ankle_d + 1e-5)
        df_clean['Knee_Ankle_Ratio_vel'] = (df_clean['Knee_Ankle_Ratio_norm'].diff() / dt).fillna(0)

        # [Axis-1] tilt angle of the trunk relative to vertical.
        # The original used only the Z component, but Z is actually the left-right axis — it measured side-to-side sway,
        # values were always <9°, so Comp_Trunk_Lean with its 40° threshold never fired.
        # Now uses the horizontal-plane resultant hypot(X, Z), independent of the subject's facing direction,
        # and the numerator magnitude goes from ~0.03m to ~0.35m, making it far more robust to depth noise.
        #
        # [Axis-4] 2026-10-02: hypot fixed the magnitude but introduced a new problem — **it is always positive**,
        # folding backward trunk lean onto the positive side and creating a fake V shape.
        # `diagnose_mediapipe_features.py` measured: computing this feature for the same movement from OpenCap and
        # MediaPipe skeletons, the hypot version agrees at only r=0.354
        # (10th percentile −0.578), while **signed X only** gives r=0.896.
        # Taking the absolute value is even worse (0.221), confirming the problem is sign folding, not Z noise.
        # Controlled by `trunk_lean_mode`; **the default is still hypot_xz to keep old results reproducible**
        #(the IC3MT numbers are tied to the hypot version, see reproducibility/).
        # The cost of signed_x: it is no longer independent of facing direction, so it only applies when “the person roughly faces
        # the camera” — which is exactly this study's single-camera deployment scenario.
        tx = df_clean['Neck_X'] - df_clean['midHip_X']
        tz = df_clean['Neck_Z'] - df_clean['midHip_Z']
        ty = df_clean['Neck_Y'] - df_clean['midHip_Y']
        if trunk_lean_mode == 'hypot_xz':
            df_clean['Trunk_Lean_Angle'] = np.degrees(np.arctan2(np.hypot(tx, tz), ty))
        elif trunk_lean_mode == 'signed_x':
            # [Axis-4] see the comment above: drop the Z term and keep the sign.
            df_clean['Trunk_Lean_Angle'] = np.degrees(np.arctan2(tx, ty))
        else:
            raise ValueError(f"unknown trunk_lean_mode: {trunk_lean_mode}")
        df_clean['Trunk_Lean_Angle_norm'] = df_clean['Trunk_Lean_Angle'] / 180.0
        df_clean['Trunk_Lean_Angle_vel'] = (df_clean['Trunk_Lean_Angle_norm'].diff() / dt).fillna(0)

        for joint in ['Shoulder', 'Knee', 'Ankle', 'Toe']:
            # [Axis-3] no X for Shoulder (experiments showed no benefit, see the columns_to_keep comment above)
            axes = ['Y', 'Z'] if joint == 'Shoulder' else ['X', 'Y', 'Z']
            for axis in axes:
                col = f"{joint}_{axis}"
                root = f"Hip_{axis}"
                if col in df_clean.columns and root in df_clean.columns:
                    c = f"{col}_centered"
                    df_clean[c] = df_clean[col] - df_clean[root]
                    n = f"{col}_norm"
                    df_clean[n] = df_clean[c] / ref_height
                    vcol = f"{col}_vel"
                    df_clean[vcol] = (df_clean[n].diff() / dt).fillna(0)
                    acol = f"{col}_acc"
                    df_clean[acol] = (df_clean[vcol].diff() / dt).fillna(0)

        df_clean['Knee_Angle_norm'] = df_clean['Knee_Angle'] / 180.0
        df_clean['Knee_Angle_vel'] = (df_clean['Knee_Angle_norm'].diff() / dt).fillna(0)
        df_clean['Knee_Toe_Diff_norm'] = df_clean['Knee_Toe_Angle_Diff'] / 180.0
        df_clean['Knee_Toe_Diff_vel'] = (df_clean['Knee_Toe_Diff_norm'].diff() / dt).fillna(0)

        # ---------------------------------------------------------------
        # combine static features + EMG
        # ---------------------------------------------------------------
        df_feat = df_clean[[c for c in df_clean.columns if c.endswith(('_norm', '_vel', '_acc'))]].copy()
        df_feat['Subj_Age'] = subject_info['Age'] / 100.0
        df_feat['Subj_Height_m'] = subject_info['Height_cm'] / 100.0
        df_feat['Subj_Weight_norm'] = subject_info['Weight_kg'] / 100.0
        df_feat['Subj_Gender'] = subject_info['Gender']

        # ★ Load_1RM_Ratio is no longer written as a feature (avoids leakage). The load is only recorded in the log.
        current_load_kg = derive_load_kg(trc_name)
        load_ratio = current_load_kg / subject_info['Squat_1RM_kg']

        df_emg_seg = df_emg_all[df_emg_all['Segment'] == segment_idx].copy().reset_index(drop=True)
        min_len = min(len(df_feat), len(df_emg_seg))
        df_feat = df_feat.iloc[:min_len].copy()

        df_feat['EMG_Main_MVC'] = np.clip(df_emg_seg['Raw0'].values[:min_len] / mvc_main_max, 0, 1.5)
        df_feat['EMG_Compass_MVC'] = np.clip(df_emg_seg['Raw1'].values[:min_len] / mvc_comp_max, 0, 1.5)

        out_name = os.path.join(output_dir, f"{subject_key}_Seg_{segment_idx}_Combined_Features.csv")
        df_feat.to_csv(out_name, index=False)
        print(f"✅ Segment {segment_idx} (load {current_load_kg}kg, intensity {load_ratio*100:.1f}%) done -> {out_name}")


# =============================================================================
# subject settings (keys are de-identified IDs; the actual file paths come from subjects.csv)
#
# ★ basis for determining the TRC pairing order (corrected 2026-07-19):
#   early versions inferred it from “file modification time (mtime) = recording order”; as a result three subjects
#   were entirely mismatched — the mtimes of their `_two` series files fell in the middle, though they were actually recorded last.
#
#   Now a “duration fingerprint” is used: actual TRC duration = last row − first row of the Time column (from line 7),
#   compared with the EMG segment duration. Note the header's NumFrames must not be used; that field was not updated
#   after the files were cropped (e.g. one subject's 0.trc header says 2039 frames, but there are only 1717).
#   Correct pairings all agree within ±0.05 s, enough to serve as a unique identifier.
#
# 2 other subjects have no matching MarkerData folder in Dataset/Open, so they are not included.
# =============================================================================
SUBJECTS = [
    {
        # all durations match (Δ ≤ 0.04s), 7 segments to 7 TRCs.
        "key": "S01",
        "trc_order": ["0.trc", "20.trc", "40.trc", "50.trc", "A.trc", "B.trc", "C.trc"],
        "segment_ids": [1, 2, 3, 4, 5, 6, 7],
    },
    {
        # ★ corrected using the duration fingerprint (all 11 pairs were mismatched under the old mtime order).
        # of the 12 non-neutral TRCs, '0.trc' is only 13.9 s and is a leftover test file; 11 remain after excluding it.
        # after re-trimming c_two.trc, Seg 9 is 30.07s vs 30.03s (Δ 0.03s).
        "key": "S02",
        "trc_order": ["0-1.trc", "20.trc", "30.trc", "40-1.trc", "40-2.trc", "a.trc",
                      "b.trc", "c_2.trc", "c_two.trc", "0_two.trc", "35_two.trc"],
        "segment_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    },
    {
        # the EMG originally has 9 segments; Segment 2 is only 843ms, an undeleted extra segment, and is excluded.
        # 8 real segments remain; all durations match (Δ ≤ 0.03s).
        "key": "S03",
        "trc_order": ["0.trc", "20.trc", "30.trc", "35-1.trc", "35-2.trc", "a.trc", "b.trc", "c_1.trc"],
        "segment_ids": [1, 3, 4, 5, 6, 7, 8, 9],
    },
    {
        # ★ corrected using the duration fingerprint (9 pairs were mismatched under the old mtime order); after correction all 11 pairs have Δ ≤ 0.05s.
        "key": "S04",
        "trc_order": ["0.trc", "20.trc", "50.trc", "70-1.trc", "70-2.trc", "a.trc",
                      "b.trc", "c_2.trc", "0_two.trc", "b_two.trc", "50_two.trc"],
        "segment_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    },
    {
        # all 8 TRCs have exactly the same mtime (exported as a batch), so originally the order could only be guessed by number/letter —
        # the duration fingerprint confirmed the guess was correct (7 pairs Δ ≤ 0.03s).
        # only Segment 2's 20.trc is 0.98s short, probably with the tail cut off during capture, but there is no other candidate file, so the pairing is correct.
        "key": "S05",
        "trc_order": ["0.trc", "20.trc", "30.trc", "50.trc", "a.trc", "a-1.trc", "b.trc", "c_1.trc"],
        "segment_ids": [1, 2, 3, 4, 5, 6, 7, 8],
    },
    {
        # of the 13 original EMG segments, the first two (3996ms / 1288ms) are too short; they are undeleted extra segments and are excluded.
        # ★ corrected using the duration fingerprint (10 pairs were mismatched under the old mtime order); after correction all 11 pairs have Δ ≤ 0.02s.
        "key": "S06",
        "trc_order": ["0.trc", "20.trc", "50.trc", "70-1.trc", "70-2.trc", "a.trc",
                      "b.trc", "c_1.trc", "0_two.trc", "50_two.trc", "b_two.trc"],
        "segment_ids": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    },
    {
        # all durations match (Δ ≤ 0.02s), 8 segments to 8 TRCs.
        "key": "S07",
        "trc_order": ["0.trc", "20.trc", "40.trc", "50-1.trc", "50-2.trc", "A.trc", "B.trc", "C.trc"],
        "segment_ids": [1, 2, 3, 4, 5, 6, 7, 8],
    },
    {
        # 11 non-neutral TRCs for 10 EMG segments; the extra one is '0.trc' (34.15s).
        # ⚠️ Segments 6 and 8 differ in duration by only about 1 frame, so duration cannot tell them apart; assigned by actual recording order
        #    (the basic set a,b,c_1 was recorded first; the _two series was shot afterwards).
        "key": "S08",
        "trc_order": ["0-1.trc", "20.trc", "40.trc", "50.trc", "a.trc",
                      "b.trc", "c_1.trc", "a_two.trc", "0_two.trc", "40_two.trc"],
        "segment_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    },
    {
        # the raw EMG has 8 segments; Segment 6 is only 0.2 s — the EMG device was pressed twice during recording,
        # and the first press was interrupted immediately, leaving an empty segment that belongs to the same take as Segment 7 (b.trc).
        #
        # ★ 2026-07-31: the empty segment has been deleted from the Clean/60FPS file, and later segments were **renumbered**:
        #     old 1,2,3,4,5,(6 deleted),7,8  ->  new 1,2,3,4,5,6,7
        #   so segment_ids changed from [1,2,3,4,5,7,8] to [1,2,3,4,5,6,7].
        #   duration check: b.trc 39.133s <-> new Seg6 39.120s (diff 0.013s),
        #                 c_2.trc 36.167s <-> new Seg7 36.151s (diff 0.016s).
        #   (b.trc additionally had 1.383s trimmed from its start to make up for the time lost between the two presses.)
        "key": "S09",
        "trc_order": ["0.trc", "60.trc", "80-1.trc", "110.trc", "a.trc", "b.trc", "c_2.trc"],
        "segment_ids": [1, 2, 3, 4, 5, 6, 7],
    },
    {
        # of the 9 non-neutral TRCs, '0.trc' is only 4.92 s and is a leftover test file; 8 remain after excluding it.
        # 8 segments to 8 TRCs; all durations match (Δ ≤ 0.01s).
        "key": "S10",
        "trc_order": ["0_1.trc", "40.trc", "80.trc", "120-1.trc", "120-2.trc",
                      "a.trc", "b.trc", "c_2.trc"],
        "segment_ids": [1, 2, 3, 4, 5, 6, 7, 8],
    },
]


if __name__ == '__main__':
    if not SUBJECT_TABLE:
        print("\n❌ No subject data was read; cannot continue. Please create subjects.csv first.")
        sys.exit(1)

    for subj in SUBJECTS:
        entry = SUBJECT_TABLE.get(subj["key"])
        if entry is None:
            print(f"\n⚠️ subjects.csv has no data for {subj['key']}; skipping for now.")
            continue
        process_subject(
            subject_key=subj["key"],
            raw_csv_path=entry["raw_csv"],
            emg_env_csv_path=entry["emg_env_csv"],
            trc_dir=entry["trc_dir"],
            trc_order=subj["trc_order"],
            segment_ids=subj["segment_ids"],
            subject_info=entry["info"],
        )

    print("\n🚀 All subjects with ready data processed (load excluded from features, ref_height made robust, MVC computed automatically)")
