"""
build_skeleton_dataset.py — produce the joint-based dataset for ST-GCN
=========================================================================
Why this program is needed:
-------------------------------------------------------------------------
ST-GCN takes a graph tensor of shape **(time T × joints V × channels C)**,
but the current `Dataset/Combined/*.csv` are **flattened 16-dim hand-crafted features**
(only 11 of which are joint coordinates, covering just four joints: Shoulder/Knee/Ankle/Toe).
With a 4-node graph, graph convolution degenerates into a fully connected layer and ST-GCN loses its point.

This program takes the **first 20 human skeleton keypoints** (Neck…RHeel, OpenPose style) directly from the TRC
and outputs one .npz per segment:

    coords (T, 20, 3)   midHip as origin, normalized by ref_height
    vel    (T, 20, 3)   first difference / dt
    acc    (T, 20, 3)   second difference / dt
    emg    (T, 2)       main / synergist %MVC (0~1.5)
    static (4,)         age / height / weight / sex
    meta                subject, seg, trc, load_pct

★ Kept consistent with `Features_insert_FIXED.py` (so results are comparable with existing experiments):
  - TRC resampled to the EMG segment's frame count (same linear interpolation)
  - ref_height = 95th percentile of (LShoulder_Y − LAnkle_Y)
  - EMG uses the same MVC normalization and clip(0, 1.5)
  - same trc_order / segment_ids pairing and duration health check

★ Deliberately different (with reasons):
  - The origin is **midHip (marker 8)** instead of the original LHip (marker 12).
    midHip is the symmetric centre of the full skeleton, so the left and right legs are treated equally.
  - No hand-crafted derived features (angles, ratios); that is what ST-GCN should learn itself.

Output: Dataset/Skeleton/{subject}_Seg_{id}.npz
"""

import os
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import Features_insert_FIXED as F  # noqa: E402

OUT_DIR = os.path.join(F.DATASET_DIR, "Skeleton")

# first 20 keypoints (TRC marker numbers, 1-based); the order is the graph node index 0-19
JOINT_NAMES = ['Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow',
               'LWrist', 'midHip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee',
               'LAnkle', 'LBigToe', 'LSmallToe', 'LHeel', 'RBigToe',
               'RSmallToe', 'RHeel']
NUM_JOINTS = len(JOINT_NAMES)
ROOT = JOINT_NAMES.index('midHip')          # 7
IDX = {n: i for i, n in enumerate(JOINT_NAMES)}

# skeleton connections (trunk + limbs + feet of OpenPose BODY_25), using 0-based node indices
BONES = [
    ('Neck', 'RShoulder'), ('Neck', 'LShoulder'), ('Neck', 'midHip'),
    ('RShoulder', 'RElbow'), ('RElbow', 'RWrist'),
    ('LShoulder', 'LElbow'), ('LElbow', 'LWrist'),
    ('midHip', 'RHip'), ('midHip', 'LHip'),
    ('RHip', 'RKnee'), ('RKnee', 'RAnkle'),
    ('LHip', 'LKnee'), ('LKnee', 'LAnkle'),
    ('LAnkle', 'LBigToe'), ('LAnkle', 'LHeel'), ('LBigToe', 'LSmallToe'),
    ('RAnkle', 'RBigToe'), ('RAnkle', 'RHeel'), ('RBigToe', 'RSmallToe'),
]
EDGES = [(IDX[a], IDX[b]) for a, b in BONES]


def adjacency(normalize=True):
    """Return the (V, V) adjacency matrix (with self-loops). Used by ST-GCN's graph convolution."""
    A = np.eye(NUM_JOINTS, dtype=np.float32)
    for i, j in EDGES:
        A[i, j] = A[j, i] = 1.0
    if normalize:                      # symmetric normalization D^-1/2 A D^-1/2
        d = A.sum(1)
        Dm = np.diag(1.0 / np.sqrt(np.maximum(d, 1e-8)))
        A = Dm @ A @ Dm
    return A.astype(np.float32)


def process_subject(subject_key, raw_csv_path, emg_env_csv_path, trc_dir,
                    trc_order, segment_ids, subject_info, output_dir=OUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n===== Subject: {subject_key} =====")
    mvc_main, mvc_comp = F.compute_subject_mvc(raw_csv_path)
    print(f"   MVC: main={mvc_main:.2f}, synergist={mvc_comp:.2f}")

    try:
        df_emg_all = pd.read_csv(emg_env_csv_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df_emg_all = pd.read_csv(emg_env_csv_path, encoding='big5')
    seg_frames = df_emg_all.groupby('Segment')['Frame'].max().to_dict()
    seg_dur = (df_emg_all.groupby('Segment')['Time_ms'].max() / 1000.0).to_dict()

    for trc_name, sid in zip(trc_order, segment_ids):
        trc_path = os.path.join(trc_dir, trc_name)
        if sid not in seg_frames:
            print(f"⚠️ EMG has no Segment {sid}; skipping {trc_name}")
            continue
        if not os.path.exists(trc_path):
            print(f"⚠️ TRC {trc_path} not found; skipping")
            continue

        # reuse the existing duration-fingerprint health check
        ts, es = F.trc_duration_s(trc_path), seg_dur.get(sid)
        if ts and es and abs(ts - es) > F.DURATION_TOLERANCE_S:
            print(f"🚨 Segment {sid} <-> {trc_name} duration mismatch: "
                  f"EMG={es:.2f}s / TRC={ts:.2f}s (diff {abs(ts-es):.2f}s)")

        target = int(seg_frames[sid])
        df = pd.read_csv(trc_path, sep='\t', skiprows=4)
        df = df.rename(columns={'Unnamed: 0': 'Frame', 'Unnamed: 1': 'Time'})
        df = df.dropna(subset=['Frame'])
        orig = np.arange(1, len(df) + 1)
        grid = np.linspace(1, target, target)

        # (T, V, 3)
        coords = np.zeros((target, NUM_JOINTS, 3), dtype=np.float32)
        for v in range(NUM_JOINTS):
            for c, ax in enumerate('XYZ'):
                col = f'{ax}{v + 1}'
                y = pd.to_numeric(df[col], errors='coerce').ffill().bfill().fillna(0).values
                coords[:, v, c] = np.interp(grid, orig, y)

        # ref_height: 95th percentile of LShoulder_Y − LAnkle_Y (consistent with the existing pipeline)
        stature = coords[:, IDX['LShoulder'], 1] - coords[:, IDX['LAnkle'], 1]
        ref = float(np.percentile(stature, F.REF_HEIGHT_PCT))
        if ref <= 1e-6:
            ref = float(np.max(stature)) or 1.0

        coords = (coords - coords[:, ROOT:ROOT + 1, :]) / ref   # midHip as origin, then normalize
        vel = np.zeros_like(coords)
        vel[1:] = np.diff(coords, axis=0) / F.dt
        acc = np.zeros_like(vel)
        acc[1:] = np.diff(vel, axis=0) / F.dt

        seg_emg = df_emg_all[df_emg_all['Segment'] == sid].reset_index(drop=True)
        n = min(target, len(seg_emg))
        emg = np.stack([
            np.clip(seg_emg['Raw0'].values[:n] / mvc_main, 0, 1.5),
            np.clip(seg_emg['Raw1'].values[:n] / mvc_comp, 0, 1.5)], axis=1).astype(np.float32)

        static = np.array([subject_info['Age'] / 100.0,
                           subject_info['Height_cm'] / 100.0,
                           subject_info['Weight_kg'] / 100.0,
                           subject_info['Gender']], dtype=np.float32)
        load_pct = F.derive_load_kg(trc_name) / subject_info['Squat_1RM_kg']

        out = os.path.join(output_dir, f"{subject_key}_Seg_{sid}.npz")
        np.savez_compressed(out,
                            coords=coords[:n], vel=vel[:n], acc=acc[:n], emg=emg,
                            static=static, subject=subject_key, seg=sid,
                            trc=trc_name, load_pct=load_pct, ref_height=ref)
        print(f"✅ Seg {sid} ({trc_name})  T={n}  ref_height={ref:.2f}  -> {os.path.basename(out)}")


def main():
    if not F.SUBJECT_TABLE:
        print("❌ No subject data read; please create subjects.csv first")
        sys.exit(1)
    for subj in F.SUBJECTS:
        e = F.SUBJECT_TABLE.get(subj['key'])
        if e is None:
            print(f"\n⚠️ subjects.csv has no {subj['key']}; skipping")
            continue
        process_subject(subj['key'], e['raw_csv'], e['emg_env_csv'], e['trc_dir'],
                        subj['trc_order'], subj['segment_ids'], e['info'])
    A = adjacency()
    np.save(os.path.join(OUT_DIR, '_adjacency.npy'), A)
    print(f"\n🦴 Skeleton graph: {NUM_JOINTS} nodes / {len(EDGES)} edges, adjacency matrix saved")
    print(f"🚀 Done -> {OUT_DIR}")


if __name__ == '__main__':
    main()
