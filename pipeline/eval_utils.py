"""
eval_utils.py  —  shared evaluation / anti-leakage utilities
=========================================================
Shared functions used by the corrected training scripts:
  1. subject_id_from_path : parse the subject ID from the file name (for grouping "by subject", so one person is never split across train/val)
  2. build_ts_cols        : generate the time-series feature columns in one place (central exclude list, guarantees the Load leak is excluded, v/a switchable)
  3. pearson_per_column   : Pearson r per output channel (per muscle) — the paper's headline metric
  4. make_windows         : sliding windows over a "list of already-split dfs"; windows always lie within a single df, never across boundaries (prevents temporal leakage)
  5. temporal_split_df    : split a single df into earlier/later parts by time (for fine-tuning, replacing random window splits)

Author's note: the original code excluded all vel/acc and used Load_1RM_Ratio as an input;
          these are fixed here centrally so all scripts share the same column logic.
"""

import os
import numpy as np

# the unified "must exclude" list.
#   - 'EMG_'          : labels (%MVC), must not be inputs
#   - 'Comp_'         : compensation labels, must not be inputs
#   - 'Load_1RM_Ratio': ★ leaks the answer: the load is highly correlated with %MVC, and at deployment the user won't know their 1RM/bar weight
#   - 'Subj_'         : subject static info, handled by the separate static branch, not part of the time-series features
#   - 'Knee_Toe_Diff' : excluded in the original code; kept that way
#   - 'Target','Unnamed': miscellaneous
EXCLUDE_ALWAYS = ['EMG_', 'Comp_', 'Load_1RM_Ratio', 'Subj_',
                  'Knee_Toe_Diff', 'Target', 'Unnamed']

# the "source features" of the compensation labels — if the compensation classification task is kept, these must be removed from the input,
# otherwise the model just learns a threshold it can already see (circular reasoning).
COMP_SOURCE_COLS = ['L_Heel_Rise_norm', 'R_Heel_Rise_norm',
                    'Knee_Ankle_Ratio_norm', 'Trunk_Lean_Angle_norm']


def subject_id_from_path(path):
    """Parse the subject ID 'S01' from 'S01_Seg_1_Combined_Features.csv'."""
    base = os.path.basename(path)
    return base.split('_')[0]


def build_ts_cols(all_columns, feature_mode="spatial", enable_comp_task=False):
    """
    Generate the list of time-series feature columns.

    feature_mode:
        "spatial"           -> positions only (_norm), vel/acc excluded (reproduces the paper's setting and avoids differentiation amplifying tracking noise)
        "spatial+kinematic" -> keep vel/acc (for testing the NSTC proposal's "physics-guided features" hypothesis;
                               recommended to apply Savitzky-Golay smoothing to the coordinates before differentiating)
    enable_comp_task:
        True -> additionally remove the compensation source columns from the input, avoiding circularity with the compensation labels
    """
    exclude = list(EXCLUDE_ALWAYS)
    if feature_mode == "spatial":
        exclude += ['vel', 'acc']
    elif feature_mode == "spatial+kinematic":
        pass  # keep vel/acc
    else:
        raise ValueError(f"unknown feature_mode: {feature_mode}")

    ts_cols = [c for c in all_columns if not any(k in c for k in exclude)]

    if enable_comp_task:
        ts_cols = [c for c in ts_cols if c not in COMP_SOURCE_COLS]

    return ts_cols


def pearson_per_column(y_true, y_pred, eps=1e-8):
    """
    Per-channel Pearson r, NaN-safe.
    Returns an array of shape=(n_outputs,); NaN for channels that are nearly constant.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_out = y_true.shape[1]
    rs = np.full(n_out, np.nan)
    for i in range(n_out):
        yt, yp = y_true[:, i], y_pred[:, i]
        if yt.std() < eps or yp.std() < eps:
            continue
        rs[i] = np.corrcoef(yt, yp)[0, 1]
    return rs


def nrmse_per_column(y_true, y_pred, mode='std', eps=1e-8):
    """
    Per-channel normalized RMSE.

    mode='std'   : RMSE / std(y_true)  ← normalized by that subject's signal standard deviation for the channel
    mode='range' : RMSE / (max-min)

    ★ Why report it together with Pearson r:
      r is immune to linear transforms (scaling the ground truth by ×0.5 or adding +0.1 leaves r unchanged),
      so a high r does not mean accurate values — systematic over/underestimation is invisible in r.
      nRMSE catches this kind of bias.
      Also, r is variance-normalized: when someone's synergist barely activates and the signal is nearly flat,
      r is mechanically depressed even if the absolute error is small. In that case nRMSE reflects the real prediction quality.
      For an ideal unbiased prediction nRMSE(std) ≈ sqrt(1 - r²); a deviation between the two indicates systematic bias.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_out = y_true.shape[1]
    out = np.full(n_out, np.nan)
    for i in range(n_out):
        yt, yp = y_true[:, i], y_pred[:, i]
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        denom = yt.std() if mode == 'std' else (yt.max() - yt.min())
        if denom < eps:
            continue
        out[i] = rmse / denom
    return out


def full_metrics(y_true, y_pred):
    """Compute r / MAE / RMSE / nRMSE(std) in one go; return a dict of arrays (per channel)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        'r': pearson_per_column(y_true, y_pred),
        'mae': np.mean(np.abs(y_true - y_pred), axis=0),
        'rmse': np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0)),
        'nrmse': nrmse_per_column(y_true, y_pred, mode='std'),
    }


def make_windows(df_list, ts_cols, static_cols, label_emg_cols,
                 label_comp_cols=None, window_size=40, step_size=5,
                 scaler_ts=None, scaler_static=None):
    """
    Sliding windows over a list of dfs that are "already split and non-overlapping".
    ★ Key: each df is windowed separately, so windows never cross df boundaries — the precondition for preventing temporal leakage.
       The caller must first split train / val into different df lists by "subject" or "time period";
       never pool all windows together and then split randomly.

    Returns: X_ts, X_static, Y_emg, (Y_comp or None)
    """
    X_ts, X_static, Y_emg, Y_comp = [], [], [], []
    for df in df_list:
        ts_raw = df[ts_cols].values
        st_raw = df[static_cols].values
        ts_data = scaler_ts.transform(ts_raw) if scaler_ts is not None else ts_raw
        st_data = scaler_static.transform(st_raw) if scaler_static is not None else st_raw
        emg_data = df[label_emg_cols].values
        comp_data = df[label_comp_cols].values if label_comp_cols else None

        for j in range(0, len(ts_data) - window_size, step_size):
            X_ts.append(ts_data[j:j + window_size, :])
            tgt = j + window_size - 1          # causal: the last frame of the window is the prediction target
            X_static.append(st_data[tgt, :])
            Y_emg.append(emg_data[tgt, :])
            if comp_data is not None:
                Y_comp.append(comp_data[tgt, :])

    X_ts = np.array(X_ts)
    X_static = np.array(X_static)
    Y_emg = np.array(Y_emg)
    Y_comp = np.array(Y_comp) if label_comp_cols else None
    return X_ts, X_static, Y_emg, Y_comp


# =========================================================
# compensation thresholds (managed centrally)
# =========================================================
# ⚠️ history lesson: these three thresholds used to be hard-coded separately in both Global_Training_FIXED.py and
#    phase2_finetune_FIXED.py. The earliest TrainModel.py comments said explicitly
#    “set to 0.02 as a demo” and “can be adjusted to the actual standard”, but after three generations of copying nobody had adjusted them,
#    and the FIXED versions even deleted the word “demo”, so later readers could not tell they were placeholders.
#    They are now centralized here; change once and both scripts stay in sync.
#
# updated 2026-07-31 from a per-frame ROC on the full dataset (10 subjects, 90 TRCs):
#   heel_raise   AUC 0.616  suggested 0.017 -> keep 0.020 (weak AUC, conservative first)
#   knee_valgus  AUC 0.768  suggested 0.758 -> ★ adopted (the old 0.92 misclassified 56% of normal frames)
#   trunk_lean   AUC 0.621  suggested 29.4  -> keep 40.0 (conservative; only 0.7% of the normal group misclassified)
#
# ⚠️ a per-frame ROC systematically underestimates separability (compensation only occurs during part of each rep).
#    Once rep segmentation is done, rerun it with “peak per rep”, and revisit these values then.
# ⚠️ the paper must not say “according to the XX standard” — these are this dataset's ROC optimal points, not literature values.
COMP_THRESHOLDS = {
    'heel_raise': 0.020,     # flagged when L/R_Heel_Rise_norm exceeds it; larger = more like compensation
    'knee_valgus': 0.758,    # flagged when Knee_Ankle_Ratio_norm is below it; smaller = more like compensation
    'trunk_lean_deg': 40.0,  # flagged when Trunk_Lean_Angle exceeds it (degrees); larger = more like compensation
}


def add_comp_labels(df, thresholds=None):
    """Add the three compensation label columns in place and return df.

    Shared by both training scripts so the thresholds can no longer drift apart.
    Note Trunk_Lean_Angle_norm stores "angle / 180", so it must be divided by 180 for the comparison.
    """
    th = thresholds or COMP_THRESHOLDS
    df['Comp_Heel_Raise'] = ((df['L_Heel_Rise_norm'] > th['heel_raise']) |
                             (df['R_Heel_Rise_norm'] > th['heel_raise'])).astype(float)
    df['Comp_Knee_Valgus'] = (df['Knee_Ankle_Ratio_norm'] < th['knee_valgus']).astype(float)
    df['Comp_Trunk_Lean'] = (df['Trunk_Lean_Angle_norm'] >
                             (th['trunk_lean_deg'] / 180.0)).astype(float)
    return df


def temporal_split_df(df, val_tail_frac=0.2):
    """
    Split a single df by time: the first (1-frac) as train, the last frac as val.
    Used during fine-tuning to replace the leaky practice of "random split over overlapping windows".
    """
    n = len(df)
    cut = int(round(n * (1.0 - val_tail_frac)))
    cut = max(1, min(cut, n - 1))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()
