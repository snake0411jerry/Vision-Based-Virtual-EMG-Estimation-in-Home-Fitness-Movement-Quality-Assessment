"""
models_teacher.py — privileged-information teacher network (TCN-Transformer + a randomly masked EMG branch)
=========================================================================
Corresponds to step 2, "Teacher Network - The Expert", of the NSTC project proposal.

The proposal says: teacher input = 3D skeleton sequence + RGB images + **real sEMG**,
while the student only sees the skeleton. This is the classic **Privileged Information (LUPI)** setting —
during training the teacher sees a modality unavailable at deployment.

---------------------------------------------------------------- The design trap

Feeding real EMG into the teacher literally, while the teacher also outputs EMG, makes the teacher degenerate into an
**autoencoder that copies its input**: soft labels ≈ ground truth, distillation loss ≈ task loss, all for nothing.

This file blocks that with two mechanisms:

1. **Random masking (mask_prob, default 0.5)**
   During training each sample has a 50% chance of not seeing EMG (the branch is zeroed and a mask flag is attached).
   The teacher therefore cannot rely on copying alone; it must also learn "how to infer EMG from motion only",
   so its latent representation becomes a **motion-grounded** joint encoding.

2. **Latent compression (latent_dim, default 64)**
   An EMG window is 40×2=80 numbers; compressing to 64 dims while also encoding motion leaves
   the teacher without the bandwidth for point-by-point copying, so it can only keep the structural information of the waveform.

---------------------------------------------------------------- Why it helps

What the student really lacks is not "a bigger model" — `loso_arch_compare.py` already showed TCN-Transformer
and Conv1D score the same; nor "more features" — `exp_c_ablation` already showed 43 dims ≈ 16 dims.

What the student lacks is **density of supervision**: a 40-frame window yields only 2 numbers (the window-end %MVC).
The teacher latent provides a 64-dim dense target, and that representation encodes **the whole window's EMG trajectory**
(which the teacher can see) — information absent from the raw labels. That is the actual mechanism of privileged distillation.

---------------------------------------------------------------- Heterogeneity

The teacher consumes the **flattened skeleton sequence** (T, V*C) through TCN + Transformer (sequence features);
the student consumes the **graph structure** (T, V, C) through ST-GCN (graph features).
Their dimensions and semantic spaces differ, hence the projection layer — see `distill_utils.py`.
"""
# --- Paths: always managed centrally by paths.py at the repo root; do not revert to absolute paths ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import PIPELINE_DIR  # noqa: E402,F401
# --- End of path setup ---

import os
import sys

import tensorflow as tf
from tensorflow.keras import layers

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from models_tcn_transformer import _tcn_block, _transformer_encoder  # noqa: E402

LATENT_NAME = 'teacher_latent'      # the layer distillation taps into
SOFT_NAME = 'out_emg'


def build_teacher(window_size, n_ts_feat, n_static,
                  n_emg=2, latent_dim=64,
                  tcn_filters=64, dilations=(1, 2, 4),
                  num_heads=2, ff_dim=128, dropout=0.1,
                  emg_proj=16, priv_frames=20, lr=1e-3):
    """
    Inputs
    ------
    ts_input     (B, T, n_ts_feat)  flattened skeleton sequence
    emg_input    (B, T, n_emg)      ★ privileged: the real EMG envelope of the window
    emg_mask     (B, 1)             1 = EMG visible, 0 = masked
    static_input (B, n_static)      subject static features

    Outputs
    -------
    out_emg         (B, n_emg)      window-end %MVC (source of soft labels)
    teacher_latent  (B, latent_dim) ★ hint target for distillation

    ⚠️ `emg_input` is privileged information **available only during training**. The student never sees it,
       and it is not needed at deployment — that is precisely the goal of "removing the electrodes".
    """
    inp_ts = layers.Input(shape=(window_size, n_ts_feat), name='ts_input')
    inp_emg = layers.Input(shape=(window_size, n_emg), name='emg_input')
    inp_mask = layers.Input(shape=(1,), name='emg_mask')
    inp_static = layers.Input(shape=(n_static,), name='static_input')

    # ---- motion branch: residual TCN stack → Transformer encoder ----
    x = inp_ts
    for d in dilations:
        x = _tcn_block(x, tcn_filters, d, dropout, name=f'feature_tcn_d{d}')
    x = _transformer_encoder(x, num_heads, tcn_filters // num_heads, ff_dim,
                             dropout, name='feature_attn')
    motion_feat = layers.GlobalAveragePooling1D(name='feature_pool')(x)

    # ---- privileged EMG branch ----
    # ★ Structural leakage prevention: **inside the model**, take only the **first priv_frames frames** of the window.
    #
    #   Why dropping just the window-end frame is not enough (the first version did that, and it was shown to fail):
    #   EMG is highly autocorrelated, so the second-to-last frame is practically the answer. Measured: after 30 epochs the teacher's
    #   r shot up to **0.993** — fully degenerated into copying the answer; the latent held only "the current EMG value",
    #   which is exactly what the student cannot observe in principle, so distillation naturally gave no gain
    #   (measured: hint arm 0.759 vs baseline 0.763, no difference at all).
    #
    #   🔴 Addendum 2026-08-06, to avoid misreading: the 0.993 above was measured **before this Lambda guard existed**.
    #      Under the current implementation, priv_frames=39 (equivalent to dropping only the window-end frame) was measured independently twice
    #      and did not degenerate: probe 0.847 (S01/S03), exp_distill_compare 0.8397 (5 seeds×10 folds).
    #      → **Do not conclude from this comment that 39 cannot be used.** It is usable, and it is the only setting the probe sweep
    #        found with gap ≥0.05. The real conclusion is "the teacher is legitimately stronger, but the knowledge does not transfer".
    #      See the headers of probe_teacher_privilege.py and exp_distill_arch.py.
    #
    #   After switching to only the first half (default 20/40 frames ≈ 0.33 s away from the target),
    #   the nature of the privileged signal changes: it is no longer "the answer" but **this subject's activation level and gain**
    #   — exactly the calibration information that cross-subject transfer lacks most.
    #   The student has to infer it from motion, which is a meaningful learning target.
    #
    #   Cropping inside the model (rather than relying on the caller to pass the right data) is the real guarantee —
    #   this project has already paid the price for "relying on caller discipline".
    hist = layers.Lambda(lambda z: z[:, :priv_frames, :],
                         name='priv_window')(inp_emg)

    e = layers.Conv1D(emg_proj, 3, padding='causal', activation='relu',
                      name='priv_emg_conv')(hist)
    e_gap = layers.GlobalAveragePooling1D(name='priv_emg_gap')(e)
    e_max = layers.GlobalMaxPooling1D(name='priv_emg_max')(e)
    e = layers.Concatenate(name='priv_emg_cat')([e_gap, e_max])

    # masking: when mask=0 the whole branch is zeroed, and the model learns from the mask flag that "there is no EMG this time"
    e = layers.Multiply(name='priv_emg_masked')([e, inp_mask])
    e = layers.Concatenate(name='priv_emg_concat')([e, inp_mask])

    # ---- fusion → latent representation ----
    fused = layers.Concatenate(name='fusion_concat')([motion_feat, e, inp_static])
    fused = layers.Dense(128, activation='relu', name='fusion_dense')(fused)
    fused = layers.Dropout(0.2, name='fusion_dropout')(fused)
    latent = layers.Dense(latent_dim, activation='relu', name=LATENT_NAME)(fused)

    out = layers.Dense(n_emg, activation='sigmoid', name=SOFT_NAME)(latent)

    # single-output model (for training). The latent representation is extracted separately with latent_extractor(),
    # avoiding Keras compatibility issues across versions with "loss=None for one of the outputs".
    model = tf.keras.Model(
        inputs=[inp_ts, inp_emg, inp_mask, inp_static],
        outputs=out, name='teacher')
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss='mse', loss_weights=[3.0], metrics=['mae'])
    return model


def build_homogeneous_teacher(window_size, n_channels, n_static,
                              filters=(32, 64, 64), latent_dim=64,
                              temporal_kernel=5, dropout=0.1, n_emg=2, lr=1e-3):
    """
    **Homogeneous** teacher: an enlarged ST-GCN — the same architecture family as the student, just wider.

    Corresponds to the Baseline of step 4, "Experiment A: verifying the necessity of a heterogeneous teacher", of the NSTC proposal:

        Baseline : distillation with a homogeneous architecture (ST-GCN teacher)  ← this function
        Ours     : Transformer teacher → ST-GCN student                          ← build_teacher(mask_prob=1.0)

    If the heterogeneous teacher does not beat the homogeneous one, we cannot claim "heterogeneous distillation beats plain
    model compression" — that would just be compressing a big model into a small one.

    The input is skeleton only (no privileged EMG); the interface matches build_teacher's vision-only mode.
    """
    from models_stgcn import build_partitions, _st_block, NUM_JOINTS

    A = build_partitions()
    inp = layers.Input(shape=(window_size, NUM_JOINTS, n_channels), name='ts_input')
    inp_static = layers.Input(shape=(n_static,), name='static_input')

    x = layers.BatchNormalization(name='feature_bn_in')(inp)
    for i, f in enumerate(filters):
        x = _st_block(x, A, f, temporal_kernel, dropout, name=f'feature_st{i}')
    x = layers.GlobalAveragePooling2D(name='feature_pool')(x)

    fused = layers.Concatenate(name='fusion_concat')([x, inp_static])
    fused = layers.Dense(128, activation='relu', name='fusion_dense')(fused)
    fused = layers.Dropout(0.2, name='fusion_dropout')(fused)
    latent = layers.Dense(latent_dim, activation='relu', name=LATENT_NAME)(fused)
    out = layers.Dense(n_emg, activation='sigmoid', name=SOFT_NAME)(latent)

    model = tf.keras.Model([inp, inp_static], out, name='teacher_homogeneous')
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss='mse', loss_weights=[3.0], metrics=['mae'])
    return model


def latent_extractor(teacher):
    """Extract the latent representation from a trained teacher (the hint target for distillation). Works for both teachers."""
    return tf.keras.Model(teacher.inputs,
                          teacher.get_layer(LATENT_NAME).output,
                          name='teacher_latent_extractor')


def homo_targets(teacher, extractor, Xg, X_static, batch=256, chunk=4096):
    """Distillation targets for the homogeneous teacher (graph input, no privileged input)."""
    import numpy as np
    softs, lats = [], []
    for i in range(0, len(Xg), chunk):
        sl = slice(i, i + chunk)
        feed = {'ts_input': Xg[sl], 'static_input': X_static[sl]}
        softs.append(teacher.predict(feed, batch_size=batch, verbose=0))
        lats.append(extractor.predict(feed, batch_size=batch, verbose=0))
    return (np.concatenate(softs).astype('float32'),
            np.concatenate(lats).astype('float32'))


def homo_predict(teacher, Xg, X_static, batch=256, chunk=4096):
    import numpy as np
    outs = []
    for i in range(0, len(Xg), chunk):
        sl = slice(i, i + chunk)
        outs.append(teacher.predict(
            {'ts_input': Xg[sl], 'static_input': X_static[sl]},
            batch_size=batch, verbose=0))
    return np.concatenate(outs).astype('float32')


def teacher_dataset(X_ts, X_emg, X_static, Y, mask_prob, batch_size, seed):
    """
    tf.data pipeline for training — the mask is **resampled every epoch**.

    This beats "one fixed mask": the teacher sees both the "with EMG" and "without EMG" versions of the same sample,
    and is forced to learn the motion–EMG relationship into a shared representation instead of two separate shortcuts.
    """
    import numpy as np
    keep = 1.0 - float(mask_prob)

    def _mask(x, y):
        m = tf.cast(tf.random.uniform((1,)) < keep, tf.float32)   # 1 = EMG visible
        return {**x, 'emg_mask': m, 'emg_input': x['emg_input'] * m}, y

    # ★ It must be built on the CPU. A single training array is ~1GB;
    #   from_tensor_slices turns it into a GPU constant by default, and an 8GB card OOMs immediately.
    with tf.device('/CPU:0'):
        ds = tf.data.Dataset.from_tensor_slices(
            ({'ts_input': X_ts, 'emg_input': X_emg, 'static_input': X_static},
             np.asarray(Y, 'float32')))
        ds = (ds.shuffle(min(len(X_ts), 20000), seed=seed,
                         reshuffle_each_iteration=True)
                .map(_mask, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE))
    return ds


def teacher_targets(teacher, extractor, X_ts, X_emg, X_static, batch=256):
    """
    Produce the two distillation targets.

    latent  ← privileged channel **on**. Contains the EMG structure of the whole window; the target of the hint loss (★ main mechanism).
    soft    ← privileged channel **off**. The teacher then reduces to vision-only inference, and its predictions do not coincide
              with ground truth, giving the smoothing a soft label should provide; in the "on" mode soft labels ≈ ground truth
              and L_soft would reduce to L_task with no information at all.

    Returns (soft, latent)
    """
    latent = _chunked_predict(extractor, X_ts, X_emg, X_static, True, batch)
    soft = _chunked_predict(teacher, X_ts, X_emg, X_static, False, batch)
    return soft.astype('float32'), latent.astype('float32')


def _chunked_predict(model, X_ts, X_emg, X_static, emg_visible, batch, chunk=4096):
    """
    Chunked inference. X_ts is easily ~1GB; passing it to predict in one go turns it into a GPU constant and OOMs;
    sending it in small chunks keeps peak memory to the size of one chunk.
    """
    import numpy as np
    outs = []
    for i in range(0, len(X_ts), chunk):
        sl = slice(i, i + chunk)
        n = len(X_ts[sl])
        m = np.full((n, 1), 1.0 if emg_visible else 0.0, dtype='float32')
        emg_in = X_emg[sl] if emg_visible else np.zeros_like(X_emg[sl])
        outs.append(model.predict(
            {'ts_input': X_ts[sl], 'emg_input': emg_in,
             'emg_mask': m, 'static_input': X_static[sl]},
            batch_size=batch, verbose=0))
    return np.concatenate(outs, axis=0)


def teacher_predict(teacher, X_ts, X_emg, X_static, emg_visible, batch=256):
    """Teacher evaluation. emg_visible=True is **diagnostic only** (it can see the privileged information; not a fair comparison)."""
    return _chunked_predict(teacher, X_ts, X_emg, X_static,
                            emg_visible, batch).astype('float32')


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    for n_ts in (16, 180):
        m = build_teacher(40, n_ts, 4)
        tag = 'hand-crafted features' if n_ts == 16 else 'flattened skeleton 20 joints×9 ch'
        print(f"teacher ({tag:<22} n_ts={n_ts:>3})  params {m.count_params():>9,}")
        tf.keras.backend.clear_session()
    print("\nReference: Conv1D 22,306 / TCN-Transformer 107,554 / ST-GCN(posva) 29,734")
