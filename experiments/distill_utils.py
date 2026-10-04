"""
distill_utils.py — data preparation, projection layers and student network for distillation
=========================================================================
Corresponds to step 2, "Feature Alignment", and step 4, experiment C, "knowledge distillation vs
vision-only training", of the NSTC project proposal.

Three losses (the "Ours" arm of the proposal's experiment C):

  L = 3.0·L_task  +  β·L_hint  +  γ·L_soft  +  δ·L_recon

  L_task   MSE of the window-end %MVC          — identical to the baseline, ensuring comparability
  L_hint   projected student features ↔ teacher latent representation (★ main mechanism)
  L_soft   student output ↔ teacher soft labels (teacher predictions in "masked EMG" mode)
  L_recon  student reconstructs the whole window's EMG waveform (the proposal's Reconstruction Loss)

★ Why hint is the main mechanism:
  A 40-frame window yields only 2 numbers of supervision — very sparse.
  The teacher latent is 64-dimensional and encodes **the whole window's EMG trajectory** (the teacher
  sees the real EMG), effectively replacing sparse labels with a dense target. This is where privileged
  distillation actually gets its effect, not "the teacher is more accurate" — `loso_arch_compare.py`
  already showed that a bigger model is not more accurate.

⚠️ **Fairness statement**: the projection layer and reconstruction head are **training-time scaffolding**,
   not used at inference. Both arms have **identical backbones and deployed parameter counts**; only the
   training loss differs. The driver reports both deployed_params and training_params.
"""
# --- Paths: always managed centrally by paths.py at the repo root; do not revert to absolute paths ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import SKELETON_DIR  # noqa: E402,F401
# --- End of path setup ---

import glob
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from models_stgcn import build_stgcn, NUM_JOINTS  # noqa: E402

WINDOW_SIZE = 40
STEP_SIZE = 5


# ============================================================ data
def load_skeleton_segments(channels=9):
    """Return {subject: [(X(T,V,C), emg(T,2), static(4)), ...]}"""
    out = {}
    for f in sorted(glob.glob(os.path.join(SKELETON_DIR, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        parts = [z['coords']]
        if channels >= 6:
            parts.append(z['vel'])
        if channels >= 9:
            parts.append(z['acc'])
        X = np.concatenate(parts, axis=2).astype(np.float32)
        out.setdefault(str(z['subject']), []).append(
            (X, z['emg'].astype(np.float32), z['static'].astype(np.float32)))
    return out


def _windows(segments, sx, ss):
    """
    Window each segment separately; windows never cross a segment boundary (same semantics as eval_utils.make_windows).

    Returns
      Xg    (N, T, V, C)  graph structure, for the student ST-GCN
      Xf    (N, T, V*C)   flattened sequence, for the teacher TCN-Transformer (two views of the same data)
      St    (N, n_static)
      Yend  (N, 2)        window-end %MVC — main task label
      Ywin  (N, T, 2)     whole-window EMG waveform — for the reconstruction loss
    """
    Xg, St, Yend, Ywin = [], [], [], []
    for X, emg, static in segments:
        T = min(len(X), len(emg))
        if T < WINDOW_SIZE:
            continue
        V, C = X.shape[1], X.shape[2]
        flat = sx.transform(X[:T].reshape(T, V * C)).reshape(T, V, C)
        st = ss.transform(static.reshape(1, -1))[0]
        for s in range(0, T - WINDOW_SIZE + 1, STEP_SIZE):
            Xg.append(flat[s:s + WINDOW_SIZE])
            St.append(st)
            Yend.append(emg[s + WINDOW_SIZE - 1])
            Ywin.append(emg[s:s + WINDOW_SIZE])
    Xg = np.asarray(Xg, np.float32)
    N = len(Xg)
    Xf = Xg.reshape(N, WINDOW_SIZE, -1) if N else np.zeros((0, WINDOW_SIZE, 1), np.float32)
    return (Xg, Xf, np.asarray(St, np.float32),
            np.asarray(Yend, np.float32), np.asarray(Ywin, np.float32))


def build_fold(data, subjects, hold, channels=9):
    """
    Data for a single LOSO fold. The scaler is fit only on the training subjects (prevents leakage).

    ★ Every arm shares this function, so what is compared is the loss function, not different windows or normalization.
    """
    tr_segs = [s for k in subjects if k != hold for s in data[k]]
    va_segs = data[hold]
    V = tr_segs[0][0].shape[1]

    big = np.concatenate([X.reshape(len(X), V * channels) for X, _, _ in tr_segs])
    sx = StandardScaler().fit(big)
    ss = StandardScaler().fit(np.stack([st for _, _, st in tr_segs]))

    tr = _windows(tr_segs, sx, ss)
    va = _windows(va_segs, sx, ss)
    keys = ('Xg', 'Xf', 'St', 'Yend', 'Ywin')
    return dict(zip(keys, tr)), dict(zip(keys, va))


# ============================================================ student
def build_student(window_size, n_channels, n_static, filters=(16, 32, 32),
                  latent_dim=64, with_distill_heads=False,
                  fusion_dense=64, emg_dense=32, lr=1e-3, heads='all'):
    """
    Student = ST-GCN backbone (default filters identical to stgcn_posva in `loso_stgcn_compare.py`).

    with_distill_heads=False → plain build_stgcn, outputs out_emg
    with_distill_heads=True  → additionally attaches **training-only** heads:
        out_soft    alias of out_emg, used for the loss against teacher soft labels
                    (Keras binds one loss per output, hence the alias). **Zero extra parameters**
        hint_proj   emg_dense-dim student features → latent_dim teacher space (★ feature-alignment layer)
        emg_recon   reconstructs the whole window's EMG waveform (T, 2)

    heads : 'all'  → out_emg + out_soft + hint_proj + emg_recon (original behaviour, default)
            'soft' → only out_emg + out_soft ★ for response-based KD
              Pure soft-label distillation needs neither hint nor recon. Attaching them (even with loss
              weight 0) still builds the projection/reconstruction layers, wasting training-time parameters
              and compute — especially noticeable for small students: with emg_dense=16, recon_dense alone
              adds 1,360 parameters and distorts the training_params column.

    ⚠️ These heads are not used at inference; **the deployed parameter count equals with_distill_heads=False**.
    """
    if heads not in ('all', 'soft'):
        raise ValueError(f"heads must be 'all' or 'soft', got {heads!r}")

    base = build_stgcn(window_size, n_channels, n_static, filters=filters,
                       lr=lr, fusion_dense=fusion_dense, emg_dense=emg_dense)
    if not with_distill_heads:
        return base

    h = base.get_layer('emg_dense').output          # (B, emg_dense) penultimate layer
    out_emg = base.output
    out_soft = layers.Activation('linear', name='out_soft')(out_emg)

    if heads == 'soft':
        return tf.keras.Model(base.inputs, [out_emg, out_soft],
                              name='student_kd_soft')

    hint = layers.Dense(latent_dim, activation='relu', name='hint_proj')(h)
    rec = layers.Dense(window_size * 2, activation='sigmoid', name='recon_dense')(h)
    rec = layers.Reshape((window_size, 2), name='emg_recon')(rec)

    model = tf.keras.Model(base.inputs,
                           [out_emg, out_soft, hint, rec], name='student_kd')
    return model


def compile_student(model, beta=1.0, gamma=0.5, delta=0.5, lr=1e-3):
    """
    Attach the distillation losses. The main task with weight 3.0 is identical to the baseline, ensuring comparability.

    Only attach outputs the model actually has — `build_student(heads='soft')` has no
    hint_proj / emg_recon, and forcing them would make Keras raise an error.
    """
    names = set(getattr(model, 'output_names', None)
                or [o.name.split('/')[0] for o in model.outputs])
    loss, weights = {'out_emg': 'mse'}, {'out_emg': 3.0}
    for name, w in (('out_soft', gamma), ('hint_proj', beta), ('emg_recon', delta)):
        if name in names:
            loss[name] = 'mse'
            weights[name] = w
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss=loss, loss_weights=weights,
                  metrics={'out_emg': 'mae'})
    return model


def deployed_param_count(window_size, n_channels, n_static, filters,
                         fusion_dense=64, emg_dense=32):
    """Parameter count actually needed at deployment (excluding training-time scaffolding)."""
    m = build_stgcn(window_size, n_channels, n_static, filters=filters,
                    fusion_dense=fusion_dense, emg_dense=emg_dense)
    n = m.count_params()
    tf.keras.backend.clear_session()
    return n


def student_dataset(Xg, St, targets, batch_size, seed, shuffle=True):
    """
    tf.data pipeline for student training.

    ★ It must be built on the CPU. Xg alone is ~955MB (34,804×40×20×9);
      handing the numpy arrays directly to model.fit() makes Keras try to turn them into GPU constants,
      and an 8GB card OOMs by fold 3 (actually hit:
      "Failed copying input tensor ... Dst tensor is not initialized").
      Built on the CPU it streams batch by batch to the GPU, so peak memory is a single batch.
    """
    with tf.device('/CPU:0'):
        ds = tf.data.Dataset.from_tensor_slices(
            ({'ts_input': Xg, 'static_input': St}, targets))
        if shuffle:
            ds = ds.shuffle(min(len(Xg), 20000), seed=seed,
                            reshuffle_each_iteration=True)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def student_predict(model, Xg, St, batch=256, chunk=4096):
    """Uniformly extract out_emg, whether or not the student has distillation heads. Chunked to avoid OOM."""
    outs = []
    for i in range(0, len(Xg), chunk):
        sl = slice(i, i + chunk)
        p = model.predict({'ts_input': Xg[sl], 'static_input': St[sl]},
                          batch_size=batch, verbose=0)
        outs.append(p[0] if isinstance(p, (list, tuple)) else p)
    return np.concatenate(outs, axis=0).astype('float32')


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    def _n(m):
        n = m.count_params()
        tf.keras.backend.clear_session()
        return n

    print("=" * 78)
    print("Teacher spec (16,32,32) + fusion head 64→32")
    print("=" * 78)
    plain = build_student(40, 9, 4, with_distill_heads=False)
    kd = build_student(40, 9, 4, with_distill_heads=True)
    n_plain, n_kd = plain.count_params(), kd.count_params()
    print(f"  deployed / no distill heads    params {n_plain:>8,}")
    print(f"  training + all distill heads   params {n_kd:>8,}")
    print(f"    → projection + reconstruction are training-only: {n_kd - n_plain:,} extra params, not loaded at inference")
    print(f"  outputs: {list(kd.output_names)}")
    tf.keras.backend.clear_session()

    print("\n" + "=" * 78)
    print("★ Student spec (8,8,8) — verify that fusion_dense / emg_dense really take effect")
    print("=" * 78)
    n_big = _n(build_student(40, 9, 4, filters=(8, 8, 8)))
    n_small = _n(build_student(40, 9, 4, filters=(8, 8, 8),
                               fusion_dense=32, emg_dense=16))
    print(f"  (8,8,8) fusion head 64→32  params {n_big:>8,}   1/{107554 / n_big:.1f}")
    print(f"  (8,8,8) fusion head 32→16  params {n_small:>8,}   1/{107554 / n_small:.1f}"
          f"   {'✅ ≤ 10,755' if n_small <= 10755 else '❌ target not met'}")
    print(f"    → shrinking the fusion head saves {n_big - n_small:,} params")

    soft = build_student(40, 9, 4, filters=(8, 8, 8), fusion_dense=32,
                         emg_dense=16, with_distill_heads=True, heads='soft')
    print(f"\n  heads='soft' training-time params {soft.count_params():>8,} "
          f"(= deployed params; out_soft is an alias and adds no parameters)")
    print(f"  outputs: {list(soft.output_names)}")
    assert soft.count_params() == n_small, "out_soft should not add parameters"
    tf.keras.backend.clear_session()
