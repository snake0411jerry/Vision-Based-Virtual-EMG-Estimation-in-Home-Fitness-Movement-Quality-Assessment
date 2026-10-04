"""
models_stgcn.py — lightweight ST-GCN student network (the core architecture of the NSTC proposal)
=========================================================================
Corresponds to "step 2: student network" and "expected outcome 1" of the proposal:
  "adopt a lightweight ST-GCN ... with only 1/10 of the parameters of mainstream Transformer models"

Key architecture points (after Yan et al. 2018):
  1. **Spatial graph convolution**: treat the body skeleton as a graph and aggregate neighbouring joints via the adjacency matrix
  2. **Temporal convolution**: a 1D convolution along time for every joint
  3. **Residual connections**: stack many layers without degradation

★ Spatial partitioning strategy
  The original paper proposes three; this file uses the best-performing **distance partitioning**,
  splitting neighbours into three groups by "distance to the root node (midHip)":
      k=0 self, k=1 centripetal (closer to the root), k=2 centrifugal (farther from the root)
  Each group has its own weights, so the model can distinguish "trunk driving the limbs" from "limbs feeding back to the trunk".
  If everything were merged into one group (K=1), graph convolution would degenerate into neighbour averaging and lose directionality.

★ Edge Importance Weighting
  A trick from the original paper: give the adjacency matrix a learnable per-edge weight mask.
  Especially useful on small data — the model can down-weight irrelevant bone segments (e.g. the wrist) by itself.

⚠️ The comparison with the current baseline is not a "pure architecture swap":
   the baseline consumes 16 **hand-crafted features** (angles, ratios, heel height…),
   while ST-GCN consumes the **raw skeleton** of 20 joints × 3 coordinates.
   The inputs are fundamentally different; what is compared is "hand-crafted features + Conv1D" vs "raw skeleton + graph convolution",
   which is exactly what the proposal asks, but the manuscript must say so clearly and not describe it as merely swapping the architecture.
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
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

HERE = os.path.dirname(os.path.abspath(__file__))
FIXED_DIR = PIPELINE_DIR
sys.path.insert(0, FIXED_DIR)
from build_skeleton_dataset import (NUM_JOINTS, EDGES, ROOT)  # noqa: E402


def hop_distance(num_joints, edges, max_hop=1):
    A = np.zeros((num_joints, num_joints))
    for i, j in edges:
        A[i, j] = A[j, i] = 1
    dist = np.full((num_joints, num_joints), np.inf)
    P = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    for d in range(max_hop, -1, -1):
        dist[P[d] > 0] = d
    return dist


def build_partitions():
    """distance partitioning: return a normalized (3, V, V) adjacency tensor.

    k=0 self; k=1 neighbours closer to the root (centripetal); k=2 neighbours farther from the root (centrifugal).
    """
    d = hop_distance(NUM_JOINTS, EDGES, max_hop=1)
    root_d = np.array([d[ROOT, v] if np.isfinite(d[ROOT, v]) else 1e9
                       for v in range(NUM_JOINTS)])
    # fill in the true distance to the root with BFS (max_hop=1 only sees first-order neighbours)
    full = np.full(NUM_JOINTS, -1)
    full[ROOT] = 0
    frontier = [ROOT]
    adj = {v: [] for v in range(NUM_JOINTS)}
    for i, j in EDGES:
        adj[i].append(j)
        adj[j].append(i)
    while frontier:
        nxt = []
        for v in frontier:
            for u in adj[v]:
                if full[u] < 0:
                    full[u] = full[v] + 1
                    nxt.append(u)
        frontier = nxt
    root_d = full.astype(float)

    A = np.zeros((3, NUM_JOINTS, NUM_JOINTS), dtype=np.float32)
    for v in range(NUM_JOINTS):
        A[0, v, v] = 1.0                       # self
        for u in adj[v]:
            if root_d[u] < root_d[v]:
                A[1, v, u] = 1.0               # centripetal
            else:
                A[2, v, u] = 1.0               # centrifugal
    # row-normalize each subset so nodes of different degree are on the same scale
    for k in range(3):
        s = A[k].sum(axis=1, keepdims=True)
        A[k] = A[k] / np.maximum(s, 1e-8)
    return A


class GraphConv(layers.Layer):
    """Spatial graph convolution: out[v] = Σ_k Σ_u A[k,v,u] · (W_k · x[u])"""

    def __init__(self, filters, A, **kw):
        super().__init__(**kw)
        self.filters = filters
        self.A_init = A.astype(np.float32)
        self.K = A.shape[0]

    def build(self, input_shape):
        self.A = self.add_weight(
            name='A', shape=self.A_init.shape,
            initializer=tf.keras.initializers.Constant(self.A_init), trainable=False)
        # Edge importance: a learnable per-edge weight mask (a trick from the original paper, very useful on small data)
        self.edge_imp = self.add_weight(
            name='edge_importance', shape=self.A_init.shape,
            initializer='ones', trainable=True)
        self.proj = layers.Dense(self.filters * self.K, use_bias=False)
        super().build(input_shape)

    def call(self, x):                       # x: (B, T, V, C)
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        h = self.proj(x)                     # (B, T, V, F*K)
        h = tf.reshape(h, [B, T, NUM_JOINTS, self.K, self.filters])
        A = self.A * self.edge_imp           # (K, V, V)
        # out[b,t,v,f] = Σ_k Σ_u A[k,v,u] · h[b,t,u,k,f]
        return tf.einsum('btukf,kvu->btvf', h, A)

    def get_config(self):
        return {**super().get_config(), 'filters': self.filters}


def _st_block(x, A, filters, temporal_kernel, dropout, name):
    """One ST-GCN unit: spatial graph convolution → temporal convolution → residual."""
    res = x
    x = GraphConv(filters, A, name=f'{name}_gcn')(x)
    x = layers.BatchNormalization(name=f'{name}_bn1')(x)
    x = layers.Activation('relu', name=f'{name}_relu1')(x)
    # temporal convolution: a (kt, 1) kernel convolves each joint along time independently
    x = layers.Conv2D(filters, (temporal_kernel, 1), padding='same',
                      name=f'{name}_tcn')(x)
    x = layers.BatchNormalization(name=f'{name}_bn2')(x)
    x = layers.Dropout(dropout, name=f'{name}_drop')(x)
    if res.shape[-1] != filters:
        res = layers.Conv2D(filters, 1, padding='same', name=f'{name}_resproj')(res)
    x = layers.Add(name=f'{name}_add')([res, x])
    return layers.Activation('relu', name=f'{name}_relu2')(x)


def build_stgcn(window_size, n_channels, n_static,
                filters=(16, 32, 32), temporal_kernel=5, dropout=0.1,
                lr=1e-3, n_out=2, fusion_dense=64, emg_dense=32):
    """Lightweight ST-GCN: (T, V, C) skeleton + static features -> n_out %MVC values.

    Layer names keep the 'feature*' prefix so phase2's frozen-feature-layer logic works unchanged.

    ★ `fusion_dense` / `emg_dense` control the widths of the two layers after GAP.
      The default 64/32 is the configuration of every existing result — **do not change the defaults**.
      Shrinking them is necessary to reach "1/10 of the parameters": with filters=(8,8,8)
      GAP leaves only 8 dims, yet Dense(64)→Dense(32)→Dense(2) still consumes about 2,978 parameters,
      a quarter of the whole model. Changing to 32/16 saves 2,000 (12,070 → 10,070).
      See the parameter budget table in exp_distill_compress.py.
    """
    A = build_partitions()
    inp = layers.Input(shape=(window_size, NUM_JOINTS, n_channels), name='ts_input')
    inp_static = layers.Input(shape=(n_static,), name='static_input')

    x = layers.BatchNormalization(name='feature_bn_in')(inp)
    for i, f in enumerate(filters):
        x = _st_block(x, A, f, temporal_kernel, dropout, name=f'feature_st{i}')

    x = layers.GlobalAveragePooling2D(name='feature_pool')(x)   # pool over (T, V) jointly
    fused = layers.Concatenate(name='fusion_concat')([x, inp_static])
    fused = layers.Dense(fusion_dense, activation='relu', name='fusion_dense')(fused)
    fused = layers.Dropout(0.2, name='fusion_dropout')(fused)
    h = layers.Dense(emg_dense, activation='relu', name='emg_dense')(fused)
    out = layers.Dense(n_out, activation='sigmoid', name='out_emg')(h)

    model = tf.keras.Model([inp, inp_static], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss={'out_emg': 'mse'}, loss_weights={'out_emg': 3.0},
                  metrics={'out_emg': 'mae'})
    return model


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    A = build_partitions()
    print(f"graph partition {A.shape}  self edges {int((A[0]>0).sum())} / centripetal {int((A[1]>0).sum())}"
          f" / centrifugal {int((A[2]>0).sum())}")
    TCN_T = 107_554          # the denominator of "1/10 parameters" (settled 2026-08-02)
    TARGET = TCN_T / 10

    print(f"\n{'Config':<30} {'C':>2} {'filters':<12} {'fusion':<9} "
          f"{'total':>8} {'trainable':>8}  vs TCN-T")
    print("-" * 88)
    for name, ch, fl, fd, ed in [
            ('ST-GCN ultra-light (pos)',        3, (16, 16, 32), 64, 32),
            ('ST-GCN standard (pos)',          3, (16, 32, 32), 64, 32),
            ('ST-GCN standard (pos+vel+acc)',  9, (16, 32, 32), 64, 32),   # ← teacher 29,734
            ('ST-GCN larger (pos+vel+acc)',  9, (32, 64, 64), 64, 32),
            ('★ student (8,8,8) original fusion head',    9, (8, 8, 8),    64, 32),
            ('★ student (8,8,8) shrunk fusion head',    9, (8, 8, 8),    32, 16)]:
        m = build_stgcn(40, ch, 4, filters=fl, fusion_dense=fd, emg_dense=ed)
        n = m.count_params()
        ntr = sum(int(tf.size(w)) for w in m.trainable_weights)
        mark = '✅' if n <= TARGET else '  '
        print(f"  {name:<28} {ch:>2} {str(fl):<12} {f'{fd}→{ed}':<9} "
              f"{n:>8,} {ntr:>8,}  1/{TCN_T / n:>4.1f} {mark}")
        tf.keras.backend.clear_session()

    print(f"\nReference: Conv1D baseline 22,306 / TCN-Transformer {TCN_T:,}"
          f"  →  1/10 target ≤ {TARGET:,.0f}")
    print("\n⚠️ Every GraphConv carries a fixed A(3×20×20=1,200, trainable=False) + "
          "edge_importance(1,200); ")
    print("   the three blocks total 7,200 parameters that **do not shrink with filters**. That is a hard floor; "
          "however small the filters, it cannot go below ~7.3k.")
    print("   When the paper reports parameter counts it should disclose this composition and also give the trainable count.")
