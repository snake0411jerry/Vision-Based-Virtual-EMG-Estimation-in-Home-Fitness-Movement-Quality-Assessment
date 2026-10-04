# -*- coding: utf-8 -*-
"""
smoke_test.py —— run this first after cloning to confirm the environment and models work.

No Dataset needed; it only uses random inputs to verify that:
  1. paths.py derives all paths correctly
  2. every kind of model file is where it should be
  3. the global model and the LOSO models load and run inference
  4. the scaler / feature_spec dimensions match the model inputs

Usage:
    python tools/smoke_test.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

FAILS = []


def check(label, fn):
    try:
        msg = fn()
        print(f"  ✅ {label}{'  — ' + msg if msg else ''}")
        return True
    except Exception as e:
        print(f"  ❌ {label}\n       {type(e).__name__}: {e}")
        FAILS.append(label)
        return False


def main():
    print("=" * 66)
    print("1. Paths")
    print("=" * 66)
    print(f"  REPO_ROOT   = {paths.REPO_ROOT}")
    print(f"  DATASET_DIR = {paths.DATASET_DIR}")
    print(f"                (exists: {os.path.isdir(paths.DATASET_DIR)}"
          f" — fine if it doesn't exist; this test needs no dataset)")

    print()
    print("=" * 66)
    print("2. Model file inventory")
    print("=" * 66)
    # key files each category must have at least (missing ones mean files were misfiled or not moved)
    expect = [
        ("global", paths.GLOBAL_MODEL_DIR,
         ['global_fitness_model.keras', 'global_scaler_ts.pkl',
          'global_scaler_static.pkl', 'global_feature_spec.pkl']),
        ("loso_zeroshot", paths.LOSO_MODEL_DIR,
         ['feature_spec.pkl'] + [f'loso_without_S{i:02d}.keras' for i in range(1, 11)]
         + [f'loso_scaler_ts_S{i:02d}.pkl' for i in range(1, 11)]),
        ("personalized/from_loso", paths.PERSONALIZED_FROM_LOSO_DIR,
         [f'S{i:02d}_personalized.keras' for i in range(1, 11)]),
        ("personalized/from_loso_nontwo", paths.PERSONALIZED_NONTWO_DIR,
         ['S02_personalized_nontwo.keras']),
        ("personalized/from_global", paths.PERSONALIZED_FROM_GLOBAL_DIR,
         [f'S{i:02d}_personalized_fitness_model.keras' for i in range(1, 9)]),
        ("ablation/before_axisfix", paths.ABL_BEFORE_AXISFIX_DIR,
         ['loso_without_S01.keras', 'feature_spec.pkl']),
        ("ablation/masked_synergist", paths.ABL_MASKED_DIR,
         ['loso_without_S01.keras', 'feature_spec.pkl']),
        ("ablation/exclude_s09", paths.ABL_EXCLUDE_S09_DIR,
         ['loso_without_S01.keras', 'feature_spec.pkl']),
        ("ablation/holdout_two", paths.ABL_HOLDOUT_TWO_DIR,
         ['model_holdout_two.keras', 'holdout_two_scaler_ts.pkl',
          'holdout_two_scaler_static.pkl']),
    ]
    for name, d, required in expect:
        def _c(d=d, required=required):
            if not os.path.isdir(d):
                raise FileNotFoundError(f"directory does not exist: {d}")
            missing = [f for f in required if not os.path.isfile(os.path.join(d, f))]
            if missing:
                raise AssertionError(f"{len(missing)} files missing, e.g. {missing[0]}")
            n = len([f for f in os.listdir(d) if not f.startswith('.')])
            return f"{n} files, all key files present"
        check(f"models/{name}", _c)

    print()
    print("=" * 66)
    print("3. Load models and run inference (random input)")
    print("=" * 66)
    try:
        import numpy as np
        import joblib
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
        import tensorflow as tf
    except ImportError as e:
        print(f"  ⏭  skipped: missing package {e.name} (pip install -r requirements.txt)")
        return summarize()

    def _global():
        spec = joblib.load(os.path.join(paths.GLOBAL_MODEL_DIR, 'global_feature_spec.pkl'))
        s_ts = joblib.load(os.path.join(paths.GLOBAL_MODEL_DIR, 'global_scaler_ts.pkl'))
        s_st = joblib.load(os.path.join(paths.GLOBAL_MODEL_DIR, 'global_scaler_static.pkl'))
        m = tf.keras.models.load_model(
            os.path.join(paths.GLOBAL_MODEL_DIR, 'global_fitness_model.keras'))
        n_ts, n_st = len(spec['ts_cols']), len(spec['static_cols'])
        assert s_ts.n_features_in_ == n_ts, f"scaler_ts {s_ts.n_features_in_} != spec {n_ts}"
        assert s_st.n_features_in_ == n_st, f"scaler_static {s_st.n_features_in_} != spec {n_st}"
        out = m.predict({'ts_input': np.random.randn(2, 40, n_ts).astype('float32'),
                         'static_input': np.random.randn(2, n_st).astype('float32')},
                        verbose=0)
        out = out[0] if isinstance(out, (list, tuple)) else out
        return f"ts={n_ts} dims static={n_st} dims → output {tuple(np.shape(out))}"

    check("global/global_fitness_model.keras", _global)

    def _loso():
        spec = joblib.load(os.path.join(paths.LOSO_MODEL_DIR, 'feature_spec.pkl'))
        s_ts = joblib.load(os.path.join(paths.LOSO_MODEL_DIR, 'loso_scaler_ts_S01.pkl'))
        s_st = joblib.load(os.path.join(paths.LOSO_MODEL_DIR, 'loso_scaler_static_S01.pkl'))
        m = tf.keras.models.load_model(
            os.path.join(paths.LOSO_MODEL_DIR, 'loso_without_S01.keras'))
        n_ts, n_st = len(spec['ts_cols']), len(spec['static_cols'])
        assert s_ts.n_features_in_ == n_ts and s_st.n_features_in_ == n_st
        out = m.predict({'ts_input': np.random.randn(2, 40, n_ts).astype('float32'),
                         'static_input': np.random.randn(2, n_st).astype('float32')},
                        verbose=0)
        out = out[0] if isinstance(out, (list, tuple)) else out
        return f"ts={n_ts} dims static={n_st} dims → output {tuple(np.shape(out))}"

    check("loso_zeroshot/loso_without_S01.keras", _loso)

    def _personal():
        m = tf.keras.models.load_model(
            os.path.join(paths.PERSONALIZED_FROM_LOSO_DIR, 'S01_personalized.keras'))
        spec = joblib.load(os.path.join(paths.LOSO_MODEL_DIR, 'feature_spec.pkl'))
        n_ts, n_st = len(spec['ts_cols']), len(spec['static_cols'])
        out = m.predict({'ts_input': np.random.randn(2, 40, n_ts).astype('float32'),
                         'static_input': np.random.randn(2, n_st).astype('float32')},
                        verbose=0)
        out = out[0] if isinstance(out, (list, tuple)) else out
        return f"output {tuple(np.shape(out))} (scaler reuses the file of the same name in loso_zeroshot/)"

    check("personalized/from_loso/S01_personalized.keras", _personal)

    return summarize()


def summarize():
    print()
    print("=" * 66)
    if FAILS:
        print(f"❌ {len(FAILS)} failures:" + ", ".join(FAILS))
        return 1
    print("✅ All passed")
    return 0


if __name__ == '__main__':
    sys.exit(main())
