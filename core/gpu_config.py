"""
core/gpu_config.py — GPU Configuration for XGBoost CUDA Training
==================================================================
Sếp Vũ: GTX 750 Ti 4GB VRAM → tree_method='hist' + device='cuda'

Usage:
    from core.gpu_config import get_xgb_params, check_gpu

    # Check GPU available
    gpu_ok = check_gpu()

    # Get XGBoost params with GPU acceleration
    params = get_xgb_params(n_estimators=500, max_depth=6)
"""

from __future__ import annotations

import os
import sys
import logging

log = logging.getLogger("GPU")

# ═══════════════════════════════════════════════════════════════════
#  GPU Detection
# ═══════════════════════════════════════════════════════════════════

def check_gpu() -> bool:
    """Check if CUDA GPU is available for XGBoost."""
    try:
        import xgboost as xgb
        # XGBoost 2.0+ uses device='cuda'
        # Try creating a small test model
        test = xgb.XGBClassifier(
            n_estimators=1, max_depth=1,
            tree_method='hist', device='cuda',
            verbosity=0,
        )
        log.info(f"XGBoost {xgb.__version__} — CUDA GPU OK")
        return True
    except Exception as e:
        log.warning(f"GPU not available: {e}. Falling back to CPU.")
        return False


def get_gpu_info() -> dict:
    """Get GPU memory info if available."""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            return {
                "name": parts[0],
                "total_mb": int(parts[1]),
                "free_mb": int(parts[2]),
            }
    except Exception:
        pass
    return {"name": "Unknown", "total_mb": 0, "free_mb": 0}


# ═══════════════════════════════════════════════════════════════════
#  XGBoost GPU Parameters
# ═══════════════════════════════════════════════════════════════════

def get_xgb_params(
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    min_child_weight: int = 5,
    gamma: float = 0.1,
    reg_alpha: float = 0.1,
    reg_lambda: float = 1.0,
    use_gpu: bool = True,
    **extra_params,
) -> dict:
    """
    Get XGBoost training parameters optimized for GTX 750 Ti 4GB VRAM.

    Key GPU params:
        tree_method='hist'  — Histogram-based algorithm, fastest on GPU
        device='cuda'       — Force CUDA GPU computation

    Args:
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth (6-8 optimal for 6GB GPU)
        learning_rate: Step size shrinkage
        use_gpu: Whether to use GPU (auto-fallback to CPU if unavailable)

    Returns:
        dict: XGBoost parameters ready for XGBClassifier/XGBRegressor
    """
    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "min_child_weight": min_child_weight,
        "gamma": gamma,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "verbosity": 0,
        "n_jobs": 1,           # GPU uses 1 thread
    }

    if use_gpu and check_gpu():
        # ═══ GTX 750 Ti 4GB VRAM — CUDA Acceleration ═══
        params["tree_method"] = "hist"
        params["device"] = "cuda"
        # GPU memory optimization for 4GB VRAM
        params["max_bin"] = 128            # Reduced for 4GB VRAM (avoid OOM)
        params["grow_policy"] = "depthwise"
        log.info(f"XGBoost GPU mode: tree_method=hist, device=cuda, "
                 f"max_depth={max_depth}, max_bin=128")
    else:
        # CPU fallback
        params["tree_method"] = "hist"
        params["device"] = "cpu"
        params["n_jobs"] = -1          # Use all CPU cores
        log.info("XGBoost CPU mode: tree_method=hist, device=cpu")

    params.update(extra_params)
    return params


def get_xgb_optuna_params(trial, use_gpu: bool = True) -> dict:
    """
    Get XGBoost params with Optuna hyperparameter suggestions.

    Usage in Optuna objective:
        from core.gpu_config import get_xgb_optuna_params
        params = get_xgb_optuna_params(trial)
        model = xgb.XGBClassifier(**params)
    """
    params = get_xgb_params(
        n_estimators=trial.suggest_int("n_estimators", 100, 1000),
        max_depth=trial.suggest_int("max_depth", 3, 8),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
        gamma=trial.suggest_float("gamma", 0.0, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 2.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.5, 5.0),
        use_gpu=use_gpu,
    )
    return params


# ═══════════════════════════════════════════════════════════════════
#  Quick Test
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="[%(levelname)s] %(message)s")

    print("=" * 50)
    print("  RabitScal GPU Config — Quick Test")
    print("=" * 50)

    # GPU Info
    info = get_gpu_info()
    print(f"\n  GPU:     {info['name']}")
    print(f"  VRAM:    {info['total_mb']} MB total, {info['free_mb']} MB free")

    # Check GPU
    gpu_ok = check_gpu()
    print(f"  CUDA:    {'AVAILABLE' if gpu_ok else 'NOT AVAILABLE'}")

    # Get params
    params = get_xgb_params(use_gpu=True)
    print(f"\n  XGBoost params:")
    for k, v in params.items():
        print(f"    {k}: {v}")

    # Quick training test
    if gpu_ok:
        try:
            import xgboost as xgb
            import numpy as np

            print(f"\n  Running GPU training test...")
            X = np.random.randn(1000, 10).astype(np.float32)
            y = (X.sum(axis=1) > 0).astype(int)

            model = xgb.XGBClassifier(**params)
            model.fit(X, y, verbose=False)
            score = model.score(X, y)
            print(f"  Training accuracy: {score:.4f}")
            print(f"  GPU TRAINING TEST PASSED!")
        except Exception as e:
            print(f"  GPU training failed: {e}")
    else:
        print("\n  [SKIP] GPU test — install xgboost-gpu:")
        print("  pip install xgboost --upgrade")

    print()
