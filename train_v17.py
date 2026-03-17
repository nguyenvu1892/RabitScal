"""
train_v17.py -- RabitScal AI Training Pipeline V17.0
=====================================================
Dan duoc vo lo roi! Script luyen nao AI moi nhat.

Pipeline:
    1. Load MTF data cho moi symbol (M1/M5/M15/H1) tu data/
    2. feature_engine.py V17.0 compute 85 features
    3. build_labels() tao nhan BUY/SELL/HOLD tu OHLCV tuong lai
    4. Gop tat ca symbols thanh 1 ma tran X/y thong nhat
    5. Optuna tune XGBoost hyperparameters (configurable trials)
    6. Luu model: data/models/xgb_all_v17.pkl (KHONG ghi de nao cu)

Usage:
    python train_v17.py                     # Full train, 100 Optuna trials
    python train_v17.py --trials 200        # Override so trials
    python train_v17.py --no-optuna         # Train voi default params (nhanh)
    python train_v17.py --gpu               # Force GPU training
    python train_v17.py --cpu               # Force CPU training

Author: Antigravity
Date:   2026-03-17
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Setup paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.feature_engine import build_feature_matrix, _atr, N_FEATURES
from core.xgb_classifier import build_labels, RabitScalClassifier

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR   = PROJECT_ROOT / "data"
MODEL_DIR  = DATA_DIR / "models"
LOGS_DIR   = PROJECT_ROOT / "logs"

# Output model path -- TUYET DOI KHONG GHI DE NAO CU
OUTPUT_MODEL = MODEL_DIR / "xgb_all_v17.pkl"

# Symbols co du 4 TF data trong data/
SYMBOLS = ["BTCUSD", "ETHUSD", "US30", "USTEC", "XAUUSD"]

# Spread cost per asset class (Exness Standard)
SPREAD_COSTS = {
    "BTCUSD": 150.0,
    "ETHUSD": 8.0,
    "US30":   0.03,
    "USTEC":  0.03,
    "XAUUSD": 20.0,
}

# Default XGBoost params (se duoc Optuna override neu chay tune)
DEFAULT_XGB_PARAMS = {
    "max_depth":        6,
    "n_estimators":     300,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma":            0.1,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "objective":        "multi:softprob",
    "num_class":        3,
    "eval_metric":      "mlogloss",
    "verbosity":        0,
}

# Labeling params
LABEL_LOOKAHEAD = 3    # nen tuong lai de xac dinh TP/SL
LABEL_RR        = 1.5  # Risk/Reward ratio

# Optuna defaults
OPTUNA_TRIALS   = 100
OPTUNA_METRIC   = "mlogloss"  # Metric: multi-class log loss (thap hon = tot hon)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def _setup_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("TrainV17")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "[%(asctime)s] - [%(levelname)-8s] - [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(LOGS_DIR / "train_v17.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


log = _setup_logger()


# ---------------------------------------------------------------------------
# Step 1: Build unified X, y from all symbols
# ---------------------------------------------------------------------------

def build_unified_dataset(symbols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Load feature_engine V17.0 features + build labels cho moi symbol.
    Gop tat ca thanh 1 unified (X, y) matrix.

    Returns:
        X: shape (N_total, 85)
        y: shape (N_total,) values in {0, 1, 2}
    """
    all_X = []
    all_y = []

    for sym in symbols:
        log.info(f"{'='*60}")
        log.info(f"  Processing: {sym}")
        log.info(f"{'='*60}")

        spread = SPREAD_COSTS.get(sym, 0.00015)

        try:
            # feature_engine.build_feature_matrix() tra ve (features, raw_m5, h1_ib)
            features, raw_m5, h1_ib = build_feature_matrix(
                symbol=sym,
                data_dir=str(DATA_DIR),
                spread_cost=spread,
            )
        except Exception as e:
            log.error(f"  [SKIP] {sym}: {e}")
            continue

        if features.shape[1] != N_FEATURES:
            log.error(
                f"  [SKIP] {sym}: Feature count mismatch! "
                f"Got {features.shape[1]}, expected {N_FEATURES}"
            )
            continue

        # Build labels tu raw_m5 OHLCV
        atr = _atr(raw_m5[:, 2], raw_m5[:, 3], raw_m5[:, 4], period=14)
        labels = build_labels(raw_m5, atr, lookahead=LABEL_LOOKAHEAD, rr=LABEL_RR)

        # Cat bo warmup bars (200 nen dau) -- noi indicators chua on dinh
        warmup = 200
        if len(features) > warmup:
            features = features[warmup:]
            labels = labels[warmup:]

        # Report
        n_buy  = (labels == 1).sum()
        n_sell = (labels == 2).sum()
        n_hold = (labels == 0).sum()
        log.info(
            f"  {sym}: {len(features)} samples | "
            f"BUY={n_buy} ({n_buy/len(labels)*100:.1f}%) | "
            f"SELL={n_sell} ({n_sell/len(labels)*100:.1f}%) | "
            f"HOLD={n_hold} ({n_hold/len(labels)*100:.1f}%)"
        )

        all_X.append(features)
        all_y.append(labels)

    if not all_X:
        raise RuntimeError("Khong co symbol nao load duoc data!")

    X = np.vstack(all_X).astype(np.float32)
    y = np.concatenate(all_y).astype(np.int32)

    log.info(f"\n{'='*60}")
    log.info(f"  UNIFIED DATASET: {X.shape[0]} samples x {X.shape[1]} features")
    log.info(f"  BUY={np.sum(y==1)}  SELL={np.sum(y==2)}  HOLD={np.sum(y==0)}")
    log.info(f"{'='*60}\n")

    return X, y


# ---------------------------------------------------------------------------
# Step 2: Optuna hyperparameter tuning
# ---------------------------------------------------------------------------

def run_optuna_tuning(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = OPTUNA_TRIALS,
    use_gpu: bool = False,
) -> dict:
    """
    Optuna Bayesian search for best XGBoost hyperparameters.
    Metric: mlogloss (cross-validated)

    Returns:
        Best params dict ready for XGBClassifier(**params)
    """
    import optuna
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    log.info(f"  [Optuna] Starting {n_trials} trials | Metric: {OPTUNA_METRIC}")

    def objective(trial: optuna.Trial) -> float:
        import xgboost as xgb

        params = {
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "n_estimators":     trial.suggest_int("n_estimators", 100, 800, step=50),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.001, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.001, 10.0, log=True),
            "objective":        "multi:softprob",
            "num_class":        3,
            "eval_metric":      "mlogloss",
            "verbosity":        0,
        }

        if use_gpu:
            params["tree_method"] = "hist"
            params["device"] = "cuda"
            params["max_bin"] = 64
            params["n_jobs"] = 1
        else:
            params["tree_method"] = "hist"
            params["device"] = "cpu"
            params["n_jobs"] = -1

        model = xgb.XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        scores = cross_val_score(
            model, X, y,
            cv=cv,
            scoring="neg_log_loss",
            n_jobs=1,
        )
        return -scores.mean()  # Optuna minimizes -> lower logloss = better

    study = optuna.create_study(
        direction="minimize",
        study_name="rabitscal_v17",
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    log.info(f"  [Optuna] Best trial #{study.best_trial.number}")
    log.info(f"  [Optuna] Best mlogloss: {study.best_value:.6f}")
    log.info(f"  [Optuna] Best params: {study.best_params}")

    # Merge best params with defaults
    best = DEFAULT_XGB_PARAMS.copy()
    best.update(study.best_params)
    return best


# ---------------------------------------------------------------------------
# Step 3: Train + Save
# ---------------------------------------------------------------------------

def train_and_save(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    use_gpu: bool = False,
    output_path: Path = OUTPUT_MODEL,
) -> Path:
    """
    Train XGBoost classifier voi params da chon va luu ra file .pkl

    Luu duoi dang dict:
        {
            "model": XGBClassifier,
            "features": [...85 feature names...],
            "n_features": 85,
            "version": "V17.0",
            "trained_at": "...",
            "params": {...},
            "samples": N,
        }
    """
    import xgboost as xgb

    # Set device params
    if use_gpu:
        params["tree_method"] = "hist"
        params["device"] = "cuda"
        params["max_bin"] = 64
        params["n_jobs"] = 1
    else:
        params["tree_method"] = "hist"
        params["device"] = "cpu"
        params["n_jobs"] = -1

    log.info(f"  [Train] Fitting XGBClassifier on {X.shape[0]} samples, {X.shape[1]} features...")
    t0 = time.time()

    model = xgb.XGBClassifier(**params)

    # Try GPU, fallback CPU
    try:
        model.fit(X.astype(np.float32), y.astype(np.int32))
    except Exception as e:
        if use_gpu:
            log.warning(f"  [Train] GPU error: {e}. Falling back to CPU...")
            params["device"] = "cpu"
            params["n_jobs"] = -1
            model = xgb.XGBClassifier(**params)
            model.fit(X.astype(np.float32), y.astype(np.int32))
        else:
            raise

    elapsed = time.time() - t0
    mode = "GPU" if params.get("device") == "cuda" else "CPU"
    log.info(f"  [Train] Done in {elapsed:.1f}s ({mode})")

    # Feature names (85 features V17.0)
    feature_names = [
        # M1 (0-3)
        "atr_m1_norm", "pinbar_m1", "vsa_m1", "spread_proxy_m1",
        # M5 (4-13)
        "atr_m5_raw", "pinbar_m5", "bull_pinbar_m5", "bear_pinbar_m5",
        "vsa_m5", "fvg_bull_m5", "fvg_bear_m5", "fvg_size_bull_m5", "fvg_size_bear_m5",
        "volume_ratio_m5",
        # M15 (14-21)
        "atr_m15_norm", "trend_ema_m15", "bos_bull_m15", "bos_bear_m15",
        "choch_bull_m15", "choch_bear_m15", "pinbar_m15", "price_vs_ema50_m15",
        # H1 (22-29)
        "atr_h1_norm", "trend_ema_h1", "price_vs_ema21_h1", "price_vs_ema50_h1",
        "bos_bull_h1", "bos_bear_h1", "choch_bull_h1", "choch_bear_h1",
        # Liquidity (30-33)
        "dist_eql_norm", "dist_eqh_norm", "eql_proximity", "eqh_proximity",
        # SMC/ICT (34-40)
        "dist_to_strong_high", "dist_to_strong_low",
        "dist_to_bull_ob", "dist_to_bear_ob", "bull_ob_active", "bear_ob_active",
        "fib_ote_level",
        # Session/ICT (41-44)
        "asia_range_norm", "in_killzone", "judas_swing", "dist_to_inst_price",
        # VSA Core (45-51)
        "relative_vol", "spread_norm", "effort_vs_result",
        "is_stopping_vol", "is_no_demand", "is_no_supply", "dist_to_rolling_poc",
        # Breaker/Flip (52-55)
        "dist_to_breaker", "dist_to_mitigation", "dist_to_s2d_flip", "dist_to_d2s_flip",
        # PA Basic (56-62)
        "body_size_norm", "upper_wick_pct", "lower_wick_pct", "close_pct_range",
        "is_bull_pinbar", "is_bear_pinbar", "is_bull_engulfing",
        # Compression (63-64)
        "is_compression", "compression_score",
        # Traps (65-68)
        "is_ib_fakeout", "is_bull_trap", "is_bear_trap", "is_opp_failure",
        # Volatility (69-70)
        "volatility_index", "spread_cost_norm",
        # Time (71-72)
        "hour_sin", "hour_cos",
        # V17.0 PA Advanced (73-80)
        "is_bear_engulfing", "is_doji", "is_outside_bar", "is_inside_bar_m5",
        "is_hammer", "is_shooting_star", "is_morning_star", "is_evening_star",
        # V17.0 VSA Session (81-84)
        "vol_vs_prev_ratio", "session_vol_percentile", "vol_acceleration", "session_vol_rank",
    ]

    # Save as dict bundle
    from datetime import datetime, timezone
    bundle = {
        "model":       model,
        "features":    feature_names,
        "n_features":  N_FEATURES,
        "version":     "V17.0",
        "trained_at":  datetime.now(timezone.utc).isoformat(),
        "params":      params,
        "samples":     len(X),
        "label_dist":  {
            "HOLD": int(np.sum(y == 0)),
            "BUY":  int(np.sum(y == 1)),
            "SELL": int(np.sum(y == 2)),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Safety: TUYET DOI KHONG ghi de nao cu
    old_path = output_path.parent / "xgb_all.pkl"
    if old_path.exists() and output_path != old_path:
        log.info(f"  [Safe] Old brain preserved at: {old_path}")

    with open(output_path, "wb") as f:
        pickle.dump(bundle, f)
    log.info(f"  [Save] Model saved: {output_path}")
    log.info(f"  [Save] Bundle keys: {list(bundle.keys())}")
    log.info(f"  [Save] Features: {N_FEATURES} | Samples: {len(X)} | Version: V17.0")

    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RabitScal AI Training Pipeline V17.0 -- Lo Luyen Dan"
    )
    parser.add_argument("--trials", type=int, default=OPTUNA_TRIALS,
                        help=f"So Optuna trials (default: {OPTUNA_TRIALS})")
    parser.add_argument("--no-optuna", action="store_true",
                        help="Skip Optuna, dung default params")
    parser.add_argument("--gpu", action="store_true",
                        help="Force GPU (CUDA) training")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU training")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS,
                        help=f"Symbols to train on (default: {SYMBOLS})")
    parser.add_argument("--output", type=str, default=str(OUTPUT_MODEL),
                        help=f"Output model path (default: {OUTPUT_MODEL})")
    args = parser.parse_args()

    use_gpu = args.gpu and not args.cpu

    print()
    print("=" * 60)
    print("  RabitScal AI -- LO LUYEN DAN V17.0")
    print("=" * 60)
    print(f"  Feature Engine : V17.0 ({N_FEATURES} features)")
    print(f"  Symbols        : {args.symbols}")
    print(f"  Optuna trials  : {'SKIP' if args.no_optuna else args.trials}")
    print(f"  Optuna metric  : {OPTUNA_METRIC}")
    print(f"  Device         : {'GPU (CUDA)' if use_gpu else 'CPU'}")
    print(f"  Output model   : {args.output}")
    print(f"  Label strategy : Lookahead={LABEL_LOOKAHEAD}, RR={LABEL_RR}")
    print("=" * 60)
    print()

    t_start = time.time()

    # Step 1: Build unified dataset
    log.info("STEP 1/3: Building unified dataset from feature_engine V17.0...")
    X, y = build_unified_dataset(args.symbols)

    # Step 2: Optuna tuning (optional)
    if args.no_optuna:
        log.info("STEP 2/3: SKIPPED (--no-optuna flag)")
        params = DEFAULT_XGB_PARAMS.copy()
    else:
        log.info(f"STEP 2/3: Optuna hyperparameter tuning ({args.trials} trials)...")
        params = run_optuna_tuning(X, y, n_trials=args.trials, use_gpu=use_gpu)

    # Step 3: Train + Save
    log.info("STEP 3/3: Training final model + saving...")
    output = train_and_save(
        X, y, params,
        use_gpu=use_gpu,
        output_path=Path(args.output),
    )

    total = time.time() - t_start
    print()
    print("=" * 60)
    print(f"  HOAN THANH! Lo Luyen Dan V17.0 da xong!")
    print(f"  Model:    {output}")
    print(f"  Features: {N_FEATURES}")
    print(f"  Samples:  {len(X)}")
    print(f"  Time:     {total/60:.1f} minutes")
    print("=" * 60)
    print()
    print("  Lenh chay production:")
    print("  1. Copy xgb_all_v17.pkl -> xgb_all.pkl (khi da test OK)")
    print("  2. Restart mt5_engine.py")
    print()


if __name__ == "__main__":
    main()
