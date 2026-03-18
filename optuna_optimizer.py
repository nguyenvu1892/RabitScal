from __future__ import annotations
"""
optuna_optimizer.py — XGBoost Hyperparameter Optimizer v1.0
============================================================
Module: Rabit_Exness AI — Phase 4, Giai đoạn Tối Ưu Hóa Lượng Tử
Author: Antigravity (Senior AI Coder)
Date:   2026-03-16
Skill:  quant_optuna_expert.md — Bộ quy tắc sinh tử Optuna Expert

Pipeline:
    Load CSV(s) → Feature Engineering (40+ features) → Label Generation (BUY/SELL/HOLD)
    → Optuna Bayesian Search (TPE) trên XGBoost với TimeSeriesSplit(5)
    → Objective = F1-Score macro + Direction Penalty (phạt dự đoán ngược xu hướng)
    → Best Trial → Retrain Full Dataset → Export .pkl

Architecture:
    • TimeSeriesSplit(5)          — KHÔNG BAO GIỜ xáo trộn dữ liệu (anti data-leakage)
    • Optuna TPESampler + SQLite  — resume được khi server crash
    • gc.collect() mỗi trial      — anti memory leak trên ngàn trials
    • try/except trong objective   — trial lỗi → score -9999 → Optuna loại bỏ
    • XGBoost tree_method='hist'  — tăng tốc trên dữ liệu lớn

Usage:
    python optuna_optimizer.py                           # Fresh run 200 trials
    python optuna_optimizer.py --resume                  # Resume từ SQLite
    python optuna_optimizer.py --trials 500              # Override số trials
    python optuna_optimizer.py --symbol XAUUSD           # Chỉ optimize 1 symbol
    python optuna_optimizer.py --data-dir data/custom/   # Custom data directory

5 QUY TẮC SINH TỬ (từ quant_optuna_expert.md):
    1. CẤM train_test_split ngẫu nhiên     → dùng TimeSeriesSplit
    2. CẤM tối ưu theo Accuracy            → dùng F1-Score + Direction Penalty
    3. CẤM rò rỉ bộ nhớ                    → gc.collect() mỗi trial
    4. SQLite bất tử                        → resume khi ngắt điện
    5. try/except bọc objective             → -9999 khi lỗi, không crash

Hàm Mục Tiêu (Objective Function):
    Score = F1_macro × (1 - α × MissDirection_Ratio)
    
    Trong đó:
        F1_macro = F1-Score trung bình 3 class (BUY/SELL/HOLD)
        MissDirection_Ratio = tỉ lệ dự đoán sai xu hướng nghiêm trọng:
            - Dự đoán BUY nhưng thực tế SELL (giá sụp mạnh)
            - Dự đoán SELL nhưng thực tế BUY (giá bật mạnh)
        α = 2.0 (hệ số phạt — phạt NẶNG sai xu hướng ngược chiều)
"""

from re import S


import argparse
import gc
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

# XGBoost — import graceful
try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    xgb = None  # type: ignore[assignment]
    _XGB_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)

# ===========================================================================
# Constants
# ===========================================================================

PROJECT_ROOT  = Path(__file__).resolve().parent
DATA_DIR      = PROJECT_ROOT / "data"
MODELS_DIR    = DATA_DIR / "models"
LOGS_DIR      = PROJECT_ROOT / "logs"
CONFIG_DIR    = PROJECT_ROOT / "config"

# Defaults
DEFAULT_N_TRIALS       = 200
DEFAULT_N_SPLITS       = 5       # TimeSeriesSplit folds
DEFAULT_DIRECTION_PENALTY_ALPHA = 2.0   # Hệ số phạt sai xu hướng
DEFAULT_LOOKAHEAD_BARS = 12      # 12 nến M5 = 1 giờ nhìn tương lai cho label
DEFAULT_THRESHOLD_PCT  = 0.001   # Ngưỡng % biến động để xác định BUY/SELL (0.1%)

# Label mapping
LABEL_HOLD = 0
LABEL_BUY  = 1
LABEL_SELL = 2
LABEL_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}

# ===========================================================================
# Logging
# ===========================================================================

def _build_logger(name: str = "OptunaXGB") -> logging.Logger:
    """Tạo logger ghi ra cả console và file."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "[%(asctime)s UTC] - [%(levelname)-8s] - [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = RotatingFileHandler(
        LOGS_DIR / "optuna_optimizer.log",
        maxBytes=10 * 1024 * 1024, backupCount=5,
    )
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = _build_logger()


# ===========================================================================
# [SECTION 1] DATA LOADING — Đọc CSV từ data/
# ===========================================================================

def load_csv_data(
    data_dir: Path | str = DATA_DIR,
    symbol: Optional[str] = None,
) -> pd.DataFrame:
    """
    Nạp tất cả file .csv trong data_dir, hoặc chỉ file khớp symbol.

    CSV Schema kỳ vọng (ít nhất phải có):
        time, open, high, low, close, volume

    Args:
        data_dir: Thư mục chứa file CSV
        symbol: (optional) Filter file theo tên symbol (vd: 'XAUUSD' → tìm file chứa 'XAUUSD')

    Returns:
        pd.DataFrame với columns [time, open, high, low, close, volume] đã sort theo time
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No .csv files found in: {data_dir}")

    if symbol:
        csv_files = [f for f in csv_files if symbol.lower() in f.stem.lower()]
        if not csv_files:
            logger.warning(
                f"No CSV files matching symbol '{symbol}' — loading ALL CSVs in {data_dir}"
            )
            csv_files = sorted(data_dir.glob("*.csv"))

    frames: list[pd.DataFrame] = []
    required_cols = {"time", "open", "high", "low", "close", "volume"}

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            # Normalize column names to lowercase
            df.columns = [c.strip().lower() for c in df.columns]

            # Kiểm tra cột bắt buộc
            actual_cols = set(df.columns)
            missing = required_cols - actual_cols
            if missing:
                logger.warning(f"CSV {csv_path.name} missing columns: {missing} — skipping")
                continue

            df = df[["time", "open", "high", "low", "close", "volume"]].copy()
            df = df.dropna()

            # Ép kiểu numeric
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["time"] = pd.to_numeric(df["time"], errors="coerce")
            df = df.dropna()

            frames.append(df)
            logger.info(f"[DataLoad] Loaded {csv_path.name} | {len(df):,} rows")

        except Exception as e:
            logger.warning(f"[DataLoad] Error reading {csv_path.name}: {e} — skipping")
            continue

    if not frames:
        raise ValueError("No valid data loaded from CSV files")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("time").reset_index(drop=True)

    logger.info(
        f"[DataLoad] ✅ Total: {len(combined):,} rows from {len(frames)} file(s)"
    )
    return combined


# ===========================================================================
# [SECTION 2] FEATURE ENGINEERING — 40+ Features từ OHLCV
# ===========================================================================

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính toàn bộ features kỹ thuật từ OHLCV raw data.

    Features (40+):
        ─ Price Action:  returns, log_returns, candle body/wick ratios
        ─ Momentum:      RSI(14), ROC(5,10,20), Stochastic %K/%D
        ─ Volatility:    ATR(14), Bollinger Bands width/position, Keltner
        ─ Volume:        volume MA ratios, OBV, VWAP proxy
        ─ Trend:         EMA(9,21,50), EMA crossovers, ADX(14)
        ─ Pattern:       Pinbar score, Doji detector, Engulfing

    QUAN TRỌNG: KHÔNG dùng bất kỳ dữ liệu tương lai nào.
    Mọi indicator đều backward-looking (dùng .shift() when needed).

    Args:
        df: DataFrame với columns [time, open, high, low, close, volume]

    Returns:
        DataFrame với tất cả features đã tính + drop NaN warmup rows
    """
    feat = df.copy()
    o, h, l, c, v = feat["open"], feat["high"], feat["low"], feat["close"], feat["volume"]

    # ── Price Action ──────────────────────────────────────────────────────
    feat["returns_1"]   = c.pct_change(1)
    feat["returns_3"]   = c.pct_change(3)
    feat["returns_5"]   = c.pct_change(5)
    feat["log_return"]  = np.log(c / c.shift(1))

    # Candle anatomy
    total_range = h - l + 1e-10
    body = (c - o).abs()
    upper_wick = h - pd.concat([c, o], axis=1).max(axis=1)
    lower_wick = pd.concat([c, o], axis=1).min(axis=1) - l

    feat["body_ratio"]       = body / total_range
    feat["upper_wick_ratio"] = upper_wick / total_range
    feat["lower_wick_ratio"] = lower_wick / total_range
    feat["candle_direction"] = np.where(c >= o, 1, -1)  # bullish=1, bearish=-1

    # ── EMA Trend ─────────────────────────────────────────────────────────
    feat["ema_9"]  = c.ewm(span=9,  adjust=False).mean()
    feat["ema_21"] = c.ewm(span=21, adjust=False).mean()
    feat["ema_50"] = c.ewm(span=50, adjust=False).mean()

    feat["ema_9_21_cross"]  = (feat["ema_9"] - feat["ema_21"]) / c  # Normalized crossover
    feat["ema_21_50_cross"] = (feat["ema_21"] - feat["ema_50"]) / c
    feat["price_vs_ema_21"] = (c - feat["ema_21"]) / c
    feat["price_vs_ema_50"] = (c - feat["ema_50"]) / c

    # ── RSI(14) ───────────────────────────────────────────────────────────
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    feat["rsi_14"] = 100 - (100 / (1 + rs))

    # ── Rate of Change ────────────────────────────────────────────────────
    feat["roc_5"]  = c.pct_change(5)
    feat["roc_10"] = c.pct_change(10)
    feat["roc_20"] = c.pct_change(20)

    # ── Stochastic %K/%D ──────────────────────────────────────────────────
    low_14  = l.rolling(14).min()
    high_14 = h.rolling(14).max()
    feat["stoch_k"] = 100 * (c - low_14) / (high_14 - low_14 + 1e-10)
    feat["stoch_d"] = feat["stoch_k"].rolling(3).mean()

    # ── ATR(14) ───────────────────────────────────────────────────────────
    prev_c = c.shift(1)
    tr = pd.concat([
        h - l,
        (h - prev_c).abs(),
        (l - prev_c).abs(),
    ], axis=1).max(axis=1)
    feat["atr_14"]     = tr.rolling(14).mean()
    feat["atr_ratio"]  = feat["atr_14"] / (c + 1e-10)  # ATR normalized

    # ── Bollinger Bands (20, 2σ) ──────────────────────────────────────────
    sma_20 = c.rolling(20).mean()
    std_20 = c.rolling(20).std()
    feat["bb_upper"]   = sma_20 + 2 * std_20
    feat["bb_lower"]   = sma_20 - 2 * std_20
    feat["bb_width"]   = (feat["bb_upper"] - feat["bb_lower"]) / (sma_20 + 1e-10)
    feat["bb_position"] = (c - feat["bb_lower"]) / (feat["bb_upper"] - feat["bb_lower"] + 1e-10)

    # ── ADX(14) ───────────────────────────────────────────────────────────
    plus_dm  = (h - h.shift(1)).clip(lower=0)
    minus_dm = (l.shift(1) - l).clip(lower=0)
    # Zero out DM when the other DM is larger
    plus_dm  = np.where(plus_dm > minus_dm, plus_dm, 0)
    minus_dm = np.where(minus_dm > plus_dm, minus_dm, 0)  # type: ignore

    atr_14_smooth = tr.ewm(span=14, adjust=False).mean()
    plus_di  = 100 * pd.Series(plus_dm, index=df.index).ewm(span=14, adjust=False).mean() / (atr_14_smooth + 1e-10)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(span=14, adjust=False).mean() / (atr_14_smooth + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    feat["adx_14"]   = dx.ewm(span=14, adjust=False).mean()
    feat["plus_di"]  = plus_di
    feat["minus_di"] = minus_di

    # ── Volume Features ───────────────────────────────────────────────────
    feat["vol_ma_20"]    = v.rolling(20).mean()
    feat["vol_ratio"]    = v / (feat["vol_ma_20"] + 1e-10)
    feat["vol_change"]   = v.pct_change(1)

    # OBV (On-Balance Volume)
    obv_sign = np.where(c > c.shift(1), 1, np.where(c < c.shift(1), -1, 0))
    feat["obv"] = (v * obv_sign).cumsum()
    feat["obv_ema_9"] = feat["obv"].ewm(span=9, adjust=False).mean()
    feat["obv_signal"] = feat["obv"] - feat["obv_ema_9"]

    # VWAP proxy (cumulative)
    feat["vwap"] = (v * (h + l + c) / 3).cumsum() / (v.cumsum() + 1e-10)
    feat["price_vs_vwap"] = (c - feat["vwap"]) / (c + 1e-10)

    # ── Pattern Features ──────────────────────────────────────────────────
    # Pinbar score (0-1): wick dominance
    max_wick = pd.concat([upper_wick, lower_wick], axis=1).max(axis=1)
    feat["pinbar_score"] = max_wick / total_range

    # Doji (body very small)
    feat["is_doji"] = (feat["body_ratio"] < 0.1).astype(int)

    # Engulfing (simplified)
    prev_body = body.shift(1)
    feat["engulfing"] = np.where(
        (body > prev_body * 1.5) & (feat["candle_direction"] != feat["candle_direction"].shift(1)),
        feat["candle_direction"],
        0,
    )

    # ── Lagged features (để XGBoost bắt patterns qua nhiều nến) ─────────
    for lag in [1, 2, 3, 5]:
        feat[f"return_lag_{lag}"] = feat["returns_1"].shift(lag)
        feat[f"vol_ratio_lag_{lag}"] = feat["vol_ratio"].shift(lag)

    # ── Drop NaN warmup rows + raw columns ────────────────────────────────
    drop_cols = ["time", "open", "high", "low", "close", "volume",
                 "bb_upper", "bb_lower", "ema_9", "ema_21", "ema_50",
                 "vol_ma_20", "obv", "obv_ema_9", "vwap", "atr_14"]
    for col in drop_cols:
        if col in feat.columns:
            feat = feat.drop(columns=[col])

    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna()
    feat = feat.reset_index(drop=True)

    logger.info(f"[Features] ✅ {feat.shape[1]} features computed | {len(feat):,} samples after warmup")
    return feat


# ===========================================================================
# [SECTION 3] LABEL GENERATION — BUY/SELL/HOLD
# ===========================================================================

def generate_labels(
    df: pd.DataFrame,
    lookahead: int = DEFAULT_LOOKAHEAD_BARS,
    threshold: float = DEFAULT_THRESHOLD_PCT,
) -> np.ndarray:
    """
    Tạo nhãn BUY/SELL/HOLD dựa trên biến động giá trong N nến tương lai.

    Logic:
        future_return = (close[i+lookahead] - close[i]) / close[i]
        
        if future_return > +threshold:  → BUY  (giá sẽ tăng đủ mạnh)
        if future_return < -threshold:  → SELL (giá sẽ giảm đủ mạnh)
        else:                           → HOLD (sideway, không đủ biến động)

    Args:
        df: DataFrame gốc với column 'close'
        lookahead: Số nến nhìn tương lai (default: 12 = 1 giờ trên M5)
        threshold: Ngưỡng % biến động tối thiểu (default: 0.1%)

    Returns:
        np.ndarray shape (N,) với giá trị 0=HOLD, 1=BUY, 2=SELL
    """
    close = df["close"].values
    n = len(close)
    labels = np.full(n, LABEL_HOLD, dtype=np.int32)

    for i in range(n - lookahead):
        future_close = close[i + lookahead]
        current_close = close[i]
        if current_close == 0:
            continue
        future_return = (future_close - current_close) / current_close

        if future_return > threshold:
            labels[i] = LABEL_BUY
        elif future_return < -threshold:
            labels[i] = LABEL_SELL
        # else: HOLD (default)

    return labels


# ===========================================================================
# [SECTION 4] OBJECTIVE FUNCTION — LINH HỒN CỦA BOT
# ===========================================================================

def _compute_direction_penalty(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Tính tỉ lệ dự đoán SAI XU HƯỚNG NGƯỢC CHIỀU (critical misprediction).

    Sai xu hướng ngược chiều = dự đoán ngược chiều thị trường:
        - Predicted BUY  nhưng actual SELL → BUY vào nhưng giá sập → thua nặng
        - Predicted SELL nhưng actual BUY  → SELL vào nhưng giá bật → thua nặng

    KHÔNG phạt:
        - Predicted HOLD khi actual BUY/SELL → bỏ lỡ cơ hội (chấp nhận được)
        - Predicted BUY/SELL khi actual HOLD → rủi ro thấp (sideway)

    Returns:
        Float [0, 1] — tỉ lệ dự đoán sai xu hướng nghiêm trọng
        0 = hoàn hảo, 1 = tất cả dự đoán đều sai ngược chiều
    """
    # Chỉ xét các sample mà model dự đoán BUY hoặc SELL (mở mồm hô lệnh)
    action_mask = (y_pred == LABEL_BUY) | (y_pred == LABEL_SELL)
    n_actions = action_mask.sum()

    if n_actions == 0:
        return 0.0  # Không dự đoán gì → không có gì để phạt

    # Đếm sai ngược chiều nghiêm trọng
    wrong_direction = (
        ((y_pred == LABEL_BUY) & (y_true == LABEL_SELL)) |
        ((y_pred == LABEL_SELL) & (y_true == LABEL_BUY))
    )
    n_wrong = wrong_direction.sum()

    return float(n_wrong / n_actions)


def objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = DEFAULT_N_SPLITS,
    alpha: float = DEFAULT_DIRECTION_PENALTY_ALPHA,
) -> float:
    """
    Hàm Objective = F1_macro × (1 - α × MissDirection_Ratio)

    ĐÂY LÀ LINH HỒN CỦA BOT.

    Chiến lược:
        - F1-Score macro (cân bằng 3 class BUY/SELL/HOLD)
        - Direction Penalty: phạt NẶNG khi dự đoán ngược xu hướng
        - TimeSeriesSplit(5): KHÔNG xáo trộn, tôn trọng thứ tự thời gian
        - gc.collect() sau mỗi fold để chống rò rỉ bộ nhớ
        - try/except bọc toàn bộ → trả -9999 nếu lỗi

    Không gian tìm kiếm (Search Space):
        learning_rate:      LogUniform [0.01, 0.3]
        max_depth:          Int [3, 10]
        n_estimators:       Int [100, 1000]
        subsample:          Uniform [0.6, 1.0]
        colsample_bytree:   Uniform [0.6, 1.0]
        gamma:              LogUniform [1e-8, 1.0]
        min_child_weight:   Int [1, 10]
        reg_alpha:          LogUniform [1e-8, 10.0]
        reg_lambda:         LogUniform [1e-8, 10.0]
        scale_pos_weight:   Uniform [0.5, 3.0]

    Returns:
        Float — score để Optuna maximize. Lớn hơn = tốt hơn.
        -9999 nếu trial gặp lỗi nghiêm trọng.
    """
    # ── TOÀN BỘ BỌC TRONG try/except (Quy tắc #5) ────────────────────────
    try:
        # ── [1] Suggest tham số XGBoost ────────────────────────────────────
        params = {
            "objective":        "multi:softmax",
            "num_class":        3,
            "eval_metric":      "mlogloss",
            "tree_method":      "hist",        # Tăng tốc trên dữ liệu lớn
            "verbosity":        0,
            "use_label_encoder": False,
            "nthread":          -1,            # Dùng tất cả CPU cores
            "tree_method": "hist",
            "device": "cuda",
            
            # Optuna search space — đúng theo quant_optuna_expert.md
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "n_estimators":     trial.suggest_int("n_estimators", 100, 1000),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma":            trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        # ── [2] TimeSeriesSplit — CẤM xáo trộn (Quy tắc #1) ─────────────
        tscv = TimeSeriesSplit(n_splits=n_splits)
        f1_scores: list[float] = []
        direction_penalties: list[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Huấn luyện XGBoost
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            # Dự đoán
            y_pred = model.predict(X_val)

            # F1-Score macro (cân bằng BUY/SELL/HOLD)
            f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
            f1_scores.append(f1)

            # Direction Penalty (phạt sai xu hướng ngược chiều)
            penalty = _compute_direction_penalty(y_val, y_pred)
            direction_penalties.append(penalty)

            # ── Dọn dẹp bộ nhớ sau mỗi fold (Quy tắc #3) ────────────────
            del model, X_train, X_val, y_train, y_val, y_pred
            gc.collect()

        # ── [3] Tính Score tổng hợp ───────────────────────────────────────
        mean_f1      = np.mean(f1_scores)
        mean_penalty = np.mean(direction_penalties)

        # Score = F1 × (1 - α × MissDirection_Ratio)
        # α = 2.0 → phạt nặng gấp đôi tỉ lệ sai xu hướng
        composite_score = mean_f1 * (1.0 - alpha * mean_penalty)

        # Clamp score về [-1, 1] để Optuna không bị confused
        composite_score = max(composite_score, -1.0)

        # ── [4] Log rõ ràng cho Sếp Vũ theo dõi (Quy tắc #5) ────────────
        logger.info(
            f"Trial {trial.number:>4d} finished | "
            f"F1={mean_f1:.4f} | DirPenalty={mean_penalty:.4f} | "
            f"Score={composite_score:.4f} | "
            f"Best so far: {trial.study.best_value:.4f}"
            if trial.study.best_trial else
            f"Trial {trial.number:>4d} finished | "
            f"F1={mean_f1:.4f} | DirPenalty={mean_penalty:.4f} | "
            f"Score={composite_score:.4f} | "
            f"(First trial)"
        )

        # ── Dọn dẹp biến tạm cuối trial (Quy tắc #3) ────────────────────
        del f1_scores, direction_penalties
        gc.collect()

        return composite_score

    except Exception as e:
        # ── Trial lỗi → -9999 → Optuna loại bỏ (Quy tắc #5) ─────────────
        logger.error(
            f"Trial {trial.number:>4d} ERROR: {type(e).__name__}: {e} | "
            f"Returning penalty score -9999"
        )
        gc.collect()
        return -9999.0


# ===========================================================================
# [SECTION 5] MODEL EXPORT — ĐÚC NÃO (Retrain + Save .pkl)
# ===========================================================================

def retrain_and_export(
    X: np.ndarray,
    y: np.ndarray,
    best_params: dict,
    symbol: str = "default",
    feature_names: Optional[list[str]] = None,
) -> Path:
    """
    Retrain XGBoost trên TOÀN BỘ dataset với best params → export .pkl.

    ĐÂY LÀ BƯỚC ĐÚC NÃO — chuẩn bị cho giai đoạn Live.

    Args:
        X: Features array (toàn bộ dataset)
        y: Labels array (toàn bộ dataset)
        best_params: Best hyperparameters từ Optuna
        symbol: Tên symbol để đặt tên file
        feature_names: Danh sách tên features (để debug)

    Returns:
        Path tới file .pkl đã lưu
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"[Export] Retraining on FULL dataset: {X.shape[0]:,} samples...")

    # Chuẩn bị params cho final model
    final_params = dict(best_params)
    final_params.update({
        "objective":         "multi:softmax",
        "num_class":         3,
        "eval_metric":       "mlogloss",
        "tree_method":       "hist",
        "verbosity":         0,
        "use_label_encoder": False,
        "nthread":           -1,
    })

    # Retrain trên toàn bộ dữ liệu
    model = xgb.XGBClassifier(**final_params)
    model.fit(X, y, verbose=False)

    # Predict trên toàn bộ dữ liệu để đánh giá final metrics
    y_pred = model.predict(X)
    train_f1 = f1_score(y, y_pred, average="macro", zero_division=0)
    dir_penalty = _compute_direction_penalty(y, y_pred)

    logger.info(
        f"[Export] Full-train metrics | "
        f"F1_macro={train_f1:.4f} | DirPenalty={dir_penalty:.4f}"
    )

    # Classification report chi tiết
    report = classification_report(
        y, y_pred,
        target_names=["HOLD", "BUY", "SELL"],
        zero_division=0,
    )
    logger.info(f"[Export] Classification Report:\n{report}")

    # Save model + metadata
    model_path = MODELS_DIR / f"xgb_{symbol.lower()}.pkl"
    meta = {
        "symbol":         symbol,
        "best_params":    best_params,
        "train_samples":  int(X.shape[0]),
        "n_features":     int(X.shape[1]),
        "feature_names":  feature_names or [],
        "train_f1":       float(train_f1),
        "dir_penalty":    float(dir_penalty),
        "label_map":      LABEL_NAMES,
        "created_at":     datetime.now(timezone.utc).isoformat(),
    }

    bundle = {
        "model": model,
        "metadata": meta,
    }
    joblib.dump(bundle, model_path)

    # Save metadata JSON riêng để dễ đọc
    meta_path = MODELS_DIR / f"xgb_{symbol.lower()}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info(f"[Export] ✅ Model saved: {model_path}")
    logger.info(f"[Export] ✅ Metadata saved: {meta_path}")

    return model_path


# ===========================================================================
# [SECTION 6] MAIN OPTIMIZER — Orchestrates everything
# ===========================================================================

def run_optimization(
    data_dir: Path | str = DATA_DIR,
    symbol: Optional[str] = None,
    n_trials: int = DEFAULT_N_TRIALS,
    n_splits: int = DEFAULT_N_SPLITS,
    alpha: float = DEFAULT_DIRECTION_PENALTY_ALPHA,
    lookahead: int = DEFAULT_LOOKAHEAD_BARS,
    threshold: float = DEFAULT_THRESHOLD_PCT,
    resume: bool = False,
) -> dict:
    """
    Entry point chính — chạy toàn bộ pipeline optimization.

    Pipeline:
        1. Load CSV data
        2. Generate labels (BUY/SELL/HOLD)
        3. Compute features (40+)
        4. Align features + labels
        5. Run Optuna optimization (TPE + SQLite)
        6. Retrain on full dataset
        7. Export .pkl model

    Args:
        data_dir:  Thư mục chứa CSV files
        symbol:    Filter theo symbol (optional)
        n_trials:  Số Optuna trials
        n_splits:  Số folds TimeSeriesSplit
        alpha:     Hệ số phạt direction penalty
        lookahead: Số nến nhìn tương lai cho label
        threshold: Ngưỡng % biến động BUY/SELL
        resume:    True = resume study từ SQLite

    Returns:
        dict với best_params, best_score, model_path, etc.
    """
    # Kiểm tra XGBoost
    if not _XGB_AVAILABLE:
        raise ImportError(
            "XGBoost chưa được cài đặt!\n"
            "  → pip install xgboost\n"
            "  → Hoặc thêm 'xgboost>=2.0.0' vào requirements.txt"
        )

    t_start = time.perf_counter()
    sym_tag = symbol or "all"

    logger.info("=" * 70)
    logger.info(
        f"[Optimizer] 🚀 Starting XGBoost Optuna Optimization | "
        f"symbol={sym_tag} | trials={n_trials} | splits={n_splits} | "
        f"alpha={alpha} | lookahead={lookahead} | threshold={threshold}"
    )

    # ── [STEP 1] Load data ─────────────────────────────────────────────────
    raw_df = load_csv_data(data_dir, symbol)

    # ── [STEP 2] Generate labels ───────────────────────────────────────────
    labels_full = generate_labels(raw_df, lookahead=lookahead, threshold=threshold)

    # Label distribution log
    unique, counts = np.unique(labels_full, return_counts=True)
    dist_str = " | ".join(
        f"{LABEL_NAMES.get(u, '?')}={c} ({c/len(labels_full)*100:.1f}%)"
        for u, c in zip(unique, counts)
    )
    logger.info(f"[Labels] Distribution: {dist_str}")

    # ── [STEP 3] Feature engineering ───────────────────────────────────────
    features_df = compute_features(raw_df)
    feature_names = list(features_df.columns)

    # ── [STEP 4] Align features + labels ───────────────────────────────────
    # features_df đã drop NaN warmup rows → cần align labels
    # features_df index đã reset → dùng iloc tương ứng
    n_features = len(features_df)
    n_labels = len(labels_full)

    # Labels tương ứng với N rows cuối cùng của raw data
    # Features bắt đầu từ row (n_labels - n_features) của raw data
    offset = n_labels - n_features
    y = labels_full[offset: offset + n_features]

    # Cắt bỏ lookahead nến cuối (labels không chính xác ở cuối)
    X = features_df.values[:-lookahead]
    y = y[:-lookahead]

    logger.info(
        f"[Align] Final dataset: X={X.shape} | y={y.shape} | "
        f"features={len(feature_names)}"
    )

    if len(X) < 500:
        logger.warning(
            f"[Align] ⚠️ Dataset quá nhỏ ({len(X)} samples). "
            f"Cần ít nhất 500 samples cho TimeSeriesSplit(5) ổn định."
        )

    # ── [STEP 5] Optuna Study — SQLite bất tử (Quy tắc #4) ────────────────
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db_path = DATA_DIR / f"optuna_ev05d_{sym_tag.lower()}.db"
    storage_url = f"sqlite:///{db_path}"
    study_name = f"xgb_optuna_{sym_tag.lower()}"

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=20)

    if resume:
        logger.info(f"[Optuna] Resuming study from: {storage_url}")
        study = optuna.load_study(
            study_name=study_name,
            storage=storage_url,
            sampler=sampler,
        )
    else:
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            storage=storage_url,
            sampler=sampler,
            load_if_exists=True,  # Resume tự động nếu DB đã tồn tại
        )

    n_done = len([
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ])
    n_remaining = max(0, n_trials - n_done)

    logger.info(
        f"[Optuna] Study ready | done={n_done} | remaining={n_remaining} | "
        f"storage={storage_url}"
    )

    if n_remaining > 0:
        # Giảm noise Optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study.optimize(
            lambda trial: objective(
                trial, X, y,
                n_splits=n_splits,
                alpha=alpha,
            ),
            n_trials=n_remaining,
            show_progress_bar=True,
            gc_after_trial=True,  # Optuna tự gc.collect() thêm
        )

    # ── [STEP 6] Kết quả tốt nhất ─────────────────────────────────────────
    best = study.best_trial
    n_complete = len([
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ])
    n_pruned = len([
        t for t in study.trials
        if t.state == optuna.trial.TrialState.PRUNED
    ])

    logger.info(
        f"[Optuna] ✅ Optimization complete | "
        f"best_score={best.value:.4f} | "
        f"total_trials={n_complete} (pruned={n_pruned})"
    )
    logger.info(f"[Optuna] Best params: {best.params}")

    # ── [STEP 7] Retrain + Export .pkl (Đúc Não — Quy tắc #4) ─────────────
    model_path = retrain_and_export(
        X, y,
        best_params=best.params,
        symbol=sym_tag,
        feature_names=feature_names,
    )

    elapsed = time.perf_counter() - t_start
    logger.info(
        f"[Optimizer] 🏁 Pipeline complete | elapsed={elapsed:.1f}s | "
        f"model={model_path}"
    )
    logger.info("=" * 70)

    return {
        "best_params":    best.params,
        "best_score":     best.value,
        "n_trials":       n_complete,
        "n_pruned":       n_pruned,
        "model_path":     str(model_path),
        "db_path":        str(db_path),
        "duration_sec":   elapsed,
        "feature_count":  len(feature_names),
        "feature_names":  feature_names,
        "dataset_size":   X.shape[0],
    }


# ===========================================================================
# [SECTION 7] CLI ENTRY POINT
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RabitScal XGBoost Optuna Optimizer — "
            "F1-Score + Direction Penalty Objective"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbol", type=str, default=None,
        help="Filter CSV files by symbol name (e.g., XAUUSD, EURUSD)",
    )
    parser.add_argument(
        "--trials", type=int, default=DEFAULT_N_TRIALS,
        help="Number of Optuna optimization trials",
    )
    parser.add_argument(
        "--splits", type=int, default=DEFAULT_N_SPLITS,
        help="Number of TimeSeriesSplit folds",
    )
    parser.add_argument(
        "--alpha", type=float, default=DEFAULT_DIRECTION_PENALTY_ALPHA,
        help="Direction penalty coefficient (higher = harsher penalty)",
    )
    parser.add_argument(
        "--lookahead", type=int, default=DEFAULT_LOOKAHEAD_BARS,
        help="Number of bars to look ahead for label generation",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD_PCT,
        help="Price change threshold (%%) for BUY/SELL label",
    )
    parser.add_argument(
        "--data-dir", type=str, default=str(DATA_DIR),
        help="Directory containing CSV data files",
    )
    parser.add_argument(
        "--resume", action="store_true", default=False,
        help="Resume optimization from existing SQLite study",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity level",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Set log level
    logging.getLogger("OptunaXGB").setLevel(getattr(logging, args.log_level))

    print("\n" + "=" * 60)
    print("🐰 RabitScal — XGBoost Optuna Optimizer")
    print(f"   Skill: quant_optuna_expert.md")
    print(f"   Objective: F1-Score + Direction Penalty (α={args.alpha})")
    print(f"   TimeSeriesSplit: {args.splits} folds")
    print(f"   Trials: {args.trials}")
    print("=" * 60 + "\n")

    try:
        result = run_optimization(
            data_dir=args.data_dir,
            symbol=args.symbol,
            n_trials=args.trials,
            n_splits=args.splits,
            alpha=args.alpha,
            lookahead=args.lookahead,
            threshold=args.threshold,
            resume=args.resume,
        )

        print("\n" + "=" * 60)
        print("✅ OPTIMIZATION COMPLETE")
        print(f"   Best Score     : {result['best_score']:.4f}")
        print(f"   Trials         : {result['n_trials']} (pruned: {result['n_pruned']})")
        print(f"   Dataset Size   : {result['dataset_size']:,} samples")
        print(f"   Features       : {result['feature_count']}")
        print(f"   Duration       : {result['duration_sec']:.1f}s")
        print(f"   Model File     : {result['model_path']}")
        print(f"   Optuna DB      : {result['db_path']}")
        print("=" * 60)
        print("\n📊 Best Hyperparameters:")
        for k, v in result["best_params"].items():
            if isinstance(v, float):
                print(f"   {k:<25} = {v:.6f}")
            else:
                print(f"   {k:<25} = {v}")
        print()

    except FileNotFoundError as e:
        logger.error(f"❌ Data not found: {e}")
        print(f"\n❌ ERROR: {e}")
        print("   → Đặt file .csv vào thư mục data/")
        print("   → Hoặc chỉ định --data-dir <path>")
        sys.exit(1)

    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Unexpected error: {type(e).__name__}: {e}", exc_info=True)
        print(f"\n❌ FATAL ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
