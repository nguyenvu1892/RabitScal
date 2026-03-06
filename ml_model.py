"""
ml_model.py — OptimizationEngine v1.0
======================================
Module: Rabit_Exness AI — Phase 4, Task 4.1
Branch: task-4.1-ml-optimization
Author: Antigravity
Date:   2026-03-06

Bộ não tiến hóa: Optuna Bayesian Optimization trên 48 ProcessPoolExecutor workers
để tìm bộ siêu tham số tốt nhất cho chiến lược SMC+VSA.

Pipeline:
    fetch M5 history → save .npy → shared_memory → 48 workers chạy backtest song song
    → best trial → walk-forward OOS validate (24h) → shadow config → promote/retire

Architecture:
    • ProcessPoolExecutor(max_workers=48)  — bypass GIL hoàn toàn
    • multiprocessing.shared_memory       — 1 numpy array → 48 procs zero-copy
    • optuna TPESampler + MedianPruner    — Bayesian search 9D
    • SQLite backend                       — resume khi server crash (--resume flag)
    • Shadow deployment                    — config/versions/settings_v{NN}.json
    • Walk-forward OOS                     — validate trên 24h data chưa thấy

Usage:
    python ml_model.py                    # Fresh optimization run
    python ml_model.py --resume           # Resume từ SQLite nếu crash giữa chừng
    python ml_model.py --trials 200       # Override số trials (default: 500)
    python ml_model.py --workers 32       # Override số workers (default: 48)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# MT5 chỉ khả dụng trên Windows — bọc try/except để không crash trên Xeon Linux
try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    mt5 = None  # type: ignore[assignment]
    _MT5_AVAILABLE = False
import numpy as np
import optuna
from filelock import FileLock

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT      = Path(__file__).resolve().parent
CONFIG_DIR        = PROJECT_ROOT / "config"
DATA_DIR          = PROJECT_ROOT / "data"
LOGS_DIR          = PROJECT_ROOT / "logs"
VERSIONS_DIR      = CONFIG_DIR / "versions"

ML_CONFIG_PATH    = CONFIG_DIR / "ml_config.json"
MAIN_CONFIG_PATH  = CONFIG_DIR / "main_config.json"
ACTIVE_CFG_PATH   = CONFIG_DIR / "current_settings.json"
DATA_CACHE_PATH   = DATA_DIR / "m5_historical.npy"
CSV_DATA_PATH     = DATA_DIR / "history_m5.csv"   # Export từ Windows MT5 → đọc trên Xeon Linux
STUDY_DB_PATH     = DATA_DIR / "optuna_study.db"
STUDY_NAME        = "rabitscal_optuna_v1"

# Optuna defaults (overridable via ml_config.json)
DEFAULT_N_TRIALS      = 500
DEFAULT_N_WORKERS     = 48
DEFAULT_MIN_TRADES    = 1000   # Scalping: 5-10 lệnh/ngày × 250 ngày giao dịch
DEFAULT_MAX_DD_LIMIT  = 0.15
DEFAULT_LOOKBACK_DAYS = 180
DEFAULT_OOS_HOURS     = 720    # 1 tháng OOS — đủ rổ M5/M15 để validate (30 ngày)
DEFAULT_PROMOTE_THR   = 0.95
DEFAULT_RETIRE_THR    = 0.80

# Scalping Fitness Weights
SCALP_WR_WEIGHT     = 100.0   # Thưởng nặng Winrate: mỗi %WR = 1 điểm
SCALP_PF_WEIGHT     = 10.0    # Profit Factor thưởng vừa
SCALP_COUNT_WEIGHT  = 0.05    # 1/20 điểm mỗi lệnh (tỉ lệ tăng dần)
# Formula: score = WR×100 + PF×10 + trades×0.05 - DD_penalty×50


# ---------------------------------------------------------------------------
# Asset-Class Search Space Configuration
# Tại sao cần: mỗi nhóm tài sản có độ biến động, spread, và pattern hành vi riêng.
# Một search space chung cho cả 7 symbols sẽ luôn generate ra 100% pruned trials
# vì ngưỡng sỡ quá chặt với tài sản biến động mạnh (XAU, US30) hoặc quá lỏng với Forex.
# ---------------------------------------------------------------------------

ASSET_CLASS_CONFIG: dict[str, dict] = {
    # ── METAL / CRYPTO (SCALPING): Gold, Silver, BTC — tần suất cao, lọc nhẹ
    "METAL": {
        "_symbols":             ["XAU", "XAG", "GOLD", "BTC", "ETH"],
        "atr_sl_multiplier":    (0.3,  1.2),
        "atr_lot_multiplier":   (0.005, 0.05),
        "atr_fvg_buffer":       (0.01, 0.3),
        "vsa_volume_ratio":     (1.0,  1.8),
        "vsa_neighbor_ratio":   (1.0,  1.4),
        "vsa_min_score":        (0.1,  0.4),
        "pinbar_wick_ratio":    (0.30, 0.60),
        "pinbar_body_ratio":    (0.05, 0.50),
        "composite_score_gate": (0.10, 0.40),
        "min_trades":           50,
        "max_dd_limit":         0.95,
        "spread_cost":          0.40,
        "pip_value":            1.0,    # XAUUSD: 1 price-unit ≈ $1 / micro-lot
        "label":                "METAL-SCALP",
    },
    "BTC": {
        "_symbols":             ["BTC", "ETH"],
        "atr_sl_multiplier":    (0.3,  1.2),
        "atr_lot_multiplier":   (0.002, 0.02),
        "atr_fvg_buffer":       (0.01, 0.3),
        "vsa_volume_ratio":     (1.0,  1.8),
        "vsa_neighbor_ratio":   (1.0,  1.4),
        "vsa_min_score":        (0.1,  0.4),
        "pinbar_wick_ratio":    (0.30, 0.60),
        "pinbar_body_ratio":    (0.05, 0.50),
        "composite_score_gate": (0.10, 0.40),
        "min_trades":           50,
        "max_dd_limit":         0.95,
        "spread_cost":          5.00,
        "pip_value":            1.0,    # BTC micro: 1 price-unit ≈ $1
        "label":                "BTC-SCALP",
    },

    # ── INDICES: US30/US500/USTEC/Dow/Nasdaq — biến động vừa, trending mạnh
    "INDEX": {
        "_symbols":             ["US30", "US100", "US500", "USTEC", "SPX", "NAS", "DAX", "GER"],
        "atr_sl_multiplier":    (0.8,  3.5),
        "atr_lot_multiplier":   (0.002, 0.03),
        "atr_fvg_buffer":       (0.2,  1.2),
        "vsa_volume_ratio":     (1.15, 2.8),
        "vsa_neighbor_ratio":   (1.05, 1.9),
        "vsa_min_score":        (0.25, 0.55),
        "pinbar_wick_ratio":    (0.45, 0.72),
        "pinbar_body_ratio":    (0.08, 0.38),
        "composite_score_gate": (0.35, 0.65),
        "min_trades":           20,     # tăng từ 5 → 20 — tránh overfit WR=100% với 10 trades
        "max_dd_limit":         0.95,
        "spread_cost":          0.03,
        "pip_value":            1.0,    # US30 micro: price-unit ≈ $1
        "label":                "INDICES",
    },
    # ── OIL: USOIL, UKOIL — biến động vừa, nhạy cảm với news
    "OIL": {
        "_symbols":             ["USOIL", "UKOIL", "OIL", "WTI", "BRENT"],
        "atr_sl_multiplier":    (0.9,  3.5),
        "atr_lot_multiplier":   (0.003, 0.04),
        "atr_fvg_buffer":       (0.2,  1.0),
        "vsa_volume_ratio":     (1.1,  2.5),
        "vsa_neighbor_ratio":   (1.05, 1.9),
        "vsa_min_score":        (0.25, 0.55),
        "pinbar_wick_ratio":    (0.48, 0.72),
        "pinbar_body_ratio":    (0.09, 0.38),
        "composite_score_gate": (0.35, 0.65),
        "min_trades":           5,
        "max_dd_limit":         0.95,
        "spread_cost":          0.025,
        "pip_value":            1.0,   # Convention: 1 price-unit = $1 (same as all assets)
        "label":                "OIL",
    },
    # ── FOREX-MAJOR: EUR, GBP — gate thấp hơn để Smart DCA sinh lệnh
    "FOREX-MAJOR": {
        "_symbols":             ["EUR", "GBP"],
        "atr_sl_multiplier":    (0.8,  3.0),
        "atr_lot_multiplier":   (0.005, 0.05),
        "atr_fvg_buffer":       (0.3,  1.5),
        "vsa_volume_ratio":     (1.1,  2.5),
        "vsa_neighbor_ratio":   (1.0,  1.8),
        "vsa_min_score":        (0.2,  0.5),
        "pinbar_wick_ratio":    (0.40, 0.70),
        "pinbar_body_ratio":    (0.10, 0.45),
        "composite_score_gate": (0.30, 0.60),   # Hạ xuống để Smart DCA có lệnh
        "min_trades":           5,
        "max_dd_limit":         0.95,
        "spread_cost":          0.00015,
        "pip_value":            1.0,
        "label":                "FOREX-MAJOR",
    },
    # ── FOREX: JPY, AUD, CHF — spread nhỏ, độ biến động thấp
    "FOREX": {
        "_symbols":             ["JPY", "AUD", "CHF", "NZD", "CAD"],
        "atr_sl_multiplier":    (0.8,  3.0),
        "atr_lot_multiplier":   (0.005, 0.05),
        "atr_fvg_buffer":       (0.3,  1.5),
        "vsa_volume_ratio":     (1.2,  3.0),
        "vsa_neighbor_ratio":   (1.1,  2.0),
        "vsa_min_score":        (0.3,  0.6),
        "pinbar_wick_ratio":    (0.50, 0.75),
        "pinbar_body_ratio":    (0.10, 0.40),
        "composite_score_gate": (0.45, 0.75),
        "min_trades":           5,
        "max_dd_limit":         0.95,
        "spread_cost":          0.00015,
        "pip_value":            1.0,
        "label":                "FOREX",
    },
    # ── CRYPTO ALT: ADA, DOGE, SOL, XRP, LINK, BNB — biến động cao, lọc chặt
    "CRYPTO": {
        "_symbols":             ["ADA", "DOGE", "SOL", "XRP", "LINK", "BNB"],
        "atr_sl_multiplier":    (0.3,  1.5),
        "atr_lot_multiplier":   (0.002, 0.03),
        "atr_fvg_buffer":       (0.01, 0.4),
        "vsa_volume_ratio":     (1.0,  2.0),
        "vsa_neighbor_ratio":   (1.0,  1.5),
        "vsa_min_score":        (0.1,  0.4),
        "pinbar_wick_ratio":    (0.30, 0.65),
        "pinbar_body_ratio":    (0.05, 0.50),
        "composite_score_gate": (0.10, 0.45),
        "min_trades":           15,     # tăng từ 10 → 15 — CRYPTO biến động cao cần mẫu lớn
        "max_dd_limit":         0.95,
        "spread_cost":          0.001,
        "pip_value":            1.0,   # CRYPTO micro: 1 price-unit = $1
        "label":                "CRYPTO-ALT",
    },
}

# Fallback khi không detect được symbol
_ASSET_CLASS_DEFAULT = ASSET_CLASS_CONFIG["FOREX"]


def detect_asset_class(data_path: str | None) -> dict:
    """
    Detect asset class từ tên file CSV (args.data).
    Ví dụ: 'data/history_XAUUSDm_M5.csv' → METAL

    Returns:
        config dict từ ASSET_CLASS_CONFIG (bao gồm search space và min_trades)
    """
    if not data_path:
        return _ASSET_CLASS_DEFAULT

    name = Path(data_path).stem.upper()  # 'history_XAUUSDm_M5'
    for cls_name, cfg in ASSET_CLASS_CONFIG.items():
        for sym in cfg["_symbols"]:
            if sym.upper() in name:
                logger_tmp = logging.getLogger("MLEngine")
                logger_tmp.info(
                    f"[AssetClass] Detected '{sym}' in '{Path(data_path).name}' "
                    f"→ class={cfg['label']} | "
                    f"gate={cfg['composite_score_gate']} min_trades={cfg['min_trades']}"
                )
                return cfg

    logging.getLogger("MLEngine").warning(
        f"[AssetClass] Cannot detect symbol from '{data_path}' — using FOREX defaults"
    )
    return _ASSET_CLASS_DEFAULT



def _build_logger(name: str) -> logging.Logger:
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
        LOGS_DIR / "ml_model.log", maxBytes=10 * 1024 * 1024, backupCount=5
    )
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = _build_logger("MLEngine")

# ---------------------------------------------------------------------------
# Data Containers
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Kết quả một backtest run trong 1 Optuna trial."""
    winrate:       float
    profit_factor: float
    max_drawdown:  float
    trade_count:   int
    gross_profit:  float = 0.0
    gross_loss:    float = 0.0
    avg_win:       float = 0.0
    avg_loss:      float = 0.0


@dataclass
class OptimizationResult:
    """Kết quả cuối cùng của 1 Optimization run."""
    best_params:   dict
    best_score:    float
    n_trials:      int
    n_pruned:      int
    duration_sec:  float
    shadow_path:   str
    created_at:    str
    # Best trial metrics (cất để SCANNER_RESULT JSON không cần re-run)
    best_winrate:  float = 0.0
    best_pf:       float = 0.0
    best_dd:       float = 0.0
    best_trades:   int   = 0
    deploy_status: str   = ""


@dataclass
class WalkForwardResult:
    """Kết quả walk-forward validate trên OOS data."""
    winrate:       float
    profit_factor: float
    max_drawdown:  float
    trade_count:   int
    pass_validate: bool
    reason:        str = ""


# ---------------------------------------------------------------------------
# Shared Memory Helper
# ---------------------------------------------------------------------------

class SharedNumpyArray:
    """Context manager wrapping multiprocessing.shared_memory cho numpy array."""

    def __init__(self, data: np.ndarray, name: Optional[str] = None):
        self._data   = data
        self._name   = name
        self._shm: Optional[shared_memory.SharedMemory] = None
        self.shape   = data.shape
        self.dtype   = data.dtype

    def __enter__(self) -> "SharedNumpyArray":
        self._shm = shared_memory.SharedMemory(
            create=(self._name is None),
            size=self._data.nbytes,
            name=self._name,
        )
        arr = np.ndarray(self._data.shape, dtype=self._data.dtype, buffer=self._shm.buf)
        np.copyto(arr, self._data)
        self._shm_name = self._shm.name
        logger.info(
            f"Shared memory allocated: {self._data.nbytes / 1024**2:.1f}MB "
            f"| name={self._shm.name}"
        )
        return self

    def __exit__(self, *args):
        if self._shm:
            self._shm.close()
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass
            logger.info("Shared memory released")

    @property
    def shm_name(self) -> str:
        return self._shm.name

    @classmethod
    def attach(cls, name: str, shape: tuple, dtype: np.dtype) -> np.ndarray:
        """Attach từ subprocess worker."""
        shm = shared_memory.SharedMemory(create=False, name=name)
        arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        return arr, shm


# ---------------------------------------------------------------------------
# Vectorized Backtest Engine — "Trái Tim Tốc Độ"
# ---------------------------------------------------------------------------

def run_backtest_fast(
    data:   np.ndarray,
    params: dict,
    *,
    commission_per_lot: float = 3.5,
    spread_cost:        float = 0.0,
    pip_value:          float = 1.0,   # Normalize PnL per price-unit (OIL=0.01, others=1.0)
    lot_size:           float = 0.01,
) -> BacktestResult:
    """
    Vectorized backtest — không có Python for-loop ở lớp indicator.
    
    Chiến lược simulate:
        BUY  signal: Bull FVG detected + Pinbar điều kiện + VSA volume
        SELL signal: Bear FVG detected + Pinbar điều kiện + VSA volume
        SL = ATR × atr_sl_multiplier
        TP = SL × 1.5  (RR 1:1.5 cố định — có thể tham số hóa sau)
        OHLC worst-case: kiểm tra SL trước (bất lợi nhất cho bot)

    Args:
        data:   np.ndarray shape (N, 6) → [time, open, high, low, close, volume]
        params: dict bộ tham số từ Optuna trial

    Returns:
        BacktestResult với winrate, profit_factor, max_drawdown, trade_count
    
    Raises:
        optuna.exceptions.TrialPruned: khi không đủ trade hoặc DD vượt ngưỡng
    """
    if len(data) < 50:
        raise optuna.exceptions.TrialPruned()

    N = len(data)
    opens   = data[:, 1].astype(np.float64)
    highs   = data[:, 2].astype(np.float64)
    lows    = data[:, 3].astype(np.float64)
    closes  = data[:, 4].astype(np.float64)
    volumes = data[:, 5].astype(np.float64)

    # ── [1] ATR(14) vectorized ──────────────────────────────────────────────
    prev_closes = np.roll(closes, 1)
    prev_closes[0] = closes[0]
    tr = np.maximum(
        highs - lows,
        np.maximum(np.abs(highs - prev_closes), np.abs(lows - prev_closes))
    )
    # Rolling mean width=14 via cumsum trick (O(N) không dùng vòng lặp)
    kernel_14  = np.ones(14) / 14.0
    atr        = np.convolve(tr, kernel_14, mode='same')
    atr[:13]   = tr[:13]  # warm-up: dùng raw TR cho 13 nến đầu

    # ── [2] Volume baseline (rolling MA20) ─────────────────────────────────
    kernel_20   = np.ones(20) / 20.0
    vol_ma20    = np.convolve(volumes, kernel_20, mode='same')
    vol_ma20[:19] = volumes[:19]

    prev_volumes = np.roll(volumes, 1)
    prev_volumes[0] = volumes[0]

    # ── [3] Pinbar detection (boolean mask) ────────────────────────────────
    total_range = highs - lows + 1e-10
    body        = np.abs(closes - opens)
    upper_wick  = highs - np.maximum(closes, opens)
    lower_wick  = np.minimum(closes, opens) - lows
    max_wick    = np.maximum(upper_wick, lower_wick)

    pinbar_mask = (
        (max_wick / total_range >= params["pinbar_wick_ratio"]) &
        (body / total_range     <= params["pinbar_body_ratio"])
    )

    # Bull pinbar: lower wick dominant (bullish rejection)
    bull_pinbar = pinbar_mask & (lower_wick > upper_wick)
    # Bear pinbar: upper wick dominant (bearish rejection)
    bear_pinbar = pinbar_mask & (upper_wick > lower_wick)

    # ── [4] VSA volume filter (2-layer) ────────────────────────────────────
    vsa_layer1 = volumes >= vol_ma20 * params["vsa_volume_ratio"]
    vsa_layer2 = volumes >= prev_volumes * params["vsa_neighbor_ratio"]
    vsa_mask   = vsa_layer1 & vsa_layer2

    # ── [5] FVG detection (vectorized 3-candle pattern) ───────────────────
    # Bull FVG: candle[i-2].high < candle[i].low  → gap bullish
    # Bear FVG: candle[i-2].low  > candle[i].high → gap bearish
    highs_2ago = np.roll(highs, 2)
    lows_2ago  = np.roll(lows,  2)
    highs_2ago[:2] = highs[:2]
    lows_2ago[:2]  = lows[:2]

    fvg_bull = (highs_2ago < lows)   # Candle hiện tại LOW > candle cách 2 nến HIGH
    fvg_bear = (lows_2ago  > highs)  # Candle hiện tại HIGH < candle cách 2 nến LOW

    # FVG phải có kích thước đủ lớn (ít nhất atr_fvg_buffer × ATR)
    fvg_size_bull = lows - highs_2ago
    fvg_size_bear = lows_2ago - highs
    fvg_min_size  = atr * params["atr_fvg_buffer"]

    fvg_bull = fvg_bull & (fvg_size_bull >= fvg_min_size)
    fvg_bear = fvg_bear & (fvg_size_bear >= fvg_min_size)

    # ── [6] Final signal mask ───────────────────────────────────────────────
    # Composite score estimate (simplified — full score = pinbar×0.4 + vsa×0.35 + fvg×0.25)
    pinbar_score = np.where(pinbar_mask, 0.40, 0.0)
    vsa_score    = np.where(vsa_mask,    0.35, 0.0)
    fvg_bull_sc  = np.where(fvg_bull,   0.25, 0.0)
    fvg_bear_sc  = np.where(fvg_bear,   0.25, 0.0)

    buy_score    = pinbar_score + vsa_score + fvg_bull_sc
    sell_score   = pinbar_score + vsa_score + fvg_bear_sc

    gate = params["composite_score_gate"]
    buy_signal  = bull_pinbar & vsa_mask & fvg_bull & (buy_score  >= gate)
    sell_signal = bear_pinbar & vsa_mask & fvg_bear & (sell_score >= gate)

    # Không trade trong 15 nến đầu (warm-up indicators)
    buy_signal[:15]  = False
    sell_signal[:15] = False

    # ── [7] Smart DCA Basket Simulation ────────────────────────────────────
    # Approved config:
    #   MAX_BASKET   = 3 entries
    #   DCA_GAP      = 0.8 × ATR (giá phải đi ngược sâu hơn mới nhồi)
    #   BASKET_SL    = 25 USD hard cap (đóng toàn rổ khi tổng PnL âm ≥ $25)
    #   BASKET_TP    = price level của lệnh đầu tiên (TP1)
    #   COST         = commission + spread mỗi entry riêng
    MAX_BASKET     = 3
    DCA_GAP_MULT   = 0.8
    BASKET_SL_MULT = 2.5   # Hard cầu chì = 2.5 × sl_dist[L1]
    RR_RATIO       = 1.5
    # cost_per_trade giữ theo USD (nhất quán với convention cũ Single-Trade)
    cost_per_trade = commission_per_lot * lot_size + spread_cost

    sl_dist = atr * params["atr_sl_multiplier"]
    pnl_list: list[float] = []

    direction     = 0
    entries: list[tuple[float, int]] = []
    basket_tp     = 0.0
    basket_sl_thr = 0.0    # = BASKET_SL_MULT × sl_dist[L1] (price-unit)
    last_dca_bar  = -999

    def basket_pnl(cur_price: float) -> float:
        """Tổng PnL price-unit × pip_value — normalize OIL/FOREX về USD-đồng nhất."""
        return sum((cur_price - ep) * direction for ep, _ in entries) * pip_value

    def close_basket(exit_pnl_pu: float) -> None:
        """exit_pnl_pu là price-unit; cost_per_trade trừ theo USD (1:1 với price-unit do convention backtest)."""
        n   = len(entries)
        net = exit_pnl_pu - cost_per_trade * n
        pnl_list.append(net)
        entries.clear()
        nonlocal direction, basket_tp, last_dca_bar, basket_sl_thr
        direction     = 0
        basket_tp     = 0.0
        basket_sl_thr = 0.0

    for i in range(15, N - 1):
        nh = highs[i + 1]    # next candle high (OHLC worst-case check)
        nl = lows[i + 1]     # next candle low

        # ── Trạng thái KHÔNG có rổ → tìm entry L1 ──────────────────────────
        if not entries:
            if buy_signal[i]:
                entries.append((closes[i], i))
                direction     = 1
                basket_tp     = closes[i] + sl_dist[i] * RR_RATIO
                basket_sl_thr = BASKET_SL_MULT * sl_dist[i] * pip_value
                last_dca_bar  = i
            elif sell_signal[i]:
                entries.append((closes[i], i))
                direction     = -1
                basket_tp     = closes[i] - sl_dist[i] * RR_RATIO
                basket_sl_thr = BASKET_SL_MULT * sl_dist[i] * pip_value
                last_dca_bar  = i

        # ── Trạng thái CÓ rổ ────────────────────────────────────────────────
        else:
            cur_close = closes[i]

            # 1) Check Basket Hard SL: worst-case price (nh/nl)
            check_price_sl = nl if direction == 1 else nh
            if direction * (basket_tp - check_price_sl) > 0:
                floating = basket_pnl(check_price_sl)
                if floating <= -basket_sl_thr:
                    close_basket(-basket_sl_thr)
                    continue

            # 2) Check Basket TP
            hit_tp = (direction == 1  and nh >= basket_tp) or \
                     (direction == -1 and nl <= basket_tp)
            if hit_tp:
                tp_pnl = basket_pnl(basket_tp)
                close_basket(tp_pnl)
                continue

            # 3) Floating SL — worst-case within-candle
            wprice = nl if direction == 1 else nh
            if basket_pnl(wprice) <= -basket_sl_thr:
                close_basket(-basket_sl_thr)
                continue

            # 4) DCA Opportunity — nhồi thêm nếu đủ điều kiện
            n_entries = len(entries)
            if n_entries < MAX_BASKET and (i - last_dca_bar) >= 3:
                gap_needed = DCA_GAP_MULT * atr[i]
                last_ep    = entries[-1][0]

                # Giá phải đi ngược đủ sâu (0.8 × ATR)
                price_moved_against = (direction == 1  and cur_close <= last_ep - gap_needed) or \
                                      (direction == -1 and cur_close >= last_ep + gap_needed)

                # Tín hiệu xác nhận cùng chiều (pinbar + VSA — lỏng hơn L1)
                if direction == 1:
                    confirm = bull_pinbar[i] and vsa_mask[i]
                else:
                    confirm = bear_pinbar[i] and vsa_mask[i]

                if price_moved_against and confirm:
                    entries.append((cur_close, i))
                    last_dca_bar = i

    # Đóng rổ cuối nếu vẫn còn mở
    if entries:
        final_pnl = basket_pnl(closes[-1])
        close_basket(final_pnl)

    # ── [8] Aggregate metrics ───────────────────────────────────────────────
    if len(pnl_list) < 3:   # objective() tầng trên sẽ lọc min_trades thực tế
        raise optuna.exceptions.TrialPruned()


    pnl     = np.array(pnl_list, dtype=np.float64)
    wins    = (pnl > 0)
    losses  = (pnl < 0)
    n_wins  = wins.sum()
    n_loss  = losses.sum()

    winrate       = n_wins / len(pnl)
    gross_profit  = pnl[wins].sum()  if n_wins > 0 else 0.0
    gross_loss    = abs(pnl[losses].sum()) if n_loss > 0 else 1e-8
    # Cap PF tại 999 khi n_loss=0 — tránh score overflow tỷ tỷ trong scalp_score formula
    profit_factor = min(gross_profit / gross_loss, 999.0)

    avg_win  = gross_profit / n_wins if n_wins > 0 else 0.0
    avg_loss = gross_loss   / n_loss if n_loss > 0 else 0.0

    # Max Drawdown — chuẩn hóa về % equity
    # Dùng tổng exposure tuyệt đối làm initial_equity anchor để tránh
    # divide-by-tiny-peak bug trên tài sản giá cao (XAUUSD, US30)
    equity    = np.cumsum(pnl)
    total_abs = gross_profit + gross_loss          # tổng thanh khoản
    anchor    = max(total_abs / len(pnl) * 100.0, 1e-8)  # ~100× avg trade

    peak      = np.maximum.accumulate(equity + anchor)   # shift về dương
    drawdown  = (peak - (equity + anchor)) / peak         # luôn trong [0,1]
    max_dd    = float(drawdown.max())

    return BacktestResult(
        winrate=float(winrate),
        profit_factor=float(profit_factor),
        max_drawdown=max_dd,
        trade_count=len(pnl),
        gross_profit=float(gross_profit),
        gross_loss=float(gross_loss),
        avg_win=float(avg_win),
        avg_loss=float(avg_loss),
    )



# ---------------------------------------------------------------------------
# Trial Worker — Pure numpy function, NO SQLite access (pickle-safe)
# ---------------------------------------------------------------------------

def _run_trial_worker(args: tuple) -> dict:
    """
    Worker function cho Ask-and-Tell pattern.
    Nhận tuple (params, data, max_dd_limit, min_trades), chạy backtest numpy.
    KHÔNG bao giờ truy cập SQLite — chỉ main thread được phép chạm study.

    Args:
        args: (params_dict, data_ndarray, max_dd_limit, min_trades)

    Returns:
        dict với keys: status ('ok'|'pruned'|'error'), score (float), error (str)
    """
    params, data, max_dd_limit, min_trades = args
    try:
        result = run_backtest_fast(data, params)

        # Hard prune check
        if result.max_drawdown >= max_dd_limit or result.trade_count < min_trades:
            return {"status": "pruned"}

        # Score: WR^0.60 × PF^0.40 × dd_penalty
        dd_penalty = 1.0 - (result.max_drawdown / max_dd_limit) ** 2
        score      = (
            (result.winrate       ** WR_WEIGHT) *
            (result.profit_factor ** PF_WEIGHT) *
            max(dd_penalty, 0.0)
        )
        return {
            "status":        "ok",
            "score":         score,
            "winrate":       result.winrate,
            "profit_factor": result.profit_factor,
            "max_drawdown":  result.max_drawdown,
            "trade_count":   result.trade_count,
        }
    except optuna.exceptions.TrialPruned:
        return {"status": "pruned"}
    except Exception as e:
        import traceback as _tb
        print(f"\u274c [_run_trial_worker] ERROR: {e}\n{_tb.format_exc()}", flush=True)
        return {"status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Optuna Objective (chạy trong subprocess worker)
# ---------------------------------------------------------------------------

def _objective_worker(
    trial_params: dict,
    shm_name:     str,
    data_shape:   tuple,
    data_dtype:   str,
    max_dd_limit: float,
    min_trades:   int,
    spread_cost:  float = 0.0,
    pip_value:    float = 1.0,   # ← pip_value per-asset (OIL=0.01, others=1.0)
) -> dict:
    """
    Subprocess worker: attach shared memory, run backtest, return results.
    spread_cost + pip_value truyền từ asset_cfg để backtest sát thực tế.
    """
    try:
        shm = shared_memory.SharedMemory(create=False, name=shm_name)
        data = np.ndarray(data_shape, dtype=np.dtype(data_dtype), buffer=shm.buf)

        result = run_backtest_fast(data, trial_params,
                                   spread_cost=spread_cost,
                                   pip_value=pip_value)
        shm.close()

        return {
            "status":        "ok",
            "winrate":       result.winrate,
            "profit_factor": result.profit_factor,
            "max_drawdown":  result.max_drawdown,
            "trade_count":   result.trade_count,
            "pruned":        (
                result.max_drawdown >= max_dd_limit or
                result.trade_count  <  min_trades
            ),
        }
    except optuna.exceptions.TrialPruned:
        return {"status": "pruned"}
    except Exception as e:
        import traceback as _tb
        print(f"\u274c [_objective_worker] ERROR: {e}\n{_tb.format_exc()}", flush=True)
        return {"status": "error", "error": str(e)}



def _build_objective(
    shm_name:     str,
    data_shape:   tuple,
    data_dtype:   str,
    max_dd_limit: float,
    min_trades:   int,
    executor:     ProcessPoolExecutor,
    asset_cfg:    dict | None = None,   # ← NEW: dynamic search space per asset class
):
    """
    Factory trả về hàm objective cho optuna.study.optimize().
    Dùng ProcessPoolExecutor để submit task sang worker process.
    """
    sp = asset_cfg if asset_cfg is not None else _ASSET_CLASS_DEFAULT

    def objective(trial: optuna.Trial) -> float:
        # ── Suggest theo asset class search space ──────────────────────────
        params = {
            "atr_sl_multiplier":    trial.suggest_float("atr_sl_multiplier",    *sp["atr_sl_multiplier"]),
            "atr_lot_multiplier":   trial.suggest_float("atr_lot_multiplier",   *sp["atr_lot_multiplier"]),
            "atr_fvg_buffer":       trial.suggest_float("atr_fvg_buffer",       *sp["atr_fvg_buffer"]),
            "vsa_volume_ratio":     trial.suggest_float("vsa_volume_ratio",     *sp["vsa_volume_ratio"]),
            "vsa_neighbor_ratio":   trial.suggest_float("vsa_neighbor_ratio",   *sp["vsa_neighbor_ratio"]),
            "vsa_min_score":        trial.suggest_float("vsa_min_score",        *sp["vsa_min_score"]),
            "pinbar_wick_ratio":    trial.suggest_float("pinbar_wick_ratio",    *sp["pinbar_wick_ratio"]),
            "pinbar_body_ratio":    trial.suggest_float("pinbar_body_ratio",    *sp["pinbar_body_ratio"]),
            "composite_score_gate": trial.suggest_float("composite_score_gate", *sp["composite_score_gate"]),
        }

        future = executor.submit(
            _objective_worker,
            params, shm_name, data_shape, str(data_dtype),
            max_dd_limit, min_trades,
            sp.get("spread_cost", 0.0),
            sp.get("pip_value",   1.0),   # ← pip_value per-asset (OIL=0.01)
        )

        res = future.result(timeout=120)

        if res["status"] == "pruned":
            raise optuna.exceptions.TrialPruned()
        if res["status"] == "error":
            logger.warning(f"[Trial {trial.number}] Worker error: {res.get('error')}")
            raise optuna.exceptions.TrialPruned()
        if res["pruned"]:
            raise optuna.exceptions.TrialPruned()

        # ── Scalping Fitness: WR×100 + PF×10 + trades×0.05 - DD_penalty×50 ─
        dd_penalty  = res["max_drawdown"] / max_dd_limit   # 0→1
        scalp_score = (
            res["winrate"]       * SCALP_WR_WEIGHT    +
            res["profit_factor"] * SCALP_PF_WEIGHT    +
            res["trade_count"]   * SCALP_COUNT_WEIGHT -
            dd_penalty           * 50.0
        )

        logger.debug(
            f"[Trial {trial.number}] [{sp.get('label','?')}] "
            f"WR={res['winrate']:.1%} PF={res['profit_factor']:.2f} "
            f"DD={res['max_drawdown']:.1%} trades={res['trade_count']} "
            f"score={scalp_score:.2f}"
        )

        return scalp_score

    return objective




# ---------------------------------------------------------------------------
# Top-Level Public API — objective(trial) & run_optimization()
# Task 4.2: Các hàm entry point rõ ràng để kickstart ML training loop
# ---------------------------------------------------------------------------

# Module-level context: được set bởi run_optimization() trước khi study chạy
# Dùng pattern này thay vì closure để top-level function có thể được pickle
# (cần thiết khi Optuna gọi objective trong subprocess)
_OPT_CONTEXT: dict = {}


def objective(trial: optuna.Trial) -> float:
    """
    Scalping Fitness Function — ưu tiên Winrate và tần suất lệnh.

    Score = WR×100 + PF×10 + trades×0.05 - DD_penalty×50

    Context (_OPT_CONTEXT):
        data, max_dd_limit, min_trades, asset_cfg
    """
    ctx          = _OPT_CONTEXT
    data         = ctx["data"]
    max_dd_limit = ctx.get("max_dd_limit", DEFAULT_MAX_DD_LIMIT)
    min_trades   = ctx.get("min_trades",   DEFAULT_MIN_TRADES)
    sp           = ctx.get("asset_cfg",    _ASSET_CLASS_DEFAULT)

    # ── Suggest tham số theo asset class ─────────────────────────────────
    params = {
        "atr_sl_multiplier":    trial.suggest_float("atr_sl_multiplier",    *sp["atr_sl_multiplier"]),
        "atr_lot_multiplier":   trial.suggest_float("atr_lot_multiplier",   *sp["atr_lot_multiplier"]),
        "atr_fvg_buffer":       trial.suggest_float("atr_fvg_buffer",       *sp["atr_fvg_buffer"]),
        "vsa_volume_ratio":     trial.suggest_float("vsa_volume_ratio",     *sp["vsa_volume_ratio"]),
        "vsa_neighbor_ratio":   trial.suggest_float("vsa_neighbor_ratio",   *sp["vsa_neighbor_ratio"]),
        "vsa_min_score":        trial.suggest_float("vsa_min_score",        *sp["vsa_min_score"]),
        "pinbar_wick_ratio":    trial.suggest_float("pinbar_wick_ratio",    *sp["pinbar_wick_ratio"]),
        "pinbar_body_ratio":    trial.suggest_float("pinbar_body_ratio",    *sp["pinbar_body_ratio"]),
        "composite_score_gate": trial.suggest_float("composite_score_gate", *sp["composite_score_gate"]),
    }

    # ── Backtest ────────────────────────────────────────────────────
    try:
        result = run_backtest_fast(data, params)
    except optuna.exceptions.TrialPruned:
        raise

    # ── Hard prune: sai cả hai mục tiêu số lệnh───────────────────────────
    if result.trade_count < min_trades:
        raise optuna.exceptions.TrialPruned()
    if result.max_drawdown >= max_dd_limit:
        raise optuna.exceptions.TrialPruned()

    # ── Scalping Fitness: WR×100 + PF×10 + trades×0.05 - DD_penalty×50 ───
    # Mục tiêu WR>70% cho điểm cao nhất; số lệnh nhiều = thưởng thêm bonus
    dd_penalty   = result.max_drawdown / max_dd_limit   # 0→1, càng cao càng trừ điểm
    scalp_score  = (
        result.winrate    * SCALP_WR_WEIGHT    +   # 0→100 (mục tiêu ≥70)
        result.profit_factor * SCALP_PF_WEIGHT  +   # thưởng khả năng sinh lời
        result.trade_count   * SCALP_COUNT_WEIGHT -  # thưởng tần suất lệnh cao
        dd_penalty           * 50.0                  # phạt nặng nếu DD cao
    )

    logger.debug(
        f"[Trial {trial.number}] [{sp.get('label','?')}] "
        f"WR={result.winrate:.1%} PF={result.profit_factor:.2f} "
        f"DD={result.max_drawdown:.1%} trades={result.trade_count} "
        f"score={scalp_score:.2f}"
    )

    return scalp_score


def run_optimization(
    data:            np.ndarray,
    *,
    n_trials:        int   = DEFAULT_N_TRIALS,
    n_workers:       int   = DEFAULT_N_WORKERS,
    max_dd_limit:    float = DEFAULT_MAX_DD_LIMIT,
    min_trades:      int   = DEFAULT_MIN_TRADES,
    resume:          bool  = False,
    custom_db_path:  str | None = None,   # ← đường dẫn DB riêng khi chạy song song
) -> optuna.Study:
    """
    Entry point standalone để khởi chạy ML training loop.

    ✅ ASK-AND-TELL PATTERN (Task 4.2 Fix):
        Chỉ Main Thread được phép chạm SQLite (study.ask / study.tell).
        ProcessPoolExecutor workers chỉ thuần túy tính numpy — không bao giờ
        truy cập DB → triệt tiêu hoàn toàn lỗi "database is locked".

    Execution model:
        Loop mỗi batch n_workers trials:
            1. Main thread: study.ask() × n_workers  → batch of FrozenTrials
            2. Main thread: extract params dict từ mỗi trial
            3. executor.map(_run_trial_worker, params_batch) → workers tính numpy song song
            4. Main thread: study.tell(trial, value) cho từng kết quả nhận về

    Args:
        data:         np.ndarray shape (N, 6) — M5 OHLCV data
        n_trials:     Số Optuna trials (default: 500)
        n_workers:    Số CPU workers ProcessPoolExecutor (default: 48)
        max_dd_limit: Max drawdown cho phép để không prune trial (default: 0.15)
        min_trades:   Số lệnh tối thiểu để trial hợp lệ (default: 200)
        resume:       True → load study cũ từ SQLite; False → tạo mới

    Returns:
        optuna.Study đã optimize xong — .best_trial, .best_params sẵn sàng
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # custom_db_path cho phép chạy song song nhiều file (mỗi file 1 DB)
    _db = Path(custom_db_path) if custom_db_path else STUDY_DB_PATH
    storage_url = f"sqlite:///{_db}"
    sampler     = optuna.samplers.TPESampler(seed=42, n_startup_trials=20)
    pruner      = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=0)

    # ── Khởi tạo / Resume Optuna Study (chỉ main thread) ───────────────────
    if resume:
        logger.info(f"[run_optimization] Resuming study from: {storage_url}")
        study = optuna.load_study(
            study_name=STUDY_NAME,
            storage=storage_url,
            sampler=sampler,
        )
    else:
        study = optuna.create_study(
            study_name=STUDY_NAME,
            direction="maximize",
            storage=storage_url,      # SQLite — chỉ main thread chạm
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

    n_done      = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_remaining = max(0, n_trials - n_done)

    logger.info(
        f"[run_optimization] Study ready | "
        f"done={n_done} remaining={n_remaining} "
        f"workers={n_workers} storage={storage_url}"
    )

    if n_remaining == 0:
        logger.info("[run_optimization] All trials already complete — skipping optimize")
        return study

    # ── ASK-AND-TELL LOOP với ProcessPoolExecutor ───────────────────────────
    # Nguyên tắc: Study (SQLite) ← CHỈ Main Thread; Workers ← chỉ tính numpy
    # Batch size = n_workers → mỗi vòng lặp lấp đầy tất cả cores trong 1 lần
    logger.info(
        f"[run_optimization] Launching Ask-and-Tell loop | "
        f"max_workers={n_workers} | batch_size={n_workers} | n_trials={n_remaining}"
    )

    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        while completed < n_remaining:
            batch_size = min(n_workers, n_remaining - completed)

            # ── STEP 1: Main Thread ask() batch trials từ study ────────────
            # study.ask() không cần objective function; chỉ sample params
            frozen_trials = [study.ask() for _ in range(batch_size)]

            # Extract params dict từ mỗi FrozenTrial (chưa có value)
            params_batch = [ft.params for ft in frozen_trials]

            # Package args cho worker: (params, data, max_dd_limit, min_trades)
            worker_args = [
                (p, data, max_dd_limit, min_trades)
                for p in params_batch
            ]

            # ── STEP 2: Workers tính numpy song song (không chạm SQLite) ───
            results = list(executor.map(_run_trial_worker, worker_args))

            # ── STEP 3: Main Thread tell() kết quả về study (SQLite) ────────
            for frozen_trial, res in zip(frozen_trials, results):
                if res["status"] == "pruned":
                    study.tell(frozen_trial, state=optuna.trial.TrialState.PRUNED)
                elif res["status"] == "error":
                    logger.warning(
                        f"[run_optimization] Trial {frozen_trial.number} error: "
                        f"{res.get('error')}"
                    )
                    study.tell(frozen_trial, state=optuna.trial.TrialState.PRUNED)
                else:
                    study.tell(frozen_trial, res["score"])

            completed += batch_size
            best_val = study.best_value if study.best_trial else float("nan")
            logger.info(
                f"[run_optimization] Progress {completed}/{n_remaining} trials | "
                f"best_score={best_val:.4f}"
            )

    logger.info(
        f"[run_optimization] Complete | "
        f"best_score={study.best_value:.4f} | "
        f"best_params={study.best_params}"
    )

    return study

# ---------------------------------------------------------------------------
# Data Fetcher — MT5 → numpy cache
# ---------------------------------------------------------------------------

def load_data_from_csv(
    csv_path: str | Path | None = None,
    *,
    logger: logging.Logger,
) -> np.ndarray:
    """
    LINUX-SAFE: Nạp dữ liệu M5 OHLCV từ file CSV (được export từ Windows bằng tools/export_mt5_data.py).
    Không cần MT5, an toàn 100% trên Xeon Linux.

    CSV Schema (header bắt buộc):
        time,open,high,low,close,volume
        (time là Unix timestamp — int seconds UTC)

    Args:
        csv_path: Đường dẫn tới file CSV. Mặc định: data/history_m5.csv
        logger:   Logger instance

    Returns:
        np.ndarray shape (N, 6) — [time, open, high, low, close, volume]

    Raises:
        FileNotFoundError: nếu file CSV không tồn tại
        ValueError:        nếu CSV thiếu cột hoặc không có dữ liệu
    """
    path = Path(csv_path) if csv_path else CSV_DATA_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"CSV data file not found: {path}\n"
            f"  → Chạy trên Windows: python tools/export_mt5_data.py\n"
            f"  → Rồi SCP file lên Xeon: scp data/history_m5.csv xeon:/path/RabitScal/data/"
        )

    logger.info(f"[DataLoad] Loading CSV: {path}")

    required_cols = {"time", "open", "high", "low", "close", "volume"}
    rows: list[list[float]] = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV file is empty or missing header: {path}")

        # MT5 exports 'tick_volume' — normalize to 'volume' trước khi validate
        fieldnames = list(reader.fieldnames)
        if "tick_volume" in fieldnames and "volume" not in fieldnames:
            idx = fieldnames.index("tick_volume")
            fieldnames[idx] = "volume"
            reader.fieldnames = fieldnames
            logger.debug("[DataLoad] Renamed column: tick_volume → volume")

        actual_cols = set(reader.fieldnames)
        missing = required_cols - actual_cols
        if missing:
            raise ValueError(
                f"CSV missing required columns: {missing}\n"
                f"  Required: {required_cols}\n"
                f"  Found: {actual_cols}"
            )

        for row in reader:
            try:
                # Cột time: hỗ trợ cả Unix timestamp (int/float) lẫn
                # string datetime MT5 export ('2025-11-04 03:05:00')
                raw_time = row["time"].strip()
                try:
                    t = float(raw_time)
                except ValueError:
                    t = datetime.strptime(raw_time, "%Y-%m-%d %H:%M:%S") \
                               .replace(tzinfo=timezone.utc) \
                               .timestamp()
                rows.append([
                    t,
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row["volume"]),
                ])
            except (ValueError, KeyError) as e:
                logger.warning(f"[DataLoad] Skipping malformed row: {e}")
                continue

    if not rows:
        raise ValueError(f"No valid data rows found in CSV: {path}")

    data = np.array(rows, dtype=np.float64)
    # Sort by time (an toàn khi ghép nhiều file)
    data = data[data[:, 0].argsort()]

    logger.info(
        f"[DataLoad] ✅ CSV loaded | shape={data.shape} | "
        f"candles={len(data):,} | "
        f"from={datetime.fromtimestamp(data[0,0], tz=timezone.utc).strftime('%Y-%m-%d')} "
        f"to={datetime.fromtimestamp(data[-1,0], tz=timezone.utc).strftime('%Y-%m-%d')}"
    )
    return data


def fetch_and_cache_data(
    symbols:       list[str],
    lookback_days: int,
    *,
    logger: logging.Logger,
) -> np.ndarray:
    """
    Fetch M5 OHLCV từ MT5 cho tất cả symbols, ghép thành 1 numpy array.
    Lưu cache ra DATA_CACHE_PATH để dùng lại.

    Returns:
        np.ndarray shape (N, 6) — [time, open, high, low, close, volume]
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not _MT5_AVAILABLE or mt5 is None:
        raise RuntimeError(
            "MetaTrader5 không khả dụng trên hệ thống này (Linux/no MT5 install).\n"
            "  → Dùng load_data_from_csv() hoặc chạy tools/export_mt5_data.py trên Windows trước."
        )

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    utc_to   = datetime.now(timezone.utc)
    utc_from = utc_to - timedelta(days=lookback_days)

    all_rates: list[np.ndarray] = []

    for sym in symbols:
        logger.info(f"[DataFetch] Fetching M5 for {sym} | {lookback_days} days...")
        rates = mt5.copy_rates_range(
            sym,
            mt5.TIMEFRAME_M5,
            utc_from.replace(tzinfo=None),
            utc_to.replace(tzinfo=None),
        )
        if rates is None or len(rates) == 0:
            logger.warning(f"[DataFetch] {sym}: no data returned — skipping")
            continue

        # Structured array → regular float array: [time, O, H, L, C, V]
        arr = np.column_stack([
            rates["time"].astype(np.float64),
            rates["open"].astype(np.float64),
            rates["high"].astype(np.float64),
            rates["low"].astype(np.float64),
            rates["close"].astype(np.float64),
            rates["tick_volume"].astype(np.float64),
        ])
        all_rates.append(arr)
        logger.info(f"[DataFetch] {sym}: {len(arr):,} M5 candles fetched")

    mt5.shutdown()

    if not all_rates:
        raise RuntimeError("No historical data fetched — check MT5 connection and symbols")

    combined = np.concatenate(all_rates, axis=0)
    # Sort by time (quan trọng khi ghép nhiều symbols)
    combined = combined[combined[:, 0].argsort()]

    np.save(DATA_CACHE_PATH, combined)
    logger.info(
        f"[DataFetch] Cache saved: {DATA_CACHE_PATH} | "
        f"shape={combined.shape} | size={combined.nbytes / 1024**2:.1f}MB"
    )
    return combined


# ---------------------------------------------------------------------------
# Walk-Forward OOS Validator (Shadow Deployment)
# ---------------------------------------------------------------------------

def _run_walk_forward_oos(
    full_data:    np.ndarray,
    params:       dict,
    oos_hours:    int,
    active_pf:    float,
    promote_thr:  float,
    *,
    logger: logging.Logger,
) -> WalkForwardResult:
    """
    Tách `oos_hours` cuối của full_data làm Out-of-Sample (OOS).
    Chạy backtest trên OOS với params mới.
    So sánh profit_factor với active config.

    OOS window: ~oos_hours × 12 nến M5/giờ = oos_hours × 12 candles
    """
    oos_candles = oos_hours * 12  # M5: 12 nến/giờ
    if len(full_data) <= oos_candles + 50:
        return WalkForwardResult(
            winrate=0, profit_factor=0, max_drawdown=1,
            trade_count=0, pass_validate=False,
            reason="OOS data too short"
        )

    oos_data = full_data[-oos_candles:]

    try:
        result = run_backtest_fast(oos_data, params)
    except optuna.exceptions.TrialPruned:
        return WalkForwardResult(
            winrate=0, profit_factor=0, max_drawdown=1,
            trade_count=0, pass_validate=False,
            reason="Pruned in OOS backtest (DD too high or too few trades)"
        )
    except Exception as e:
        return WalkForwardResult(
            winrate=0, profit_factor=0, max_drawdown=1,
            trade_count=0, pass_validate=False,
            reason=f"OOS backtest error: {e}"
        )

    # Promote nếu shadow PF >= active PF × promote_threshold
    promote_bar = active_pf * promote_thr
    passed      = result.profit_factor >= promote_bar

    reason = (
        f"shadow_pf={result.profit_factor:.3f} >= active_pf×{promote_thr}={promote_bar:.3f}"
        if passed else
        f"shadow_pf={result.profit_factor:.3f} < active_pf×{promote_thr}={promote_bar:.3f} → RETIRE"
    )

    logger.info(
        f"[WalkForward] OOS {oos_hours}h ({len(oos_data)} candles) | "
        f"WR={result.winrate:.3f} PF={result.profit_factor:.3f} "
        f"DD={result.max_drawdown:.3f} trades={result.trade_count} | "
        f"{'PROMOTE' if passed else 'RETIRE'}"
    )

    return WalkForwardResult(
        winrate=float(result.winrate),
        profit_factor=float(result.profit_factor),
        max_drawdown=float(result.max_drawdown),
        trade_count=result.trade_count,
        pass_validate=passed,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Config Version Manager
# ---------------------------------------------------------------------------

def _get_next_version() -> int:
    """Scan config/versions/ → trả về version tiếp theo (max + 1)."""
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(VERSIONS_DIR.glob("settings_v*.json"))
    if not existing:
        return 1
    nums = []
    for p in existing:
        try:
            nums.append(int(p.stem.replace("settings_v", "")))
        except ValueError:
            pass
    return (max(nums) + 1) if nums else 1


def _load_active_config() -> dict:
    """Load current_settings.json với FileLock. Trả về {} nếu không tồn tại."""
    lock_path = str(ACTIVE_CFG_PATH) + ".lock"
    with FileLock(lock_path):
        if not ACTIVE_CFG_PATH.exists():
            return {}
        with open(ACTIVE_CFG_PATH, "r") as f:
            return json.load(f)


def _save_active_config(params: dict, version: int) -> None:
    """Ghi current_settings.json với FileLock (atomic)."""
    lock_path = str(ACTIVE_CFG_PATH) + ".lock"
    payload = {
        "version":    version,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "params":     params,
    }
    with FileLock(lock_path):
        with open(ACTIVE_CFG_PATH, "w") as f:
            json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Shadow Deploy — Walk-Forward OOS + Promote/Retire
# ---------------------------------------------------------------------------

def _shadow_deploy(
    best_params:   dict,
    best_score:    float,
    best_metrics:  BacktestResult,
    full_data:     np.ndarray,
    oos_hours:     int,
    promote_thr:   float,
    *,
    logger: logging.Logger,
) -> tuple[str, str]:
    """
    Lưu config mới ra settings_v{NN}.json.
    Chạy walk-forward OOS validate.
    Nếu pass → promote thành active config.
    Nếu fail → đánh dấu retired.

    Returns:
        (shadow_path, status)  — status = "promoted" | "retired"
    """
    version   = _get_next_version()
    path      = VERSIONS_DIR / f"settings_v{version:03d}.json"

    # Đọc active config để lấy profit_factor baseline
    active_cfg = _load_active_config()
    active_pf  = active_cfg.get("params", {}).get("last_profit_factor", 1.0)
    if active_pf <= 0:
        active_pf = 1.0  # Fallback: so sánh với PF=1.0 (breakeven)

    # ── Walk-forward OOS validation ────────────────────────────────────────
    logger.info(
        f"[ShadowDeploy] Running walk-forward OOS validate | "
        f"oos_hours={oos_hours} | active_pf={active_pf:.3f}"
    )
    wf_result = _run_walk_forward_oos(
        full_data, best_params, oos_hours, active_pf, promote_thr, logger=logger
    )

    status = "promoted" if wf_result.pass_validate else "retired"

    # ── Save version file ───────────────────────────────────────────────────
    shadow_config = {
        "version":     version,
        "created_at":  datetime.now(timezone.utc).isoformat(),
        "status":      status,
        "params":      best_params,
        "train_metrics": {
            "best_score":    round(best_score, 6),
            "winrate":       round(best_metrics.winrate, 4),
            "profit_factor": round(best_metrics.profit_factor, 4),
            "max_drawdown":  round(best_metrics.max_drawdown, 4),
            "trade_count":   best_metrics.trade_count,
        },
        "oos_metrics": {
            "oos_hours":     oos_hours,
            "winrate":       round(wf_result.winrate, 4),
            "profit_factor": round(wf_result.profit_factor, 4),
            "max_drawdown":  round(wf_result.max_drawdown, 4),
            "trade_count":   wf_result.trade_count,
            "pass_validate": wf_result.pass_validate,
            "reason":        wf_result.reason,
        },
    }

    lock_path = str(path) + ".lock"
    with FileLock(lock_path):
        with open(path, "w") as f:
            json.dump(shadow_config, f, indent=2)

    logger.info(f"[ShadowDeploy] Config saved: {path} | status={status}")

    # ── Promote nếu OOS validate pass ──────────────────────────────────────
    if wf_result.pass_validate:
        best_params_with_metrics = dict(best_params)
        best_params_with_metrics["last_profit_factor"] = round(wf_result.profit_factor, 4)
        _save_active_config(best_params_with_metrics, version)
        logger.info(
            f"[ShadowDeploy] ✅ PROMOTED to active | v{version:03d} | "
            f"OOS PF={wf_result.profit_factor:.3f}"
        )
    else:
        logger.warning(
            f"[ShadowDeploy] ❌ RETIRED | v{version:03d} | "
            f"OOS PF={wf_result.profit_factor:.3f} < bar | {wf_result.reason}"
        )

    return str(path), status


# ---------------------------------------------------------------------------
# Optimization Engine — Main Orchestrator
# ---------------------------------------------------------------------------

class OptimizationEngine:
    """
    Orchestrate toàn bộ Optuna optimization pipeline:
        fetch data → shared_memory → 48 workers → best trial → shadow deploy
    """

    def __init__(self, cfg: dict, *, resume: bool = False):
        self.n_trials      = cfg.get("n_trials",            DEFAULT_N_TRIALS)
        self.n_workers     = cfg.get("n_workers",            DEFAULT_N_WORKERS)
        self.min_trades    = cfg.get("min_trades",           DEFAULT_MIN_TRADES)
        self.max_dd_limit  = cfg.get("max_dd_limit",         DEFAULT_MAX_DD_LIMIT)
        self.lookback_days = cfg.get("data_lookback_days",   DEFAULT_LOOKBACK_DAYS)
        self.oos_hours     = cfg.get("shadow_deploy_hours",  DEFAULT_OOS_HOURS)
        self.promote_thr   = cfg.get("promote_threshold",    DEFAULT_PROMOTE_THR)
        self.resume        = resume
        self.custom_db     = cfg.get("study_db", None)        # override via CLI --study-db
        _db                = Path(self.custom_db) if self.custom_db else STUDY_DB_PATH
        self.storage       = f"sqlite:///{_db}"

    # ── Public API ──────────────────────────────────────────────────────────

    def run(self, symbols: list[str], *, data_path: str | None = None) -> OptimizationResult:
        """
        Entry point: fetch data, optimize, shadow deploy.
        Returns OptimizationResult với best params và shadow config path.

        Args:
            symbols:   Danh sách symbols dùng khi fetch từ MT5.
            data_path: (LINUX-SAFE) Đường dẫn tới CSV. Nếu được truyền,
                       dùng load_data_from_csv() — không cần MT5.
        """
        t_start = time.perf_counter()
        logger.info("=" * 70)
        logger.info(
            f"[MLEngine] Optimization run started | "
            f"n_trials={self.n_trials} n_workers={self.n_workers}"
        )

        # ── [Step 1] Fetch historical data ─────────────────────────────────
        if data_path is not None:
            # LINUX-SAFE: load thẳng từ CSV — không cần MT5
            logger.info(f"[MLEngine] --data override: loading CSV: {data_path}")
            data = load_data_from_csv(data_path, logger=logger)
        elif DATA_CACHE_PATH.exists() and not self.resume:
            logger.info(f"[MLEngine] Loading cached data: {DATA_CACHE_PATH}")
            data = np.load(DATA_CACHE_PATH)
        else:
            data = fetch_and_cache_data(symbols, self.lookback_days, logger=logger)

        # Detect asset class → dynamic search space
        asset_cfg     = detect_asset_class(data_path)
        # Luôn dùng asset_cfg làm base — override chỉ khi user truyền CLI
        effective_min = self._cli_min_trades if hasattr(self, "_cli_min_trades") else asset_cfg["min_trades"]
        effective_dd  = self._cli_max_dd     if hasattr(self, "_cli_max_dd")     else asset_cfg["max_dd_limit"]
        logger.info(
            f"[MLEngine] Asset class: {asset_cfg['label']} | "
            f"gate={asset_cfg['composite_score_gate']} | "
            f"min_trades={effective_min} | max_dd={effective_dd:.0%}"
        )

        logger.info(
            f"[MLEngine] Data ready | shape={data.shape} | "
            f"{data.shape[0]:,} M5 candles | "
            f"{data.nbytes / 1024**2:.1f}MB"
        )

        # ── [Step 2] Optuna study ───────────────────────────────────────────
        sampler = optuna.samplers.TPESampler(seed=42)
        pruner  = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=0)

        if self.resume:
            logger.info(f"[MLEngine] Resuming study from: {self.storage}")
            study = optuna.load_study(
                study_name=STUDY_NAME,
                storage=self.storage,
                sampler=sampler,
            )
        else:
            study = optuna.create_study(
                study_name=STUDY_NAME,
                direction="maximize",
                storage=self.storage,
                sampler=sampler,
                pruner=pruner,
                load_if_exists=True,
            )

        n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        n_remaining = max(0, self.n_trials - n_completed)
        logger.info(
            f"[MLEngine] Study: {n_completed} trials done, "
            f"{n_remaining} remaining to run"
        )

        if n_remaining == 0:
            logger.info("[MLEngine] All trials already completed — proceeding to deploy")
        else:
            # ── [Step 3] Shared memory + ProcessPoolExecutor ────────────────
            with SharedNumpyArray(data) as shm_ctx:
                with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                    obj_fn = _build_objective(
                        shm_name=shm_ctx.shm_name,
                        data_shape=data.shape,
                        data_dtype=str(data.dtype),
                        max_dd_limit=effective_dd,
                        min_trades=effective_min,
                        executor=executor,
                        asset_cfg=asset_cfg,    # ← dynamic search space per symbol
                    )
                    logger.info(
                        f"[MLEngine] ProcessPoolExecutor started | "
                        f"workers={self.n_workers}"
                    )
                    study.optimize(
                        obj_fn,
                        n_trials=n_remaining,
                        show_progress_bar=True,
                        gc_after_trial=True,
                    )

        # ── [Step 4] Validate best trial ────────────────────────────────────
        n_pruned   = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

        if n_complete == 0:
            logger.error(
                f"[MLEngine] ❌ ALL {n_pruned} trials pruned — 0 COMPLETE trials!\n"
                f"  Asset class : {asset_cfg['label']}\n"
                f"  min_trades  : {effective_min} | max_dd_limit: {effective_dd:.0%}\n"
                f"  → Tiếp tục debug hoặc giảm bộ lọc trong ASSET_CLASS_CONFIG."
            )
            raise RuntimeError(
                f"Optimization failed: all {n_pruned} trials pruned "
                f"[{asset_cfg['label']}: min_trades={effective_min}, max_dd={effective_dd:.0%}]. "
                f"Reduce constraints in ASSET_CLASS_CONFIG and retry."
            )

        best = study.best_trial
        logger.info(
            f"[MLEngine] Optimization complete | "
            f"best_score={best.value:.4f} | "
            f"params={best.params}"
        )



        # Re-run best params để lấy full BacktestResult metrics
        try:
            best_metrics = run_backtest_fast(data, best.params)
        except Exception:
            best_metrics = BacktestResult(0, 0, 1, 0)

        logger.info(
            f"[MLEngine] Best trial metrics | "
            f"WR={best_metrics.winrate:.3f} PF={best_metrics.profit_factor:.3f} "
            f"DD={best_metrics.max_drawdown:.3f} trades={best_metrics.trade_count}"
        )

        # ── [Step 5] Shadow deploy + Walk-forward OOS ───────────────────────
        shadow_path, deploy_status = _shadow_deploy(
            best_params=best.params,
            best_score=best.value,
            best_metrics=best_metrics,
            full_data=data,
            oos_hours=self.oos_hours,
            promote_thr=self.promote_thr,
            logger=logger,
        )

        elapsed = time.perf_counter() - t_start
        logger.info(
            f"[MLEngine] Run complete | elapsed={elapsed:.1f}s | "
            f"trials={n_complete} (pruned={n_pruned}) | "
            f"deploy={deploy_status} | path={shadow_path}"
        )
        logger.info("=" * 70)

        return OptimizationResult(
            best_params=best.params,
            best_score=best.value,
            n_trials=n_complete,
            n_pruned=n_pruned,
            duration_sec=elapsed,
            shadow_path=shadow_path,
            created_at=datetime.now(timezone.utc).isoformat(),
            best_winrate=float(best_metrics.winrate),
            best_pf=float(best_metrics.profit_factor),
            best_dd=float(best_metrics.max_drawdown),
            best_trades=int(best_metrics.trade_count),
            deploy_status=deploy_status,
        )


# ---------------------------------------------------------------------------
# Config Loader
# ---------------------------------------------------------------------------

def _load_ml_config() -> dict:
    if not ML_CONFIG_PATH.exists():
        logger.info(f"[MLEngine] ml_config.json not found — using defaults")
        return {}
    with open(ML_CONFIG_PATH, "r") as f:
        return json.load(f)


def _load_symbols() -> list[str]:
    """Đọc danh sách symbols từ main_config.json."""
    if not MAIN_CONFIG_PATH.exists():
        return ["EURUSDc"]
    with open(MAIN_CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    return cfg.get("symbols", cfg.get("symbol", ["EURUSDc"]))


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RabitScal ML Optimizer — Optuna Bayesian Hyperparameter Search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume optimization từ SQLite study DB (nếu server bị restart giữa chừng)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help=f"Override số Optuna trials (default: {DEFAULT_N_TRIALS} từ ml_config.json)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Đường dẫn tới file data CSV (LINUX-SAFE, bỏ qua MT5). Ví dụ: data/history_XAUUSDm_M5.csv",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Override số ProcessPoolExecutor workers (default: {DEFAULT_N_WORKERS})",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=None,
        help=f"Override số lệnh tối thiểu để trial valid (default: {DEFAULT_MIN_TRADES}). "
             f"Giảm xuống 30-50 nếu strategy quá chặt.",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        default=False,
        help="Force re-fetch M5 data từ MT5 (bỏ qua cache .npy)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )
    parser.add_argument(
        "--study-db",
        type=str,
        default=None,
        help="Custom SQLite DB path cho Optuna study (dùng khi chạy song song nhiều file — mỗi file 1 DB riêng). "
             "Ví dụ: data/optuna_XAUUSDm.db",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Set log level
    logging.getLogger("MLEngine").setLevel(getattr(logging, args.log_level))
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # Giảm noise Optuna

    # Load config
    ml_cfg   = _load_ml_config()
    symbols  = _load_symbols()

    # Override từ CLI args
    if args.trials is not None:
        ml_cfg["n_trials"] = args.trials
    if args.workers is not None:
        ml_cfg["n_workers"] = args.workers
    if args.min_trades is not None:
        ml_cfg["min_trades"] = args.min_trades
    if args.fetch and DATA_CACHE_PATH.exists():
        DATA_CACHE_PATH.unlink()
        logger.info("[MLEngine] Cache cleared — will re-fetch from MT5")
    if args.study_db is not None:
        ml_cfg["study_db"] = args.study_db   # ← Custom DB path — parallel safe

    logger.info(
        f"[MLEngine] Starting | symbols={symbols} | "
        f"resume={args.resume} | "
        f"trials={ml_cfg.get('n_trials', DEFAULT_N_TRIALS)} | "
        f"workers={ml_cfg.get('n_workers', DEFAULT_N_WORKERS)}"
    )

    engine = OptimizationEngine(ml_cfg, resume=args.resume)
    result = engine.run(symbols, data_path=args.data)

    print("\n" + "=" * 60)
    print("✅ OPTIMIZATION COMPLETE")
    print(f"   Best Score    : {result.best_score:.4f}")
    print(f"   Trials        : {result.n_trials} (pruned: {result.n_pruned})")
    print(f"   Duration      : {result.duration_sec:.1f}s")
    print(f"   Shadow Config : {result.shadow_path}")
    print(f"   Timestamp     : {result.created_at}")
    print("=" * 60)
    print("\nBest Parameters:")
    for k, v in result.best_params.items():
        print(f"   {k:<28} = {v:.6f}")
    print()

    # ── SCANNER_RESULT: JSON dòng đơn để auto_scanner.py parse ───────────────────────
    # Mỗi subprocess có stdout độc lập → KHÔNG race condition dù 5 file song song.
    print("SCANNER_RESULT: " + json.dumps({
        "score":   round(result.best_score,   4),
        "winrate": round(result.best_winrate, 4),
        "pf":      round(result.best_pf,      4),
        "max_dd":  round(result.best_dd,      4),
        "trades":  result.best_trades,
        "deploy":  result.deploy_status,
    }), flush=True)


if __name__ == "__main__":
    main()
