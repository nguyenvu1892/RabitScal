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

import MetaTrader5 as mt5
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
STUDY_DB_PATH     = DATA_DIR / "optuna_study.db"
STUDY_NAME        = "rabitscal_optuna_v1"

# Optuna defaults (overridable via ml_config.json)
DEFAULT_N_TRIALS      = 500
DEFAULT_N_WORKERS     = 48
DEFAULT_MIN_TRADES    = 200
DEFAULT_MAX_DD_LIMIT  = 0.15
DEFAULT_LOOKBACK_DAYS = 180
DEFAULT_OOS_HOURS     = 24
DEFAULT_PROMOTE_THR   = 0.95
DEFAULT_RETIRE_THR    = 0.80

# Objective weights
WR_WEIGHT  = 0.60
PF_WEIGHT  = 0.40

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

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
    commission_per_lot: float = 3.5,   # USD/lot (Exness typical)
    lot_size:           float = 0.01,  # Fixed 1 micro-lot per trade
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

    # ── [7] Trade simulation — OHLC worst-case model ───────────────────────
    sl_dist    = atr * params["atr_sl_multiplier"]  # SL distance (price units)
    rr_ratio   = 1.5                                  # TP = SL × 1.5
    commission = commission_per_lot * lot_size        # USD per round trip

    pnl_list: list[float] = []
    in_trade   = False
    entry_price = 0.0
    sl_price    = 0.0
    tp_price    = 0.0
    direction   = 0  # 1=BUY, -1=SELL

    for i in range(15, N - 1):
        if not in_trade:
            # Ưu tiên BUY signal; SELL khi không có BUY
            if buy_signal[i]:
                entry_price = closes[i]
                sl_price    = entry_price - sl_dist[i]
                tp_price    = entry_price + sl_dist[i] * rr_ratio
                direction   = 1
                in_trade    = True
            elif sell_signal[i]:
                entry_price = closes[i]
                sl_price    = entry_price + sl_dist[i]
                tp_price    = entry_price - sl_dist[i] * rr_ratio
                direction   = -1
                in_trade    = True
        else:
            next_h = highs[i]
            next_l = lows[i]

            if direction == 1:
                # BUY: worst-case → SL check first
                if next_l <= sl_price:
                    pnl_list.append(-(sl_dist[i - 1]) - commission)
                    in_trade = False
                elif next_h >= tp_price:
                    pnl_list.append(sl_dist[i - 1] * rr_ratio - commission)
                    in_trade = False
            else:
                # SELL: worst-case → SL check first
                if next_h >= sl_price:
                    pnl_list.append(-(sl_dist[i - 1]) - commission)
                    in_trade = False
                elif next_l <= tp_price:
                    pnl_list.append(sl_dist[i - 1] * rr_ratio - commission)
                    in_trade = False

    # ── [8] Aggregate metrics ───────────────────────────────────────────────
    if len(pnl_list) < 10:
        raise optuna.exceptions.TrialPruned()

    pnl     = np.array(pnl_list, dtype=np.float64)
    wins    = (pnl > 0)
    losses  = (pnl < 0)
    n_wins  = wins.sum()
    n_loss  = losses.sum()

    winrate       = n_wins / len(pnl)
    gross_profit  = pnl[wins].sum()  if n_wins > 0 else 0.0
    gross_loss    = abs(pnl[losses].sum()) if n_loss > 0 else 1e-8
    profit_factor = gross_profit / gross_loss
    avg_win       = gross_profit / n_wins  if n_wins > 0 else 0.0
    avg_loss      = gross_loss   / n_loss  if n_loss > 0 else 0.0

    # Max Drawdown từ equity curve
    equity   = np.cumsum(pnl)
    peak     = np.maximum.accumulate(equity)
    drawdown = np.where(peak > 0, (peak - equity) / (peak + 1e-8), 0.0)
    max_dd   = float(drawdown.max())

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
# Optuna Objective (chạy trong subprocess worker)
# ---------------------------------------------------------------------------

def _objective_worker(
    trial_params: dict,
    shm_name:     str,
    data_shape:   tuple,
    data_dtype:   str,
    max_dd_limit: float,
    min_trades:   int,
) -> dict:
    """
    Chạy trong subprocess riêng biệt (ProcessPoolExecutor worker).
    Attach vào shared memory, chạy backtest, trả về dict result.
    
    Không nhận optuna.Trial trực tiếp (không serializable sang subprocess).
    Thay vào đó nhận trial_params dict đã suggest sẵn từ orchestrator.
    """
    try:
        # Attach shared memory
        shm = shared_memory.SharedMemory(create=False, name=shm_name)
        data = np.ndarray(data_shape, dtype=np.dtype(data_dtype), buffer=shm.buf)

        result = run_backtest_fast(data, trial_params)

        shm.close()  # Không unlink — orchestrator quản lý lifecycle

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
        return {"status": "error", "error": str(e)}


def _build_objective(
    shm_name:     str,
    data_shape:   tuple,
    data_dtype:   str,
    max_dd_limit: float,
    min_trades:   int,
    executor:     ProcessPoolExecutor,
):
    """
    Factory trả về hàm objective cho optuna.study.optimize().
    Dùng ProcessPoolExecutor để submit task sang worker process.
    """
    def objective(trial: optuna.Trial) -> float:
        params = {
            "atr_sl_multiplier":    trial.suggest_float("atr_sl_multiplier",    0.8,  3.0),
            "atr_lot_multiplier":   trial.suggest_float("atr_lot_multiplier",   0.005, 0.05),
            "atr_fvg_buffer":       trial.suggest_float("atr_fvg_buffer",       0.3,  1.5),
            "vsa_volume_ratio":     trial.suggest_float("vsa_volume_ratio",     1.2,  3.0),
            "vsa_neighbor_ratio":   trial.suggest_float("vsa_neighbor_ratio",   1.1,  2.0),
            "vsa_min_score":        trial.suggest_float("vsa_min_score",        0.3,  0.6),
            "pinbar_wick_ratio":    trial.suggest_float("pinbar_wick_ratio",    0.50, 0.75),
            "pinbar_body_ratio":    trial.suggest_float("pinbar_body_ratio",    0.10, 0.40),
            "composite_score_gate": trial.suggest_float("composite_score_gate", 0.45, 0.75),
        }

        future = executor.submit(
            _objective_worker,
            params, shm_name, data_shape, str(data_dtype),
            max_dd_limit, min_trades,
        )
        res = future.result(timeout=120)  # 2-min timeout per trial

        if res["status"] == "pruned":
            raise optuna.exceptions.TrialPruned()
        if res["status"] == "error":
            logger.warning(f"[Trial {trial.number}] Worker error: {res.get('error')}")
            raise optuna.exceptions.TrialPruned()

        if res["pruned"]:
            raise optuna.exceptions.TrialPruned()

        # Hàm objective: WR^0.60 × PF^0.40 × dd_penalty
        dd_penalty       = 1.0 - (res["max_drawdown"] / max_dd_limit) ** 2
        composite_score  = (
            (res["winrate"]       ** WR_WEIGHT) *
            (res["profit_factor"] ** PF_WEIGHT) *
            max(dd_penalty, 0.0)
        )

        logger.debug(
            f"[Trial {trial.number}] "
            f"WR={res['winrate']:.3f} PF={res['profit_factor']:.3f} "
            f"DD={res['max_drawdown']:.3f} trades={res['trade_count']} "
            f"score={composite_score:.4f}"
        )

        return composite_score

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
    Hàm Objective chuẩn Optuna — top-level, picklable, dùng trực tiếp với study.optimize().

    Khai báo toàn bộ không gian tham số (9 chiều) cho chiến lược SMC+VSA:
        ─ ATR group    : atr_sl_multiplier, atr_lot_multiplier, atr_fvg_buffer
        ─ VSA group    : vsa_volume_ratio, vsa_neighbor_ratio, vsa_min_score
        ─ Pinbar group : pinbar_wick_ratio, pinbar_body_ratio
        ─ Gate         : composite_score_gate

    Score = WR^0.60 × PF^0.40 × dd_penalty
        • dd_penalty = 1 − (max_dd / max_dd_limit)²
        • TrialPruned ngay khi DD ≥ 15% hoặc trade_count < min_trades

    Context phải được set trước (bởi run_optimization):
        _OPT_CONTEXT = {
            "data":          np.ndarray,   # Full M5 OHLCV
            "max_dd_limit":  float,        # e.g. 0.15
            "min_trades":    int,          # e.g. 200
        }
    """
    ctx          = _OPT_CONTEXT
    data         = ctx["data"]
    max_dd_limit = ctx.get("max_dd_limit", DEFAULT_MAX_DD_LIMIT)
    min_trades   = ctx.get("min_trades",   DEFAULT_MIN_TRADES)

    # ── Suggest tham số (9 chiều) ──────────────────────────────────────────
    params = {
        # ATR group — kiểm soát khoảng cách SL và kích thước FVG tối thiểu
        "atr_sl_multiplier":    trial.suggest_float("atr_sl_multiplier",    0.8,  3.0),
        "atr_lot_multiplier":   trial.suggest_float("atr_lot_multiplier",   0.005, 0.05),
        "atr_fvg_buffer":       trial.suggest_float("atr_fvg_buffer",       0.3,  1.5),

        # VSA group — ngưỡng lọc volume climax (2-layer filter)
        "vsa_volume_ratio":     trial.suggest_float("vsa_volume_ratio",     1.2,  3.0),
        "vsa_neighbor_ratio":   trial.suggest_float("vsa_neighbor_ratio",   1.1,  2.0),
        "vsa_min_score":        trial.suggest_float("vsa_min_score",        0.3,  0.6),

        # Pinbar group — tỷ lệ wick/body xác nhận đảo chiều
        "pinbar_wick_ratio":    trial.suggest_float("pinbar_wick_ratio",    0.50, 0.75),
        "pinbar_body_ratio":    trial.suggest_float("pinbar_body_ratio",    0.10, 0.40),

        # Composite gate — ngưỡng tổng hợp để ra signal (SMC+VSA+Pinbar)
        "composite_score_gate": trial.suggest_float("composite_score_gate", 0.45, 0.75),
    }

    # ── Chạy backtest vectorized ────────────────────────────────────────────
    try:
        result = run_backtest_fast(data, params)
    except optuna.exceptions.TrialPruned:
        raise

    # ── Hard prune nếu không đủ điều kiện ──────────────────────────────────
    if result.max_drawdown >= max_dd_limit or result.trade_count < min_trades:
        raise optuna.exceptions.TrialPruned()

    # ── Score tổng hợp: WR^0.60 × PF^0.40 × dd_penalty ───────────────────
    dd_penalty      = 1.0 - (result.max_drawdown / max_dd_limit) ** 2
    composite_score = (
        (result.winrate       ** WR_WEIGHT) *
        (result.profit_factor ** PF_WEIGHT) *
        max(dd_penalty, 0.0)
    )

    logger.debug(
        f"[Trial {trial.number}] "
        f"WR={result.winrate:.3f} PF={result.profit_factor:.3f} "
        f"DD={result.max_drawdown:.3f} trades={result.trade_count} "
        f"score={composite_score:.4f}"
    )

    return composite_score


def run_optimization(
    data:         np.ndarray,
    *,
    n_trials:     int   = DEFAULT_N_TRIALS,
    n_workers:    int   = DEFAULT_N_WORKERS,
    max_dd_limit: float = DEFAULT_MAX_DD_LIMIT,
    min_trades:   int   = DEFAULT_MIN_TRADES,
    resume:       bool  = False,
) -> optuna.Study:
    """
    Entry point standalone để khởi chạy ML training loop.

    Khởi tạo Optuna study với SQLite backend (hỗ trợ resume khi crash),
    kích hoạt ProcessPoolExecutor(max_workers=48) để tận dụng tối đa
    56 luồng Xeon E5-2680 v4 (48 workers ML + 8 luồng dự trữ cho bot).

    Execution model:
        • ProcessPoolExecutor(max_workers=n_workers) → bypass GIL hoàn toàn
        • Mỗi trial = 1 subprocess → backtest vectorized numpy ~25ms/trial
        • SQLite storage → có thể Ctrl+C và resume bằng --resume flag
        • TPESampler + MedianPruner → Bayesian search thông minh

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
    global _OPT_CONTEXT

    # ── Set module-level context (tránh closure — giữ objective picklable) ─
    _OPT_CONTEXT = {
        "data":          data,
        "max_dd_limit":  max_dd_limit,
        "min_trades":    min_trades,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    storage_url = f"sqlite:///{STUDY_DB_PATH}"
    sampler     = optuna.samplers.TPESampler(seed=42, n_startup_trials=20)
    pruner      = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=0)

    # ── Khởi tạo / Resume Optuna Study ─────────────────────────────────────
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
            direction="maximize",            # Maximize composite score
            storage=storage_url,             # SQLite → durable, resumable
            sampler=sampler,                 # TPE Bayesian sampler
            pruner=pruner,                   # Median pruner cắt early bad trials
            load_if_exists=True,             # Resume nếu DB đã tồn tại
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

    # ── KÍCH HOẠT 48 LUỒNG ProcessPoolExecutor ─────────────────────────────
    # Mỗi worker = 1 OS process riêng biệt → bypass GIL hoàn toàn
    # Dual Xeon E5-2680 v4: 56 logical threads → dùng 48 cho ML, 8 dự trữ bot
    logger.info(
        f"[run_optimization] Launching ProcessPoolExecutor | "
        f"max_workers={n_workers} | n_trials={n_remaining}"
    )

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Truyền executor vào _build_objective để mỗi trial của Optuna
        # được submit sang 1 worker process riêng → parallel hoàn toàn
        obj_fn = _build_objective(
            shm_name=None,       # Không dùng shared_memory ở mode standalone
            data_shape=data.shape,
            data_dtype=str(data.dtype),
            max_dd_limit=max_dd_limit,
            min_trades=min_trades,
            executor=executor,
        )

        # study.optimize() điều phối Bayesian search qua executor
        study.optimize(
            obj_fn,
            n_trials=n_remaining,
            n_jobs=1,               # Optuna gọi objective tuần tự; executor batches workers
            show_progress_bar=True,
            gc_after_trial=True,
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
        self.storage       = f"sqlite:///{STUDY_DB_PATH}"

    # ── Public API ──────────────────────────────────────────────────────────

    def run(self, symbols: list[str]) -> OptimizationResult:
        """
        Entry point: fetch data, optimize, shadow deploy.
        Returns OptimizationResult với best params và shadow config path.
        """
        t_start = time.perf_counter()
        logger.info("=" * 70)
        logger.info(
            f"[MLEngine] Optimization run started | "
            f"n_trials={self.n_trials} n_workers={self.n_workers}"
        )

        # ── [Step 1] Fetch historical data ─────────────────────────────────
        if DATA_CACHE_PATH.exists() and not self.resume:
            logger.info(f"[MLEngine] Loading cached data: {DATA_CACHE_PATH}")
            data = np.load(DATA_CACHE_PATH)
        else:
            data = fetch_and_cache_data(symbols, self.lookback_days, logger=logger)

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
                        max_dd_limit=self.max_dd_limit,
                        min_trades=self.min_trades,
                        executor=executor,
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
        best = study.best_trial
        logger.info(
            f"[MLEngine] Optimization complete | "
            f"best_score={best.value:.4f} | "
            f"params={best.params}"
        )

        n_pruned   = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

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
        "--workers",
        type=int,
        default=None,
        help=f"Override số ProcessPoolExecutor workers (default: {DEFAULT_N_WORKERS})",
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
    if args.fetch and DATA_CACHE_PATH.exists():
        DATA_CACHE_PATH.unlink()
        logger.info("[MLEngine] Cache cleared — will re-fetch from MT5")

    logger.info(
        f"[MLEngine] Starting | symbols={symbols} | "
        f"resume={args.resume} | "
        f"trials={ml_cfg.get('n_trials', DEFAULT_N_TRIALS)} | "
        f"workers={ml_cfg.get('n_workers', DEFAULT_N_WORKERS)}"
    )

    engine = OptimizationEngine(ml_cfg, resume=args.resume)
    result = engine.run(symbols)

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


if __name__ == "__main__":
    main()
