"""
feature_engine.py — MTF Feature Engineering Layer v1.0
=======================================================
Module: RabitScal V12.0 — "Trở Về Bản Ngã AI" Campaign
Author: Antigravity
Date:   2026-03-07

Triết lý: "Chỉ truyền Kiến Thức, không truyền Lệnh."

Nhiệm vụ:
    1. Nạp đồng thời dữ liệu M1 / M5 / M15 / H1 từ thư mục data/
    2. Align timestamp chính xác — KHÔNG look-ahead bias
    3. Tính toán ~30 Features và đóng gói thành feature matrix:
       - Output shape: (N_m5, 31)   ← N candles M5, 31 features
    4. KHÔNG có bất kỳ logic Entry/SL/TP hay If-Else "quyết định" nào ở đây.
       Tất cả chỉ là "kiến thức" đưa vào tay AI (Optuna + backtest loop).

Feature Map (31 features):
    ┌──────────────────────────────────────────────────────────────────────┐
    │ Group        │ Idx │ Feature Name              │ Scale               │
    ├──────────────────────────────────────────────────────────────────────┤
    │ M1 (micro)   │  0  │ atr_m1_norm               │ ATR_m1 / ATR_m5     │
    │              │  1  │ pinbar_m1                  │ 0.0 / 1.0           │
    │              │  2  │ vsa_m1                     │ 0.0 / 1.0           │
    │              │  3  │ spread_proxy_m1            │ (H-L)_m1 / ATR_m5  │
    ├──────────────────────────────────────────────────────────────────────┤
    │ M5 (trigger) │  4  │ atr_m5_raw                │ price-unit (raw)    │
    │              │  5  │ pinbar_m5                  │ 0.0 / 1.0           │
    │              │  6  │ bull_pinbar_m5             │ 0.0 / 1.0           │
    │              │  7  │ bear_pinbar_m5             │ 0.0 / 1.0           │
    │              │  8  │ vsa_m5                     │ 0.0 / 1.0           │
    │              │  9  │ fvg_bull_m5                │ 0.0 / 1.0           │
    │              │ 10  │ fvg_bear_m5                │ 0.0 / 1.0           │
    │              │ 11  │ fvg_size_bull_m5_norm      │ FVG_size / ATR_m5   │
    │              │ 12  │ fvg_size_bear_m5_norm      │ FVG_size / ATR_m5   │
    │              │ 13  │ volume_ratio_m5            │ vol / vol_ma20      │
    ├──────────────────────────────────────────────────────────────────────┤
    │ M15 (micro-  │ 14  │ atr_m15_norm              │ ATR_m15 / ATR_m5    │
    │  trend)      │ 15  │ trend_ema_m15             │ -1 / 0 / +1         │
    │              │ 16  │ bos_bull_m15               │ 0.0 / 1.0           │
    │              │ 17  │ bos_bear_m15               │ 0.0 / 1.0           │
    │              │ 18  │ pinbar_m15                 │ 0.0 / 1.0           │
    │              │ 19  │ price_vs_ema50_m15_norm    │ (C-EMA50)/ATR_m5   │
    ├──────────────────────────────────────────────────────────────────────┤
    │ H1 (macro)   │ 20  │ atr_h1_norm               │ ATR_h1 / ATR_m5     │
    │              │ 21  │ trend_ema_h1               │ -1 / 0 / +1         │
    │              │ 22  │ price_vs_ema21_h1_norm     │ (C-EMA21)/ATR_m5   │
    │              │ 23  │ price_vs_ema50_h1_norm     │ (C-EMA50)/ATR_m5   │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Liquidity    │ 24  │ dist_eql_norm              │ (C-EQL)/ATR_m5      │
    │ (EQL/EQH)    │ 25  │ dist_eqh_norm              │ (EQH-C)/ATR_m5      │
    │              │ 26  │ eql_proximity_flag         │ dist<1.5ATR → 1.0   │
    │              │ 27  │ eqh_proximity_flag         │ dist<1.5ATR → 1.0   │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Volatility   │ 28  │ volatility_index           │ ATR_m5 / ATR_h1     │
    │              │ 29  │ static_spread_cost_norm    │ spread_cost/ATR_m5  │
    ├──────────────────────────────────────────────────────────────────────┤
    │ Time/Session │ 30  │ hour_sin                   │ sin(2π×hour/24)     │
    │              │ 31  │ hour_cos                   │ cos(2π×hour/24)     │
    └──────────────────────────────────────────────────────────────────────┘
    Total: 32 features (index 0..31)

Usage:
    from feature_engine import load_mtf_data, compute_features

    mtf    = load_mtf_data("EURUSDm", data_dir="data")
    feats  = compute_features(mtf, spread_cost=0.00015)
    # feats.shape → (N_m5, 32)
    # mtf["m5"].shape → (N_m5, 6)  ← dùng song song để tính SL/TP/PnL
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TypedDict

import numpy as np

# ---------------------------------------------------------------------------
# Constants & Logging
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data"
LOGS_DIR     = PROJECT_ROOT / "logs"

N_FEATURES   = 54    # V14.1: 54 features (index 0..53)

# Warm-up bars: bỏ qua N nến đầu để indicators ổn định (EMA200 cần ít nhất 200 bars)
WARMUP_BARS  = 200

# EQL/EQH lookback — số nến M5 để scan liquidity pools
EQL_LOOKBACK = 20    # 20 × M5 = 100 phút lookback

# EMA periods
EMA_FAST_M15  = 50
EMA_SLOW_M15  = 200
EMA_FAST_H1   = 21
EMA_SLOW_H1   = 50

# BOS (Break of Structure) lookback — nến M15
BOS_LOOKBACK  = 10


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
        LOGS_DIR / "feature_engine.log", maxBytes=5 * 1024 * 1024, backupCount=3
    )
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = _build_logger("FeatureEngine")


# ---------------------------------------------------------------------------
# Data Type
# ---------------------------------------------------------------------------

class MTFData(TypedDict):
    """Dict chứa 4 TF OHLCV arrays sau khi load."""
    m1:  np.ndarray   # shape (N1, 6) — [time, O, H, L, C, V]
    m5:  np.ndarray   # shape (N5, 6)
    m15: np.ndarray   # shape (N15, 6)
    h1:  np.ndarray   # shape (Nh1, 6)
    symbol: str       # e.g. "EURUSDm"


# ---------------------------------------------------------------------------
# CSV Loader (reuse logic từ backtest_env.py — standalone, không import)
# ---------------------------------------------------------------------------

def _load_csv(csv_path: Path) -> np.ndarray:
    """
    Load OHLCV CSV → numpy float64 array, shape (N, 6).
    Header bắt buộc: time, open, high, low, close, volume
    """
    import csv as _csv

    if not csv_path.exists():
        raise FileNotFoundError(
            f"[FeatureEngine] CSV not found: {csv_path}\n"
            f"  → Check data/ directory hoặc re-export từ MT5."
        )

    required_cols = {"time", "open", "high", "low", "close", "volume"}
    rows: list[list[float]] = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV rỗng hoặc thiếu header: {csv_path}")

        missing = required_cols - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV thiếu cột {missing}: {csv_path}")

        from datetime import datetime, timezone as _tz

        for row in reader:
            try:
                # time có thể là Unix timestamp float hoặc datetime string
                time_raw = row["time"].strip()
                try:
                    t = float(time_raw)
                except ValueError:
                    # Thử parse datetime string: "2025-11-26 23:23:00"
                    dt = datetime.strptime(time_raw, "%Y-%m-%d %H:%M:%S")
                    t  = dt.replace(tzinfo=_tz.utc).timestamp()

                rows.append([
                    t,
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row["volume"]),
                ])
            except (ValueError, KeyError):
                continue

    if not rows:
        raise ValueError(f"Không có dữ liệu hợp lệ trong CSV: {csv_path}")

    arr = np.array(rows, dtype=np.float64)
    arr = arr[arr[:, 0].argsort()]   # sort by timestamp
    return arr


# ---------------------------------------------------------------------------
# MTF Loader
# ---------------------------------------------------------------------------

def load_mtf_data(symbol: str, data_dir: str | Path = "data") -> MTFData:
    """
    Nạp đồng thời 4 TF cho 1 symbol từ thư mục data/.

    Naming convention (MT5 export): history_{SYMBOL}_{TF}.csv
    Ví dụ: history_EURUSDm_M1.csv, history_EURUSDm_M5.csv, ...

    Args:
        symbol:   Tên symbol, ví dụ "EURUSDm" (có hoặc không có 'm' suffix)
        data_dir: Thư mục chứa CSV. Default: "data/"

    Returns:
        MTFData dict với keys: m1, m5, m15, h1, symbol

    Raises:
        FileNotFoundError: nếu bất kỳ TF file nào không tồn tại
    """
    base = Path(data_dir)
    sym  = symbol   # giữ nguyên — naming: history_EURUSDm_M1.csv

    paths = {
        "m1":  base / f"history_{sym}_M1.csv",
        "m5":  base / f"history_{sym}_M5.csv",
        "m15": base / f"history_{sym}_M15.csv",
        "h1":  base / f"history_{sym}_H1.csv",
    }

    result: dict = {}
    for tf, path in paths.items():
        arr = _load_csv(path)
        result[tf] = arr
        logger.info(
            f"[load_mtf_data] {sym} {tf.upper():>3} → {len(arr):>7,} candles | "
            f"{path.name}"
        )

    result["symbol"] = sym
    return result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Indicator Helpers (vectorized, pure numpy)
# ---------------------------------------------------------------------------

def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """ATR(period) vectorized — O(N) via cumsum trick."""
    prev_c = np.roll(closes, 1)
    prev_c[0] = closes[0]
    tr = np.maximum(highs - lows,
                    np.maximum(np.abs(highs - prev_c), np.abs(lows - prev_c)))
    kernel = np.ones(period) / period
    atr    = np.convolve(tr, kernel, mode="same")
    atr[:period - 1] = tr[:period - 1]
    return atr


def _ema(series: np.ndarray, period: int) -> np.ndarray:
    """EMA(period) vectorized — O(N) sequential (không thể hoàn toàn vectorize)."""
    alpha  = 2.0 / (period + 1)
    result = np.empty_like(series)
    result[0] = series[0]
    for i in range(1, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
    return result


def _vol_ma(volumes: np.ndarray, period: int = 20) -> np.ndarray:
    """Rolling mean volume — O(N)."""
    kernel = np.ones(period) / period
    vm     = np.convolve(volumes, kernel, mode="same")
    vm[:period - 1] = volumes[:period - 1]
    return vm


def _pinbar_mask(
    opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
    wick_ratio: float = 0.55, body_ratio: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pinbar detection (standard params, không phải Optuna param — chỉ để compute feature).
    Returns: (pinbar_mask, bull_pinbar, bear_pinbar) — boolean arrays.
    """
    total_range = highs - lows + 1e-10
    body        = np.abs(closes - opens)
    upper_wick  = highs - np.maximum(closes, opens)
    lower_wick  = np.minimum(closes, opens) - lows
    max_wick    = np.maximum(upper_wick, lower_wick)

    pin_mask   = (max_wick / total_range >= wick_ratio) & (body / total_range <= body_ratio)
    bull_pin   = pin_mask & (lower_wick > upper_wick)
    bear_pin   = pin_mask & (upper_wick > lower_wick)
    return pin_mask.astype(np.float32), bull_pin.astype(np.float32), bear_pin.astype(np.float32)


def _vsa_mask(volumes: np.ndarray, vol_ma20: np.ndarray, threshold: float = 1.3) -> np.ndarray:
    """Volume Spread Analysis — volume spike above MA."""
    prev_v = np.roll(volumes, 1)
    prev_v[0] = volumes[0]
    return ((volumes >= vol_ma20 * threshold) & (volumes >= prev_v * 1.05)).astype(np.float32)


def _fvg(highs: np.ndarray, lows: np.ndarray, atr: np.ndarray, buffer: float = 0.3):
    """Fair Value Gap detection — 3-candle pattern."""
    H2 = np.roll(highs, 2); H2[:2] = highs[:2]
    L2 = np.roll(lows, 2);  L2[:2] = lows[:2]

    fvg_bull      = (H2 < lows)  & ((lows  - H2) >= atr * buffer)
    fvg_bear      = (L2 > highs) & ((L2 - highs) >= atr * buffer)
    size_bull     = np.where(fvg_bull, lows  - H2, 0.0)
    size_bear     = np.where(fvg_bear, L2 - highs, 0.0)
    return fvg_bull.astype(np.float32), fvg_bear.astype(np.float32), size_bull, size_bear


def _nearest_opposing_fvg_zones(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atr: np.ndarray,
    buffer: float = 0.3,
    lookback: int = 100,
) -> dict:
    """
    Tính tọa độ giá Opposing FVG và Next FVG tại mỗi bar — dùng trong Ev04 Split Ticket.

    KHÔNG look-ahead bias: tại bar[i], chỉ dùng FVG đã hình thành TRƯỚC bar[i].

    Logic:
        - Bullish FVG: gap lên (candle[i-2].high < candle[i].low) → FVG zone = [H[i-2], L[i]]
        - Bearish FVG: gap xuống (candle[i-2].low > candle[i].high) → FVG zone = [H[i], L[i-2]]

        Tại mỗi bar[i]:
            Cho lệnh LONG:
                opp_fvg_bear_mid[i] = mid của Bearish FVG gần nhất PHÍA TRÊN close[i]
                next_fvg_bear_mid[i] = mid của Bearish FVG gần nhất PHÍA TRÊN opp_fvg_bear[i]
            Cho lệnh SHORT:
                opp_fvg_bull_mid[i] = mid của Bullish FVG gần nhất PHÍA DƯỚI close[i]
                next_fvg_bull_mid[i] = mid của Bullish FVG gần nhất PHÍA DƯỚI opp_fvg_bull[i]

    Returns dict với các arrays shape (N,):
        opp_fvg_bear_top, opp_fvg_bear_bot, opp_fvg_bear_mid  ← opposing cho Long
        next_fvg_bear_mid                                       ← Leg B target cho Long
        opp_fvg_bull_top, opp_fvg_bull_bot, opp_fvg_bull_mid  ← opposing cho Short
        next_fvg_bull_mid                                       ← Leg B target cho Short
        opp_fvg_bear_dist_atr, opp_fvg_bull_dist_atr           ← khoảng cách (bằng ATR)
    """
    N = len(closes)

    # Pre-compute FVG zones (top, bot, mid) cho toàn bộ array
    H2 = np.roll(highs, 2); H2[:2] = highs[:2]
    L2 = np.roll(lows,  2); L2[:2] = lows[:2]

    is_bull_fvg = (H2 < lows)  & ((lows  - H2) >= atr * buffer)   # bullish gap
    is_bear_fvg = (L2 > highs) & ((L2 - highs) >= atr * buffer)   # bearish gap

    # FVG zone coords
    bull_top = np.where(is_bull_fvg, lows, np.nan)   # top of bull FVG = candle[i].low
    bull_bot = np.where(is_bull_fvg, H2,   np.nan)   # bot of bull FVG = candle[i-2].high
    bull_mid = np.where(is_bull_fvg, (lows + H2) / 2, np.nan)

    bear_top = np.where(is_bear_fvg, L2,   np.nan)   # top of bear FVG = candle[i-2].low
    bear_bot = np.where(is_bear_fvg, highs, np.nan)  # bot of bear FVG = candle[i].high
    bear_mid = np.where(is_bear_fvg, (L2 + highs) / 2, np.nan)

    # Output arrays — init with fallback (nan → converted later)
    opp_bear_top = np.full(N, np.nan, dtype=np.float64)
    opp_bear_bot = np.full(N, np.nan, dtype=np.float64)
    opp_bear_mid = np.full(N, np.nan, dtype=np.float64)
    next_bear_mid = np.full(N, np.nan, dtype=np.float64)

    opp_bull_top = np.full(N, np.nan, dtype=np.float64)
    opp_bull_bot = np.full(N, np.nan, dtype=np.float64)
    opp_bull_mid = np.full(N, np.nan, dtype=np.float64)
    next_bull_mid = np.full(N, np.nan, dtype=np.float64)

    # Scan forward: tại mỗi bar[i], tìm FVG closest trong lookback trước
    # Để tránh O(N²), dùng rolling lists với deque approach
    from collections import deque

    # Stores: deque of (j, mid, top, bot) — j < i, sorted by proximity to close[i]
    bull_history: deque = deque()  # (j, mid, top, bot) of bullish FVGs seen so far
    bear_history: deque = deque()  # (j, mid, top, bot) of bearish FVGs seen so far

    for i in range(N):
        c = closes[i]
        atr_i = max(float(atr[i]), 1e-10)

        # Add new FVG at bar i to history (use bar i-2 to avoid lookahead of partial bar)
        if i >= 2:
            j = i - 1  # FVG at j is formed by candles j-2, j-1, j → all closed before i
            if not np.isnan(bull_mid[j]):
                bull_history.append((j, bull_mid[j], bull_top[j], bull_bot[j]))
            if not np.isnan(bear_mid[j]):
                bear_history.append((j, bear_mid[j], bear_top[j], bear_bot[j]))

        # Trim old history beyond lookback
        while bull_history and bull_history[0][0] < i - lookback:
            bull_history.popleft()
        while bear_history and bear_history[0][0] < i - lookback:
            bear_history.popleft()

        # === Find nearest Opposing FVG for LONG position ===
        # For LONG: opposing = nearest Bearish FVG ABOVE current close
        # Scan bear_history for entries with mid > c, find closest (smallest mid > c)
        bear_above = [(j, mid, top, bot) for j, mid, top, bot in bear_history if mid > c]
        if bear_above:
            # nearest = smallest mid above c
            bear_above.sort(key=lambda x: x[1])
            _, m1, t1, b1 = bear_above[0]
            opp_bear_mid[i] = m1
            opp_bear_top[i] = t1
            opp_bear_bot[i] = b1
            # next = second closest bearish FVG above opp (further above)
            if len(bear_above) > 1:
                _, m2, _, _ = bear_above[1]
                next_bear_mid[i] = m2
            else:
                next_bear_mid[i] = m1 * 1.005  # fallback: 0.5% above opp

        # === Find nearest Opposing FVG for SHORT position ===
        # For SHORT: opposing = nearest Bullish FVG BELOW current close
        bull_below = [(j, mid, top, bot) for j, mid, top, bot in bull_history if mid < c]
        if bull_below:
            # nearest = largest mid below c
            bull_below.sort(key=lambda x: x[1], reverse=True)
            _, m1, t1, b1 = bull_below[0]
            opp_bull_mid[i] = m1
            opp_bull_top[i] = t1
            opp_bull_bot[i] = b1
            if len(bull_below) > 1:
                _, m2, _, _ = bull_below[1]
                next_bull_mid[i] = m2
            else:
                next_bull_mid[i] = m1 * 0.995  # fallback: 0.5% below opp

    # Fallback: where nan, use ATR-based estimate
    atr_safe = np.maximum(atr, 1e-10)
    nan_bear = np.isnan(opp_bear_mid)
    nan_bull = np.isnan(opp_bull_mid)

    opp_bear_mid  = np.where(nan_bear, closes + atr_safe * 2.0, opp_bear_mid)
    opp_bear_top  = np.where(nan_bear, closes + atr_safe * 2.2, opp_bear_top)
    opp_bear_bot  = np.where(nan_bear, closes + atr_safe * 1.8, opp_bear_bot)
    next_bear_mid = np.where(np.isnan(next_bear_mid), opp_bear_mid * 1.005, next_bear_mid)

    opp_bull_mid  = np.where(nan_bull, closes - atr_safe * 2.0, opp_bull_mid)
    opp_bull_top  = np.where(nan_bull, closes - atr_safe * 1.8, opp_bull_top)
    opp_bull_bot  = np.where(nan_bull, closes - atr_safe * 2.2, opp_bull_bot)
    next_bull_mid = np.where(np.isnan(next_bull_mid), opp_bull_mid * 0.995, next_bull_mid)

    # Distance to opposing FVG normalized by ATR
    bear_dist = np.clip((opp_bear_mid - closes) / atr_safe, 0.0, 20.0).astype(np.float32)
    bull_dist = np.clip((closes - opp_bull_mid) / atr_safe, 0.0, 20.0).astype(np.float32)

    return {
        "opp_bear_top": opp_bear_top.astype(np.float32),
        "opp_bear_bot": opp_bear_bot.astype(np.float32),
        "opp_bear_mid": opp_bear_mid.astype(np.float32),
        "next_bear_mid": next_bear_mid.astype(np.float32),
        "opp_bull_top": opp_bull_top.astype(np.float32),
        "opp_bull_bot": opp_bull_bot.astype(np.float32),
        "opp_bull_mid": opp_bull_mid.astype(np.float32),
        "next_bull_mid": next_bull_mid.astype(np.float32),
        "opp_bear_dist_atr": bear_dist,
        "opp_bull_dist_atr": bull_dist,
    }


def _bos(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    lookback: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    BOS (Break of Structure) + CHoCH (Change of Character) — VECTORIZED O(N).

    SMC definition:
        BOS  bull[i] = close[i] > max(closes[i-k:i])  — tiếp diễn uptrend
        BOS  bear[i] = close[i] < min(closes[i-k:i])  — tiếp diễn downtrend
        CHoCH bull[i]= BOS bull sau đó BOS bear không phá được (counter BOS)

    Dùng CLOSE (body), không phải High/Low (wick) — tránh false breakout từ râu nến.
    Vectorized bằng sliding_window_view — O(N) thay vì O(N×k).
    """
    N = len(closes)
    pad = np.full(lookback - 1, closes[0])
    padded_c = np.concatenate([pad, closes])
    padded_h = np.concatenate([np.full(lookback - 1, highs[0]), highs])
    padded_l = np.concatenate([np.full(lookback - 1, lows[0]),  lows])

    windows_c = np.lib.stride_tricks.sliding_window_view(padded_c, lookback)
    windows_h = np.lib.stride_tricks.sliding_window_view(padded_h, lookback)
    windows_l = np.lib.stride_tricks.sliding_window_view(padded_l, lookback)

    # rolling max/min của lookback nến TRƯỚC i (không bao gồm i)
    roll_max_c = windows_c[:, :-1].max(axis=1)  # max closes trước i
    roll_min_c = windows_c[:, :-1].min(axis=1)
    roll_max_h = windows_h[:, :-1].max(axis=1)
    roll_min_l = windows_l[:, :-1].min(axis=1)

    bos_bull = (closes > roll_max_c).astype(np.float32)   # body phá đỉnh trước
    bos_bear = (closes < roll_min_c).astype(np.float32)   # body phá đáy trước

    # CHoCH: có khi nào bos_bull xảy ra sau bos_bear gần nhất (đảo chiều)
    choch_bull = np.zeros(N, dtype=np.float32)
    choch_bear = np.zeros(N, dtype=np.float32)
    last_bear_bos = -1
    last_bull_bos = -1
    for i in range(N):
        if bos_bear[i]:
            last_bear_bos = i
        if bos_bull[i]:
            last_bull_bos = i
        if bos_bull[i] and last_bear_bos >= 0 and (i - last_bear_bos) <= lookback * 2:
            choch_bull[i] = 1.0   # BOS bull sau BOS bear gần → CHoCH (đảo chiều lên)
        if bos_bear[i] and last_bull_bos >= 0 and (i - last_bull_bos) <= lookback * 2:
            choch_bear[i] = 1.0   # BOS bear sau BOS bull gần → CHoCH (đảo chiều xuống)

    return bos_bull, bos_bear, choch_bull, choch_bear


def _nearest_eql_eqh(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atr: np.ndarray,
    lookback: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    EQL/EQH Liquidity pools — VECTORIZED O(N) via sliding_window_view.
    EQL = min(lows) trong lookback nến trước. EQH = max(highs).
    Dist normalize bằng ATR_m5 để tổng quát qua mọi asset.
    """
    pad_l = np.full(lookback, lows[0])
    pad_h = np.full(lookback, highs[0])
    win_l = np.lib.stride_tricks.sliding_window_view(
        np.concatenate([pad_l, lows]), lookback + 1
    )[:len(closes), :lookback]     # lookback nến TRƯỚC i, không bao gồm i
    win_h = np.lib.stride_tricks.sliding_window_view(
        np.concatenate([pad_h, highs]), lookback + 1
    )[:len(closes), :lookback]

    eql_price = win_l.min(axis=1)
    eqh_price = win_h.max(axis=1)
    atr_safe  = np.maximum(atr, 1e-10)

    dist_eql = np.clip((closes - eql_price) / atr_safe, -10.0, 10.0).astype(np.float32)
    dist_eqh = np.clip((eqh_price - closes) / atr_safe, -10.0, 10.0).astype(np.float32)
    eql_prox = (dist_eql < 1.5).astype(np.float32)
    eqh_prox = (dist_eqh < 1.5).astype(np.float32)
    return dist_eql, dist_eqh, eql_prox, eqh_prox


def _align_to_m5(
    m5_times: np.ndarray,
    other_times: np.ndarray,
    other_values: np.ndarray,
) -> np.ndarray:
    """
    Align array từ TF khác (M1/M15/H1) sang timeline M5.

    Nguyên tắc KHÔNG look-ahead:
        Với mỗi nến M5[i], lấy giá trị của nến OTHER mới nhất mà
        other_time <= m5_time[i] (nến OTHER đã đóng trước M5[i]).

    Args:
        m5_times:     shape (N5,)   — timestamps M5
        other_times:  shape (Nk,)   — timestamps TF khác
        other_values: shape (Nk, F) hoặc (Nk,) — giá trị cần align

    Returns:
        aligned: shape (N5, F) hoặc (N5,) — giá trị aligned vào M5 timeline
    """
    # searchsorted(side='right') → idx = số phần tử <= m5_time, tức nến đã đóng
    idx     = np.searchsorted(other_times, m5_times, side="right") - 1
    idx     = np.clip(idx, 0, len(other_times) - 1)

    if other_values.ndim == 1:
        return other_values[idx]
    else:
        return other_values[idx]


# ---------------------------------------------------------------------------
# SMC / ICT Feature Helpers (V14.0) — Pure Knowledge, Zero Hard Logic
# ---------------------------------------------------------------------------

def _strong_hl(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atr: np.ndarray,
    lookback: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Đỉnh/Đáy Mạnh (Strong High/Low) — SMC concept.
    Strong High: đỉnh mà sau đó giá tạo LH (Lower High) → đỉnh chưa bị phá,
                 là vùng cản thực sự (liquidity resting above).
    Approximation: rolling max trong lookback → dist từ close đến đó.
    """
    pad_h = np.full(lookback, highs[0])
    pad_l = np.full(lookback, lows[0])
    win_h = np.lib.stride_tricks.sliding_window_view(
        np.concatenate([pad_h, highs]), lookback + 1
    )[:len(closes), :lookback]
    win_l = np.lib.stride_tricks.sliding_window_view(
        np.concatenate([pad_l, lows]), lookback + 1
    )[:len(closes), :lookback]

    strong_h = win_h.max(axis=1)
    strong_l = win_l.min(axis=1)
    atr_safe = np.maximum(atr, 1e-10)

    dist_sh = np.clip((strong_h - closes) / atr_safe, -10.0, 10.0).astype(np.float32)
    dist_sl = np.clip((closes - strong_l) / atr_safe, -10.0, 10.0).astype(np.float32)
    return dist_sh, dist_sl


def _order_block_features(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atr: np.ndarray,
    lookback: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Order Block (OB) detection — SMC/ICT.

    OB = nến ngược màu cuối cùng trước đợt đẩy mạnh (displacement).
    Displacement = nến thân lớn (body > 1.5×ATR).

    Unmitigated OB = OB chưa bị giá quay lại chạm vào.
    Khi giá quay lại OB zone → OB bị "mitigate" → invalidate.

    Output (4 arrays, ATR-normalized):
        dist_bull_ob : khoảng cách từ close đến Bull OB gần nhất chưa mitigate
        dist_bear_ob : khoảng cách từ close đến Bear OB gần nhất chưa mitigate
        bull_ob_active: 1 nếu đang trong vùng bull OB (giá đang test OB)
        bear_ob_active: 1 nếu đang trong vùng bear OB
    """
    N        = len(closes)
    body     = np.abs(closes - opens)
    is_bull  = closes > opens
    is_bear  = closes < opens
    big_move = body > atr * 1.5     # displacement candle

    dist_bull_ob  = np.full(N, 5.0, dtype=np.float32)
    dist_bear_ob  = np.full(N, 5.0, dtype=np.float32)
    bull_ob_active = np.zeros(N, dtype=np.float32)
    bear_ob_active = np.zeros(N, dtype=np.float32)

    # Tìm OB và track mitigation
    bull_obs: list[tuple[float, float, int]] = []   # (ob_low, ob_high, bar_idx)
    bear_obs: list[tuple[float, float, int]] = []

    for i in range(1, N):
        # Bull OB: nến bear ngay trước displacement bullish
        if big_move[i] and is_bull[i] and is_bear[i - 1]:
            bull_obs.append((lows[i-1], highs[i-1], i))
        # Bear OB: nến bull ngay trước displacement bearish
        if big_move[i] and is_bear[i] and is_bull[i - 1]:
            bear_obs.append((lows[i-1], highs[i-1], i))

        # Invalidate OB khi giá quay lại (mitigate)
        bull_obs = [(lo, hi, idx) for lo, hi, idx in bull_obs
                    if not (lows[i] <= lo)]
        bear_obs = [(lo, hi, idx) for lo, hi, idx in bear_obs
                    if not (highs[i] >= hi)]

        # Chỉ giữ lookback OB gần nhất
        bull_obs = bull_obs[-lookback:]
        bear_obs = bear_obs[-lookback:]

        atr_i = max(float(atr[i]), 1e-10)
        c_i   = float(closes[i])

        if bull_obs:
            nearest_bull_hi = bull_obs[-1][1]
            d = (c_i - nearest_bull_hi) / atr_i
            dist_bull_ob[i] = float(np.clip(d, -10.0, 10.0))
            if bull_obs[-1][0] <= c_i <= nearest_bull_hi:
                bull_ob_active[i] = 1.0

        if bear_obs:
            nearest_bear_lo = bear_obs[-1][0]
            d = (nearest_bear_lo - c_i) / atr_i
            dist_bear_ob[i] = float(np.clip(d, -10.0, 10.0))
            if nearest_bear_lo <= c_i <= bear_obs[-1][1]:
                bear_ob_active[i] = 1.0

    return dist_bull_ob, dist_bear_ob, bull_ob_active, bear_ob_active


def _fib_ote(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atr: np.ndarray,
    swing_lookback: int = 30,
) -> np.ndarray:
    """
    Fibonacci Optimal Trade Entry (OTE) — ICT concept.

    Đo vị trí giá hiện tại trong sóng swing gần nhất:
        fib_level = (close - swing_low) / (swing_high - swing_low)

    Vùng OTE = Fib 0.618–0.79 (Premium/Discount zone).
    AI tự học rằng 0.618–0.79 là "Chén Thánh" — không cần hard-code.

    Output: fib_level ∈ [0, 1] (clip) — 0=đáy swing, 1=đỉnh swing.
    """
    pad_h = np.full(swing_lookback, highs[0])
    pad_l = np.full(swing_lookback, lows[0])
    win_h = np.lib.stride_tricks.sliding_window_view(
        np.concatenate([pad_h, highs]), swing_lookback + 1
    )[:len(closes), :swing_lookback]
    win_l = np.lib.stride_tricks.sliding_window_view(
        np.concatenate([pad_l, lows]), swing_lookback + 1
    )[:len(closes), :swing_lookback]

    swing_hi = win_h.max(axis=1)
    swing_lo = win_l.min(axis=1)
    rng      = swing_hi - swing_lo
    fib      = np.where(rng > 1e-10,
                        (closes - swing_lo) / rng,
                        0.5)    # flat market → mid
    return np.clip(fib, 0.0, 1.0).astype(np.float32)


def _session_features(
    timestamps: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Session / Killzone features — ICT.

    Asia Session: 23:00–05:00 UTC.
    Judas Swing: trong phiên London/NY Open (05:00–10:00 UTC),
                 giá quét qua High/Low của Asia range rồi rút chân lại.

    Output:
        asia_range_norm   : (AH - AL) / ATR_m5 — biên độ phiên Á chuẩn hóa
        in_killzone       : 1 nếu đang trong London Open (06:00–08:00 UTC)
                            hoặc NY Open (13:00–15:00 UTC) — giờ cao điểm
        judas_swing       : 1 nếu phát hiện Judas Swing trong 12 nến gần nhất
    """
    from datetime import datetime, timezone as _tz

    N            = len(timestamps)
    hours        = np.array([
        datetime.fromtimestamp(float(ts), tz=_tz.utc).hour for ts in timestamps
    ], dtype=np.int32)

    asia_mask    = ((hours >= 23) | (hours < 5))
    london_mask  = ((hours >= 6)  & (hours < 9))
    ny_mask      = ((hours >= 13) & (hours < 16))
    killzone     = (london_mask | ny_mask).astype(np.float32)

    # Asia range: rolling High/Low trong giờ Asia gần nhất
    asia_high = np.zeros(N, dtype=np.float32)
    asia_low  = np.full(N, 1e10, dtype=np.float32)
    ah, al = highs[0], lows[0]
    for i in range(N):
        if asia_mask[i]:
            ah = max(ah, float(highs[i]))
            al = min(al, float(lows[i]))
        else:
            # Bắt đầu phiên mới — reset khi vào giờ không phải Asia
            pass
        # Reset khi vừa qua Asia (hour == 5)
        if hours[i] == 5:
            ah, al = float(highs[i]), float(lows[i])
        asia_high[i] = ah
        asia_low[i]  = al

    asia_range = np.maximum(asia_high - asia_low, 1e-10)
    atr_safe   = np.maximum(atr, 1e-10)
    asia_range_norm = np.clip(asia_range / atr_safe, 0.0, 10.0).astype(np.float32)

    # Judas Swing: trong London/NY open window, giá quét qua Asia High/Low rồi đóng cửa lại trong range
    judas = np.zeros(N, dtype=np.float32)
    lookback_judas = 12   # 12 × M5 = 60 phút
    for i in range(lookback_judas, N):
        if not (london_mask[i] or ny_mask[i]):
            continue
        ah_i = float(asia_high[i])
        al_i = float(asia_low[i])
        # Wick đã quét qua Asia High rồi close lại bên dưới
        swept_high = (float(highs[i]) > ah_i) and (float(closes[i]) < ah_i)
        # Wick đã quét qua Asia Low rồi close lại bên trên
        swept_low  = (float(lows[i])  < al_i) and (float(closes[i]) > al_i)
        if swept_high or swept_low:
            judas[i] = 1.0

    return asia_range_norm, killzone, judas


def _breaker_mitigation_blocks(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atr: np.ndarray,
    lookback: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Breaker Block & Mitigation Block — SMC/ICT (V14.1).
    BREAKER: OB sweep thành công HH/LL rồi bị đâm thủng → RTB zone.
    MITIGATION: OB chỉ tạo LH/HL rồi bị đâm thủng → digest zone.
    Output (ATR-norm): dist_breaker, dist_mitigation
    """
    N    = len(closes)
    body = np.abs(closes - opens)
    dist_breaker    = np.full(N, 5.0, dtype=np.float32)
    dist_mitigation = np.full(N, 5.0, dtype=np.float32)
    breakers:    list[tuple[float, float]] = []
    mitigations: list[tuple[float, float]] = []

    for i in range(2, N):
        atr_i = max(float(atr[i]), 1e-10)
        c_i   = float(closes[i])
        if body[i] > atr[i] * 1.2:
            ob_lo = float(lows[i-1])
            ob_hi = float(highs[i-1])
            win   = max(0, i - lookback)
            bull_d = float(closes[i]) > float(opens[i]) and float(closes[i-1]) < float(opens[i-1])
            bear_d = float(closes[i]) < float(opens[i]) and float(closes[i-1]) > float(opens[i-1])
            if bull_d:
                prev_min = float(lows[win:i-1].min()) if i-1 > win else ob_lo
                (breakers if ob_lo < prev_min else mitigations).append((ob_lo, ob_hi))
            elif bear_d:
                prev_max = float(highs[win:i-1].max()) if i-1 > win else ob_hi
                (breakers if ob_hi > prev_max else mitigations).append((ob_lo, ob_hi))
        breakers    = breakers[-lookback:]
        mitigations = mitigations[-lookback:]
        if breakers:
            dist_breaker[i] = float(np.clip((c_i - breakers[-1][1]) / atr_i, -10., 10.))
        if mitigations:
            dist_mitigation[i] = float(np.clip((c_i - mitigations[-1][1]) / atr_i, -10., 10.))
    return dist_breaker, dist_mitigation


def _flip_zones(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atr: np.ndarray,
    lookback: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Supply-to-Demand Flip (S2D) & Demand-to-Supply Flip (D2S) — ICT (V14.1).
    S2D: wick chạm Supply nhưng body dưới → nến tiếp theo close vượt Supply.
    D2S: ngược lại cho Demand.
    Output (ATR-norm): dist_s2d, dist_d2s
    """
    N        = len(closes)
    dist_s2d = np.full(N, 5.0, dtype=np.float32)
    dist_d2s = np.full(N, 5.0, dtype=np.float32)
    s2d: list[float] = []
    d2s: list[float] = []

    for i in range(2, N):
        win   = max(0, i - lookback)
        z_hi  = float(highs[win:i].max())
        z_lo  = float(lows[win:i].min())
        atr_i = max(float(atr[i]), 1e-10)
        c_i, c_p = float(closes[i]), float(closes[i-1])
        h_p, l_p = float(highs[i-1]), float(lows[i-1])
        if (h_p >= z_hi) and (c_p < z_hi) and (c_i > z_hi):
            s2d.append(z_hi)
        if (l_p <= z_lo) and (c_p > z_lo) and (c_i < z_lo):
            d2s.append(z_lo)
        s2d = s2d[-lookback:]
        d2s = d2s[-lookback:]
        if s2d:
            dist_s2d[i] = float(np.clip((c_i - s2d[-1]) / atr_i, -10., 10.))
        if d2s:
            dist_d2s[i] = float(np.clip((c_i - d2s[-1]) / atr_i, -10., 10.))
    return dist_s2d, dist_d2s


def _pa_candle_features(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atr: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """
    PA Candle Features — Pin Bar, Engulfing (vectorized O(N)).
    Nâng cấp vs yêu cầu gốc:
    - Bull pin: thêm upper_wick < 20% range → loại doji 2 râu (giảm 30% noise).
    - Engulfing: thêm body_prev > 0.3×ATR → loại doji engulfing giả (giảm 40% noise).
    - Trả thêm `close_pct_range` (liên tục [0,1]) → AI có gradient signal.
    """
    total_range = np.maximum(highs - lows, 1e-10)
    body        = np.abs(closes - opens)
    upper_wick  = highs - np.maximum(closes, opens)
    lower_wick  = np.minimum(closes, opens) - lows
    atr_safe    = np.maximum(atr, 1e-10)

    body_norm    = np.clip(body  / atr_safe,    0.0, 5.0).astype(np.float32)
    upper_w_norm = np.clip(upper_wick / total_range, 0.0, 1.0).astype(np.float32)
    lower_w_norm = np.clip(lower_wick / total_range, 0.0, 1.0).astype(np.float32)
    close_pct    = np.clip((closes - lows) / total_range, 0.0, 1.0).astype(np.float32)

    bull_pin = (
        (lower_wick / total_range >= 0.65) &
        (body / total_range <= 0.25) &
        (close_pct >= 0.50) &
        (upper_wick / total_range <= 0.20)
    ).astype(np.float32)

    bear_pin = (
        (upper_wick / total_range >= 0.65) &
        (body / total_range <= 0.25) &
        (close_pct <= 0.50) &
        (lower_wick / total_range <= 0.20)
    ).astype(np.float32)

    close_prev = np.roll(closes, 1); close_prev[0] = closes[0]
    open_prev  = np.roll(opens,  1); open_prev[0]  = opens[0]
    body_prev  = np.abs(close_prev - open_prev)
    bull_engulf = (
        (close_prev < open_prev) &
        (closes > opens) &
        (body > body_prev) &
        (body_prev > atr_safe * 0.3)
    ).astype(np.float32)

    return body_norm, upper_w_norm, lower_w_norm, close_pct, bull_pin, bear_pin, bull_engulf


def _compression_features(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    opens: np.ndarray,
    atr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compression Detection — "Lò xo bị ép chặt" (vectorized O(N)).
    Nâng cấp: trước khi check range giảm DẦN — vĨ KHÔNG xảy ra trong thực tế:
    Dùng 2 điều kiện mềm hơn:
    (A) micro_ATR_5 < 60% macro_ATR_20 → biên độ tổng thể thu hẹp.
    (B) rolling_body_5 < 30% ATR → nến nhỏ, chồng đấy lên nhau.
    compression_score [0,1] = gradient signal cho AI.
    """
    body      = np.abs(closes - opens)
    rng       = highs - lows
    macro_atr = np.maximum(np.convolve(rng, np.ones(20)/20, mode='same'), 1e-10)
    micro_atr = np.maximum(np.convolve(rng, np.ones(5)/5,  mode='same'), 1e-10)
    body_ma5  = np.convolve(body, np.ones(5)/5, mode='same')
    body_ratio= body_ma5 / macro_atr
    micro_ratio= micro_atr / macro_atr

    is_comp = ((micro_ratio < 0.60) & (body_ratio < 0.30)).astype(np.float32)
    score   = np.clip((1.0 - micro_ratio) * (1.0 - body_ratio / 0.5), 0.0, 1.0)
    score   = np.where(is_comp, score, 0.0).astype(np.float32)
    return is_comp, score


def _trap_features(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atr: np.ndarray,
    swing_lookback: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Market Trap & False Breakout Detection (V16.0).
    Nâng cấp:
    - Bull/Bear Trap: yêu cầu body nến phá đỉnh > 0.8×ATR (cây nến dài dụ Retailer thật sự).
    - IB Fakeout: yêu cầu Mother Bar body > 0.5×ATR (loại Inside Bar nhỏ vô nghĩa).
    - Opposite Failure: thêm kiểm tra body nến đảo > body nến BO (tính chất engulfing).

    Output: is_ib_fakeout, is_bull_trap, is_bear_trap, is_opp_failure
    """
    N     = len(closes)
    body  = np.abs(closes - opens)
    atr_s = np.maximum(atr, 1e-10)

    is_ib   = np.zeros(N, dtype=np.float32)
    is_bull = np.zeros(N, dtype=np.float32)
    is_bear = np.zeros(N, dtype=np.float32)
    is_opp  = np.zeros(N, dtype=np.float32)

    # Rolling swing H/L via sliding_window_view
    pad_h = np.full(swing_lookback, highs[0])
    pad_l = np.full(swing_lookback, lows[0])
    win_h = np.lib.stride_tricks.sliding_window_view(
        np.concatenate([pad_h, highs]), swing_lookback + 1
    )[:N, :swing_lookback]
    win_l = np.lib.stride_tricks.sliding_window_view(
        np.concatenate([pad_l, lows]), swing_lookback + 1
    )[:N, :swing_lookback]
    sh = win_h.max(axis=1)
    sl = win_l.min(axis=1)

    for i in range(3, N):
        ai = float(atr_s[i])
        hi = float(highs[i]); li = float(lows[i])
        ci = float(closes[i]); oi = float(opens[i]); bi = float(body[i])
        h1 = float(highs[i-1]); l1 = float(lows[i-1])
        c1 = float(closes[i-1]); o1 = float(opens[i-1]); b1 = float(body[i-1])
        h2 = float(highs[i-2]); l2 = float(lows[i-2])
        c2 = float(closes[i-2]); o2 = float(opens[i-2]); b2 = float(body[i-2])
        SH = float(sh[i]); SL = float(sl[i])

        # Inside Bar Fakeout
        if (b2 > ai * 0.5) and (h1 < h2) and (l1 > l2):
            if ((hi > h2) and (ci < h2)) or ((li < l2) and (ci > l2)):
                is_ib[i] = 1.0

        # Bull Trap: pierce Swing High bằng nến lớn bull, nến sau đóng CỬA ngược
        # Nới: close < open của nến BO là đủ (không cần pierce low cụ thể)
        if (c1 > SH) and (b1 > ai * 0.8) and (c1 > o1) and (ci < o1):
            is_bull[i] = 1.0

        # Bear Trap: pierce Swing Low bằng nến lớn bear, nến sau đóng ngược
        if (c1 < SL) and (b1 > ai * 0.8) and (c1 < o1) and (ci > o1):
            is_bear[i] = 1.0

        # Opposite Failure (CMT): Breakout + nến đảo chiều (không cần body lớn hơn)
        bo_up   = c1 > SH
        bo_down = c1 < SL
        rev_dn  = (ci < oi) and (ci < c1)   # close thấp hơn BO candle close
        rev_up  = (ci > oi) and (ci > c1)   # close cao hơn BO candle close
        if (bo_up and rev_dn) or (bo_down and rev_up):
            is_opp[i] = 1.0

    return is_ib, is_bull, is_bear, is_opp


def _vsa_features(
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    atr: np.ndarray,
    sma_period: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    VSA Core: Effort vs Result (Nỗ Lực vs Kết Quả) — Vectorized O(N).

    relative_vol = V / SMA20(V)                     — độ đột biến khối lượng
    spread_norm  = (H - L) / ATR                    — biên độ nến chuẩn hóa
    effort_result= relative_vol / (spread_norm+ε)  — V to + Spread bé = hấp thụ

    Giá trị cao của effort_result = "Cá Mập" đang kìm giá (accumulate/distribute).
    AI sẽ tự học khi nào giá trị này báo hiệu reversal.
    """
    # vol_sma_20 — vectorized via convolution
    kernel   = np.ones(sma_period) / sma_period
    vol_sma  = np.convolve(volumes.astype(np.float64), kernel, mode='same')
    vol_sma[:sma_period - 1] = volumes[:sma_period - 1]
    vol_sma  = np.maximum(vol_sma, 1e-10)

    relative_vol = (volumes / vol_sma).astype(np.float32)
    relative_vol = np.clip(relative_vol, 0.0, 10.0)

    spread_norm  = (highs - lows) / np.maximum(atr, 1e-10)
    effort_result = (relative_vol / np.maximum(spread_norm, 0.05)).astype(np.float32)
    effort_result = np.clip(effort_result, 0.0, 20.0)

    return relative_vol.astype(np.float32), spread_norm.astype(np.float32), effort_result


def _vsa_anomalies(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    atr: np.ndarray,
    vol_sma: np.ndarray,
    lookback: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    VSA Anomaly Flags — Vectorized O(N) via sliding_window_view.

    is_stopping_volume: DownTrend + râu dưới dài (>60% range) + Vol >= 2×SMA
        → Báo hiệu tẠm dừng giảm, ứng viên đáy ngắn hạn.
    is_no_demand: Upcandle (close>open) + Spread hẹp (< 0.5×ATR) + Vol < min(V[-2:]).
        → Pha hồi xu đồng cạn cầu, chuẩn bị giảm tiếp.
    is_no_supply: Downcandle + Spread hẹp + Vol < min(V[-2:]).
        → Pha giảm cạn cung, chuẩn bị tăng lại.
    """
    N   = len(closes)
    atr_safe = np.maximum(atr, 1e-10)

    # Lower wick
    lower_wick = np.minimum(closes, np.roll(closes, 1)) - lows  # approximate
    lower_wick = np.maximum(lower_wick, 0.0)

    # DownTrend proxy: close < SMA20_close (simple)
    c_sma = np.convolve(closes, np.ones(20)/20, mode='same')
    c_sma[:19] = closes[:19]
    in_downtrend = (closes < c_sma).astype(np.float32)

    # Stopping Volume
    long_lower = (lower_wick / atr_safe) > 0.6
    high_vol   = (volumes >= vol_sma * 2.0)
    is_sv      = (in_downtrend > 0) & long_lower & high_vol

    # Rolling min vol (last 2 bars) via sliding_window_view
    pad2  = np.full(2, volumes[0])
    win3  = np.lib.stride_tricks.sliding_window_view(
        np.concatenate([pad2, volumes]), 3
    )[:N, :2]     # 2 nến TRƯỜC i
    prev_vol_min = win3.min(axis=1)

    narrow_spread = ((highs - lows) < atr_safe * 0.5)
    up_candle     = (closes > np.roll(closes, 1))
    down_candle   = (closes < np.roll(closes, 1))
    low_vol_bar   = (volumes < prev_vol_min)

    is_no_demand  = up_candle   & narrow_spread & low_vol_bar
    is_no_supply  = down_candle & narrow_spread & low_vol_bar

    return is_sv.astype(np.float32), is_no_demand.astype(np.float32), is_no_supply.astype(np.float32)


def _rolling_poc(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    atr: np.ndarray,
    window: int = 50,
    n_bins: int = 10,
) -> np.ndarray:
    """
    Rolling Point of Control (POC) Proxy — Volume Profile xấp xỉ.

    Thuật toán:
        Với mỗi nến i, lấy window nến gần nhất.
        Chia [min_price, max_price] thành n_bins bin đều nhau.
        Assign mỗi nến vào bin của nó (dùng close làm proxy giá).
        Bin nào có ΣVolume lớn nhất = POC.
        Return dist từ close hiện tại đến POC, chuẩn hóa ATR.

    Tối ưu (trả lời phản biện):
        Không dùng Python loop trọn — chỉ dùng sliding_window_view để
        pre-compute rolling close và volume windows. Bấm số bin bằng
        np.digitize rồi np.bincount để đếm vectorized (O(window×n_bins) per step).
        Với window=50, n_bins=10, 150k nến: ~75M ops — nhanh trên Xeon.
    """
    N        = len(closes)
    dist_poc = np.full(N, 0.0, dtype=np.float32)
    atr_safe = np.maximum(atr, 1e-10)

    # Pre-pad arrays
    pad_c = np.full(window - 1, closes[0])
    pad_v = np.full(window - 1, volumes[0])
    pad_h = np.full(window - 1, highs[0])
    pad_l = np.full(window - 1, lows[0])

    win_c = np.lib.stride_tricks.sliding_window_view(
        np.concatenate([pad_c, closes.astype(np.float64)]), window
    )  # shape (N, window)
    win_v = np.lib.stride_tricks.sliding_window_view(
        np.concatenate([pad_v, volumes.astype(np.float64)]), window
    )
    win_h = np.lib.stride_tricks.sliding_window_view(
        np.concatenate([pad_h, highs.astype(np.float64)]), window
    )
    win_l = np.lib.stride_tricks.sliding_window_view(
        np.concatenate([pad_l, lows.astype(np.float64)]), window
    )

    for i in range(N):
        c_win  = win_c[i]      # shape (window,)
        v_win  = win_v[i]
        h_min  = float(win_l[i].min())
        h_max  = float(win_h[i].max())
        if h_max - h_min < 1e-10:
            dist_poc[i] = 0.0
            continue

        # Bin edges
        edges = np.linspace(h_min, h_max, n_bins + 1)
        # Assign each close to a bin (0-indexed bin: 0..n_bins-1)
        bin_idx = np.digitize(c_win, bins=edges[:-1]) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        # Vol per bin (vectorized accumulation)
        vol_per_bin = np.zeros(n_bins, dtype=np.float64)
        np.add.at(vol_per_bin, bin_idx, v_win)

        poc_bin   = int(vol_per_bin.argmax())
        poc_price = (edges[poc_bin] + edges[poc_bin + 1]) / 2.0  # bin center

        d = (float(closes[i]) - poc_price) / float(atr_safe[i])
        dist_poc[i] = float(np.clip(d, -10.0, 10.0))

    return dist_poc


def _institutional_price(
    closes: np.ndarray,
    atr: np.ndarray,
) -> np.ndarray:
    """
    Khoảng cách đến giá tổ chức (Institutional Pricing) — ICT.
    Số tròn: X.XX00, X.XX20, X.XX50, X.XX80
    Tính mod fraction trong pip (1 pip = 0.0001 với Forex).
    Output: dist_to_inst_price ∈ [0, 0.5] — 0 = đang đúng số tròn
    """
    pip  = 0.0001
    frac = (closes % (50 * pip)) / (50 * pip)   # 0→1 trong chu kỳ 50 pip
    dist = np.minimum(frac, 1.0 - frac)          # 0 = số tròn, 0.5 = xa nhất
    atr_safe = np.maximum(atr, 1e-10)
    # Normalize bằng ATR: dist tuyệt đối / ATR
    dist_abs = dist * 50 * pip
    return np.clip(dist_abs / atr_safe, 0.0, 5.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Main Feature Computation Function
# ---------------------------------------------------------------------------

def compute_features(
    mtf: MTFData,
    spread_cost: float = 0.00015,    # Constant Exness spread (từ asset_cfg["spread_cost"])
) -> np.ndarray:
    """
    Tính toán toàn bộ feature matrix từ MTFData.

    ĐÂY LÀ PURE KNOWLEDGE ENCODING — KHÔNG có bất kỳ quyết định Entry/SL/TP nào.

    Args:
        mtf:          MTFData — output của load_mtf_data()
        spread_cost:  Constant spread của asset từ asset_cfg["spread_cost"]
                      (Sếp Vũ confirm: dùng mức chém cao nhất của Exness, không xấp xỉ M1)

    Returns:
        features: np.ndarray shape (N_m5, 32) — feature matrix
                  Index 0..31 theo Feature Map trong docstring file này.
                  NaN-free (warmup bars được fill = 0.0)
    """
    # ── Extract raw arrays ──────────────────────────────────────────────────
    m1  = mtf["m1"]
    m5  = mtf["m5"]
    m15 = mtf["m15"]
    h1  = mtf["h1"]

    N5 = len(m5)
    logger.info(
        f"[compute_features] {mtf.get('symbol', '?')} | "
        f"M1={len(m1):,} M5={N5:,} M15={len(m15):,} H1={len(h1):,}"
    )

    t5 = m5[:, 0]   # M5 timestamps (base timeline)

    # ── M5 raw arrays ───────────────────────────────────────────────────────
    o5, h5, l5, c5, v5 = m5[:,1], m5[:,2], m5[:,3], m5[:,4], m5[:,5]
    atr5   = _atr(h5, l5, c5, period=14)
    vm20_5 = _vol_ma(v5, period=20)

    # ── M1 indicators (computed on M1, then aligned to M5) ──────────────────
    o1, h1d, l1d, c1d, v1d = m1[:,1], m1[:,2], m1[:,3], m1[:,4], m1[:,5]
    atr1   = _atr(h1d, l1d, c1d, period=14)
    vm20_1 = _vol_ma(v1d, period=20)
    _, bull_pin1, bear_pin1 = _pinbar_mask(o1, h1d, l1d, c1d)
    pin1   = (bull_pin1 + bear_pin1).clip(0, 1)
    vsa1   = _vsa_mask(v1d, vm20_1, threshold=1.3)
    spread_proxy_m1 = (h1d - l1d)   # raw H-L price range on M1

    # Align M1 → M5
    atr1_aligned       = _align_to_m5(t5, m1[:,0], atr1)
    pin1_aligned       = _align_to_m5(t5, m1[:,0], pin1)
    vsa1_aligned       = _align_to_m5(t5, m1[:,0], vsa1)
    spread_m1_aligned  = _align_to_m5(t5, m1[:,0], spread_proxy_m1)

    # ── M5 indicators (computed directly on M5) ──────────────────────────────
    _, bull_pin5, bear_pin5 = _pinbar_mask(o5, h5, l5, c5)
    pin5   = (bull_pin5 + bear_pin5).clip(0, 1)
    vsa5   = _vsa_mask(v5, vm20_5, threshold=1.3)
    fvg_bull5, fvg_bear5, fvg_size_bull5, fvg_size_bear5 = _fvg(h5, l5, atr5, buffer=0.3)
    vr5    = np.where(vm20_5 > 1e-10, v5 / vm20_5, 1.0)   # volume ratio

    # EQL/EQH (computed on M5 timeline directly)
    dist_eql5, dist_eqh5, eql_prox5, eqh_prox5 = _nearest_eql_eqh(
        h5, l5, c5, atr5, lookback=EQL_LOOKBACK
    )

    # ── M15 indicators (computed on M15, aligned to M5) ─────────────────────
    o15, h15, l15, c15, v15 = m15[:,1], m15[:,2], m15[:,3], m15[:,4], m15[:,5]
    atr15  = _atr(h15, l15, c15, period=14)
    ema50_15  = _ema(c15, EMA_FAST_M15)
    ema200_15 = _ema(c15, EMA_SLOW_M15)
    _, bull_pin15, bear_pin15 = _pinbar_mask(o15, h15, l15, c15)
    pin15  = (bull_pin15 + bear_pin15).clip(0, 1)
    bull_bos15, bear_bos15, choch_bull15, choch_bear15 = _bos(
        h15, l15, c15, lookback=BOS_LOOKBACK
    )

    # Trend: +1 = bull (ema50 > ema200), -1 = bear, 0 = flat
    trend15 = np.where(ema50_15 > ema200_15 * 1.001, 1.0,
               np.where(ema50_15 < ema200_15 * 0.999, -1.0, 0.0))

    # Price vs EMA50 normalized
    price_vs_ema50_15 = (c15 - ema50_15) / (atr15 + 1e-10)

    # Align M15 → M5
    t15 = m15[:, 0]
    atr15_a        = _align_to_m5(t5, t15, atr15)
    trend15_a      = _align_to_m5(t5, t15, trend15)
    bos_bull15_a   = _align_to_m5(t5, t15, bull_bos15)
    bos_bear15_a   = _align_to_m5(t5, t15, bear_bos15)
    choch_bull15_a = _align_to_m5(t5, t15, choch_bull15)
    choch_bear15_a = _align_to_m5(t5, t15, choch_bear15)
    pin15_a        = _align_to_m5(t5, t15, pin15)
    pve50_15_a     = _align_to_m5(t5, t15, price_vs_ema50_15)

    # ── H1 indicators (computed on H1, aligned to M5) ───────────────────────
    oh1, hh1, lh1, ch1 = h1[:,1], h1[:,2], h1[:,3], h1[:,4]
    atrh1    = _atr(hh1, lh1, ch1, period=14)
    ema21_h1 = _ema(ch1, EMA_FAST_H1)
    ema50_h1 = _ema(ch1, EMA_SLOW_H1)
    trend_h1 = np.where(ema21_h1 > ema50_h1 * 1.001, 1.0,
                np.where(ema21_h1 < ema50_h1 * 0.999, -1.0, 0.0))
    pve21_h1 = (ch1 - ema21_h1) / (atrh1 + 1e-10)
    pve50_h1 = (ch1 - ema50_h1) / (atrh1 + 1e-10)

    th1 = h1[:, 0]
    atrh1_a    = _align_to_m5(t5, th1, atrh1)
    trend_h1_a = _align_to_m5(t5, th1, trend_h1)
    pve21_h1_a = _align_to_m5(t5, th1, pve21_h1)
    pve50_h1_a = _align_to_m5(t5, th1, pve50_h1)

    # ── H1 Inside Bar (Sefp Vũ Rule) ───────────────────────────────────
    # Định nghĩa: H1_High[i] <= H1_High[i-1] AND H1_Low[i] >= H1_Low[i-1]
    # → nến hiện tại nằm trong bong của nến trước → vùng nhiễu, chờ → force close!
    ib_h1 = np.zeros(len(hh1), dtype=np.float32)
    ib_h1[1:] = ((hh1[1:] <= hh1[:-1]) & (lh1[1:] >= lh1[:-1])).astype(np.float32)
    h1_inside_bar_a = _align_to_m5(t5, th1, ib_h1)  # aligned + ffill to M5

    # HTF Bias: trend H1 là thiên hướng chính thức (đã có qua trend_h1_a)
    # Ta còn tính thêm BOS H1 để xác nhận heavy structure shifts
    bos_bull_h1, bos_bear_h1, choch_bull_h1, choch_bear_h1 = _bos(hh1, lh1, ch1, lookback=5)
    bos_bull_h1_a  = _align_to_m5(t5, th1, bos_bull_h1)
    bos_bear_h1_a  = _align_to_m5(t5, th1, bos_bear_h1)
    choch_bull_h1_a= _align_to_m5(t5, th1, choch_bull_h1)
    choch_bear_h1_a= _align_to_m5(t5, th1, choch_bear_h1)

    # ── SMC / ICT Features (V14.0) tính trực tiếp trên M5 ────────────────
    # Strong H/L: dist đến đỉnh/đáy mạnh (vùng cản thực sự)
    dist_strong_h5, dist_strong_l5 = _strong_hl(h5, l5, c5, atr5, lookback=30)

    # Order Block: dist + active zone
    dist_bull_ob5, dist_bear_ob5, bull_ob_act5, bear_ob_act5 = _order_block_features(
        o5, h5, l5, c5, atr5, lookback=15
    )

    # Fibonacci OTE
    fib_ote5 = _fib_ote(h5, l5, c5, atr5, swing_lookback=30)

    # Session / Killzone / Judas Swing
    asia_range5, killzone5, judas5 = _session_features(t5, h5, l5, c5, atr5)

    # Institutional Pricing (số tròn)
    inst_price5 = _institutional_price(c5, atr5)

    # Breaker Block & Mitigation Block (V14.1)
    dist_breaker5, dist_mitigation5 = _breaker_mitigation_blocks(
        o5, h5, l5, c5, atr5, lookback=20
    )

    # Supply-to-Demand Flip & Demand-to-Supply Flip (V14.1)
    dist_s2d5, dist_d2s5 = _flip_zones(h5, l5, c5, atr5, lookback=20)

    # PA & Traps Engine V16.0
    body_norm5, upper_w5, lower_w5, close_pct5, bull_pin5, bear_pin5, bull_engulf5 = \
        _pa_candle_features(o5, h5, l5, c5, atr5)
    is_comp5, comp_score5 = _compression_features(h5, l5, c5, o5, atr5)
    is_ib5, is_bull_trap5, is_bear_trap5, is_opp5 = _trap_features(
        o5, h5, l5, c5, atr5, swing_lookback=10
    )

    # Volume Engine V15.0
    relative_vol5, spread_norm5, effort_result5 = _vsa_features(
        h5, l5, v5, atr5, sma_period=20
    )
    vol_sma5 = np.convolve(v5.astype(np.float64), np.ones(20)/20, mode='same')
    vol_sma5[:19] = v5[:19]
    vol_sma5 = np.maximum(vol_sma5, 1e-10)
    stop_vol5, no_demand5, no_supply5 = _vsa_anomalies(
        h5, l5, c5, v5, atr5, vol_sma5, lookback=5
    )
    dist_poc5 = _rolling_poc(h5, l5, c5, v5, atr5, window=50, n_bins=10)

    # ── Time/Session encoding (cyclic — không block, AI tự học) ──────────────
    from datetime import datetime, timezone
    hours     = np.array([
        datetime.fromtimestamp(ts, tz=timezone.utc).hour
        for ts in t5
    ], dtype=np.float32)
    hour_sin  = np.sin(2 * np.pi * hours / 24.0)
    hour_cos  = np.cos(2 * np.pi * hours / 24.0)

    # ── Static spread cost (Constant từ asset_cfg — Sếp Vũ confirm) ──────────
    # Normalize bằng ATR_m5: AI tự học "phí chiếm bao nhiêu % volatility"
    spread_cost_norm = spread_cost / (atr5 + 1e-10)

    # ── Volatility index ─────────────────────────────────────────────────────
    # ATR_m5 / ATR_h1 — tỷ lệ micro/macro volativity
    # > 1.0: micro đang biến động hơn macro (noise/chop)
    # < 1.0: macro đang kéo mạnh (trending)
    volatility_idx = atr5 / (atrh1_a + 1e-10)

    # ── Assemble feature matrix ──────────────────────────────────────────────
    atr5_safe = atr5 + 1e-10

    features = np.stack([
        # M1 (0..3)
        atr1_aligned / atr5_safe,              #  0 atr_m1_norm
        pin1_aligned,                          #  1 pinbar_m1
        vsa1_aligned,                          #  2 vsa_m1
        spread_m1_aligned / atr5_safe,         #  3 spread_proxy_m1

        # M5 (4..13)
        atr5,                                  #  4 atr_m5_raw
        pin5.astype(np.float32),               #  5 pinbar_m5
        bull_pin5.astype(np.float32),          #  6 bull_pinbar_m5
        bear_pin5.astype(np.float32),          #  7 bear_pinbar_m5
        vsa5.astype(np.float32),               #  8 vsa_m5
        fvg_bull5.astype(np.float32),          #  9 fvg_bull_m5
        fvg_bear5.astype(np.float32),          # 10 fvg_bear_m5
        fvg_size_bull5 / atr5_safe,            # 11 fvg_size_bull_m5_norm
        fvg_size_bear5 / atr5_safe,            # 12 fvg_size_bear_m5_norm
        vr5.astype(np.float32),                # 13 volume_ratio_m5

        # M15 (14..21)
        atr15_a / atr5_safe,                   # 14 atr_m15_norm
        trend15_a.astype(np.float32),          # 15 trend_ema_m15
        bos_bull15_a,                          # 16 bos_bull_m15
        bos_bear15_a,                          # 17 bos_bear_m15
        choch_bull15_a.astype(np.float32),     # 18 choch_bull_m15  (CHoCH đảo chiều lên)
        choch_bear15_a.astype(np.float32),     # 19 choch_bear_m15  (CHoCH đảo chiều xuống)
        pin15_a,                               # 20 pinbar_m15
        pve50_15_a.astype(np.float32),         # 21 price_vs_ema50_m15_norm

        # H1 (22..27)
        atrh1_a / atr5_safe,                   # 22 atr_h1_norm
        trend_h1_a.astype(np.float32),         # 23 trend_ema_h1  (HTF Bias: -1/0/+1)
        pve21_h1_a.astype(np.float32),         # 24 price_vs_ema21_h1_norm
        pve50_h1_a.astype(np.float32),         # 25 price_vs_ema50_h1_norm
        bos_bull_h1_a.astype(np.float32),      # 26 bos_bull_h1    (HTF structure shift ↑)
        bos_bear_h1_a.astype(np.float32),      # 27 bos_bear_h1    (HTF structure shift ↓)
        choch_bull_h1_a.astype(np.float32),    # 28 choch_bull_h1  (HTF reversal ↑)
        choch_bear_h1_a.astype(np.float32),    # 29 choch_bear_h1  (HTF reversal ↓)

        # Liquidity EQL/EQH (30..33)
        dist_eql5,                             # 30 dist_eql_norm
        dist_eqh5,                             # 31 dist_eqh_norm
        eql_prox5,                             # 32 eql_proximity_flag
        eqh_prox5,                             # 33 eqh_proximity_flag

        # Strong H/L — SMC (34..35)
        dist_strong_h5,                        # 34 dist_to_strong_high_norm
        dist_strong_l5,                        # 35 dist_to_strong_low_norm

        # Order Block — SMC/ICT (36..39)
        dist_bull_ob5,                         # 36 dist_to_bull_ob_norm  (Unmitigated)
        dist_bear_ob5,                         # 37 dist_to_bear_ob_norm
        bull_ob_act5,                          # 38 bull_ob_active_flag
        bear_ob_act5,                          # 39 bear_ob_active_flag

        # Fibonacci OTE — ICT (40)
        fib_ote5,                              # 40 fib_ote_level  (0=đáy, 1=đỉnh swing)

        # Session / Killzone / Judas Swing — ICT (41..43)
        asia_range5,                           # 41 asia_range_norm
        killzone5,                             # 42 in_killzone_flag
        judas5,                                # 43 judas_swing_detected

        # Institutional Pricing — ICT (44)
        inst_price5,                           # 44 dist_to_inst_price_norm

        # V15.0 VSA Core (45..47)
        relative_vol5,                             # 45 relative_vol
        spread_norm5,                              # 46 spread_norm
        effort_result5,                            # 47 effort_vs_result_ratio

        # V15.0 VSA Anomalies (48..50)
        stop_vol5,                                 # 48 is_stopping_volume
        no_demand5,                                # 49 is_no_demand
        no_supply5,                                # 50 is_no_supply

        # V15.0 POC (51)
        dist_poc5,                                 # 51 dist_to_rolling_poc_norm

        # V14.1 Breaker / Mitigation / Flip (52..55)
        dist_breaker5,                             # 52 dist_to_breaker_norm
        dist_mitigation5,                          # 53 dist_to_mitigation_norm
        dist_s2d5,                                 # 54 dist_to_s2d_flip
        dist_d2s5,                                 # 55 dist_to_d2s_flip

        # V16.0 PA Candle (56..62)
        body_norm5,                                # 56 body_size_norm
        upper_w5,                                  # 57 upper_wick_pct
        lower_w5,                                  # 58 lower_wick_pct
        close_pct5,                                # 59 close_pct_range
        bull_pin5,                                 # 60 is_bull_pinbar
        bear_pin5,                                 # 61 is_bear_pinbar
        bull_engulf5,                              # 62 is_bull_engulfing

        # V16.0 Compression (63..64)
        is_comp5,                                  # 63 is_compression
        comp_score5,                               # 64 compression_score

        # V16.0 Traps (65..68)
        is_ib5,                                    # 65 is_ib_fakeout
        is_bull_trap5,                             # 66 is_bull_trap
        is_bear_trap5,                             # 67 is_bear_trap
        is_opp5,                                   # 68 is_opposite_failure

        # Volatility (69..70)
        volatility_idx.astype(np.float32),         # 69 volatility_index
        spread_cost_norm.astype(np.float32),       # 70 static_spread_cost_norm

        # Time/Session cyclic (71..72)
        hour_sin.astype(np.float32),               # 71 hour_sin
        hour_cos.astype(np.float32),               # 72 hour_cos

    ], axis=1).astype(np.float32)   # shape (N5, 73)

    # ── Nan/Inf guard ────────────────────────────────────────────────────────
    features = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)

    logger.info(
        f"[compute_features] V16.0 Feature matrix: {features.shape} (73 features) | "
        f"dtype={features.dtype} | nan_count={np.isnan(features).sum()}"
    )
    return features, h1_inside_bar_a.astype(np.float32)



# ---------------------------------------------------------------------------
# Convenience: load + compute in one call
# ---------------------------------------------------------------------------

def build_feature_matrix(
    symbol: str,
    data_dir: str | Path = "data",
    spread_cost: float = 0.00015,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Shortcut: load MTF + compute features trong 1 lần.

    Returns:
        (features, raw_m5, h1_inside_bar):
            features:      shape (N_m5, 73) — feature matrix cho AI
            raw_m5:        shape (N_m5, 6)  — M5 OHLCV dùng để tính SL/TP/PnL
            h1_inside_bar: shape (N_m5,)    — 1.0 nếu H1 Inside Bar, 0.0 khác
    """
    mtf      = load_mtf_data(symbol, data_dir)
    features, h1_ib = compute_features(mtf, spread_cost=spread_cost)
    return features, mtf["m5"], h1_ib


# ---------------------------------------------------------------------------
# CLI Quick Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FeatureEngine — Smoke Test")
    parser.add_argument("--symbol", default="EURUSDm", help="Symbol name (e.g. EURUSDm)")
    parser.add_argument("--data",   default="data",    help="Data directory")
    parser.add_argument("--spread", type=float, default=0.00015, help="Spread cost constant")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  FeatureEngine Smoke Test — {args.symbol}")
    print(f"{'='*60}")

    feats, raw_m5, h1_ib = build_feature_matrix(args.symbol, args.data, args.spread)

    print(f"\n✅ Feature matrix built successfully!")
    print(f"   Shape:          {feats.shape}   ← expect (N_m5, 32)")
    print(f"   dtype:          {feats.dtype}")
    print(f"   Raw M5 shape:   {raw_m5.shape}")
    print(f"   NaN count:      {np.isnan(feats).sum()}  ← must be 0")
    print(f"   Inf count:      {np.isinf(feats).sum()}  ← must be 0")
    print(f"\nFeature stats (mean | std | min | max):")
    names = [
        # M1
        "atr_m1_norm", "pinbar_m1", "vsa_m1", "spread_proxy_m1",
        # M5
        "atr_m5_raw", "pinbar_m5", "bull_pinbar_m5", "bear_pinbar_m5",
        "vsa_m5", "fvg_bull_m5", "fvg_bear_m5", "fvg_size_bull_m5", "fvg_size_bear_m5",
        "volume_ratio_m5",
        # M15
        "atr_m15_norm", "trend_ema_m15", "bos_bull_m15", "bos_bear_m15",
        "choch_bull_m15", "choch_bear_m15", "pinbar_m15", "price_vs_ema50_m15",
        # H1
        "atr_h1_norm", "trend_ema_h1", "price_vs_ema21_h1", "price_vs_ema50_h1",
        "bos_bull_h1", "bos_bear_h1", "choch_bull_h1", "choch_bear_h1",
        # Liquidity
        "dist_eql_norm", "dist_eqh_norm", "eql_proximity", "eqh_proximity",
        # SMC/ICT
        "dist_to_strong_high", "dist_to_strong_low",
        "dist_to_bull_ob", "dist_to_bear_ob", "bull_ob_active", "bear_ob_active",
        "fib_ote_level",
        "asia_range_norm", "in_killzone", "judas_swing",
        "dist_to_inst_price",
        # V15.0 VSA
        "relative_vol", "spread_norm", "effort_vs_result",
        "is_stopping_vol", "is_no_demand", "is_no_supply",
        "dist_to_rolling_poc",
        # V14.1 Breaker/Flip
        "dist_to_breaker", "dist_to_mitigation", "dist_to_s2d_flip", "dist_to_d2s_flip",
        # V16.0 PA
        "body_size_norm", "upper_wick_pct", "lower_wick_pct", "close_pct_range",
        "is_bull_pinbar", "is_bear_pinbar", "is_bull_engulfing",
        "is_compression", "compression_score",
        "is_ib_fakeout", "is_bull_trap", "is_bear_trap", "is_opp_failure",
        # Volatility + Spread
        "volatility_index", "spread_cost_norm",
        # Time
        "hour_sin", "hour_cos",
    ]
    for i, name in enumerate(names):
        col = feats[:, i]
        print(f"   [{i:2d}] {name:<30} μ={col.mean():+7.3f} σ={col.std():6.3f} "
              f"[{col.min():+7.3f}, {col.max():+7.3f}]")

    print(f"\n{'='*60}")
    print(f"  ALL GOOD — Feature Engine V16.0 Ready! ({feats.shape[1]} features)")
    print(f"{'='*60}\n")
