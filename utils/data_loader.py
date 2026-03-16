"""
utils/data_loader.py — Centralized OHLCV Data Loader
======================================================
Module: RabitScal — utils
DRY: Gộp load_ohlcv_from_csv() từ backtest_env.py + CSV load logic từ ml_model.py

Usage:
    from utils.data_loader import load_ohlcv_from_csv
    data = load_ohlcv_from_csv("data/history_m5.csv")
"""

from __future__ import annotations

import csv as _csv
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_CSV_PATH = Path("data") / "history_m5.csv"
logger = logging.getLogger("DataLoader")


def load_ohlcv_from_csv(
    csv_path: str | Path | None = None,
) -> np.ndarray:
    """
    Nạp dữ liệu M5 OHLCV từ file CSV (export từ Windows bằng tools/export_mt5_data.py).
    LINUX-SAFE: Không yêu cầu MetaTrader5.

    CSV Schema (header bắt buộc):
        time,open,high,low,close,volume
        (time là Unix timestamp — int seconds UTC)

    Args:
        csv_path: Đường dẫn tới CSV file.
                  Mặc định: data/history_m5.csv

    Returns:
        np.ndarray shape (N, 6) — dtype float64
        Columns: [time, open, high, low, close, volume]

    Raises:
        FileNotFoundError: nếu file không tồn tại
        ValueError:        nếu CSV sai format hoặc không có dữ liệu hợp lệ
    """
    path = Path(csv_path) if csv_path else DEFAULT_CSV_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"CSV data file not found: {path}\n"
            f"  → Chạy trên Windows: python tools/export_mt5_data.py\n"
            f"  → SCP lên Xeon:       scp data/history_m5.csv xeon:/path/RabitScal/data/"
        )

    required_cols = {"time", "open", "high", "low", "close", "volume"}
    rows: list[list[float]] = []

    with open(path, "r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV file rỗng hoặc thiếu header: {path}")

        missing = required_cols - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"CSV thiếu các cột bắt buộc: {missing}\n"
                f"  Required: {required_cols}"
            )

        for row in reader:
            try:
                rows.append([
                    float(row["time"]),
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row["volume"]),
                ])
            except (ValueError, KeyError):
                continue

    if not rows:
        raise ValueError(f"Không có dữ liệu hợp lệ trong CSV: {path}")

    data = np.array(rows, dtype=np.float64)
    data = data[data[:, 0].argsort()]   # Sort by time ascending

    from_date = datetime.fromtimestamp(data[0, 0], tz=timezone.utc).strftime("%Y-%m-%d")
    to_date   = datetime.fromtimestamp(data[-1, 0], tz=timezone.utc).strftime("%Y-%m-%d")
    logger.info(
        f"[load_ohlcv_from_csv] ✅ {len(data):,} candles | "
        f"from={from_date} to={to_date} | {path}"
    )
    return data
