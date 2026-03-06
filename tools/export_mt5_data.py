"""
tools/export_mt5_data.py — MT5 Data Exporter (Windows Only)
=============================================================
Module: Rabit_Exness AI — Microservices Bridge
Author: Antigravity
Date:   2026-03-06

MỤC ĐÍCH:
    Script này CHỈ chạy trên Windows (có cài MetaTrader5).
    Kết nối MT5, tải dữ liệu nến M5 của EURUSDc, lưu ra CSV.
    File CSV sau đó được SCP lên Xeon Linux để ml_model.py và backtest_env.py
    đọc trực tiếp mà KHÔNG cần chạm vào MT5 trên Linux.

LUỒNG:
    Windows MT5 → export_mt5_data.py → data/history_m5.csv
                                              ↓ scp / shared drive
    Xeon Linux → ml_model.py (load CSV) → Optuna 56 workers → best_params
    Xeon Linux → backtest_env.py (load CSV) → BacktestReport → HTML

USAGE:
    python tools/export_mt5_data.py
    python tools/export_mt5_data.py --symbol EURUSDc --days 365 --out data/history_m5.csv
    python tools/export_mt5_data.py --bars 100000     # Theo số nến thay vì số ngày

OUTPUT CSV columns:
    time (Unix int), open, high, low, close, volume

REQUIREMENTS (Windows only):
    pip install MetaTrader5 pandas
"""

from __future__ import annotations

import argparse
import csv
import sys
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Bảo vệ: chỉ import MetaTrader5 trên Windows ──────────────────────────────
try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    mt5 = None  # type: ignore[assignment]
    _MT5_AVAILABLE = False

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    pd = None  # type: ignore[assignment]
    _PANDAS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Config mặc định
# ---------------------------------------------------------------------------

PROJECT_ROOT    = Path(__file__).resolve().parent.parent
DEFAULT_OUT     = str(PROJECT_ROOT / "data" / "history_m5.csv")
DEFAULT_SYMBOL  = "EURUSDc"
DEFAULT_DAYS    = 365     # 1 năm dữ liệu
DEFAULT_BARS    = None    # None = dùng --days; số nguyên = lấy đúng N bars gần nhất

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)-8s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("MT5Exporter")


# ---------------------------------------------------------------------------
# Core: fetch + save
# ---------------------------------------------------------------------------

def _check_prerequisites() -> None:
    """Kiểm tra môi trường trước khi chạy."""
    if not _MT5_AVAILABLE:
        logger.error(
            "❌ MetaTrader5 library không tìm thấy.\n"
            "   Script này phải chạy trên Windows với MT5 đã cài:\n"
            "       pip install MetaTrader5\n"
            "   Nếu đang ở Xeon Linux → đây là lỗi kiến trúc, ĐỪNG chạy!"
        )
        sys.exit(1)

    if sys.platform != "win32":
        logger.error(
            f"❌ Platform hiện tại: '{sys.platform}'.\n"
            "   Script này CHỈ chạy trên Windows (win32).\n"
            "   Xeon Linux sẽ đọc file CSV đã export — KHÔNG chạy script này trên Linux!"
        )
        sys.exit(1)


def fetch_by_range(
    symbol: str,
    days: int,
) -> list[tuple]:
    """
    Fetch M5 rates theo khoảng thời gian (date range).

    Args:
        symbol: Tên symbol MT5, ví dụ 'EURUSDc'
        days:   Số ngày lookback từ hiện tại

    Returns:
        List of tuples: (time_unix, open, high, low, close, volume)
    """
    utc_to   = datetime.now(timezone.utc)
    utc_from = utc_to - timedelta(days=days)

    logger.info(
        f"[Fetch] Symbol={symbol} | Timeframe=M5 | "
        f"Range: {utc_from.date()} → {utc_to.date()} ({days} days)"
    )

    rates = mt5.copy_rates_range(
        symbol,
        mt5.TIMEFRAME_M5,
        utc_from.replace(tzinfo=None),
        utc_to.replace(tzinfo=None),
    )

    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        raise RuntimeError(
            f"MT5 copy_rates_range failed for {symbol}: {err}"
        )

    logger.info(f"[Fetch] ✅ {len(rates):,} candles received")
    return [
        (int(r["time"]), float(r["open"]), float(r["high"]),
         float(r["low"]), float(r["close"]), float(r["tick_volume"]))
        for r in rates
    ]


def fetch_by_bars(
    symbol: str,
    n_bars: int,
) -> list[tuple]:
    """
    Fetch M5 rates theo số nến (N bars gần nhất).

    Args:
        symbol: Tên symbol MT5
        n_bars: Số nến muốn lấy (đếm từ bar mới nhất về quá khứ)

    Returns:
        List of tuples: (time_unix, open, high, low, close, volume)
    """
    logger.info(
        f"[Fetch] Symbol={symbol} | Timeframe=M5 | "
        f"Bars={n_bars:,} (from latest)"
    )

    rates = mt5.copy_rates_from_pos(
        symbol,
        mt5.TIMEFRAME_M5,
        0,       # start_pos=0 → bar mới nhất
        n_bars,
    )

    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        raise RuntimeError(
            f"MT5 copy_rates_from_pos failed for {symbol}: {err}"
        )

    logger.info(f"[Fetch] ✅ {len(rates):,} candles received")
    return [
        (int(r["time"]), float(r["open"]), float(r["high"]),
         float(r["low"]), float(r["close"]), float(r["tick_volume"]))
        for r in rates
    ]


def save_to_csv(rows: list[tuple], out_path: str) -> Path:
    """
    Lưu dữ liệu OHLCV ra CSV.

    CSV Schema:
        time,open,high,low,close,volume
        (time là Unix timestamp — integer seconds UTC)

    Args:
        rows:     List tuples (time, open, high, low, close, volume)
        out_path: Đường dẫn file output

    Returns:
        Path object trỏ tới file đã ghi
    """
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "open", "high", "low", "close", "volume"])
        writer.writerows(rows)

    size_mb = out_file.stat().st_size / 1024 / 1024
    logger.info(
        f"[Save] ✅ CSV saved: {out_file}\n"
        f"         Rows: {len(rows):,} | Size: {size_mb:.2f} MB"
    )
    return out_file


def run_export(
    symbol:  str = DEFAULT_SYMBOL,
    days:    int = DEFAULT_DAYS,
    n_bars:  int | None = None,
    out:     str = DEFAULT_OUT,
    login:   int | None = None,
    password: str | None = None,
    server:  str | None = None,
) -> Path:
    """
    Hàm xuất dữ liệu chính — kết nối MT5, fetch, save CSV.

    Args:
        symbol:   Tên symbol (default: EURUSDc)
        days:     Số ngày lookback (dùng khi n_bars=None)
        n_bars:   Số nến cụ thể (override days nếu được set)
        out:      Đường dẫn file CSV output
        login:    Số tài khoản MT5 (None = dùng account đang đăng nhập)
        password: Password tài khoản MT5
        server:   Tên server (vd: 'Exness-MT5Trial')

    Returns:
        Path object của CSV file đã lưu
    """
    _check_prerequisites()

    # ── Khởi tạo MT5 ────────────────────────────────────────────────────────
    logger.info("[MT5] Initializing MetaTrader5...")

    init_kwargs: dict = {}
    if login and password and server:
        init_kwargs = {"login": login, "password": password, "server": server}

    if not mt5.initialize(**init_kwargs):
        err = mt5.last_error()
        raise RuntimeError(f"MT5 initialize() failed: {err}")

    # In thông tin account đang dùng
    account_info = mt5.account_info()
    if account_info:
        logger.info(
            f"[MT5] Connected | Account: {account_info.login} | "
            f"Server: {account_info.server} | "
            f"Balance: {account_info.balance:.2f} {account_info.currency}"
        )
    else:
        logger.warning("[MT5] account_info() returned None — continuing anyway")

    # Kiểm tra symbol có tồn tại không
    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        mt5.shutdown()
        available = [s.name for s in (mt5.symbols_get() or [])
                     if "EUR" in s.name.upper()][:10]
        raise ValueError(
            f"Symbol '{symbol}' not found in MT5.\n"
            f"  EUR* symbols available: {available}"
        )

    if not sym_info.visible:
        mt5.symbol_select(symbol, True)
        logger.info(f"[MT5] Symbol {symbol} added to MarketWatch")

    # ── Fetch data ──────────────────────────────────────────────────────────
    try:
        if n_bars is not None:
            rows = fetch_by_bars(symbol, n_bars)
        else:
            rows = fetch_by_range(symbol, days)
    finally:
        mt5.shutdown()
        logger.info("[MT5] Shutdown complete")

    # ── Save CSV ────────────────────────────────────────────────────────────
    return save_to_csv(rows, out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "MT5 M5 Data Exporter — Windows Only\n"
            "Export lịch sử nến M5 từ MetaTrader5 ra CSV cho Xeon Linux."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ví dụ:\n"
            "  python tools/export_mt5_data.py\n"
            "  python tools/export_mt5_data.py --symbol EURUSDc --days 365\n"
            "  python tools/export_mt5_data.py --bars 105120     # ~1 năm M5\n"
            "  python tools/export_mt5_data.py --login 12345 --password MyPass --server Exness-MT5Trial\n"
            "\n"
            "Sau khi export, SCP file lên Xeon:\n"
            "  scp data/history_m5.csv xeon-user@xeon-ip:/path/RabitScal/data/"
        ),
    )
    parser.add_argument(
        "--symbol", type=str, default=DEFAULT_SYMBOL,
        help=f"Tên symbol MT5 (default: {DEFAULT_SYMBOL})",
    )
    parser.add_argument(
        "--days", type=int, default=DEFAULT_DAYS,
        help=f"Số ngày lookback (default: {DEFAULT_DAYS}). Bị override bởi --bars.",
    )
    parser.add_argument(
        "--bars", type=int, default=None,
        help="Số nến M5 muốn lấy (tính từ bar mới nhất). Override --days nếu set.",
    )
    parser.add_argument(
        "--out", type=str, default=DEFAULT_OUT,
        help=f"Đường dẫn file CSV output (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--login", type=int, default=None,
        help="Số tài khoản MT5 (bỏ qua nếu dùng account đang login sẵn)",
    )
    parser.add_argument(
        "--password", type=str, default=None,
        help="Password tài khoản MT5",
    )
    parser.add_argument(
        "--server", type=str, default=None,
        help="Tên MT5 server (vd: Exness-MT5Trial)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logger.info("=" * 60)
    logger.info("  RabitScal — MT5 M5 Data Exporter (Windows Only)")
    logger.info("=" * 60)

    try:
        out_file = run_export(
            symbol=args.symbol,
            days=args.days,
            n_bars=args.bars,
            out=args.out,
            login=args.login,
            password=args.password,
            server=args.server,
        )

        print("\n" + "=" * 60)
        print("✅ EXPORT COMPLETE")
        print(f"   Symbol  : {args.symbol}")
        print(f"   CSV     : {out_file}")
        print(f"\n   Tiếp theo — SCP lên Xeon Linux:")
        print(f"   scp {out_file} xeon-user@<xeon-ip>:/path/RabitScal/data/")
        print(f"\n   Rồi chạy trên Xeon:")
        print(f"   python ml_model.py --csv data/history_m5.csv")
        print(f"   python backtest_env.py --csv data/history_m5.csv --params config/current_settings.json")
        print("=" * 60 + "\n")

    except KeyboardInterrupt:
        logger.info("\n[MT5Exporter] Interrupted by user")
        sys.exit(0)
    except Exception as exc:
        logger.error(f"❌ Export failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
