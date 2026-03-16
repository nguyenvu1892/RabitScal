"""
tools/export_mt5_data.py — MT5 Multi-Timeframe Data Exporter v2.0
==================================================================
Author: Antigravity / RabitScal Team
Date:   2026-03-15

MODES:
  1. Single TF (legacy):   Export 1 symbol, 1 TF (backward-compatible)
  2. Multi-TF batch (new): Export 1+ symbols x 4 TF = N×4 CSV files

OUTPUT FILE NAMING (matches core/feature_engine.load_mtf_data()):
    data/history_{SYMBOL}_{TF}.csv
    Example: data/history_XAUUSD_M1.csv
             data/history_XAUUSD_M5.csv
             data/history_XAUUSD_M15.csv
             data/history_XAUUSD_H1.csv

CSV COLUMNS: time, open, high, low, close, volume
    time = Unix timestamp (integer seconds UTC)

REQUIREMENTS (Windows only — MT5 must be installed and logged in):
    pip install MetaTrader5

USAGE:
    # Multi-TF batch for XAUUSD (4 files):
    python tools/export_mt5_data.py --mtf --symbol XAUUSD --days 365

    # Multi-TF batch for all 5 symbols (20 files):
    python tools/export_mt5_data.py --mtf --days 365

    # Legacy single-TF (backward-compatible):
    python tools/export_mt5_data.py --symbol EURUSDc --days 365 --out data/history_m5.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Platform guard ────────────────────────────────────────────────────────────
try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    mt5 = None  # type: ignore[assignment]
    _MT5_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────────

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DATA_DIR       = PROJECT_ROOT / "data"
DEFAULT_OUT    = str(DATA_DIR / "history_m5.csv")
DEFAULT_SYMBOL = "EURUSDc"
DEFAULT_DAYS   = 365

# Allowed symbols for --mtf mode (Seep Vu mandate — no XAG)
MTF_SYMBOLS = ["XAUUSD", "US30", "USTEC", "BTCUSD", "ETHUSD"]

# Timeframes for --mtf mode
MTF_TIMEFRAMES = {
    "M1":  "TIMEFRAME_M1",
    "M5":  "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "H1":  "TIMEFRAME_H1",
}

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)-8s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("MT5Exporter")


# ── Pre-checks ────────────────────────────────────────────────────────────────

def _check_prerequisites() -> None:
    if not _MT5_AVAILABLE:
        logger.error(
            "MetaTrader5 library not found.\n"
            "  Install: pip install MetaTrader5\n"
            "  This script must run on Windows with MT5 installed."
        )
        sys.exit(1)
    if sys.platform != "win32":
        logger.error(
            f"Platform: '{sys.platform}' — this script requires Windows (win32)."
        )
        sys.exit(1)


def _mt5_init(login=None, password=None, server=None) -> None:
    """Initialize MT5 connection. Call mt5.shutdown() when done."""
    kwargs: dict = {}
    if login and password and server:
        kwargs = {"login": login, "password": password, "server": server}
    if not mt5.initialize(**kwargs):
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
    info = mt5.account_info()
    if info:
        logger.info(
            f"[MT5] Connected | Account: {info.login} | "
            f"Server: {info.server} | Balance: {info.balance:.2f} {info.currency}"
        )


def _ensure_symbol(symbol: str) -> None:
    """Make sure symbol is visible in MarketWatch."""
    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        available = [s.name for s in (mt5.symbols_get() or [])
                     if symbol[:3].upper() in s.name.upper()][:8]
        raise ValueError(
            f"Symbol '{symbol}' not found in MT5.\n"
            f"  Closest matches: {available}"
        )
    if not sym_info.visible:
        mt5.symbol_select(symbol, True)
        logger.info(f"[MT5] Symbol {symbol} added to MarketWatch")


# ── Fetch helpers ─────────────────────────────────────────────────────────────

def fetch_by_range(symbol: str, tf_const, tf_name: str, days: int) -> list[tuple]:
    """Fetch OHLCV for `days` days on the given timeframe."""
    utc_to   = datetime.now(timezone.utc)
    utc_from = utc_to - timedelta(days=days)
    logger.info(
        f"  [{tf_name}] {symbol} | "
        f"{utc_from.strftime('%Y-%m-%d')} → {utc_to.strftime('%Y-%m-%d')}"
    )
    rates = mt5.copy_rates_range(
        symbol, tf_const,
        utc_from.replace(tzinfo=None),
        utc_to.replace(tzinfo=None),
    )
    if rates is None or len(rates) == 0:
        raise RuntimeError(
            f"copy_rates_range failed for {symbol} {tf_name}: {mt5.last_error()}"
        )
    logger.info(f"  [{tf_name}] {len(rates):,} candles received")
    return [
        (int(r["time"]), float(r["open"]), float(r["high"]),
         float(r["low"]),  float(r["close"]), float(r["tick_volume"]))
        for r in rates
    ]


def fetch_by_bars(symbol: str, tf_const, tf_name: str, n_bars: int) -> list[tuple]:
    """Fetch last N bars from the given timeframe."""
    logger.info(f"  [{tf_name}] {symbol} | {n_bars:,} bars from latest")
    rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, n_bars)
    if rates is None or len(rates) == 0:
        raise RuntimeError(
            f"copy_rates_from_pos failed for {symbol} {tf_name}: {mt5.last_error()}"
        )
    logger.info(f"  [{tf_name}] {len(rates):,} candles received")
    return [
        (int(r["time"]), float(r["open"]), float(r["high"]),
         float(r["low"]),  float(r["close"]), float(r["tick_volume"]))
        for r in rates
    ]


# ── Save ──────────────────────────────────────────────────────────────────────

def save_to_csv(rows: list[tuple], out_path: str) -> Path:
    """Save OHLCV rows to CSV with header: time,open,high,low,close,volume"""
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "open", "high", "low", "close", "volume"])
        writer.writerows(rows)
    size_kb = out_file.stat().st_size / 1024
    logger.info(f"  Saved: {out_file.name} ({len(rows):,} rows, {size_kb:.1f} KB)")
    return out_file


# ── Multi-TF Batch Export (NEW) ───────────────────────────────────────────────

def export_mtf_batch(
    symbols: list[str],
    days: int,
    n_bars: int | None,
    data_dir: Path,
    login=None, password=None, server=None,
) -> list[Path]:
    """
    Export M1/M5/M15/H1 for each symbol to data/history_{SYMBOL}_{TF}.csv.

    Naming matches core/feature_engine.load_mtf_data() exactly.

    Returns:
        List of Path objects for all CSV files created.
    """
    _check_prerequisites()
    _mt5_init(login, password, server)

    saved_files: list[Path] = []
    total = len(symbols) * len(MTF_TIMEFRAMES)
    done  = 0

    try:
        for symbol in symbols:
            logger.info(f"\n[{symbol}] ========================================")
            try:
                _ensure_symbol(symbol)
            except ValueError as e:
                logger.warning(f"  SKIP: {e}")
                continue

            for tf_name, tf_attr in MTF_TIMEFRAMES.items():
                tf_const = getattr(mt5, tf_attr)
                out_path = data_dir / f"history_{symbol}_{tf_name}.csv"
                try:
                    if n_bars is not None:
                        rows = fetch_by_bars(symbol, tf_const, tf_name, n_bars)
                    else:
                        rows = fetch_by_range(symbol, tf_const, tf_name, days)
                    saved_files.append(save_to_csv(rows, str(out_path)))
                except Exception as e:
                    logger.error(f"  FAIL [{symbol} {tf_name}]: {e}")

                done += 1
                logger.info(f"  Progress: {done}/{total}")

    finally:
        mt5.shutdown()
        logger.info("\n[MT5] Shutdown.")

    return saved_files


# ── Legacy Single-TF Export ───────────────────────────────────────────────────

def run_export(
    symbol: str = DEFAULT_SYMBOL,
    days: int = DEFAULT_DAYS,
    n_bars: int | None = None,
    out: str = DEFAULT_OUT,
    login=None, password=None, server=None,
) -> Path:
    """Legacy single-TF export (M5 only). Backward-compatible."""
    _check_prerequisites()
    _mt5_init(login, password, server)
    try:
        _ensure_symbol(symbol)
        tf_const = mt5.TIMEFRAME_M5
        if n_bars is not None:
            rows = fetch_by_bars(symbol, tf_const, "M5", n_bars)
        else:
            rows = fetch_by_range(symbol, tf_const, "M5", days)
    finally:
        mt5.shutdown()
        logger.info("[MT5] Shutdown.")
    return save_to_csv(rows, out)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MT5 Multi-Timeframe Data Exporter v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Multi-TF batch (recommended):\n"
            "  python tools/export_mt5_data.py --mtf --symbol XAUUSD --days 365\n"
            "  python tools/export_mt5_data.py --mtf --days 365  # all 5 symbols\n\n"
            "  # Legacy single-TF:\n"
            "  python tools/export_mt5_data.py --symbol EURUSDc --days 365\n"
        ),
    )

    # MTF mode
    parser.add_argument(
        "--mtf", action="store_true",
        help="Multi-TF batch mode: export M1/M5/M15/H1 "
             "as history_{SYMBOL}_{TF}.csv (use with load_mtf_data())",
    )

    # Symbol (for MTF mode: defaults to all 5; for legacy: defaults to EURUSDc)
    parser.add_argument(
        "--symbol", type=str, default=None,
        help=(
            f"Symbol to export (MTF mode default: all 5 symbols {MTF_SYMBOLS}; "
            f"legacy default: {DEFAULT_SYMBOL})"
        ),
    )
    parser.add_argument(
        "--days", type=int, default=DEFAULT_DAYS,
        help=f"Lookback days (default: {DEFAULT_DAYS}). Overridden by --bars.",
    )
    parser.add_argument(
        "--bars", type=int, default=None,
        help="Number of bars to fetch instead of --days.",
    )

    # Legacy output (ignored in MTF mode)
    parser.add_argument(
        "--out", type=str, default=DEFAULT_OUT,
        help=f"Output path (legacy single-TF only, default: {DEFAULT_OUT})",
    )

    # Auth (optional)
    parser.add_argument("--login",    type=int, default=None)
    parser.add_argument("--password", type=str, default=None)
    parser.add_argument("--server",   type=str, default=None)

    return parser.parse_args()


def main() -> None:
    args   = _parse_args()
    width  = 60

    logger.info("=" * width)
    logger.info("  RabitScal — MT5 Data Exporter v2.0")
    logger.info("=" * width)

    try:
        if args.mtf:
            # ─── Multi-TF Batch Mode ───────────────────────────────────────
            symbols = [args.symbol] if args.symbol else MTF_SYMBOLS
            logger.info(f"  Mode:     MULTI-TF BATCH")
            logger.info(f"  Symbols:  {symbols}")
            logger.info(f"  TFs:      {list(MTF_TIMEFRAMES.keys())}")
            logger.info(f"  Days:     {args.days}")
            logger.info(f"  Data dir: {DATA_DIR}")
            logger.info("=" * width)

            saved = export_mtf_batch(
                symbols  = symbols,
                days     = args.days,
                n_bars   = args.bars,
                data_dir = DATA_DIR,
                login    = args.login,
                password = args.password,
                server   = args.server,
            )

            print("\n" + "=" * width)
            print(f"  EXPORT COMPLETE: {len(saved)} files")
            print("=" * width)
            for f in saved:
                print(f"  OK  {f.name}")
            print()
            print("  Next — train XGBoost model:")
            print("  python -c \"")
            print("    from core.feature_engine import load_mtf_data, compute_features")
            print("    from core.xgb_classifier import RabitScalClassifier, build_labels")
            print("    mtf = load_mtf_data('XAUUSD', data_dir='data')")
            print("    X   = compute_features(mtf, spread_cost=0.0)")
            print("    y   = build_labels(mtf['m5'], atr=X[:,4])")
            print("    clf = RabitScalClassifier()")
            print("    clf.fit(X, y, symbol='XAUUSD', use_gpu=True)")
            print("    clf.save('XAUUSD')")
            print("  \"")
            print("=" * width)

        else:
            # ─── Legacy Single-TF Mode ─────────────────────────────────────
            symbol = args.symbol or DEFAULT_SYMBOL
            logger.info(f"  Mode:   LEGACY (M5 only)")
            logger.info(f"  Symbol: {symbol}")
            logger.info(f"  Days:   {args.days}")
            logger.info(f"  Out:    {args.out}")
            logger.info("=" * width)

            out_file = run_export(
                symbol   = symbol,
                days     = args.days,
                n_bars   = args.bars,
                out      = args.out,
                login    = args.login,
                password = args.password,
                server   = args.server,
            )
            print("\n" + "=" * width)
            print(f"  EXPORT COMPLETE")
            print(f"  CSV: {out_file}")
            print("=" * width)

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user.")
        sys.exit(0)
    except Exception as exc:
        logger.error(f"Export failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
