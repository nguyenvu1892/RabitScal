"""
mt5_engine.py — RabitScal AI Trading Engine v3.0
==================================================
Direct MetaTrader5 Python API — No Socket, No EA, No Asyncio.

Architecture:
    Main loop (1s sleep) → detect M5 candle close
        → pull 300 bars × 4 TFs × 5 symbols via mt5.copy_rates_from_pos()
        → FeatureEngine → XGBoost → Signal / AI Dynamic Exit
        → mt5.order_send() direct execution

Author:  Antigravity (Re-architecture: Sếp Vũ 2026-03-17)
"""
from __future__ import annotations

import logging
import sys
import time
import traceback
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import numpy as np

# -- MT5 ------------------------------------------------------------------
try:
    import MetaTrader5 as mt5
except ImportError:
    print("FATAL: MetaTrader5 not installed. Run: pip install MetaTrader5")
    sys.exit(1)

# -- AI Modules -----------------------------------------------------------
from core.feature_engine import compute_features, MTFData, WARMUP_BARS
from core.xgb_classifier import RabitScalClassifier, MIN_CONFIDENCE, get_classifier
from core.signal_engine import (
    ALLOWED_SYMBOLS, LOT_MAP,
    RiskGuard, SLTPCalculator, SL_ATR_MULT, TP_ATR_MULT,
    validate_lot_with_broker, check_spread,
    process_ai_signal,
)
from core.position_tracker import PositionTracker

# -- Config ----------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
LOGS_DIR     = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

ELITE_5 = ["XAUUSD", "US30", "USTEC", "BTCUSD", "ETHUSD"]
TF_MAP  = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1":  mt5.TIMEFRAME_H1,
}
CANDLE_COUNT = 300
MAGIC_NUMBER = 202603

# -- Logger ----------------------------------------------------------------

def _build_logger() -> logging.Logger:
    logger = logging.getLogger("MT5Engine")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)-7s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = RotatingFileHandler(
        LOGS_DIR / "mt5_engine.log", maxBytes=10 * 1024 * 1024, backupCount=5,
    )
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

log = _build_logger()





# =============================================================================
#  MT5 Helper Functions
# =============================================================================

def resolve_symbol(symbol: str) -> str:
    """Try bare name, then with 'm' suffix (Exness convention)."""
    info = mt5.symbol_info(symbol)
    if info is not None:
        return symbol
    alt = symbol + "m"
    info = mt5.symbol_info(alt)
    if info is not None:
        log.info(f"[Symbol] {symbol} -> {alt} (Exness suffix)")
        return alt
    log.warning(f"[Symbol] {symbol} not found on broker -- skipping")
    return ""


def pull_candles(broker_symbol: str, tf_name: str, count: int = CANDLE_COUNT) -> Optional[np.ndarray]:
    """
    Pull OHLCV from MT5 → numpy array shape (N, 6): [time, O, H, L, C, V].
    Returns None if MT5 returns no data.
    """
    tf_const = TF_MAP.get(tf_name)
    if tf_const is None:
        return None

    rates = mt5.copy_rates_from_pos(broker_symbol, tf_const, 0, count)
    if rates is None or len(rates) == 0:
        return None

    arr = np.column_stack([
        rates["time"].astype(np.float64),
        rates["open"].astype(np.float64),
        rates["high"].astype(np.float64),
        rates["low"].astype(np.float64),
        rates["close"].astype(np.float64),
        rates["tick_volume"].astype(np.float64),
    ])
    return arr


def get_filling_mode(broker_sym: str) -> int:
    """Auto-detect supported filling mode for a symbol (fixes retcode=10030)."""
    info = mt5.symbol_info(broker_sym)
    if info is None:
        return mt5.ORDER_FILLING_IOC
    modes = info.filling_mode
    # Try FOK first (most brokers), then IOC, then RETURN
    if modes & mt5.SYMBOL_FILLING_FOK:
        return mt5.ORDER_FILLING_FOK
    elif modes & mt5.SYMBOL_FILLING_IOC:
        return mt5.ORDER_FILLING_IOC
    else:
        return mt5.ORDER_FILLING_RETURN


# =============================================================================
#  Order Execution (direct mt5.order_send)
# =============================================================================

def mt5_open_position(
    broker_sym: str,
    clean_sym:  str,
    action:     str,
    lot:        float,
    sl:         float,
    tp:         float,
    comment:    str = "RabitScal_AI",
) -> Optional[int]:
    """Open a position. Returns ticket or None."""
    order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
    price = mt5.symbol_info_tick(broker_sym)
    if price is None:
        log.error(f"[ORDER] No tick data for {broker_sym}")
        return None

    fill_price = price.ask if action == "BUY" else price.bid

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       broker_sym,
        "volume":       lot,
        "type":         order_type,
        "price":        fill_price,
        "sl":           sl,
        "tp":           tp,
        "deviation":    30,
        "magic":        MAGIC_NUMBER,
        "comment":      comment,
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": get_filling_mode(broker_sym),
    }

    result = mt5.order_send(request)
    if result is None:
        log.error(f"[ORDER] order_send returned None for {broker_sym}")
        return None

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        log.info(
            f"[ORDER] OK: {action} {clean_sym} lot={lot} @ {result.price} "
            f"ticket={result.order} SL={sl} TP={tp}"
        )
        return result.order
    else:
        log.error(
            f"[ORDER] FAIL: {action} {clean_sym} FAILED: "
            f"retcode={result.retcode} comment={result.comment}"
        )
        return None


def mt5_close_position(ticket: int, reason: str = "AI_EXIT") -> Optional[float]:
    """Close an entire position by ticket. Returns close_price or None."""
    pos = mt5.positions_get(ticket=ticket)
    if pos is None or len(pos) == 0:
        log.warning(f"[CLOSE] Ticket {ticket} not found -- already closed?")
        return None

    p = pos[0]
    close_type = mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    tick = mt5.symbol_info_tick(p.symbol)
    if tick is None:
        log.error(f"[CLOSE] No tick for {p.symbol}")
        return None

    close_price = tick.bid if p.type == mt5.ORDER_TYPE_BUY else tick.ask

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       p.symbol,
        "volume":       p.volume,
        "type":         close_type,
        "position":     ticket,
        "price":        close_price,
        "deviation":    30,
        "magic":        MAGIC_NUMBER,
        "comment":      f"RabitScal_{reason}",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": get_filling_mode(p.symbol),
    }

    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        log.info(f"[CLOSE] OK: ticket={ticket} {p.symbol} vol={p.volume} @ {close_price} | {reason}")
        return close_price
    else:
        rc = result.retcode if result else "None"
        log.error(f"[CLOSE] FAIL: ticket={ticket} failed: retcode={rc}")
        return None


def mt5_partial_close(ticket: int, close_vol: float, broker_sym: str) -> bool:
    """Close part of a position. Auto-fallback to full close if remaining < min_lot."""
    pos = mt5.positions_get(ticket=ticket)
    if pos is None or len(pos) == 0:
        log.warning(f"[PARTIAL] Ticket {ticket} not found")
        return False

    p = pos[0]
    info = mt5.symbol_info(broker_sym)
    if info is None:
        log.error(f"[PARTIAL] symbol_info({broker_sym}) returned None")
        return False

    min_lot  = info.volume_min
    lot_step = info.volume_step

    # Round close_vol to lot step
    if lot_step > 0:
        close_vol = int(close_vol / lot_step) * lot_step
        close_vol = round(close_vol, 8)

    remaining = round(p.volume - close_vol, 8)

    # Fallback: if remaining < min_lot → full close
    if remaining < min_lot:
        log.warning(f"[PARTIAL] remaining={remaining} < min={min_lot} -> FULL CLOSE")
        return mt5_close_position(ticket, "SCALE_OUT_FULL_FALLBACK")

    if close_vol < min_lot:
        log.warning(f"[PARTIAL] close_vol={close_vol} < min={min_lot} -> REJECTED")
        return False

    close_type = mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    tick = mt5.symbol_info_tick(broker_sym)
    if tick is None:
        return False

    close_price = tick.bid if p.type == mt5.ORDER_TYPE_BUY else tick.ask

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       broker_sym,
        "volume":       close_vol,
        "type":         close_type,
        "position":     ticket,
        "price":        close_price,
        "deviation":    30,
        "magic":        MAGIC_NUMBER,
        "comment":      "RabitScal_SCALE_OUT",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": get_filling_mode(broker_sym),
    }

    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        log.info(f"[PARTIAL] OK: ticket={ticket} closed {close_vol} lot, was {p.volume}")
        return True
    else:
        rc = result.retcode if result else "None"
        log.error(f"[PARTIAL] FAIL: ticket={ticket} failed: retcode={rc}")
        return False


def mt5_modify_sl(ticket: int, new_sl: float) -> bool:
    """Move SL to break-even (or any new level)."""
    pos = mt5.positions_get(ticket=ticket)
    if pos is None or len(pos) == 0:
        return False
    p = pos[0]

    request = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "symbol":   p.symbol,
        "position": ticket,
        "sl":       new_sl,
        "tp":       p.tp,
        "magic":    MAGIC_NUMBER,
    }

    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        log.info(f"[MODIFY] OK: ticket={ticket} SL->{new_sl}")
        return True
    else:
        rc = result.retcode if result else "None"
        log.error(f"[MODIFY] FAIL: ticket={ticket} SL modify failed: retcode={rc}")
        return False


# =============================================================================
#  Position Sync — reconcile tracker with broker reality
# =============================================================================

def sync_positions(tracker: PositionTracker, symbol_map: dict[str, str]):
    """
    Sync PositionTracker with actual MT5 positions.
    Handles manual closes, MT5 restarts, etc.
    """
    broker_positions = mt5.positions_get()
    if broker_positions is None:
        broker_positions = []

    our_tickets = set()
    for p in broker_positions:
        if p.magic == MAGIC_NUMBER:
            our_tickets.add(p.ticket)
            clean_sym = p.symbol.rstrip("m") if p.symbol.endswith("m") and len(p.symbol) > 3 else p.symbol
            direction = "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL"

            if not tracker.has_position(clean_sym):
                tracker.on_fill(
                    ticket=p.ticket,
                    symbol=clean_sym,
                    direction=direction,
                    lot=p.volume,
                    entry_price=p.price_open,
                )
                log.info(f"[SYNC] Registered existing position: {clean_sym} {direction} ticket={p.ticket}")

    tracked_symbols = list(tracker._symbol_index.keys())
    for sym in tracked_symbols:
        ticket = tracker._symbol_index.get(sym)
        if ticket and ticket not in our_tickets:
            tracker.on_close(ticket)
            log.info(f"[SYNC] Removed stale position: {sym} ticket={ticket}")


# =============================================================================
#  AI Pipeline — runs every M5 candle
# =============================================================================

def run_ai_for_symbol(
    clean_sym:  str,
    broker_sym: str,
    tracker:    PositionTracker,
) -> None:
    """Full AI pipeline for one symbol. Called every M5 candle close."""
    t0 = time.time()

    # 1. Pull 4-TF candle data
    mtf_arrays = {}
    for tf_name, tf_attr in [("M1", "m1"), ("M5", "m5"), ("M15", "m15"), ("H1", "h1")]:
        arr = pull_candles(broker_sym, tf_name, CANDLE_COUNT)
        if arr is None or len(arr) < 50:
            log.debug(f"[AI] {clean_sym}: not enough {tf_name} bars -- skipping")
            return
        mtf_arrays[tf_attr] = arr

    # 2. Build MTFData dict
    mtf: MTFData = {
        "m1":     mtf_arrays["m1"],
        "m5":     mtf_arrays["m5"],
        "m15":    mtf_arrays["m15"],
        "h1":     mtf_arrays["h1"],
        "symbol": clean_sym,
    }

    # 3. Compute features
    X_mat, _h1_bar = compute_features(mtf, spread_cost=0.0)
    if X_mat is None or len(X_mat) <= WARMUP_BARS:
        log.debug(f"[AI] {clean_sym}: insufficient feature rows -- skipping")
        return

    # 4. XGBoost predict — last row only
    last_feat = X_mat[-1:].astype(np.float32)
    clf = get_classifier()
    result = clf.predict_single(last_feat, clean_sym)

    action     = result["action"]
    confidence = result["confidence"]

    atr_m5        = float(last_feat[0, 4]) if last_feat.shape[1] > 4 else 1.0
    current_price = float(mtf_arrays["m5"][-1, 4])

    elapsed_ms = (time.time() - t0) * 1000
    log.info(
        f"[AI] {clean_sym} M5: {action} conf={confidence:.2f} "
        f"in {elapsed_ms:.1f}ms"
    )

    # === 4a: AI SMART EXIT v2.1 -- check open positions first ===
    if tracker.has_position(clean_sym):
        # Get broker volume_min for smart scale-out decision
        sym_info = mt5.symbol_info(broker_sym)
        broker_vol_min = sym_info.volume_min if sym_info else 0.01

        exit_signal = tracker.evaluate_exit(
            symbol        = clean_sym,
            ai_result     = result,
            current_price = current_price,
            atr           = atr_m5,
            broker_vol_min = broker_vol_min,
        )
        if exit_signal is not None:
            exit_action = exit_signal["action"]
            ticket      = exit_signal["ticket"]
            pos_info    = tracker.get_position(clean_sym)

            if exit_action == "MODIFY_SL":
                # L1: Fast break-even -- just move SL, keep position open
                new_sl = exit_signal["new_sl"]
                mt5_modify_sl(ticket, new_sl)
                return

            elif exit_action == "CLOSE":
                # L3/L4/L5: Close full position
                reason = exit_signal.get("exit", "AI_EXIT")
                close_price = mt5_close_position(ticket, reason)
                if close_price is not None and pos_info:
                    # Calculate PnL
                    if pos_info.direction == "BUY":
                        pnl = (close_price - pos_info.entry_price) * pos_info.lot
                    else:
                        pnl = (pos_info.entry_price - close_price) * pos_info.lot
                    # Rough USD conversion (tick_value approximation)
                    tick_val = sym_info.trade_tick_value if sym_info else 1.0
                    tick_size = sym_info.trade_tick_size if sym_info else 0.00001
                    if tick_size > 0:
                        pnl_usd = pnl / tick_size * tick_val
                    else:
                        pnl_usd = pnl
                    tracker.on_close(ticket, close_price, reason, pnl_usd)
                elif close_price is not None:
                    tracker.on_close(ticket, close_price, reason)
                return

            elif exit_action == "CLOSE_PARTIAL":
                # L2: Smart scale-out
                close_vol  = exit_signal["volume"]
                remaining  = exit_signal["remaining"]
                new_sl     = exit_signal["new_sl"]

                val_lot, val_reason = validate_lot_with_broker(clean_sym, remaining)
                if val_lot is None:
                    log.warning(
                        f"[EXIT] {clean_sym} remaining={remaining} "
                        f"< broker min -> FULL CLOSE fallback"
                    )
                    close_price = mt5_close_position(ticket, "SCALE_OUT_FULL_FALLBACK")
                    if close_price is not None:
                        tracker.on_close(ticket, close_price, "SCALE_OUT_FULL_FALLBACK")
                else:
                    if mt5_partial_close(ticket, close_vol, broker_sym):
                        tracker.on_partial_close(ticket, val_lot)
                        mt5_modify_sl(ticket, new_sl)
                        log.info(
                            f"[EXIT] {clean_sym} SCALE-OUT: closed {close_vol} lot, "
                            f"remaining={val_lot}, SL->{new_sl}"
                        )
                return

    # === 4b: Entry signal (per-symbol independent -- no global lock) ===
    if action == "HOLD":
        return

    if tracker.has_position(clean_sym):
        log.debug(f"[AI] {clean_sym}: already have open position -- skip entry")
        return

    # 5a. Spread Filter
    tick = mt5.symbol_info_tick(broker_sym)
    if tick is None:
        return
    sym_info = mt5.symbol_info(broker_sym)
    if sym_info is None or sym_info.point == 0:
        return
    spread_points = int(round((tick.ask - tick.bid) / sym_info.point))
    sp_ok, sp_reason = check_spread(clean_sym, abs(spread_points))
    if not sp_ok:
        log.warning(f"[SpreadFilter] {action} {clean_sym} BLOCKED | {sp_reason}")
        return
    log.info(f"[SpreadFilter] {action} {clean_sym} PASS | {sp_reason}")

    # 5b. SL/TP
    digits = 5 if current_price < 1000 else 2
    entry_price = tick.ask if action == "BUY" else tick.bid
    sl, tp = SLTPCalculator.compute(action, entry_price, atr_m5, digits=digits)

    # 5c. Lot validation
    desired_lot = LOT_MAP.get(clean_sym, 0.01)
    val_lot, val_reason = validate_lot_with_broker(clean_sym, desired_lot)
    if val_lot is None:
        log.warning(f"[Entry] {clean_sym} lot REJECTED: {val_reason}")
        return

    # 5d. Execute order
    ticket = mt5_open_position(
        broker_sym = broker_sym,
        clean_sym  = clean_sym,
        action     = action,
        lot        = val_lot,
        sl         = sl,
        tp         = tp,
        comment    = f"RabitScal_AI|conf={confidence:.2f}",
    )

    if ticket is not None:
        tracker.on_fill(
            ticket      = ticket,
            symbol      = clean_sym,
            direction   = action,
            lot         = val_lot,
            entry_price = entry_price,
        )


# =============================================================================
#  Main Loop
# =============================================================================

def main():
    log.info("=" * 60)
    log.info("  RabitScal AI Engine v3.0 -- MT5 Direct API")
    log.info("=" * 60)

    # 1. Initialize MT5
    if not mt5.initialize():
        log.error("FATAL: mt5.initialize() failed. Is MetaTrader 5 running?")
        sys.exit(1)

    terminal = mt5.terminal_info()
    account  = mt5.account_info()
    log.info(f"  MT5: {terminal.name} build {terminal.build}")
    log.info(f"  Account: {account.login} | Balance: {account.balance} {account.currency}")

    # 2. Resolve broker symbol names (XAUUSD vs XAUUSDm)
    symbol_map: dict[str, str] = {}
    for sym in ELITE_5:
        broker_sym = resolve_symbol(sym)
        if broker_sym:
            symbol_map[sym] = broker_sym
            mt5.symbol_select(broker_sym, True)

    if not symbol_map:
        log.error("FATAL: No symbols resolved. Check broker symbol names.")
        mt5.shutdown()
        sys.exit(1)

    log.info(f"  Symbols: {symbol_map}")

    # 3. Load XGBoost models
    clf = get_classifier()
    loaded = clf.load_all(list(ALLOWED_SYMBOLS))
    log.info(f"  [AI] Loaded {loaded} XGBoost models")

    # 4. Initialize tracker + sync with broker
    tracker = PositionTracker()
    sync_positions(tracker, symbol_map)

    log.info("=" * 60)
    log.info("  Engine running -- waiting for M5 candle close...")
    log.info("=" * 60)

    # 5. Main loop
    last_m5_minute = -1
    cycle_count    = 0

    try:
        while True:
            now = datetime.now(timezone.utc)
            minute = now.minute
            second = now.second

            # Detect M5 candle close: minute divisible by 5, first 5 seconds
            is_m5_close = (minute % 5 == 0) and (second < 5) and (minute != last_m5_minute)

            if is_m5_close:
                last_m5_minute = minute
                cycle_count += 1
                log.info(f"\n{'-' * 50}")
                log.info(
                    f"[CYCLE #{cycle_count}] M5 candle closed at "
                    f"{now.strftime('%H:%M:%S')} UTC"
                )
                log.info(f"{'-' * 50}")

                # Sync positions with broker reality
                sync_positions(tracker, symbol_map)

                # Run AI for each symbol
                for clean_sym, broker_sym in symbol_map.items():
                    try:
                        run_ai_for_symbol(clean_sym, broker_sym, tracker)
                    except Exception as e:
                        log.error(
                            f"[AI] {clean_sym} EXCEPTION: {e}\n"
                            f"{traceback.format_exc()}"
                        )

                # Log tracker status
                status = tracker.get_status()
                if status:
                    log.info(f"[Tracker] Open positions: {status}")
                else:
                    log.info("[Tracker] No open positions")

            # Heartbeat log every 60 seconds while waiting
            if cycle_count == 0 and second == 0:
                log.debug(f"[Heartbeat] Waiting for M5... {now.strftime('%H:%M:%S')}")

            time.sleep(1)

    except KeyboardInterrupt:
        log.info("\nInterrupted by user — shutting down...")
    finally:
        mt5.shutdown()
        log.info("MT5 connection closed. Goodbye!")


if __name__ == "__main__":
    main()
