"""
core/signal_engine.py — RabitScal Signal Engine & Risk Guard v1.0
==================================================================
Converts XGBoost predictions → validated ORDER messages for MT5.

LỆNH TỐI THƯỢNG (Sếp Vũ):
  1. LOT_MAP is hardcoded and IMMUTABLE — AI cannot override lot size.
  2. XAUUSD lot = 0.01 ALWAYS. No exceptions.
  3. Only ALLOWED_SYMBOLS trade — XAG and all unlisted symbols BLOCKED.

Signal flow:
    XGBoost output
        → RiskGuard.validate()       — symbol whitelist + lot override
        → SLTPCalculator.compute()   — dynamic SL/TP from ATR
        → build_order_msg()          — format JSON for MT5 socket
        → socket_bridge._send_message()
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("SignalEngine")

# ═══════════════════════════════════════════════════════════════════════════════
#  STATIC RISK CONFIGURATION — Sếp Vũ 2026-03-15 (DO NOT MODIFY WITHOUT APPROVAL)
# ═══════════════════════════════════════════════════════════════════════════════

# Whitelist: only these 5 symbols can generate orders.
# XAG is explicitly absent and must NEVER be added without Sếp Vũ approval.
ALLOWED_SYMBOLS: frozenset[str] = frozenset({
    "XAUUSD",
    "US30",
    "USTEC",
    "BTCUSD",
    "ETHUSD",
})

# Hardcoded lot sizes per symbol.
# These values are IMMUTABLE — AI model output must never override these.
LOT_MAP: dict[str, float] = {
    "XAUUSD": 0.01,   # HARDCODED — micro-lot for safety (Sếp Vũ mandate)
    "US30":   0.01,
    "USTEC":  0.01,
    "BTCUSD": 0.01,
    "ETHUSD": 0.01,
}

# ATR multipliers for dynamic SL/TP
SL_ATR_MULT  = 1.5   # SL = 1.5 × ATR
TP_ATR_MULT  = 2.5   # TP = 2.5 × ATR   (RR ≈ 1:1.67)

# Minimum ATR value to guard against division-by-zero
MIN_ATR = 0.0001

# ==========================================================================
#  SPREAD FILTER CONFIG (Seep Vu 2026-03-15)
#  Max allowed spread in POINTS before signal is blocked.
#  Source: typical Exness spreads during London/NY sessions + safety buffer.
# ==========================================================================
MAX_SPREAD_POINTS: dict[str, int] = {
    "XAUUSD": 35,    # Gold:   typical 15-25 pts, block spikes > 35 pts
    "US30":   50,    # Dow:    typical 20-35 pts
    "USTEC":  50,    # Nasdaq: typical 20-35 pts
    "BTCUSD": 200,   # BTC:    wide crypto spread
    "ETHUSD": 150,   # ETH:    wide crypto spread
}


def check_spread(symbol: str, spread_points: int) -> tuple[bool, str]:
    """
    Validate live market spread before sending an order.

    Args:
        symbol:        e.g. "XAUUSD"
        spread_points: current spread in MT5 integer points
                       (from TICK payload 'sp' field)

    Returns:
        (ok: bool, reason: str)
        ok=False → order must be blocked (spread too wide)
    """
    max_sp = MAX_SPREAD_POINTS.get(symbol, 100)
    if spread_points > max_sp:
        return False, (
            f"SPREAD_BLOCK: {symbol} spread={spread_points}pts "
            f"> max={max_sp}pts"
        )
    return True, f"spread={spread_points}pts OK (max={max_sp})"


# ═══════════════════════════════════════════════════════════════════════════════
#  Risk Guard
# ═══════════════════════════════════════════════════════════════════════════════

class RiskGuard:
    """
    Stateless risk validation layer.

    Every signal MUST pass through here before being sent to MT5.
    The guard enforces:
      - Symbol whitelist (no XAG or unknown symbols)
      - Hardcoded lot size (LOT_MAP — AI cannot override)
    """

    @staticmethod
    def validate(signal: dict) -> tuple[bool, str]:
        """
        Validate and normalise a raw AI signal.

        Args:
            signal: dict with keys: symbol, action, lot (optional), sl, tp

        Returns:
            (ok: bool, reason: str)
            If ok=True, signal["lot"] is overwritten with LOT_MAP value.
        """
        symbol = signal.get("symbol", "")
        action = signal.get("action", "")

        # 1. Symbol whitelist check
        if symbol not in ALLOWED_SYMBOLS:
            return False, f"BLOCKED: '{symbol}' not in ALLOWED_SYMBOLS (XAG and others are banned)"

        # 2. Action check
        if action not in ("BUY", "SELL"):
            return False, f"BLOCKED: invalid action '{action}' (must be BUY or SELL)"

        # 3. Hardcode lot — AI output is IGNORED, LOT_MAP is absolute
        ai_lot = signal.get("lot", None)
        hardcoded_lot = LOT_MAP[symbol]
        if ai_lot is not None and ai_lot != hardcoded_lot:
            log.debug(
                f"[RiskGuard] {symbol}: AI requested lot={ai_lot} → "
                f"overridden to {hardcoded_lot} (hardcoded)"
            )
        signal["lot"] = hardcoded_lot

        # 4. SL/TP sanity check (must be positive, non-zero)
        sl = signal.get("sl", 0.0)
        tp = signal.get("tp", 0.0)
        if sl <= 0 or tp <= 0:
            return False, f"BLOCKED: invalid SL={sl} or TP={tp} (must be > 0)"

        return True, "OK"


# ═══════════════════════════════════════════════════════════════════════════════
#  SL/TP Calculator
# ═══════════════════════════════════════════════════════════════════════════════

class SLTPCalculator:
    """Dynamic SL/TP based on ATR — no fixed pip values."""

    @staticmethod
    def compute(
        action: str,
        current_price: float,
        atr: float,
        sl_mult: float = SL_ATR_MULT,
        tp_mult: float = TP_ATR_MULT,
        digits: int = 5,
    ) -> tuple[float, float]:
        """
        Compute SL and TP prices from current price and ATR.

        Args:
            action:        "BUY" or "SELL"
            current_price: Current bid/ask price
            atr:           ATR value for the symbol/timeframe
            sl_mult:       SL = sl_mult × ATR
            tp_mult:       TP = tp_mult × ATR
            digits:        Decimal places for rounding

        Returns:
            (sl_price, tp_price) rounded to `digits` decimal places
        """
        atr_safe = max(atr, MIN_ATR)
        sl_dist  = atr_safe * sl_mult
        tp_dist  = atr_safe * tp_mult

        if action == "BUY":
            sl = round(current_price - sl_dist, digits)
            tp = round(current_price + tp_dist, digits)
        else:  # SELL
            sl = round(current_price + sl_dist, digits)
            tp = round(current_price - tp_dist, digits)

        return sl, tp


# ═══════════════════════════════════════════════════════════════════════════════
#  Order Message Builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_order_msg(
    action:  str,
    symbol:  str,
    lot:     float,
    sl:      float,
    tp:      float,
    comment: str = "RabitScal_AI",
) -> dict:
    """
    Build the ORDER message dict that socket_bridge will send to MT5.

    Format (matches MT5 EA ProcessOrder parser):
        {"type":"ORDER","id":"sig_...","action":"BUY","symbol":"XAUUSD",
         "lot":0.01,"sl":2340.0,"tp":2360.0,"ticket":0}
    """
    order_id = f"sig_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
    return {
        "type":    "ORDER",
        "id":      order_id,
        "action":  action,
        "symbol":  symbol,
        "lot":     lot,
        "sl":      sl,
        "tp":      tp,
        "ticket":  0,
        "magic":   202603,
        "comment": comment,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Signal Processing Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def process_ai_signal(
    symbol:        str,
    action:        str,
    confidence:    float,
    current_price: float,
    atr:           float,
    digits:        int = 5,
) -> Optional[dict]:
    """
    Full pipeline: AI output → Risk Guard → SL/TP → ORDER dict.

    Called from socket_bridge._run_ai_analysis() (background thread).

    Args:
        symbol:        e.g. "XAUUSD"
        action:        "BUY" | "SELL" | "HOLD"
        confidence:    model confidence [0,1]
        current_price: current bid or ask
        atr:           ATR from feature engine (M5 timeframe)
        digits:        price rounding

    Returns:
        ORDER dict (ready to send via socket_bridge._send_message()) or None if blocked.
    """
    # HOLD → nothing to do
    if action == "HOLD":
        return None

    # Compute SL/TP
    sl, tp = SLTPCalculator.compute(action, current_price, atr, digits=digits)

    # Build raw signal
    signal = {
        "symbol": symbol,
        "action": action,
        "lot":    LOT_MAP.get(symbol, 0.01),   # initial set, will be overridden by RiskGuard
        "sl":     sl,
        "tp":     tp,
    }

    # Risk Guard validation (symbol whitelist + lot hardcode)
    ok, reason = RiskGuard.validate(signal)
    if not ok:
        log.warning(f"[SignalEngine] {symbol} {action} BLOCKED: {reason}")
        return None

    # Build ORDER message
    order_msg = build_order_msg(
        action  = signal["action"],
        symbol  = signal["symbol"],
        lot     = signal["lot"],    # guaranteed = LOT_MAP value after RiskGuard
        sl      = signal["sl"],
        tp      = signal["tp"],
        comment = f"RabitScal_AI|conf={confidence:.2f}",
    )

    log.info(
        f"[SignalEngine] SIGNAL OK: {action} {symbol} "
        f"lot={signal['lot']} SL={sl} TP={tp} "
        f"conf={confidence:.2f} id={order_msg['id']}"
    )

    return order_msg


# ═══════════════════════════════════════════════════════════════════════════════
#  Thread-Safe Async Sender Helper
# ═══════════════════════════════════════════════════════════════════════════════

def schedule_order_send(
    order_msg:   dict,
    send_coro_fn,            # Callable that returns coroutine: async def _send_message(msg)
    event_loop:  asyncio.AbstractEventLoop,
) -> None:
    """
    Schedule an ORDER send from a background thread back to the asyncio event loop.

    Because _run_ai_analysis() runs in asyncio.to_thread() (thread pool),
    it cannot directly await coroutines. Use this helper to safely pass
    the order back to the event loop.

    Args:
        order_msg:   ORDER dict from process_ai_signal()
        send_coro_fn: lambda that calls bridge._send_message(order_msg)
        event_loop:  the bridge's running event loop (stored in bridge.__init__)
    """
    future = asyncio.run_coroutine_threadsafe(
        send_coro_fn(order_msg),
        event_loop,
    )
    # Non-blocking: don't wait for result in the thread
    # Log result when future completes (fire-and-forget with callback)
    def _on_done(fut):
        try:
            sent = fut.result(timeout=5)
            if sent:
                log.info(f"[SignalEngine] ORDER sent OK: {order_msg['id']}")
            else:
                log.warning(f"[SignalEngine] ORDER send failed: {order_msg['id']}")
        except Exception as e:
            log.error(f"[SignalEngine] ORDER send exception: {e}")

    future.add_done_callback(_on_done)
