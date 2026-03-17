"""
position_tracker.py -- RabitScal AI Smart Exit v2.1
====================================================
Track open positions and evaluate dynamic exit signals.
Includes CSV trade logger for post-analysis.

Exit layers (Sep Vu 2026-03-17):
    L1: Fast Break-even -- profit >= 1.5*ATR -> move SL to entry
    L2: Smart Scale-out -- profit >= 2.5*ATR -> partial close (if broker allows)
    L3: AI Counter-signal -- opposite signal conf >= 0.65 -> full close
    L4: Dynamic Trailing Stop -- after BE, trail tightens with profit:
        < 3*ATR -> 1.5*ATR | 3-5*ATR -> 1.0*ATR | > 5*ATR -> 0.7*ATR
    L5: Safety Time-stop -- 24 bars (2h) without BE -> cut loss
"""
from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("MT5Engine")

# CSV trade log path
TRADES_CSV = Path(__file__).resolve().parent.parent / "trades_history.csv"
_CSV_COLUMNS = [
    "Timestamp", "Symbol", "Direction", "Entry_Price", "Exit_Price",
    "PnL_USD", "Bars_Held", "Exit_Reason",
]


def _ensure_csv_header():
    """Create CSV with header if it doesn't exist."""
    if not TRADES_CSV.exists():
        with open(TRADES_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(_CSV_COLUMNS)


def log_trade_csv(
    symbol:      str,
    direction:   str,
    entry_price: float,
    exit_price:  float,
    pnl_usd:     float,
    bars_held:   int,
    exit_reason: str,
):
    """Append one trade record to trades_history.csv."""
    _ensure_csv_header()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(TRADES_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                ts, symbol, direction, f"{entry_price:.5f}", f"{exit_price:.5f}",
                f"{pnl_usd:.2f}", bars_held, exit_reason,
            ])
        log.info(f"[CSV] Trade logged: {symbol} {direction} PnL=${pnl_usd:.2f} | {exit_reason}")
    except Exception as e:
        log.error(f"[CSV] Failed to write trade log: {e}")


# =============================================================================
#  Position Data
# =============================================================================

@dataclass
class TrackedPosition:
    """State of one open position."""
    ticket:          int
    symbol:          str
    direction:       str          # "BUY" or "SELL"
    lot:             float
    entry_price:     float
    open_time:       float = field(default_factory=time.time)

    # AI tracking
    max_confidence:  float = 0.0  # Peak confidence of the holding direction
    bars_held:       int   = 0    # M5 candles since entry

    # Scale-out tracking
    is_scaled_out:   bool  = False
    original_lot:    float = 0.0  # Lot at entry (before partial close)

    # Break-even & trailing tracking
    is_breakeven:    bool  = False  # SL already moved to entry?
    peak_price:      float = 0.0   # Best price since entry (for trailing)

    def __post_init__(self):
        if self.original_lot == 0.0:
            self.original_lot = self.lot
        if self.peak_price == 0.0:
            self.peak_price = self.entry_price


# =============================================================================
#  Position Tracker
# =============================================================================

class PositionTracker:
    """
    Track open positions in RAM for AI Smart Exit v2.1.

    Per-symbol independent: each symbol can have 1 open position.
    No global lock -- XAUUSD and ETHUSD trade independently.
    """

    def __init__(self):
        self._positions: dict[int, TrackedPosition] = {}
        self._symbol_index: dict[str, int] = {}

    @property
    def open_count(self) -> int:
        return len(self._positions)

    def get_position(self, symbol: str) -> Optional[TrackedPosition]:
        ticket = self._symbol_index.get(symbol)
        if ticket is None:
            return None
        return self._positions.get(ticket)

    def has_position(self, symbol: str) -> bool:
        return symbol in self._symbol_index

    # -- Lifecycle ---------------------------------------------------------

    def on_fill(
        self,
        ticket:      int,
        symbol:      str,
        direction:   str,
        lot:         float,
        entry_price: float,
    ) -> None:
        """Register a newly filled position."""
        pos = TrackedPosition(
            ticket=ticket,
            symbol=symbol,
            direction=direction,
            lot=lot,
            entry_price=entry_price,
        )
        self._positions[ticket] = pos
        self._symbol_index[symbol] = ticket
        log.info(
            f"[Tracker] REGISTERED: ticket={ticket} {direction} {symbol} "
            f"lot={lot} entry={entry_price}"
        )

    def on_close(
        self,
        ticket:      int,
        exit_price:  float = 0.0,
        exit_reason: str = "UNKNOWN",
        pnl_usd:     float = 0.0,
    ) -> None:
        """Remove a fully closed position and log to CSV."""
        pos = self._positions.pop(ticket, None)
        if pos:
            self._symbol_index.pop(pos.symbol, None)
            elapsed = time.time() - pos.open_time
            log.info(
                f"[Tracker] REMOVED: ticket={ticket} {pos.symbol} "
                f"held={pos.bars_held} bars, {elapsed:.0f}s | {exit_reason}"
            )
            # Log to CSV
            log_trade_csv(
                symbol=pos.symbol,
                direction=pos.direction,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                pnl_usd=pnl_usd,
                bars_held=pos.bars_held,
                exit_reason=exit_reason,
            )

    def on_partial_close(self, ticket: int, remaining_lot: float) -> None:
        """Update lot after partial close."""
        pos = self._positions.get(ticket)
        if pos:
            pos.lot = remaining_lot
            pos.is_scaled_out = True
            log.info(
                f"[Tracker] PARTIAL CLOSE: ticket={ticket} {pos.symbol} "
                f"remaining_lot={remaining_lot}"
            )

    # -- AI Smart Exit v2.1 ------------------------------------------------

    # Exit action constants
    EXIT_NONE           = None
    EXIT_BREAKEVEN      = "FAST_BREAKEVEN"
    EXIT_REVERSAL       = "AI_COUNTER_SIGNAL"
    EXIT_SCALE_OUT      = "SMART_SCALE_OUT"
    EXIT_TRAILING       = "TRAILING_STOP"
    EXIT_TIME_SAFETY    = "TIME_SAFETY_NET"

    # Tunable parameters
    BE_ATR_MULT            = 1.5    # Move SL to entry when profit >= 1.5*ATR
    SCALE_OUT_ATR_MULT     = 2.5    # Partial close when profit >= 2.5*ATR
    SCALE_OUT_LOT_PCT      = 0.50   # Close 50% of lot
    COUNTER_SIGNAL_CONF    = 0.65   # AI reversal threshold
    MAX_BARS_NO_BE         = 24     # 24 * M5 = 2 hours -> cut if no BE

    # Dynamic trailing tiers (profit in ATR multiples -> trail distance)
    TRAIL_TIER_1_MULT      = 1.5   # profit < 3*ATR -> trail at 1.5*ATR
    TRAIL_TIER_2_MULT      = 1.0   # profit 3-5*ATR -> trail at 1.0*ATR
    TRAIL_TIER_3_MULT      = 0.7   # profit > 5*ATR -> trail at 0.7*ATR

    def _get_trail_distance(self, profit_dist: float, atr: float) -> float:
        """Dynamic trailing: tighter as profit grows."""
        if atr <= 0:
            return atr * self.TRAIL_TIER_1_MULT
        profit_atr = profit_dist / atr
        if profit_atr >= 5.0:
            return atr * self.TRAIL_TIER_3_MULT   # Lock tight
        elif profit_atr >= 3.0:
            return atr * self.TRAIL_TIER_2_MULT   # Tighten
        else:
            return atr * self.TRAIL_TIER_1_MULT   # Breathe

    def evaluate_exit(
        self,
        symbol:        str,
        ai_result:     dict,
        current_price: float,
        atr:           float,
        broker_vol_min: float = 0.01,
    ) -> Optional[dict]:
        """
        Evaluate whether an open position should be exited.

        Returns:
            None -- hold position
            dict -- exit action with keys: exit, action, ticket, symbol, ...
        """
        pos = self.get_position(symbol)
        if pos is None:
            return None

        # Increment bar counter
        pos.bars_held += 1

        proba  = ai_result.get("proba", [1.0, 0.0, 0.0])
        action = ai_result.get("action", "HOLD")

        # Track peak price for trailing stop
        if pos.direction == "BUY":
            profit_dist = current_price - pos.entry_price
            if current_price > pos.peak_price:
                pos.peak_price = current_price
        else:  # SELL
            profit_dist = pos.entry_price - current_price
            if current_price < pos.peak_price:
                pos.peak_price = current_price

        # Confidence of opposing direction
        if pos.direction == "BUY":
            reverse_conf = float(proba[2])    # P_SELL
        else:
            reverse_conf = float(proba[1])    # P_BUY

        # == L1: FAST BREAK-EVEN ==========================================
        if not pos.is_breakeven and atr > 0 and profit_dist >= self.BE_ATR_MULT * atr:
            pos.is_breakeven = True
            log.info(
                f"[EXIT L1] {symbol} FAST BREAKEVEN: profit={profit_dist:.5f} "
                f">= {self.BE_ATR_MULT}*ATR={atr:.5f} -> SL to entry"
            )
            return {
                "exit":     self.EXIT_BREAKEVEN,
                "action":   "MODIFY_SL",
                "ticket":   pos.ticket,
                "symbol":   symbol,
                "new_sl":   pos.entry_price,
                "reason":   f"Fast BE at {self.BE_ATR_MULT}*ATR",
            }

        # == L2: SMART SCALE-OUT ==========================================
        if not pos.is_scaled_out and atr > 0 and profit_dist >= self.SCALE_OUT_ATR_MULT * atr:
            half_vol = round(pos.original_lot * self.SCALE_OUT_LOT_PCT, 10)

            if half_vol >= broker_vol_min:
                remaining = round(pos.original_lot - half_vol, 10)
                log.info(
                    f"[EXIT L2] {symbol} SCALE-OUT: profit={profit_dist:.5f} "
                    f">= {self.SCALE_OUT_ATR_MULT}*ATR. "
                    f"Closing {half_vol} lot, keeping {remaining}"
                )
                return {
                    "exit":       self.EXIT_SCALE_OUT,
                    "action":     "CLOSE_PARTIAL",
                    "ticket":     pos.ticket,
                    "symbol":     symbol,
                    "volume":     half_vol,
                    "remaining":  remaining,
                    "new_sl":     pos.entry_price,
                    "reason":     f"Scale-out at {self.SCALE_OUT_ATR_MULT}*ATR",
                }
            else:
                log.debug(
                    f"[EXIT L2] {symbol} SKIP scale-out: half_vol={half_vol} "
                    f"< broker_min={broker_vol_min} -> holding full position"
                )

        # == L3: AI COUNTER-SIGNAL EXIT ===================================
        if pos.direction == "BUY" and action == "SELL" and reverse_conf >= self.COUNTER_SIGNAL_CONF:
            log.warning(
                f"[EXIT L3] {symbol} AI COUNTER: holding BUY but "
                f"P_SELL={reverse_conf:.4f} >= {self.COUNTER_SIGNAL_CONF}"
            )
            return {
                "exit":   self.EXIT_REVERSAL,
                "action": "CLOSE",
                "ticket": pos.ticket,
                "symbol": symbol,
                "reason": f"AI counter-signal: P_SELL={reverse_conf:.4f}",
            }
        if pos.direction == "SELL" and action == "BUY" and reverse_conf >= self.COUNTER_SIGNAL_CONF:
            log.warning(
                f"[EXIT L3] {symbol} AI COUNTER: holding SELL but "
                f"P_BUY={reverse_conf:.4f} >= {self.COUNTER_SIGNAL_CONF}"
            )
            return {
                "exit":   self.EXIT_REVERSAL,
                "action": "CLOSE",
                "ticket": pos.ticket,
                "symbol": symbol,
                "reason": f"AI counter-signal: P_BUY={reverse_conf:.4f}",
            }

        # == L4: DYNAMIC TRAILING STOP ====================================
        if pos.is_breakeven and atr > 0:
            trail_dist = self._get_trail_distance(profit_dist, atr)
            if pos.direction == "BUY":
                trail_sl = pos.peak_price - trail_dist
                if current_price <= trail_sl and trail_sl > pos.entry_price:
                    tier = "TIGHT" if trail_dist < atr else "NORMAL"
                    log.warning(
                        f"[EXIT L4] {symbol} TRAILING STOP ({tier}): "
                        f"price={current_price:.5f} <= trail={trail_sl:.5f} "
                        f"(peak={pos.peak_price:.5f} - {trail_dist:.5f})"
                    )
                    return {
                        "exit":   self.EXIT_TRAILING,
                        "action": "CLOSE",
                        "ticket": pos.ticket,
                        "symbol": symbol,
                        "reason": f"Trailing stop ({tier}): peak={pos.peak_price:.5f}",
                    }
            else:  # SELL
                trail_sl = pos.peak_price + trail_dist
                if current_price >= trail_sl and trail_sl < pos.entry_price:
                    tier = "TIGHT" if trail_dist < atr else "NORMAL"
                    log.warning(
                        f"[EXIT L4] {symbol} TRAILING STOP ({tier}): "
                        f"price={current_price:.5f} >= trail={trail_sl:.5f} "
                        f"(peak={pos.peak_price:.5f} + {trail_dist:.5f})"
                    )
                    return {
                        "exit":   self.EXIT_TRAILING,
                        "action": "CLOSE",
                        "ticket": pos.ticket,
                        "symbol": symbol,
                        "reason": f"Trailing stop ({tier}): peak={pos.peak_price:.5f}",
                    }

        # == L5: SAFETY TIME-STOP =========================================
        if not pos.is_breakeven and pos.bars_held >= self.MAX_BARS_NO_BE:
            log.warning(
                f"[EXIT L5] {symbol} TIME SAFETY: {pos.bars_held} bars "
                f"(>= {self.MAX_BARS_NO_BE}) without BE -> cutting loss"
            )
            return {
                "exit":   self.EXIT_TIME_SAFETY,
                "action": "CLOSE",
                "ticket": pos.ticket,
                "symbol": symbol,
                "reason": f"Safety time-stop: {pos.bars_held} bars without BE",
            }

        # Hold -- no exit signal
        trail_info = ""
        if pos.is_breakeven and atr > 0:
            td = self._get_trail_distance(profit_dist, atr)
            trail_info = f" trail_dist={td:.5f}"
        log.debug(
            f"[Tracker] {symbol} HOLD: bars={pos.bars_held} "
            f"profit={profit_dist:.5f} BE={pos.is_breakeven} "
            f"scaled={pos.is_scaled_out} peak={pos.peak_price:.5f}{trail_info}"
        )
        return None

    def get_status(self) -> dict:
        """Summary for logging/dashboard."""
        return {
            sym: {
                "ticket": pos.ticket,
                "dir": pos.direction,
                "lot": pos.lot,
                "entry": pos.entry_price,
                "bars": pos.bars_held,
                "BE": pos.is_breakeven,
                "scaled": pos.is_scaled_out,
                "peak": round(pos.peak_price, 5),
            }
            for sym, ticket in self._symbol_index.items()
            if (pos := self._positions.get(ticket))
        }
