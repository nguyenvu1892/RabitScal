"""
main.py — BotOrchestrator v1.0 (State Machine)
===============================================
Module: Rabit_Exness AI — Phase 3, Task 3.1
Branch: task-3.1-main-orchestrator
Author: Antigravity
Date:   2026-03-05

Bộ não điều phối bot theo mô hình State Machine 6 trạng thái:
    IDLE → SCANNING → SIGNAL_FOUND → PENDING_ORDER → IN_TRADE → CLOSING → IDLE

Architecture notes:
    • Multi-symbol: 7 cặp, loop IDLE → scan từng cặp theo thứ tự
    • Global Lock Rule: tối đa 1 lệnh mở toàn account
    • Anti-spam: mỗi symbol chỉ được scan sau MIN_SCAN_INTERVAL_SEC (300s ≈ 1 nến M5)
    • IN_TRADE poll 0.5s (nhanh hơn để bắt floating DD sớm)
    • Tất cả state khác: 1s loop tick
    • Daily reset theo Exness server time (không dùng UTC raw)
    • Ghost Order: timeout 30s → kiểm tra mt5.orders_get() → cancel hoặc IN_TRADE
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import MetaTrader5 as mt5
import numpy as np

from data_pipeline   import DataPipeline
from execution       import OrderManager
from risk_manager    import RiskManager, TradeResult
from strategy_engine import StrategyEngine, SignalResult

# Dashboard — import với fallback để bot vẫn chạy nếu dashboard bị lỗi
try:
    from dashboard import dashboard_pub, start_dashboard_server, set_pipeline as _dash_set_pipeline
    _DASHBOARD_AVAILABLE = True
except ImportError:
    _DASHBOARD_AVAILABLE = False
    def _dash_set_pipeline(*a, **kw): pass   # no-op fallback
    class _NullPub:             # Null-object pattern: publish() là no-op
        def publish(self, *a, **kw): pass
    dashboard_pub = _NullPub()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOOP_INTERVAL_SEC          = 1.0    # Tick chính (tất cả state trừ IN_TRADE)
IN_TRADE_POLL_INTERVAL_SEC = 0.5    # Poll nhanh hơn khi đang giữ lệnh
PENDING_ORDER_TIMEOUT_SEC  = 30     # Ghost order timeout


# ---------------------------------------------------------------------------
# Enums & Containers
# ---------------------------------------------------------------------------

class BotState(Enum):
    IDLE          = auto()
    SCANNING      = auto()
    SIGNAL_FOUND  = auto()
    PENDING_ORDER = auto()
    IN_TRADE      = auto()
    CLOSING       = auto()


@dataclass
class TradeInfo:
    """Thông tin lệnh đang mở — maintained trong self.open_trade."""
    symbol:      str
    signal:      SignalResult
    lot:         float
    sl_pips:     float
    order_time:  datetime
    ticket:      int                   = 0
    fill_price:  float                 = 0.0
    fill_time:   Optional[datetime]    = None
    close_reason: str                  = ""
    pnl:         float                 = 0.0


# ---------------------------------------------------------------------------
# BotOrchestrator
# ---------------------------------------------------------------------------

class BotOrchestrator:
    """
    State Machine điều phối toàn bộ bot Rabit_Exness AI.

    Khởi tạo từ main_config.json, spawn DataPipeline + StrategyEngine
    (một instance per symbol) + RiskManager (global) + OrderManager.
    """

    def __init__(self, config_path: str = "config/main_config.json") -> None:
        # --- Load config ---
        raw = Path(config_path).read_text(encoding="utf-8")
        self.cfg = json.loads(raw)

        self.symbols: list[str] = self.cfg.get("symbols", ["EURUSD"])

        # --- Logger ---
        log_cfg  = self.cfg.get("log", {})
        self.logger = self._build_logger(
            path         = log_cfg.get("path", "logs/system.log"),
            max_bytes    = log_cfg.get("max_bytes", 10_485_760),
            backup_count = log_cfg.get("backup_count", 5),
        )

        # --- State ---
        self.state:          BotState          = BotState.IDLE
        self.active_symbol:  Optional[str]     = None    # Symbol đang được scan/trade
        self.open_trade:     Optional[TradeInfo] = None
        self.last_signal:    Optional[SignalResult] = None
        self._running:       bool              = False
        self.daily_reset_done: bool            = False

        # Candle sync: {symbol: timestamp của nến M5 đã scan lần cuối}
        # So sánh với candle[-1].time (nến vừa đóng) — chỉ scan khi có nến mới thực sự
        self.last_candle_time: dict[str, int] = {s: 0 for s in self.symbols}

        # Magic number từ config (tránh hardcode)
        self.magic_number: int = int(self.cfg.get("magic_number", 20260305))

        # --- Modules ---
        pip_cfg  = self.cfg.get("pipeline", {})
        strat_cfg = self.cfg.get("strategy", {})
        risk_cfg = self.cfg.get("risk", {})

        self.pipeline = DataPipeline(pip_cfg,  self.logger)

        # Một StrategyEngine per symbol (mỗi symbol có FVG pool riêng)
        self.strategies: dict[str, StrategyEngine] = {
            sym: StrategyEngine(strat_cfg, self.logger, symbol=sym)
            for sym in self.symbols
        }

        self.risk     = RiskManager(
            config      = risk_cfg,
            logger      = self.logger,
            state_path  = "config/state.json",
            config_path = "config/risk_config.json",
            symbol      = self.symbols[0],   # RiskManager là global — symbol chỉ dùng cho pip_value
        )
        self.executor = OrderManager(self.cfg, self.logger)

        # Graceful shutdown on SIGTERM / SIGINT
        signal.signal(signal.SIGTERM, self._sig_handler)
        signal.signal(signal.SIGINT,  self._sig_handler)

        # Dashboard: khởi động uvicorn trong daemon thread (non-blocking)
        self._start_dashboard()

        self.logger.info(
            f"[BotOrchestrator] Initialized | symbols={self.symbols}"
            f" | daily_dd_limit=15% | balance_floor=50%"
        )

    # ==========================================================================
    # PUBLIC API
    # ==========================================================================

    def start(self) -> None:
        """Entry point: kết nối MT5, start pipeline, bắt đầu vòng lặp."""
        self.logger.info("[BotOrchestrator] Starting...")
        self.pipeline.start()
        self._running = True
        self._run_loop()

    def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        self.pipeline.stop()
        self.logger.info("[BotOrchestrator] Stopped.")

    # ==========================================================================
    # MAIN LOOP
    # ==========================================================================

    def _run_loop(self) -> None:
        """Vòng lặp chính. Dispatch tới handler của state hiện tại."""
        while self._running:
            try:
                self._check_daily_reset()
                self._dispatch_state()

            except KeyboardInterrupt:
                self.stop()
                break
            except Exception as e:
                self.logger.critical(
                    f"[BotOrchestrator] UNHANDLED error in state={self.state.name}: {e}",
                    exc_info=True,
                )
                self._transition(BotState.IDLE)

            # Poll interval — IN_TRADE dùng interval ngắn hơn (0.5s)
            if self.state == BotState.IN_TRADE:
                time.sleep(IN_TRADE_POLL_INTERVAL_SEC)
            else:
                time.sleep(LOOP_INTERVAL_SEC)

    def _dispatch_state(self) -> None:
        """Router: gọi handler tương ứng với state hiện tại."""
        handlers = {
            BotState.IDLE:          self._state_idle,
            BotState.SCANNING:      self._state_scanning,
            BotState.SIGNAL_FOUND:  self._state_signal_found,
            BotState.PENDING_ORDER: self._state_pending_order,
            BotState.IN_TRADE:      self._state_in_trade,
            BotState.CLOSING:       self._state_closing,
        }
        handler = handlers.get(self.state)
        if handler:
            handler()
        else:
            self.logger.error(f"[BotOrchestrator] Unknown state: {self.state}")
            self._transition(BotState.IDLE)

    # ==========================================================================
    # STATE HANDLERS
    # ==========================================================================

    def _state_idle(self) -> None:
        """
        IDLE: Kiểm tra điều kiện rồi chọn symbol để scan.

        Multi-symbol loop với Candle Sync (Global Lock Rule):
            1. Nếu đang có lệnh mở (open_trade is not None) → không scan mới.
            2. Bot phải ACTIVE, Session phải ACTIVE.
            3. Với từng symbol theo thứ tự ưu tiên:
               Fetch nhanh 2 nến M5 cuối (OHLCV chỉ lấy time).
               Nếu candle_closed.time > self.last_candle_time[symbol]:
                   → Có nến mới vừa đóng → cập nhật mốc → SCANNING.
               → Không dùng elapsed 300s: bắt đúng pha nến, không bỏ lỡ hay trễ thêm.

        Chỉ chọn 1 symbol mỗi tick — return ngay sau khi chọn được.
        """
        # Global Lock: không scan khi đang có lệnh mở
        if self.open_trade is not None:
            self.logger.debug(
                f"[IDLE] Trade open on {self.open_trade.symbol} → skip scan"
            )
            return

        # Bot phải ở trạng thái ACTIVE
        if not self.risk.is_active():
            self.logger.debug(
                f"[IDLE] Bot not active | status={self.risk.bot_status}"
            )
            return

        # Session phải là giờ giao dịch (London / NY)
        if not self.pipeline.is_session_active():
            self.logger.debug("[IDLE] Outside active session — waiting")
            return

        # Candle Sync: loop từng symbol, kiểm tra nến M5 mới nhất đã đóng
        for symbol in self.symbols:
            # Fetch nhanh 2 nến M5 để lấy timestamp (không fetch full 1000 nến)
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 2)

            if rates is None or len(rates) < 2:
                # API lỗi hoặc không đủ data — bỏ qua symbol này
                self.logger.debug(f"[IDLE] {symbol}: cannot fetch M5 rates — skip")
                continue

            # rates[-2] = nến đã đóng cuối cùng (anti-repainting)
            # rates[-1] = nến đang chạy (không dùng)
            candle_closed_time = int(rates[-2]["time"])   # Unix timestamp (giây)

            if candle_closed_time <= self.last_candle_time.get(symbol, 0):
                # Chưa có nến mới — nến này đã cầm rồi, không scan lại
                self.logger.debug(
                    f"[IDLE] {symbol}: same candle (t={candle_closed_time}) — skip"
                )
                continue

            # Có nến M5 mới vừa đóng! Cập nhật mốc và chuyển sang SCANNING
            self.last_candle_time[symbol] = candle_closed_time
            self.active_symbol = symbol
            self.logger.info(
                f"[IDLE] {symbol}: new M5 candle detected"
                f" (t={candle_closed_time}) → SCANNING"
            )
            self._transition(BotState.SCANNING)
            return   # ← Chỉ chọn 1 symbol mỗi tick, return ngay

        # Không có symbol nào có nến mới — ở lại IDLE
        self.logger.debug("[IDLE] No new M5 candle on any symbol — waiting")

    def _state_scanning(self) -> None:
        """
        SCANNING: Fetch data đầy đủ cho active_symbol và phân tích tín hiệu.

        Không cần update last_candle_time ở đây — đã update trong _state_idle().
        Về IDLE sau mỗi lần scan (dù có signal hay không) để round-robin qua các symbol.
        """
        symbol = self.active_symbol
        if symbol is None:
            self.logger.error("[SCANNING] active_symbol is None — fallback IDLE")
            self._transition(BotState.IDLE)
            return

        try:
            data = self.pipeline.fetch_all(symbol=symbol)

            if data is None:
                self.logger.warning(f"[SCANNING] {symbol}: fetch_all returned None — skip")
                self._update_scan_time(symbol)
                self._transition(BotState.IDLE)
                return

            # Kiểm tra data quality
            quality = data.get("quality_score", 1.0)
            if quality < self.cfg.get("min_data_quality", 0.60):
                self.logger.warning(
                    f"[SCANNING] {symbol}: data quality too low ({quality:.2f}) — skip"
                )
                self._update_scan_time(symbol)
                self._transition(BotState.IDLE)
                return

            # Phân tích tín hiệu cho symbol này
            engine = self.strategies[symbol]
            signal = engine.analyze(
                h1_data      = data["H1"],
                m15_data     = data["M15"],
                m5_data      = data["M5"],
                current_time = datetime.now(timezone.utc),
            )

            if signal.has_signal:
                self.last_signal = signal
                self.logger.info(
                    f"[SCANNING] {symbol}: SIGNAL {signal.direction}"
                    f" score={signal.score:.4f} → SIGNAL_FOUND"
                )
                # Dashboard: publish signal_found với FVG info
                fvg = getattr(signal, 'fvg', None)
                # fvg_created_time: timestamp thực của nến tạo ra FVG
                # Dùng fvg.created_time nếu có, fallback candle đang xét
                fvg_created_ts = (
                    getattr(fvg, 'created_time', None) or
                    getattr(fvg, 'candle_time',  None) or
                    self.last_candle_time.get(symbol, int(time.time()))
                )
                # Convert sang ISO string cho JS (Plotly dùng ISO làm xref='x')
                fvg_created_iso = datetime.fromtimestamp(
                    int(fvg_created_ts), tz=timezone.utc
                ).isoformat() if fvg_created_ts else datetime.now(timezone.utc).isoformat()

                dashboard_pub.publish({
                    "type":    "signal_found",
                    "ts":      int(time.time()),
                    "symbol":  symbol,
                    "payload": {
                        "direction":        signal.direction,
                        "score":            round(float(signal.score), 4),
                        "entry_price":      round(float(getattr(signal, 'entry_price', 0)), 5),
                        "sl_price":         round(float(getattr(signal, 'sl_price', 0)), 5),
                        "fvg_top":          round(float(fvg.top    if fvg else 0), 5),
                        "fvg_bottom":       round(float(fvg.bottom if fvg else 0), 5),
                        "fvg_created_time": fvg_created_iso,  # ← Gốc thực của FVG box
                    },
                })
                self._transition(BotState.SIGNAL_FOUND)
            else:
                self.logger.debug(f"[SCANNING] {symbol}: no signal → IDLE")
                self._transition(BotState.IDLE)

        except Exception as e:
            self.logger.error(f"[SCANNING] {symbol}: error: {e}", exc_info=True)
            self._transition(BotState.IDLE)

    def _state_signal_found(self) -> None:
        """
        SIGNAL_FOUND: Validate trade parameters trước khi gửi lệnh.
        Double-check risk, tính lot size, validate → PENDING_ORDER hoặc IDLE.
        """
        signal = self.last_signal
        symbol = self.active_symbol

        if signal is None or symbol is None:
            self._transition(BotState.IDLE)
            return

        # Double-check: vẫn active?
        if not self.risk.is_active():
            self.logger.warning(f"[SIGNAL_FOUND] {symbol}: bot not active → abort")
            self._transition(BotState.IDLE)
            return

        # Lấy balance + tính ATR
        account = mt5.account_info()
        if account is None:
            self.logger.error("[SIGNAL_FOUND] Cannot get account_info — abort")
            self._transition(BotState.IDLE)
            return

        balance  = float(account.balance)
        atr_val  = self._get_atr14(symbol)

        sl_pips = self.risk.calculate_sl_distance(atr_val, signal.direction)
        lot     = self.risk.calculate_lot_size(balance, atr_val, sl_pips)
        ok, reason = self.risk.validate_trade(balance, lot, sl_pips)

        if not ok:
            self.logger.warning(f"[SIGNAL_FOUND] {symbol}: rejected: {reason} → IDLE")
            self._transition(BotState.IDLE)
            return

        self.open_trade = TradeInfo(
            symbol     = symbol,
            signal     = signal,
            lot        = lot,
            sl_pips    = sl_pips,
            order_time = datetime.now(timezone.utc),
        )

        self.logger.info(
            f"[SIGNAL_FOUND] {symbol}: trade validated | dir={signal.direction}"
            f" | lot={lot:.2f} | sl_pips={sl_pips:.2f} → PENDING_ORDER"
        )
        self._transition(BotState.PENDING_ORDER)

    def _state_pending_order(self) -> None:
        """
        PENDING_ORDER: Gửi lệnh ra MT5, chờ FILLED.

        Ghost Order Handling:
            Sau PENDING_ORDER_TIMEOUT_SEC (30s) nếu chưa nhận FILLED:
            1. Kiểm tra mt5.orders_get(symbol) xem có lệnh treo không.
            2. Nếu CÓ lệnh treo → CANCEL lệnh đó → về IDLE (tránh ghost order).
            3. Nếu KHÔNG có lệnh treo nhưng CÓ vị thế mở → lệnh thực ra đã FILLED
               (confirmation về trễ) → ghi nhận ticket → IN_TRADE.
            4. Nếu không có gì → về IDLE.
        """
        trade = self.open_trade
        if trade is None:
            self._transition(BotState.IDLE)
            return

        symbol = trade.symbol

        # --- Bước 1: Gửi lệnh ---
        result = self.executor.send_order(
            symbol    = symbol,
            direction = trade.signal.direction,
            lot       = trade.lot,
            sl_price  = trade.signal.sl_price,
            comment   = f"Rabit_{trade.signal.score:.3f}",
        )

        if result.success:
            # FILLED ngay lập tức (trường hợp thường gặp)
            trade.ticket     = result.ticket
            trade.fill_price = result.fill_price
            trade.fill_time  = datetime.now(timezone.utc)
            self.logger.info(
                f"[PENDING_ORDER] {symbol}: FILLED | ticket={result.ticket}"
                f" | fill={result.fill_price:.5f} → IN_TRADE"
            )
            # Dashboard: vẽ Entry arrow + SL/TP lines
            dashboard_pub.publish({
                "type":    "order_filled",
                "ts":      int(time.time()),
                "symbol":  symbol,
                "payload": {
                    "ticket":    result.ticket,
                    "direction": trade.signal.direction,
                    "entry":     round(float(result.fill_price), 5),
                    "sl":        round(float(getattr(trade.signal, 'sl_price', 0)), 5),
                    "tp":        round(float(getattr(trade.signal, 'tp_price', 0)), 5),
                    "lot":       round(float(trade.lot), 2),
                },
            })
            self._transition(BotState.IN_TRADE)
            return

        # --- Bước 2: Lệnh bị reject ngay (retcode != success) ---
        if not result.may_be_pending:
            # Broker reject rõ ràng (REQUOTE, INVALID_PRICE, etc.)
            self.logger.warning(
                f"[PENDING_ORDER] {symbol}: REJECTED | reason={result.error} → IDLE"
            )
            self.open_trade  = None
            self.last_signal = None
            self._transition(BotState.IDLE)
            return

        # --- Bước 3: Timeout wait (30s) — lệnh có thể đang queue ---
        self.logger.info(
            f"[PENDING_ORDER] {symbol}: order sent but not confirmed"
            f" — waiting {PENDING_ORDER_TIMEOUT_SEC}s for fill..."
        )
        deadline = time.monotonic() + PENDING_ORDER_TIMEOUT_SEC
        ticket_confirmed = 0

        while time.monotonic() < deadline:
            time.sleep(1)

            # Kiểm tra positions trước — lệnh đã khớp?
            positions = mt5.positions_get(symbol=symbol)
            if positions:
                for pos in positions:
                    # Lọc đúng magic number để tránh nhầm lệnh tay
                    if pos.magic == self.magic_number:
                        # Lệnh đã vào! confirmation về trễ
                        ticket_confirmed = pos.ticket
                        self.logger.info(
                            f"[PENDING_ORDER] {symbol}: position confirmed"
                            f" | ticket={ticket_confirmed} (late confirmation)"
                        )
                        break
                if ticket_confirmed:
                    break

            # Kiểm tra pending orders
            orders = mt5.orders_get(symbol=symbol)
            if orders is None:
                # API trả None (lỗi mạng tạm thời) — thử lại ở giây tiếp theo
                self.logger.debug(
                    f"[PENDING_ORDER] {symbol}: orders_get returned None — retrying..."
                )
                continue   # ← FIX: continue, không break khi None
            if len(orders) == 0:
                # Chắc chắn không có lệnh treo — thoát vòng chờ
                break

        # --- Bước 4: Phân loại kết quả sau timeout ---
        if ticket_confirmed:
            # Lệnh đã fill (confirmation về chậm)
            trade.ticket     = ticket_confirmed
            trade.fill_price = self._get_position_price(ticket_confirmed, symbol)
            trade.fill_time  = datetime.now(timezone.utc)
            self.logger.info(
                f"[PENDING_ORDER] {symbol}: Confirmed fill ticket={ticket_confirmed}"
                f" | price={trade.fill_price:.5f} → IN_TRADE"
            )
            self._transition(BotState.IN_TRADE)
            return

        # Ghost Order: vẫn còn lệnh treo → CANCEL
        pending_orders = mt5.orders_get(symbol=symbol)
        if pending_orders:
            for order in pending_orders:
                if order.magic == self.magic_number:
                    cancel_req = {
                        "action": mt5.TRADE_ACTION_REMOVE,
                        "order":  order.ticket,
                    }
                    cancel_result = mt5.order_send(cancel_req)
                    self.logger.warning(
                        f"[PENDING_ORDER] {symbol}: GHOST ORDER canceled"
                        f" | ticket={order.ticket}"
                        f" | cancel_retcode={cancel_result.retcode if cancel_result else 'N/A'}"
                    )

        # Về IDLE — không có lệnh xác nhận nào
        self.logger.warning(
            f"[PENDING_ORDER] {symbol}: TIMEOUT {PENDING_ORDER_TIMEOUT_SEC}s"
            f" — no fill confirmed → IDLE"
        )
        self.open_trade  = None
        self.last_signal = None
        self._transition(BotState.IDLE)

    def _state_in_trade(self) -> None:
        """
        IN_TRADE: Giữ lệnh, monitor floating DD và SL/TP.

        KHÔNG scan signal mới khi đang ở state này.
        Poll mỗi IN_TRADE_POLL_INTERVAL_SEC (0.5s) để bắt DD breach sớm.
        """
        trade = self.open_trade
        if trade is None:
            self.logger.error("[IN_TRADE] open_trade is None — fallback IDLE")
            self._transition(BotState.IDLE)
            return

        # --- Check 1: Floating Drawdown ---
        account = mt5.account_info()
        if account is None:
            self.logger.warning("[IN_TRADE] Cannot get account_info — reconnecting...")
            return   # Chờ tick tiếp theo (pipeline heartbeat sẽ reconnect)

        safe = self.risk.check_floating_drawdown(float(account.equity))
        if not safe:
            self.logger.critical(
                f"[IN_TRADE] {trade.symbol}: Floating DD breach"
                f" equity={account.equity:.2f} → Emergency close all → CLOSING"
            )
            self._close_all_orders_market()
            self._transition(BotState.CLOSING)
            return

        # Dashboard: publish equity realtime (mỗi 0.5s tick)
        balance_ref = getattr(self.risk, '_daily_start_balance', account.balance) or account.balance
        dd_pct = round((1.0 - account.equity / balance_ref) * 100.0, 2) if balance_ref > 0 else 0.0
        dashboard_pub.publish({
            "type":    "equity_update",
            "ts":      int(time.time()),
            "symbol":  trade.symbol,
            "payload": {
                "equity":       round(float(account.equity),  2),
                "balance":      round(float(account.balance), 2),
                "floating_pnl": round(float(account.equity - account.balance), 2),
                "dd_pct":       dd_pct,
            },
        })

        # --- Check 2: SL/TP hit (position đã đóng tự động bởi MT5?) ---
        positions = mt5.positions_get(ticket=trade.ticket)
        if positions is None or len(positions) == 0:
            # Vị thế đã biến mất → SL/TP đã hit (hoặc manual close)
            self.logger.info(
                f"[IN_TRADE] {trade.symbol}: position {trade.ticket} closed"
                f" (SL/TP hit or manual) → CLOSING"
            )
            self._transition(BotState.CLOSING)

        # Nếu vị thế vẫn còn → ở lại IN_TRADE, không làm gì thêm

    def _state_closing(self) -> None:
        """
        CLOSING: Lấy lịch sử lệnh từ MT5, báo cáo cho RiskManager, dọn dẹp state.
        """
        trade = self.open_trade

        if trade is not None and trade.ticket > 0:
            # Lấy history deals của position này
            history = mt5.history_deals_get(position=trade.ticket)
            if history:
                pnl          = sum(float(d.profit) for d in history)
                close_reason = self._parse_close_reason(history[-1])
                trade.pnl          = pnl
                trade.close_reason = close_reason

                account = mt5.account_info()
                balance = float(account.balance) if account else 0.0

                result = TradeResult(
                    ticket       = trade.ticket,
                    pnl          = pnl,
                    close_reason = close_reason,
                    timestamp    = datetime.now(timezone.utc),
                )

                self.risk.on_trade_closed(result, balance)

                self.logger.info(
                    f"[CLOSING] {trade.symbol} ticket={trade.ticket}"
                    f" | pnl={pnl:+.4f} | reason={close_reason}"
                    f" | balance_after={balance:.2f}"
                )

                # Dashboard: vẽ close marker + PnL badge
                dashboard_pub.publish({
                    "type":    "trade_closed",
                    "ts":      int(time.time()),
                    "symbol":  trade.symbol,
                    "payload": {
                        "ticket":      trade.ticket,
                        "pnl":         round(float(pnl), 4),
                        "close_reason": close_reason,
                        "balance_after": round(float(balance), 2),
                    },
                })
            else:
                self.logger.warning(
                    f"[CLOSING] {trade.symbol}: no deal history for ticket={trade.ticket}"
                )

        # Dọn dẹp state
        self.open_trade   = None
        self.last_signal  = None
        self.active_symbol = None
        self._transition(BotState.IDLE)

    # ==========================================================================
    # DAILY RESET — Server Time
    # ==========================================================================

    def _check_daily_reset(self) -> None:
        """
        Daily reset theo Exness server time (không dùng UTC raw).

        Exness server time = UTC+2 (winter) / UTC+3 (summer DST).
        DataPipeline._detect_server_tz() đã detect offset khi start.
        Dùng mt5.symbol_info_tick() timestamp làm tham chiếu server time.

        Trigger: khi server_hour == 0 và server_minute == 0 (00:00 server).
        """
        # Lấy server time qua mt5 terminal
        terminal_info = mt5.terminal_info()
        if terminal_info is None:
            return

        # Server time từ MT5 (seconds since epoch, server timezone)
        # mt5.symbol_info_tick() trả về server timestamp
        # Cách reliable nhất: đọc từ một tick gần nhất
        try:
            tick = mt5.symbol_info_tick(self.symbols[0])
            if tick is None:
                return
            server_dt = datetime.fromtimestamp(tick.time, tz=timezone.utc)
            # Exness server offset: detect từ pipeline hoặc fallback UTC+2
            server_offset_h = getattr(self.pipeline, '_server_tz_offset', 2)
            server_hour   = (server_dt.hour + server_offset_h) % 24
            server_minute = server_dt.minute
        except Exception:
            return

        if server_hour == 0 and server_minute == 0:
            if not self.daily_reset_done:
                account = mt5.account_info()
                if account:
                    self.risk.reset_daily(float(account.balance))
                    self.daily_reset_done = True
                    self.logger.info(
                        f"[DAILY_RESET] Server 00:00"
                        f" | balance={account.balance:.2f}"
                    )
        else:
            # Reset flag khi không còn là 00:00 (để ngày tiếp theo có thể trigger lại)
            self.daily_reset_done = False

    # ==========================================================================
    # HELPERS
    # ==========================================================================

    def _transition(self, new_state: BotState) -> None:
        """Log và thực hiện chuyển state. Publish state_change event cho Dashboard."""
        if self.state != new_state:
            self.logger.info(
                f"[STATE] {self.state.name} → {new_state.name}"
                + (f" | symbol={self.active_symbol}" if self.active_symbol else "")
            )
            # Dashboard publish — put_nowait, zero blocking
            dashboard_pub.publish({
                "type":    "state_change",
                "ts":      int(time.time()),
                "symbol":  self.active_symbol or "",
                "payload": {"old": self.state.name, "new": new_state.name},
            })
        self.state = new_state

    def _update_scan_time(self, symbol: str) -> None:
        """[DEPRECATED] Lưu vết scan — không còn dùng elapsed-time. Giữ lại để tương thích."""
        pass   # last_candle_time được update trong _state_idle()

    def _start_dashboard(self) -> None:
        """
        Khởi động uvicorn trong daemon thread nếu dashboard available.

        Daemon=True đảm bảo Dashboard tự tắt khi main process kết thúc.
        Nếu dashboard.py không available → skip silently, bot vẫn chạy bình thường.
        """
        if not _DASHBOARD_AVAILABLE:
            self.logger.warning(
                "[BotOrchestrator] Dashboard not available "
                "(fastapi/uvicorn not installed) — running without dashboard"
            )
            return

        dash_cfg  = self.cfg.get("dashboard", {})
        host      = dash_cfg.get("host",  "127.0.0.1")
        port      = int(dash_cfg.get("port",  8888))

        dash_thread = threading.Thread(
            target    = start_dashboard_server,
            kwargs    = {"host": host, "port": port, "log_level": "warning"},
            daemon    = True,
            name      = "DashboardServer",
        )
        # Inject DataPipeline reference TRƯỚC khi uvicorn start
        # (uvicorn chạy async — không thể inject sau vì event loop đã khởi chạy)
        _dash_set_pipeline(self.pipeline)
        dash_thread.start()
        self.logger.info(
            f"[BotOrchestrator] Dashboard started | http://{host}:{port}"
        )

    def _close_all_orders_market(self) -> None:
        """Emergency: đóng tất cả positions theo giá market ngay lập tức."""
        for symbol in self.symbols:
            positions = mt5.positions_get(symbol=symbol)
            if not positions:
                continue
            for pos in positions:
                if pos.magic == self.cfg.get("magic_number", 20250305):
                    try:
                        self.executor.close_order_market(
                            ticket  = pos.ticket,
                            volume  = pos.volume,
                            pos_type = pos.type,
                        )
                        self.logger.critical(
                            f"[EMERGENCY] Market closed: {symbol}"
                            f" ticket={pos.ticket}"
                        )
                    except Exception as e:
                        self.logger.error(
                            f"[EMERGENCY] Failed to close {pos.ticket}: {e}"
                        )

    def _get_atr14(self, symbol: str) -> float:
        """
        Tính ATR(14) trực tiếp từ M5 data đã cache trong pipeline.
        Dùng True Range = max(H-L, |H-prev_C|, |L-prev_C|).
        """
        try:
            data = self.pipeline.get_data("M5", symbol=symbol)
            if data is None or len(data) < 16:
                return 0.001   # Fallback an toàn
            # Anti-repainting: chỉ dùng nến đã đóng
            bars   = data[:-1]
            highs  = bars["high"].astype(float)
            lows   = bars["low"].astype(float)
            closes = bars["close"].astype(float)
            n = 14
            trs = np.maximum(
                highs[-n:] - lows[-n:],
                np.maximum(
                    np.abs(highs[-n:] - closes[-n-1:-1]),
                    np.abs(lows[-n:] - closes[-n-1:-1]),
                )
            )
            return float(np.mean(trs))
        except Exception:
            return 0.001

    def _get_position_price(self, ticket: int, symbol: str) -> float:
        """Lấy giá fill của position theo ticket."""
        positions = mt5.positions_get(ticket=ticket)
        if positions:
            return float(positions[0].price_open)
        return 0.0

    def _parse_close_reason(self, deal) -> str:
        """Parse close reason từ MT5 deal object."""
        reason_map = {
            mt5.DEAL_REASON_SL:     "SL",
            mt5.DEAL_REASON_TP:     "TP",
            mt5.DEAL_REASON_CLIENT: "MANUAL",
            mt5.DEAL_REASON_EXPERT: "EA",
        }
        return reason_map.get(getattr(deal, "reason", -1), f"MT5_{getattr(deal, 'reason', '?')}")

    def _sig_handler(self, signum, frame) -> None:
        """SIGTERM / SIGINT handler — graceful shutdown."""
        self.logger.info(f"[BotOrchestrator] Signal {signum} received — stopping...")
        self.stop()
        sys.exit(0)

    @staticmethod
    def _build_logger(path: str, max_bytes: int, backup_count: int) -> logging.Logger:
        """Khởi tạo logger với RotatingFileHandler + StreamHandler."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        logger    = logging.getLogger("RabitScal")
        logger.setLevel(logging.DEBUG)
        fmt       = logging.Formatter(
            "[%(asctime)s UTC] - [%(levelname)-8s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # File handler (rotating)
        fh = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count,
                                  encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(logging.DEBUG)
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        ch.setLevel(logging.INFO)

        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Rabit_Exness AI Trading Bot")
    parser.add_argument(
        "--config", default="config/main_config.json",
        help="Path to main_config.json (default: config/main_config.json)"
    )
    args = parser.parse_args()

    bot = BotOrchestrator(config_path=args.config)
    bot.start()


if __name__ == "__main__":
    main()
