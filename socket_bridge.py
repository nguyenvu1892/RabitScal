#!/usr/bin/env python3
"""
socket_bridge.py — RabitScal AI Bridge Server v2.1 (Single Socket + MTF Candles)
==================================================================================
Single TCP connection, bidirectional.

EA sends: TICK (real-time), CANDLES (300 bars × 4 TF), ORDER_RESULT
Server sends: HEARTBEAT, ORDER

CANDLES format from EA:
  {"type":"CANDLES","s":"XAUUSDm","tf":"M5","c":300,
   "d":[[unix,O,H,L,C,vol],...]}

Usage:
  python socket_bridge.py                    # Production (0.0.0.0:15555)
  python socket_bridge.py --host 127.0.0.1   # Localhost only
  python socket_bridge.py --test             # Self-test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import socket
import sys
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# ── AI Pipeline (lazy imports to avoid slowing startup if xgboost not installed)
try:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from core.feature_engine import compute_features, MTFData
    from core.xgb_classifier import get_classifier
    from core.signal_engine import process_ai_signal, schedule_order_send
    _AI_AVAILABLE = True
except ImportError as _e:
    _AI_AVAILABLE = False
    import logging as _lg
    _lg.getLogger("Bridge").warning(f"AI modules not available: {_e}. Running in data-only mode.")

PROJECT_ROOT = Path(__file__).resolve().parent
LOGS_DIR = PROJECT_ROOT / "logs"

HEADER_SIZE = 4
MAX_MSG_SIZE = 2_000_000  # 2MB (candle data can be large)
HEARTBEAT_INTERVAL = 5.0
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 17777

ELITE_5 = ["XAUUSD", "US30", "USTEC", "BTCUSD", "ETHUSD"]
TIMEFRAMES = ["M1", "M5", "M15", "H1"]


def _build_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("Bridge")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)-7s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = RotatingFileHandler(
        LOGS_DIR / "bridge_server.log", maxBytes=10 * 1024 * 1024, backupCount=5
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


log = _build_logger()


class RabitScalBridge:
    """Single-socket TCP Bridge with MTF candle data support."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.host = host
        self.port = port

        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None

        self.running = False
        self.tick_count = 0
        self.last_tick_time: float = 0.0
        self.tick_buffer: dict[str, dict] = {}

        # Candle data: candle_buffer[symbol][tf] = list of [time, O, H, L, C, vol]
        self.candle_buffer: dict[str, dict[str, list]] = {}
        self.candle_update_count: int = 0

        # Spread buffer: spread_buffer[symbol][tf] = spread in points at CANDLES send-time
        # More accurate than tick_buffer because it's atomic with the candle batch.
        self.spread_buffer: dict[str, dict[str, int]] = {}

        self.order_results: dict[str, dict] = {}
        self._order_callbacks: dict[str, asyncio.Future] = {}

        self._start_time: float = 0.0
        self._bytes_received: int = 0
        self._bytes_sent: int = 0
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Event loop reference — captured in start() for thread-safe order sending
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Producer-Consumer AI Queue (Sep Vu 2026-03-16)
        # Socket loop = pure producer (put_nowait, never blocks)
        # AI worker  = single consumer (serial, no GIL contention)
        self._ai_queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        self._ai_worker_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start TCP server."""
        self.running = True
        self._start_time = time.time()

        server = await asyncio.start_server(
            self._on_client_connected,
            self.host, self.port,
        )

        # Capture the running event loop so background threads can schedule sends
        self._loop = asyncio.get_event_loop()

        # Pre-load XGBoost models if available
        if _AI_AVAILABLE:
            clf = get_classifier()
            from core.signal_engine import ALLOWED_SYMBOLS
            loaded = clf.load_all(list(ALLOWED_SYMBOLS))
            log.info(f"[AI] Loaded {loaded} XGBoost models")

        # Start the single AI worker (Consumer) — runs for lifetime of server
        self._ai_worker_task = asyncio.create_task(
            self._ai_worker_loop(), name="ai_worker"
        )
        log.info("[AI] Worker task started (Producer-Consumer queue, maxsize=50)")

        for sock in server.sockets:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        log.info("=" * 60)
        log.info("  RabitScal AI Bridge v2.1 — SINGLE SOCKET + MTF")
        log.info("=" * 60)
        log.info(f"  Address:    tcp://{self.host}:{self.port}")
        log.info(f"  Heartbeat:  every {HEARTBEAT_INTERVAL}s")
        log.info(f"  Symbols:    {ELITE_5}")
        log.info(f"  Timeframes: {TIMEFRAMES}")
        log.info(f"  Max msg:    {MAX_MSG_SIZE:,} bytes")
        log.info("=" * 60)
        log.info("Waiting for MT5 EA to connect...")

        try:
            await server.serve_forever()
        except asyncio.CancelledError:
            log.info("Server shutting down...")
        finally:
            self.running = False
            server.close()

    async def _on_client_connected(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        addr = writer.get_extra_info("peername")
        log.info(f">>> MT5 CONNECTED: {addr}")

        if self._writer and not self._writer.is_closing():
            log.warning(f"Rejecting {addr} — existing connection active")
            writer.close()
            return

        sock = writer.get_extra_info("socket")
        if sock:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        self._reader = reader
        self._writer = writer
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        try:
            while self.running:
                msg = await self._read_message(reader)
                if msg is None:
                    break
                await self._process_message(msg)
        except asyncio.IncompleteReadError:
            log.warning(f"MT5 disconnected unexpectedly: {addr}")
        except ConnectionResetError:
            log.warning(f"MT5 connection reset: {addr}")
        except Exception as e:
            log.error(f"Client error: {type(e).__name__}: {e}")
        finally:
            log.warning(f"<<< MT5 DISCONNECTED: {addr}")
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                self._heartbeat_task = None
            self._reader = None
            self._writer = None
            if not writer.is_closing():
                writer.close()

    # ── Protocol ──────────────────────────────────────────────────

    async def _read_message(self, reader: asyncio.StreamReader) -> Optional[dict]:
        # Dòng hứng data cực kỳ quan trọng Sếp lỡ tay xóa mất
        raw = await reader.readline()
        
        # Nếu không có data (rớt mạng)
        if not raw:
            return None
            
        # Kiểm tra dung lượng
        if len(raw) > MAX_MSG_SIZE:
            log.error(f"Message too large: {len(raw):,} bytes")
            return None
            
        self._bytes_received += len(raw)
        
        try:
            return json.loads(raw.decode("utf-8").strip())
        except json.JSONDecodeError as e:
            log.error(f"Invalid JSON: {e}")
            return None
            
    async def _send_message(self, msg: dict) -> bool:
        if self._writer is None or self._writer.is_closing():
            return False
        try:
            payload = json.dumps(msg, separators=(",", ":")).encode("utf-8") + b"\n"
            self._writer.write(payload)
            await self._writer.drain()
            self._bytes_sent += len(payload)
            return True
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            log.error(f"Send failed: {e}")
            return False

    # ── Message Processing ────────────────────────────────────────

    async def _process_message(self, msg: dict):
        msg_type = msg.get("type", "")
        if msg_type == "TICK":
            self._process_tick(msg)
        elif msg_type == "CANDLES":
            await self._process_candles(msg)
        elif msg_type == "ORDER_RESULT":
            self._process_order_result(msg)
        else:
            log.warning(f"Unknown message type: {msg_type}")

    def _process_tick(self, msg: dict):
        self.tick_count += 1
        self.last_tick_time = time.time()
        for item in msg.get("data", []):
            symbol = item.get("s", "")
            if symbol:
                self.tick_buffer[symbol] = {
                    "bid": item.get("b", 0.0),
                    "ask": item.get("a", 0.0),
                    "spread": item.get("sp", 0),
                    "timestamp": msg.get("ts", ""),
                }
        if self.tick_count % 100 == 0:
            symbols_str = " | ".join(
                f"{s}: {d['bid']:.2f}/{d['ask']:.2f}"
                for s, d in self.tick_buffer.items()
            )
            log.info(f"[TICK #{self.tick_count:,}] {symbols_str}")

    async def _process_candles(self, msg: dict):
        """
        Process CANDLES message from MT5.

        ARCHITECTURE (Event Loop Blocking Fix — Sếp Vũ 2026-03-14):
          1. Store raw data in buffer IMMEDIATELY (non-blocking, ~0ms).
          2. Offload heavy AI processing (Pandas, XGBoost) to a
             background thread via asyncio.to_thread().
          3. The main asyncio event loop stays 100% free to handle
             TICK and HEARTBEAT — no missed beats.

        Format:
        {
            "type": "CANDLES",
            "s": "XAUUSDm",      # symbol
            "tf": "M5",          # timeframe
            "c": 300,            # count
            "d": [               # data array (oldest first)
                [unix_time, open, high, low, close, volume],
                ...
            ]
        }
        """
        symbol = msg.get("s", "")
        tf = msg.get("tf", "")
        count = msg.get("c", 0)
        data = msg.get("d", [])

        if not symbol or not tf or not data:
            log.warning(f"Invalid CANDLES message: s={symbol}, tf={tf}, len={len(data)}")
            return

        # ═══ STEP 1: Store in buffer IMMEDIATELY (non-blocking) ═══
        if symbol not in self.candle_buffer:
            self.candle_buffer[symbol] = {}
        self.candle_buffer[symbol][tf] = data

        # Store spread from CANDLES message (atomic with candle data — Phase 3)
        candle_spread = int(msg.get("sp", 0))
        if symbol not in self.spread_buffer:
            self.spread_buffer[symbol] = {}
        self.spread_buffer[symbol][tf] = candle_spread

        self.candle_update_count += 1

        # Log receipt
        first_time = data[0][0] if data else 0
        last_time = data[-1][0] if data else 0
        last_close = data[-1][4] if data else 0

        log.info(
            f"[CANDLES] {symbol} {tf}: {len(data)} bars | "
            f"last_close={last_close} | "
            f"range={datetime.fromtimestamp(first_time).strftime('%m-%d %H:%M') if first_time else '?'}"
            f" -> {datetime.fromtimestamp(last_time).strftime('%m-%d %H:%M') if last_time else '?'}"
        )

        # ═══ STEP 2: Producer — put into AI queue, return IMMEDIATELY (Sep Vu 2026-03-16) ═══
        # Socket loop is PURE PRODUCER. Never blocks, never touches AI.
        # AI Worker (single consumer) will pick this up serially.
        try:
            self._ai_queue.put_nowait((symbol, tf, data))
        except asyncio.QueueFull:
            log.warning(
                f"[Queue] FULL (size={self._ai_queue.qsize()}) — "
                f"dropping {symbol} {tf} candle batch"
            )

    async def _ai_worker_loop(self):
        """
        Single AI Consumer — Producer-Consumer architecture (Sep Vu 2026-03-16).

        Design:
          - Runs FOREVER as a background asyncio Task.
          - Pulls (symbol, tf, data) tuples from _ai_queue ONE AT A TIME.
          - Runs _run_ai_analysis() in a thread pool via await to_thread().
          - Serial execution: ZERO concurrent threads → no GIL fight,
            no thread-safety race on the asyncio socket writer.

        Thread safety:
          _run_ai_analysis() sends orders via schedule_order_send() which
          uses loop.call_soon_threadsafe() — already safe.
        """
        log.info("[AIWorker] Consumer loop started — waiting for CANDLES jobs")
        while True:
            try:
                symbol, tf, data = await self._ai_queue.get()
                log.info(f"[AIWorker] Got job: {symbol} {tf} | queue_remaining={self._ai_queue.qsize()}")
                await asyncio.to_thread(self._run_ai_analysis, symbol, tf, data)
                self._ai_queue.task_done()
            except asyncio.CancelledError:
                log.info("[AIWorker] Consumer loop cancelled — shutting down")
                break
            except Exception as e:
                import traceback
                log.error(f"[AIWorker] Unexpected error: {e}\n{traceback.format_exc()}")
                # Never crash the loop — keep consuming even on errors

    def _run_ai_analysis(self, symbol: str, tf: str, data: list):
        """
        AI analysis pipeline — runs in a BACKGROUND THREAD via asyncio.to_thread().

        Flow:
            1. Convert candle list → numpy array (CPU, ~0ms)
            2. Build MTFData from candle_buffer (M1/M5/M15/H1 all TFs)
            3. Run feature_engine.compute_features() — CPU, ~5-20ms
            4. XGBoost predict_single() — GPU (GTX 750 Ti), ~2-5ms
            5. signal_engine.process_ai_signal() — Risk Guard + SL/TP
            6. schedule_order_send() — thread-safe back to asyncio event loop

        IMPORTANT: Never touch asyncio objects directly here.
        """
        if not _AI_AVAILABLE:
            return
        # Only run full pipeline on M5 (trigger timeframe)
        if tf != "M5":
            return

        # Normalize symbol name: strip trailing 'm' suffix if EA sends e.g. 'XAUUSDm'
        # EA v3.1 uses clean names, but keep this guard for backward-compatibility.
        clean_symbol = symbol[:-1] if symbol.endswith("m") and len(symbol) > 3 else symbol

        t0 = time.time()
        try:
            import numpy as np

            # 1. Check we have all 4 TF buffers for this symbol
            sym_buffer = self.candle_buffer.get(symbol, {})
            required_tfs = {"M1": "m1", "M5": "m5", "M15": "m15", "H1": "h1"}
            mtf_arrays = {}
            for tf_key, tf_attr in required_tfs.items():
                buf = sym_buffer.get(tf_key, [])
                if len(buf) < 50:
                    log.debug(f"[AI] {clean_symbol}: not enough {tf_key} bars ({len(buf)}) — skipping")
                    return
                mtf_arrays[tf_attr] = np.array(buf, dtype=np.float64)

            # 2. Build MTFData dict
            mtf: MTFData = {
                "m1":    mtf_arrays["m1"],
                "m5":    mtf_arrays["m5"],
                "m15":   mtf_arrays["m15"],
                "h1":    mtf_arrays["h1"],
                "symbol": clean_symbol,
            }

            # 3. Compute features (CPU)
            # compute_features() returns TUPLE: (X_mat, h1_inside_bar)
            # X_mat shape: (N_m5, 73) — full feature matrix
            from core.feature_engine import WARMUP_BARS
            X_mat, _h1_bar = compute_features(mtf, spread_cost=0.0)
            if X_mat is None or len(X_mat) <= WARMUP_BARS:
                log.debug(f"[AI] {clean_symbol}: insufficient feature rows — skipping")
                return

            # Take LAST ROW as 2D array shape (1, 73) — XGBoost predict() requires 2D input
            last_feat = X_mat[-1:].astype(np.float32)   # shape (1, 73)  ← CRITICAL FIX

            # 4. XGBoost predict (GPU)
            clf = get_classifier()
            result = clf.predict_single(last_feat, clean_symbol)
            action     = result["action"]
            confidence = result["confidence"]

            elapsed_ms = (time.time() - t0) * 1000
            log.info(
                f"[AI] {clean_symbol} M5: {action} conf={confidence:.2f} "
                f"in {elapsed_ms:.1f}ms"
            )

            if action == "HOLD":
                return

            # 5a. Spread Filter — block order if spread is too wide
            # Priority: spread from CANDLES msg (atomic) > TICK buffer (fallback)
            live_spread = self.spread_buffer.get(clean_symbol, {}).get("M5", 0)
            if live_spread == 0:
                live_spread = int(
                    (self.tick_buffer.get(clean_symbol) or
                     self.tick_buffer.get(symbol, {})).get("spread", 0)
                )

            from core.signal_engine import check_spread
            sp_ok, sp_reason = check_spread(clean_symbol, live_spread)
            if not sp_ok:
                log.warning(f"[SpreadFilter] {action} {clean_symbol} BLOCKED | {sp_reason}")
                return
            log.info(f"[SpreadFilter] {action} {clean_symbol} PASS | {sp_reason}")

            # 5b. Signal engine: Risk Guard + SL/TP
            # atr_m5_raw is feature index [0, 4] (2D: row 0, col 4)
            atr_m5        = float(last_feat[0, 4]) if last_feat.shape[1] > 4 else 1.0
            current_price = float(mtf_arrays["m5"][-1, 4])  # last M5 close

            # Approximate price digits from magnitude
            digits = 5 if current_price < 1000 else 2

            order_msg = process_ai_signal(
                symbol        = clean_symbol,
                action        = action,
                confidence    = confidence,
                current_price = current_price,
                atr           = atr_m5,
                digits        = digits,
            )

            if order_msg is None:
                return

            # 6. Thread-safe send back to event loop
            if self._loop and self._loop.is_running():
                schedule_order_send(
                    order_msg    = order_msg,
                    send_coro_fn = self._send_message,
                    event_loop   = self._loop,
                )
            else:
                log.warning("[AI] Event loop not running — ORDER dropped")

        except Exception as e:
            # ARMOR: log the error but NEVER re-raise — a symbol failure must NOT
            # crash the socket thread. Other symbols continue normally.
            import traceback
            log.error(
                f"[AI] _run_ai_analysis SWALLOWED error for {clean_symbol}: "
                f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            )

    def _process_order_result(self, msg: dict):
        order_id = msg.get("id", "unknown")
        status = msg.get("status", "UNKNOWN")
        self.order_results[order_id] = msg
        if status == "FILLED":
            log.info(f"[ORDER FILLED] id={order_id} | ticket={msg.get('ticket', 0)}")
        else:
            log.warning(f"[ORDER REJECTED] id={order_id} | error={msg.get('error_msg', '')}")
        if order_id in self._order_callbacks:
            future = self._order_callbacks.pop(order_id)
            if not future.done():
                future.set_result(msg)

    # ── Trade Signal API ──────────────────────────────────────────

    async def send_order(self, action: str, symbol: str, lot: float = 0.01,
                         sl: float = 0.0, tp: float = 0.0, ticket: int = 0,
                         magic: int = 202603, comment: str = "RabitScal_AI",
                         wait_result: bool = False, timeout: float = 10.0) -> Optional[dict]:
        order_id = f"sig_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        msg = {
            "type": "ORDER", "id": order_id, "action": action,
            "symbol": symbol, "lot": lot, "sl": sl, "tp": tp,
            "ticket": ticket, "magic": magic, "comment": comment,
        }
        success = await self._send_message(msg)
        if not success:
            log.error(f"Failed to send order: {order_id}")
            return None
        log.info(f"[ORDER SENT] {action} {symbol} {lot} lot | id={order_id}")
        if wait_result:
            future: asyncio.Future = asyncio.get_event_loop().create_future()
            self._order_callbacks[order_id] = future
            try:
                return await asyncio.wait_for(future, timeout=timeout)
            except asyncio.TimeoutError:
                self._order_callbacks.pop(order_id, None)
                return None
        return {"id": order_id, "status": "SENT"}

    # ── Candle Data API ───────────────────────────────────────────

    def get_candles(self, symbol: str, tf: str) -> list:
        """Get stored candle data for a symbol/timeframe.

        Returns list of [time, open, high, low, close, volume].
        """
        return self.candle_buffer.get(symbol, {}).get(tf, [])

    def get_candle_status(self) -> dict:
        """Get summary of stored candle data."""
        status = {}
        for symbol in self.candle_buffer:
            status[symbol] = {}
            for tf in self.candle_buffer[symbol]:
                data = self.candle_buffer[symbol][tf]
                status[symbol][tf] = {
                    "count": len(data),
                    "last_close": data[-1][4] if data else 0,
                    "last_time": data[-1][0] if data else 0,
                }
        return status

    # ── Heartbeat ─────────────────────────────────────────────────

    async def _heartbeat_loop(self):
        while self.running:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            msg = {
                "type": "HEARTBEAT",
                "ts": datetime.now(timezone.utc).isoformat(),
                "ticks": self.tick_count,
                "candles": self.candle_update_count,
                "uptime": int(time.time() - self._start_time),
            }
            ok = await self._send_message(msg)
            if ok:
                log.info(
                    f"HB SENT | ticks={self.tick_count} "
                    f"| candles={self.candle_update_count} "
                    f"| symbols={list(self.candle_buffer.keys())}"
                )

    # ── Status ────────────────────────────────────────────────────

    @property
    def is_mt5_connected(self) -> bool:
        return self._writer is not None and not self._writer.is_closing()

    def get_status(self) -> dict:
        uptime = time.time() - self._start_time if self._start_time else 0
        return {
            "running": self.running,
            "mt5_connected": self.is_mt5_connected,
            "tick_count": self.tick_count,
            "candle_updates": self.candle_update_count,
            "candle_symbols": list(self.candle_buffer.keys()),
            "uptime_seconds": int(uptime),
            "bytes_received": self._bytes_received,
            "bytes_sent": self._bytes_sent,
        }


# ═══════════════════════════════════════════════════════════════════
# Self-Test
# ═══════════════════════════════════════════════════════════════════

async def _run_self_test(host: str, port: int):
    import random
    log.info("=" * 60)
    log.info("  SELF-TEST MODE")
    log.info("=" * 60)

    bridge = RabitScalBridge(host="127.0.0.1", port=port)
    server_task = asyncio.create_task(bridge.start())
    await asyncio.sleep(1.0)

    passed = failed = 0

    def _send_raw(writer, msg_dict):
        """Helper to send newline-delimited JSON."""
        payload = json.dumps(msg_dict, separators=(",", ":")).encode("utf-8") + b"\n"

        writer.write(payload)

    try:
        # Test 1: Connect
        log.info("\nTest 1: Connect...")
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        log.info("   PASS: Connected")
        passed += 1
        await asyncio.sleep(0.3)

        # Test 2: Send tick
        log.info("\nTest 2: Send tick data...")
        _send_raw(writer, {
            "type": "TICK", "ts": "2026-03-14 20:00:00",
            "data": [
                {"s": "XAUUSDm", "b": 2350.10, "a": 2350.30, "sp": 20},
                {"s": "US30m", "b": 39150.0, "a": 39152.0, "sp": 20},
            ],
        })
        await writer.drain()
        await asyncio.sleep(0.3)
        if bridge.tick_count >= 1:
            log.info(f"   PASS: Tick received (count={bridge.tick_count})")
            passed += 1
        else:
            log.error("   FAIL")
            failed += 1

        # Test 3: Send CANDLES (300 bars M5)
        log.info("\nTest 3: Send 300 candles (XAUUSDm M5)...")
        t0 = 1710400000
        candle_data = [[t0 + i * 300, 2350 + random.uniform(-5, 5),
                         2352 + random.uniform(0, 3), 2348 + random.uniform(-3, 0),
                         2350 + random.uniform(-2, 2), random.randint(50, 500)]
                       for i in range(300)]
        _send_raw(writer, {
            "type": "CANDLES", "s": "XAUUSDm", "tf": "M5",
            "c": 300, "d": candle_data,
        })
        await writer.drain()
        await asyncio.sleep(0.5)
        stored = bridge.get_candles("XAUUSDm", "M5")
        if len(stored) == 300:
            log.info(f"   PASS: 300 candles stored for XAUUSDm M5")
            passed += 1
        else:
            log.error(f"   FAIL: Only {len(stored)} candles stored")
            failed += 1

        # Test 4: Receive heartbeat (same socket!)
        log.info("\nTest 4: Receive heartbeat...")
        try:
            hb_line = await asyncio.wait_for(reader.readline(), timeout=HEARTBEAT_INTERVAL + 2)
            hb_msg = json.loads(hb_line.decode("utf-8").strip())
            if hb_msg.get("type") == "HEARTBEAT":
                log.info(f"   PASS: Heartbeat received!")
                passed += 1
            else:
                log.error(f"   FAIL: Expected HEARTBEAT, got: {hb_msg.get('type')}")
                failed += 1
        except asyncio.TimeoutError:
            log.error("   FAIL: Heartbeat timeout")
            failed += 1

        # Test 5: Send order
        log.info("\nTest 5: Send order signal...")
        order_task = asyncio.create_task(
            bridge.send_order("BUY", "XAUUSDm", lot=0.01, sl=2340.0, tp=2360.0)
        )
        try:
            for _ in range(5):
                ord_line = await asyncio.wait_for(reader.readline(), timeout=5)
                ord_msg = json.loads(ord_line.decode("utf-8").strip())
                if ord_msg.get("type") == "ORDER":
                    log.info(f"   PASS: Order received: {ord_msg['action']} {ord_msg['symbol']}")
                    passed += 1
                    break
            else:
                log.error("   FAIL")
                failed += 1
        except asyncio.TimeoutError:
            log.error("   FAIL: Order timeout")
            failed += 1
        await order_task

    except Exception as e:
        log.error(f"Self-test error: {type(e).__name__}: {e}")
        failed += 1
    finally:
        try:
            writer.close()
        except Exception:
            pass
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    log.info("\n" + "=" * 60)
    log.info(f"  RESULTS: {passed} passed, {failed} failed")
    log.info("=" * 60)
    status = bridge.get_status()
    log.info(f"  Ticks: {status['tick_count']} | Candle updates: {status['candle_updates']}")
    log.info(f"  Bytes recv: {status['bytes_received']:,} | sent: {status['bytes_sent']:,}")
    candle_status = bridge.get_candle_status()
    for sym, tfs in candle_status.items():
        for tf, info in tfs.items():
            log.info(f"  {sym} {tf}: {info['count']} bars | last={info['last_close']:.2f}")
    if failed == 0:
        log.info("  ALL TESTS PASSED!")
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="RabitScal AI Bridge v2.1")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        success = asyncio.run(_run_self_test(args.host, args.port))
        sys.exit(0 if success else 1)
    else:
        bridge = RabitScalBridge(host=args.host, port=args.port)
        loop = asyncio.new_event_loop()

        def _shutdown():
            for task in asyncio.all_tasks(loop):
                task.cancel()

        try:
            loop.run_until_complete(bridge.start())
        except KeyboardInterrupt:
            log.info("Interrupted by user")
        finally:
            loop.close()


if __name__ == "__main__":
    main()
