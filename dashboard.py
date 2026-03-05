"""
dashboard.py — Rabit_Exness AI Web Dashboard Backend v1.0
===========================================================
Module: Rabit_Exness AI — Phase 5, Task 5.2
Branch: task-5.2-dashboard
Author: Antigravity
Date:   2026-03-06

Architecture (Option B — 1 Process):
    BotOrchestrator (main thread) → dashboard_pub.publish() [put_nowait, non-blocking]
        → asyncio.Queue → _queue_drain_loop() → DashboardHub.broadcast() → WS clients

FastAPI app được khởi chạy bởi uvicorn trong daemon thread từ main.py.
Không cần file-based IPC. Communication qua in-memory asyncio.Queue.

Endpoints:
    GET  /                        → Serve index.html
    GET  /api/candles             → M5 OHLCV history (Plotly candlestick data)
    GET  /api/trades              → Trade log history
    GET  /api/status              → Current bot state snapshot
    WS   /ws                      → WebSocket realtime stream

Usage (from main.py):
    from dashboard import app as dashboard_app, dashboard_pub
    # Spawn uvicorn daemon thread, then call dashboard_pub.publish(event)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import MetaTrader5 as mt5
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

# ---------------------------------------------------------------------------
# Constants & Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT  = Path(__file__).resolve().parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"
STATIC_DIR    = PROJECT_ROOT / "static"
TRADE_LOG     = PROJECT_ROOT / "data" / "trade_log.csv"
CONFIG_DIR    = PROJECT_ROOT / "config"

DASHBOARD_HOST    = "127.0.0.1"
DASHBOARD_PORT    = 8888
WS_MAX_CLIENTS    = 10
EVENT_QUEUE_SIZE  = 500   # max events in queue before discard

logger = logging.getLogger("Dashboard")

# ---------------------------------------------------------------------------
# Pipeline Reference — Injected by BotOrchestrator via set_pipeline()
# ---------------------------------------------------------------------------

_pipeline_ref = None   # DataPipeline instance — set sau khi BotOrchestrator init


def set_pipeline(pipeline) -> None:
    """
    Inject DataPipeline reference vào dashboard module.
    Gọi từ BotOrchestrator.__init__() sau khi self.pipeline được tạo.
    Thread-safe: chỉ ghi 1 lần dừi module-level variable.
    """
    global _pipeline_ref
    _pipeline_ref = pipeline
    logger.info("[Dashboard] DataPipeline reference injected")


# ---------------------------------------------------------------------------
# Shared State — populated by BotOrchestrator via dashboard_pub.publish()
# ---------------------------------------------------------------------------

_shared_state: dict[str, Any] = {
    "bot_state":          "IDLE",
    "current_symbol":     "—",
    "equity":             0.0,
    "balance":            0.0,
    "floating_pnl":       0.0,
    "daily_dd_pct":       0.0,
    "daily_dd_limit_pct": 6.0,
    "safety_net_status":  "active",
    "open_trade":         None,
    "uptime_sec":         0,
    "last_updated":       0,
    "start_time":         int(time.time()),
}


def _update_shared_state(event: dict) -> None:
    """Cập nhật shared state từ event — dùng để phục vụ /api/status endpoint."""
    etype   = event.get("type", "")
    payload = event.get("payload", {})

    if etype == "state_change":
        _shared_state["bot_state"]      = payload.get("new", _shared_state["bot_state"])
        _shared_state["current_symbol"] = event.get("symbol", _shared_state["current_symbol"])

    elif etype == "equity_update":
        _shared_state["equity"]         = payload.get("equity",       _shared_state["equity"])
        _shared_state["balance"]        = payload.get("balance",      _shared_state["balance"])
        _shared_state["floating_pnl"]   = payload.get("floating_pnl", _shared_state["floating_pnl"])
        _shared_state["daily_dd_pct"]   = payload.get("dd_pct",       _shared_state["daily_dd_pct"])

    elif etype == "order_filled":
        _shared_state["open_trade"] = payload

    elif etype == "trade_closed":
        _shared_state["open_trade"] = None

    elif etype == "safety_event":
        _shared_state["safety_net_status"] = payload.get("level", "active")

    _shared_state["last_updated"] = int(time.time())
    _shared_state["uptime_sec"]   = int(time.time()) - _shared_state["start_time"]


# ---------------------------------------------------------------------------
# DashboardHub — WebSocket Connection Manager
# ---------------------------------------------------------------------------

class DashboardHub:
    """
    Quản lý tất cả WebSocket connections.
    Thread-safe broadcast qua asyncio event loop.
    """

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        if len(self._clients) >= WS_MAX_CLIENTS:
            await ws.send_json({"type": "error", "payload": {"msg": "Max clients reached"}})
            await ws.close()
            return
        self._clients.add(ws)
        logger.info(f"[WS] Client connected | total={len(self._clients)}")

        # Gửi ngay snapshot hiện tại cho client mới
        snapshot = _build_snapshot()
        try:
            await ws.send_json(snapshot)
        except Exception:
            pass

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.discard(ws)
        logger.info(f"[WS] Client disconnected | total={len(self._clients)}")

    async def broadcast(self, message: dict) -> None:
        """Broadcast JSON tới tất cả connected clients. Tự dọn client đã chết."""
        dead: set[WebSocket] = set()
        for client in list(self._clients):
            try:
                await client.send_json(message)
            except (WebSocketDisconnect, RuntimeError, Exception):
                dead.add(client)
        for d in dead:
            self._clients.discard(d)

    @property
    def client_count(self) -> int:
        return len(self._clients)


# ---------------------------------------------------------------------------
# DashboardPublisher — Non-Blocking Bridge (main.py → asyncio Queue)
# ---------------------------------------------------------------------------

class DashboardPublisher:
    """
    Bridge giữa synchronous main.py và async FastAPI event loop.

    main.py gọi: dashboard_pub.publish(event)   ← put_nowait() — ZERO BLOCKING
    FastAPI drains queue qua: _queue_drain_loop()  ← background asyncio task
    """

    def __init__(self) -> None:
        # Queue được tạo trong asyncio event loop của FastAPI khi startup
        self._queue: Optional[asyncio.Queue] = None
        self._loop:  Optional[asyncio.AbstractEventLoop] = None

    def _init_queue(self, loop: asyncio.AbstractEventLoop) -> None:
        """Được gọi từ FastAPI lifespan startup — phải chạy trong asyncio loop."""
        self._loop  = loop
        self._queue = asyncio.Queue(maxsize=EVENT_QUEUE_SIZE)

    def publish(self, event: dict) -> None:
        """
        Non-blocking publish từ main.py (synchronous context).

        Dùng call_soon_threadsafe → put_nowait vào asyncio Queue.
        Nếu queue đầy → event bị discard (không block trading 1ms).
        """
        if self._loop is None or self._queue is None:
            return  # Dashboard chưa ready — skip silently

        # Cập nhật shared state ngay (thread-safe đủ cho dict simple types)
        _update_shared_state(event)

        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, event)
        except asyncio.QueueFull:
            # Queue đầy → discard oldest bằng cách lấy ra và bỏ
            try:
                self._loop.call_soon_threadsafe(self._queue.get_nowait)
                self._loop.call_soon_threadsafe(self._queue.put_nowait, event)
            except Exception:
                pass  # Discard silently — bot không bao giờ biết
        except Exception:
            pass  # Dashboard issue không ảnh hưởng trading


# ---------------------------------------------------------------------------
# Singletons — import bởi main.py
# ---------------------------------------------------------------------------

dashboard_hub = DashboardHub()
dashboard_pub = DashboardPublisher()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_snapshot() -> dict:
    """Build snapshot event để gửi ngay khi WS client connect."""
    return {
        "type":    "snapshot",
        "ts":      int(time.time()),
        "payload": dict(_shared_state),
    }


def _candles_from_pipeline(symbol: str, tf: str, limit: int) -> dict | None:
    """
    Đọc OHLCV từ DataPipeline cache trước (RAM, zero MT5 call).
    DataPipeline trả về numpy structured ndarray với fields:
        time, open, high, low, close, tick_volume, ...

    Returns:
        Plotly-compatible dict {x, open, high, low, close, volume}
        None nếu pipeline chưa sẵn sàng hoặc không có data cho tf.
    """
    if _pipeline_ref is None:
        return None

    try:
        # get_data() là thread-safe (dùng RLock nội bộ)
        # Trả về numpy structured ndarray hoặc None
        rates = _pipeline_ref.get_data(tf.upper())
        if rates is None or len(rates) == 0:
            return None

        # Anti-repainting: bỏ nến cuối (nến đang chạy chưa đóng)
        rates = rates[:-1] if len(rates) > 1 else rates

        # Clamp theo limit (lấy limit nến cuối)
        if len(rates) > limit:
            rates = rates[-limit:]

        # Numpy structured array → Plotly-compatible dict
        # rates["time"] = unix timestamp (seconds)
        times = [
            datetime.fromtimestamp(int(t), tz=timezone.utc).isoformat()
            for t in rates["time"]
        ]
        return {
            "x":      times,
            "open":   [float(v) for v in rates["open"]],
            "high":   [float(v) for v in rates["high"]],
            "low":    [float(v) for v in rates["low"]],
            "close":  [float(v) for v in rates["close"]],
            "volume": [int(v)   for v in rates["tick_volume"]],
        }
    except Exception as e:
        logger.warning(f"[Dashboard] Pipeline candles error ({tf}): {e}")
        return None


def _mt5_candles_to_plotly(rates) -> dict:
    """
    Convert MT5 structured array → Plotly candlestick trace dict.

    Returns dict với keys: x, open, high, low, close, volume
    Plotly Go.Candlestick() nhận trực tiếp format này.
    """
    if rates is None or len(rates) == 0:
        return {"x": [], "open": [], "high": [], "low": [], "close": [], "volume": []}

    # MT5 timestamp (unix seconds) → ISO datetime string (Plotly dùng)
    times  = [datetime.fromtimestamp(int(r["time"]), tz=timezone.utc).isoformat() for r in rates]
    return {
        "x":      times,
        "open":   [float(r["open"])        for r in rates],
        "high":   [float(r["high"])        for r in rates],
        "low":    [float(r["low"])         for r in rates],
        "close":  [float(r["close"])       for r in rates],
        "volume": [int(r["tick_volume"])   for r in rates],
    }


_TF_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}


def _fetch_candles_from_mt5(symbol: str, tf: str, limit: int) -> dict:
    """
    Fetch candle data trực tiếp từ MT5.
    MT5 phải đã initialize (được đảm bảo bởi BotOrchestrator startup).
    """
    mt5_tf = _TF_MAP.get(tf.upper(), mt5.TIMEFRAME_M5)
    rates  = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, limit)
    return _mt5_candles_to_plotly(rates)


def _load_trade_history(limit: int = 50) -> list[dict]:
    """
    Đọc trade_log.csv từ execution.py.
    CSV format: ticket,symbol,direction,entry_price,sl_price,tp_price,
                close_price,lot,pnl,open_time,close_time,close_reason,
                signal_score,slippage_pips,spread,commission,attempts
    """
    if not TRADE_LOG.exists():
        return []

    trades: list[dict] = []
    try:
        with open(TRADE_LOG, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if len(lines) < 2:
            return []

        headers = [h.strip() for h in lines[0].split(",")]
        for line in reversed(lines[1:]):
            if len(trades) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < len(headers):
                continue
            row = dict(zip(headers, parts))
            # Type cast
            for fld in ("entry_price", "sl_price", "tp_price", "close_price",
                        "lot", "pnl", "signal_score", "slippage_pips",
                        "spread", "commission"):
                try:
                    row[fld] = float(row.get(fld, 0))
                except (ValueError, TypeError):
                    row[fld] = 0.0
            for fld in ("ticket", "open_time", "close_time", "attempts"):
                try:
                    row[fld] = int(row.get(fld, 0))
                except (ValueError, TypeError):
                    row[fld] = 0
            trades.append(row)
    except Exception as e:
        logger.error(f"[Dashboard] Failed to read trade_log.csv: {e}")

    return trades


# ---------------------------------------------------------------------------
# FastAPI App + Lifespan
# ---------------------------------------------------------------------------

from contextlib import asynccontextmanager


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Lifespan: khởi tạo queue + background drain task khi startup."""
    loop = asyncio.get_event_loop()
    dashboard_pub._init_queue(loop)
    logger.info(f"[Dashboard] Event queue initialized | maxsize={EVENT_QUEUE_SIZE}")

    # Background task: drain queue → broadcast
    async def _drain_loop():
        while True:
            try:
                event = await dashboard_pub._queue.get()
                if dashboard_hub.client_count > 0:
                    await dashboard_hub.broadcast(event)
                dashboard_pub._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[Dashboard] drain_loop error: {e}")

    drain_task = asyncio.create_task(_drain_loop())
    logger.info(f"[Dashboard] Background drain task started")

    yield  # ← App running

    drain_task.cancel()
    logger.info("[Dashboard] Shutdown complete")


app = FastAPI(
    title="RabitScal Dashboard",
    description="Realtime trading bot monitor — Rabit_Exness AI",
    version="1.0.0",
    lifespan=_lifespan,
)

# Mount static files (CSS, JS)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Templates
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve main dashboard page."""
    index_path = TEMPLATES_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse(
            content="<h1>Dashboard template not found.</h1>"
                    "<p>Please create <code>templates/index.html</code></p>",
            status_code=404,
        )
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/candles")
async def get_candles(
    symbol: str = "EURUSDc",
    tf:     str = "M5",
    limit:  int = 500,
):
    """
    Fetch M5 OHLCV cho Plotly candlestick chart.

    Ưu tiên đọc từ DataPipeline cache (RAM, zero MT5 call).
    Chỉ fallback sang mt5.copy_rates_from_pos() khi pipeline chưa ready.

    Điều này ngăn spam MT5 Terminal khi user F5 liên tục.
    """
    limit = max(10, min(limit, 2000))

    try:
        # ƯU TIÊN 1: Đọc từ DataPipeline cache (no MT5 call)
        data = _candles_from_pipeline(symbol, tf, limit)

        if data is not None:
            source = "pipeline_cache"

        else:
            # FALLBACK: Pipeline chưa ready, gọi MT5 trực tiếp
            logger.info(
                f"[Dashboard] Pipeline cache miss ({symbol}/{tf}) — "
                f"falling back to mt5.copy_rates_from_pos()"
            )
            data = _fetch_candles_from_mt5(symbol, tf, limit)
            source = "mt5_direct"

        if not data or not data.get("x"):
            return JSONResponse(content={
                "error":  f"No data for {symbol}/{tf} — MT5 connected?",
                "source": source if 'source' in dir() else "unknown",
                "x": [], "open": [], "high": [], "low": [], "close": [], "volume": [],
            }, status_code=200)

        return JSONResponse(content={
            "symbol": symbol,
            "tf":     tf,
            "count":  len(data["x"]),
            "source": source,
            **data,
        })

    except Exception as e:
        logger.error(f"[Dashboard] /api/candles error: {e}", exc_info=True)
        return JSONResponse(
            content={"error": str(e), "x": []},
            status_code=500,
        )


@app.get("/api/trades")
async def get_trades(limit: int = 50, status: str = "all"):
    """
    Đọc lịch sử trade từ data/trade_log.csv.
    Params:
        limit:  số lệnh trả về (mới nhất trước), max 200
        status: "all" | "win" | "loss" — filter theo PnL
    """
    limit = max(1, min(limit, 200))
    trades = _load_trade_history(limit * 3)  # Lấy dư để filter

    if status == "win":
        trades = [t for t in trades if float(t.get("pnl", 0)) > 0]
    elif status == "loss":
        trades = [t for t in trades if float(t.get("pnl", 0)) < 0]

    return JSONResponse(content={
        "count":  len(trades[:limit]),
        "trades": trades[:limit],
    })


@app.get("/api/status")
async def get_status():
    """
    Current bot state snapshot.
    Đọc từ _shared_state được cập nhật realtime bởi dashboard_pub.publish().
    Fallback: đọc từ config/state.json nếu shared_state chưa được populate.
    """
    state = dict(_shared_state)

    # Fallback: đọc state.json nếu bot chưa publish equity update nào
    if state["balance"] == 0.0:
        state_json = CONFIG_DIR / "state.json"
        if state_json.exists():
            try:
                with open(state_json) as f:
                    persisted = json.load(f)
                state["balance"] = persisted.get("initial_balance", 0.0)
            except Exception:
                pass

    return JSONResponse(content=state)


@app.get("/api/health")
async def health():
    """Health check endpoint — dùng để kiểm tra server còn sống."""
    return {"status": "ok", "ts": int(time.time()), "ws_clients": dashboard_hub.client_count}


# ---------------------------------------------------------------------------
# WebSocket Endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    WebSocket endpoint — realtime event stream từ BotOrchestrator.

    Client nhận JSON events:
        {"type": "state_change",  "ts": 1741194000, "symbol": "EURUSDc", "payload": {...}}
        {"type": "candle_close",  "ts": ..., "payload": {time, o, h, l, c, vol}}
        {"type": "order_filled",  "ts": ..., "payload": {ticket, direction, entry, sl, tp}}
        {"type": "equity_update", "ts": ..., "payload": {equity, balance, dd_pct}}
        {"type": "trade_closed",  "ts": ..., "payload": {ticket, pnl, reason}}
        {"type": "signal_found",  "ts": ..., "payload": {direction, score, fvg, entry}}
        {"type": "safety_event",  "ts": ..., "payload": {level, reason}}
        {"type": "snapshot",      ...}       ← Gửi ngay sau khi connect

    Client có thể gửi ping để keepalive:
        {"type": "ping"}  →  server trả {"type": "pong"}
    """
    await dashboard_hub.connect(ws)
    try:
        while True:
            try:
                data = await asyncio.wait_for(ws.receive_json(), timeout=30.0)
                if data.get("type") == "ping":
                    await ws.send_json({"type": "pong", "ts": int(time.time())})
            except asyncio.TimeoutError:
                # Keepalive: gửi heartbeat nếu không nhận message trong 30s
                await ws.send_json({"type": "heartbeat", "ts": int(time.time())})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"[WS] Error: {e}")
    finally:
        dashboard_hub.disconnect(ws)


# ---------------------------------------------------------------------------
# Uvicorn launcher — được gọi bởi main.py trong daemon thread
# ---------------------------------------------------------------------------

def start_dashboard_server(
    host:      str = DASHBOARD_HOST,
    port:      int = DASHBOARD_PORT,
    log_level: str = "warning",
) -> None:
    """
    Khởi động uvicorn server trong thread hiện tại.
    Được gọi từ main.py trong daemon thread:

        import threading
        from dashboard import start_dashboard_server
        t = threading.Thread(target=start_dashboard_server, daemon=True)
        t.start()

    Daemon=True đảm bảo Dashboard tự tắt khi main process kết thúc.
    """
    try:
        import uvicorn
        logger.info(f"[Dashboard] Starting uvicorn on http://{host}:{port}")
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level=log_level,
            access_log=False,   # Tắt access log để không spam console
        )
    except ImportError:
        logger.error("[Dashboard] uvicorn not installed. Run: pip install uvicorn[standard]")
    except Exception as e:
        logger.error(f"[Dashboard] Server error: {e}", exc_info=True)


# ---------------------------------------------------------------------------
# Standalone run (for testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Dashboard in standalone mode (no bot data)")
    uvicorn.run(app, host=DASHBOARD_HOST, port=DASHBOARD_PORT, log_level="info")
