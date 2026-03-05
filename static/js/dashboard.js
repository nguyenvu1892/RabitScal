/**
 * dashboard.js — RabitScal Frontend Logic v1.0
 * =============================================
 * Module: Task 5.2 — Realtime Web Dashboard
 * Author: Antigravity | Date: 2026-03-06
 *
 * Responsibilities:
 *   1. Init Plotly candlestick chart via /api/candles
 *   2. WebSocket listener — 7 event types → UI updates
 *   3. FVG box rendering via Plotly shapes API
 *   4. SL/TP line rendering via Plotly layout.shapes
 *   5. Entry arrow via Plotly annotations
 *   6. Trade history table via /api/trades
 *   7. WS auto-reconnect với exponential backoff
 */

'use strict';

// ─── Config ────────────────────────────────────────────────────────────────
const WS_URL = `ws://${location.host}/ws`;
const CANDLES_API = '/api/candles';
const TRADES_API = '/api/trades?limit=30';
const STATUS_API = '/api/status';
const RECONNECT_MAX = 30000;  // ms — max backoff

// ─── State ──────────────────────────────────────────────────────────────────
let ws = null;
let wsReconnDelay = 1000;
let chartInited = false;

// Track active overlays by ID để update/remove đúng cái
let activeSL = null;   // {id: 'sl-xxxxx', price}
let activeTP = null;   // {id: 'tp-xxxxx', price}
let activeEntry = null;   // annotation index
let fvgShapes = [];     // [{id: 'fvg-...', shape obj}, ...]
const MAX_FVG_BOXES = 5;     // Giữ tối đa 5 FVG boxes trên chart

// Current symbol/tf selection
let currentSymbol = document.getElementById('sel-symbol').value;
let currentTf = document.getElementById('sel-tf').value;

// ─── Plotly chart layout ────────────────────────────────────────────────────
const CHART_LAYOUT = {
    paper_bgcolor: '#0d0f14',
    plot_bgcolor: '#0d0f14',
    margin: { t: 8, b: 40, l: 55, r: 16 },
    xaxis: {
        type: 'date',
        gridcolor: '#1a1d26',
        linecolor: '#2a2d36',
        tickfont: { color: '#5a6370', size: 10 },
        rangeslider: { visible: false },
        showgrid: true,
        tickformat: '%H:%M\n%d/%m',
    },
    yaxis: {
        gridcolor: '#1a1d26',
        linecolor: '#2a2d36',
        tickfont: { color: '#5a6370', size: 10 },
        side: 'right',
        showgrid: true,
        tickformat: '.5f',
    },
    showlegend: false,
    hovermode: 'x unified',
    hoverlabel: {
        bgcolor: '#1a1d26', bordercolor: '#2a2d36',
        font: { color: '#c9d1e0', size: 11 }
    },
    shapes: [],     // SL/TP/FVG shapes injected here
    annotations: [],     // Entry arrows injected here
};

const CHART_CONFIG = {
    displayModeBar: false,
    responsive: true,
};

// Candlestick trace template
const candleTrace = () => ({
    type: 'candlestick',
    name: '',
    x: [],
    open: [],
    high: [],
    low: [],
    close: [],
    increasing: { line: { color: '#00d4a0', width: 1 }, fillcolor: '#00d4a0' },
    decreasing: { line: { color: '#ff4f6d', width: 1 }, fillcolor: '#ff4f6d' },
    whiskerwidth: 0.3,
    hoverinfo: 'x+open+high+low+close',
});

// Volume trace template
const volumeTrace = () => ({
    type: 'bar',
    name: 'Vol',
    x: [],
    y: [],
    yaxis: 'y2',
    marker: { color: [], opacity: 0.4 },
    showlegend: false,
    hoverinfo: 'none',
});

// ─── Init chart ─────────────────────────────────────────────────────────────
async function initChart() {
    const msgEl = document.getElementById('chart-msg');

    try {
        const res = await fetch(`${CANDLES_API}?symbol=${currentSymbol}&tf=${currentTf}&limit=500`);
        const data = await res.json();

        if (!data.x || data.x.length === 0) {
            msgEl.textContent = '⚠️ No candle data — waiting for MT5 connection…';
            return;
        }

        msgEl.classList.add('hidden');

        // Build volume colors (bull=teal, bear=red, 40% opacity)
        const volColors = data.close.map((c, i) =>
            c >= (data.open[i] ?? c) ? 'rgba(0,212,160,0.4)' : 'rgba(255,79,109,0.4)'
        );

        const layout = {
            ...CHART_LAYOUT,
            yaxis2: {
                overlaying: 'y',
                side: 'right',
                showgrid: false,
                showticklabels: false,
                domain: [0, 0.18],     // Volume bars chiếm 18% dưới
            },
            yaxis: { ...CHART_LAYOUT.yaxis, domain: [0.2, 1.0] },
        };

        const ct = candleTrace();
        ct.x = data.x; ct.open = data.open; ct.high = data.high;
        ct.low = data.low; ct.close = data.close;

        const vt = volumeTrace();
        vt.x = data.x; vt.y = data.volume; vt.marker.color = volColors;

        await Plotly.newPlot('chart', [ct, vt], layout, CHART_CONFIG);
        chartInited = true;

        // Attach resize observer
        const resizeObs = new ResizeObserver(() => Plotly.Plots.resize('chart'));
        resizeObs.observe(document.getElementById('chart-wrap'));

        console.log(`[Dashboard] Chart inited | ${data.count} candles | source=${data.source}`);
    } catch (err) {
        msgEl.textContent = `❌ Chart error: ${err.message}`;
        console.error('[Dashboard] initChart error:', err);
    }
}

// ─── Candle update (on candle_close event) ──────────────────────────────────
function onCandleClose(p) {
    // p = {time: ISO_string, o, h, l, c, vol}
    if (!chartInited) return;

    const isGreen = p.c >= p.o;
    const volColor = isGreen ? 'rgba(0,212,160,0.4)' : 'rgba(255,79,109,0.4)';

    // Extend trace 0 (candles) và trace 1 (volume)
    Plotly.extendTraces('chart', {
        x: [[p.time], [p.time]],
        open: [[p.o]], high: [[p.h]],
        low: [[p.l]], close: [[p.c]],
        y: [undefined, [p.vol]],   // trace 0 không cần y, trace 1 cần
        'marker.color': [undefined, [volColor]],
    }, [0, 1]);

    // Giữ tối đa 1000 candles (prune oldest)
    const chartDiv = document.getElementById('chart');
    const currentLen = chartDiv.data[0].x.length;
    if (currentLen > 1000) {
        Plotly.deleteTraces('chart', [0, 1]);
        initChart();  // Re-init từ pipeline nếu quá dài
    }
}

// ─── FVG Box (core snippet cho TechLead review) ────────────────────────────
/**
 * drawFVGBox — Vẽ hộp FVG dùng Plotly layout.shapes (rectangle).
 *
 * Plotly shapes API: thêm shape vào layout.shapes array → Plotly.relayout().
 * Mỗi FVG shape có id riêng để đè/xoá đúng cái.
 *
 * @param {string} fvgId    - Unique ID, dùng để remove khi hết hiệu lực
 * @param {string} dir      - "BUY" | "SELL"
 * @param {number} top      - Giá trên của FVG gap
 * @param {number} bottom   - Giá dưới của FVG gap
 * @param {string} timeFrom - ISO datetime (thời điểm FVG xuất hiện)
 * @param {string} timeTo   - ISO datetime (kéo dài đến đây, hoặc 'now')
 */
function drawFVGBox(fvgId, dir, top, bottom, timeFrom, timeTo) {
    if (!chartInited) return;

    const isBull = dir === 'BUY';
    const fillColor = isBull ? 'rgba(0,212,160,0.10)' : 'rgba(255,79,109,0.10)';
    const lineColor = isBull ? 'rgba(0,212,160,0.35)' : 'rgba(255,79,109,0.35)';

    const shape = {
        type: 'rect',
        xref: 'x',         // tham chiếu theo trục thời gian chart
        yref: 'y',         // tham chiếu theo trục giá
        x0: timeFrom,
        x1: timeTo || new Date().toISOString(),
        y0: bottom,
        y1: top,
        fillcolor: fillColor,
        line: { color: lineColor, width: 1, dash: 'dot' },
        layer: 'below',     // vẽ phía sau nến (không che khuất)
        name: fvgId,       // dùng để identify (Plotly chưa hỗ trợ ID chính thức)
    };

    // Quản lý pool: giữ tối đa MAX_FVG_BOXES
    if (fvgShapes.length >= MAX_FVG_BOXES) {
        fvgShapes.shift();      // Remove oldest FVG
    }
    fvgShapes.push({ id: fvgId, shape });

    // Rebuild tất cả shapes: FVG boxes + SL + TP lines
    _rebuildShapes();
}

// ─── SL/TP + Entry rendering ────────────────────────────────────────────────

function drawOrderOverlay(direction, entry, sl, tp) {
    if (!chartInited) return;

    const now = new Date().toISOString();
    const isBuy = direction === 'BUY';

    // SL line (dashed horizontal red)
    activeSL = {
        id: 'sl-active',
        shape: {
            type: 'line', xref: 'paper', yref: 'y',
            x0: 0, x1: 1,     // xref=paper → span toàn bộ chiều ngang
            y0: sl, y1: sl,
            line: { color: '#ff4f6d', width: 1.5, dash: 'dash' },
            layer: 'above',
        }
    };

    // TP line (dashed horizontal green)
    activeTP = {
        id: 'tp-active',
        shape: {
            type: 'line', xref: 'paper', yref: 'y',
            x0: 0, x1: 1,
            y0: tp, y1: tp,
            line: { color: '#00d4a0', width: 1.5, dash: 'dash' },
            layer: 'above',
        }
    };

    // Entry arrow annotation
    activeEntry = {
        x: now,
        y: entry,
        xref: 'x', yref: 'y',
        text: isBuy ? '▲ BUY' : '▼ SELL',
        font: { color: isBuy ? '#00d4a0' : '#ff4f6d', size: 12, family: 'Inter' },
        showarrow: true,
        arrowcolor: isBuy ? '#00d4a0' : '#ff4f6d',
        arrowhead: 2,
        arrowsize: 1.2,
        ay: isBuy ? 30 : -30,    // arrow points down for BUY (below bar)
        bgcolor: 'rgba(13,15,20,0.8)',
        bordercolor: isBuy ? '#00d4a0' : '#ff4f6d',
        borderwidth: 1, borderpad: 3,
    };

    _rebuildShapes();
    _rebuildAnnotations();
}

function clearOrderOverlay() {
    activeSL = null;
    activeTP = null;
    activeEntry = null;
    _rebuildShapes();
    _rebuildAnnotations();
}

// ─── Internal: rebuild Plotly layout.shapes + annotations ──────────────────
function _rebuildShapes() {
    const shapes = [
        ...fvgShapes.map(f => f.shape),
        ...(activeSL ? [activeSL.shape] : []),
        ...(activeTP ? [activeTP.shape] : []),
    ];
    Plotly.relayout('chart', { shapes });
}

function _rebuildAnnotations() {
    const annotations = activeEntry ? [activeEntry] : [];
    Plotly.relayout('chart', { annotations });
}

// ─── WebSocket Event Handlers ───────────────────────────────────────────────

function onWsMessage(evt) {
    let msg;
    try { msg = JSON.parse(evt.data); }
    catch { return; }

    const type = msg.type;
    const payload = msg.payload ?? {};
    const symbol = msg.symbol ?? '';

    switch (type) {

        case 'snapshot':
            // Toàn bộ state ngay khi connect — populate tất cả widgets
            applySnapshot(payload);
            break;

        case 'candle_close':
            // Chỉ update chart nếu đúng symbol + tf đang xem
            if (symbol === currentSymbol) {
                onCandleClose(payload);
            }
            break;

        case 'state_change':
            updateStateBadge(payload.new);
            break;

        case 'equity_update':
            updateEquityWidgets(payload);
            break;

        case 'signal_found':
            // Chỉ vẽ FVG khi đúng symbol đang xem
            if (symbol === currentSymbol) {
                // x0 = fvg_created_time (gốc thực của FVG từ quá khứ, không phải "now")
                // x1 = 24h tương lai — box kéo dài cho đến khi bị mitigated
                const fvgFrom = payload.fvg_created_time || new Date().toISOString();
                const fvgTo = new Date(Date.now() + 24 * 3600 * 1000).toISOString();
                drawFVGBox(
                    `fvg-${msg.ts}`,
                    payload.direction,
                    payload.fvg_top || payload.entry_price + 0.0005,
                    payload.fvg_bottom || payload.entry_price - 0.0005,
                    fvgFrom,   // ← gốc đúng của FVG (nến tạo ra gap)
                    fvgTo,
                );
                showSignalCard(payload);
            }
            break;

        case 'order_filled':
            // Vẽ SL/TP lines + Entry arrow
            if (symbol === currentSymbol) {
                drawOrderOverlay(payload.direction, payload.entry, payload.sl, payload.tp);
            }
            showOpenTradeCard(payload);
            break;

        case 'trade_closed':
            // Xoá SL/TP/Entry overlays và reload trade table
            clearOrderOverlay();
            hideOpenTradeCard();
            showClosedMarker(payload);
            loadTradeHistory();
            break;

        case 'safety_event':
            showSafetyBanner(payload);
            updateSafetyStatus(payload.level);
            break;

        case 'heartbeat':
        case 'pong':
            // Keepalive — bỏ qua
            break;

        default:
            break;
    }
}

// ─── UI Update functions ────────────────────────────────────────────────────

function applySnapshot(s) {
    updateStateBadge(s.bot_state ?? 'IDLE');
    updateEquityWidgets({
        equity: s.equity ?? 0,
        balance: s.balance ?? 0,
        floating_pnl: s.floating_pnl ?? 0,
        dd_pct: s.daily_dd_pct ?? 0,
    });
    updateSafetyStatus(s.safety_net_status ?? 'active');
    if (s.open_trade) showOpenTradeCard(s.open_trade);
}

function updateStateBadge(state) {
    const badge = document.getElementById('state-badge');
    const text = document.getElementById('state-text');
    text.textContent = state;
    badge.className = 'state-badge';
    if (state === 'IN_TRADE') badge.classList.add('trade');
    else if (state === 'CLOSING') badge.classList.add('warn');
    else if (state === 'IDLE') { }  // default blue
}

function updateEquityWidgets(p) {
    const eq = p.equity ?? 0;
    const bal = p.balance ?? 0;
    const pnl = p.floating_pnl ?? (eq - bal);
    const dd = p.dd_pct ?? 0;

    document.getElementById('equity-num').textContent = fmtMoney(eq);
    document.getElementById('balance-num').textContent = fmtMoney(bal);

    const pnlEl = document.getElementById('pnl-num');
    pnlEl.textContent = `${pnl >= 0 ? '+' : ''}${fmtMoney(pnl)}`;
    pnlEl.className = pnl >= 0 ? 'pos' : 'neg';

    document.getElementById('dd-pct').textContent = `${Math.abs(dd).toFixed(2)}%`;

    // Gauge: limit = 6%, vẽ lên 100% của gauge = 15% (hard halt)
    const pct = Math.min(Math.abs(dd) / 15 * 100, 100);
    const gauge = document.getElementById('dd-gauge');
    gauge.style.width = `${pct}%`;
    gauge.className = 'gauge-fill';
    if (Math.abs(dd) >= 6) gauge.classList.add('warn');
    if (Math.abs(dd) >= 12) gauge.classList.add('danger');
}

function updateSafetyStatus(level) {
    const el = document.getElementById('safety-status');
    const map = {
        'active': ['ACTIVE ✅', '#00d4a0'],
        'cooldown': ['COOLDOWN ⚠️', '#ffc44d'],
        'paused': ['PAUSED ⏸', '#ffc44d'],
        'halted': ['HALTED 🔴', '#ff4f6d'],
    };
    const [text, color] = map[level] ?? ['UNKNOWN', '#5a6370'];
    el.textContent = text;
    el.style.color = color;
}

function showSafetyBanner(p) {
    const banner = document.getElementById('safety-banner');
    banner.textContent = `⚠️ SAFETY: ${p.level?.toUpperCase()} — ${p.reason ?? ''}`;
    banner.classList.add('show');
    setTimeout(() => banner.classList.remove('show'), 8000);
}

function showSignalCard(p) {
    const card = document.getElementById('signal-card');
    card.style.display = '';
    const isBuy = p.direction === 'BUY';
    document.getElementById('signal-dir').textContent = isBuy ? '▲ BUY' : '▼ SELL';
    document.getElementById('signal-dir').style.color = isBuy ? '#00d4a0' : '#ff4f6d';
    document.getElementById('signal-score').textContent = p.score?.toFixed(4) ?? '—';
    document.getElementById('signal-entry').textContent = p.entry_price?.toFixed(5) ?? '—';
}

function showOpenTradeCard(p) {
    const card = document.getElementById('open-trade-card');
    card.style.display = '';
    const isBuy = p.direction === 'BUY';
    document.getElementById('open-trade-body').innerHTML = `
    <div class="trade-row">
      <span>Ticket</span><span>${p.ticket ?? '—'}</span>
    </div>
    <div class="trade-row">
      <span>Dir</span>
      <span class="${isBuy ? 'dir-buy' : 'dir-sell'}">${p.direction ?? '—'}</span>
    </div>
    <div class="trade-row"><span>Entry</span><span>${fmtPrice(p.entry)}</span></div>
    <div class="trade-row">
      <span>SL</span>
      <span style="color:#ff4f6d">${fmtPrice(p.sl)}</span>
    </div>
    <div class="trade-row">
      <span>TP</span>
      <span style="color:#00d4a0">${fmtPrice(p.tp)}</span>
    </div>
    <div class="trade-row"><span>Lot</span><span>${p.lot?.toFixed(2) ?? '—'}</span></div>
  `;
}

function hideOpenTradeCard() {
    document.getElementById('open-trade-card').style.display = 'none';
}

function showClosedMarker(p) {
    // Thêm annotation "CLOSED PnL: +X.XX" tại nến hiện tại
    if (!chartInited) return;
    const isProfit = (p.pnl ?? 0) >= 0;
    const note = {
        x: new Date().toISOString(),
        y: document.getElementById('chart').layout?.yaxis?.range?.[1] ?? 0,
        xref: 'x', yref: 'y',
        text: `${p.close_reason}  ${isProfit ? '+' : ''}${(p.pnl ?? 0).toFixed(2)}`,
        showarrow: false,
        bgcolor: isProfit ? 'rgba(0,212,160,0.2)' : 'rgba(255,79,109,0.2)',
        bordercolor: isProfit ? '#00d4a0' : '#ff4f6d',
        font: { color: isProfit ? '#00d4a0' : '#ff4f6d', size: 11 },
        borderwidth: 1, borderpad: 4,
    };
    const existing = (document.getElementById('chart').layout?.annotations ?? [])
        .filter(a => a.text?.startsWith('▲') || a.text?.startsWith('▼'));
    Plotly.relayout('chart', { annotations: [...existing, note] });

    // Auto-remove annotation sau 5s
    setTimeout(() => {
        Plotly.relayout('chart', { annotations: [] });
    }, 5000);
}

// ─── Trade History Table ────────────────────────────────────────────────────

async function loadTradeHistory() {
    try {
        const res = await fetch(TRADES_API);
        const data = await res.json();
        const trades = data.trades ?? [];
        const tbody = document.getElementById('trade-tbody');

        if (trades.length === 0) {
            tbody.innerHTML = `<tr><td colspan="9" style="color:var(--muted);text-align:center;padding:12px;">No trades yet</td></tr>`;
            return;
        }

        tbody.innerHTML = trades.map(t => {
            const isProfit = parseFloat(t.pnl ?? 0) >= 0;
            const isBuy = (t.direction ?? '').toUpperCase() === 'BUY';
            return `<tr>
        <td>${t.ticket}</td>
        <td class="${isBuy ? 'dir-buy' : 'dir-sell'}">${t.direction}</td>
        <td>${fmtPrice(t.entry_price)}</td>
        <td style="color:#ff4f6d">${fmtPrice(t.sl_price)}</td>
        <td style="color:#00d4a0">${fmtPrice(t.tp_price)}</td>
        <td>${fmtPrice(t.close_price)}</td>
        <td>${parseFloat(t.lot ?? 0).toFixed(2)}</td>
        <td class="${isProfit ? 'pnl-pos' : 'pnl-neg'}">
          ${isProfit ? '+' : ''}${parseFloat(t.pnl ?? 0).toFixed(2)}
        </td>
        <td><span class="reason">${t.close_reason ?? '—'}</span></td>
      </tr>`;
        }).join('');
    } catch (err) {
        console.error('[Dashboard] loadTradeHistory error:', err);
    }
}

// ─── WebSocket connection + auto-reconnect ──────────────────────────────────

function connectWS() {
    const dot = document.getElementById('ws-dot');
    const label = document.getElementById('ws-label');

    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        wsReconnDelay = 1000;   // Reset backoff
        dot.classList.add('connected');
        label.textContent = 'Live';
        console.log('[WS] Connected');

        // Keepalive: gửi ping mỗi 20s
        setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 20000);
    };

    ws.onmessage = onWsMessage;

    ws.onclose = ws.onerror = () => {
        dot.classList.remove('connected');
        label.textContent = `Reconnecting (${Math.round(wsReconnDelay / 1000)}s)…`;
        console.warn(`[WS] Disconnected. Retry in ${wsReconnDelay}ms`);
        setTimeout(connectWS, wsReconnDelay);
        wsReconnDelay = Math.min(wsReconnDelay * 2, RECONNECT_MAX);  // exponential backoff
    };
}

// ─── Symbol / TF selectors ──────────────────────────────────────────────────

function onSymbolChange() {
    currentSymbol = document.getElementById('sel-symbol').value;
    fvgShapes = [];
    clearOrderOverlay();
    chartInited = false;
    initChart();
}

function onTfChange() {
    currentTf = document.getElementById('sel-tf').value;
    fvgShapes = [];
    clearOrderOverlay();
    chartInited = false;
    initChart();
}

// ─── Format helpers ─────────────────────────────────────────────────────────

function fmtMoney(v) {
    const n = parseFloat(v) || 0;
    return '$' + n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function fmtPrice(v) {
    const n = parseFloat(v) || 0;
    return n === 0 ? '—' : n.toFixed(5);
}

// ─── Startup ────────────────────────────────────────────────────────────────

(async function startup() {
    console.log('[Dashboard] Startup — RabitScal v1.0');
    await initChart();
    await loadTradeHistory();
    connectWS();
})();
