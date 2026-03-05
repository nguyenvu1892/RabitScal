"""
backtest_env.py — BacktestEnv v1.0
=====================================
Module: Rabit_Exness AI — Phase 6, Task 6
Branch: task-6-backtest-env
Author: Antigravity
Date:   2026-03-06

Backtest Engine chuyên dụng xuất báo cáo:
    • Chạy walk-through simulation trên dữ liệu lịch sử
    • Export chi tiết lệnh ra CSV (trade_log)
    • Generate Equity Curve + Drawdown Chart dạng HTML (Plotly)
    • OHLC worst-case model (kiểm tra SL trước — bất lợi nhất cho bot)
    • Slippage model: gauss(μ=avg_spread, σ=spread_std) per trade

Usage:
    from backtest_env import BacktestEnv
    env = BacktestEnv(config)
    report = env.run(data, params)
    env.export_trade_log(report, "data/trade_log.csv")
    env.generate_html_report(report, "reports/backtest_report.html")

CLI:
    python backtest_env.py --params config/current_settings.json
    python backtest_env.py --params config/versions/settings_v001.json --out reports/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import numpy as np

# Plotly import — graceful fallback nếu chưa cài
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data"
REPORTS_DIR  = PROJECT_ROOT / "reports"
LOGS_DIR     = PROJECT_ROOT / "logs"

DEFAULT_COMMISSION_PER_LOT = 3.5    # USD/lot (Exness Standard Cent)
DEFAULT_LOT_SIZE           = 0.01   # 1 micro-lot per trade
DEFAULT_RR_RATIO           = 1.5    # TP = SL × 1.5
DEFAULT_AVG_SPREAD         = 0.0002 # ~2 pips EURUSD Cent avg spread
DEFAULT_SPREAD_STD         = 0.0001 # Spread volatility

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _build_logger(name: str) -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "[%(asctime)s UTC] - [%(levelname)-8s] - [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = RotatingFileHandler(
        LOGS_DIR / "backtest_env.log", maxBytes=5 * 1024 * 1024, backupCount=3
    )
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = _build_logger("BacktestEnv")

# ---------------------------------------------------------------------------
# Data Containers
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """Chi tiết 1 lệnh — exported ra CSV và hiển thị trên HTML report."""
    trade_id:       int
    open_time:      float      # Unix timestamp
    close_time:     float
    direction:      str        # "BUY" | "SELL"
    entry_price:    float
    sl_price:       float
    tp_price:       float
    close_price:    float
    close_reason:   str        # "TP_HIT" | "SL_HIT" | "FORCE_CLOSE"
    pnl_raw:        float      # PnL trước phí
    commission:     float
    slippage:       float
    pnl_net:        float      # PnL sau phí + slippage
    cumulative_pnl: float      # Equity tại điểm đóng lệnh
    atr_at_entry:   float


@dataclass
class BacktestReport:
    """Kết quả đầy đủ 1 backtest run."""
    # ── Metadata ──────────────────────────────────────────────────────────
    created_at:    str
    params:        dict
    data_shape:    tuple
    duration_sec:  float

    # ── Aggregate Metrics ─────────────────────────────────────────────────
    winrate:       float
    profit_factor: float
    max_drawdown:  float       # Max drawdown % (peak-to-trough)
    trade_count:   int
    gross_profit:  float
    gross_loss:    float
    avg_win:       float
    avg_loss:      float
    avg_rrr:       float       # Realized RR ratio (avg_win / avg_loss)
    sharpe_ratio:  float       # Simplified Sharpe trên PnL series
    calmar_ratio:  float       # Ann. return / Max DD

    # ── Trade List ────────────────────────────────────────────────────────
    trades:        list[TradeRecord] = field(default_factory=list)

    # ── Arrays (for charting) ─────────────────────────────────────────────
    equity_curve:  Optional[np.ndarray] = None   # Cumulative PnL per trade
    drawdown_pct:  Optional[np.ndarray] = None   # Drawdown % per trade
    timestamps:    Optional[np.ndarray] = None   # Unix timestamps per trade close


# ---------------------------------------------------------------------------
# BacktestEnv — Main Class
# ---------------------------------------------------------------------------

class BacktestEnv:
    """
    Backtest engine chuyên dụng tạo báo cáo chi tiết.

    Pipeline:
        data (numpy) + params (dict)
        → run()            → BacktestReport (trades + metrics)
        → export_trade_log() → data/trade_log_{ts}.csv
        → generate_html_report() → reports/backtest_{ts}.html (Plotly)
    """

    def __init__(
        self,
        *,
        commission_per_lot: float = DEFAULT_COMMISSION_PER_LOT,
        lot_size:           float = DEFAULT_LOT_SIZE,
        rr_ratio:           float = DEFAULT_RR_RATIO,
        avg_spread:         float = DEFAULT_AVG_SPREAD,
        spread_std:         float = DEFAULT_SPREAD_STD,
        rng_seed:           int   = 42,
    ):
        self.commission_per_lot = commission_per_lot
        self.lot_size           = lot_size
        self.rr_ratio           = rr_ratio
        self.avg_spread         = avg_spread
        self.spread_std         = spread_std
        self._rng               = np.random.default_rng(rng_seed)

    # ── Public API ──────────────────────────────────────────────────────────

    def run(self, data: np.ndarray, params: dict) -> BacktestReport:
        """
        Chạy full backtest simulation trên data với params.

        Args:
            data:   np.ndarray shape (N, 6) — [time, open, high, low, close, volume]
            params: dict bộ tham số từ Optuna best trial

        Returns:
            BacktestReport với danh sách lệnh đầy đủ + các aggregate metrics
        """
        t_start = __import__("time").perf_counter()
        logger.info(
            f"[BacktestEnv] run() started | candles={len(data):,} | params={params}"
        )

        N       = len(data)
        times   = data[:, 0].astype(np.float64)
        opens   = data[:, 1].astype(np.float64)
        highs   = data[:, 2].astype(np.float64)
        lows    = data[:, 3].astype(np.float64)
        closes  = data[:, 4].astype(np.float64)
        volumes = data[:, 5].astype(np.float64)

        # ── [1] Indicators (vectorized) ─────────────────────────────────────
        prev_closes  = np.roll(closes, 1); prev_closes[0] = closes[0]
        tr           = np.maximum(highs - lows,
                           np.maximum(np.abs(highs - prev_closes),
                                      np.abs(lows  - prev_closes)))
        atr          = np.convolve(tr, np.ones(14) / 14.0, mode="same")
        atr[:13]     = tr[:13]

        vol_ma20     = np.convolve(volumes, np.ones(20) / 20.0, mode="same")
        vol_ma20[:19] = volumes[:19]
        prev_volumes = np.roll(volumes, 1); prev_volumes[0] = volumes[0]

        # ── [2] Signal masks ────────────────────────────────────────────────
        total_range = highs - lows + 1e-10
        body        = np.abs(closes - opens)
        upper_wick  = highs - np.maximum(closes, opens)
        lower_wick  = np.minimum(closes, opens) - lows
        max_wick    = np.maximum(upper_wick, lower_wick)

        pinbar_mask = (
            (max_wick / total_range >= params["pinbar_wick_ratio"]) &
            (body     / total_range <= params["pinbar_body_ratio"])
        )
        bull_pinbar = pinbar_mask & (lower_wick > upper_wick)
        bear_pinbar = pinbar_mask & (upper_wick > lower_wick)

        vsa_mask  = (
            (volumes >= vol_ma20 * params["vsa_volume_ratio"]) &
            (volumes >= prev_volumes * params["vsa_neighbor_ratio"])
        )

        highs_2ago = np.roll(highs, 2); highs_2ago[:2] = highs[:2]
        lows_2ago  = np.roll(lows,  2); lows_2ago[:2]  = lows[:2]
        fvg_bull   = (highs_2ago < lows)  & ((lows - highs_2ago)  >= atr * params["atr_fvg_buffer"])
        fvg_bear   = (lows_2ago  > highs) & ((lows_2ago - highs)  >= atr * params["atr_fvg_buffer"])

        gate        = params["composite_score_gate"]
        buy_score   = (np.where(pinbar_mask, 0.40, 0) + np.where(vsa_mask, 0.35, 0)
                       + np.where(fvg_bull, 0.25, 0))
        sell_score  = (np.where(pinbar_mask, 0.40, 0) + np.where(vsa_mask, 0.35, 0)
                       + np.where(fvg_bear, 0.25, 0))

        buy_signal  = bull_pinbar & vsa_mask & fvg_bull & (buy_score  >= gate)
        sell_signal = bear_pinbar & vsa_mask & fvg_bear & (sell_score >= gate)
        buy_signal[:15] = sell_signal[:15] = False

        # ── [3] Trade simulation — OHLC worst-case + slippage model ─────────
        sl_dist    = atr * params["atr_sl_multiplier"]
        commission = self.commission_per_lot * self.lot_size

        trades:   list[TradeRecord] = []
        trade_id  = 0
        in_trade  = False
        entry_i   = 0
        direction = 0  # 1=BUY, -1=SELL
        entry_price = sl_price = tp_price = 0.0

        for i in range(15, N - 1):
            if not in_trade:
                if buy_signal[i]:
                    slip        = abs(self._rng.normal(self.avg_spread, self.spread_std))
                    entry_price = closes[i] + slip
                    sl_price    = entry_price - sl_dist[i]
                    tp_price    = entry_price + sl_dist[i] * self.rr_ratio
                    direction   = 1
                    in_trade    = True
                    entry_i     = i
                elif sell_signal[i]:
                    slip        = abs(self._rng.normal(self.avg_spread, self.spread_std))
                    entry_price = closes[i] - slip
                    sl_price    = entry_price + sl_dist[i]
                    tp_price    = entry_price - sl_dist[i] * self.rr_ratio
                    direction   = -1
                    in_trade    = True
                    entry_i     = i
            else:
                next_h = highs[i]
                next_l = lows[i]
                close_price = None
                close_reason = ""

                if direction == 1:
                    if next_l <= sl_price:
                        close_price  = sl_price
                        close_reason = "SL_HIT"
                    elif next_h >= tp_price:
                        close_price  = tp_price
                        close_reason = "TP_HIT"
                else:
                    if next_h >= sl_price:
                        close_price  = sl_price
                        close_reason = "SL_HIT"
                    elif next_l <= tp_price:
                        close_price  = tp_price
                        close_reason = "TP_HIT"

                if close_price is not None:
                    price_diff  = (close_price - entry_price) * direction
                    slip_exit   = abs(self._rng.normal(self.avg_spread / 2, self.spread_std / 2))
                    pnl_raw     = price_diff
                    pnl_net     = pnl_raw - commission - slip_exit
                    cum_pnl     = (trades[-1].cumulative_pnl if trades else 0.0) + pnl_net

                    trades.append(TradeRecord(
                        trade_id      = trade_id,
                        open_time     = times[entry_i],
                        close_time    = times[i],
                        direction     = "BUY" if direction == 1 else "SELL",
                        entry_price   = round(entry_price, 6),
                        sl_price      = round(sl_price, 6),
                        tp_price      = round(tp_price, 6),
                        close_price   = round(close_price, 6),
                        close_reason  = close_reason,
                        pnl_raw       = round(pnl_raw, 6),
                        commission    = round(commission, 6),
                        slippage      = round(slip_exit, 6),
                        pnl_net       = round(pnl_net, 6),
                        cumulative_pnl= round(cum_pnl, 6),
                        atr_at_entry  = round(atr[entry_i], 6),
                    ))
                    trade_id += 1
                    in_trade  = False

        # ── [4] Aggregate Metrics ────────────────────────────────────────────
        if not trades:
            logger.warning("[BacktestEnv] No trades generated — check signal params")
            return BacktestReport(
                created_at=datetime.now(timezone.utc).isoformat(),
                params=params, data_shape=data.shape,
                duration_sec=0, winrate=0, profit_factor=0,
                max_drawdown=1, trade_count=0, gross_profit=0,
                gross_loss=0, avg_win=0, avg_loss=0, avg_rrr=0,
                sharpe_ratio=0, calmar_ratio=0,
            )

        pnl_arr   = np.array([t.pnl_net for t in trades], dtype=np.float64)
        wins      = pnl_arr > 0
        losses    = pnl_arr < 0
        n_wins    = wins.sum()
        n_loss    = losses.sum()

        gross_profit  = float(pnl_arr[wins].sum())  if n_wins > 0 else 0.0
        gross_loss    = float(abs(pnl_arr[losses].sum())) if n_loss > 0 else 1e-8
        profit_factor = gross_profit / gross_loss
        avg_win       = gross_profit / n_wins if n_wins > 0 else 0.0
        avg_loss      = gross_loss   / n_loss if n_loss > 0 else 0.0
        winrate       = float(n_wins / len(pnl_arr))

        equity        = np.array([t.cumulative_pnl for t in trades])
        peak          = np.maximum.accumulate(equity)
        dd_abs        = peak - equity
        dd_pct        = np.where(peak > 0, dd_abs / (peak + 1e-8), 0.0)
        max_dd        = float(dd_pct.max())

        # Sharpe (simplified — annualized assuming 1 trade ≠ 1 period)
        pnl_mean  = pnl_arr.mean()
        pnl_std   = pnl_arr.std() + 1e-10
        sharpe    = float((pnl_mean / pnl_std) * np.sqrt(252))

        # Calmar = Total Return / Max Drawdown
        total_ret = float(equity[-1]) if len(equity) > 0 else 0.0
        calmar    = (total_ret / (max_dd * abs(equity[-1]) + 1e-8)) if max_dd > 0 else 0.0

        timestamps_arr = np.array([t.close_time for t in trades])

        duration = __import__("time").perf_counter() - t_start
        logger.info(
            f"[BacktestEnv] run() done | "
            f"trades={len(trades)} WR={winrate:.3f} PF={profit_factor:.3f} "
            f"MaxDD={max_dd:.3f} Sharpe={sharpe:.2f} | {duration:.3f}s"
        )

        return BacktestReport(
            created_at    = datetime.now(timezone.utc).isoformat(),
            params        = params,
            data_shape    = data.shape,
            duration_sec  = duration,
            winrate       = winrate,
            profit_factor = profit_factor,
            max_drawdown  = max_dd,
            trade_count   = len(trades),
            gross_profit  = gross_profit,
            gross_loss    = gross_loss,
            avg_win       = avg_win,
            avg_loss      = abs(avg_loss),
            avg_rrr       = avg_win / (abs(avg_loss) + 1e-8),
            sharpe_ratio  = sharpe,
            calmar_ratio  = calmar,
            trades        = trades,
            equity_curve  = equity,
            drawdown_pct  = dd_pct,
            timestamps    = timestamps_arr,
        )

    # ── Report Exporters ────────────────────────────────────────────────────

    def export_trade_log(
        self,
        report:   BacktestReport,
        out_path: Optional[str] = None,
    ) -> Path:
        """
        Xuất danh sách lệnh chi tiết ra file CSV.

        Cột CSV:
            trade_id, open_time (UTC ISO), close_time (UTC ISO),
            direction, entry_price, sl_price, tp_price, close_price,
            close_reason, pnl_raw, commission, slippage, pnl_net,
            cumulative_pnl, atr_at_entry, winrate_running, pf_running

        Args:
            report:   BacktestReport từ run()
            out_path: Path output CSV — auto-generated nếu None

        Returns:
            Path to written CSV file
        """
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        if out_path is None:
            ts       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out_path = str(DATA_DIR / f"trade_log_{ts}.csv")

        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "trade_id", "open_time", "close_time",
            "direction", "entry_price", "sl_price", "tp_price",
            "close_price", "close_reason",
            "pnl_raw", "commission", "slippage", "pnl_net",
            "cumulative_pnl", "atr_at_entry",
            "winrate_running", "pf_running",
        ]

        n_wins_running = 0
        gross_w        = 0.0
        gross_l        = 0.0

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for t in report.trades:
                if t.pnl_net > 0:
                    n_wins_running += 1
                    gross_w        += t.pnl_net
                else:
                    gross_l        += abs(t.pnl_net)

                idx = t.trade_id + 1  # 1-indexed
                wr_run = n_wins_running / idx
                pf_run = gross_w / (gross_l + 1e-8)

                writer.writerow({
                    "trade_id":       t.trade_id,
                    "open_time":      datetime.fromtimestamp(t.open_time,  tz=timezone.utc).isoformat(),
                    "close_time":     datetime.fromtimestamp(t.close_time, tz=timezone.utc).isoformat(),
                    "direction":      t.direction,
                    "entry_price":    t.entry_price,
                    "sl_price":       t.sl_price,
                    "tp_price":       t.tp_price,
                    "close_price":    t.close_price,
                    "close_reason":   t.close_reason,
                    "pnl_raw":        round(t.pnl_raw,  6),
                    "commission":     round(t.commission, 6),
                    "slippage":       round(t.slippage, 6),
                    "pnl_net":        round(t.pnl_net,  6),
                    "cumulative_pnl": round(t.cumulative_pnl, 6),
                    "atr_at_entry":   round(t.atr_at_entry, 6),
                    "winrate_running": round(wr_run, 4),
                    "pf_running":      round(pf_run, 4),
                })

        logger.info(
            f"[BacktestEnv] Trade log exported | "
            f"{len(report.trades)} trades | {path}"
        )
        return path

    def generate_html_report(
        self,
        report:   BacktestReport,
        out_path: Optional[str] = None,
    ) -> Path:
        """
        Tạo báo cáo HTML tương tác với Plotly:
            Row 1: Equity Curve (line) + TP/SL/Entry markers
            Row 2: Drawdown Chart (area, filled red)
            Row 3: Trade PnL Bar chart (green/red per trade)
            Sidebar: Metrics summary table

        Args:
            report:   BacktestReport từ run()
            out_path: Path output HTML — auto-generated nếu None

        Returns:
            Path to written HTML file

        Raises:
            ImportError: nếu plotly chưa được cài đặt
        """
        if not _PLOTLY_AVAILABLE:
            raise ImportError(
                "plotly is not installed. Run: pip install plotly"
            )

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        if out_path is None:
            ts       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out_path = str(REPORTS_DIR / f"backtest_report_{ts}.html")

        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not report.trades:
            logger.warning("[BacktestEnv] No trades to chart — skipping HTML report")
            return path

        # ── Prepare series ───────────────────────────────────────────────────
        trade_ids  = [t.trade_id    for t in report.trades]
        close_dts  = [
            datetime.fromtimestamp(t.close_time, tz=timezone.utc).isoformat()
            for t in report.trades
        ]
        equity_arr = report.equity_curve
        dd_arr     = report.drawdown_pct * 100  # Convert to %
        pnl_arr    = [t.pnl_net for t in report.trades]
        directions = [t.direction     for t in report.trades]
        reasons    = [t.close_reason  for t in report.trades]

        colors_pnl = [
            "#26a69a" if p > 0 else "#ef5350"
            for p in pnl_arr
        ]

        # ── Build subplots: 3 rows ───────────────────────────────────────────
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=(
                "📈 Equity Curve (Cumulative Net PnL)",
                "📉 Drawdown (%)",
                "🎯 Trade PnL per Trade",
            ),
            row_heights=[0.50, 0.25, 0.25],
            vertical_spacing=0.06,
        )

        # ── Row 1: Equity Curve ──────────────────────────────────────────────
        fig.add_trace(
            go.Scatter(
                x=close_dts,
                y=equity_arr.tolist(),
                mode="lines+markers",
                name="Equity",
                line=dict(color="#2196F3", width=2),
                marker=dict(
                    size=6,
                    color=colors_pnl,
                    symbol=["triangle-up" if d == "BUY" else "triangle-down" for d in directions],
                    line=dict(width=1, color="#ffffff"),
                ),
                hovertemplate=(
                    "<b>Trade #%{customdata[0]}</b><br>"
                    "Close: %{x}<br>"
                    "Equity: %{y:.5f}<br>"
                    "Direction: %{customdata[1]}<br>"
                    "Reason: %{customdata[2]}<extra></extra>"
                ),
                customdata=list(zip(trade_ids, directions, reasons)),
            ),
            row=1, col=1,
        )

        # Zero reference line
        fig.add_hline(y=0, line_dash="dash", line_color="#78909C", opacity=0.5, row=1, col=1)

        # ── Row 2: Drawdown (area filled red) ───────────────────────────────
        fig.add_trace(
            go.Scatter(
                x=close_dts,
                y=(-dd_arr).tolist(),   # Inverted: drawdown plotted downward
                mode="lines",
                name="Drawdown %",
                fill="tozeroy",
                fillcolor="rgba(239,83,80,0.25)",
                line=dict(color="#ef5350", width=1.5),
                hovertemplate="Drawdown: %{customdata:.2f}%<extra></extra>",
                customdata=dd_arr.tolist(),
            ),
            row=2, col=1,
        )

        # Max drawdown marker
        max_dd_idx = int(np.argmax(dd_arr))
        fig.add_annotation(
            x=close_dts[max_dd_idx],
            y=-float(dd_arr[max_dd_idx]),
            text=f"MaxDD {dd_arr[max_dd_idx]:.1f}%",
            showarrow=True, arrowhead=2,
            arrowcolor="#ef5350", font=dict(color="#ef5350", size=11),
            row=2, col=1,
        )

        # ── Row 3: PnL bar chart per trade ───────────────────────────────────
        fig.add_trace(
            go.Bar(
                x=close_dts,
                y=pnl_arr,
                name="PnL/Trade",
                marker_color=colors_pnl,
                opacity=0.85,
                hovertemplate=(
                    "Trade #%{customdata}<br>"
                    "PnL: %{y:.5f}<extra></extra>"
                ),
                customdata=trade_ids,
            ),
            row=3, col=1,
        )

        # ── Layout ───────────────────────────────────────────────────────────
        summary_text = (
            f"<b>Backtest Report</b><br>"
            f"Trades: {report.trade_count} | "
            f"WR: {report.winrate*100:.1f}% | "
            f"PF: {report.profit_factor:.3f} | "
            f"MaxDD: {report.max_drawdown*100:.1f}% | "
            f"Sharpe: {report.sharpe_ratio:.2f} | "
            f"Calmar: {report.calmar_ratio:.2f}<br>"
            f"Gross Profit: {report.gross_profit:.4f} | "
            f"Gross Loss: {report.gross_loss:.4f} | "
            f"Net: {report.gross_profit - report.gross_loss:.4f}"
        )

        fig.update_layout(
            title=dict(
                text=summary_text,
                x=0.01, font=dict(size=14, color="#E0E0E0"),
            ),
            paper_bgcolor="#1A1A2E",
            plot_bgcolor="#16213E",
            font=dict(color="#E0E0E0", family="Inter, monospace"),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                bgcolor="rgba(0,0,0,0.3)",
            ),
            hovermode="x unified",
            height=900,
            margin=dict(l=60, r=40, t=100, b=40),
        )

        fig.update_xaxes(
            gridcolor="#1f2d3d", showgrid=True, zeroline=False,
            rangeslider=dict(visible=False),
        )
        fig.update_yaxes(gridcolor="#1f2d3d", showgrid=True, zeroline=True)

        # ── Write HTML ───────────────────────────────────────────────────────
        fig.write_html(
            str(path),
            full_html=True,
            include_plotlyjs="cdn",      # Dùng CDN — file nhỏ hơn, không cần offline
            config={
                "scrollZoom":     True,
                "displayModeBar": True,
                "modeBarButtonsToAdd": ["drawline", "eraseshape"],
            },
        )

        logger.info(
            f"[BacktestEnv] HTML report generated | "
            f"{report.trade_count} trades | {path}"
        )
        return path


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RabitScal Backtest Env — CSV + Plotly HTML Report Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--params",
        type=str,
        default=str(PROJECT_ROOT / "config" / "current_settings.json"),
        help="Path to params JSON file (current_settings.json or settings_vNNN.json)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_ROOT / "data" / "m5_historical.npy"),
        help="Path to M5 historical numpy cache (.npy)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(REPORTS_DIR),
        help="Output directory for CSV and HTML report",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        default=False,
        help="Skip HTML report generation (CSV only)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.getLogger("BacktestEnv").setLevel(getattr(logging, args.log_level))

    # Load params
    params_path = Path(args.params)
    if not params_path.exists():
        logger.error(f"Params file not found: {params_path}")
        sys.exit(1)

    with open(params_path, "r") as f:
        cfg = json.load(f)
    params = cfg.get("params", cfg)  # Support both raw params and versioned format

    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(
            f"Data cache not found: {data_path}\n"
            f"Run: python ml_model.py --fetch  to download M5 data first"
        )
        sys.exit(1)

    logger.info(f"[BacktestEnv] Loading data: {data_path}")
    data = np.load(data_path)
    logger.info(f"[BacktestEnv] Data loaded | shape={data.shape}")

    # Run backtest
    env    = BacktestEnv()
    report = env.run(data, params)

    out_dir    = Path(args.out)
    ts         = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path   = str(out_dir / f"trade_log_{ts}.csv")
    html_path  = str(out_dir / f"backtest_report_{ts}.html")

    # Export CSV
    env.export_trade_log(report, csv_path)

    # Generate HTML
    if not args.no_html:
        env.generate_html_report(report, html_path)

    # Print summary
    print("\n" + "=" * 60)
    print("✅ BACKTEST COMPLETE")
    print(f"   Trades       : {report.trade_count}")
    print(f"   Winrate      : {report.winrate * 100:.1f}%")
    print(f"   Profit Factor: {report.profit_factor:.3f}")
    print(f"   Max Drawdown : {report.max_drawdown * 100:.1f}%")
    print(f"   Sharpe Ratio : {report.sharpe_ratio:.2f}")
    print(f"   Calmar Ratio : {report.calmar_ratio:.2f}")
    print(f"   Net PnL      : {report.gross_profit - report.gross_loss:.4f}")
    print(f"   Trade Log    : {csv_path}")
    if not args.no_html:
        print(f"   HTML Report  : {html_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
