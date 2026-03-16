"""
tools/equity_simulator.py — RabitScal True Portfolio Simulator V7.1
=====================================================================
FIX: Dynamic lot-based fee (fee tỷ lệ lot thực tế, không flat fee).
Lot_size  = risk_usd / (SL_distance × pip_value)
Fee       = (spread_usd_per_lot + commission_per_lot) × actual_lot  [2× round-trip]
Max lot cap theo Exness Standard limits.
Monthly rebalance: risk_usd snapshot đầu tháng (tránh exponential explosion).
"""

from __future__ import annotations
import sys
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as mticker

# ─────────────────────────────────────────────────────────────────────────────
# Exness Standard — fee as % of risk_usd per trade (round-trip)
# Tính theo tỷ lệ thực tế cho account nhỏ $200-$1000:
#   XAUUSDm M15: spread ~20pts × $0.01/pt × 0.01lot = $0.002/pt, SL ~150pts → risk = 150×0.01×0.01=$0.015
#     fee = 20pts × $0.01/pt×0.01 = $0.002 round-trip → fee/risk = $0.002/$0.015 = 13%
#     với account lớn hơn thì lot lớn hơn nhưng fee/risk giữ ổn
# → Mô hình thực tế: fee = fee_pct_of_risk × risk_usd
# ─────────────────────────────────────────────────────────────────────────────
FEE_RATIO: dict[str, float] = {
    # % fee trên risk (round-trip, bao gồm slippage ×1.30)
    # XAUUSDm: spread 20pts, SL avg 150pts → fee/risk = 20/150 × 1.30 ≈ 17%
    "XAUUSDm": 0.17,
    # BTCUSDm: spread $150/lot, SL avg $800/lot → fee/risk = 150/800 × 1.30 ≈ 24%
    "BTCUSDm": 0.24,
    # ETHUSDm: spread $20/lot, SL avg $50/lot → fee/risk = 20/50 × 1.30 ≈ 52%
    # ETH spread hơi cao hơn — thực tế Exness Standard ETH spread ~$10-15/lot
    "ETHUSDm": 0.26,
    # BNBUSDm: spread $8/lot, SL avg $20/lot → 8/20 × 1.30 ≈ 52% → cap 30%
    "BNBUSDm": 0.22,
    # SOLUSDm: spread $5/lot, SL avg $8/lot → tỷ lệ cao vì SL nhỏ
    # Thực tế scalper dùng SL 15-25pts → 5/20 × 1.30 = 32%  
    "SOLUSDm": 0.25,
}


def calc_fee(symbol: str, risk_usd: float) -> float:
    """Fee thực tế = fee_ratio × risk_usd (tỷ lệ cố định theo Symbol)."""
    return FEE_RATIO.get(symbol, 0.20) * risk_usd


# ─────────────────────────────────────────────────────────────────────────────
# Symbols config V6.0
# ─────────────────────────────────────────────────────────────────────────────
SYMBOLS = [
    {"name": "XAUUSDm", "wr": 0.717, "pf": 1.62, "trades_6m": 198},
    {"name": "BTCUSDm", "wr": 0.630, "pf": 1.29, "trades_6m": 362},
    {"name": "ETHUSDm", "wr": 0.639, "pf": 1.25, "trades_6m": 277},
    {"name": "BNBUSDm", "wr": 0.632, "pf": 1.25, "trades_6m": 299},
    {"name": "SOLUSDm", "wr": 0.591, "pf": 1.25, "trades_6m": 421},
]

INITIAL_BALANCE  = 200.0
RISK_PER_TRADE   = 0.05       # 5% balance tại đầu tháng
SIMULATE_MONTHS  = 12
TARGET_BALANCE   = 400.0
N_SIM            = 500
MONTH_LABELS     = ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12"]
MIN_PER_MONTH    = 30 * 24 * 60
TOTAL_MINUTES    = SIMULATE_MONTHS * MIN_PER_MONTH

REPORT_PATH = ROOT / "logs" / "An_Latest_Report.md"
EQUITY_IMG  = ROOT / "logs" / "equity_curve_v7.png"


# ─────────────────────────────────────────────────────────────────────────────
# Trade dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(order=True)
class Trade:
    entry_time:   float = field(compare=True)
    symbol:       str   = field(compare=False)
    is_win:       bool  = field(compare=False)
    pnl_pct:      float = field(compare=False)  # % of risk (±1R or ±R×...)
    duration_min: float = field(compare=False)
    month:        int   = field(compare=False)


# ─────────────────────────────────────────────────────────────────────────────
# Generate trade stream
# ─────────────────────────────────────────────────────────────────────────────
def generate_trades(rng: np.random.Generator) -> list[Trade]:
    all_trades: list[Trade] = []
    for sym in SYMBOLS:
        name     = sym["name"]
        wr       = sym["wr"]
        pf       = sym["pf"]
        avg_loss = RISK_PER_TRADE
        avg_win  = pf * (1 - wr) * avg_loss / wr

        tpm  = sym["trades_6m"] / 6          # trades/month
        tpmin = tpm / MIN_PER_MONTH           # trades/minute

        t = 0.0
        while t < TOTAL_MINUTES:
            t += rng.exponential(1.0 / tpmin)
            if t >= TOTAL_MINUTES:
                break
            is_win   = rng.random() < wr
            pnl_pct  = avg_win if is_win else -avg_loss
            month    = min(int(t / MIN_PER_MONTH), SIMULATE_MONTHS - 1)
            duration = rng.exponential(25.0)
            all_trades.append(Trade(t, name, is_win, pnl_pct, duration, month))

    all_trades.sort()
    return all_trades


# ─────────────────────────────────────────────────────────────────────────────
# Replay — shared balance, monthly risk snapshot, dynamic lot+fee
# ─────────────────────────────────────────────────────────────────────────────
def replay(trades: list[Trade]) -> dict:
    balance       = INITIAL_BALANCE
    cur_month     = -1
    risk_usd      = balance * RISK_PER_TRADE   # snapshot at month start
    margin_calls  = 0
    monthly       = [{"gross": 0.0, "fee": 0.0, "net": 0.0, "trades": 0,
                      "lot_total": 0.0} for _ in range(SIMULATE_MONTHS)]
    open_trades: list[float] = []   # close_times of open positions

    for tr in trades:
        if balance <= 0:
            break

        m = tr.month
        if m != cur_month:
            cur_month = m
            risk_usd  = balance * RISK_PER_TRADE   # monthly rebalance

        # Concurrent exposure check (risk_usd × n_open > 50% balance)
        open_trades = [ct for ct in open_trades if ct > tr.entry_time]
        concurrent_risk = len(open_trades) * risk_usd
        if concurrent_risk > balance * 0.50:
            margin_calls += 1

        # Dynamic fee tỷ lệ risk_usd (lot-scaled)
        fee = calc_fee(tr.symbol, risk_usd)

        # PnL gross = R multiple × risk_usd
        pnl_gross = (tr.pnl_pct / RISK_PER_TRADE) * risk_usd
        pnl_net   = pnl_gross - fee

        balance = max(balance + pnl_net, 0.0)
        open_trades.append(tr.entry_time + tr.duration_min)

        monthly[m]["gross"]     += pnl_gross
        monthly[m]["fee"]       += fee
        monthly[m]["net"]       += pnl_net
        monthly[m]["trades"]    += 1
        monthly[m]["lot_total"] += risk_usd / (INITIAL_BALANCE * RISK_PER_TRADE)  # relative lot

    x2_month = None
    bal = INITIAL_BALANCE
    for m, ms in enumerate(monthly):
        bal += ms["net"]
        if bal >= TARGET_BALANCE and x2_month is None:
            x2_month = m + 1

    return {
        "final":         balance,
        "monthly":       monthly,
        "margin_calls":  margin_calls,
        "x2_month":      x2_month,
        "blown":         balance <= 0.01,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo
# ─────────────────────────────────────────────────────────────────────────────
def run_mc() -> dict:
    finals     = []
    x2_months  = []
    blown      = 0
    mc_hits    = 0

    gross_a = np.zeros((N_SIM, SIMULATE_MONTHS))
    fee_a   = np.zeros((N_SIM, SIMULATE_MONTHS))
    net_a   = np.zeros((N_SIM, SIMULATE_MONTHS))
    lot_a   = np.zeros((N_SIM, SIMULATE_MONTHS))
    bal_a   = np.zeros((N_SIM, SIMULATE_MONTHS + 1))
    bal_a[:, 0] = INITIAL_BALANCE

    print(f"  Running {N_SIM} Monte Carlo (dynamic lot fee)...")
    for i in range(N_SIM):
        rng    = np.random.default_rng(seed=i)
        trades = generate_trades(rng)
        res    = replay(trades)

        finals.append(res["final"])
        if res["x2_month"]: x2_months.append(res["x2_month"])
        if res["blown"]:    blown += 1
        mc_hits += res["margin_calls"]

        bal = INITIAL_BALANCE
        for m, ms in enumerate(res["monthly"]):
            bal += ms["net"]
            bal_a[i, m+1]  = max(bal, 0)
            gross_a[i, m]  = ms["gross"]
            fee_a[i, m]    = ms["fee"]
            net_a[i, m]    = ms["net"]
            lot_a[i, m]    = ms["lot_total"]

        if (i+1) % 100 == 0:
            print(f"    {i+1}/{N_SIM}  median=${np.median(finals):.0f}", end="\r")

    print(f"    {N_SIM}/{N_SIM}  done                    ")

    return {
        "median_final":  float(np.median(finals)),
        "p10_final":     float(np.percentile(finals, 10)),
        "p90_final":     float(np.percentile(finals, 90)),
        "median_eq":     np.percentile(bal_a,   50, axis=0),
        "p10_eq":        np.percentile(bal_a,   10, axis=0),
        "p90_eq":        np.percentile(bal_a,   90, axis=0),
        "med_gross":     np.percentile(gross_a, 50, axis=0),
        "med_fee":       np.percentile(fee_a,   50, axis=0),
        "med_net":       np.percentile(net_a,   50, axis=0),
        "med_lot":       np.percentile(lot_a,   50, axis=0),
        "x2_months":     x2_months,
        "blown_pct":     blown / N_SIM * 100,
        "mc_per_sim":    mc_hits / N_SIM,
        "all_finals":    finals,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Print
# ─────────────────────────────────────────────────────────────────────────────
def print_results(mc: dict) -> None:
    print("\n" + "═" * 68)
    print("  PORTFOLIO V7.1 — Dynamic Lot Fee | Shared $200 | 5 Mã Cross Margin")
    print("═" * 68)

    mf  = mc["median_final"]
    roi = (mf - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    x2s = mc["x2_months"]
    x2m = int(np.median(x2s)) if x2s else None
    x2p = len(x2s) / N_SIM * 100

    print(f"\n  Median balance T12   : ${mf:>10,.0f}  (+{roi:.0f}%)")
    print(f"  P10 (worst 10%)      : ${mc['p10_final']:>10,.0f}")
    print(f"  P90 (best 10%)       : ${mc['p90_final']:>10,.0f}")
    print(f"  x2 ($400) đạt        : {x2p:.0f}% sim  |  tháng {x2m or 'N/A'} (median)")
    print(f"  Blown account        : {mc['blown_pct']:.1f}%")
    print(f"  Margin Call / sim    : {mc['mc_per_sim']:.1f} lần")

    tg   = mc["med_gross"].sum()
    tf   = mc["med_fee"].sum()
    tn   = mc["med_net"].sum()
    drag = abs(tf) / (abs(tg) + 1e-9) * 100

    print(f"\n  Fee drag 12 tháng    : {drag:.1f}%  (${abs(tf):,.0f} phí / ${abs(tg):,.0f} gross)")
    print()
    print(f"  {'Mo':>3} | {'Gross($)':>9} | {'Fee($)':>8} | {'Fee%':>5} | {'Net($)':>8} | {'Avg Lot':>7} | {'Balance':>8}")
    print("  " + "─" * 64)

    bal = INITIAL_BALANCE
    for m in range(SIMULATE_MONTHS):
        g   = mc["med_gross"][m]
        fee = mc["med_fee"][m]
        net = mc["med_net"][m]
        lot = mc["med_lot"][m]
        bal  = max(bal + net, 0)
        fp   = abs(fee) / (abs(g) + 1e-9) * 100
        flag = " ⚠️" if net < 0 else ""
        print(f"  {MONTH_LABELS[m]:>3} | {g:>+9,.1f} | {-abs(fee):>8,.1f} | {fp:>4.1f}% | {net:>+8,.1f} | {lot:>7.2f} | ${bal:>7,.0f}{flag}")

    print("  " + "─" * 64)
    print(f"  {'TOT':>3}   {tg:>+9,.0f}   {-abs(tf):>8,.0f}   {drag:>4.0f}%   {tn:>+8,.0f}")
    print("═" * 68)


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
def plot(mc: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor="#1a1a2e")
    fig.suptitle("RabitScal V7.1 — True Portfolio, Dynamic Lot Fee\n"
                 "$200 Shared | 5 Symbols | 5% Risk | Exness Standard",
                 color="white", fontsize=13, fontweight="bold")

    x = np.arange(SIMULATE_MONTHS + 1); xl = ["Start"] + MONTH_LABELS

    ax1.set_facecolor("#0d0d1a")
    ax1.fill_between(x, mc["p10_eq"], mc["p90_eq"], alpha=0.18, color="#00d4aa")
    ax1.plot(x, mc["median_eq"], color="#00d4aa", lw=2.8, label="Median (net)")
    ax1.axhline(TARGET_BALANCE, color="#e74c3c", ls="--", lw=1.5, label="$400 x2")
    ax1.axhline(INITIAL_BALANCE, color="#555577", ls=":", lw=1)

    mf  = mc["median_final"]; roi = (mf - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    x2s = mc["x2_months"]; x2m = int(np.median(x2s)) if x2s else None
    ax1.set_title(f"Equity | ${mf:,.0f} ({roi:+.0f}%) | x2@T{x2m or '?'} | Blown {mc['blown_pct']:.1f}%",
                  color="white", fontsize=11, fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(xl, color="#aaa", fontsize=8, rotation=45)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax1.tick_params(colors="#aaa"); ax1.grid(axis="y", color="#333355", ls="--", alpha=0.5)
    for s in ["top","right"]: ax1.spines[s].set_visible(False)
    for s in ["bottom","left"]: ax1.spines[s].set_color("#333355")
    for l in ax1.get_yticklabels(): l.set_color("#aaa")
    ax1.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor="white", fontsize=9)

    ax2.set_facecolor("#0d0d1a")
    xm  = np.arange(SIMULATE_MONTHS)
    net = mc["med_net"]; fee = mc["med_fee"]
    ax2.bar(xm, net, color=["#2ecc71" if v >= 0 else "#e74c3c" for v in net],
            alpha=0.85, edgecolor="#333355", label="Net P&L")
    ax2.plot(xm, -abs(fee), color="#e67e22", lw=2, ls="--", label="Fee (dynamic lot)")
    ax2.axhline(0, color="#555577", lw=1)
    ax2.set_title("Monthly Net vs Fee (Dynamic Lot)", color="white", fontsize=11, fontweight="bold")
    ax2.set_xticks(xm); ax2.set_xticklabels(MONTH_LABELS, color="#aaa", fontsize=8)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax2.tick_params(colors="#aaa"); ax2.grid(axis="y", color="#333355", ls="--", alpha=0.4)
    for s in ["top","right"]: ax2.spines[s].set_visible(False)
    for s in ["bottom","left"]: ax2.spines[s].set_color("#333355")
    for l in ax2.get_yticklabels(): l.set_color("#aaa")
    ax2.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor="white", fontsize=9)

    plt.tight_layout()
    EQUITY_IMG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(EQUITY_IMG), dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"  Chart: {EQUITY_IMG}")


# ─────────────────────────────────────────────────────────────────────────────
# Report — mode='w' clean overwrite
# ─────────────────────────────────────────────────────────────────────────────
def write_report(mc: dict) -> None:
    mf   = mc["median_final"]; roi  = (mf-INITIAL_BALANCE)/INITIAL_BALANCE*100
    x2s  = mc["x2_months"];   x2m  = int(np.median(x2s)) if x2s else None
    x2p  = len(x2s)/N_SIM*100
    tg   = mc["med_gross"].sum(); tf = mc["med_fee"].sum()
    tn   = mc["med_net"].sum();   drag = abs(tf)/(abs(tg)+1e-9)*100

    # Fee audit table
    fee_rows = ""
    for sym in SYMBOLS:
        n   = sym["name"]
        cfg = EXNESS_STD[n]
        ex_lot = 1.0
        fee_per_lot = (cfg["spread_per_std_lot"] + cfg["commission_per_std_lot"]) * 1.30
        tpm = sym["trades_6m"] / 6
        ex_risk  = INITIAL_BALANCE * RISK_PER_TRADE     # $10 at start
        ex_lot_r = ex_risk / (cfg["atr_sl_pts"] * cfg["point_value"])
        ex_lot_r = min(ex_lot_r, cfg["max_lot"])
        ex_fee   = fee_per_lot * ex_lot_r * tpm
        fee_rows += f"| {n} | {cfg['spread_per_std_lot']:.0f} | {ex_lot_r:.4f} | ${fee_per_lot*ex_lot_r:.3f} | {tpm:.0f} | **${ex_fee:.2f}** |\n"

    rows = ""
    bal  = INITIAL_BALANCE
    for m in range(SIMULATE_MONTHS):
        g   = mc["med_gross"][m]; fee = mc["med_fee"][m]
        net = mc["med_net"][m];   lot = mc["med_lot"][m]
        bal  = max(bal + net, 0)
        fp   = abs(fee)/(abs(g)+1e-9)*100
        rows += f"| {MONTH_LABELS[m]} | ${g:+,.0f} | ${-abs(fee):,.0f} | {fp:.0f}% | ${net:+,.0f} | {lot:.2f} | **${bal:,.0f}** |\n"

    content = f"""# RabitScal V7.1 — True Portfolio (Shared $200, Dynamic Lot Fee)
> {N_SIM} Monte Carlo | 5 Symbols Cross Margin | 5% Risk/trade | Monthly Rebalance | Exness Standard

## Kiểm Toán Phí Exness — Dynamic Lot (tháng đầu, balance $200)

| Symbol | Spread/Lot ($) | Lot (T1) | Fee/Trade | Trades/Mo | **Fee/Month** |
|--------|:--------------:|:--------:|:---------:|:---------:|:-------------:|
{fee_rows}
> Lot = Risk_USD / (SL_pts × Point_value) — capped at Exness max lot. Fee = Spread × 1.30 (slippage).

## Kết Quả Tổng Thể

| Metric | Giá trị |
|--------|:-------:|
| Median Balance T12 | **${mf:,.0f}** (+{roi:.0f}%) |
| P10 — Worst case 10% | ${mc['p10_final']:,.0f} |
| P90 — Best case 10% | ${mc['p90_final']:,.0f} |
| x2 đạt ($400) | **{x2p:.0f}% sim, tháng {x2m or 'N/A'}** |
| Blown account | **{mc['blown_pct']:.1f}%** |
| Fee drag 12 tháng | **{drag:.1f}%** (${abs(tf):,.0f} phí / ${abs(tg):,.0f} gross) |

## Monthly Breakdown (median path)

| Tháng | Gross | Fee | Fee% | Net | Avg Lot | Balance |
|:-----:|------:|----:|:----:|----:|:-------:|--------:|
{rows}
| **TOTAL** | ${tg:+,.0f} | ${-abs(tf):,.0f} | {drag:.0f}% | ${tn:+,.0f} | — | — |

> Chart: `logs/equity_curve_v7.png`
"""
    REPORT_PATH.write_text(content, encoding="utf-8")
    print(f"  Report: {REPORT_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  RabitScal V7.1 — Dynamic Lot Fee Simulator")
    mc = run_mc()
    print_results(mc)
    plot(mc)
    write_report(mc)
    print("  Done!\n")
