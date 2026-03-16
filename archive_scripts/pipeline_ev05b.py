#!/usr/bin/env python3
"""
pipeline_ev05b.py — Ev05b: Super Portfolio Engine
===================================================
Kế thừa Ev05 + 3 nâng cấp quan trọng:
  1. Margin-safe lot sizing (calc_lot_size_with_margin)
  2. H1 Trend Filter (block lệnh ngược trend H1 — Option B, N=73 intact)
  3. Global Risk Gate (max 3 lệnh đồng thời / 9% exposure)

Super 6 × 3 TF = 18 studies × 2000 trials = 36,000 kịch bản
"""
from __future__ import annotations

import os, gc, json, logging, time
import numpy as np
import optuna
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Any

# ─── Import internal modules ──────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.symbol_registry import (
    SYMBOL_PROPS, SUPER_6_SYMBOLS, BARS_PER_DAY,
    calc_lot_size_with_margin, get_swap_per_bar,
)
from core.feature_engine import build_feature_matrix, load_mtf_data

logger = logging.getLogger("EV05B")

# ─── Constants ────────────────────────────────────────────────────────────────
SUPER_6      = SUPER_6_SYMBOLS       # 6 symbols
TF_LIST      = ["M1", "M5", "M15"]  # bỏ H1 để né swap
N_TRIALS     = 2000
EQUITY       = 20_000.0
RISK_PCT     = 0.03         # 3% / lệnh = $600
MAX_CONCURRENT = 3          # Global Risk Gate: tối đa 3 lệnh/time
MAX_MARGIN_PCT  = 0.30      # tối đa 30% equity margin / lệnh

# ─── Dynamic Search Space per TF ─────────────────────────────────────────────
# Giống Ev05 nhưng tập trung M1/M5/M15 (không có H1 Swing rule)
SEARCH_SPACE = {
    "M1": {
        "threshold_range":  (0.65, 0.95),
        "sl_mult_range":    (1.5, 4.0),
        "rr_range":         (1.0, 2.5),
        "cooldown_range":   (1, 10),
    },
    "M5": {
        "threshold_range":  (0.55, 0.90),
        "sl_mult_range":    (1.0, 3.0),
        "rr_range":         (1.0, 2.5),
        "cooldown_range":   (1, 5),
    },
    "M15": {
        "threshold_range":  (0.45, 0.85),
        "sl_mult_range":    (1.5, 4.0),
        "rr_range":         (1.5, 5.0),
        "cooldown_range":   (1, 5),
    },
}

# ─── Emas for H1 trend extraction from existing feature matrix ────────────────
# Feature #23 (index 22) = trend_ema_h1 (-1/0/+1) — đã có sẵn trong N=73!
FEAT_IDX_H1_TREND = 23  # trend_ema_h1 (-1/0/+1): feature doc #23 (1-indexed) = col 23 (0-indexed)
# Col 22 = atr_h1_norm (float), Col 23 = trend_ema_h1 (-1.0/0.0/+1.0)


def _extract_h1_trend(feats: np.ndarray) -> np.ndarray:
    """
    Trích xuất H1 trend direction từ feature matrix đã tính.
    Feature #23 (idx=22) = trend_ema_h1: -1 (bear) / 0 (sideways) / +1 (bull)
    Đây là H1 signal đã được ffill() và shift đúng trong feature_engine.
    KHÔNG look-ahead vì feature_engine đã xử lý.
    """
    return feats[:, FEAT_IDX_H1_TREND].copy()


# ─── Backtester EV05B ─────────────────────────────────────────────────────────

def _backtest_ev05b(
    feats:   np.ndarray,       # (N, 73) feature matrix
    raw_m5:  np.ndarray,       # (N, 6) OHLCV+time M5 bars
    params:  dict,
    symbol:  str,
    tf:      str,
    spread:  float,
    is_mask: np.ndarray,       # boolean mask for IS/OOS split
    h1_trend: np.ndarray,      # (N,) H1 trend: -1/0/+1
) -> dict:
    """
    Vectorized backtest với:
    - H1 Trend Filter: chỉ Long khi h1_trend >= 0, chỉ Short khi h1_trend <= 0
    - Margin-safe lot sizing
    - Global Risk Gate simulation: tối đa MAX_CONCURRENT lệnh
    """
    threshold  = params["threshold"]
    sl_mult    = params["sl_mult"]
    rr_fb      = params["rr_fallback"]
    cooldown   = params["cooldown"]
    split_ratio= params.get("split_ratio", 0.5)

    # Raw columns
    o5 = raw_m5[:, 1]; h5 = raw_m5[:, 2]
    l5 = raw_m5[:, 3]; c5 = raw_m5[:, 4]
    pc5 = np.roll(c5, 1); pc5[0] = c5[0]

    # ATR (simple range proxy)
    atr5 = feats[:, 2].astype(np.float64)  # feat #3 = atr_norm → unnormalize
    atr_abs = np.abs(c5 - pc5)             # fallback price-based ATR
    # Dùng atr từ feature nếu có, else price-range
    high_arr = h5; low_arr = l5

    # ML score: dùng feature tổng hợp làm proxy score
    # (trong thực tế sẽ có model predict, đây dùng feat composite)
    # Score = ema_trend_score approximate từ feat #3,4,5
    bull_score = feats[:, 3].astype(np.float64)  # price vs ema21
    bear_score = feats[:, 4].astype(np.float64)  # price vs ema50

    N = len(c5)
    prices  = c5
    equity  = EQUITY
    results: list[dict] = []
    concurrent_open = 0
    cooldown_cnt    = 0
    last_exit_bar   = -999

    for i in range(50, N):
        if not is_mask[i]:
            continue
        if cooldown_cnt > 0:
            cooldown_cnt -= 1
            continue
        if i - last_exit_bar < cooldown:
            continue
        if concurrent_open >= MAX_CONCURRENT:
            continue  # Global Risk Gate

        # ── H1 Trend Filter (Option B) ──────────────────────────────
        h1_t = h1_trend[i]  # -1 / 0 / +1

        # Signal từ feature ema context
        bull_sig = bull_score[i] > threshold
        bear_sig = bear_score[i] < -threshold

        # Apply H1 filter: block nghịch chiều
        if bull_sig and h1_t < 0:   # muốn Long nhưng H1 Bear → block
            continue
        if bear_sig and h1_t > 0:   # muốn Short nhưng H1 Bull → block
            continue

        direction = 0
        if bull_sig:  direction = +1
        elif bear_sig: direction = -1
        else: continue

        entry = prices[i]
        atr_i = max(atr_abs[i], entry * 0.001)

        # ── FVG zones cho dynamic target ──────────────────────────────
        # Tận dụng feat index gần nhất FVG context (feat #5-12 là FVG features)
        fvg_dist_norm = feats[i, 9]  # nearest_fvg_bull_dist_norm
        target_dist   = max(atr_i * rr_fb, atr_i * 1.2)

        sl_dist  = atr_i * sl_mult
        tp_dist  = target_dist

        # Lot sizing với margin check
        lot = calc_lot_size_with_margin(
            equity, RISK_PCT, sl_dist, symbol, entry,
            max_margin_pct=MAX_MARGIN_PCT,
        )
        if lot <= 0:
            continue

        # ── Simulate trade ────────────────────────────────────────────
        sl_price = (entry - sl_dist) if direction == +1 else (entry + sl_dist)
        tp_price = (entry + tp_dist) if direction == +1 else (entry - tp_dist)

        hit_sl = hit_tp = False
        bars_in_trade = 0
        concurrent_open += 1

        for j in range(i+1, min(i + 500, N)):
            h_j = high_arr[j]; l_j = low_arr[j]
            bars_in_trade += 1

            if direction == +1:
                if l_j <= sl_price: hit_sl = True; break
                if h_j >= tp_price: hit_tp = True; break
            else:
                if h_j >= sl_price: hit_sl = True; break
                if l_j <= tp_price: hit_tp = True; break

        concurrent_open -= 1
        last_exit_bar = j if (hit_sl or hit_tp) else i + bars_in_trade

        # PnL
        pv    = SYMBOL_PROPS.get(symbol, {}).get("pip_size", 0.00001)
        pv_l  = pv * SYMBOL_PROPS.get(symbol, {}).get("contract_size", 100_000)
        sp_c  = spread / pv * pv_l * lot if pv > 0 else 0

        if hit_tp:
            pnl = tp_dist / pv * pv_l * lot - sp_c
        elif hit_sl:
            pnl = -sl_dist / pv * pv_l * lot - sp_c
        else:
            pnl = -sp_c * 0.3  # exit at bar limit, small cost

        equity  += pnl
        cooldown_cnt = cooldown
        results.append({
            "bar": i, "dir": direction, "pnl": pnl,
            "equity": equity, "won": hit_tp,
        })

    # ── Stats ─────────────────────────────────────────────────────────
    if len(results) < 10:
        return {"fitness": -999.0, "n_trades": len(results)}

    pnls       = np.array([r["pnl"] for r in results])
    equities   = np.array([r["equity"] for r in results])
    wins       = sum(1 for r in results if r["won"])
    n          = len(results)
    net_profit = float(equities[-1] - EQUITY)
    winrate    = wins / n
    pf         = float(pnls[pnls > 0].sum() / max(-pnls[pnls < 0].sum(), 1e-6))

    # Max drawdown
    peak = EQUITY
    max_dd = 0.0
    for eq in equities:
        if eq > peak: peak = eq
        dd = (peak - eq) / max(peak, 1)
        max_dd = max(max_dd, dd)

    # Fitness: giống Ev04 — penalize DD, reward profit/trade
    if max_dd > 0.35:
        return {"fitness": -999.0, "n_trades": n}

    fitness = (net_profit / EQUITY) * (1 - max_dd * 2) * (pf ** 0.5)

    return {
        "fitness": fitness,
        "net_profit": net_profit,
        "n_trades": n,
        "winrate": winrate,
        "pf": pf,
        "max_dd": max_dd,
        "oos_ok": 1,
    }


# ─── Optuna Objective ─────────────────────────────────────────────────────────

def _objective_ev05b(
    trial, feats, raw_m5, symbol, tf, spread, is_split, h1_trend
):
    ss = SEARCH_SPACE[tf]
    params = {
        "threshold":   trial.suggest_float("threshold", *ss["threshold_range"]),
        "sl_mult":     trial.suggest_float("sl_mult",   *ss["sl_mult_range"]),
        "rr_fallback": trial.suggest_float("rr_fallback", *ss["rr_range"]),
        "cooldown":    trial.suggest_int("cooldown",    *ss["cooldown_range"]),
    }

    N = len(feats)
    is_cut = int(N * (1 - is_split))
    is_mask = np.zeros(N, dtype=bool)
    is_mask[:is_cut] = True   # IS = first 70%

    res = _backtest_ev05b(feats, raw_m5, params, symbol, tf,
                          spread, is_mask, h1_trend)

    if res["fitness"] <= -999 or res.get("n_trades", 0) < 30:
        raise optuna.TrialPruned()

    # OOS validation
    oos_mask = ~is_mask
    oos_res  = _backtest_ev05b(feats, raw_m5, params, symbol, tf,
                               spread, oos_mask, h1_trend)

    trial.set_user_attr("net_profit",  res.get("net_profit", 0))
    trial.set_user_attr("winrate",     res.get("winrate", 0))
    trial.set_user_attr("pf",          res.get("pf", 0))
    trial.set_user_attr("max_dd",      res.get("max_dd", 0))
    trial.set_user_attr("n_trades",    res.get("n_trades", 0))
    trial.set_user_attr("oos_profit",  oos_res.get("net_profit", 0))
    trial.set_user_attr("oos_dd",      oos_res.get("max_dd", -1))
    trial.set_user_attr("oos_ok",      int(oos_res.get("max_dd", 1) < 0.25))

    return res["fitness"]


# ─── Run single study ─────────────────────────────────────────────────────────

def run_study_ev05b(
    symbol:   str,
    tf:       str,
    data_dir: str    = "data",
    n_trials: int    = N_TRIALS,
    n_workers: int   = 40,
    spread:   float  = 0.00015,
    is_split: float  = 0.30,
) -> dict:
    """Chạy 1 Optuna study cho 1 symbol × 1 TF."""
    study_name = f"ev05b_{symbol.lower().replace('m','')}_{tf.lower()}"
    db_path    = f"data/optuna_{study_name}.db"

    logger.info(f"[{symbol}|{tf}] Loading features...")
    try:
        feats, raw = build_feature_matrix(symbol, data_dir)
    except Exception as e:
        logger.error(f"[{symbol}|{tf}] FAILED: {e}")
        return {"symbol": symbol, "tf": tf, "error": str(e)}

    # Extract H1 trend context (Option B — separate, N=73 intact)
    h1_trend = _extract_h1_trend(feats)

    logger.info(f"[{symbol}|{tf}] Matrix: {feats.shape} | H1 trend unique: {np.unique(h1_trend)}")

    tf_key = tf if tf in SEARCH_SPACE else "M15"

    study = optuna.create_study(
        study_name  = study_name,
        storage     = f"sqlite:///{db_path}",
        direction   = "maximize",
        load_if_exists = True,
        sampler     = optuna.samplers.TPESampler(
            n_startup_trials=50, multivariate=True,
            group=True, constant_liar=True, seed=42,
        ),
    )

    # Progress bar
    try:
        from tqdm import tqdm
        pbar = tqdm(total=n_trials, desc=f"  [{tf}] {symbol}", ncols=80,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, best={postfix[0]:.4f}, cpl={postfix[1]}]",
                    postfix=[0.0, 0])
    except Exception:
        pbar = None

    done  = len(study.trials)
    remaining = n_trials - done
    if remaining <= 0:
        logger.info(f"[{symbol}|{tf}] Already done ({done} trials)")
        return {"symbol": symbol, "tf": tf, "done": done}

    def _cb(study, trial):
        if pbar is not None:
            comp = [t for t in study.trials if t.state.name == "COMPLETE"]
            try:
                bv = study.best_value
            except Exception:
                bv = 0.0
            pbar.set_postfix([bv, len(comp)], refresh=False)
            pbar.update(1)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for _ in range(remaining):
            f = executor.submit(
                _worker_trial,
                study_name, db_path, symbol, tf, feats, raw,
                spread, is_split, h1_trend
            )
            futures.append(f)
        for f in futures:
            try:
                f.result(timeout=120)
                if pbar: pbar.update(1)
            except Exception:
                pass

    if pbar: pbar.close()

    try:
        bt = study.best_trial
        result = {
            "symbol": symbol, "tf": tf, "done": len(study.trials),
            "best_fitness": bt.value,
            "best_params": bt.params,
            **bt.user_attrs,
        }
        logger.info(
            f"[{symbol}|{tf}] ✓ best={bt.value:.3f} "
            f"IS${bt.user_attrs.get('net_profit',0):,.0f} "
            f"OOS${bt.user_attrs.get('oos_profit',0):,.0f}"
        )
        return result
    except Exception:
        return {"symbol": symbol, "tf": tf, "error": "No valid trial"}


def _worker_trial(study_name, db_path, symbol, tf, feats, raw,
                  spread, is_split, h1_trend):
    """Worker chạy trong subprocess."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(
        study_name=study_name, storage=f"sqlite:///{db_path}"
    )
    trial = study.ask()
    try:
        value = _objective_ev05b(trial, feats, raw, symbol, tf,
                                  spread, is_split, h1_trend)
        study.tell(trial, value)
    except optuna.TrialPruned:
        study.tell(trial, state=optuna.trial.TrialState.PRUNED)
    except Exception as e:
        study.tell(trial, state=optuna.trial.TrialState.FAIL)


# ─── Queue Runner ─────────────────────────────────────────────────────────────

def run_ev05b_queue(
    symbols:     list[str] = SUPER_6_SYMBOLS,
    tfs:         list[str] = TF_LIST,
    data_dir:    str       = "data",
    n_trials:    int       = N_TRIALS,
    n_workers:   int       = 40,
    spread:      float     = 0.00015,
    results_dir: str       = "data/ev05b_results",
    report_out:  str       = "logs/An_Latest_Report.md",
) -> list[dict]:
    """
    Queue tuần tự: symbol by symbol, TF by TF.
    Tổng: len(symbols) × len(tfs) studies.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s UTC] - [%(levelname)-7s] - [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    total   = len(symbols) * len(tfs)
    done_n  = 0
    all_res: list[dict] = []

    for i, sym in enumerate(symbols):
        print(f"\n{'─'*55}")
        print(f"  [{i+1}/{len(symbols)}] SYMBOL: {sym}")
        print(f"{'─'*55}")

        for j, tf in enumerate(tfs):
            done_n += 1
            pct  = done_n / total * 100
            bar  = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  [{i+1}/{len(symbols)}] {sym} | [{j+1}/{len(tfs)}] {tf} "
                  f"| Overall [{bar}] {pct:.0f}% ({done_n}/{total})")

            res = run_study_ev05b(sym, tf, data_dir, n_trials, n_workers, spread)
            all_res.append(res)

            # Save intermediate
            out_file = Path(results_dir) / f"{sym}_{tf}.json"
            with open(out_file, "w") as f:
                json.dump(res, f, indent=2, default=str)

            gc.collect()

    print(f"\n✅ Ev05b hoàn tất toàn mặt trận Super 6!")
    print(f"   Leaderboard → {report_out}")

    # Generate report
    _write_ev05b_report(all_res, report_out)
    return all_res


def _write_ev05b_report(results: list[dict], out_path: str):
    """Ghi leaderboard Ev05b (sẽ được portfolio_report.py mở rộng thêm equity curve)."""
    from datetime import datetime
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"# 🏆 EV05b — SUPER PORTFOLIO INTERIM REPORT",
        f"**{now}** | 6 Symbols × 3 TF × 2000 trials = 36,000 kịch bản",
        f"",
        f"| Symbol | TF | Best Fit | IS$ | OOS$ | OOS DD | WR | PF | H1 Filter | Status |",
        f"|--------|-----|---------|-----|------|--------|----|----|-----------|--------|",
    ]

    for r in sorted(results, key=lambda x: x.get("best_fitness", -999), reverse=True):
        sym   = r.get("symbol", "?")
        tf    = r.get("tf", "?")
        bf    = r.get("best_fitness", -999)
        ip    = r.get("net_profit", 0)
        op    = r.get("oos_profit", 0)
        od    = r.get("oos_dd", -1)
        wr    = r.get("winrate", 0)
        pf_   = r.get("pf", 0)
        ok    = "✅" if r.get("oos_ok", 0) else "🔴"
        err   = r.get("error", None)
        if err:
            lines.append(f"| {sym} | {tf} | — | — | — | — | — | — | — | 🔴 ERR |")
        else:
            lines.append(
                f"| {sym} | {tf} | {bf:.3f} | ${ip:,.0f} | ${op:,.0f} | "
                f"{od:.1%} | {wr:.1%} | {pf_:.2f} | ✅ Active | {ok} |"
            )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
