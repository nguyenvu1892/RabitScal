#!/usr/bin/env python3
"""
pipeline_ev05d.py — Ev05d: $200 Live Rehearsal (Hybrid Risk Engine)
====================================================================
Kế thừa CHÍNH XÁC Ev05c engine:
  signal = feats @ weights  (dot-product linear classifier, weights ∈ optuna)
  TP     = Opposing FVG mid (Split Ticket 2-leg)
  SL     = ± sl_mult × ATR

Thay đổi so với Ev05c (Lệnh Sếp Vũ):
  1. Initial Capital: $200 (thay vì $20,000)
  2. Hybrid Lot Sizing:
     - XAUUSDm: FIX CỨNG lot = 0.01 cho mọi lệnh
     - Còn lại: Dynamic 3% risk = $6/trade, chặn sàn min_lot=0.01
  3. Đội hình 5 mã (xóa XAGUSDm): ELITE_5_SYMBOLS
  4. 15 studies = 5 mã × 3 TF (M1, M5, M15)
  5. Giữ nguyên: 73 features, H1 filter, Inside Bar force close, Gate=3
"""
from __future__ import annotations

import gc
import json
import logging
import math
import multiprocessing
import multiprocessing.shared_memory as shm_mod
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.symbol_registry import (
    SYMBOL_PROPS, ELITE_5_SYMBOLS,
    calc_lot_size, calc_margin,
)
from core.feature_engine import build_feature_matrix

logger = logging.getLogger("EV05D")

# ─── Constants (Ev05d: $200 Live Rehearsal) ───────────────────────────────────
N_FEATURES      = 73
IS_RATIO        = 0.70
BATCH_SIZE      = 50
INITIAL_CAPITAL = 200.0            # ← $200 (Sếp Vũ chốt)
RISK_PER_TRADE  = 0.03             # 3% = $6/trade
MAX_DD_LIMIT    = 0.28
OOS_DD_GATE     = 0.25
MAX_CONCURRENT  = 3                # Global Risk Gate
MAX_MARGIN_PCT  = 0.30
FEAT_H1_TREND   = 23

# Features frozen (kế thừa Ev05c)
FROZEN_FEATURES = [0, 3, 5, 17, 23, 24, 39, 41, 42, 44, 48, 50, 52, 53, 54, 55, 57, 58, 59, 68]
ACTIVE_FEATURES = [i for i in range(N_FEATURES) if i not in FROZEN_FEATURES]

# TF search space (kế thừa Ev05c)
TF_SS = {
    "M1":  {"threshold": (0.55, 0.95), "sl": (1.5, 4.0), "rfb": (1.5, 5.0), "slip": (0.001, 0.01), "cd": (5, 20)},
    "M5":  {"threshold": (0.05, 0.80), "sl": (1.5, 3.5), "rfb": (0.5, 3.0), "slip": (0.0005, 0.005), "cd": (1, 8)},
    "M15": {"threshold": (0.05, 0.80), "sl": (2.0, 4.0), "rfb": (0.5, 3.0), "slip": (0.0005, 0.003), "cd": (1, 5)},
}
MIN_TRADES = {"M1": 80, "M5": 200, "M15": 150}


# ─── Hybrid Lot Sizing (Ev05d Core Change) ───────────────────────────────────

def calc_lot_ev05d(
    equity: float,
    risk_pct: float,
    sl_price_dist: float,
    symbol: str,
    entry_price: float,
    max_margin_pct: float = MAX_MARGIN_PCT,
) -> float:
    """
    Lot sizing Hybrid:
    - XAUUSDm: FIX CỨNG 0.01 (bất chấp equity/SL)
    - Còn lại: Dynamic 3% risk, chặn sàn min_lot=0.01
    """
    if symbol == "XAUUSDm":
        return 0.01   # ← Lệnh Sếp Vũ: Gold fix cứng

    # Dynamic: 3% risk ($6 khi equity=$200)
    props   = SYMBOL_PROPS.get(symbol, {})
    min_lot = props.get("min_lot", 0.01)
    max_lot = props.get("max_lot", 100.0)

    risk_usd = equity * risk_pct
    # lot = risk_usd / (sl_price_dist × contract_size)
    cs = props.get("contract_size", 100)
    denom = sl_price_dist * cs
    if denom <= 0:
        return min_lot
    lot = risk_usd / denom

    # Margin check
    margin_needed = calc_margin(symbol, lot, entry_price)
    max_margin    = equity * max_margin_pct
    if margin_needed > max_margin and lot > min_lot:
        scale = max_margin / max(margin_needed, 1e-10)
        lot   = max(min_lot, lot * scale)

    return float(max(min_lot, min(max_lot, round(lot, 3))))


# ─── Core Backtest (kế thừa Ev05c + Hybrid Lot) ──────────────────────────────

def run_backtest_ev05d(
    features:   np.ndarray,    # (N, 73)
    raw_ohlcv:  np.ndarray,    # (N, 6) OHLCV+time
    fvg_zones:  dict,
    weights:    np.ndarray,    # (73,) learned weights
    threshold:  float,
    sl_mult:    float,
    rr_fallback: float,
    slippage_pct: float,
    cooldown:   int,
    symbol:     str,
    spread_cost: float = 0.00015,
    max_dd:     float  = MAX_DD_LIMIT,
    initial_capital: float = INITIAL_CAPITAL,
    risk_pct:   float  = RISK_PER_TRADE,
    h1_filter:  bool   = True,
    h1_inside_bar: np.ndarray | None = None,
) -> dict | None:
    """
    Split Ticket backtest — Ev05d version ($200, Hybrid Lot).
    Giống hệt Ev05c, chỉ thay calc_lot_size bằng calc_lot_ev05d.
    """
    N = len(features)
    if N < 500:
        return None

    # ── Signal generation ─────────────────────────────────────────────────
    score = features.astype("float32") @ weights
    h1_trend = features[:, FEAT_H1_TREND]

    if h1_filter:
        ls = (score >  threshold) & (h1_trend >= 0)
        ss = (score < -threshold) & (h1_trend <= 0)
    else:
        ls = score >  threshold
        ss = score < -threshold

    ls[:200] = False; ss[:200] = False

    # ── OHLCV ─────────────────────────────────────────────────────────────
    o = raw_ohlcv[:, 1].astype("float64")
    h = raw_ohlcv[:, 2].astype("float64")
    l = raw_ohlcv[:, 3].astype("float64")
    c = raw_ohlcv[:, 4].astype("float64")

    pc  = np.roll(c, 1); pc[0] = c[0]
    tr  = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = np.convolve(tr, np.ones(14) / 14, mode="same"); atr[:13] = tr[:13]

    sl_d = atr * sl_mult
    slip = atr * slippage_pct

    # ── FVG zones ─────────────────────────────────────────────────────────
    opp_bear  = fvg_zones["opp_bear_mid"]
    next_bear = fvg_zones["next_bear_mid"]
    opp_bull  = fvg_zones["opp_bull_mid"]
    next_bull = fvg_zones["next_bull_mid"]

    pnls   = []
    equity = initial_capital
    peak   = initial_capital
    max_dd_hit = 0.0

    _ib = h1_inside_bar if (h1_inside_bar is not None and len(h1_inside_bar) == N) \
          else np.zeros(N, dtype=np.float32)

    # ── Leg state ─────────────────────────────────────────────────────────
    in_a = False; d_a = 0; ep_a = sl_a = tp_a = lot_a = 0.0
    in_b = False; d_b = 0; ep_b = sl_b = tp_b = lot_b = 0.0
    leg_a_hit    = False
    cooldown_left = 0

    for i in range(N):
        hi = float(h[i]); lo = float(l[i]); op = float(o[i])
        sl_i   = float(sl_d[i])
        slip_i = float(slip[i])

        # ── H1 INSIDE BAR FORCE CLOSE (Sếp Vũ Rule) ─────────────
        if _ib[i] > 0.5 and (in_a or in_b):
            if in_a:
                pnl = (op - ep_a) * d_a * lot_a - spread_cost * lot_a
                pnls.append(pnl); equity += pnl; in_a = False
            if in_b:
                pnl = (op - ep_b) * d_b * lot_b - spread_cost * lot_b
                pnls.append(pnl); equity += pnl; in_b = False
            leg_a_hit = False; cooldown_left = cooldown
            if equity > peak: peak = equity
            if peak > initial_capital:
                dd = (peak - equity) / peak
                if dd > max_dd_hit: max_dd_hit = dd
            continue

        # ── Process Leg A ─────────────────────────────────────────
        if in_a:
            hit_sl_a = (d_a ==  1 and lo <= sl_a) or (d_a == -1 and hi >= sl_a)
            hit_tp_a = (d_a ==  1 and hi >= tp_a) or (d_a == -1 and lo <= tp_a)
            if hit_tp_a:
                pnl = (tp_a - ep_a) * d_a * lot_a - spread_cost * lot_a - slip_i * lot_a
                pnls.append(pnl); equity += pnl
                in_a = False; leg_a_hit = True
                if in_b: sl_b = ep_b + slip_i * d_b
            elif hit_sl_a:
                pnl = (sl_a - ep_a) * d_a * lot_a - spread_cost * lot_a - slip_i * lot_a
                pnls.append(pnl); equity += pnl; in_a = False
            if equity > peak: peak = equity
            if peak > initial_capital:
                dd = (peak - equity) / peak
                if dd > max_dd_hit: max_dd_hit = dd
            if max_dd_hit > max_dd: break

        # ── Process Leg B ─────────────────────────────────────────
        if in_b:
            hit_sl_b = (d_b ==  1 and lo <= sl_b) or (d_b == -1 and hi >= sl_b)
            hit_tp_b = (d_b ==  1 and hi >= tp_b) or (d_b == -1 and lo <= tp_b)
            if hit_tp_b:
                pnl = (tp_b - ep_b) * d_b * lot_b - spread_cost * lot_b - slip_i * lot_b
                pnls.append(pnl); equity += pnl
                in_b = False; leg_a_hit = False; cooldown_left = cooldown
            elif hit_sl_b:
                pnl = (sl_b - ep_b) * d_b * lot_b - spread_cost * lot_b - slip_i * lot_b
                pnls.append(pnl); equity += pnl
                in_b = False; leg_a_hit = False; cooldown_left = cooldown
            if equity > peak: peak = equity
            if peak > initial_capital:
                dd = (peak - equity) / peak
                if dd > max_dd_hit: max_dd_hit = dd

        # ── New Entry ─────────────────────────────────────────────
        if not in_a and not in_b:
            if cooldown_left > 0:
                cooldown_left -= 1
                continue

            # ★ HYBRID LOT SIZING (Ev05d core) ★
            entry_price = op
            lot_total = calc_lot_ev05d(
                equity, risk_pct, sl_d[i] + slip[i],
                symbol, entry_price, MAX_MARGIN_PCT,
            )
            if lot_total <= 0:
                continue
            lot_half = lot_total * 0.5

            opp_b_i   = float(opp_bear[i])
            next_b_i  = float(next_bear[i])
            opp_bu_i  = float(opp_bull[i])
            next_bu_i = float(next_bull[i])

            if ls[i]:
                ep_entry   = op + slip_i
                sl_price   = ep_entry - sl_i
                tp_a_price = opp_b_i  if opp_b_i  > ep_entry else ep_entry + sl_i * rr_fallback
                tp_b_price = next_b_i if next_b_i > tp_a_price else tp_a_price * 1.005
                in_a = True; d_a = 1
                ep_a = ep_entry; sl_a = sl_price; tp_a = tp_a_price; lot_a = lot_half
                in_b = True; d_b = 1
                ep_b = ep_entry; sl_b = sl_price; tp_b = tp_b_price; lot_b = lot_half
                leg_a_hit = False
            elif ss[i]:
                ep_entry   = op - slip_i
                sl_price   = ep_entry + sl_i
                tp_a_price = opp_bu_i  if opp_bu_i  < ep_entry else ep_entry - sl_i * rr_fallback
                tp_b_price = next_bu_i if next_bu_i < tp_a_price else tp_a_price * 0.995
                in_a = True; d_a = -1
                ep_a = ep_entry; sl_a = sl_price; tp_a = tp_a_price; lot_a = lot_half
                in_b = True; d_b = -1
                ep_b = ep_entry; sl_b = sl_price; tp_b = tp_b_price; lot_b = lot_half
                leg_a_hit = False

    n = len(pnls)
    if n == 0:
        return None

    arr   = np.array(pnls, dtype="float64")
    wins  = arr[arr > 0]; loss = arr[arr <= 0]
    gp    = float(wins.sum()) if len(wins) else 0.0
    gl    = float(loss.sum()) if len(loss) else 0.0
    wr    = len(wins) / n
    pf    = min(abs(gp / gl), 999.0) if gl != 0 else (999.0 if gp > 0 else 0.0)
    net_ev     = (gp + gl) / n
    net_profit = gp + gl
    dd_ratio   = min(max_dd_hit, 1.0)

    freq_bonus = math.log10(max(10, n))
    dd_penalty = 1.0 - (dd_ratio / MAX_DD_LIMIT)
    fitness    = (net_profit / initial_capital) * freq_bonus * dd_penalty

    return {
        "wr": wr, "pf": pf, "n": n,
        "net_ev": net_ev, "net_profit": net_profit,
        "max_dd": max_dd_hit, "fitness": fitness,
    }


# ─── Worker subprocess ────────────────────────────────────────────────────────

def _ev05d_worker(tid, params, symbol, tf,
                  fn, fs, fd, rn, rs, rd, mn, ms, md,
                  zn_names, zn_shapes, zn_dtypes,
                  ibn, ibs, ibd,
                  spread):
    """Subprocess: load feats+zones+ib từ SHM, chạy IS+OOS backtest."""
    time.sleep(random.uniform(0, 1.5))
    try:
        def att(name, shape, dtype):
            s = shm_mod.SharedMemory(name=name)
            return np.ndarray(shape, dtype=np.dtype(dtype), buffer=s.buf), s

        feats,  _s1 = att(fn, fs, fd)
        raw,    _s2 = att(rn, rs, rd)
        is_msk, _s3 = att(mn, ms, md)
        h1_ib,  _s4 = att(ibn, ibs, ibd)

        zones = {}; zone_shms = []
        for key, zname, zshape, zdtype in zip(
            ["opp_bear_mid","next_bear_mid","opp_bull_mid","next_bull_mid",
             "opp_bear_top","opp_bear_bot","opp_bull_top","opp_bull_bot",
             "opp_bear_dist_atr","opp_bull_dist_atr"],
            zn_names, zn_shapes, zn_dtypes
        ):
            arr, zs = att(zname, zshape, zdtype)
            zones[key] = arr.copy()
            zone_shms.append(zs)

        w   = np.array(params["weights"], dtype="float32")
        th  = float(params["threshold"])
        sl  = float(params["sl_mult"])
        rfb = float(params["rr_fallback"])
        sp  = float(params["slippage_pct"])
        cd  = int(params["cooldown"])
        min_tr = MIN_TRADES.get(tf, 100)

        is_mask_arr = is_msk.copy()
        h1_ib_arr   = h1_ib.copy()

        r_is = run_backtest_ev05d(
            feats[is_mask_arr], raw[is_mask_arr],
            {k: v[is_mask_arr] for k, v in zones.items()},
            w, th, sl, rfb, sp, cd, symbol, spread,
            h1_inside_bar=h1_ib_arr[is_mask_arr],
        )

        oos_mask = ~is_mask_arr
        r_oos = run_backtest_ev05d(
            feats[oos_mask], raw[oos_mask],
            {k: v[oos_mask] for k, v in zones.items()},
            w, th, sl, rfb, sp, cd, symbol, spread,
            h1_inside_bar=h1_ib_arr[oos_mask],
        )

        for s in (_s1, _s2, _s3, _s4): s.close()
        for zs in zone_shms: zs.close()

        if r_is is None or r_is["n"] < min_tr:
            return {"tid": tid, "status": "pruned_trades", "n": r_is["n"] if r_is else 0}
        if r_is["max_dd"] > MAX_DD_LIMIT:
            return {"tid": tid, "status": "pruned_dd"}

        fitness  = r_is["fitness"]
        oos_ok   = True
        oos_dd   = r_oos["max_dd"]   if r_oos else 1.0
        oos_prof = r_oos["net_profit"] if r_oos else -1.0

        if r_oos and oos_dd > OOS_DD_GATE:
            fitness = -abs(fitness); oos_ok = False
        elif r_oos and oos_prof < 0:
            fitness *= 0.5

        return {
            "tid": tid, "status": "ok",
            "wr": r_is["wr"], "pf": r_is["pf"], "n_is": r_is["n"],
            "net_ev": r_is["net_ev"], "net_profit": r_is["net_profit"],
            "max_dd": r_is["max_dd"],
            "oos_dd": oos_dd, "oos_profit": oos_prof, "oos_ok": oos_ok,
            "fitness": fitness,
        }

    except Exception as e:
        return {"tid": tid, "status": "error", "error": str(e)}


# ─── Train single study ───────────────────────────────────────────────────────

def train_study_ev05d(
    symbol:    str,
    tf:        str,
    data_dir:  str   = "data",
    n_trials:  int   = 2000,
    n_workers: int   = 40,
    spread:    float = 0.00015,
) -> optuna.Study:
    from tqdm import tqdm
    from core.feature_engine import _nearest_opposing_fvg_zones

    sym_plain = symbol.lower().replace("m", "")
    study_name = f"ev05d_{sym_plain}_{tf.lower()}"
    db_path    = f"data/optuna_{study_name}.db"

    logger.info(f"[{symbol}|{tf}] Loading features ({N_FEATURES} feats) + H1 InsideBar...")
    feats, raw, h1_ib = build_feature_matrix(symbol, data_dir)

    logger.info(f"[{symbol}|{tf}] Computing FVG zones...")
    o5 = raw[:, 1]; h5 = raw[:, 2]; l5 = raw[:, 3]; c5 = raw[:, 4]
    pc5 = np.roll(c5, 1); pc5[0] = c5[0]
    tr5  = np.maximum(h5-l5, np.maximum(np.abs(h5-pc5), np.abs(l5-pc5)))
    atr5 = np.convolve(tr5, np.ones(14)/14, mode="same").astype("float32")
    atr5[:13] = tr5[:13]
    zones = _nearest_opposing_fvg_zones(o5, h5, l5, c5, atr5, lookback=100)

    N = len(feats)
    is_cut  = int(N * IS_RATIO)
    is_mask = np.zeros(N, dtype=bool); is_mask[:is_cut] = True

    shms = []
    def alloc(arr):
        arr = np.ascontiguousarray(arr)
        s = shm_mod.SharedMemory(create=True, size=arr.nbytes)
        np.ndarray(arr.shape, dtype=arr.dtype, buffer=s.buf)[:] = arr[:]
        shms.append(s)
        return s.name, arr.shape, str(arr.dtype)

    fn, fs, fd = alloc(feats)
    rn, rs, rd = alloc(raw)
    mn, ms, md = alloc(is_mask)
    ibn, ibs, ibd = alloc(h1_ib)

    zone_keys = ["opp_bear_mid","next_bear_mid","opp_bull_mid","next_bull_mid",
                 "opp_bear_top","opp_bear_bot","opp_bull_top","opp_bull_bot",
                 "opp_bear_dist_atr","opp_bull_dist_atr"]
    zn_names, zn_shapes, zn_dtypes = [], [], []
    for k in zone_keys:
        zname, zshape, zdtype = alloc(zones[k])
        zn_names.append(zname); zn_shapes.append(zshape); zn_dtypes.append(zdtype)

    ss = TF_SS[tf]
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{db_path}",
        engine_kwargs={"connect_args": {"timeout": 60}},
    )
    study = optuna.create_study(
        study_name=study_name, direction="maximize",
        storage=storage, load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=80, multivariate=True, group=True,
            constant_liar=True, warn_independent_sampling=False,
            seed=(42 + hash(f"{symbol}{tf}") % 1000),
        ),
    )

    already_done = len(study.trials)
    remaining    = n_trials - already_done
    if remaining <= 0:
        logger.info(f"[{symbol}|{tf}] Already done ({already_done} trials)")
        for s in shms:
            try: s.close(); s.unlink()
            except: pass
        return study

    best_v = -1e9; done = 0; completed_c = 0; pruned_c = 0
    pbar = tqdm(total=remaining, desc=f"  [{tf}] {symbol}", ncols=90)

    try:
        with ProcessPoolExecutor(max_workers=n_workers) as exe:
            while done < remaining:
                batch  = min(BATCH_SIZE, remaining - done)
                asked  = []

                for _ in range(batch):
                    t = study.ask()
                    w = np.zeros(N_FEATURES, dtype="float32")
                    for i in ACTIVE_FEATURES:
                        w[i] = t.suggest_float(f"w{i}", -1.0, 1.0)
                    nm = float(np.linalg.norm(w))
                    w  = w / nm if nm > 1e-6 else w

                    th  = t.suggest_float("threshold",    *ss["threshold"])
                    sl  = t.suggest_float("sl_mult",      *ss["sl"])
                    rfb = t.suggest_float("rr_fallback",  *ss["rfb"])
                    sp  = t.suggest_float("slippage_pct", *ss["slip"])
                    cd  = t.suggest_int("cooldown",        *ss["cd"])

                    params = {
                        "weights": w.tolist(), "threshold": th,
                        "sl_mult": sl, "rr_fallback": rfb,
                        "slippage_pct": sp, "cooldown": cd,
                    }
                    asked.append((t, params))

                futures = {
                    exe.submit(
                        _ev05d_worker, t.number, p, symbol, tf,
                        fn, fs, fd, rn, rs, rd, mn, ms, md,
                        zn_names, zn_shapes, zn_dtypes,
                        ibn, ibs, ibd,
                        spread,
                    ): t
                    for t, p in asked
                }

                for fut in as_completed(futures):
                    trial = futures[fut]
                    res   = fut.result()

                    if res["status"] == "ok":
                        trial.set_user_attr("winrate",    res["wr"])
                        trial.set_user_attr("pf",         res["pf"])
                        trial.set_user_attr("n_is",       res["n_is"])
                        trial.set_user_attr("net_ev",     res["net_ev"])
                        trial.set_user_attr("net_profit", res["net_profit"])
                        trial.set_user_attr("max_dd",     res["max_dd"])
                        trial.set_user_attr("oos_dd",     res.get("oos_dd", -1))
                        trial.set_user_attr("oos_profit", res.get("oos_profit", 0))
                        trial.set_user_attr("oos_ok",     int(res.get("oos_ok", True)))
                        study.tell(trial, res["fitness"])
                        completed_c += 1
                        if res["fitness"] > best_v: best_v = res["fitness"]
                    else:
                        study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                        pruned_c += 1

                    done += 1
                    pbar.set_postfix({"best": f"{best_v:.3f}" if best_v > -1e9 else "N/A",
                                      "cpl": completed_c, "prn": pruned_c}, refresh=False)
                    pbar.update(1)

    finally:
        pbar.close()
        for s in shms:
            try: s.close(); s.unlink()
            except: pass

    logger.info(
        f"[{symbol}|{tf}] ✓ done={already_done+done} cpl={completed_c} "
        f"prune={pruned_c} best={best_v:.4f}"
    )
    return study


# ─── Queue Runner ─────────────────────────────────────────────────────────────

def run_ev05d_queue(
    symbols:     list[str] = ELITE_5_SYMBOLS,
    tfs:         list[str] | None = None,
    data_dir:    str   = "data",
    n_trials:    int   = 2000,
    n_workers:   int   = 40,
    spread:      float = 0.00015,
    report_out:  str   = "logs/An_Latest_Report.md",
) -> None:
    """Queue tuần tự: symbol × TF cho Ev05d ($200 Live Rehearsal)."""
    if tfs is None:
        tfs = ["M1", "M5", "M15"]

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s UTC] - [%(levelname)-7s] - [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    total  = len(symbols) * len(tfs)
    done_n = 0

    print(f"\n{'='*65}")
    print(f"  EV05d $200 LIVE REHEARSAL — {len(symbols)} symbols × {len(tfs)} TF")
    print(f"  Capital: $200 | XAU=0.01 fix | Others=3% dynamic ($6/trade)")
    print(f"  Total: {total} studies × {n_trials:,} trials = {total*n_trials:,} kịch bản")
    print(f"{'='*65}\n")

    for i, sym in enumerate(symbols):
        for j, tf in enumerate(tfs):
            done_n += 1
            pct = done_n / total * 100
            bar = "█" * int(pct/5) + "░" * (20 - int(pct/5))
            print(f"\n[{done_n}/{total}] {sym}|{tf} | Overall [{bar}] {pct:.0f}%")

            try:
                train_study_ev05d(sym, tf, data_dir, n_trials, n_workers, spread)
            except Exception as e:
                logger.error(f"[{sym}|{tf}] FAILED: {e}")

            gc.collect()

    print(f"\n{'='*65}")
    print(f"  ✅ EV05d HOÀN TẤT — {total} studies × {n_trials} trials!")
    print(f"  Generating Portfolio Fact Sheet...")
    print(f"{'='*65}")

    # Auto-generate Report khi xong
    try:
        from engine.portfolio_report import generate_fact_sheet
        generate_fact_sheet(
            output_path="logs/ev05d_fact_sheet.md",
            data_dir=data_dir,
        )
    except Exception as e:
        logger.warning(f"Fact sheet generation failed: {e}")

    print(f"  Report → {report_out}")
