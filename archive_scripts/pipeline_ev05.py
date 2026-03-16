#!/usr/bin/env python3
"""
pipeline_ev05.py — Ev05: 40 Battlefields Multi-Asset Engine
=============================================================
Architecture: Queue tuần tự (symbol A → symbol B) để bảo vệ server.
Mỗi symbol: chạy 4 TF lần lượt (M1 → M5 → M15 → H1).
Tổng: 10 symbols × 4 TF = 40 Optuna studies × 2000 trials = 80,000 kịch bản.

Key upgrades vs Ev04:
  - Symbol-aware lot sizing (từ symbol_registry.py)
  - H1 Swap fee trong backtest
  - Dynamic Search Space per TF (Swing/Intraday/Scalping/Sniper)
  - Queue system: không chạy song song 40 studies
  - FVG Split Ticket giữ nguyên từ Ev04
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

logging.basicConfig(
    format="[%(asctime)s UTC] - [%(levelname)-8s] - [EV05] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger("EV05")

# ─── Constants ────────────────────────────────────────────────────────────────
N_FEATURES      = 73
IS_RATIO        = 0.70
BATCH_SIZE      = 50
INITIAL_CAPITAL = 20_000.0
RISK_PER_TRADE  = 0.01
MAX_DD_LIMIT    = 0.28
OOS_DD_GATE     = 0.25

FROZEN_FEATURES = [0, 3, 5, 17, 23, 24, 39, 41, 42, 44, 48, 50, 52, 53, 54, 55, 57, 58, 59, 68]
ACTIVE_FEATURES = [i for i in range(N_FEATURES) if i not in FROZEN_FEATURES]

# ─── Dynamic Search Space per TF ─────────────────────────────────────────────
TF_SEARCH_SPACES = {
    "H1": {
        # Swing — lệnh ôm lâu, cần RR cao, threshold vừa (nhiều hơn M1 nhưng chặt hơn M5)
        "threshold_range":  (0.40, 0.85),
        "sl_range":         (2.0, 5.0),
        "rr_fallback_range":(2.0, 8.0),
        "slip_range":       (0.0005, 0.003),
        "cooldown_range":   (1, 4),         # H1 bars = 4 giờ cooldown max
        "min_trades_is":    50,             # H1 ít lệnh hơn
        "fitness_mode":     "dynamic_fvg",
        "apply_swap":       True,
    },
    "M15": {
        # Intraday — chuẩn Ev04
        "threshold_range":  (0.45, 0.85),
        "sl_range":         (1.5, 4.0),
        "rr_fallback_range":(1.5, 5.0),
        "slip_range":       (0.0005, 0.003),
        "cooldown_range":   (1, 6),
        "min_trades_is":    150,
        "fitness_mode":     "dynamic_fvg",
        "apply_swap":       False,
    },
    "M5": {
        # Scalping — threshold cao hơn, RR thấp hơn (spread bào mòn)
        "threshold_range":  (0.55, 0.90),
        "sl_range":         (1.0, 3.0),
        "rr_fallback_range":(1.0, 2.5),
        "slip_range":       (0.0005, 0.005),
        "cooldown_range":   (1, 10),
        "min_trades_is":    200,
        "fitness_mode":     "dynamic_fvg",
        "apply_swap":       False,
    },
    "M1": {
        # Scalping Sniper — threshold siết chặt nhất (từ fix Ev04)
        "threshold_range":  (0.65, 0.95),
        "sl_range":         (1.5, 4.0),
        "rr_fallback_range":(1.0, 2.5),
        "slip_range":       (0.001, 0.01),
        "cooldown_range":   (5, 25),
        "min_trades_is":    80,
        "fitness_mode":     "sniper_fvg",
        "apply_swap":       False,
    },
}

TF_ORDER = ["M1", "M5", "M15", "H1"]   # thứ tự chạy per symbol

BARS_PER_DAY = {"M1": 1440, "M5": 288, "M15": 96, "H1": 24}


# ─── Backtest Ev05 (extend Ev04 + swap + symbol lot) ─────────────────────────

def run_backtest_ev05(
    features: np.ndarray,
    raw_ohlcv: np.ndarray,
    fvg_zones: dict,
    weights: np.ndarray,
    threshold: float,
    sl_mult: float,
    rr_fallback: float,
    slippage_pct: float,
    cooldown: int,
    spread_cost: float,
    pip_value_per_lot: float,
    pip_size: float,
    contract_size: float,
    swap_per_bar: float = 0.0,      # USD/lot/bar (H1 swap)
    max_dd: float = MAX_DD_LIMIT,
    initial_capital: float = INITIAL_CAPITAL,
    risk_pct: float = RISK_PER_TRADE,
    fitness_mode: str = "dynamic_fvg",
    apply_swap: bool = False,
) -> dict | None:
    """
    Split Ticket backtest với symbol-aware lot sizing và H1 swap.
    Lot size = risk_usd / (sl_pips × pip_value_per_lot).
    """
    N = len(features)
    if N < 500:
        return None

    score = features.astype("float32") @ weights
    ls = score >  threshold
    ss = score < -threshold
    ls[:200] = False; ss[:200] = False

    o = raw_ohlcv[:, 1].astype("float64")
    h = raw_ohlcv[:, 2].astype("float64")
    l = raw_ohlcv[:, 3].astype("float64")
    c = raw_ohlcv[:, 4].astype("float64")

    pc  = np.roll(c, 1); pc[0] = c[0]
    tr  = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = np.convolve(tr, np.ones(14) / 14, mode="same"); atr[:13] = tr[:13]

    sl_d = atr * sl_mult
    slip = atr * slippage_pct

    opp_bear = fvg_zones["opp_bear_mid"]
    next_bear = fvg_zones["next_bear_mid"]
    opp_bull  = fvg_zones["opp_bull_mid"]
    next_bull  = fvg_zones["next_bull_mid"]

    pnls = []
    equity = initial_capital
    peak   = initial_capital
    max_dd_hit = 0.0

    in_a = False; d_a = 0; ep_a = sl_a = tp_a = lot_a = 0.0
    in_b = False; d_b = 0; ep_b = sl_b = tp_b = lot_b = 0.0
    bars_held_a = bars_held_b = 0
    cooldown_left = 0

    for i in range(N):
        hi = float(h[i]); lo = float(l[i]); op = float(o[i]); cl = float(c[i])
        sl_i   = float(sl_d[i])
        slip_i = float(slip[i])

        # ── Leg A ──
        if in_a:
            bars_held_a += 1
            if apply_swap:
                swap_cost_a = abs(swap_per_bar) * lot_a
                equity -= swap_cost_a
                pnls.append(-swap_cost_a)  # record swap as micro-loss

            hit_sl_a = (d_a ==  1 and lo <= sl_a) or (d_a == -1 and hi >= sl_a)
            hit_tp_a = (d_a ==  1 and hi >= tp_a) or (d_a == -1 and lo <= tp_a)

            if hit_tp_a:
                pnl = (tp_a - ep_a) * d_a * lot_a - spread_cost * lot_a - slip_i * lot_a
                pnls.append(pnl); equity += pnl; in_a = False
                if in_b: sl_b = ep_b + slip_i * d_b   # BE trigger
            elif hit_sl_a:
                pnl = (sl_a - ep_a) * d_a * lot_a - spread_cost * lot_a - slip_i * lot_a
                pnls.append(pnl); equity += pnl; in_a = False

            if equity > peak: peak = equity
            if peak > initial_capital:
                dd = (peak - equity) / peak
                if dd > max_dd_hit: max_dd_hit = dd

        # ── Leg B ──
        if in_b:
            bars_held_b += 1
            if apply_swap:
                swap_cost_b = abs(swap_per_bar) * lot_b
                equity -= swap_cost_b
                pnls.append(-swap_cost_b)

            hit_sl_b = (d_b ==  1 and lo <= sl_b) or (d_b == -1 and hi >= sl_b)
            hit_tp_b = (d_b ==  1 and hi >= tp_b) or (d_b == -1 and lo <= tp_b)

            if hit_tp_b:
                pnl = (tp_b - ep_b) * d_b * lot_b - spread_cost * lot_b - slip_i * lot_b
                pnls.append(pnl); equity += pnl; in_b = False; cooldown_left = cooldown
            elif hit_sl_b:
                pnl = (sl_b - ep_b) * d_b * lot_b - spread_cost * lot_b - slip_i * lot_b
                pnls.append(pnl); equity += pnl; in_b = False; cooldown_left = cooldown

            if equity > peak: peak = equity
            if peak > initial_capital:
                dd = (peak - equity) / peak
                if dd > max_dd_hit: max_dd_hit = dd

        # ── New Entry ──
        if not in_a and not in_b:
            if cooldown_left > 0:
                cooldown_left -= 1; continue

            sl_dist  = float(sl_d[i]) + float(slip[i])
            # Symbol-aware lot sizing
            sl_pips  = sl_dist / max(pip_size, 1e-10)
            denom    = sl_pips * pip_value_per_lot
            risk_usd = equity * risk_pct
            lot_total = risk_usd / max(denom, 1e-10)
            lot_total = max(0.001, min(lot_total, 500.0))
            lot_half  = lot_total * 0.5

            opp_b_i   = float(opp_bear[i]); next_b_i  = float(next_bear[i])
            opp_bu_i  = float(opp_bull[i]); next_bu_i = float(next_bull[i])

            if ls[i]:
                ep_e = op + slip_i; sl_p = ep_e - sl_dist / 2
                tp_a_ = max(opp_b_i,  ep_e + sl_dist * rr_fallback * 0.5)
                tp_b_ = max(next_b_i, tp_a_ * 1.005)
                in_a = True; d_a = 1; ep_a = ep_e; sl_a = sl_p; tp_a = tp_a_; lot_a = lot_half; bars_held_a = 0
                in_b = True; d_b = 1; ep_b = ep_e; sl_b = sl_p; tp_b = tp_b_; lot_b = lot_half; bars_held_b = 0

            elif ss[i]:
                ep_e = op - slip_i; sl_p = ep_e + sl_dist / 2
                tp_a_ = min(opp_bu_i,  ep_e - sl_dist * rr_fallback * 0.5)
                tp_b_ = min(next_bu_i, tp_a_ * 0.995)
                in_a = True; d_a = -1; ep_a = ep_e; sl_a = sl_p; tp_a = tp_a_; lot_a = lot_half; bars_held_a = 0
                in_b = True; d_b = -1; ep_b = ep_e; sl_b = sl_p; tp_b = tp_b_; lot_b = lot_half; bars_held_b = 0

    n = len(pnls)
    # Filter out tiny swap micro-entries
    pnl_arr = np.array([p for p in pnls if abs(p) > 1e-6], dtype="float64")
    n_trades = len(pnl_arr)
    if n_trades == 0:
        return None

    wins = pnl_arr[pnl_arr > 0]; loss = pnl_arr[pnl_arr <= 0]
    gp   = float(wins.sum()) if len(wins) else 0.0
    gl   = float(loss.sum()) if len(loss) else 0.0
    wr   = len(wins) / n_trades
    pf   = min(abs(gp / gl), 999.0) if gl != 0 else (999.0 if gp > 0 else 0.0)
    net_ev     = (gp + gl) / n_trades
    net_profit = gp + gl
    dd_ratio   = min(max_dd_hit, 1.0)

    freq_bonus = math.log10(max(10, n_trades))
    dd_penalty = max(0.0, 1.0 - (dd_ratio / MAX_DD_LIMIT))

    if fitness_mode == "sniper_fvg":
        ev_bonus = math.log10(max(1.0, abs(net_ev) * 1000))
        fitness  = (net_profit / initial_capital) * ev_bonus * dd_penalty
    else:
        fitness = (net_profit / initial_capital) * freq_bonus * dd_penalty

    return {
        "wr": wr, "pf": pf, "n": n_trades,
        "net_ev": net_ev, "net_profit": net_profit,
        "max_dd": max_dd_hit, "gp": gp, "gl": abs(gl),
        "fitness": fitness,
    }


# ─── Worker ───────────────────────────────────────────────────────────────────

def _ev05_worker(tid, params, ss, symbol_cfg,
                 fn, fs, fd, rn, rs, rd, mn, ms, md,
                 zn_names, zn_shapes, zn_dtypes, spread):
    time.sleep(random.uniform(0, 1.5))
    try:
        def att(name, shape, dtype):
            s = shm_mod.SharedMemory(name=name)
            return np.ndarray(shape, dtype=np.dtype(dtype), buffer=s.buf), s

        feats, _s1 = att(fn, fs, fd)
        raw,   _s2 = att(rn, rs, rd)
        ismsk, _s3 = att(mn, ms, md)

        zones = {}; zone_shms = []
        for key, zname, zshape, zdtype in zip(
            ["opp_bear_mid","next_bear_mid","opp_bull_mid","next_bull_mid",
             "opp_bear_top","opp_bear_bot","opp_bull_top","opp_bull_bot",
             "opp_bear_dist_atr","opp_bull_dist_atr"],
            zn_names, zn_shapes, zn_dtypes
        ):
            arr, zs = att(zname, zshape, zdtype)
            zones[key] = arr.copy(); zone_shms.append(zs)

        w   = np.array(params["weights"], dtype="float32")
        th  = float(params["threshold"])
        sl  = float(params["sl_mult"])
        rfb = float(params["rr_fallback"])
        sp  = float(params["slippage_pct"])
        cd  = int(params["cooldown"])

        pv   = symbol_cfg["pip_value_per_lot"]
        ps   = symbol_cfg["pip_size"]
        cs   = symbol_cfg["contract_size"]
        swpb = symbol_cfg["swap_per_bar"]
        appl = symbol_cfg["apply_swap"]
        fm   = ss["fitness_mode"]

        is_mask = ismsk.copy()
        feats_is = feats[is_mask]; raw_is = raw[is_mask]
        zones_is = {k: v[is_mask] for k, v in zones.items()}

        r_is = run_backtest_ev05(
            feats_is, raw_is, zones_is, w, th, sl, rfb, sp, cd, spread,
            pv, ps, cs, swpb, fitness_mode=fm, apply_swap=appl,
        )

        oos_mask = ~is_mask
        feats_oos = feats[oos_mask]; raw_oos = raw[oos_mask]
        zones_oos = {k: v[oos_mask] for k, v in zones.items()}

        r_oos = run_backtest_ev05(
            feats_oos, raw_oos, zones_oos, w, th, sl, rfb, sp, cd, spread,
            pv, ps, cs, swpb, fitness_mode=fm, apply_swap=appl,
        )

        for s in (_s1, _s2, _s3): s.close()
        for zs in zone_shms: zs.close()

        if r_is is None:
            return {"tid": tid, "status": "pruned"}
        if r_is["n"] < ss["min_trades_is"]:
            return {"tid": tid, "status": "pruned_trades", "n": r_is["n"]}
        if r_is["max_dd"] > MAX_DD_LIMIT:
            return {"tid": tid, "status": "pruned_dd", "dd": r_is["max_dd"]}

        fitness = r_is["fitness"]
        oos_ok  = True
        oos_dd  = r_oos["max_dd"] if r_oos else 1.0
        oos_pnl = r_oos["net_profit"] if r_oos else -1.0

        if r_oos and oos_dd > OOS_DD_GATE:
            fitness = -abs(fitness); oos_ok = False
        elif r_oos and oos_pnl < 0:
            fitness *= 0.5

        return {
            "tid": tid, "status": "ok",
            "wr": r_is["wr"], "pf": r_is["pf"], "n_is": r_is["n"],
            "net_ev": r_is["net_ev"], "net_profit": r_is["net_profit"],
            "max_dd": r_is["max_dd"], "fitness": fitness,
            "oos_dd": oos_dd, "oos_profit": oos_pnl, "oos_ok": oos_ok,
        }
    except Exception as e:
        return {"tid": tid, "status": "error", "error": str(e)}


# ─── Single Study ─────────────────────────────────────────────────────────────

def run_study_ev05(symbol: str, tf: str, data_dir: str,
                   n_trials: int, n_workers: int, spread: float,
                   results_dir: str) -> dict:
    from tqdm import tqdm
    from core.feature_engine import build_feature_matrix, _nearest_opposing_fvg_zones
    from core.symbol_registry import (SYMBOL_PROPS, get_pip_value_per_lot,
                                      get_swap_per_bar)

    ss = TF_SEARCH_SPACES[tf]
    props = SYMBOL_PROPS.get(symbol, SYMBOL_PROPS["EURUSDm"])

    study_name = f"ev05_{symbol.lower().replace('m','')}_{tf.lower()}"
    db_path    = f"data/optuna_{study_name}.db"

    logger.info(f"[{symbol}|{tf}] Loading features...")
    feats, raw = build_feature_matrix(symbol, data_dir)

    o5 = raw[:, 1]; h5 = raw[:, 2]; l5 = raw[:, 3]; c5 = raw[:, 4]
    pc5 = np.roll(c5, 1); pc5[0] = c5[0]
    tr5 = np.maximum(h5 - l5, np.maximum(np.abs(h5 - pc5), np.abs(l5 - pc5)))
    atr5 = np.convolve(tr5, np.ones(14) / 14, mode="same").astype("float32")
    atr5[:13] = tr5[:13]

    zones = _nearest_opposing_fvg_zones(o5, h5, l5, c5, atr5, lookback=100)

    N = len(feats)
    is_cut = int(N * IS_RATIO)
    is_mask = np.zeros(N, dtype=bool); is_mask[:is_cut] = True

    # Symbol-aware configs
    avg_price = float(np.nanmedian(c5))
    pv = get_pip_value_per_lot(symbol, avg_price)
    ps = props["pip_size"]; cs_ = props["contract_size"]
    swpb = get_swap_per_bar(symbol, +1, tf) if ss["apply_swap"] else 0.0

    symbol_cfg = {
        "pip_value_per_lot": pv, "pip_size": ps, "contract_size": cs_,
        "swap_per_bar": swpb, "apply_swap": ss["apply_swap"],
    }

    shms = []
    def alloc(arr):
        arr = np.ascontiguousarray(arr)
        s = shm_mod.SharedMemory(create=True, size=arr.nbytes)
        np.ndarray(arr.shape, dtype=arr.dtype, buffer=s.buf)[:] = arr[:]
        shms.append(s); return s.name, arr.shape, str(arr.dtype)

    fn, fs, fd = alloc(feats)
    rn, rs, rd = alloc(raw)
    mn, ms, md = alloc(is_mask)

    zone_keys = ["opp_bear_mid","next_bear_mid","opp_bull_mid","next_bull_mid",
                 "opp_bear_top","opp_bear_bot","opp_bull_top","opp_bull_bot",
                 "opp_bear_dist_atr","opp_bull_dist_atr"]
    zn_names, zn_shapes, zn_dtypes = [], [], []
    for k in zone_keys:
        zname, zshape, zdtype = alloc(zones[k])
        zn_names.append(zname); zn_shapes.append(zshape); zn_dtypes.append(zdtype)

    p_db = Path(db_path)
    if p_db.exists(): p_db.unlink()

    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{db_path}",
        engine_kwargs={"connect_args": {"timeout": 60}},
    )
    study = optuna.create_study(
        study_name=study_name, direction="maximize",
        storage=storage, load_if_exists=False,
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=80, multivariate=True, group=True,
            constant_liar=True, warn_independent_sampling=False,
            seed=(42 + hash(symbol + tf) % 10000)),
    )

    t0 = time.time()
    done = 0; best_v = -1e9; pruned_c = 0; completed_c = 0
    pbar = tqdm(total=n_trials,
                desc=f"  [{tf}] {symbol}",
                unit="trial", dynamic_ncols=True, leave=False)

    try:
        with ProcessPoolExecutor(max_workers=n_workers) as exe:
            while done < n_trials:
                batch = min(BATCH_SIZE, n_trials - done)
                asked = []

                for _ in range(batch):
                    t_trial = study.ask()
                    w = np.zeros(N_FEATURES, dtype="float32")
                    for i in ACTIVE_FEATURES:
                        w[i] = t_trial.suggest_float(f"w{i}", -1.0, 1.0)
                    for i in FROZEN_FEATURES:
                        t_trial.suggest_float(f"w{i}", 0.0, 0.0)
                    nm = float(np.linalg.norm(w))
                    w = w / nm if nm > 1e-6 else w

                    th  = t_trial.suggest_float("threshold",    *ss["threshold_range"])
                    sl  = t_trial.suggest_float("sl_mult",      *ss["sl_range"])
                    rfb = t_trial.suggest_float("rr_fallback",  *ss["rr_fallback_range"], log=True)
                    sp  = t_trial.suggest_float("slippage_pct", *ss["slip_range"], log=True)
                    cd  = t_trial.suggest_int("cooldown",       *ss["cooldown_range"])

                    params = {"weights": w.tolist(), "threshold": th,
                              "sl_mult": sl, "rr_fallback": rfb,
                              "slippage_pct": sp, "cooldown": cd}
                    asked.append((t_trial, params))

                futures = {
                    exe.submit(_ev05_worker, t_trial.number, p, ss, symbol_cfg,
                               fn, fs, fd, rn, rs, rd, mn, ms, md,
                               zn_names, zn_shapes, zn_dtypes, spread): t_trial
                    for t_trial, p in asked
                }

                for fut in as_completed(futures):
                    trial = futures[fut]; res = fut.result()
                    if res["status"] == "ok":
                        for k in ["wr","pf","n_is","net_ev","net_profit","max_dd","oos_dd","oos_profit","oos_ok"]:
                            if k in res: trial.set_user_attr(k if k!="wr" else "winrate", res[k])
                        trial.set_user_attr("winrate", res["wr"])
                        study.tell(trial, res["fitness"])
                        completed_c += 1
                        if res["fitness"] > best_v: best_v = res["fitness"]
                    else:
                        study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                        pruned_c += 1
                    done += 1; pbar.update(1)
                    if done % 100 == 0:
                        pbar.set_postfix(best=f"{best_v:.4f}" if best_v > -1e9 else "N/A",
                                         cpl=completed_c)
    finally:
        pbar.close()
        for s in shms:
            try: s.close(); s.unlink()
            except: pass

    elapsed = time.time() - t0
    logger.info(
        f"[{symbol}|{tf}] DONE | {done} trials | {elapsed:.0f}s | "
        f"cpl={completed_c} | pruned={pruned_c} | best={best_v:.4f}"
    )

    # Save result summary JSON
    result = {
        "symbol": symbol, "tf": tf, "study_name": study_name,
        "n_trials": done, "completed": completed_c, "pruned": pruned_c,
        "best_fitness": best_v, "elapsed_s": round(elapsed),
        "best_trial": None, "top3": [],
    }
    try:
        completed = [t for t in study.trials if t.state.name == "COMPLETE"]
        completed.sort(key=lambda t: t.value, reverse=True)
        if completed:
            bt = study.best_trial; ua = bt.user_attrs; p = bt.params
            result["best_trial"] = {
                "trial_no": bt.number, "fitness": bt.value,
                "is_profit": ua.get("net_profit", 0),
                "oos_profit": ua.get("oos_profit", 0),
                "oos_dd": ua.get("oos_dd", -1),
                "oos_ok": ua.get("oos_ok", 1),
                "winrate": ua.get("winrate", 0),
                "pf": ua.get("pf", 0),
                "max_dd": ua.get("max_dd", 0),
                "threshold": p.get("threshold", 0),
                "sl_mult": p.get("sl_mult", 0),
                "rr_fallback": p.get("rr_fallback", 0),
                "cooldown": p.get("cooldown", 0),
            }
            for t in completed[:3]:
                ua2 = t.user_attrs; p2 = t.params
                result["top3"].append({
                    "trial_no": t.number, "fitness": t.value,
                    "is_profit": ua2.get("net_profit", 0),
                    "oos_profit": ua2.get("oos_profit", 0),
                    "oos_dd": ua2.get("oos_dd", -1),
                    "winrate": ua2.get("winrate", 0),
                    "pf": ua2.get("pf", 0),
                })
    except Exception as e:
        logger.warning(f"[{symbol}|{tf}] Result parse error: {e}")

    out_json = Path(results_dir) / f"ev05_{symbol}_{tf}.json"
    out_json.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    del feats, raw, zones; gc.collect()
    return result


# ─── Queue Runner ─────────────────────────────────────────────────────────────

def run_ev05_queue(
    symbols: list[str],
    data_dir: str,
    n_trials: int,
    n_workers: int,
    spread: float,
    results_dir: str,
    report_out: str,
) -> None:
    """
    Queue tuần tự: symbol A (4 TF) → symbol B (4 TF) → ...
    Bảo vệ RAM server: giải phóng cache sau mỗi study.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    n_sym  = len(symbols)
    n_tf   = len(TF_ORDER)
    total  = n_sym * n_tf
    done   = 0
    all_results = []

    print(f"\n{'='*65}")
    print(f"  EV05 — {n_sym} SYMBOLS × {n_tf} TFs = {total} STUDIES")
    print(f"  {n_trials} trials/study | {n_sym*n_tf*n_trials:,} total kịch bản")
    print(f"  Queue mode: tuần tự per-symbol → bảo vệ Xeon server")
    print(f"{'='*65}\n")

    for sym_idx, symbol in enumerate(symbols, 1):
        print(f"\n{'─'*55}")
        print(f"  [{sym_idx}/{n_sym}] SYMBOL: {symbol}")
        print(f"{'─'*55}")

        for tf_idx, tf in enumerate(TF_ORDER, 1):
            done += 1
            pct  = done / total * 100
            bar  = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  [{sym_idx}/{n_sym}] {symbol} | [{tf_idx}/{n_tf}] {tf} "
                  f"| Overall [{bar}] {pct:.0f}% ({done}/{total})")

            try:
                result = run_study_ev05(
                    symbol=symbol, tf=tf, data_dir=data_dir,
                    n_trials=n_trials, n_workers=n_workers,
                    spread=spread, results_dir=results_dir,
                )
                all_results.append(result)
                bf = result.get("best_fitness", -999)
                bt = result.get("best_trial") or {}
                ip = bt.get("is_profit", 0)
                op = bt.get("oos_profit", 0)
                ok = "✅" if bt.get("oos_ok", 1) else "🔴"
                print(f"  ✓ {symbol}|{tf} → best={bf:.4f} {ok} | IS=${ip:,.0f} OOS=${op:,.0f}")
            except Exception as e:
                logger.error(f"[{symbol}|{tf}] FAILED: {e}")
                all_results.append({"symbol": symbol, "tf": tf, "error": str(e)})

            gc.collect()

    # Write leaderboard
    _write_leaderboard(all_results, report_out)
    print(f"\n✅ Đã hoàn tất thử lửa toàn mặt trận!")
    print(f"   Leaderboard → {report_out}")


# ─── Leaderboard Report ───────────────────────────────────────────────────────

def _write_leaderboard(results: list[dict], out_file: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# 🏆 EV05 — Multi-Asset Leaderboard ({now})",
        f"> 10 Symbols × 4 TFs = 40 Battlefields | 2000 trials/study | $20,000 Vốn",
        f"",
        f"## Bảng Phong Thần — Top Performers (IS + OOS)",
        f"",
        f"| Rank | Symbol | TF | Best Fit | IS Profit | OOS Profit | OOS DD | OOS Gate | WR |",
        f"|------|--------|----|----------|-----------|------------|--------|----------|----|",
    ]

    # Sort by best_fitness descending
    valid = [r for r in results if "best_trial" in r and r.get("best_trial")]
    valid.sort(key=lambda r: r.get("best_fitness", -999), reverse=True)

    medals = ["🥇","🥈","🥉"] + [f"#{i}" for i in range(4, 42)]
    for rank, r in enumerate(valid[:40]):
        medal = medals[rank] if rank < len(medals) else f"#{rank+1}"
        sym   = r["symbol"]
        tf    = r["tf"]
        bf    = r.get("best_fitness", -999)
        bt    = r.get("best_trial", {})
        ip    = bt.get("is_profit", 0)
        op    = bt.get("oos_profit", 0)
        od    = bt.get("oos_dd", -1)
        ok    = "✅" if bt.get("oos_ok", 1) else "🔴"
        wr    = bt.get("winrate", 0)
        lines.append(
            f"| {medal} | {sym} | {tf} | `{bf:.4f}` | "
            f"${ip:,.0f} | ${op:,.0f} | {od:.1%} | {ok} | {wr:.1%} |"
        )

    # Errors
    errors = [r for r in results if "error" in r]
    if errors:
        lines += [f"", f"## ⚠️ Failures", f""]
        for e in errors:
            lines.append(f"- {e['symbol']}|{e['tf']}: {e['error']}")

    # Per-symbol summary
    lines += [f"", f"## 📊 Kết Quả Chi Tiết Theo Symbol", f""]
    symbols_seen = []
    for r in results:
        sym = r["symbol"]
        if sym not in symbols_seen:
            symbols_seen.append(sym)
            lines += [f"", f"### {sym}", f"",
                      f"| TF | Best Fit | IS Profit | OOS Profit | OOS DD |",
                      f"|----|---------:|----------:|-----------:|-------:|"]
        tf  = r.get("tf", "?")
        bf  = r.get("best_fitness", -999)
        bt  = r.get("best_trial") or {}
        ip  = bt.get("is_profit", 0)
        op  = bt.get("oos_profit", 0)
        od  = bt.get("oos_dd", -1)
        lines.append(f"| {tf} | `{bf:.4f}` | ${ip:,.0f} | ${op:,.0f} | {od:.1%} |")

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(out_file).write_text("\n".join(lines), encoding="utf-8")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    from core.symbol_registry import TOP_10_SYMBOLS, ALL_SYMBOLS

    p = argparse.ArgumentParser(description="Ev05 — 40 Battlefields Multi-Asset")
    p.add_argument("--symbols", default="TOP10",
                   help="Comma-separated symbols, or TOP10, or ALL")
    p.add_argument("--data",    default="data")
    p.add_argument("--trials",  type=int, default=2000)
    p.add_argument("--workers", type=int, default=40)
    p.add_argument("--spread",  type=float, default=0.00015)
    p.add_argument("--out",     default="logs/An_Latest_Report.md")
    p.add_argument("--results", default="data/ev05_results")
    args = p.parse_args()

    if args.symbols == "TOP10":
        symbols = TOP_10_SYMBOLS
    elif args.symbols == "ALL":
        symbols = ALL_SYMBOLS
    else:
        symbols = [s.strip() for s in args.symbols.split(",")]

    n_total = len(symbols) * len(TF_ORDER) * args.trials
    print(f"✅ Đã rẽ nhánh feature/ev05_multi_asset an toàn.")
    print(f"   Server Xeon đã lên nòng {n_total:,} kịch bản! Khởi động!")

    run_ev05_queue(
        symbols=symbols, data_dir=args.data,
        n_trials=args.trials, n_workers=args.workers,
        spread=args.spread, results_dir=args.results,
        report_out=args.out,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
