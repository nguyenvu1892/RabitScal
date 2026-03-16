#!/usr/bin/env python3
"""
pipeline_ev04.py — Ev04: FVG Dynamic Trade Management
======================================================
Kiến trúc: Split Ticket 2-Leg dựa trên Opposing FVG cùng TF.
  Leg A (50%): TP = Opposing FVG gần nhất (đối diện chiều vào lệnh)
  Leg B (50%): Sau khi Leg A hit → SL chuyển BE + spread, TP = Next FVG tiếp theo

OOS Gate: Sau IS training → tự động validate OOS 30%.
  DD_oos > 25% → fitness = -abs(fitness)  (loại thẳng)
  Net_oos < 0  → fitness *= 0.5           (phạt 50%)

Changes vs Ev03:
  - Backtest: run_backtest_ev04() thay run_backtest_tf()
  - M1: sl_range=[1.5, 4.0], threshold_range=[0.75, 0.99]
  - Fitness: tích hợp OOS validation trong mỗi trial
"""
from __future__ import annotations

import json
import logging
import math
import multiprocessing
import multiprocessing.shared_memory as shm_mod
import random
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    format="[%(asctime)s UTC] - [%(levelname)-8s] - [EV04] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger("EV04")

# ─── Constants ────────────────────────────────────────────────────────────────
N_FEATURES      = 73
IS_RATIO        = 0.70
BATCH_SIZE      = 50
INITIAL_CAPITAL = 20_000.0
RISK_PER_TRADE  = 0.01
MAX_DD_LIMIT    = 0.28
OOS_DD_GATE     = 0.25     # OOS DD > 25% → loại

# ─── TF Configs ───────────────────────────────────────────────────────────────
TF_CONFIGS = {
    "M15": {
        "timeframe":       "M15",
        "study_name":      "rabitscal_ev04_m15_EURUSDm",
        "db_path":         "data/optuna_ev04_m15_EURUSDm.db",
        "n_trials":        2000,
        "n_workers":       30,
        "n_startup":       80,
        "min_trades_is":   200,
        "threshold_range": (0.05, 0.80),
        "sl_range":        (2.0, 4.0),
        "rr_fallback_range": (0.5, 3.0),  # Leg B fallback RR (khi không có Next FVG)
        "slip_range":      (0.0005, 0.003),
        "cooldown_range":  (1, 5),
        "fitness_mode":    "dynamic_fvg",
    },
    "M5": {
        "timeframe":       "M5",
        "study_name":      "rabitscal_ev04_m5_EURUSDm",
        "db_path":         "data/optuna_ev04_m5_EURUSDm.db",
        "n_trials":        2000,
        "n_workers":       50,
        "n_startup":       80,
        "min_trades_is":   300,
        "threshold_range": (0.05, 0.80),
        "sl_range":        (1.5, 3.5),
        "rr_fallback_range": (0.5, 3.0),
        "slip_range":      (0.0005, 0.005),
        "cooldown_range":  (1, 8),
        "fitness_mode":    "dynamic_fvg",
    },
    "M1": {
        "timeframe":       "M1",
        "study_name":      "rabitscal_ev04_m1_sniper_EURUSDm",
        "db_path":         "data/optuna_ev04_m1_sniper_EURUSDm.db",
        "n_trials":        2000,
        "n_workers":       20,
        "n_startup":       120,
        "min_trades_is":   100,
        # 🔧 FIX từ Ev03: sl_range từ [0.5,1.5] → [1.5,4.0]
        "threshold_range": (0.75, 0.99),   # Sniper: chặt hơn
        "sl_range":        (1.5, 4.0),     # ← FIX: không còn 0.5 nữa
        "rr_fallback_range": (1.5, 5.0),
        "slip_range":      (0.001, 0.01),
        "cooldown_range":  (5, 20),         # 5-20 M1 bars = 5-20 phút
        "fitness_mode":    "sniper_fvg",
    },
}

FROZEN_FEATURES = [0, 3, 5, 17, 23, 24, 39, 41, 42, 44, 48, 50, 52, 53, 54, 55, 57, 58, 59, 68]
ACTIVE_FEATURES = [i for i in range(N_FEATURES) if i not in FROZEN_FEATURES]


# ─── Split Ticket Backtest ────────────────────────────────────────────────────

def run_backtest_ev04(
    features: np.ndarray,
    raw_ohlcv: np.ndarray,
    fvg_zones: dict,
    weights: np.ndarray,
    threshold: float,
    sl_mult: float,
    rr_fallback: float,       # Leg B fallback RR khi không có Next FVG
    slippage_pct: float,
    cooldown: int,
    spread_cost: float = 0.00015,
    max_dd: float = MAX_DD_LIMIT,
    initial_capital: float = INITIAL_CAPITAL,
    risk_pct: float = RISK_PER_TRADE,
    fitness_mode: str = "dynamic_fvg",
) -> dict | None:
    """
    Split Ticket backtest với Opposing FVG targets.

    Mỗi tín hiệu vào lệnh:
      Leg A (50% lot): TP = Opposing FVG mid, SL = entry ± sl_mult×ATR
      Leg B (50% lot): SL ban đầu = sl_price, sau khi Leg A hit → SL = BE + spread
                       TP = Next FVG mid (cùng TF)

    Tracking 2 legs riêng biệt (không nest nhau về mặt cooldown).
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

    # FVG zones
    opp_bear = fvg_zones["opp_bear_mid"]   # opposing for LONG
    next_bear = fvg_zones["next_bear_mid"]  # next target for LONG leg B
    opp_bull  = fvg_zones["opp_bull_mid"]   # opposing for SHORT
    next_bull  = fvg_zones["next_bull_mid"]  # next target for SHORT leg B

    pnls = []
    equity = initial_capital
    peak   = initial_capital
    max_dd_hit = 0.0

    # State for Leg A
    in_a = False
    d_a = 0; ep_a = sl_a = tp_a = lot_a = 0.0

    # State for Leg B
    in_b   = False
    d_b = 0; ep_b = sl_b = tp_b = lot_b = 0.0
    leg_a_hit = False   # flag: Leg A đã hit → Leg B chuyển BE

    cooldown_left = 0

    for i in range(N):
        hi = float(h[i]); lo = float(l[i]); op = float(o[i])
        sl_i   = float(sl_d[i])
        slip_i = float(slip[i])

        # ── Process Leg A ──
        if in_a:
            hit_sl_a = (d_a ==  1 and lo <= sl_a) or (d_a == -1 and hi >= sl_a)
            hit_tp_a = (d_a ==  1 and hi >= tp_a) or (d_a == -1 and lo <= tp_a)

            if hit_tp_a:
                pnl = (tp_a - ep_a) * d_a * lot_a - spread_cost * lot_a - slip_i * lot_a
                pnls.append(pnl); equity += pnl
                in_a = False; leg_a_hit = True
                # Leg B: chuyển SL về BE
                if in_b:
                    be_sl = ep_b + slip_i * d_b   # BE + spread phía đi ngược
                    sl_b  = be_sl
            elif hit_sl_a:
                pnl = (sl_a - ep_a) * d_a * lot_a - spread_cost * lot_a - slip_i * lot_a
                pnls.append(pnl); equity += pnl
                in_a = False
                # Leg A bị stop → Leg B vẫn giữ SL gốc

            if equity > peak: peak = equity
            if peak > initial_capital:
                dd = (peak - equity) / peak
                if dd > max_dd_hit: max_dd_hit = dd

        # ── Process Leg B ──
        if in_b:
            hit_sl_b = (d_b ==  1 and lo <= sl_b) or (d_b == -1 and hi >= sl_b)
            hit_tp_b = (d_b ==  1 and hi >= tp_b) or (d_b == -1 and lo <= tp_b)

            if hit_tp_b:
                pnl = (tp_b - ep_b) * d_b * lot_b - spread_cost * lot_b - slip_i * lot_b
                pnls.append(pnl); equity += pnl
                in_b = False; leg_a_hit = False
                cooldown_left = cooldown
            elif hit_sl_b:
                pnl = (sl_b - ep_b) * d_b * lot_b - spread_cost * lot_b - slip_i * lot_b
                pnls.append(pnl); equity += pnl
                in_b = False; leg_a_hit = False
                cooldown_left = cooldown

            if equity > peak: peak = equity
            if peak > initial_capital:
                dd = (peak - equity) / peak
                if dd > max_dd_hit: max_dd_hit = dd

        # ── New Entry (only if both legs free) ──
        if not in_a and not in_b:
            if cooldown_left > 0:
                cooldown_left -= 1
                continue

            sl_dist = sl_i + slip_i
            lot_total = (equity * risk_pct) / max(sl_dist, 1e-10)
            lot_total = max(lot_total, 1e-10)
            lot_half  = lot_total * 0.5

            opp_b_i  = float(opp_bear[i])
            next_b_i = float(next_bear[i])
            opp_bu_i = float(opp_bull[i])
            next_bu_i = float(next_bull[i])

            if ls[i]:
                ep_entry = op + slip_i
                sl_price = ep_entry - sl_i

                tp_a_price  = opp_b_i   if opp_b_i  > ep_entry else ep_entry + sl_i * rr_fallback
                tp_b_price  = next_b_i  if next_b_i > tp_a_price else tp_a_price * 1.005

                in_a = True; d_a = 1
                ep_a = ep_entry; sl_a = sl_price; tp_a = tp_a_price; lot_a = lot_half

                in_b = True; d_b = 1
                ep_b = ep_entry; sl_b = sl_price; tp_b = tp_b_price; lot_b = lot_half
                leg_a_hit = False

            elif ss[i]:
                ep_entry = op - slip_i
                sl_price = ep_entry + sl_i

                tp_a_price  = opp_bu_i  if opp_bu_i  < ep_entry else ep_entry - sl_i * rr_fallback
                tp_b_price  = next_bu_i if next_bu_i < tp_a_price else tp_a_price * 0.995

                in_a = True; d_a = -1
                ep_a = ep_entry; sl_a = sl_price; tp_a = tp_a_price; lot_a = lot_half

                in_b = True; d_b = -1
                ep_b = ep_entry; sl_b = sl_price; tp_b = tp_b_price; lot_b = lot_half
                leg_a_hit = False

    n = len(pnls)
    if n == 0:
        return None

    arr  = np.array(pnls, dtype="float64")
    wins = arr[arr > 0]; loss = arr[arr <= 0]
    gp   = float(wins.sum()) if len(wins) else 0.0
    gl   = float(loss.sum()) if len(loss) else 0.0
    wr   = len(wins) / n
    pf   = min(abs(gp / gl), 999.0) if gl != 0 else (999.0 if gp > 0 else 0.0)
    net_ev     = (gp + gl) / n
    net_profit = gp + gl
    dd_ratio   = min(max_dd_hit, 1.0)

    freq_bonus = math.log10(max(10, n))
    dd_penalty = 1.0 - (dd_ratio / MAX_DD_LIMIT)

    if fitness_mode == "sniper_fvg":
        ev_bonus = math.log10(max(1.0, abs(net_ev) * 1000))
        fitness  = (net_profit / initial_capital) * ev_bonus * dd_penalty
    else:
        fitness = (net_profit / initial_capital) * freq_bonus * dd_penalty

    return {
        "wr": wr, "pf": pf, "n": n,
        "net_ev": net_ev, "net_profit": net_profit,
        "max_dd": max_dd_hit, "gp": gp, "gl": abs(gl),
        "fitness": fitness,
    }


# ─── Shared Memory Worker (subprocess) ───────────────────────────────────────

def _ev04_worker(tid, params, cfg,
                 fn, fs, fd, rn, rs, rd, mn, ms, md,
                 zn_names, zn_shapes, zn_dtypes,
                 spread):
    """Subprocess worker: load feats + zones từ SHM, chạy IS + OOS backtest."""
    time.sleep(random.uniform(0, 2.0))
    try:
        def att(name, shape, dtype):
            s = shm_mod.SharedMemory(name=name)
            return np.ndarray(shape, dtype=np.dtype(dtype), buffer=s.buf), s

        feats,  _s1 = att(fn, fs, fd)
        raw,    _s2 = att(rn, rs, rd)
        is_msk, _s3 = att(mn, ms, md)

        # Load FVG zones từ SHM
        zones = {}
        zone_shms = []
        for key, zname, zshape, zdtype in zip(
            ["opp_bear_mid", "next_bear_mid", "opp_bull_mid", "next_bull_mid",
             "opp_bear_top", "opp_bear_bot", "opp_bull_top", "opp_bull_bot",
             "opp_bear_dist_atr", "opp_bull_dist_atr"],
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
        fm  = cfg["fitness_mode"]

        # IS backtest
        is_mask_arr = is_msk.copy()
        feats_is = feats[is_mask_arr]
        raw_is   = raw[is_mask_arr]
        zones_is = {k: v[is_mask_arr] for k, v in zones.items()}

        r_is = run_backtest_ev04(
            feats_is, raw_is, zones_is, w, th, sl, rfb, sp, cd, spread,
            fitness_mode=fm,
        )

        # OOS backtest (cuối 30%)
        oos_mask = ~is_mask_arr
        feats_oos = feats[oos_mask]
        raw_oos   = raw[oos_mask]
        zones_oos = {k: v[oos_mask] for k, v in zones.items()}

        r_oos = run_backtest_ev04(
            feats_oos, raw_oos, zones_oos, w, th, sl, rfb, sp, cd, spread,
            fitness_mode=fm,
        )

        for s in (_s1, _s2, _s3): s.close()
        for zs in zone_shms: zs.close()

        # ── Gate checks ──
        if r_is is None:
            return {"tid": tid, "status": "pruned"}
        if r_is["n"] < cfg["min_trades_is"]:
            return {"tid": tid, "status": "pruned_trades", "n": r_is["n"]}
        if r_is["max_dd"] > MAX_DD_LIMIT:
            return {"tid": tid, "status": "pruned_dd", "dd": r_is["max_dd"]}

        fitness = r_is["fitness"]

        # OOS Gate
        oos_ok = True
        oos_dd = r_oos["max_dd"] if r_oos else 1.0
        oos_pf = r_oos["net_profit"] if r_oos else -1.0
        if r_oos and oos_dd > OOS_DD_GATE:
            fitness = -abs(fitness)     # loại thẳng tay
            oos_ok = False
        elif r_oos and oos_pf < 0:
            fitness *= 0.5              # phạt nhẹ

        return {
            "tid": tid, "status": "ok",
            "wr": r_is["wr"], "pf": r_is["pf"], "n_is": r_is["n"],
            "net_ev": r_is["net_ev"], "net_profit": r_is["net_profit"],
            "max_dd": r_is["max_dd"],
            "oos_dd": oos_dd, "oos_profit": oos_pf, "oos_ok": oos_ok,
            "fitness": fitness,
        }

    except Exception as e:
        return {"tid": tid, "status": "error", "error": str(e)}


# ─── Training ─────────────────────────────────────────────────────────────────

def train_tf_ev04(cfg: dict, symbol: str, data_dir: str,
                  n_trials: int | None, n_workers: int | None,
                  spread: float) -> optuna.Study:
    from tqdm import tqdm
    from core.feature_engine import build_feature_matrix, _nearest_opposing_fvg_zones

    tf_name   = cfg["timeframe"]
    n_trials  = n_trials  or cfg["n_trials"]
    n_workers = n_workers or cfg["n_workers"]
    db_path   = cfg["db_path"]

    logger.info(f"[{tf_name}] Loading features + FVG zones for {symbol}...")
    feats, raw = build_feature_matrix(symbol, data_dir)

    # Tính FVG zones từ raw OHLCV M5
    o5 = raw[:, 1]; h5 = raw[:, 2]; l5 = raw[:, 3]; c5 = raw[:, 4]
    pc5 = np.roll(c5, 1); pc5[0] = c5[0]
    tr5 = np.maximum(h5 - l5, np.maximum(np.abs(h5 - pc5), np.abs(l5 - pc5)))
    atr5 = np.convolve(tr5, np.ones(14) / 14, mode="same"); atr5[:13] = tr5[:13]
    atr5 = atr5.astype("float32")

    logger.info(f"[{tf_name}] Computing FVG zones...")
    zones = _nearest_opposing_fvg_zones(o5, h5, l5, c5, atr5, lookback=100)
    logger.info(f"[{tf_name}] FVG zones ready: {list(zones.keys())}")

    N = len(feats)
    is_cut   = int(N * IS_RATIO)
    is_mask  = np.zeros(N, dtype=bool); is_mask[:is_cut] = True

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

    # Alloc FVG zones in SHM
    zone_keys = ["opp_bear_mid", "next_bear_mid", "opp_bull_mid", "next_bull_mid",
                 "opp_bear_top", "opp_bear_bot", "opp_bull_top", "opp_bull_bot",
                 "opp_bear_dist_atr", "opp_bull_dist_atr"]
    zn_names, zn_shapes, zn_dtypes = [], [], []
    for k in zone_keys:
        zname, zshape, zdtype = alloc(zones[k])
        zn_names.append(zname); zn_shapes.append(zshape); zn_dtypes.append(zdtype)

    # Delete old DB
    p_db = Path(db_path)
    if p_db.exists():
        p_db.unlink()
        logger.info(f"[{tf_name}] Deleted old DB.")

    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{db_path}",
        engine_kwargs={"connect_args": {"timeout": 60}},
    )
    study = optuna.create_study(
        study_name=cfg["study_name"], direction="maximize",
        storage=storage, load_if_exists=False,
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=cfg["n_startup"], multivariate=True, group=True,
            constant_liar=True, warn_independent_sampling=False,
            seed=(42 + hash(tf_name) % 1000)),
    )
    logger.info(f"[{tf_name}] Study created. Fresh Ev04 exploration.")

    t0 = time.time()
    done = 0; best_v = -1e9; pruned_c = 0; completed_c = 0
    pbar = tqdm(total=n_trials, desc=f"Phase4 🔥 {tf_name}", unit="trial",
                dynamic_ncols=True,
                position={"M15": 0, "M5": 1, "M1": 2}.get(tf_name, 0))

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

                    th  = t_trial.suggest_float("threshold",    *cfg["threshold_range"])
                    sl  = t_trial.suggest_float("sl_mult",      *cfg["sl_range"])
                    rfb = t_trial.suggest_float("rr_fallback",  *cfg["rr_fallback_range"], log=True)
                    sp  = t_trial.suggest_float("slippage_pct", *cfg["slip_range"], log=True)
                    cd  = t_trial.suggest_int("cooldown",       *cfg["cooldown_range"])

                    params = {
                        "weights": w.tolist(), "threshold": th,
                        "sl_mult": sl, "rr_fallback": rfb,
                        "slippage_pct": sp, "cooldown": cd,
                    }
                    asked.append((t_trial, params))

                futures = {
                    exe.submit(_ev04_worker, t_trial.number, p, cfg,
                               fn, fs, fd, rn, rs, rd, mn, ms, md,
                               zn_names, zn_shapes, zn_dtypes, spread): t_trial
                    for t_trial, p in asked
                }

                for fut in as_completed(futures):
                    trial = futures[fut]
                    res   = fut.result()

                    if res["status"] == "ok":
                        trial.set_user_attr("winrate",     res["wr"])
                        trial.set_user_attr("pf",          res["pf"])
                        trial.set_user_attr("n_is",        res["n_is"])
                        trial.set_user_attr("net_ev",      res["net_ev"])
                        trial.set_user_attr("net_profit",  res["net_profit"])
                        trial.set_user_attr("max_dd",      res["max_dd"])
                        trial.set_user_attr("oos_dd",      res.get("oos_dd", -1))
                        trial.set_user_attr("oos_profit",  res.get("oos_profit", 0))
                        trial.set_user_attr("oos_ok",      int(res.get("oos_ok", True)))
                        study.tell(trial, res["fitness"])
                        completed_c += 1
                        if res["fitness"] > best_v: best_v = res["fitness"]
                    else:
                        study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                        pruned_c += 1

                    done += 1
                    pbar.update(1)
                    if done % 100 == 0:
                        pbar.set_postfix(
                            best=f"{best_v:.4f}" if best_v > -1e9 else "N/A",
                            cpl=completed_c, prune=pruned_c,
                        )
    finally:
        pbar.close()
        for s in shms:
            try: s.close(); s.unlink()
            except: pass

    elapsed = time.time() - t0
    logger.info(
        f"[{tf_name}] DONE | {done} trials | {elapsed:.0f}s | {done/elapsed:.2f}t/s | "
        f"completed={completed_c} | pruned={pruned_c} | best={best_v:.4f}"
    )
    return study


def _run_tf_process_ev04(cfg_key: str, symbol: str, data_dir: str,
                         n_trials, n_workers, spread, out) -> None:
    cfg   = TF_CONFIGS[cfg_key]
    study = train_tf_ev04(cfg, symbol, data_dir, n_trials, n_workers, spread)
    _write_tf_report_ev04(study, cfg, out)


def train_mtf_ev04(symbol: str, data_dir: str, n_trials,
                   n_workers, spread: float, out: str) -> None:
    logger.info("=" * 65)
    logger.info("  PHASE 4 — EV04 FVG DYNAMIC MTF (Split Ticket + OOS Gate)")
    logger.info(f"  Symbol: {symbol} | 3 TFs: M15 / M5 / M1-Sniper-Fix")
    logger.info(f"  OOS Gate: DD_oos > {OOS_DD_GATE:.0%} → fitness = -abs(fitness)")
    logger.info(f"  Capital: ${INITIAL_CAPITAL:,.0f} | Risk: {RISK_PER_TRADE:.0%}/trade")
    logger.info("=" * 65)

    processes = []
    for cfg_key in ["M15", "M5", "M1"]:
        p = multiprocessing.Process(
            target=_run_tf_process_ev04,
            args=(cfg_key, symbol, data_dir, n_trials, n_workers, spread, out),
            name=f"EV04-{cfg_key}", daemon=False,
        )
        p.start()
        logger.info(f"[Dispatcher] Spawned {cfg_key} PID={p.pid}")
        processes.append(p)

    for p in processes:
        p.join()
        logger.info(f"[Dispatcher] {p.name} done (exitcode={p.exitcode})")

    logger.info("✅ EV04 MTF PARALLEL HOÀN TẤT")


# ─── Report ───────────────────────────────────────────────────────────────────

def _write_tf_report_ev04(study: optuna.Study, cfg: dict, out_file: str) -> None:
    tf_name   = cfg["timeframe"]
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    pruned    = [t for t in study.trials if t.state.name == "PRUNED"]
    total     = len(study.trials)
    now       = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if not completed:
        logger.error(f"[{tf_name}] No completed trials.")
        return

    top3 = sorted(completed, key=lambda t: t.value, reverse=True)[:3]
    # Only show trials with positive fitness (real champions)
    oos_ok_trials = [t for t in completed if t.user_attrs.get("oos_ok", 1)]

    L = []
    a = lambda *lines: L.extend(lines)
    icon = {"M15": "🟩", "M5": "🟦", "M1": "🔫"}.get(tf_name, "⬛")

    a(f"\n---\n",
      f"## {icon} TF: {tf_name} | Ev04 FVG Dynamic",
      f"",
      f"| Metrics | Value |", f"|---|---|",
      f"| Total Trials | {total:,} |",
      f"| ✅ Completed | **{len(completed):,}** ({len(completed)/max(total,1):.1%}) |",
      f"| 🎯 OOS Pass  | **{len(oos_ok_trials):,}** ({len(oos_ok_trials)/max(len(completed),1):.1%} của completed) |",
      f"| ✂️ Pruned    | {len(pruned):,} |",
      f"| Best Fitness | **{study.best_value:.6f}** |",
      f"| Best Trial # | #{study.best_trial.number} |",
      f"| OOS DD Gate  | {OOS_DD_GATE:.0%} |",
      f"",
      f"### 🥇 Top 3 — {tf_name} (IS+OOS)",
      f"")

    for rank, t in enumerate(top3, 1):
        p     = t.params
        wr    = t.user_attrs.get("winrate", 0)
        pf_v  = t.user_attrs.get("pf", 0)
        n_is  = t.user_attrs.get("n_is", 0)
        nev   = t.user_attrs.get("net_ev", 0)
        dd    = t.user_attrs.get("max_dd", 0)
        oos_d = t.user_attrs.get("oos_dd", -1)
        oos_p = t.user_attrs.get("oos_profit", 0)
        oos_ok_flag = "✅" if t.user_attrs.get("oos_ok", 1) else "🔴"
        sl    = p.get("sl_mult", 0)
        rfb   = p.get("rr_fallback", 0)
        thr   = p.get("threshold", 0)
        cd    = p.get("cooldown", 0)

        a(f"**{'🥇' if rank==1 else '🥈' if rank==2 else '🥉'} #{rank}**"
          f" Trial #{t.number} | Fitness: `{t.value:.4f}` {oos_ok_flag}",
          f"",
          f"| KPI | IS | OOS |", f"|---|---|---|",
          f"| N Trades | {n_is:,} | — |",
          f"| Net Profit | ${t.user_attrs.get('net_profit',0):,.0f} | ${oos_p:,.0f} |",
          f"| Win Rate | {wr:.1%} | — |",
          f"| PF | {pf_v:.3f} | — |",
          f"| Max DD | {dd:.1%} | {oos_d:.1%} |",
          f"| sl_mult | {sl:.3f} | rr_fallback={rfb:.3f} |",
          f"| threshold | {thr:.4f} | cooldown={cd} bars |",
          f"")

    out_path = Path(out_file)
    mode = "a" if out_path.exists() else "w"
    with open(out_path, mode, encoding="utf-8") as f:
        f.write("\n".join(L))
    logger.info(f"[{tf_name}] Report appended → {out_file}")


def write_ev04_header(out_file: str, symbol: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = f"""# 🏆 Ev04 MTF FVG Dynamic Report — {symbol} $20,000
> **Generated:** {now} | **Branch:** `Ev04_FVG_Dynamic`
> **Architecture:** Split Ticket (Leg A→Opposing FVG | Leg B→Next FVG + BE) | OOS Gate DD>{OOS_DD_GATE:.0%}
> **M1 Fix:** sl_range=[1.5, 4.0] | threshold=[0.75, 0.99]

"""
    Path(out_file).write_text(header, encoding="utf-8")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Phase 4 Ev04 FVG Dynamic MTF")
    p.add_argument("--symbol",  default="EURUSDm")
    p.add_argument("--data",    default="data")
    p.add_argument("--trials",  type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--spread",  type=float, default=0.00015)
    p.add_argument("--out",     default="logs/An_Latest_Report.md")
    args = p.parse_args()

    write_ev04_header(args.out, args.symbol)
    train_mtf_ev04(args.symbol, args.data, args.trials, args.workers, args.spread, args.out)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
