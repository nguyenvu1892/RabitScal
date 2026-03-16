#!/usr/bin/env python3
"""
pipeline_mtf.py — Ev03: Multi-Timeframe Parallel Optimization
==============================================================
Kiến trúc: 3 Optuna study độc lập (M15 / M5 / M1-Sniper) chạy song song.
Mỗi TF có Search Space riêng. Fitness M1 thưởng thêm EV/trade (Sniper quality).

Risk Gate (live trading enforcement):
  - Tối đa 3 lệnh đồng thời trên thị trường.
  - Mỗi TF chỉ được mở tối đa 1 lệnh tại 1 thời điểm.
  - Được enforce tại lớp live dispatcher (main.py), KHÔNG phải tại đây.
  - Trong Optuna: mỗi study tối ưu độc lập, không simulate multi-TF concurrency.

Usage:
    python quant_main.py train --phase 3 --mode mtf --trials 2000 --workers 50

⚠️ FOREGROUND ONLY — không nohup, không &
"""
from __future__ import annotations

import argparse
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
    format="[%(asctime)s UTC] - [%(levelname)-8s] - [MTF] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger("MTF")

# ─── Shared Constants ────────────────────────────────────────────────────────
N_FEATURES      = 73          # main branch: 73 features (Ev02 vol_pa_signal on Ev02 branch only)
IS_RATIO        = 0.70
BATCH_SIZE      = 50
INITIAL_CAPITAL = 20_000.0
RISK_PER_TRADE  = 0.01        # 1% per trade — dynamic lot
MAX_DD_LIMIT    = 0.28

# ─── Per-TF Configurations ───────────────────────────────────────────────────

TF_CONFIGS = {
    "M15": {
        "timeframe":       "M15",
        "study_name":      "rabitscal_ev03_m15_EURUSDm",
        "db_path":         "data/optuna_ev03_m15_EURUSDm.db",
        "data_tf":         "m15",           # key trong MTFData dict
        "n_trials":        2000,
        "n_workers":       30,
        "n_startup":       80,
        "min_trades_is":   200,             # M15: ít lệnh hơn vì TF lớn

        # Search Space — Standard
        "threshold_range": (0.05, 0.80),
        "sl_range":        (2.0,  4.0),    # SL rộng hơn — M15 bar lớn
        "rr_range":        (0.5,  5.0),
        "slip_range":      (0.0005, 0.003),
        "cooldown_range":  (1, 5),          # In M15 bars

        # Fitness: standard Ev02
        "fitness_mode":    "standard",
    },
    "M5": {
        "timeframe":       "M5",
        "study_name":      "rabitscal_ev03_m5_EURUSDm",
        "db_path":         "data/optuna_ev03_m5_EURUSDm.db",
        "data_tf":         "m5",
        "n_trials":        2000,
        "n_workers":       50,
        "n_startup":       80,
        "min_trades_is":   300,

        # Search Space — Standard (giống Ev02)
        "threshold_range": (0.05, 0.80),
        "sl_range":        (1.5,  3.5),
        "rr_range":        (0.3,  5.0),
        "slip_range":      (0.0005, 0.005),
        "cooldown_range":  (1, 8),

        "fitness_mode":    "standard",
    },
    "M1": {
        "timeframe":       "M1",
        "study_name":      "rabitscal_ev03_m1_sniper_EURUSDm",
        "db_path":         "data/optuna_ev03_m1_sniper_EURUSDm.db",
        "data_tf":         "m1",
        "n_trials":        2000,
        "n_workers":       20,              # Ít workers hơn — M1 nhiều bars hơn
        "n_startup":       120,             # Cần nhiều random start hơn để explore
        "min_trades_is":   100,             # Sniper: ít lệnh hơn nhưng chất lượng cao

        # Search Space — STRICT Sniper Mode
        "threshold_range": (0.50, 0.99),   # Ngưỡng cao bắt buộc — cắt 80% nhiễu M1
        "sl_range":        (0.5,  1.5),    # SL hẹp hơn — sniper precise
        "rr_range":        (2.0,  8.0),    # RR cao hơn bù cho WR thấp
        "slip_range":      (0.001, 0.01),  # M1 slippage cao hơn thực tế
        "cooldown_range":  (10, 30),        # 10-30 M1 bars = 10-30 phút cooldown

        # Fitness: Sniper mode — thưởng thêm EV/trade
        "fitness_mode":    "sniper",
    },
}

# Frozen features từ Ev02 Phase 1 SHAP (bottom 20)
FROZEN_FEATURES = [0, 3, 5, 17, 23, 24, 39, 41, 42, 44, 48, 50, 52, 53, 54, 55, 57, 58, 59, 68]
ACTIVE_FEATURES = [i for i in range(N_FEATURES) if i not in FROZEN_FEATURES]


# ─── Backtest (Parameterized by TF Config) ───────────────────────────────────

def run_backtest_tf(
    features: np.ndarray,
    raw_ohlcv: np.ndarray,
    weights: np.ndarray,
    threshold: float,
    sl_mult: float,
    rr_ratio: float,
    slippage_pct: float,
    cooldown: int,
    spread_cost: float = 0.00015,
    max_dd: float = MAX_DD_LIMIT,
    initial_capital: float = INITIAL_CAPITAL,
    risk_pct: float = RISK_PER_TRADE,
    fitness_mode: str = "standard",
) -> dict | None:
    """
    Backtest chung cho tất cả TF. Parameterized theo TF config.
    Dynamic Lot: equity × risk_pct / SL_distance.
    """
    N = len(features)
    if N < 500:
        return None

    score = features.astype("float32") @ weights
    ls = score >  threshold
    ss = score < -threshold
    ls[:200] = False; ss[:200] = False

    o = raw_ohlcv[:,1].astype("float64"); h = raw_ohlcv[:,2].astype("float64")
    l = raw_ohlcv[:,3].astype("float64"); c = raw_ohlcv[:,4].astype("float64")
    pc  = np.roll(c, 1); pc[0] = c[0]
    tr  = np.maximum(h-l, np.maximum(np.abs(h-pc), np.abs(l-pc)))
    atr = np.convolve(tr, np.ones(14)/14, mode="same"); atr[:13] = tr[:13]
    sl_d = atr * sl_mult
    slip = atr * slippage_pct

    pnls=[]; equity=initial_capital; peak=initial_capital; max_dd_hit=0.0
    in_t=False; d=ep=sp_=tp_=lot_size=0.0; cooldown_left=0

    for i in range(N):
        hi=float(h[i]); lo=float(l[i]); op=float(o[i])

        if in_t:
            sl_i = float(sl_d[i]); slip_i = float(slip[i])
            hit_sl = (d== 1 and lo<=sp_) or (d==-1 and hi>=sp_)
            hit_tp = (d== 1 and hi>=tp_) or (d==-1 and lo<=tp_)
            if hit_sl:
                pnl = (sp_-ep)*d*lot_size - spread_cost*lot_size - slip_i*lot_size
                pnls.append(pnl); equity += pnl
                in_t=False; cooldown_left=cooldown
            elif hit_tp:
                pnl = (tp_-ep)*d*lot_size - spread_cost*lot_size - slip_i*lot_size
                pnls.append(pnl); equity += pnl
                in_t=False; cooldown_left=cooldown

            if equity > peak: peak = equity
            if peak > initial_capital:
                dd = (peak - equity) / peak
                if dd > max_dd_hit: max_dd_hit = dd
        else:
            if cooldown_left > 0:
                cooldown_left -= 1
                continue
            sl_i = float(sl_d[i]); slip_i = float(slip[i])
            sl_dist = sl_i + slip_i
            lot_size = (equity * risk_pct) / max(sl_dist, 1e-10)
            lot_size = max(lot_size, 1e-10)
            if ls[i]:
                in_t=True; d=1
                ep = op + slip_i
                sp_ = ep - sl_i; tp_ = ep + sl_i * rr_ratio
            elif ss[i]:
                in_t=True; d=-1
                ep = op - slip_i
                sp_ = ep + sl_i; tp_ = ep - sl_i * rr_ratio

    n = len(pnls)
    if n == 0:
        return None

    arr  = np.array(pnls, dtype="float64")
    wins = arr[arr>0]; loss = arr[arr<=0]
    gp   = float(wins.sum()) if len(wins) else 0.0
    gl   = float(loss.sum()) if len(loss) else 0.0
    wr   = len(wins)/n
    pf   = min(abs(gp/gl), 999.0) if gl!=0 else (999.0 if gp>0 else 0.0)
    net_ev     = (gp + gl) / n
    net_profit = gp + gl
    dd_ratio   = min(max_dd_hit, 1.0)

    freq_bonus = math.log10(max(10, n))
    dd_penalty = 1.0 - (dd_ratio / MAX_DD_LIMIT)

    if fitness_mode == "sniper":
        # M1 Sniper: thưởng thêm EV/trade quality — ép tìm setup chất lượng cao
        # log10(|ev| × 1000) normalizes EV vào scale dương
        ev_bonus = math.log10(max(1.0, abs(net_ev) * 1000))
        fitness  = (net_profit / initial_capital) * ev_bonus * dd_penalty
    else:
        # Standard Ev02: freq bonus
        fitness = (net_profit / initial_capital) * freq_bonus * dd_penalty

    return {
        "wr": wr, "pf": pf, "n": n,
        "net_ev": net_ev, "net_profit": net_profit,
        "max_dd": max_dd_hit, "gp": gp, "gl": abs(gl),
        "fitness": fitness,
    }


# ─── Worker (subprocess — no SQLite) ─────────────────────────────────────────

def _tf_worker(tid, params, cfg,
               fn, fs, fd, rn, rs, rd, mn, ms, md,
               spread):
    """Subprocess worker cho một TF. Pure NumPy, không đụng SQLite."""
    time.sleep(random.uniform(0, 2.0))

    try:
        def att(name, shape, dtype):
            s = shm_mod.SharedMemory(name=name)
            return np.ndarray(shape, dtype=np.dtype(dtype), buffer=s.buf), s

        feats,   _s1 = att(fn, fs, fd)
        raw,     _s2 = att(rn, rs, rd)
        is_msk,  _s3 = att(mn, ms, md)

        w   = np.array(params["weights"], dtype="float32")
        th  = float(params["threshold"])
        sl  = float(params["sl_mult"])
        rr  = float(params["rr_ratio"])
        sp  = float(params["slippage_pct"])
        cd  = int(params["cooldown"])

        r_is = run_backtest_tf(
            feats[is_msk], raw[is_msk], w, th, sl, rr, sp, cd, spread,
            fitness_mode=cfg["fitness_mode"],
        )

        for s in (_s1, _s2, _s3): s.close()

        if r_is is None:
            return {"tid": tid, "status": "pruned"}
        if r_is["n"] < cfg["min_trades_is"]:
            return {"tid": tid, "status": "pruned_trades", "n": r_is["n"]}
        if r_is["max_dd"] > MAX_DD_LIMIT:
            return {"tid": tid, "status": "pruned_dd", "dd": r_is["max_dd"]}

        return {
            "tid": tid, "status": "ok",
            "wr": r_is["wr"], "pf": r_is["pf"], "n_is": r_is["n"],
            "net_ev": r_is["net_ev"], "net_profit": r_is["net_profit"],
            "max_dd": r_is["max_dd"], "gp": r_is["gp"], "gl": r_is["gl"],
            "fitness": r_is["fitness"],
        }

    except Exception as e:
        return {"tid": tid, "status": "error", "error": str(e)}


# ─── Single-TF Training ───────────────────────────────────────────────────────

def train_tf(cfg: dict, symbol: str, data_dir: str,
             n_trials: int | None, n_workers: int | None, spread: float) -> optuna.Study:
    """
    Chạy Optuna study cho 1 TF. Được gọi từ spawn process.
    """
    from tqdm import tqdm
    from core.feature_engine import build_feature_matrix

    tf_name   = cfg["timeframe"]
    n_trials  = n_trials  or cfg["n_trials"]
    n_workers = n_workers or cfg["n_workers"]
    db_path   = cfg["db_path"]

    logger.info(f"[{tf_name}] Loading features for {symbol}...")
    feats, raw = build_feature_matrix(symbol, data_dir)

    N = len(feats)
    is_cut   = int(N * IS_RATIO)
    is_mask  = np.zeros(N, dtype=bool); is_mask[:is_cut] = True

    shms = []
    def alloc(arr):
        s = shm_mod.SharedMemory(create=True, size=arr.nbytes)
        np.ndarray(arr.shape, dtype=arr.dtype, buffer=s.buf)[:] = arr[:]
        shms.append(s)
        return s.name, arr.shape, str(arr.dtype)

    fn, fs, fd = alloc(feats)
    rn, rs, rd = alloc(raw)
    mn, ms, md = alloc(is_mask)

    # Delete old DB — fresh start
    p2_db = Path(db_path)
    if p2_db.exists():
        p2_db.unlink()
        logger.info(f"[{tf_name}] Deleted old DB: {db_path}")

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
    logger.info(f"[{tf_name}] Study created. NO warm-start — fresh {tf_name} exploration.")

    t0 = time.time()
    done = 0; best_v = -1e9; pruned_c = 0; completed_c = 0
    pbar = tqdm(total=n_trials,
                desc=f"Phase3 🔥 {tf_name}",
                unit="trial", dynamic_ncols=True,
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
                    rr  = t_trial.suggest_float("rr_ratio",     *cfg["rr_range"], log=True)
                    sp  = t_trial.suggest_float("slippage_pct", *cfg["slip_range"], log=True)
                    cd  = t_trial.suggest_int("cooldown",       *cfg["cooldown_range"])

                    params = {
                        "weights": w.tolist(), "threshold": th,
                        "sl_mult": sl, "rr_ratio": rr,
                        "slippage_pct": sp, "cooldown": cd,
                    }
                    asked.append((t_trial, params))

                futures = {
                    exe.submit(_tf_worker, t_trial.number, p, cfg,
                               fn, fs, fd, rn, rs, rd, mn, ms, md, spread): t_trial
                    for t_trial, p in asked
                }

                for fut in as_completed(futures):
                    trial = futures[fut]
                    res   = fut.result()

                    if res["status"] == "ok":
                        trial.set_user_attr("winrate",    res["wr"])
                        trial.set_user_attr("pf",         res["pf"])
                        trial.set_user_attr("n_is",       res["n_is"])
                        trial.set_user_attr("net_ev",     res["net_ev"])
                        trial.set_user_attr("net_profit", res.get("net_profit", 0.0))
                        trial.set_user_attr("max_dd",     res["max_dd"])
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


# ─── MTF Dispatcher (spawn 3 processes in parallel) ──────────────────────────

def _run_tf_process(cfg_key: str, symbol: str, data_dir: str,
                    n_trials: int | None, n_workers: int | None,
                    spread: float, out: str) -> None:
    """Target function for each TF process."""
    cfg   = TF_CONFIGS[cfg_key]
    study = train_tf(cfg, symbol, data_dir, n_trials, n_workers, spread)
    _write_tf_report(study, cfg, out)


def train_mtf(symbol: str, data_dir: str, n_trials: int | None,
              n_workers: int | None, spread: float, out: str) -> None:
    """
    MTF Dispatcher: spawn 3 independent processes (M15, M5, M1) in parallel.

    ─── Risk Gate Architecture ───────────────────────────────────────────────
    Live Trading (main.py / risk_manager.py):
      - MAX_CONCURRENT_POSITIONS = 3
      - MAX_PER_TF = 1 (M15 ≤1, M5 ≤1, M1 ≤1)
      - Implemented via multiprocessing.Manager dict in live dispatcher
    Optimization (here): each study runs independently, no cross-TF constraints.
    ──────────────────────────────────────────────────────────────────────────
    """
    logger.info("=" * 65)
    logger.info("  PHASE 3 — MTF PARALLEL OPTIMIZATION (Ev03)")
    logger.info(f"  Symbol: {symbol} | 3 TFs: M15 / M5 / M1-Sniper")
    logger.info(f"  Risk Gate: Max 3 concurrent positions, 1 per TF (live)")
    logger.info(f"  Capital: ${INITIAL_CAPITAL:,.0f} | Risk: {RISK_PER_TRADE:.0%}/trade")
    logger.info("=" * 65)

    processes = []
    for cfg_key in ["M15", "M5", "M1"]:
        p = multiprocessing.Process(
            target=_run_tf_process,
            args=(cfg_key, symbol, data_dir, n_trials, n_workers, spread, out),
            name=f"MTF-{cfg_key}",
            daemon=False,
        )
        p.start()
        logger.info(f"[Dispatcher] Spawned {cfg_key} process PID={p.pid}")
        processes.append(p)

    # Wait for all 3 to finish
    for p in processes:
        p.join()
        logger.info(f"[Dispatcher] {p.name} finished (exitcode={p.exitcode})")

    logger.info("✅ MTF PARALLEL OPTIMIZATION HOÀN TẤT")
    logger.info(f"   Report: {out}")


# ─── Report ───────────────────────────────────────────────────────────────────

def _write_tf_report(study: optuna.Study, cfg: dict, out_file: str) -> None:
    """Ghi report cho 1 TF vào file (append mode với header). """
    tf_name   = cfg["timeframe"]
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    pruned    = [t for t in study.trials if t.state.name == "PRUNED"]
    total     = len(study.trials)
    now       = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    top3      = sorted(completed, key=lambda t: t.value, reverse=True)[:3]

    if not completed:
        logger.error(f"[{tf_name}] No completed trials — no report written.")
        return

    best = study.best_trial
    L    = []
    a    = lambda *lines: L.extend(lines)

    icon = {"M15": "🟩", "M5": "🟦", "M1": "🔴"}.get(tf_name, "⬛")
    fitness_desc = ("Net_Profit/CAPITAL × log10(n) × (1-DD/0.28)"
                    if cfg["fitness_mode"] == "standard"
                    else "Net_Profit/CAPITAL × log10(|EV|×1000) × (1-DD/0.28) [Sniper]")

    a(f"\n---\n",
      f"## {icon} TF: {tf_name} | Study: `{cfg['study_name']}`",
      f"",
      f"| Metrics | Value |", f"|---|---|",
      f"| Total Trials | {total:,} |",
      f"| ✅ Completed | **{len(completed):,}** ({len(completed)/max(total,1):.1%}) |",
      f"| ✂️ Pruned | {len(pruned):,} |",
      f"| Best Fitness | **{study.best_value:.6f}** |",
      f"| Best Trial # | #{best.number} |",
      f"| Fitness Formula | `{fitness_desc}` |",
      f"| Search Space | threshold={cfg['threshold_range']} sl={cfg['sl_range']} rr={cfg['rr_range']} |",
      f"| Min IS Trades | {cfg['min_trades_is']} |",
      f"",
      f"### 🥇 Top 3 — {tf_name}",
      f"")

    for rank, t in enumerate(top3, 1):
        p    = t.params
        wr   = t.user_attrs.get("winrate", 0)
        pf_v = t.user_attrs.get("pf", 0)
        n_is = t.user_attrs.get("n_is", 0)
        nev  = t.user_attrs.get("net_ev", 0)
        dd   = t.user_attrs.get("max_dd", 0)
        rr   = p.get("rr_ratio", 0); thr = p.get("threshold", 0)
        sl   = p.get("sl_mult", 0); cd = p.get("cooldown", 0)
        safe = "✅ SAFE" if dd < MAX_DD_LIMIT else "⚠️ OVER"

        a(f"**{'🥇' if rank==1 else '🥈' if rank==2 else '🥉'} #{rank}** "
          f"Trial #{t.number} | Fitness: `{t.value:.4f}`",
          f"",
          f"| KPI | Value |", f"|---|---|",
          f"| Fitness | **{t.value:.6f}** |",
          f"| IS Trades | **{n_is:,}** |",
          f"| Net EV/trade | **{nev:+.4f}** |",
          f"| Winrate | **{wr:.1%}** |",
          f"| Profit Factor | **{pf_v:.3f}** |",
          f"| Max DD | **{dd:.1%}** | {safe} |",
          f"| R/R | **{rr:.3f}** |",
          f"| Threshold | **{thr:.4f}** |",
          f"| SL×ATR | **{sl:.3f}** |",
          f"| Cooldown | **{cd} bars** |",
          f"")

    out_path = Path(out_file)
    mode = "a" if out_path.exists() else "w"
    with open(out_path, mode, encoding="utf-8") as f:
        f.write("\n".join(L))
    logger.info(f"[{tf_name}] Report appended → {out_file}")


def write_mtf_header(out_file: str, symbol: str) -> None:
    """Ghi header cho report trước khi 3 TF processes chạy."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = f"""# 🏆 Ev03 MTF Parallel Report — {symbol} $20,000
> **Generated:** {now} | **Branch:** `Ev03_MTF_Parallel`  
> **Architecture:** 3 Independent Optuna Studies (M15 / M5 / M1-Sniper)  
> **Risk Gate (Live):** Max 3 concurrent positions | 1 per TF

"""
    Path(out_file).write_text(header, encoding="utf-8")


# ─── Main (standalone) ────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Phase 3 MTF Parallel Optimization")
    p.add_argument("--symbol",  default="EURUSDm")
    p.add_argument("--data",    default="data")
    p.add_argument("--trials",  type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--spread",  type=float, default=0.00015)
    p.add_argument("--out",     default="logs/An_Latest_Report.md")
    args = p.parse_args()

    write_mtf_header(args.out, args.symbol)
    train_mtf(args.symbol, args.data, args.trials, args.workers, args.spread, args.out)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
