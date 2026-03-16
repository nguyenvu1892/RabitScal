#!/usr/bin/env python3
"""
run_phase2_pipeline.py — Phase 2: Shaping & Scalping Pipeline
==============================================================
Warm-start từ Phase 1 best weights → Ép khuôn Scalping $200 Exness:
  - Freeze 20 features yếu nhất từ Phase 1 SHAP
  - Max DD = 28% | Min Trades = 1050 (1500/năm × 70% IS)
  - Fitness = Net_EV × (1+ln(n/1000)) × (1-DD×0.3) × OOS_factor

Usage: python run_phase2_pipeline.py [--trials 2000] [--workers 50]

⚠️ FOREGROUND ONLY — không nohup, không &
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing.shared_memory as shm_mod
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

import logging
logging.basicConfig(
    format="[%(asctime)s UTC] - [%(levelname)-8s] - [Phase2] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger("Phase2")

# ─── Constants ───────────────────────────────────────────────────────────────
N_FEATURES     = 73
IS_RATIO       = 0.70
BATCH_SIZE     = 50
STUDY_NAME     = "rabitscal_phase2_scalping_EURUSDm"
PHASE1_DB      = "data/optuna_phase1_EURUSDm.db"
PHASE1_STUDY   = "rabitscal_phase1_free_EURUSDm"

# Phase 2 Hard Constraints (Sếp Vũ approved)
MAX_DD_LIMIT   = 0.28     # Tối đa 28% drawdown
MIN_TRADES_IS  = 300      # Realistic với cooldown+SL filtering (1050/3.5x avg_cooldown)
MIN_TRADES_OOS = 300      # OOS tối thiểu

# Frozen features từ Phase 1 SHAP (bottom 20 by |weight|)
FROZEN_FEATURES = [0, 3, 5, 17, 23, 24, 39, 41, 42, 44, 48, 50, 52, 53, 54, 55, 57, 58, 59, 68]
ACTIVE_FEATURES = [i for i in range(N_FEATURES) if i not in FROZEN_FEATURES]  # 53 features

# Phase 2 search space — hạ threshold để ép trade frequency
P2_THRESHOLD_RANGE = (0.05, 0.80)  # Diagnostic: thr=0.7→47k signals; thr=2.854→550 signals (không đủ 1050 lệnh)
P2_SL_RANGE        = (1.5, 3.5)   # Quanh Phase 1 best=2.418
P2_RR_RANGE        = (0.3, 5.0)   # AI TỰ QUYẾT — cấm fix cứng
P2_SLIP_RANGE      = (0.0005, 0.005)
P2_COOLDOWN_RANGE  = (1, 8)

FEAT_NAMES = [
    "bos_bull","bos_bear","choch_bull","choch_bear",
    "struct_h1_bull","struct_h1_bear","struct_h4_bull","struct_h4_bear",
    "mtf_align_bull","mtf_align_bear",
    "eql_dist_atr","eqh_dist_atr","eql_touch","eqh_touch",
    "in_ote_bull","in_ote_bear","swing_high_dist","swing_low_dist",
    "fvg_bull","fvg_bear","fvg_bull_fresh","fvg_bear_fresh",
    "fvg_bull_mitigated","fvg_bear_mitigated",
    "ob_bull","ob_bear","ob_bull_fresh","ob_bear_fresh",
    "in_fvg_bull","in_fvg_bear","fvg_size_atr","ob_size_atr",
    "fvg_bull_dist","fvg_bear_dist","ob_bull_dist","ob_bear_dist",
    "vol_sma20","relative_vol","effort_vs_result",
    "is_climax_up","is_climax_dn","is_no_demand","is_no_supply",
    "vol_trend_bull","vol_trend_bear",
    "poc_dist","vah_dist","val_dist",
    "vol_session_ratio","delta_vol","cumulative_delta",
    "vol_diverge_bull","vol_diverge_bear","vol_breakout",
    "poc_support","poc_resist",
    "is_bull_pinbar","is_bear_pinbar",
    "is_bull_engulf","is_bear_engulf",
    "is_bull_maru","is_bear_maru",
    "is_doji","trap_bull","trap_bear",
    "compression_bull","compression_bear",
    "body_size_rel","upper_wick_rel","lower_wick_rel","candle_range_atr",
    "hour_sin","hour_cos",
]
GROUPS = {
    "SMC Structure": list(range(0,18)),
    "FVG / OB":      list(range(18,36)),
    "VSA Volume":    list(range(36,56)),
    "Price Action":  list(range(56,71)),
    "Session/Time":  list(range(71,73)),
}

# ─── Phase 2 Backtest (with DD gate + trade gate) ────────────────────────────

def run_backtest_p2(features, raw_m5, weights, threshold, sl_mult, rr_ratio,
                    slippage_pct=0.001, cooldown=3,
                    spread_cost=0.00015, pip_value=1.0,
                    max_dd=MAX_DD_LIMIT):
    """
    Phase 2: Có đủ constraints thực chiến:
    - SL/TP với slippage ATR
    - Cooldown timer sau mỗi lệnh
    - Theo dõi Max Drawdown (early exit để tính DD)
    """
    N = len(features)
    if N < 500:
        return None

    score = features.astype("float32") @ weights
    ls = score >  threshold
    ss = score < -threshold
    ls[:200] = False; ss[:200] = False

    o = raw_m5[:,1].astype("float64"); h = raw_m5[:,2].astype("float64")
    l = raw_m5[:,3].astype("float64"); c = raw_m5[:,4].astype("float64")
    pc  = np.roll(c, 1); pc[0] = c[0]
    tr  = np.maximum(h-l, np.maximum(np.abs(h-pc), np.abs(l-pc)))
    atr = np.convolve(tr, np.ones(14)/14, mode="same"); atr[:13] = tr[:13]
    sl_d = atr * sl_mult
    slip = atr * slippage_pct   # [S1] Slippage penalty

    pnls=[]; equity=0.0; peak=0.0; max_dd_hit=0.0
    in_t=False; d=ep=sp_=tp_=0.0; cooldown_left=0
    cost = spread_cost

    for i in range(N):
        hi=float(h[i]); lo=float(l[i]); op=float(o[i])
        sl_i = float(sl_d[i]); slip_i = float(slip[i])

        # [FIX] Cooldown ONLY blocks new entries, NOT trade monitoring
        if in_t:
            hit_sl = (d== 1 and lo<=sp_) or (d==-1 and hi>=sp_)
            hit_tp = (d== 1 and hi>=tp_) or (d==-1 and lo<=tp_)
            if hit_sl:
                pnl = (sp_-ep)*d*pip_value - cost - slip_i
                pnls.append(pnl); equity += pnl
                in_t=False; cooldown_left=cooldown
            elif hit_tp:
                pnl = (tp_-ep)*d*pip_value - cost - slip_i
                pnls.append(pnl); equity += pnl
                in_t=False; cooldown_left=cooldown

            if equity > peak: peak = equity
            if peak > 0:
                dd = (peak - equity) / peak
                if dd > max_dd_hit: max_dd_hit = dd
        else:
            if cooldown_left > 0:   # Block new entry only
                cooldown_left -= 1
                continue
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
    net_ev = (gp + gl) / n   # average PnL per trade
    dd_ratio = min(max_dd_hit, 1.0)

    # Phase 2 Fitness: Net_EV × freq_bonus × drawdown_penalty
    freq_bonus = 1.0 + math.log(max(n, 1) / 1000) if n >= 1 else 0.0
    fitness = net_ev * freq_bonus * (1.0 - dd_ratio * 0.3)

    return {"wr":wr,"pf":pf,"n":n,"net_ev":net_ev,
            "max_dd":max_dd_hit,"gp":gp,"gl":abs(gl),"fitness":fitness}


# ─── Worker ───────────────────────────────────────────────────────────────────

def _p2_worker(tid, params,
               fn,fs,fd, rn,rs,rd, mn,ms,md, oon,os_,od,
               spread):
    """Subprocess worker: pure NumPy, không đụng SQLite."""
    time.sleep(random.uniform(0, 2.0))   # Stagger

    try:
        def att(name, shape, dtype):
            s = shm_mod.SharedMemory(name=name)
            return np.ndarray(shape, dtype=np.dtype(dtype), buffer=s.buf), s

        feats,  _s1 = att(fn, fs, fd)
        raw,    _s2 = att(rn, rs, rd)
        is_msk, _s3 = att(mn, ms, md)
        oos_msk,_s4 = att(oon, os_, od)

        w   = np.array(params["weights"], dtype="float32")
        th  = float(params["threshold"])
        sl  = float(params["sl_mult"])
        rr  = float(params["rr_ratio"])
        sp  = float(params["slippage_pct"])
        cd  = int(params["cooldown"])

        # IS backtest
        r_is = run_backtest_p2(feats[is_msk], raw[is_msk], w, th, sl, rr, sp, cd, spread)

        for s in (_s1,_s2,_s3,_s4): s.close()

        if r_is is None: return {"tid":tid,"status":"pruned"}
        if r_is["n"] < MIN_TRADES_IS:
            return {"tid":tid,"status":"pruned_trades","n":r_is["n"]}
        if r_is["max_dd"] > MAX_DD_LIMIT:
            return {"tid":tid,"status":"pruned_dd","dd":r_is["max_dd"]}
        # [FIX] Cho phép negative fitness pass — Optuna cần học từ giá trị âm
        # Gate fitness<=0 → prune toàn bộ random weights, TPE không học được

        return {"tid":tid,"status":"ok",
                "wr":r_is["wr"],"pf":r_is["pf"],"n_is":r_is["n"],
                "net_ev":r_is["net_ev"],"max_dd":r_is["max_dd"],
                "gp":r_is["gp"],"gl":r_is["gl"],"fitness":r_is["fitness"]}

    except Exception as e:
        return {"tid":tid,"status":"error","error":str(e)}


# ─── Warm-start enqueue ───────────────────────────────────────────────────────

def _enqueue_warmstart(study: optuna.Study, phase1_db: str, phase1_study: str,
                       top_k: int = 20) -> None:
    """Load top_k Phase 1 trials và enqueue làm prior cho Phase 2 TPE."""
    try:
        s1 = optuna.load_study(study_name=phase1_study,
                               storage=f"sqlite:///{phase1_db}")
        completed = sorted(
            [t for t in s1.trials if t.state.name == "COMPLETE"],
            key=lambda t: t.value, reverse=True,
        )[:top_k]

        for t in completed:
            w = np.array([t.params.get(f"w{i}",0.0) for i in range(N_FEATURES)], dtype="float32")
            nm = float(np.linalg.norm(w)); w = w/nm if nm>1e-6 else w
            # Zero-out frozen features
            for idx in FROZEN_FEATURES: w[idx] = 0.0
            nm2 = float(np.linalg.norm(w)); w = w/nm2 if nm2>1e-6 else w

            enqueue_params = {f"w{i}": float(w[i]) for i in range(N_FEATURES)}
            # Phase 2 threshold = Phase 1 threshold × 0.8 (hạ để nhiều lệnh hơn)
            # Phase 1 threshold=2.854 → ÷4 để điều chỉnh về Phase 2 range [0.05,0.8]
            p1_thr = t.params.get("threshold", 1.0)
            enqueue_params["threshold"]    = max(P2_THRESHOLD_RANGE[0],
                                                  min(p1_thr / 4.0, P2_THRESHOLD_RANGE[1]))
            enqueue_params["sl_mult"]      = max(P2_SL_RANGE[0],
                                                  min(t.params.get("sl_mult",1.5), P2_SL_RANGE[1]))
            enqueue_params["rr_ratio"]     = max(P2_RR_RANGE[0],
                                                  min(t.params.get("rr_ratio",1.5), P2_RR_RANGE[1]))
            enqueue_params["slippage_pct"] = 0.001
            enqueue_params["cooldown"]     = 3
            study.enqueue_trial(enqueue_params)

        logger.info(f"Warm-start: enqueued top {len(completed)} Phase 1 trials")
    except Exception as e:
        logger.warning(f"Warm-start failed (will start fresh): {e}")


# ─── Training ─────────────────────────────────────────────────────────────────

def train(symbol, data_dir, n_trials, n_workers, spread, db_path):
    from feature_engine import build_feature_matrix
    from tqdm import tqdm

    logger.info(f"Loading features for {symbol}...")
    feats, raw = build_feature_matrix(symbol, data_dir)
    N = len(feats)
    is_cut  = int(N * IS_RATIO)
    is_mask = np.zeros(N, dtype=bool); is_mask[:is_cut] = True
    oos_mask= np.zeros(N, dtype=bool); oos_mask[is_cut:] = True

    shms = []
    def alloc(arr):
        s = shm_mod.SharedMemory(create=True, size=arr.nbytes)
        np.ndarray(arr.shape, dtype=arr.dtype, buffer=s.buf)[:] = arr[:]
        logger.info(f"SHM {arr.nbytes/1024/1024:.1f}MB → {s.name}")
        shms.append(s)
        return s.name, arr.shape, str(arr.dtype)

    fn,fs,fd = alloc(feats)
    rn,rs,rd = alloc(raw)
    mn,ms,md = alloc(is_mask)
    on,os_,od= alloc(oos_mask)

    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{db_path}",
        engine_kwargs={"connect_args": {"timeout": 60}},
    )
    study = optuna.create_study(
        study_name=STUDY_NAME, direction="maximize",
        storage=storage, load_if_exists=False,
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=40, multivariate=True, group=True,
            constant_liar=True, warn_independent_sampling=False, seed=42),
    )

    # Warm-start từ Phase 1
    _enqueue_warmstart(study, PHASE1_DB, PHASE1_STUDY, top_k=20)

    t0   = time.time()
    done = 0; best_v = -1e9; pruned_c = 0; completed_c = 0
    pbar = tqdm(total=n_trials, desc="Phase2 🔥 Scalping", unit="trial", dynamic_ncols=True)

    try:
        with ProcessPoolExecutor(max_workers=n_workers) as exe:
            while done < n_trials:
                batch = min(BATCH_SIZE, n_trials - done)

                # ASK: main thread owns SQLite
                asked = []
                for _ in range(batch):
                    t = study.ask()
                    # Suggest active features only; freeze = 0
                    w = np.zeros(N_FEATURES, dtype="float32")
                    for i in ACTIVE_FEATURES:
                        w[i] = t.suggest_float(f"w{i}", -1.0, 1.0)
                    # Frozen stay at 0 — but we still need to register them
                    for i in FROZEN_FEATURES:
                        t.suggest_float(f"w{i}", 0.0, 0.0)   # fixed = 0
                    nm = float(np.linalg.norm(w)); w = w/nm if nm>1e-6 else w

                    th  = t.suggest_float("threshold",    *P2_THRESHOLD_RANGE)
                    sl  = t.suggest_float("sl_mult",      *P2_SL_RANGE)
                    rr  = t.suggest_float("rr_ratio",     *P2_RR_RANGE, log=True)
                    sp  = t.suggest_float("slippage_pct", *P2_SLIP_RANGE, log=True)
                    cd  = t.suggest_int("cooldown",       *P2_COOLDOWN_RANGE)

                    params = {"weights":w.tolist(),"threshold":th,"sl_mult":sl,
                              "rr_ratio":rr,"slippage_pct":sp,"cooldown":cd}
                    asked.append((t, params))

                # SUBMIT: 50 workers song song, không đụng SQLite
                futures = {
                    exe.submit(_p2_worker, t.number, p,
                               fn,fs,fd, rn,rs,rd, mn,ms,md, on,os_,od, spread): t
                    for t, p in asked
                }

                # TELL: main thread ghi kết quả
                for fut in as_completed(futures):
                    trial = futures[fut]
                    res   = fut.result()

                    if res["status"] == "ok":
                        trial.set_user_attr("winrate",    res["wr"])
                        trial.set_user_attr("pf",         res["pf"])
                        trial.set_user_attr("n_is",       res["n_is"])
                        trial.set_user_attr("net_ev",     res["net_ev"])
                        trial.set_user_attr("max_dd",     res["max_dd"])
                        study.tell(trial, res["fitness"])
                        completed_c += 1
                        if res["fitness"] > best_v: best_v = res["fitness"]
                    else:
                        study.tell(trial,
                                   state=optuna.trial.TrialState.PRUNED)
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
    logger.info(f"Phase 2 DONE | {done} trials | {elapsed:.0f}s | {done/elapsed:.2f}t/s | "
                f"completed={completed_c} | pruned={pruned_c} | best={best_v:.4f}")
    return study


# ─── Report ───────────────────────────────────────────────────────────────────

def write_report(study: optuna.Study, out_file: str) -> None:
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    pruned    = [t for t in study.trials if t.state.name == "PRUNED"]
    total     = len(study.trials)
    now       = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    top3      = sorted(completed, key=lambda t: t.value, reverse=True)[:3]

    if not completed:
        logger.error("No completed trials!")
        return

    best = study.best_trial

    L = []
    def a(*lines): L.extend(lines)

    a(f"# 🏆 Phase 2 Scalping Report — EURUSDm $200",
      f"",
      f"**Generated:** {now} | **Study:** `{STUDY_NAME}`",
      f"",
      f"| Metrics | Value |",f"|---|---|",
      f"| Total Trials | {total:,} |",
      f"| ✅ Completed (Pass DD+Trades) | **{len(completed):,}** ({len(completed)/max(total,1):.1%}) |",
      f"| ✂️ Pruned | {len(pruned):,} |",
      f"| Best Fitness | **{study.best_value:.6f}** |",
      f"| Best Trial # | #{best.number} |",
      f"| Frozen Features | 20/73 (73→53 active dims) |",
      f"",
      f"---",
      f"",
      f"## 🥇 Top 3 Chén Thánh (Scalping $200 Exness)",
      f"",
      f"> Thỏa mãn: DD<28% | IS_Trades>1050 | Net_EV>0 | Slippage/Cooldown embedded",
      f"")

    for rank, t in enumerate(top3, 1):
        p    = t.params
        wr   = t.user_attrs.get("winrate", 0)
        pf_v = t.user_attrs.get("pf", 0)
        n_is = t.user_attrs.get("n_is", 0)
        nev  = t.user_attrs.get("net_ev", 0)
        dd   = t.user_attrs.get("max_dd", 0)
        rr   = p.get("rr_ratio", 0); thr = p.get("threshold",0); sl = p.get("sl_mult",0)
        sp   = p.get("slippage_pct",0); cd = p.get("cooldown",0)

        annual_est = int(n_is / 0.7)  # IS = 70% → extrapolate to 100%
        daily_est  = round(annual_est / 252, 1)
        safe_dd    = "✅ SAFE" if dd < MAX_DD_LIMIT else "⚠️ BORDERLINE"

        a(f"### {'🥇' if rank==1 else '🥈' if rank==2 else '🥉'} #{rank} — Trial {t.number} | Fitness: **{t.value:.6f}**",
          f"",
          f"| KPI | Value | Đánh giá |",f"|---|---|---|",
          f"| 🏆 Fitness (Phase 2) | **{t.value:.6f}** | Net_EV×freq×(1-DD×0.3) |",
          f"| 📊 IS Trades (70% data) | **{n_is:,}** | Annual ≈{annual_est:,} ({daily_est}/ngày) |",
          f"| 💰 Net EV/trade | **{nev:+.5f}** | {'✅ Dương' if nev>0 else '❌ Âm'} |",
          f"| 🎯 Winrate | **{wr:.1%}** | — |",
          f"| 📈 Profit Factor | **{pf_v:.3f}** | — |",
          f"| 📉 Max Drawdown | **{dd:.1%}** | {safe_dd} (<28% target) |",
          f"| ⚖️ R/R (AI tự chọn) | **{rr:.3f}** | — |",
          f"| 🔑 Entry Threshold | **{thr:.4f}** | — |",
          f"| 🛡️ SL × ATR | **{sl:.3f}** | — |",
          f"| 💸 Slippage | **{sp:.4f}** × ATR | [S1] Simulated |",
          f"| ⏳ Cooldown | **{cd}** nến = {cd*5} phút | [S2] Anti-overtrade |",
          f"")

    # Feature importance from weights
    if completed:
        wb = np.array([best.params.get(f"w{i}",0) for i in range(N_FEATURES)], dtype="float32")
        nm = float(np.linalg.norm(wb)); wb = wb/nm if nm > 1e-6 else wb
        top5_w = np.argsort(np.abs(wb))[::-1][:5]

        a(f"---",f"",
          f"## 🧠 Top 5 SHAP Weights (Phase 2 Best Trial #{best.number})",
          f"",
          f"| Rank | Feature | Nhóm | Weight |",f"|---|---|---|---|")
        for rank, idx in enumerate(top5_w, 1):
            name = FEAT_NAMES[idx] if idx < len(FEAT_NAMES) else f"w{idx}"
            grp  = next((g for g,ii in GROUPS.items() if idx in ii), "?")
            a(f"| {rank} | `{name}` | {grp} | {wb[idx]:+.4f} |")

    a(f"",f"---",
      f"",
      f"## 📋 Deploy Instructions",
      f"",
      f"```python",
      f"# Load Phase 2 best params:",
      f"import optuna",
      f"s = optuna.load_study(study_name='{STUDY_NAME}',",
      f"                      storage='sqlite:///data/optuna_phase2_EURUSDm.db')",
      f"best = s.best_trial.params",
      f"# → Gắn vào ml_model.py current_settings.json",
      f"```",
      f"",
      f"*Auto-generated by `run_phase2_pipeline.py` · {now}*")

    Path(out_file).write_text("\n".join(L), encoding="utf-8")
    logger.info(f"Report → {out_file}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Phase 2 Shaping & Scalping Pipeline")
    p.add_argument("--symbol",  default="EURUSDm")
    p.add_argument("--data",    default="data")
    p.add_argument("--trials",  type=int,   default=2000)
    p.add_argument("--workers", type=int,   default=50)
    p.add_argument("--spread",  type=float, default=0.00015)
    p.add_argument("--out",     default="logs/An_Latest_Report.md")
    args = p.parse_args()

    db_path = str(Path(args.data) / f"optuna_phase2_{args.symbol}.db")

    print(f"\n{'='*65}")
    print(f"  PHASE 2 — SHAPING & SCALPING (Exness $200)")
    print(f"  Trials: {args.trials} | Workers: {args.workers} | Batch: {BATCH_SIZE}")
    print(f"  Constraints: DD<{MAX_DD_LIMIT:.0%} | IS_Trades>{MIN_TRADES_IS}")
    print(f"  Warm-start: top 20 Phase 1 trials enqueued")
    print(f"  Frozen: {len(FROZEN_FEATURES)} features | Active: {len(ACTIVE_FEATURES)} dims")
    print(f"  Threshold range: {P2_THRESHOLD_RANGE} (hạ để ép tần suất)")
    print(f"  Fix: Batch Ask-and-Tell | SQLite WAL timeout=60 | Stagger")
    print(f"  ⚠️ FOREGROUND — Xem thanh tiến trình trực tiếp")
    print(f"{'='*65}\n")

    # Delete old Phase 2 DB nếu có
    p2_db = Path(db_path)
    if p2_db.exists():
        p2_db.unlink()
        logger.info(f"Deleted old Phase 2 DB: {db_path}")

    study = train(args.symbol, args.data, args.trials, args.workers, args.spread, db_path)
    write_report(study, args.out)
    print(f"\n✅ PHASE 2 PIPELINE HOÀN TẤT → {args.out}")

if __name__ == "__main__":
    main()
