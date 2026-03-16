"""
ml_engine_v2.py — Composite Score Training Engine V1.5
=======================================================
Module: RabitScal Phase 2 — AI Training (Giai Đoạn 2)
Author: Antigravity
Date:   2026-03-07

Triết lý: Feature Matrix 73 chiều × Learned Weights → Total_Score → Entry.
AI (Optuna) tự tìm bộ trọng số tối ưu. Không có If-Else quy tắc cứng.

Walk-Forward Anti-Overfit:
    70% In-Sample (Train) → 30% Out-of-Sample (Test)
    OOS Net_EV <= 0 → PRUNE (overfit kịch độc)
"""

from __future__ import annotations

import time
import logging
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from typing import Optional

import numpy as np
import optuna

# Import từ ml_model.py cùng thư mục
from ml_model import (
    BacktestResult,
    OptimizationResult,
    SharedNumpyArray,
    _ASSET_CLASS_DEFAULT,
    DATA_DIR,
    _build_logger,
)

logger = _build_logger("MLEngineV2")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_FEATURES_V2 = 73   # Phải khớp với feature_engine.py N_FEATURES
IS_RATIO      = 0.70  # 70% In-Sample, 30% Out-of-Sample

# ---------------------------------------------------------------------------
# Scalping Profile — Exness $200 Account
# ---------------------------------------------------------------------------
# Triết lý: Bào lãi kép, 5-10 lệnh/ngày = 1500-3000/năm (250 ngày giao dịch)
# Chấp nhận DD cao hơn (25-30%) để đổi lấy tần suất cao liên tục.
# Cost per 0.01 lot EURUSD trên Exness Standard: ~0.00015 USD spread.
# ── Quản lý vốn $200, DD 30% (Phân tích Sếp Vũ Q2) ─────────────────────
# Max acceptable loss: $200 × 30% = $60
# Losing streak worst-case (WR=40%, p=lose 5 consecutive): (0.6)^5 ≈ 7.8%
# → 5 lệnh thua liên tiếp xác suất 7.8%. Mỗi lệnh risk tối đa $60/5 = $12
# → Position Size = $12 / (SL_pips × pip_value_per_0.01lot)
# EURUSD: 1 pip = $0.10/0.01lot → SL 15 pips → risk = 15×$0.10 = $1.5/lot
# → Max lots = $12 / $1.5 = 0.08 lot (tối đa 8 micro lots khi SL=15pips)
# Mặc định an toàn: 0.01 lot cố định trong backtest (1 micro lot)
SCALPING_CAPITAL      = 200.0    # USD
SCALPING_MAX_DD_PCT   = 0.28     # 28% — buffer dưới 30%
SCALPING_MIN_TRADES   = 1050     # 1500/năm × 0.70 IS split ≈ 1050 lệnh IS
SCALPING_MAX_TRADES   = 2100     # 3000/năm × 0.70 IS split
SCALPING_OOS_MIN      = 300      # OOS 30% must have ≥ 300 trades
SCALPING_LOT          = 0.01     # 1 micro lot mặc định
SCALPING_THRESH_RANGE = (0.05, 2.0)   # Nới thấp — bắt nhiều tín hiệu hơn


# ---------------------------------------------------------------------------
# Core Backtest — Composite Score Engine
# ---------------------------------------------------------------------------

def run_backtest_composite(
    features: np.ndarray,   # shape (N, 73)
    raw_m5:   np.ndarray,   # shape (N, 6)  [time, open, high, low, close, volume]
    weights:  np.ndarray,   # shape (73,)   learned weights in [-1, 1]
    threshold: float,       # Entry threshold: score > thresh → Long
    sl_mult:   float,       # ATR multiplier for SL
    rr_ratio:  float,       # TP = SL × rr_ratio
    *,
    commission_per_lot: float = 3.5,
    spread_cost:        float = 0.0,
    pip_value:          float = 1.0,
    lot_size:           float = 0.01,
    max_dd_limit:       float = 0.15,
    min_trades:         int   = 50,
    # ── Security Layer 1: Slippage Penalty ────────────────────────────
    # Phạt trượt giá: entry/exit giá thực = open ± slippage_pct × ATR
    # Mô phỏng thực tế hơn: Exness M5 có latency ≈ 0.5-2% ATR slippage
    slippage_pct: float = 0.001,   # 0.1% ATR mặc định (Optuna tự tìm)
    # ── Security Layer 2: Cooldown Timer ──────────────────────────────
    # Sau mỗi lệnh đóng, chờ N nến trước khi entry tiếp theo.
    # Tránh over-trade liên tiếp — AI tự tìm N tối ưu (Optuna param)
    cooldown_candles: int = 3,     # 3 nến ≈ 15 phút cool-down với M5
) -> BacktestResult:
    """
    Composite Score Backtest V1.5 — 3 Security Layers:
    [S1] Slippage: entry_price = open ± slippage_pct × ATR (hướng chống)
    [S2] Cooldown: skip `cooldown_candles` nến sau mỗi trade đóng
    [S3] TPE: đã fix warn_independent_sampling=False trong study (bên ngoài)
    """
    N = len(features)
    if N < 200 or len(raw_m5) < 200:
        raise optuna.exceptions.TrialPruned()

    # ── [1] Composite Score — PURE VECTORIZED ────────────────────────────────
    # (N, 73) @ (73,) = (N,): numpy BLAS dot ~1ms cho 150k×73
    total_score = features.astype(np.float32) @ weights   # shape (N,)

    # ── [2] Entry signals ────────────────────────────────────────────────────
    long_signal  = total_score >  threshold
    short_signal = total_score < -threshold
    long_signal[:200]  = False   # warmup
    short_signal[:200] = False

    # ── [3] ATR cho SL/TP ────────────────────────────────────────────────────
    opens  = raw_m5[:, 1].astype(np.float64)
    highs  = raw_m5[:, 2].astype(np.float64)
    lows   = raw_m5[:, 3].astype(np.float64)
    closes = raw_m5[:, 4].astype(np.float64)
    prev_c = np.roll(closes, 1); prev_c[0] = closes[0]
    tr = np.maximum(highs - lows, np.maximum(
        np.abs(highs - prev_c), np.abs(lows - prev_c)
    ))
    atr = np.convolve(tr, np.ones(14) / 14, mode='same')
    atr[:13] = tr[:13]
    sl_dist  = atr * sl_mult

    # ── [4] Sequential simulation ────────────────────────────────────────────
    cost_per_trade = commission_per_lot * lot_size + spread_cost
    pnl_list:    list[float] = []
    equity:      float = 0.0
    peak_eq:     float = 0.0
    max_dd:      float = 0.0
    in_trade:    bool  = False
    direction:   int   = 0
    entry_price: float = 0.0
    sl_price:    float = 0.0
    tp_price:    float = 0.0
    cooldown_left: int = 0    # Security Layer 2: cooldown counter

    for i in range(N):
        hi  = float(highs[i]); lo = float(lows[i]); op = float(opens[i])
        atr_i = float(atr[i])
        slip  = slippage_pct * atr_i    # Security Layer 1: slippage distance

        # Cooldown countdown
        if cooldown_left > 0:
            cooldown_left -= 1

        if in_trade:
            hit_sl = (direction ==  1 and lo <= sl_price) or \
                     (direction == -1 and hi >= sl_price)
            hit_tp = (direction ==  1 and hi >= tp_price) or \
                     (direction == -1 and lo <= tp_price)

            if hit_sl:
                # [S1] SL exit: worst-case — exit giá bị trượt thêm slip
                exit_price = sl_price - direction * slip
                exit_pnl   = (exit_price - entry_price) * direction
                pnl = exit_pnl * pip_value - cost_per_trade
                pnl_list.append(pnl)
                equity    += pnl
                in_trade   = False
                cooldown_left = cooldown_candles   # [S2] trigger cooldown
            elif hit_tp:
                # [S1] TP exit: bị trượt theo hướng TP (gọi là favorable slip)
                exit_price = tp_price - direction * slip   # cả TP cũng bị slip
                exit_pnl   = (exit_price - entry_price) * direction
                pnl = exit_pnl * pip_value - cost_per_trade
                pnl_list.append(pnl)
                equity    += pnl
                in_trade   = False
                cooldown_left = cooldown_candles   # [S2] trigger cooldown

            # Track drawdown với early-exit
            peak_eq = max(peak_eq, equity)
            if peak_eq > 0:
                dd_cur = (peak_eq - equity) / peak_eq
                max_dd = max(max_dd, dd_cur)
                if max_dd > max_dd_limit:
                    raise optuna.exceptions.TrialPruned()
        else:
            # [S2] Chỉ entry sau khi hết cooldown
            if cooldown_left == 0:
                if long_signal[i]:
                    in_trade    = True; direction  =  1
                    entry_price = op + slip   # [S1] entry bị slip hướng chống
                    sl_price    = entry_price - float(sl_dist[i])
                    tp_price    = entry_price + float(sl_dist[i]) * rr_ratio
                elif short_signal[i]:
                    in_trade    = True; direction  = -1
                    entry_price = op - slip   # [S1] entry bị slip hướng chống
                    sl_price    = entry_price + float(sl_dist[i])
                    tp_price    = entry_price - float(sl_dist[i]) * rr_ratio

    n_trades = len(pnl_list)
    if n_trades < max(min_trades, 1):
        raise optuna.exceptions.TrialPruned()

    # Quá nhiều lệnh → AI đang mở tất cả mọi lúc (noise entry) → prune
    # Ngưỡng trên: 3×SCALPING_MAX_TRADES để không quá chặt với IS data
    if n_trades > SCALPING_MAX_TRADES * 3:
        raise optuna.exceptions.TrialPruned()

    arr    = np.array(pnl_list, dtype=np.float64)
    wins   = arr[arr > 0]
    losses = arr[arr <= 0]
    gross_p = float(wins.sum())   if len(wins)   > 0 else 0.0
    gross_l = float(losses.sum()) if len(losses) > 0 else 0.0   # âm
    winrate = len(wins) / n_trades
    pf      = abs(gross_p / gross_l) if gross_l != 0 else float('inf')

    return BacktestResult(
        winrate=winrate, profit_factor=pf, max_drawdown=max_dd,
        trade_count=n_trades, gross_profit=gross_p, gross_loss=abs(gross_l),
        avg_win=float(wins.mean())    if len(wins)   > 0 else 0.0,
        avg_loss=float(losses.mean()) if len(losses) > 0 else 0.0,
    )


# ---------------------------------------------------------------------------
# Subprocess Worker — Pickle-safe, No SQLite
# ---------------------------------------------------------------------------

def _composite_worker(
    trial_params:  dict,
    feat_shm_name: str,
    feat_shape:    tuple,
    feat_dtype:    str,
    raw_shm_name:  str,
    raw_shape:     tuple,
    raw_dtype:     str,
    is_shm_name:   str,
    is_shape:      tuple,
    is_dtype:      str,
    max_dd_limit:  float,
    min_trades:    int,
    spread_cost:   float = 0.0,
    pip_value:     float = 1.0,
    slippage_pct:  float = 0.001,   # [S1]
    cooldown_candles: int = 3,      # [S2]
) -> dict:
    """Worker subprocess: attach shm, run IS+OOS backtests, return results."""
    try:
        feats  = SharedNumpyArray.attach(feat_shm_name, feat_shape, np.dtype(feat_dtype))
        raw    = SharedNumpyArray.attach(raw_shm_name,  raw_shape,  np.dtype(raw_dtype))
        is_msk = SharedNumpyArray.attach(is_shm_name,   is_shape,   np.dtype(is_dtype))

        weights   = np.array(trial_params["weights"],   dtype=np.float32)
        threshold = float(trial_params["threshold"])
        sl_mult   = float(trial_params["sl_mult"])
        rr_ratio  = float(trial_params["rr_ratio"])
        slippage_pct_v = float(trial_params.get("slippage_pct",  0.001))
        cooldown_v     =   int(trial_params.get("cooldown_candles", 3))

        # IN-SAMPLE (70%)
        res_is = run_backtest_composite(
            feats[is_msk], raw[is_msk], weights, threshold, sl_mult, rr_ratio,
            spread_cost=spread_cost, pip_value=pip_value,
            max_dd_limit=max_dd_limit, min_trades=min_trades,
            slippage_pct=slippage_pct_v, cooldown_candles=cooldown_v,
        )

        # OUT-OF-SAMPLE (30%) — Walk-Forward validation
        oos_mask  = ~is_msk
        n_oos_min = max(5, min_trades // 5)
        res_oos = run_backtest_composite(
            feats[oos_mask], raw[oos_mask], weights, threshold, sl_mult, rr_ratio,
            spread_cost=spread_cost, pip_value=pip_value,
            max_dd_limit=max_dd_limit * 1.2,
            min_trades=n_oos_min,
            slippage_pct=slippage_pct_v, cooldown_candles=cooldown_v,
        )

        def _r(res: BacktestResult) -> dict:
            return {
                "gross_profit": res.gross_profit, "gross_loss": res.gross_loss,
                "max_drawdown": res.max_drawdown, "trade_count": res.trade_count,
                "winrate": res.winrate,
            }

        return {"status": "ok", "is": _r(res_is), "oos": _r(res_oos)}

    except optuna.exceptions.TrialPruned:
        return {"status": "pruned"}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Objective Factory V2.0
# ---------------------------------------------------------------------------

def _build_objective_v2(
    feat_shm_name: str, feat_shape: tuple, feat_dtype: str,
    raw_shm_name:  str, raw_shape:  tuple, raw_dtype:  str,
    is_shm_name:   str, is_shape:   tuple, is_dtype:   str,
    max_dd_limit:  float,
    min_trades:    int,
    executor:      ProcessPoolExecutor,
    asset_cfg:     dict | None = None,
    n_features:    int = N_FEATURES_V2,
):
    """
    Optuna Objective Factory — V2.0 Composite Score + Walk-Forward.

    Search Space: 73 feature weights ∈ [-1,1] + threshold + sl_mult + rr_ratio
    Fitness: IS Net_EV × (1-DD_ratio) × OOS_consistency_bonus
    Anti-Overfit Gate: OOS Net_EV <= 0 → PRUNE ngay lập tức

    Tốc độ ước tính (Xeon 56 luồng):
        dot product: ~1ms | simulation: ~50ms | total: ~51ms/trial
        500 trials × 56 parallel = ~10 phút (500 trials tổng, không phải 500/luồng)
    """
    sp = asset_cfg or _ASSET_CLASS_DEFAULT

    def objective(trial: optuna.Trial) -> float:
        # ── Suggest 73 feature weights + 3 meta params ─────────────────────
        # Threshold range nới thấp [0.05, 2.0] cho scalping (bắt nhiều tín hiệu)
        weights = np.array([
            trial.suggest_float(f"w{i}", -1.0, 1.0)
            for i in range(n_features)
        ], dtype=np.float32)

        # L2 normalize — tránh trivial all-zero solution
        w_norm = float(np.linalg.norm(weights))
        if w_norm > 1e-6:
            weights = weights / w_norm

        threshold = trial.suggest_float(
            "threshold", *SCALPING_THRESH_RANGE
        )
        sl_mult      = trial.suggest_float("sl_mult",   0.3, 2.0)
        rr_ratio     = trial.suggest_float("rr_ratio",  0.3, 5.0, log=True)  # AI tự quyết
        # ── Security Params (Optuna tự tìm mức tối ưu) ────────────────────
        slippage_pct = trial.suggest_float("slippage_pct", 0.0005, 0.005)  # [S1] 0.05-0.5% ATR
        cooldown_c   = trial.suggest_int("cooldown_candles", 1, 10)        # [S2] 1-10 nến

        trial_params = {
            "weights":          weights.tolist(),
            "threshold":        threshold,
            "sl_mult":          sl_mult,
            "rr_ratio":         rr_ratio,
            "slippage_pct":     slippage_pct,
            "cooldown_candles": cooldown_c,
        }

        future = executor.submit(
            _composite_worker,
            trial_params,
            feat_shm_name, feat_shape, feat_dtype,
            raw_shm_name,  raw_shape,  raw_dtype,
            is_shm_name,   is_shape,   is_dtype,
            max_dd_limit, min_trades,
            sp.get("spread_cost", 0.0),
            sp.get("pip_value",   1.0),
        )

        res = future.result(timeout=180)

        if res["status"] in ("pruned", "error"):
            raise optuna.exceptions.TrialPruned()

        is_r  = res["is"];  oos_r = res["oos"]
        net_ev_is  = is_r["gross_profit"]  - is_r["gross_loss"]
        net_ev_oos = oos_r["gross_profit"] - oos_r["gross_loss"]
        dd_is      = is_r["max_drawdown"]
        n_is  = is_r["trade_count"]
        n_oos = oos_r["trade_count"]

        # ── Hard Survival Constraints (Scalping $200 profile) ───────────────
        if net_ev_is  <= 0:               raise optuna.exceptions.TrialPruned()  # Lỗ IS
        if n_is < SCALPING_MIN_TRADES:    raise optuna.exceptions.TrialPruned()  # Quá ít lệnh
        if net_ev_oos <= 0:               raise optuna.exceptions.TrialPruned()  # Overfit!
        if n_oos < SCALPING_OOS_MIN:      raise optuna.exceptions.TrialPruned()  # OOS thiếu lệnh
        if dd_is > SCALPING_MAX_DD_PCT:   raise optuna.exceptions.TrialPruned()  # DD vượt 28%

        # ── Aggressive Fitness: Thưởng mạnh tần suất (Sếp Vũ yêu cầu) ──────
        # Fitness = Net_EV × (1 + ln(n_trades/1000)) × (1 - DD_ratio×0.3) × OOS_factor
        # Giải thích từng nhân tử:
        #   (1 + ln(trades/1000)): bonus tần suất — 1000 lệnh → +0, 2000 → +0.69, 3000 → +1.1
        #   (1 - DD×0.3): phạt nhẹ DD (0.3 thay vì 0.4 — ưu tiên lệnh hơn DD thấp)
        #   OOS_factor: thưởng stability IS→OOS
        import math
        freq_bonus  = 1.0 + math.log(max(n_is, 1) / 1000.0) if n_is >= 1000 else 0.5
        dd_ratio    = dd_is / max(SCALPING_MAX_DD_PCT, 1e-8)
        oos_ratio   = min(net_ev_oos / max(net_ev_is, 1e-8), 1.0)
        oos_factor  = 0.65 + 0.35 * oos_ratio
        score = net_ev_is * freq_bonus * (1.0 - dd_ratio * 0.3) * oos_factor

        logger.debug(
            f"[T{trial.number}][SCALP] IS_EV={net_ev_is:.4f} OOS_EV={net_ev_oos:.4f} "
            f"DD={dd_is:.1%} IS_n={n_is} OOS_n={n_oos} freq_bonus={freq_bonus:.2f} score={score:.4f}"
        )
        return score

    return objective


# ---------------------------------------------------------------------------
# Public API — run_optimization_v2()
# ---------------------------------------------------------------------------

def run_optimization_v2(
    features:     np.ndarray,
    raw_m5:       np.ndarray,
    *,
    n_trials:     int   = 500,
    n_workers:    int   = 48,
    max_dd_limit: float = SCALPING_MAX_DD_PCT,   # 28% default
    min_trades:   int   = SCALPING_MIN_TRADES,   # 1050 IS trades default
    study_name:   str   = "rabitscal_composite_v2",
    db_path:      str   = str(DATA_DIR / "optuna_v2.db"),
    asset_cfg:    dict | None = None,
    is_ratio:     float = IS_RATIO,
) -> OptimizationResult:
    """
    Phase 2 Training Entry Point — Exness Scalping $200 Profile.

    Default settings theo Scalping profile:
        min_trades  = SCALPING_MIN_TRADES (1050 IS ≈ 1500/năm)
        max_dd_limit= SCALPING_MAX_DD_PCT (28%)
        is_ratio    = 0.70 (70% IS, 30% OOS Walk-Forward)
    """
    N = len(features)
    split_idx = int(N * is_ratio)
    is_mask = np.zeros(N, dtype=bool)
    is_mask[:split_idx] = True

    logger.info(
        f"[V2] Optimization START | trials={n_trials} workers={n_workers} | "
        f"IS={split_idx:,} ({is_ratio:.0%}) OOS={N-split_idx:,} ({1-is_ratio:.0%}) | "
        f"DD_limit={max_dd_limit:.0%} min_trades={min_trades}"
    )

    with SharedNumpyArray(features.astype(np.float32)) as feat_shm, \
         SharedNumpyArray(raw_m5.astype(np.float32))   as raw_shm, \
         SharedNumpyArray(is_mask)                     as is_shm:

        storage = optuna.storages.RDBStorage(
            url=f"sqlite:///{db_path}",
            engine_kwargs={"pool_pre_ping": True, "pool_size": 1},
        )
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=20,
                n_warmup_steps=0,
                interval_steps=1,
            ),
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=40,
                multivariate=True,              # CRITICAL: 78 params → multivariate TPE
                group=True,
                constant_liar=True,             # Giảm collision khi parallel 50 workers
                warn_independent_sampling=False, # [S3] Suppress log nhiễu — không ảnh hưởng kết quả
                seed=42,
            ),
        )

        start_t = time.time()
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            obj = _build_objective_v2(
                feat_shm.shm_name, tuple(features.shape), str(features.dtype),
                raw_shm.shm_name,  tuple(raw_m5.shape),   str(raw_m5.dtype),
                is_shm.shm_name,   tuple(is_mask.shape),  str(is_mask.dtype),
                max_dd_limit, min_trades, executor, asset_cfg,
            )
            study.optimize(obj, n_trials=n_trials, show_progress_bar=True)

        elapsed = time.time() - start_t
        best    = study.best_trial

        logger.info(
            f"[V2] DONE | best_score={best.value:.4f} | "
            f"n_trials={len(study.trials)} n_pruned={sum(1 for t in study.trials if t.state.name=='PRUNED')} | "
            f"elapsed={elapsed:.1f}s ({elapsed/60:.1f} min)"
        )

        return OptimizationResult(
            best_params=best.params,
            best_score=float(best.value),
            n_trials=len(study.trials),
            n_pruned=sum(1 for t in study.trials if t.state.name == "PRUNED"),
            duration_sec=elapsed,
            shadow_path=db_path,
            created_at=datetime.now(timezone.utc).isoformat(),
        )


# ---------------------------------------------------------------------------
# CLI Quick Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from feature_engine import build_feature_matrix

    parser = argparse.ArgumentParser(description="Phase 2: Composite Score Optimization V2.0")
    parser.add_argument("--symbol",   default="EURUSDm",  help="Symbol name")
    parser.add_argument("--data",     default="data",      help="Data directory")
    parser.add_argument("--spread",   type=float, default=0.00015)
    parser.add_argument("--trials",   type=int,   default=500)
    parser.add_argument("--workers",  type=int,   default=48)
    parser.add_argument("--max-dd",     type=float, default=SCALPING_MAX_DD_PCT)
    parser.add_argument("--min-trades", type=int,   default=SCALPING_MIN_TRADES)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Phase 2 Training — {args.symbol}")
    print(f"  trials={args.trials} workers={args.workers}")
    print(f"{'='*60}\n")

    feats, raw = build_feature_matrix(args.symbol, args.data, args.spread)
    print(f"Features: {feats.shape} | Raw M5: {raw.shape}")

    result = run_optimization_v2(
        feats, raw,
        n_trials=args.trials,
        n_workers=args.workers,
        max_dd_limit=args.max_dd,
        min_trades=args.min_trades,
        study_name=f"rabitscal_v2_{args.symbol}",
        db_path=f"data/optuna_v2_{args.symbol}.db",
    )

    print(f"\n{'='*60}")
    print(f"  DONE! Best score: {result.best_score:.4f}")
    print(f"  Trials: {result.n_trials} | Pruned: {result.n_pruned}")
    print(f"  Time: {result.duration_sec/60:.1f} min")
    print(f"  DB: {result.shadow_path}")
    print(f"{'='*60}\n")
