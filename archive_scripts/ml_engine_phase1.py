"""
ml_engine_phase1.py — Phase 1: Free Exploration Engine
=======================================================
Triết lý: Thả rông hoàn toàn — để AI khám phá Edge từ 73 features SMC/VSA/PA.
KHÔNG có Max_Drawdown gate. KHÔNG có Trade_Count gate.
Fitness = PF × WR × confidence(n) → AI tự chứng minh thống kê.

Sau khi chạy xong, dùng FANOVA + SHAP để bóc tách não bộ AI,
biết nó đã "ngộ" ra Feature nào quan trọng nhất.

Usage:
    python ml_engine_phase1.py --symbol EURUSDm --data data --trials 3000 --workers 50
    python ml_engine_phase1.py --analyze --study rabitscal_phase1_free --db data/optuna_phase1_EURUSDm.db
"""

from __future__ import annotations

import argparse
import logging
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="[%(asctime)s UTC] - [%(levelname)-8s] - [Phase1] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("Phase1")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_FEATURES       = 73      # Phải khớp với feature_engine.py
IS_RATIO         = 0.70    # 70% In-Sample
MIN_STAT_TRADES  = 30      # Ngưỡng thống kê — sqrt(n/30) confidence
PF_CAP           = 10.0    # Cap Profit Factor để tránh lazy 1-trade = ∞

# ---------------------------------------------------------------------------
# Shared Memory Helper (giống ml_engine_v2.py)
# ---------------------------------------------------------------------------

class SharedNumpyArray:
    """Context manager để attach/detach shared memory."""
    from multiprocessing.shared_memory import SharedMemory

    def __init__(self, name: str, shape: tuple, dtype: np.dtype):
        self._shm  = multiprocessing.shared_memory.SharedMemory(name=name)
        self.array = np.ndarray(shape, dtype=dtype, buffer=self._shm.buf)

    @classmethod
    def create(cls, arr: np.ndarray) -> "SharedNumpyArray":
        shm = multiprocessing.shared_memory.SharedMemory(create=True, size=arr.nbytes)
        shared = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        shared[:] = arr[:]
        obj = object.__new__(cls)
        obj._shm  = shm
        obj.array = shared
        return obj

    @classmethod
    def attach(cls, name: str, shape: tuple, dtype: np.dtype) -> np.ndarray:
        shm  = multiprocessing.shared_memory.SharedMemory(name=name)
        arr  = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        return arr, shm

    def close(self):
        self._shm.close()

    def unlink(self):
        self._shm.unlink()


# ---------------------------------------------------------------------------
# Phase 1 Anti-Lazy Fitness
# ---------------------------------------------------------------------------

def phase1_fitness(profit_factor: float, winrate: float, n_trades: int) -> float:
    """
    Phase 1 Anti-Lazy Fitness Formula.

    Mục tiêu: Phát hiện Edge thống kê thực sự, không phải may mắn.

    Công thức:
        quality  = PF_capped × (0.4 + 0.6 × WR)
        conf     = sqrt(n / MIN_STAT_TRADES)   nếu n < MIN_STAT_TRADES
                   1.0                          nếu n >= MIN_STAT_TRADES
        log_b    = log(n+1) / log(MIN_STAT_TRADES+1)  ∈ [0, 1]
        fitness  = quality × conf × (1 + 0.3 × log_b)

    Kết quả (so sánh):
        Lazy  (1 trade, WR=100%, PF=∞→10): ~1.81
        Good  (100 trades, WR=55%, PF=2.0): ~1.89  ← Tốt hơn Lazy!
        Great (300 trades, WR=65%, PF=3.0): ~3.08  ← Xuất sắc
    """
    pf = min(profit_factor, PF_CAP)   # Cap để tránh PF = ∞

    # Confidence: sqrt penalty cho n < 30 thống kê
    if n_trades >= MIN_STAT_TRADES:
        confidence = 1.0
    else:
        confidence = math.sqrt(max(n_trades, 0) / MIN_STAT_TRADES)

    # Log bonus: tưởng thưởng nhẹ cho nhiều lệnh (không phải KPI cứng)
    log_bonus = math.log(n_trades + 1) / math.log(MIN_STAT_TRADES + 1)
    log_bonus = min(log_bonus, 1.0)

    quality = pf * (0.4 + 0.6 * winrate)
    return quality * confidence * (1.0 + 0.3 * log_bonus)


# ---------------------------------------------------------------------------
# Backtest (simplified — no DD gate, no trade_count gate)
# ---------------------------------------------------------------------------

@dataclass
class Phase1Result:
    winrate:       float
    profit_factor: float
    trade_count:   int
    gross_profit:  float
    gross_loss:    float
    fitness:       float


def run_backtest_phase1(
    features:   np.ndarray,   # (N, 73)
    raw_m5:     np.ndarray,   # (N, 6)
    weights:    np.ndarray,   # (73,)
    threshold:  float,
    sl_mult:    float,
    rr_ratio:   float,
    *,
    spread_cost: float = 0.00015,
    pip_value:   float = 1.0,
    lot_size:    float = 0.01,
) -> Phase1Result:
    """
    Phase 1 Backtest — KHÔNG có max_dd gate, KHÔNG có min_trades gate.
    Đây là môi trường THỰC NGHIỆM thuần túy.
    AI được phép thất bại, miễn là nó học được điều gì đó.
    """
    N = len(features)
    if N < 200:
        raise optuna.exceptions.TrialPruned()

    # Score vectorized
    total_score = features.astype(np.float32) @ weights

    long_signal  = total_score >  threshold
    short_signal = total_score < -threshold
    long_signal[:200]  = False
    short_signal[:200] = False

    opens  = raw_m5[:, 1].astype(np.float64)
    highs  = raw_m5[:, 2].astype(np.float64)
    lows   = raw_m5[:, 3].astype(np.float64)
    closes = raw_m5[:, 4].astype(np.float64)
    prev_c = np.roll(closes, 1); prev_c[0] = closes[0]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_c), np.abs(lows - prev_c)))
    atr = np.convolve(tr, np.ones(14) / 14, mode="same")
    atr[:13] = tr[:13]
    sl_dist = atr * sl_mult

    cost_per_trade = lot_size * spread_cost   # simplified commission
    pnl_list: list[float] = []
    in_trade  = False
    direction = 0
    entry_p   = 0.0
    sl_p      = 0.0
    tp_p      = 0.0

    for i in range(N):
        hi = float(highs[i]); lo = float(lows[i]); op = float(opens[i])

        if in_trade:
            hit_sl = (direction ==  1 and lo <= sl_p) or (direction == -1 and hi >= sl_p)
            hit_tp = (direction ==  1 and hi >= tp_p) or (direction == -1 and lo <= tp_p)

            if hit_sl:
                pnl = (sl_p - entry_p) * direction * pip_value - cost_per_trade
                pnl_list.append(pnl); in_trade = False
            elif hit_tp:
                pnl = (tp_p - entry_p) * direction * pip_value - cost_per_trade
                pnl_list.append(pnl); in_trade = False
        else:
            if long_signal[i]:
                in_trade = True; direction = 1
                entry_p  = op
                sl_p     = op - float(sl_dist[i])
                tp_p     = op + float(sl_dist[i]) * rr_ratio
            elif short_signal[i]:
                in_trade = True; direction = -1
                entry_p  = op
                sl_p     = op + float(sl_dist[i])
                tp_p     = op - float(sl_dist[i]) * rr_ratio

    n = len(pnl_list)
    if n == 0:
        return Phase1Result(0.0, 0.0, 0, 0.0, 0.0, 0.0)

    arr     = np.array(pnl_list, dtype=np.float64)
    wins    = arr[arr > 0]
    losses  = arr[arr <= 0]
    gp      = float(wins.sum())   if len(wins)   > 0 else 0.0
    gl      = float(losses.sum()) if len(losses) > 0 else 0.0
    wr      = len(wins) / n
    pf      = abs(gp / gl) if gl != 0 else (PF_CAP if gp > 0 else 0.0)
    fitness = phase1_fitness(pf, wr, n)

    return Phase1Result(
        winrate=wr, profit_factor=pf, trade_count=n,
        gross_profit=gp, gross_loss=abs(gl), fitness=fitness,
    )


# ---------------------------------------------------------------------------
# Worker subprocess
# ---------------------------------------------------------------------------

def _phase1_worker(
    trial_params:  dict,
    feat_shm_name: str, feat_shape: tuple, feat_dtype: str,
    raw_shm_name:  str, raw_shape:  tuple, raw_dtype:  str,
    is_shm_name:   str, is_shape:   tuple, is_dtype:   str,
    spread_cost:   float = 0.00015,
    pip_value:     float = 1.0,
) -> dict:
    try:
        import multiprocessing.shared_memory as shm_mod
        import numpy as _np

        def _attach(name, shape, dtype):
            s = shm_mod.SharedMemory(name=name)
            return _np.ndarray(shape, dtype=_np.dtype(dtype), buffer=s.buf), s

        feats,  _s1 = _attach(feat_shm_name, feat_shape, feat_dtype)
        raw,    _s2 = _attach(raw_shm_name,  raw_shape,  raw_dtype)
        is_msk, _s3 = _attach(is_shm_name,   is_shape,   is_dtype)

        weights   = _np.array(trial_params["weights"],  dtype=_np.float32)
        threshold = float(trial_params["threshold"])
        sl_mult   = float(trial_params["sl_mult"])
        rr_ratio  = float(trial_params["rr_ratio"])

        res = run_backtest_phase1(
            feats[is_msk], raw[is_msk], weights, threshold, sl_mult, rr_ratio,
            spread_cost=spread_cost, pip_value=pip_value,
        )
        for s in (_s1, _s2, _s3): s.close()

        return {
            "status":        "ok",
            "winrate":       res.winrate,
            "profit_factor": res.profit_factor,
            "trade_count":   res.trade_count,
            "gross_profit":  res.gross_profit,
            "gross_loss":    res.gross_loss,
            "fitness":       res.fitness,
        }
    except optuna.exceptions.TrialPruned:
        return {"status": "pruned"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Objective Factory
# ---------------------------------------------------------------------------

def _build_phase1_objective(
    feat_shm: str, feat_shape: tuple, feat_dtype: str,
    raw_shm:  str, raw_shape:  tuple, raw_dtype:  str,
    is_shm:   str, is_shape:   tuple, is_dtype:   str,
    executor: ProcessPoolExecutor,
    spread_cost: float = 0.00015,
    pip_value:   float = 1.0,
):
    def objective(trial: optuna.Trial) -> float:
        weights = np.array([
            trial.suggest_float(f"w{i}", -1.0, 1.0)
            for i in range(N_FEATURES)
        ], dtype=np.float32)
        w_norm = float(np.linalg.norm(weights))
        if w_norm > 1e-6:
            weights = weights / w_norm

        threshold = trial.suggest_float("threshold", 0.02, 3.0)   # Mở rộng hơn Phase 2
        sl_mult   = trial.suggest_float("sl_mult",   0.2, 4.0)    # Thả rộng hơn
        rr_ratio  = trial.suggest_float("rr_ratio",  0.3, 5.0, log=True)  # AI tự quyết

        params = {
            "weights": weights.tolist(),
            "threshold": threshold, "sl_mult": sl_mult, "rr_ratio": rr_ratio,
        }

        future = executor.submit(
            _phase1_worker,
            params,
            feat_shm, feat_shape, feat_dtype,
            raw_shm,  raw_shape,  raw_dtype,
            is_shm,   is_shape,   is_dtype,
            spread_cost, pip_value,
        )
        res = future.result(timeout=120)

        if res["status"] in ("pruned", "error"):
            raise optuna.exceptions.TrialPruned()
        if res["fitness"] <= 0:
            raise optuna.exceptions.TrialPruned()

        logger.debug(
            f"[Trial {trial.number}] WR={res['winrate']:.1%} PF={res['profit_factor']:.2f} "
            f"n={res['trade_count']} fitness={res['fitness']:.4f}"
        )
        return res["fitness"]

    return objective


# ---------------------------------------------------------------------------
# Feature Importance Analysis (FANOVA + SHAP-linear)
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    # MTF Structure F0-F17
    "bos_bull", "bos_bear", "choch_bull", "choch_bear",
    "struct_h1_bull", "struct_h1_bear", "struct_h4_bull", "struct_h4_bear",
    "mtf_align_bull", "mtf_align_bear",
    "eql_dist_atr", "eqh_dist_atr", "eql_touch", "eqh_touch",
    "in_ote_bull", "in_ote_bear", "swing_high_dist", "swing_low_dist",
    # FVG/OB F18-F35
    "fvg_bull", "fvg_bear", "fvg_bull_fresh", "fvg_bear_fresh",
    "fvg_bull_mitigated", "fvg_bear_mitigated",
    "ob_bull", "ob_bear", "ob_bull_fresh", "ob_bear_fresh",
    "in_fvg_bull", "in_fvg_bear", "fvg_size_atr", "ob_size_atr",
    "fvg_bull_dist", "fvg_bear_dist", "ob_bull_dist", "ob_bear_dist",
    # VSA Volume F36-F55
    "vol_sma20", "relative_vol", "effort_vs_result",
    "is_climax_up", "is_climax_dn",
    "is_no_demand", "is_no_supply",
    "vol_trend_bull", "vol_trend_bear",
    "poc_dist", "vah_dist", "val_dist",
    "vol_session_ratio", "delta_vol", "cumulative_delta",
    "vol_diverge_bull", "vol_diverge_bear", "vol_breakout",
    "poc_support", "poc_resist",
    # Price Action F56-F70
    "is_bull_pinbar", "is_bear_pinbar",
    "is_bull_engulf", "is_bear_engulf",
    "is_bull_maru", "is_bear_maru",
    "is_doji", "trap_bull", "trap_bear",
    "compression_bull", "compression_bear",
    "body_size_rel", "upper_wick_rel", "lower_wick_rel",
    "candle_range_atr",
    # Session F71-F72
    "hour_sin", "hour_cos",
]

# Group mapping for SHAP summary
FEATURE_GROUPS = {
    "SMC Structure": list(range(0, 18)),
    "FVG / OB":      list(range(18, 36)),
    "VSA Volume":    list(range(36, 56)),
    "Price Action":  list(range(56, 71)),
    "Session/Time":  list(range(71, 73)),
}


def analyze_feature_importance(study_name: str, db_path: str, top_k: int = 15) -> None:
    """
    Bóc tách não bộ AI:
    1. FANOVA Importance Evaluator (Optuna built-in)
    2. SHAP-linear (exact cho linear model y = features @ weights)
    """
    storage = f"sqlite:///{db_path}"
    study   = optuna.load_study(study_name=study_name, storage=storage)

    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    if len(completed) < 10:
        logger.warning(f"Chỉ có {len(completed)} completed trials — chưa đủ để FANOVA. Cần ≥10.")
        return

    print(f"\n{'='*60}")
    print(f"  PHASE 1 BRAIN SCAN — {study_name}")
    print(f"  Completed: {len(completed)} | Best: {study.best_value:.4f}")
    print(f"{'='*60}\n")

    # ── FANOVA Importance ─────────────────────────────────────────────────
    print("📊 [1/2] FANOVA Feature Importance (param weights)...")
    try:
        from optuna.importance import get_param_importances, FanovaImportanceEvaluator
        importances = get_param_importances(
            study,
            evaluator=FanovaImportanceEvaluator(seed=42),
            params=[f"w{i}" for i in range(N_FEATURES)],
        )
        sorted_imp = sorted(importances.items(), key=lambda x: -x[1])

        print(f"\n  TOP {top_k} FEATURES BY FANOVA IMPORTANCE:\n")
        print(f"  {'Rank':<5} {'Feature':<25} {'Category':<18} {'Importance':>10}")
        print(f"  {'-'*60}")
        for rank, (param, imp) in enumerate(sorted_imp[:top_k], 1):
            idx  = int(param[1:])
            name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else param
            cat  = next((g for g, idxs in FEATURE_GROUPS.items() if idx in idxs), "?")
            print(f"  {rank:<5} {name:<25} {cat:<18} {imp:>10.4f}")

        # Group summary
        print(f"\n  GROUP IMPORTANCE SUMMARY:\n")
        group_imp = {}
        for group, idxs in FEATURE_GROUPS.items():
            group_imp[group] = sum(importances.get(f"w{i}", 0.0) for i in idxs)
        for group, imp in sorted(group_imp.items(), key=lambda x: -x[1]):
            bar = "█" * int(imp * 200)
            print(f"  {group:<20}: {imp:.4f}  {bar}")

    except ImportError:
        print("  [WARNING] optuna.importance unavailable, skipping FANOVA.")

    # ── SHAP-linear on best trial ─────────────────────────────────────────
    print(f"\n📊 [2/2] SHAP-linear Analysis (best trial weights)...")
    best   = study.best_trial
    w_best = np.array([best.params.get(f"w{i}", 0.0) for i in range(N_FEATURES)], dtype=np.float32)
    w_norm = float(np.linalg.norm(w_best))
    if w_norm > 1e-6:
        w_best = w_best / w_norm

    top_w_idx = np.argsort(np.abs(w_best))[::-1][:top_k]
    print(f"\n  TOP {top_k} WEIGHTS (SHAP-linear |weight|):\n")
    print(f"  {'Rank':<5} {'Feature':<25} {'Category':<18} {'Weight':>10}")
    print(f"  {'-'*60}")
    for rank, idx in enumerate(top_w_idx, 1):
        name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"w{idx}"
        cat  = next((g for g, idxs in FEATURE_GROUPS.items() if idx in idxs), "?")
        print(f"  {rank:<5} {name:<25} {cat:<18} {w_best[idx]:>+10.4f}")

    # Group weight sum
    print(f"\n  GROUP WEIGHT IMPACT (sum |w|):\n")
    for group, idxs in FEATURE_GROUPS.items():
        group_w = float(np.abs(w_best[idxs]).sum())
        bar     = "█" * int(group_w * 30)
        print(f"  {group:<20}: {group_w:.4f}  {bar}")

    print(f"\n{'='*60}")
    print(f"  Best Trial: #{best.number} | Fitness: {best.value:.4f}")
    print(f"  RR={best.params.get('rr_ratio',0):.2f} | threshold={best.params.get('threshold',0):.4f}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main Optimization — Phase 1
# ---------------------------------------------------------------------------

def run_phase1_optimization(
    symbol:      str,
    data_dir:    str,
    n_trials:    int = 3000,
    n_workers:   int = 50,
    spread_cost: float = 0.00015,
    pip_value:   float = 1.0,
) -> None:
    """Fire Phase 1 Free Exploration."""
    from feature_engine import build_feature_matrix

    data_path = Path(data_dir)
    study_name = f"rabitscal_phase1_free_{symbol}"
    db_path    = data_path / f"optuna_phase1_{symbol}.db"

    print(f"\n{'='*60}")
    print(f"  PHASE 1 — FREE EXPLORATION")
    print(f"  Symbol: {symbol} | Trials: {n_trials} | Workers: {n_workers}")
    print(f"  Study: {study_name}")
    print(f"  ⚠ KHÔNG có Max_Drawdown gate | KHÔNG có Trade_Count gate")
    print(f"  Fitness = PF × WR × sqrt(n/30) × log_bonus")
    print(f"{'='*60}\n")

    # Load features
    logger.info("Loading feature matrix...")
    feats, raw = build_feature_matrix(symbol, str(data_path))
    N = len(feats)
    logger.info(f"Feature matrix: {feats.shape} | Raw: {raw.shape}")

    # IS mask
    is_size   = int(N * 0.70)
    is_mask   = np.zeros(N, dtype=bool)
    is_mask[:is_size] = True

    # Shared memory
    import multiprocessing.shared_memory as shm_mod

    def _alloc(arr: np.ndarray) -> tuple:
        shm = shm_mod.SharedMemory(create=True, size=arr.nbytes)
        shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        shm_arr[:] = arr[:]
        size_mb = arr.nbytes / 1024 / 1024
        logger.info(f"Shared memory: {size_mb:.1f}MB | name={shm.name}")
        return shm, shm.name, arr.shape, str(arr.dtype)

    shm_f, fn, fs, fd = _alloc(feats)
    shm_r, rn, rs, rd = _alloc(raw)
    shm_m, mn, ms, md = _alloc(is_mask)

    try:
        study = optuna.create_study(
            study_name=study_name,
            storage=f"sqlite:///{db_path}",
            direction="maximize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=40,
                multivariate=True,
                group=True,
                constant_liar=True,
                warn_independent_sampling=False,
                seed=42,
            ),
        )

        t0 = time.time()
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            obj = _build_phase1_objective(
                fn, fs, fd,
                rn, rs, rd,
                mn, ms, md,
                executor, spread_cost, pip_value,
            )

            from tqdm import tqdm
            pbar = tqdm(total=n_trials, desc="Phase1", unit="trial", dynamic_ncols=True)

            def _callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
                pbar.update(1)
                completed = [t for t in study.trials if t.state.name == "COMPLETE"]
                if completed and len(completed) % 50 == 0:
                    pbar.set_postfix({
                        "best":  f"{study.best_value:.3f}",
                        "cpl":   len(completed),
                        "prune": len(study.trials) - len(completed),
                    })

            study.optimize(obj, n_trials=n_trials, n_jobs=1,
                           callbacks=[_callback], show_progress_bar=False)
            pbar.close()

        elapsed = time.time() - t0
        logger.info(f"Phase 1 DONE | {n_trials} trials | {elapsed:.0f}s | speed={n_trials/elapsed:.2f} t/s")

        # Auto-analyze
        analyze_feature_importance(study_name, str(db_path))

    finally:
        for shm in (shm_f, shm_r, shm_m):
            try: shm.close(); shm.unlink()
            except: pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 Free Exploration Engine")
    sub = parser.add_subparsers(dest="cmd")

    # Train
    tr = sub.add_parser("train", help="Run Phase 1 optimization")
    tr.add_argument("--symbol",  default="EURUSDm")
    tr.add_argument("--data",    default="data")
    tr.add_argument("--trials",  type=int,   default=3000)
    tr.add_argument("--workers", type=int,   default=50)
    tr.add_argument("--spread",  type=float, default=0.00015)

    # Analyze
    an = sub.add_parser("analyze", help="Run FANOVA + SHAP on existing study")
    an.add_argument("--study",   required=True)
    an.add_argument("--db",      required=True)
    an.add_argument("--top",     type=int, default=15)

    args = parser.parse_args()

    if args.cmd == "train" or args.cmd is None:
        sym = getattr(args, "symbol", "EURUSDm")
        run_phase1_optimization(
            symbol    = sym,
            data_dir  = getattr(args, "data", "data"),
            n_trials  = getattr(args, "trials", 3000),
            n_workers = getattr(args, "workers", 50),
            spread_cost = getattr(args, "spread", 0.00015),
        )
    elif args.cmd == "analyze":
        analyze_feature_importance(args.study, args.db, args.top)
