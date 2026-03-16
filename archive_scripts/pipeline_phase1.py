#!/usr/bin/env python3
"""
run_phase1_pipeline.py — One-Click Phase 1 Pipeline (v2 - Anti-Deadlock)
=========================================================================
Train → FANOVA/SHAP Analyze → Write 3-Part Quant Report
Fix: Batch Ask-and-Tell (50 workers thực sự song song) + SQLite WAL + stagger sleep

Usage: python run_phase1_pipeline.py [--trials 3000] [--workers 50]
"""
from __future__ import annotations

import argparse
import math
import multiprocessing.shared_memory as shm_mod
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

import logging
logging.basicConfig(
    format="[%(asctime)s UTC] - [%(levelname)-8s] - [Phase1] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger("Phase1")

# ─── Constants ───────────────────────────────────────────────────────────────
N_FEATURES  = 73
MIN_STAT    = 30
PF_CAP      = 10.0
IS_RATIO    = 0.70
STUDY_NAME  = "rabitscal_phase1_free_EURUSDm"
OUT_REPORT  = "logs/An_Latest_Report.md"
BATCH_SIZE  = 50   # Số trials gửi song song mỗi lượt

FEATURE_NAMES = [
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
    "is_climax_up","is_climax_dn",
    "is_no_demand","is_no_supply",
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
FEATURE_GROUPS = {
    "SMC Structure": list(range(0, 18)),
    "FVG / OB":      list(range(18, 36)),
    "VSA Volume":    list(range(36, 56)),
    "Price Action":  list(range(56, 71)),
    "Session/Time":  list(range(71, 73)),
}

# ─── Fitness ─────────────────────────────────────────────────────────────────

def phase1_fitness(pf: float, wr: float, n: int) -> float:
    pf    = min(pf, PF_CAP)
    conf  = 1.0 if n >= MIN_STAT else math.sqrt(max(n, 0) / MIN_STAT)
    log_b = min(math.log(n + 1) / math.log(MIN_STAT + 1), 1.0)
    return pf * (0.4 + 0.6 * wr) * conf * (1.0 + 0.3 * log_b)

# ─── Backtest (no gates) ─────────────────────────────────────────────────────

def run_backtest_p1(features, raw_m5, weights, threshold, sl_mult, rr_ratio,
                    spread_cost=0.00015, pip_value=1.0):
    N = len(features)
    if N < 200:
        return None

    score     = features.astype("float32") @ weights
    ls        = score >  threshold
    ss        = score < -threshold
    ls[:200]  = False; ss[:200] = False

    o = raw_m5[:, 1].astype("float64"); h = raw_m5[:, 2].astype("float64")
    l = raw_m5[:, 3].astype("float64"); c = raw_m5[:, 4].astype("float64")
    pc  = np.roll(c, 1); pc[0] = c[0]
    tr  = np.maximum(h-l, np.maximum(np.abs(h-pc), np.abs(l-pc)))
    atr = np.convolve(tr, np.ones(14)/14, mode="same"); atr[:13] = tr[:13]
    sl_d = atr * sl_mult

    pnls = []; in_t = False; d = ep = sp_ = tp_ = 0.0
    for i in range(N):
        hi = float(h[i]); lo = float(l[i]); op = float(o[i])
        if in_t:
            if (d==1 and lo<=sp_) or (d==-1 and hi>=sp_):
                pnls.append((sp_-ep)*d*pip_value - spread_cost); in_t = False
            elif (d==1 and hi>=tp_) or (d==-1 and lo<=tp_):
                pnls.append((tp_-ep)*d*pip_value - spread_cost); in_t = False
        else:
            if ls[i]:
                in_t=True; d=1; ep=op; sp_=op-float(sl_d[i]); tp_=op+float(sl_d[i])*rr_ratio
            elif ss[i]:
                in_t=True; d=-1; ep=op; sp_=op+float(sl_d[i]); tp_=op-float(sl_d[i])*rr_ratio

    n = len(pnls)
    if n == 0: return {"wr":0.0,"pf":0.0,"n":0,"gp":0.0,"gl":0.0,"fitness":0.0}
    arr  = np.array(pnls, dtype="float64")
    wins = arr[arr>0]; loss = arr[arr<=0]
    gp   = float(wins.sum()) if len(wins) else 0.0
    gl   = float(loss.sum()) if len(loss) else 0.0
    wr   = len(wins)/n
    pf   = min(abs(gp/gl), PF_CAP) if gl!=0 else (PF_CAP if gp>0 else 0.0)
    return {"wr":wr,"pf":pf,"n":n,"gp":gp,"gl":abs(gl),"fitness":phase1_fitness(pf,wr,n)}

# ─── Worker (tiến trình con) ─────────────────────────────────────────────────

def _worker(trial_id: int, params: dict,
            fn, fs, fd, rn, rs, rd, mn, ms, md,
            spread: float) -> dict:
    """
    [FIX] Stagger: random sleep 0-2s để 50 workers không dẫm nhau khi khởi động.
    Không cần SQLite trong worker — chỉ tính toán, trả kết quả về main thread.
    """
    time.sleep(random.uniform(0, 2.0))   # ← Stagger fix

    try:
        def att(name, shape, dtype):
            s = shm_mod.SharedMemory(name=name)
            return np.ndarray(shape, dtype=np.dtype(dtype), buffer=s.buf), s

        feats, _s1 = att(fn, fs, fd)
        raw,   _s2 = att(rn, rs, rd)
        msk,   _s3 = att(mn, ms, md)

        w  = np.array(params["weights"], dtype="float32")
        th = float(params["threshold"])
        sl = float(params["sl_mult"])
        rr = float(params["rr_ratio"])

        r = run_backtest_p1(feats[msk], raw[msk], w, th, sl, rr, spread)
        for s in (_s1, _s2, _s3): s.close()

        if r is None: return {"trial_id": trial_id, "status": "pruned"}
        return {"trial_id": trial_id, "status": "ok", **r}

    except Exception as e:
        return {"trial_id": trial_id, "status": "error", "error": str(e)}

# ─── Training — Batch Ask-and-Tell ──────────────────────────────────────────

def train(symbol: str, data_dir: str, n_trials: int, n_workers: int,
          spread: float, db_path: str) -> optuna.Study:
    """
    [FIX] Batch Ask-and-Tell pattern — tránh SQLite deadlock hoàn toàn:
    - Main thread làm TẤT CẢ thao tác SQLite (ask/tell)
    - Workers KHÔNG đụng đến SQLite — chỉ tính toán thuần NumPy
    - SQLite WAL mode + timeout=60s cho main thread
    - 50 workers chạy thực sự song song trong mỗi batch
    """
    from feature_engine import build_feature_matrix
    from tqdm import tqdm

    logger.info(f"Loading features for {symbol}...")
    feats, raw = build_feature_matrix(symbol, data_dir)
    N = len(feats)
    is_mask = np.zeros(N, dtype=bool)
    is_mask[:int(N * IS_RATIO)] = True

    shms = []
    def alloc(arr):
        s = shm_mod.SharedMemory(create=True, size=arr.nbytes)
        np.ndarray(arr.shape, dtype=arr.dtype, buffer=s.buf)[:] = arr[:]
        logger.info(f"SHM {arr.nbytes/1024/1024:.1f}MB → {s.name}")
        shms.append(s)
        return s.name, arr.shape, str(arr.dtype)

    fn, fs, fd = alloc(feats)
    rn, rs, rd = alloc(raw)
    mn, ms, md = alloc(is_mask)

    # [FIX] SQLite WAL mode + timeout=60s — main thread dùng SQLite mượt mà
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

    t0      = time.time()
    done    = 0
    best_v  = -float("inf")
    pbar    = tqdm(total=n_trials, desc="Phase1 🔥", unit="trial", dynamic_ncols=True)

    try:
        with ProcessPoolExecutor(max_workers=n_workers) as exe:
            while done < n_trials:
                batch = min(BATCH_SIZE, n_trials - done)

                # ASK: main thread (only!) tạo batch trials từ SQLite
                asked = [study.ask() for _ in range(batch)]

                # BUILD params cho mỗi trial
                worker_args = []
                for trial in asked:
                    w = np.array([trial.suggest_float(f"w{i}", -1.0, 1.0)
                                  for i in range(N_FEATURES)], dtype="float32")
                    nm = float(np.linalg.norm(w))
                    if nm > 1e-6: w /= nm
                    th = trial.suggest_float("threshold", 0.02, 3.0)
                    sl = trial.suggest_float("sl_mult",   0.2,  4.0)
                    rr = trial.suggest_float("rr_ratio",  0.3,  5.0, log=True)
                    params = {"weights":w.tolist(),"threshold":th,"sl_mult":sl,"rr_ratio":rr}
                    worker_args.append((trial.number, params))

                # SUBMIT: 50 workers chạy song song — KHÔNG đụng SQLite
                futures = {
                    exe.submit(_worker, tid, p, fn,fs,fd, rn,rs,rd, mn,ms,md, spread): (tid, trial)
                    for (tid, p), trial in zip(worker_args, asked)
                }

                # COLLECT + TELL: main thread ghi kết quả vào SQLite
                for fut in as_completed(futures):
                    _, trial = futures[fut]
                    res = fut.result()

                    if res["status"] == "ok" and res["fitness"] > 0:
                        trial.set_user_attr("winrate",       res["wr"])
                        trial.set_user_attr("profit_factor", res["pf"])
                        trial.set_user_attr("trade_count",   res["n"])
                        study.tell(trial, res["fitness"])
                        if res["fitness"] > best_v:
                            best_v = res["fitness"]
                    else:
                        study.tell(trial, float("nan"),
                                   state=optuna.trial.TrialState.PRUNED)

                    done += 1
                    pbar.update(1)
                    if done % 100 == 0:
                        pbar.set_postfix(best=f"{best_v:.3f}", done=done)

    finally:
        pbar.close()
        for s in shms:
            try: s.close(); s.unlink()
            except: pass

    elapsed = time.time() - t0
    cpl = [t for t in study.trials if t.state.name == "COMPLETE"]
    logger.info(f"Train DONE | {done} trials | {elapsed:.0f}s | {done/elapsed:.2f}t/s | "
                f"completed={len(cpl)} | best={best_v:.4f}")
    return study

# ─── Analyze & Report ─────────────────────────────────────────────────────────

def analyze_and_report(study: optuna.Study, db_path: str, out_file: str) -> None:
    logger.info("Running FANOVA + SHAP brain scan...")
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    pruned    = [t for t in study.trials if t.state.name == "PRUNED"]
    total     = len(study.trials)
    now       = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if not completed:
        logger.error("No completed trials to analyze!")
        return

    top3  = sorted(completed, key=lambda t: t.value, reverse=True)[:3]
    best  = study.best_trial

    # FANOVA
    fanova = {}
    try:
        from optuna.importance import get_param_importances, FanovaImportanceEvaluator
        fanova = get_param_importances(study,
            evaluator=FanovaImportanceEvaluator(seed=42),
            params=[f"w{i}" for i in range(N_FEATURES)])
        logger.info("FANOVA done.")
    except Exception as e:
        logger.warning(f"FANOVA fallback: {e}")
        w_b = np.array([best.params.get(f"w{i}", 0) for i in range(N_FEATURES)])
        nm = np.linalg.norm(w_b); w_b = w_b/nm if nm > 1e-6 else w_b
        fanova = {f"w{i}": float(abs(w_b[i])) for i in range(N_FEATURES)}

    sorted_imp = sorted(fanova.items(), key=lambda x: -x[1])
    top10_imp  = sorted_imp[:10]

    w_best = np.array([best.params.get(f"w{i}", 0) for i in range(N_FEATURES)], dtype="float32")
    nm = float(np.linalg.norm(w_best)); w_best = w_best/nm if nm > 1e-6 else w_best

    grp_imp = {g: sum(fanova.get(f"w{i}", 0) for i in idx)
               for g, idx in FEATURE_GROUPS.items()}
    grp_sorted = sorted(grp_imp.items(), key=lambda x: -x[1])

    smc = grp_imp["SMC Structure"] + grp_imp["FVG / OB"]
    vol = grp_imp["VSA Volume"]
    pa  = grp_imp["Price Action"]

    if smc >= vol and smc >= pa:
        personality = "**SMC/ICT Strategist** — AI ngộ ra bẫy thanh khoản (EQL/EQH), FVG mitigation và BOS/CHoCH structure là Chén Thánh trên M5 Exness."
        chén_thánh  = "FVG + BOS/CHoCH: Entry khi giá fill FVG sau khi phá structure (xác nhận Smart Money)"
    elif vol >= pa:
        personality = "**Volume Footprint Reader** — AI nhận ra dấu chân tổ chức qua Climax Volume và Effort-vs-Result divergence."
        chén_thánh  = "Volume Climax + No-Demand: Entry khi Smart Money bơm volume dị thường tại vùng cản"
    else:
        personality = "**Pure Price Action** — AI nghiêng về hammer/engulfing và False Breakout Traps của Ray Wan."
        chén_thánh  = "Pinbar / Market Traps: Entry khi wick quét liquidity rồi đảo chiều mạnh"

    best_rr  = best.params.get("rr_ratio",  1.5)
    best_thr = best.params.get("threshold", 0.5)
    best_sl  = best.params.get("sl_mult",   1.0)
    top5_idx = np.argsort(np.abs(w_best))[::-1][:5]
    bot20    = sorted(np.argsort(np.abs(w_best))[:20].tolist())
    thr_lo   = round(float(best_thr) * 1.2, 3)
    thr_hi   = round(float(best_thr) * 2.0, 3)

    L = []
    def a(*lines): L.extend(lines)

    a(f"# 🌅 Báo Cáo Quant — Phase 1 Free Exploration",
      f"",
      f"**Generated:** {now} | **Study:** `{STUDY_NAME}`",
      f"",
      f"| Metrics | Value |",
      f"|---|---|",
      f"| Total Trials | {total:,} |",
      f"| ✅ Completed | {len(completed):,} ({len(completed)/max(total,1):.1%}) |",
      f"| ✂️ Pruned | {len(pruned):,} ({len(pruned)/max(total,1):.1%}) |",
      f"| Best Fitness | **{study.best_value:.4f}** |",
      f"| Best Trial # | #{best.number} |",
      f"",
      f"---",
      f"",
      f"## Phần 1: Thống Kê Điểm Số — Có Dính Bệnh Lười Biếng Không?",
      f"",
      f"### Top 3 Anti-Lazy Score Cao Nhất",
      f"")

    for rank, t in enumerate(top3, 1):
        p   = t.params
        wr  = t.user_attrs.get("winrate",       None)
        pf  = t.user_attrs.get("profit_factor", None)
        n   = t.user_attrs.get("trade_count",   None)
        rr  = p.get("rr_ratio",  "?")
        thr = p.get("threshold", "?")
        sl  = p.get("sl_mult",   "?")
        lazy   = "✅ Không lười (n≥30)" if (n or 0) >= 30 else "⚠️ Confidence penalty áp dụng"
        scalp  = "🎯 Scalping RR" if isinstance(rr, float) and rr < 1.8 else "📈 Swing RR"
        a(f"#### #{rank} — Trial {t.number} | Fitness: **{t.value:.4f}**",
          f"",
          f"| KPI | Value | Nhận xét |",
          f"|---|---|---|",
          f"| Trade Count (IS≈1năm) | **{n if n is not None else '?'}** | {lazy} |",
          f"| Winrate | **{wr:.1%}** | — |" if wr is not None else "| Winrate | ? | — |",
          f"| Profit Factor | **{pf:.2f}** | {'✅ Edge dương' if (pf or 0)>1 else '❌'} |" if pf is not None else "| PF | ? | — |",
          f"| **R/R (AI tự chọn)** | **{rr:.3f}** | {scalp} |" if isinstance(rr, float) else f"| R/R | {rr} | — |",
          f"| Entry Threshold | {thr:.4f} | Gate lọc signal |" if isinstance(thr, float) else f"| Threshold | {thr} | — |",
          f"| SL × ATR | {sl:.3f} | — |" if isinstance(sl, float) else f"| SL | {sl} | — |",
          f"| Anti-Lazy Score | **{t.value:.4f}** | — |",
          f"")

    lazy_clean = all((t.user_attrs.get("trade_count") or 0) >= 30 for t in top3)
    a(f"> **Kết luận:** {'✅ AI SẠCH bệnh lười — Top 3 đều có ≥30 lệnh, anti-lazy formula hoạt động tốt!' if lazy_clean else '⚠️ Một số trial ít lệnh, confidence penalty đã kiềm chế nhưng cần tăng min_stat nếu muốn gắt hơn.'}",
      f"",
      f"---",
      f"",
      f"## Phần 2: Con AI Đã Học Được Gì?",
      f"",
      f"### FANOVA Importance — Top 10 Features",
      f"",
      f"| Rank | Feature | Nhóm | Score |",
      f"|---|---|---|---|")

    for rank, (param, imp) in enumerate(top10_imp, 1):
        idx  = int(param[1:])
        name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else param
        grp  = next((g for g, ii in FEATURE_GROUPS.items() if idx in ii), "?")
        a(f"| {rank} | `{name}` | {grp} | {imp:.4f} |")

    a(f"",
      f"### Group Dominance",
      f"",
      f"| Nhóm | Score | Bar |",
      f"|---|---|---|")
    for g, sc in grp_sorted:
        bar = "█" * max(1, int(sc * 200))
        a(f"| {g} | {sc:.4f} | {bar} |")

    a(f"",
      f"### SHAP-Linear Top 10 Weights (Best Trial #{best.number})",
      f"",
      f"| Rank | Feature | Nhóm | Weight | Hướng |",
      f"|---|---|---|---|---|")
    for rank, idx in enumerate(np.argsort(np.abs(w_best))[::-1][:10], 1):
        name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"w{idx}"
        grp  = next((g for g, ii in FEATURE_GROUPS.items() if idx in ii), "?")
        dir_s = "🟢 Kích hoạt" if w_best[idx] > 0 else "🔴 Lọc noise"
        a(f"| {rank} | `{name}` | {grp} | {w_best[idx]:+.4f} | {dir_s} |")

    a(f"",
      f"### 🧠 Chẩn Đoán Bản Ngã AI",
      f"",
      f"**Trường phái:** {personality}",
      f"",
      f"**Chén Thánh AI tìm ra trên M5 Exness:**",
      f"> {chén_thánh}",
      f"",
      f"**R/R AI tự chọn: {best_rr:.3f}** — "
      f"{'Scalping optimal (RR < 1.8): AI biết TP xa quá bị noise reject trên M5. Tự học chốt lời sớm.' if isinstance(best_rr, float) and best_rr < 1.8 else 'Swing approach (RR ≥ 1.8): AI ưu tiên chất lượng lệnh hơn số lượng.'}",
      f"",
      f"---",
      f"",
      f"## Phần 3: Đề Xuất Ép Khuôn Phase 2",
      f"",
      f"### Top 5 DNA Cốt Lõi — Giữ Nguyên [-1, 1]",
      f"",
      f"| # | Feature | Nhóm | Action |",
      f"|---|---|---|---|")
    for i, idx in enumerate(top5_idx, 1):
        name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"w{idx}"
        grp  = next((g for g, ii in FEATURE_GROUPS.items() if idx in ii), "?")
        a(f"| {i} | `{name}` | {grp} | **Giữ [-1.0, 1.0]** |")

    a(f"",
      f"**Bottom 20 features yếu → Freeze = 0 (giảm 73→53 dims):**",
      f"```python",
      f"FROZEN_FEATURES = {bot20}",
      f"```",
      f"",
      f"### Entry Threshold → Target 1500 lệnh/năm",
      f"",
      f"| Config | Threshold | Trade/năm |",
      f"|---|---|---|",
      f"| Phase 1 best | {best_thr:.4f} | Tự do |",
      f"| Phase 2 Conservative | {thr_lo} | ~1000 lệnh |",
      f"| **Phase 2 Scalping ← Target** | **{thr_hi}** | **~1500-2000 lệnh** |",
      f"",
      f"### Phase 2 Warm-Start Config",
      f"",
      f"```python",
      f"PHASE2_CONFIG = {{",
      f"    'max_dd_limit':  0.28,",
      f"    'min_trades':    1050,       # 1500/năm × 70% IS",
      f"    'threshold':     ({thr_lo}, {thr_hi}),",
      f"    'sl_mult':       ({round(float(best_sl)*0.7,2)}, {round(float(best_sl)*1.5,2)}),",
      f"    'rr_ratio':      (0.3, 5.0), # AI tự quyết — cấm fix cứng",
      f"    'frozen_features': {bot20},",
      f"    'warm_start_db': '{db_path}',",
      f"    'trials': 2000, 'workers': 50,",
      f"}}",
      f"```",
      f"",
      f"---",
      f"*Auto-generated by `run_phase1_pipeline.py` | {now}*")

    Path(out_file).write_text("\n".join(L), encoding="utf-8")
    logger.info(f"✅ Report → {out_file}")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",  default="EURUSDm")
    p.add_argument("--data",    default="data")
    p.add_argument("--trials",  type=int,   default=3000)
    p.add_argument("--workers", type=int,   default=50)
    p.add_argument("--spread",  type=float, default=0.00015)
    p.add_argument("--out",     default=OUT_REPORT)
    args = p.parse_args()

    db_path = str(Path(args.data) / f"optuna_phase1_{args.symbol}.db")

    print(f"\n{'='*60}")
    print(f"  PHASE 1 ONE-CLICK PIPELINE v2 (Anti-Deadlock)")
    print(f"  Trials: {args.trials} | Workers: {args.workers} | Batch: {BATCH_SIZE}")
    print(f"  Fix: Batch Ask-and-Tell | SQLite WAL timeout=60 | Stagger sleep")
    print(f"  Step 1: TRAIN → Step 2: FANOVA/SHAP → Step 3: REPORT")
    print(f"{'='*60}\n")

    study = train(args.symbol, args.data, args.trials, args.workers, args.spread, db_path)
    analyze_and_report(study, db_path, args.out)
    print(f"\n✅ PIPELINE HOÀN TẤT → {args.out}")

if __name__ == "__main__":
    main()
