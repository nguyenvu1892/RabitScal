#!/usr/bin/env python3
"""
morning_report.py — TOP 10 CHÉN THÁNH Report Generator
========================================================
Chạy sáng hôm sau để xuất TOP 10 bộ tham số tốt nhất từ Optuna study.
Usage: python tools/morning_report.py [--study rabitscal_v2_EURUSDm] [--db data/optuna_v2_EURUSDm.db]
"""
import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def fmt_score(v: float) -> str:
    return f"{v:+.4f}" if v is not None else "N/A"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study",  default="rabitscal_v2_EURUSDm")
    parser.add_argument("--db",     default="data/optuna_v2_EURUSDm.db")
    parser.add_argument("--top",    type=int, default=10)
    parser.add_argument("--out",    default="logs/An_Latest_Report.md")
    args = parser.parse_args()

    storage = f"sqlite:///{args.db}"
    study = optuna.load_study(study_name=args.study, storage=storage)

    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    pruned    = [t for t in study.trials if t.state.name == "PRUNED"]
    total     = len(study.trials)

    print(f"Study: {args.study}")
    print(f"Total trials: {total} | Completed: {len(completed)} | Pruned: {len(pruned)}")

    if not completed:
        print("❌ Không có trial nào hoàn chỉnh. Chờ thêm!")
        return

    # Sort by value descending
    top_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:args.top]

    # ── Build report ────────────────────────────────────────────────────────
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# 🏆 MORNING REPORT — TOP {args.top} CHÉN THÁNH",
        f"",
        f"**Generated:** {now}  ",
        f"**Study:** `{args.study}`  ",
        f"**DB:** `{args.db}`  ",
        f"",
        f"## 📊 Tổng quan Optimization Run",
        f"",
        f"| Metrics | Value |",
        f"|---|---|",
        f"| Total Trials | {total:,} |",
        f"| Completed | {len(completed):,} ({len(completed)/total:.1%}) |",
        f"| Pruned | {len(pruned):,} ({len(pruned)/total:.1%}) |",
        f"| Best Score | {study.best_value:.4f} |",
        f"| Best Trial # | {study.best_trial.number} |",
        f"",
        f"---",
        f"",
        f"## 🥇 TOP {args.top} BỘ THAM SỐ (CHÉN THÁNH)",
        f"",
        f"> Sắp xếp theo **Fitness Score** = Net_EV × (1+ln(trades/1000)) × (1-DD×0.3) × OOS_factor  ",
        f"> Tất cả đã vượt qua: IS_EV>0, OOS_EV>0, IS_trades≥1050, OOS_trades≥300, DD≤28%",
        f"",
    ]

    for rank, t in enumerate(top_trials, 1):
        p = t.params
        rr     = p.get("rr_ratio",         "?")
        sl     = p.get("sl_mult",           "?")
        thresh = p.get("threshold",         "?")
        slip   = p.get("slippage_pct",      "?")
        cd     = p.get("cooldown_candles",  "?")

        # Reconstruct weight stats
        weights = [p.get(f"w{i}", 0.0) for i in range(73)]
        w_nonzero = sum(1 for w in weights if abs(w) > 0.05)
        w_pos     = sum(1 for w in weights if w > 0.05)
        w_neg     = sum(1 for w in weights if w < -0.05)
        top_w_idx = sorted(range(73), key=lambda i: abs(weights[i]), reverse=True)[:5]

        lines += [
            f"### #{rank} — Trial {t.number} | Score: **{t.value:.4f}**",
            f"",
            f"| Parameter | Value |",
            f"|---|---|",
            f"| 🎯 Fitness Score | **{t.value:.4f}** |",
            f"| RR Ratio | {rr:.3f} | " if isinstance(rr, float) else f"| RR Ratio | {rr} |",
            f"| SL Multiplier | {sl:.3f} |" if isinstance(sl, float) else f"| SL Multiplier | {sl} |",
            f"| Entry Threshold | {thresh:.4f} |" if isinstance(thresh, float) else f"| Entry Threshold | {thresh} |",
            f"| Slippage % ATR | {slip:.4f} |" if isinstance(slip, float) else f"| Slippage | {slip} |",
            f"| Cooldown Candles | {cd} candles = {int(cd)*5 if isinstance(cd, (int,float)) else '?'} min |",
            f"| Active Weights | {w_nonzero}/73 (pos={w_pos} neg={w_neg}) |",
            f"| Top 5 Feature Idx | {top_w_idx} |",
            f"",
        ]

    lines += [
        f"---",
        f"",
        f"## 🔧 Hướng Dẫn Dùng Best Params",
        f"",
        f"```python",
        f"# Query best params:",
        f"import optuna",
        f"s = optuna.load_study(study_name='{args.study}', storage='sqlite:///{args.db}')",
        f"best = s.best_trial.params",
        f"print(f\"Best RR={{ best['rr_ratio']:.3f }}\")",
        f"print(f\"Best threshold={{ best['threshold']:.4f }}\")",
        f"```",
        f"",
        f"> **Monitoring:** `tail -f logs/training_overnight_v15.log`  ",
        f"> **PID:** 63906 | **Workers:** 50 | **Target:** 5000 trials",
    ]

    report = "\n".join(lines)
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"\n✅ Report saved → {args.out}")
    print(report[:800] + "\n...")


if __name__ == "__main__":
    main()
