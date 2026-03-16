#!/usr/bin/env python3
"""
quant_main.py — RabitScal Quant/ML Entry Point
================================================
Cổng điều hướng duy nhất cho toàn bộ hệ thống ML, Backtest, và Reporting.

CÁCH DÙNG:
    python quant_main.py train --phase 1
    python quant_main.py train --phase 1 --trials 3000 --workers 15
    python quant_main.py train --phase 2 --mode evolution
    python quant_main.py train --phase 2 --trials 2000 --symbol EURUSDm
    python quant_main.py backtest --params config/current_settings.json
    python quant_main.py backtest --params config/current_settings.json --out-dir logs/
    python quant_main.py report --study data/optuna_phase1_EURUSDm.db --phase 1
    python quant_main.py report --study data/optuna_phase2_EURUSDm.db --phase 2

NOTE:
    - Bot live trading vẫn chạy qua: python main.py
    - Mọi pipeline ML đều qua file này
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ─── Thêm root vào PYTHONPATH để các module tìm thấy nhau ───────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─── Entry Point Dispatcher ──────────────────────────────────────────────────

def cmd_train(args: argparse.Namespace) -> None:
    """Điều hướng lệnh train sang đúng pipeline."""
    # ── Windows RAM Safety: Cap workers at 15 ──
    import platform
    MAX_WORKERS_WIN = 15
    if args.workers > MAX_WORKERS_WIN:
        print(f"  [WARN] Workers {args.workers} > {MAX_WORKERS_WIN}! "
              f"Windows spawn sẽ nhân bản RAM. Ép về {MAX_WORKERS_WIN}.")
        args.workers = MAX_WORKERS_WIN

    if args.phase == 1:
        print(f"\n{'='*65}")
        print(f"  QUANT_MAIN → PHASE 1 PIPELINE (Alpha Exploration)")
        print(f"  Symbol: {args.symbol} | Trials: {args.trials} | Workers: {args.workers}")
        print(f"  Spread: {args.spread} | Data: {args.data} | Out: {args.out}")
        print(f"{'='*65}\n")
        from engine.pipeline_phase1 import main as phase1_main
        # Ghi đè sys.argv để pipeline_phase1.main() parse args đúng
        sys.argv = [
            "pipeline_phase1",
            "--symbol",  args.symbol,
            "--data",    args.data,
            "--trials",  str(args.trials),
            "--workers", str(args.workers),
            "--spread",  str(args.spread),
            "--out",     args.out,
        ]
        phase1_main()

    elif args.phase == 2:
        mode_label = args.mode.upper() if args.mode else "DEFAULT"
        print(f"\n{'='*65}")
        print(f"  QUANT_MAIN → PHASE 2 PIPELINE (KPI Shaping — {mode_label})")
        print(f"  Symbol: {args.symbol} | Trials: {args.trials} | Workers: {args.workers}")
        print(f"  Spread: {args.spread} | Data: {args.data} | Out: {args.out}")
        print(f"{'='*65}\n")
        from engine.pipeline_phase2 import main as phase2_main
        sys.argv = [
            "pipeline_phase2",
            "--symbol",  args.symbol,
            "--data",    args.data,
            "--trials",  str(args.trials),
            "--workers", str(args.workers),
            "--spread",  str(args.spread),
            "--out",     args.out,
        ]
        phase2_main()

    elif args.phase == 3:
        import multiprocessing as _mp
        try:
            _mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # Already set
        from engine.pipeline_mtf import train_mtf, write_mtf_header
        write_mtf_header(args.out, args.symbol)
        train_mtf(
            symbol    = args.symbol,
            data_dir  = args.data,
            n_trials  = args.trials,
            n_workers = args.workers,
            spread    = args.spread,
            out       = args.out,
        )

    elif args.phase == 4:
        import multiprocessing as _mp
        try:
            _mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        from engine.pipeline_ev04 import train_mtf_ev04, write_ev04_header
        write_ev04_header(args.out, args.symbol)
        train_mtf_ev04(
            symbol    = args.symbol,
            data_dir  = args.data,
            n_trials  = args.trials,
            n_workers = args.workers,
            spread    = args.spread,
            out       = args.out,
        )

    elif args.phase == 5:
        import multiprocessing as _mp
        try:
            _mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        from engine.pipeline_ev05 import run_ev05_queue, TF_ORDER
        from core.symbol_registry import TOP_10_SYMBOLS, ALL_SYMBOLS
        if not hasattr(args, 'symbols') or args.symbols == 'TOP10':
            symbols = TOP_10_SYMBOLS
        elif args.symbols == 'ALL':
            symbols = ALL_SYMBOLS
        else:
            symbols = [s.strip() for s in args.symbols.split(',')]
        n_total = len(symbols) * len(TF_ORDER) * (args.trials or 2000)
        print(f'✅ Đã rẽ nhánh feature/ev05_multi_asset an toàn.')
        print(f'   Server Xeon đã lên nòng {n_total:,} kịch bản! Khởi động!')
        run_ev05_queue(
            symbols   = symbols,
            data_dir  = args.data,
            n_trials  = args.trials or 2000,
            n_workers = args.workers or 15,
            spread    = args.spread,
            results_dir = 'data/ev05_results',
            report_out  = args.out,
        )

    elif args.phase == 6:
        import multiprocessing as _mp
        try:
            _mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        from engine.pipeline_ev05b import run_ev05b_queue
        from core.symbol_registry import SUPER_6_SYMBOLS
        n_total = len(SUPER_6_SYMBOLS) * 3 * (args.trials or 2000)
        print(f'\n{"="*65}')
        print(f'  SUPER PORTFOLIO EV05B — {len(SUPER_6_SYMBOLS)} symbols x 3 TF x {args.trials or 2000} trials')
        print(f'  Total: {n_total:,} kịch bản | H1 MTF Filter ON | Global Risk Gate: 3 lệnh')
        print(f'{"="*65}\n')
        run_ev05b_queue(
            symbols     = SUPER_6_SYMBOLS,
            tfs         = ["M1", "M5", "M15"],
            data_dir    = args.data,
            n_trials    = args.trials or 2000,
            n_workers   = args.workers or 15,
            spread      = args.spread,
            results_dir = "data/ev05b_results",
            report_out  = args.out,
        )
        # Generate Fact Sheet after training
        from engine.portfolio_report import generate_fact_sheet
        generate_fact_sheet(
            output_path = "logs/ev05b_fact_sheet.md",
            data_dir    = args.data,
        )

    elif args.phase == 7:
        import multiprocessing as _mp
        try:
            _mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        from engine.pipeline_ev05c import run_ev05c_queue
        from core.symbol_registry import SUPER_6_SYMBOLS
        n_total = len(SUPER_6_SYMBOLS) * 3 * (args.trials or 2000)
        print(f'\n{"="*65}')
        print(f'  EV05c SUPER PORTFOLIO — Real ML Signal (feats@weights)')
        print(f'  {len(SUPER_6_SYMBOLS)} symbols × 3 TF × {args.trials or 2000} trials = {n_total:,} kịch bản')
        print(f'  H1 Filter ON | 3% risk | OOS Gate 25% | Split Ticket FVG')
        print(f'{"="*65}\n')
        run_ev05c_queue(
            symbols    = SUPER_6_SYMBOLS,
            tfs        = ["M1", "M5", "M15"],
            data_dir   = args.data,
            n_trials   = args.trials or 2000,
            n_workers  = args.workers or 15,
            spread     = args.spread,
            report_out = args.out,
        )

    elif args.phase == 8:
        import multiprocessing as _mp
        try:
            _mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        from engine.pipeline_ev05d import run_ev05d_queue
        from core.symbol_registry import ELITE_5_SYMBOLS
        n_total = len(ELITE_5_SYMBOLS) * 3 * (args.trials or 2000)
        print(f'\n{"="*65}')
        print(f'  EV05d $200 LIVE REHEARSAL — Hybrid Risk Engine')
        print(f'  {len(ELITE_5_SYMBOLS)} symbols × 3 TF × {args.trials or 2000} trials = {n_total:,} kịch bản')
        print(f'  Capital: $200 | XAU=0.01 fix | Others=3% ($6/trade)')
        print(f'{"="*65}\n')
        run_ev05d_queue(
            symbols    = ELITE_5_SYMBOLS,
            tfs        = ["M1", "M5", "M15"],
            data_dir   = args.data,
            n_trials   = args.trials or 2000,
            n_workers  = args.workers or 15,
            spread     = args.spread,
            report_out = args.out,
        )

    else:
        print(f"[ERROR] --phase phải là 1–8. Nhận được: {args.phase}", file=sys.stderr)
        sys.exit(1)


def cmd_backtest(args: argparse.Namespace) -> None:
    """Chạy BacktestEnv với bộ params từ JSON file."""
    import json
    import numpy as np

    print(f"\n{'='*65}")
    print(f"  QUANT_MAIN → BACKTEST ENGINE")
    print(f"  Params: {args.params} | Data: {args.data} | Out: {args.out_dir}")
    print(f"{'='*65}\n")

    from engine.backtest_engine import BacktestEnv
    from utils.data_loader import load_ohlcv_from_csv

    # Load params
    params_path = Path(args.params)
    if not params_path.exists():
        print(f"[ERROR] Params file không tồn tại: {params_path}", file=sys.stderr)
        sys.exit(1)
    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    # Load data
    data = load_ohlcv_from_csv(args.data)

    # Run backtest
    env    = BacktestEnv()
    report = env.run(data, params)

    # Export
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = env.export_trade_log(report, out_path=str(out_dir / "trade_log.csv"))
    html_path = None
    try:
        html_path = env.generate_html_report(report, out_path=str(out_dir / "backtest_report.html"))
    except ImportError:
        print("[WARN] plotly chưa cài — bỏ qua HTML report. Cài bằng: pip install plotly")

    print(f"\n✅ BACKTEST HOÀN TẤT")
    print(f"   WR={report.winrate:.1%} | PF={report.profit_factor:.2f} "
          f"| MaxDD={report.max_drawdown:.1%} | Trades={report.trade_count}")
    print(f"   CSV  → {csv_path}")
    if html_path:
        print(f"   HTML → {html_path}")


def cmd_report(args: argparse.Namespace) -> None:
    """Tạo Markdown report từ Optuna study DB."""
    print(f"\n{'='*65}")
    print(f"  QUANT_MAIN → REPORT GENERATOR")
    print(f"  Study: {args.study} | Phase: {args.phase} | Out: {args.out}")
    print(f"{'='*65}\n")

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    db_path = Path(args.study)
    if not db_path.exists():
        print(f"[ERROR] Study DB không tồn tại: {db_path}", file=sys.stderr)
        sys.exit(1)

    # Tìm study name trong DB
    storage = f"sqlite:///{db_path}"
    study_names = optuna.get_all_study_names(storage)
    if not study_names:
        print(f"[ERROR] Không tìm thấy study nào trong: {db_path}", file=sys.stderr)
        sys.exit(1)

    study_name = study_names[-1]  # Lấy study mới nhất
    print(f"[INFO] Tìm thấy study: {study_name}")
    study = optuna.load_study(study_name=study_name, storage=storage)

    if args.phase == 1:
        from engine.pipeline_phase1 import analyze_and_report
        analyze_and_report(study, str(db_path), args.out)
    elif args.phase == 2:
        from engine.pipeline_phase2 import write_report
        write_report(study, args.out)
    else:
        print(f"[ERROR] --phase phải là 1 hoặc 2", file=sys.stderr)
        sys.exit(1)

    print(f"\n✅ REPORT HOÀN TẤT → {args.out}")


# ─── Argparse Setup ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quant_main.py",
        description=(
            "╔══════════════════════════════════════════════════════════╗\n"
            "║       RabitScal — Quant/ML Entry Point (quant_main)     ║\n"
            "║                                                          ║\n"
            "║  Single cổng điều hướng cho toàn bộ hệ thống ML/Quant  ║\n"
            "║  Bot live trading → python main.py                      ║\n"
            "╚══════════════════════════════════════════════════════════╝"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        metavar="<command>",
        help="Chọn chế độ chạy",
    )
    subparsers.required = True

    # ── train ─────────────────────────────────────────────────────────────
    train_p = subparsers.add_parser(
        "train",
        help="Chạy ML training pipeline (Phase 1 hoặc Phase 2)",
        description=(
            "Chạy Optuna optimization pipeline.\n\n"
            "Ví dụ:\n"
            "  python quant_main.py train --phase 1\n"
            "  python quant_main.py train --phase 1 --trials 3000 --workers 15\n"
            "  python quant_main.py train --phase 2 --mode evolution\n"
            "  python quant_main.py train --phase 2 --trials 2000 --symbol EURUSDm"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    train_p.add_argument(
        "--phase", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8], required=True,
        help="Phase: 1=Alpha, 2=KPI, 3=MTF, 4=FVG, 5=Ev05, 6=Ev05b, 7=Ev05c, 8=Ev05d $200 Rehearsal"
    )
    train_p.add_argument(
        "--mode", default="evolution", choices=["evolution", "hard-constraint"],
        help="Chế độ Phase 2 (mặc định: evolution)"
    )
    train_p.add_argument("--symbol",  default="EURUSDm", help="Symbol (mặc định: EURUSDm)")
    train_p.add_argument("--data",    default="data",    help="Thư mục data (mặc định: data/)")
    train_p.add_argument("--trials",  type=int,   default=None,
                         help="Số trials Optuna (mặc định: 3000 cho P1, 2000 cho P2)")
    train_p.add_argument("--workers", type=int,   default=15, help="Số worker song song (mặc định: 15, max 15 trên Windows)")
    train_p.add_argument("--spread",  type=float, default=0.00015, help="Spread cost (mặc định: 0.00015)")
    train_p.add_argument("--out",     default="logs/An_Latest_Report.md",
                         help="File output report (mặc định: logs/An_Latest_Report.md)")
    train_p.set_defaults(func=cmd_train)

    # ── backtest ──────────────────────────────────────────────────────────
    bt_p = subparsers.add_parser(
        "backtest",
        help="Chạy BacktestEnv đầy đủ, xuất CSV + HTML report",
        description=(
            "Chạy backtest chi tiết với bộ params cho trước.\n\n"
            "Ví dụ:\n"
            "  python quant_main.py backtest --params config/current_settings.json\n"
            "  python quant_main.py backtest --params best.json --out-dir logs/backtest/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    bt_p.add_argument("--params",  required=True, help="JSON file chứa best_params từ Optuna")
    bt_p.add_argument("--data",    default=None,  help="CSV data file (mặc định: data/history_m5.csv)")
    bt_p.add_argument("--out-dir", default="logs", dest="out_dir",
                      help="Thư mục output cho CSV + HTML (mặc định: logs/)")
    bt_p.set_defaults(func=cmd_backtest)

    # ── report ────────────────────────────────────────────────────────────
    rp_p = subparsers.add_parser(
        "report",
        help="Tạo Markdown report từ Optuna study database",
        description=(
            "Tạo báo cáo từ SQLite Optuna DB.\n\n"
            "Ví dụ:\n"
            "  python quant_main.py report --study data/optuna_phase1_EURUSDm.db --phase 1\n"
            "  python quant_main.py report --study data/optuna_phase2_EURUSDm.db --phase 2 --out logs/report.md"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    rp_p.add_argument("--study",  required=True, help="Đường dẫn tới Optuna SQLite DB")
    rp_p.add_argument("--phase",  type=int, choices=[1, 2], required=True,
                      help="Phase (chọn đúng report template)")
    rp_p.add_argument("--out",    default="logs/An_Latest_Report.md",
                      help="Output markdown file (mặc định: logs/An_Latest_Report.md)")
    rp_p.set_defaults(func=cmd_report)

    return parser


def main() -> None:
    parser  = build_parser()
    args    = parser.parse_args()

    # Tự động set default trials theo phase
    if hasattr(args, "trials") and args.trials is None:
        args.trials = 3000 if args.phase == 1 else 2000

    args.func(args)


if __name__ == "__main__":
    main()
