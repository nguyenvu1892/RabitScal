#!/usr/bin/env python3
"""
tools/auto_scanner.py — Auto-Scanner Pipeline v3.0  (ALL-IN PARALLEL MODE)
============================================================================
Kiến trúc: "Chọn Lọc Tự Nhiên 2 Lớp Lưới" — Parallel, full Xeon utilization.

LƯỚI 1 — SHALLOW SCAN (PARALLEL):
    Chạy PARALLEL_JOBS file CSV cùng lúc, mỗi file 10 workers.
    Dual Xeon 56 threads: 5 file × 10 workers = 50 luồng (đạt ~90% CPU).
    Tiêu chuẩn pass Lưới 1: Winrate > 50% AND min_trades >= 10.
    Bao nhiêu mã qua ngưỡng → bê TẤT CẢ vào Lưới 2.

LƯỚI 2 — DEEP DIVE (PARALLEL):
    Tương tự Lưới 1 nhưng 500 trials, toàn bộ mã pass Lưới 1.

NGHIỆM THU GẮT GAO:
    • ✅ THÀNH CÔNG : Winrate > 65%  AND  OOS Pass (live | shadow)
    • ❌ FAILED     : Winrate ≤ 65%  OR   OOS Fail

SQLite ISOLATION:
    Mỗi subprocess nhận --study-db data/optuna_<symbol>.db riêng biệt.
    → Tuyệt đối không SQLite lock, dù 5 process chạy cùng lúc.

RESOURCE MANAGEMENT:
    • ThreadPoolExecutor điều phối subprocess (không tốn RAM như ProcessPool)
    • cleanup_shm() trước mỗi file → dọn /dev/shm orphans
    • cleanup_db() sau mỗi file → xóa SQLite tạm
    • gc.collect() sau mỗi batch
    • Timeout 10 phút (Shallow) / 40 phút (Deep)

Usage:
    python tools/auto_scanner.py                    # Full pipeline
    python tools/auto_scanner.py --skip-deep        # Chỉ Shallow Scan
    python tools/auto_scanner.py --jobs 3           # Override số file song song
    python tools/auto_scanner.py --workers 8        # Override workers mỗi file
    python tools/auto_scanner.py --shallow-trials 150
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────

ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
LOGS_DIR  = ROOT / "logs"
PYTHON    = sys.executable
ML_SCRIPT = str(ROOT / "ml_model.py")

# ── Parallel config (Dual Xeon 56 threads) ────────────────────────────────
PARALLEL_JOBS    = 14   # file chạy cùng lúc
WORKERS_PER_JOB  = 4    # workers mỗi file  →  14 × 4 = 56 luồng Dual Xeon

# ── Lưới 1 ────────────────────────────────────────────────────────────────
SHALLOW_TRIALS     = 100
SHALLOW_MIN_TRADES = 10
SHALLOW_TIMEOUT    = 600    # 10 phút/file
SHALLOW_GATE_WR    = 0.50

# ── Lưới 2 ────────────────────────────────────────────────────────────────
DEEP_TRIALS      = 1000
DEEP_TIMEOUT     = 4800   # 80 phút/file (1000 trials, tăng từ 40 phút)

# ── Nghiệm thu ────────────────────────────────────────────────────────────
CONFIRM_WR = 0.65

# ── Asset class map ───────────────────────────────────────────────────────
ASSET_CLASS_DETECT: list[tuple[list[str], str]] = [
    (["XAU", "XAG", "XPT", "GOLD"],                "METAL"),
    (["BTC", "ETH"],                                "BTC"),
    (["ADA", "DOGE", "SOL", "XRP", "LINK", "BNB"], "CRYPTO"),
    (["US30", "US500", "USTEC", "NAS", "DAX"],      "INDEX"),
    (["USOIL", "UKOIL", "OIL", "WTI"],              "OIL"),
    (["EUR", "GBP"],                                "FOREX-MAJOR"),
    (["JPY", "AUD", "CHF", "NZD", "CAD"],           "FOREX"),
]

RE_SCORE         = re.compile(r"Best Score\s*:\s*([\d.eE+\-]+)")   # fallback
RE_SCANNER_RESULT = re.compile(r"^SCANNER_RESULT:\s*(\{.+\})\s*$", re.MULTILINE)


# ─────────────────────────────────────────────────────────────────────────────
# Data class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScanResult:
    csv_path:      Path
    symbol:        str
    tf:            str
    asset_class:   str
    phase:         str
    status:        str   # "ok"|"pruned"|"timeout"|"missing"|"filtered"
    score:         float = 0.0
    winrate:       float = 0.0
    pf:            float = 0.0
    max_dd:        float = 0.0
    trades:        int   = 0
    duration_sec:  float = 0.0
    oos_status:    str   = "—"
    reject_reason: str   = ""
    confirmed:     bool  = False


# ─────────────────────────────────────────────────────────────────────────────
# Terminal helpers
# ─────────────────────────────────────────────────────────────────────────────

class C:
    RESET   = "\033[0m";  BOLD  = "\033[1m";  DIM   = "\033[2m"
    RED     = "\033[91m"; GREEN = "\033[92m";  YELLOW= "\033[93m"
    BLUE    = "\033[94m"; MAGENTA="\033[95m";  CYAN  = "\033[96m"


def banner(text: str, color: str = C.CYAN) -> None:
    w = 72
    pad = (w - len(text) - 2) // 2
    print(f"\n{color}{'═'*w}{C.RESET}")
    print(f"{color}║{' '*pad}{C.BOLD}{text}{C.RESET}{color}{' '*(w-pad-len(text)-1)}║{C.RESET}")
    print(f"{color}{'═'*w}{C.RESET}\n")


def section(text: str, color: str = C.BLUE) -> None:
    print(f"\n{color}{'─'*72}{C.RESET}")
    print(f"{color}  ▶  {C.BOLD}{text}{C.RESET}")
    print(f"{color}{'─'*72}{C.RESET}")


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log_ok(msg: str)   -> None: print(f"  {C.DIM}[{_ts()}]{C.RESET} {C.GREEN}✅{C.RESET}  {msg}", flush=True)
def log_warn(msg: str) -> None: print(f"  {C.DIM}[{_ts()}]{C.RESET} {C.YELLOW}⚠️ {C.RESET}  {C.YELLOW}{msg}{C.RESET}", flush=True)
def log_err(msg: str)  -> None: print(f"  {C.DIM}[{_ts()}]{C.RESET} {C.RED}❌{C.RESET}  {C.RED}{msg}{C.RESET}", flush=True)
def log_info(msg: str) -> None: print(f"  {C.DIM}[{_ts()}]{C.RESET} {C.CYAN}ℹ️ {C.RESET}  {msg}", flush=True)


def pbar(cur: int, total: int, w: int = 36) -> str:
    f = int(w * cur / total) if total else 0
    return f"{C.CYAN}[{'█'*f}{'░'*(w-f)}]{C.RESET} {C.BOLD}{cur/total*100 if total else 0:5.1f}%{C.RESET} ({cur}/{total})"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def detect_asset_class(csv_path: Path) -> str:
    name = csv_path.stem.upper()
    for keywords, cls in ASSET_CLASS_DETECT:
        for kw in keywords:
            if kw in name:
                return cls
    return "FOREX"


def detect_symbol_tf(csv_path: Path) -> tuple[str, str]:
    parts  = csv_path.stem.split("_")
    tf     = parts[-1] if len(parts) >= 2 else "?"
    symbol = parts[-2] if len(parts) >= 3 else csv_path.stem
    return symbol, tf


def study_db_path(csv_path: Path) -> Path:
    """DB riêng cho từng file — tuyệt đối không SQLite lock."""
    sym, _ = detect_symbol_tf(csv_path)
    return DATA_DIR / f"optuna_{sym}.db"


def cleanup_shm() -> None:
    count = 0
    for f in glob.glob("/dev/shm/psm_*"):
        try: Path(f).unlink(); count += 1
        except: pass


def cleanup_db(csv_path: Path) -> None:
    study_db_path(csv_path).unlink(missing_ok=True)
    (DATA_DIR / "optuna_study.db").unlink(missing_ok=True)


def parse_output(stdout: str, stderr: str) -> Optional[dict]:
    """Parse stdout của ml_model.py.
    Ưu tiên #1: dòng 'SCANNER_RESULT: {...}' — JSON chính xác, không race condition.
    Fallback: RE_SCORE cho score, còn WR/PF/DD/trades = 0 (chưa có dữ liệu).
    """
    full = stdout + stderr

    # ── Ưu tiên: SCANNER_RESULT JSON ─────────────────────────────────────────
    sm_json = RE_SCANNER_RESULT.search(full)
    if sm_json:
        try:
            d = json.loads(sm_json.group(1))
            raw = d.get("deploy", "")
            oos = ("✅ Live"   if raw == "live"    else
                   "📦 Shadow" if raw == "shadow"  else
                   "❌ Retired" if raw == "retired" else "—")
            return {
                "score":    float(d.get("score",   0)),
                "winrate":  float(d.get("winrate", 0)),
                "pf":       float(d.get("pf",      0)),
                "max_dd":   float(d.get("max_dd",  0)),
                "trades":   int(d.get("trades",    0)),
                "oos_status": oos,
                "oos_pass": raw in ("live", "shadow"),
            }
        except (json.JSONDecodeError, KeyError):
            pass   # fall through

    # ── Fallback: chỉ lấy score từ Best Score line ────────────────────────────
    sm = RE_SCORE.search(full)
    if not sm:
        return None
    return {
        "score":    float(sm.group(1)),
        "winrate":  0.0,
        "pf":       0.0,
        "max_dd":   0.0,
        "trades":   0,
        "oos_status": "—",
        "oos_pass": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core runner — 1 file, runs in a thread (subprocess isolated)
# ─────────────────────────────────────────────────────────────────────────────

def run_one_file(
    csv_path:   Path,
    trials:     int,
    workers:    int,
    min_trades: int,
    timeout:    int,
    phase:      str,
) -> ScanResult:
    """Chạy ml_model.py cho 1 file CSV. Thread-safe: subprocess isolation + DB riêng."""
    symbol, tf  = detect_symbol_tf(csv_path)
    asset_class = detect_asset_class(csv_path)
    db_path     = study_db_path(csv_path)

    if not csv_path.exists():
        return ScanResult(csv_path, symbol, tf, asset_class, phase, "missing",
                          reject_reason="File không tồn tại")

    cleanup_shm()
    cleanup_db(csv_path)

    cmd = [
        PYTHON, ML_SCRIPT,
        "--data",       str(csv_path),
        "--trials",     str(trials),
        "--workers",    str(workers),
        "--min-trades", str(min_trades),
        "--study-db",   str(db_path),    # ← DB riêng — không lock nhau
        "--log-level",  "WARNING",
    ]

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=ROOT, timeout=timeout)
        elapsed = time.perf_counter() - t0
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        cleanup_shm(); cleanup_db(csv_path)
        return ScanResult(csv_path, symbol, tf, asset_class, phase, "timeout",
                          duration_sec=elapsed, reject_reason=f"Timeout {timeout}s")

    cleanup_db(csv_path)
    cleanup_shm()

    parsed = parse_output(proc.stdout, proc.stderr)
    if proc.returncode != 0 or parsed is None:
        return ScanResult(csv_path, symbol, tf, asset_class, phase, "pruned",
                          duration_sec=elapsed, reject_reason="Pruned/error")

    return ScanResult(
        csv_path=csv_path, symbol=symbol, tf=tf,
        asset_class=asset_class, phase=phase, status="ok",
        score=parsed["score"], winrate=parsed["winrate"], pf=parsed["pf"],
        max_dd=parsed["max_dd"], trades=parsed["trades"],
        duration_sec=elapsed, oos_status=parsed["oos_status"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Parallel batch runner
# ─────────────────────────────────────────────────────────────────────────────

def run_batch_parallel(
    csv_files:  list[Path],
    trials:     int,
    workers:    int,
    min_trades: int,
    timeout:    int,
    phase:      str,
    jobs:       int,
) -> list[ScanResult]:
    """
    Chạy `jobs` file song song bằng ThreadPoolExecutor.
    Mỗi thread chạy 1 subprocess ml_model.py độc lập.
    In kết quả ngay khi xong từng file (không đợi cả batch).
    """
    results:     list[ScanResult]  = []
    done_count   = 0
    total        = len(csv_files)

    phase_color  = C.YELLOW if phase == "shallow" else C.MAGENTA
    phase_label  = "SHALLOW" if phase == "shallow" else "DEEP DIVE"

    print(f"\n  {phase_color}{C.BOLD}▣ {phase_label}: {jobs} file song song × {workers} workers{C.RESET}  "
          f"{C.DIM}(total {jobs*workers} luồng CPU){C.RESET}\n", flush=True)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        future_map = {
            pool.submit(run_one_file, csv_path, trials, workers, min_trades, timeout, phase): csv_path
            for csv_path in csv_files
        }

        # In header tiến trình khi bắt đầu
        for csv_path in csv_files:
            sym, tf = detect_symbol_tf(csv_path)
            ac      = detect_asset_class(csv_path)
            db      = study_db_path(csv_path).name
            print(f"    {C.DIM}▶ Đang chạy: {sym:12} {tf:>4} [{ac}]  DB={db}{C.RESET}", flush=True)
        print(flush=True)

        for future in as_completed(future_map):
            r          = future.result()
            done_count += 1

            # In kết quả ngay khi 1 file xong
            wr_col = (C.GREEN  if r.winrate > CONFIRM_WR     else
                      C.YELLOW if r.winrate > SHALLOW_GATE_WR else C.RED)

            if r.status == "ok":
                log_ok(
                    f"{phase_color}[{phase_label} {done_count}/{total}]{C.RESET}  "
                    f"{C.BOLD}{r.symbol:12}{C.RESET} {r.tf}  "
                    f"score={C.BOLD}{r.score:.2f}{C.RESET}  "
                    f"{wr_col}WR={r.winrate:.1%}{C.RESET}  "
                    f"PF={r.pf:.2f}  DD={r.max_dd:.1%}  rổ={r.trades}  "
                    f"{r.duration_sec:.0f}s"
                )
            elif r.status == "timeout":
                log_warn(f"{phase_label} {done_count}/{total} | {r.symbol}: TIMEOUT")
            else:
                log_err(f"{phase_label} {done_count}/{total} | {r.symbol}: {r.status} — {r.reject_reason}")

            results.append(r)

    gc.collect()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Filter logic
# ─────────────────────────────────────────────────────────────────────────────

def filter_luoi_1(results: list[ScanResult]) -> tuple[list[ScanResult], list[ScanResult]]:
    passed, rejected = [], []
    for r in results:
        if r.status != "ok":
            r.reject_reason = r.reject_reason or r.status
            rejected.append(r); continue
        fails = []
        if r.winrate <= SHALLOW_GATE_WR:
            fails.append(f"WR={r.winrate:.1%} ≤ {SHALLOW_GATE_WR:.0%}")
        if r.trades < SHALLOW_MIN_TRADES:
            fails.append(f"rổ={r.trades} < {SHALLOW_MIN_TRADES}")
        if fails:
            r.status = "filtered"
            r.reject_reason = " | ".join(fails)
            rejected.append(r)
        else:
            passed.append(r)
    passed.sort(key=lambda x: x.score, reverse=True)
    return passed, rejected


def validate_luoi_2(deep_results: list[ScanResult]) -> tuple[list[ScanResult], list[ScanResult]]:
    confirmed, failed = [], []
    for r in deep_results:
        if r.status != "ok":
            r.confirmed = False; failed.append(r); continue
        oos_pass = r.oos_status in ("✅ Live", "📦 Shadow")
        wr_pass  = r.winrate > CONFIRM_WR
        if wr_pass and oos_pass:
            r.confirmed = True; confirmed.append(r)
        else:
            r.confirmed = False
            reasons = []
            if not wr_pass:  reasons.append(f"WR={r.winrate:.1%} ≤ {CONFIRM_WR:.0%}")
            if not oos_pass: reasons.append(f"OOS={r.oos_status}")
            r.reject_reason = " | ".join(reasons)
            failed.append(r)
    confirmed.sort(key=lambda x: x.score, reverse=True)
    return confirmed, failed


# ─────────────────────────────────────────────────────────────────────────────
# Report writer
# ─────────────────────────────────────────────────────────────────────────────

def _wr_badge(wr: float) -> str:
    if wr > 0.80: return f"✅ **{wr:.1%}**"
    if wr > 0.65: return f"✅ {wr:.1%}"
    if wr > 0.50: return f"⚠️ {wr:.1%}"
    return f"❌ {wr:.1%}"

def _dd_badge(dd: float) -> str:
    if dd <= 0.10: return f"✅ **{dd:.1%}**"
    if dd <= 0.25: return f"✅ {dd:.1%}"
    if dd <= 0.40: return f"⚠️ {dd:.1%}"
    return f"❌ {dd:.1%}"


def write_report(
    shallow_all:   list[ScanResult],
    luoi1_passed:  list[ScanResult],
    deep_all:      list[ScanResult],
    confirmed:     list[ScanResult],
    failed_deep:   list[ScanResult],
    total_elapsed: float,
    args:          argparse.Namespace,
) -> Path:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# An_Latest_Report.md — Auto-Scanner v3.0 ALL-IN PARALLEL",
        f"> **[CẬP NHẬT LÚC: {now}]**  ",
        f"> Pipeline: {len(shallow_all)} file Shallow ({args.jobs} parallel) → ALL pass Lưới 1 → Deep 500t OOS 720h",
        f"",
        f"---",
        f"",
        f"## ✅ MASTER TABLE — THÀNH CÔNG ({len(confirmed)} mã)",
        f"",
        f"| # | Symbol | TF | Asset Class | Score | Winrate | PF | Max DD | Rổ | OOS |",
        f"|---|--------|----|-------------|------:|:-------:|:--:|:------:|:--:|:---:|",
    ]
    medals = {1:"🥇", 2:"🥈", 3:"🥉"}
    for i, r in enumerate(confirmed, 1):
        lines.append(
            f"| {medals.get(i, str(i))} | **{r.symbol}** | {r.tf} | {r.asset_class}"
            f" | **{r.score:.2f}** | {_wr_badge(r.winrate)}"
            f" | {r.pf:.2f} | {_dd_badge(r.max_dd)} | {r.trades} | {r.oos_status} |"
        )
    if not confirmed:
        lines.append("| — | *(không có mã nào đạt tiêu chuẩn)* | | | | | | | | |")

    lines += [
        f"",
        f"---",
        f"",
        f"## ❌ DANH SÁCH FAILED — Deep Dive ({len(failed_deep)} mã)",
        f"",
        f"| Symbol | TF | WR | OOS | Lý do |",
        f"|--------|:--:|:--:|:---:|-------|",
    ]
    for r in failed_deep:
        lines.append(
            f"| {r.symbol} | {r.tf} | {r.winrate:.1%} | {r.oos_status} | {r.reject_reason} |"
        )

    lines += [
        f"",
        f"---",
        f"",
        f"## 📊 SHALLOW SCAN — Toàn bộ {len(shallow_all)} mã",
        f"",
        f"| # | Symbol | TF | Asset Class | Score | WR | PF | DD | Rổ | Lưới 1 |",
        f"|---|--------|----|-------------|------:|:--:|:--:|:--:|:--:|:------:|",
    ]
    def _sk(r: ScanResult):
        order = {"ok": 0, "filtered": 1, "pruned": 2, "timeout": 3, "missing": 4}
        return (order.get(r.status, 9), -r.score)

    for i, r in enumerate(sorted(shallow_all, key=_sk), 1):
        in_luoi1 = any(t.csv_path == r.csv_path for t in luoi1_passed)
        badge = ("🔥 Pass"        if r.status == "ok" and in_luoi1 else
                 f"🚫 {r.reject_reason[:22]}" if r.status == "filtered"  else
                 "⏱️ Timeout"     if r.status == "timeout"              else
                 "📁 Missing"     if r.status == "missing"              else
                 f"❌ {r.status}")
        lines.append(
            f"| {i} | {r.symbol} | {r.tf} | {r.asset_class}"
            f" | {r.score:.2f} | {r.winrate:.1%} | {r.pf:.2f}"
            f" | {r.max_dd:.1%} | {r.trades} | {badge} |"
        )

    total_min = total_elapsed / 60
    ok_s  = sum(1 for r in shallow_all if r.status == "ok")
    ok_d  = sum(1 for r in deep_all    if r.status == "ok")
    lines += [
        f"",
        f"---",
        f"",
        f"## ⚙️ CONFIG",
        f"| Tham số | Lưới 1 | Lưới 2 |",
        f"|---------|:------:|:------:|",
        f"| Parallel jobs | {args.jobs} file song song | {args.jobs} file song song |",
        f"| Workers/file | {args.workers} | {args.workers} |",
        f"| Tổng threads | {args.jobs}×{args.workers}={args.jobs*args.workers} | {args.jobs}×{args.workers}={args.jobs*args.workers} |",
        f"| Trials/file | {args.shallow_trials} | {DEEP_TRIALS} |",
        f"| Ngưỡng WR | > {SHALLOW_GATE_WR:.0%} | > {CONFIRM_WR:.0%} + OOS |",
        f"| Min Trades | {SHALLOW_MIN_TRADES} | — |",
        f"| Soft Cap | **KHÔNG** (All-In) | — |",
        f"| SQLite | DB riêng mỗi file | DB riêng mỗi file |",
        f"",
        f"**Tổng thời gian:** {total_elapsed:.0f}s ({total_min:.1f} phút)  ",
        f"**Shallow OK:** {ok_s}/{len(shallow_all)} | **Pass Lưới 1:** {len(luoi1_passed)} | "
        f"**Deep OK:** {ok_d}/{len(deep_all)} | **✅ Xác nhận:** {len(confirmed)}",
        f"",
        f"---",
        f"> Resume: `python ml_model.py --data data/history_SYMBOL_TF.csv --workers 40 --trials 1000 --resume`",
    ]

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGS_DIR / "An_Latest_Report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(args: argparse.Namespace) -> None:
    t0       = time.perf_counter()
    data_dir = Path(args.data_dir)

    banner(f"AUTO-SCANNER v3.0 — ALL-IN PARALLEL 🔥  ({args.jobs} file × {args.workers} workers)", C.CYAN)

    # ── Discover ──────────────────────────────────────────────────────────
    csv_files = sorted(data_dir.glob("history_*.csv"))
    if not csv_files:
        log_err(f"Không tìm thấy history_*.csv trong {data_dir}")
        sys.exit(1)

    est_s = len(csv_files) * args.shallow_trials / args.jobs * 0.018   # ~18ms/trial rough estimate
    est_d = 0  # updated after Lưới 1
    log_info(
        f"Phát hiện {C.BOLD}{len(csv_files)}{C.RESET} file  "
        f"| Cấu hình: {args.jobs} song song × {args.workers} workers/file "
        f"= {C.BOLD}{args.jobs*args.workers} luồng CPU{C.RESET}  "
        f"| Ước tính Shallow: ~{est_s/60:.0f} phút"
    )

    # ════════════════════════════════════════════════════════════════════════
    # LƯỚI 1 — SHALLOW SCAN (PARALLEL)
    # ════════════════════════════════════════════════════════════════════════
    section(
        f"LƯỚI 1 — SHALLOW SCAN PARALLEL: {len(csv_files)} mã × {args.shallow_trials} trials  "
        f"[{args.jobs} file song song]", C.YELLOW
    )
    print(f"  {C.DIM}Ngưỡng pass: WR > {SHALLOW_GATE_WR:.0%} | trades ≥ {SHALLOW_MIN_TRADES}{C.RESET}")

    t_s = time.perf_counter()
    shallow_all = run_batch_parallel(
        csv_files, args.shallow_trials, args.workers,
        SHALLOW_MIN_TRADES, SHALLOW_TIMEOUT, "shallow", args.jobs,
    )
    t_s_elapsed = time.perf_counter() - t_s
    log_ok(f"Shallow hoàn tất: {t_s_elapsed:.0f}s ({t_s_elapsed/60:.1f} phút)")

    # ── Filter Lưới 1 ─────────────────────────────────────────────────────
    section("FILTER LƯỚI 1 — Chọn lọc tự nhiên (WR > 50%)", C.MAGENTA)
    luoi1_passed, luoi1_rejected = filter_luoi_1(shallow_all)

    print(f"\n  {C.GREEN}🔥 Pass Lưới 1:{C.RESET}  {C.BOLD}{len(luoi1_passed)} mã{C.RESET}")
    for r in luoi1_passed:
        wr_col = C.GREEN if r.winrate > CONFIRM_WR else C.YELLOW
        print(f"    {C.BOLD}{r.symbol:12}{C.RESET} {r.tf:>4}  "
              f"score={C.CYAN}{r.score:.2f}{C.RESET}  "
              f"{wr_col}WR={r.winrate:.1%}{C.RESET}  PF={r.pf:.2f}  DD={r.max_dd:.1%}  rổ={r.trades}")

    print(f"\n  {C.RED}🚫 Loại:{C.RESET}  {len(luoi1_rejected)} mã")
    for r in luoi1_rejected:
        print(f"    {C.DIM}{r.symbol:12} {r.tf:>4}  → {r.reject_reason}{C.RESET}")

    if not luoi1_passed:
        log_warn("Không có mã nào pass Lưới 1 — dừng pipeline.")
        write_report(shallow_all, [], [], [], [], time.perf_counter()-t0, args)
        return

    # ════════════════════════════════════════════════════════════════════════
    # LƯỚI 2 — DEEP DIVE (PARALLEL)
    # ════════════════════════════════════════════════════════════════════════
    deep_all: list[ScanResult] = []

    if args.skip_deep:
        log_warn("--skip-deep: Bỏ qua Deep Dive.")
    else:
        n    = len(luoi1_passed)
        est  = n * DEEP_TRIALS / args.jobs * 0.018 / 60
        section(
            f"LƯỚI 2 — DEEP DIVE PARALLEL: {n} mã × {DEEP_TRIALS} trials  "
            f"[{args.jobs} song song]  ~{est:.0f} phút", C.MAGENTA
        )
        print(f"  {C.YELLOW}{C.BOLD}⚡ ALL-IN PARALLEL — Xeon toàn lực!{C.RESET}  "
              f"{C.DIM}Nghiệm thu: WR > {CONFIRM_WR:.0%} + OOS Pass{C.RESET}\n")

        t_d = time.perf_counter()
        deep_all = run_batch_parallel(
            [r.csv_path for r in luoi1_passed],
            DEEP_TRIALS, args.workers,
            SHALLOW_MIN_TRADES, DEEP_TIMEOUT, "deep", args.jobs,
        )
        t_d_elapsed = time.perf_counter() - t_d
        log_ok(f"Deep Dive hoàn tất: {t_d_elapsed:.0f}s ({t_d_elapsed/60:.1f} phút)")

    # ════════════════════════════════════════════════════════════════════════
    # NGHIỆM THU
    # ════════════════════════════════════════════════════════════════════════
    section(f"NGHIỆM THU GẮT GAO — WR > {CONFIRM_WR:.0%} + OOS Pass", C.CYAN)
    confirmed, failed_deep = validate_luoi_2(deep_all)

    if confirmed:
        print(f"\n  {C.GREEN}{C.BOLD}✅ THÀNH CÔNG — {len(confirmed)} MÃ ĐƯỢC VINH DANH:{C.RESET}")
        medals = {1:"🥇", 2:"🥈", 3:"🥉"}
        for i, r in enumerate(confirmed, 1):
            print(f"    {medals.get(i,'  ')}  {C.BOLD}{r.symbol:12}{C.RESET} {r.tf}  "
                  f"{C.GREEN}WR={r.winrate:.1%}{C.RESET}  "
                  f"PF={r.pf:.2f}  DD={r.max_dd:.1%}  OOS={r.oos_status}")
    else:
        log_warn("Không có mã nào pass Nghiệm thu Lưới 2!")

    if failed_deep:
        print(f"\n  {C.RED}❌ FAILED — {len(failed_deep)} mã bị gạch tên:{C.RESET}")
        for r in failed_deep:
            print(f"    {C.DIM}{r.symbol:12} {r.tf}  WR={r.winrate:.1%}  "
                  f"OOS={r.oos_status}  → {r.reject_reason}{C.RESET}")

    # ════════════════════════════════════════════════════════════════════════
    # REPORT
    # ════════════════════════════════════════════════════════════════════════
    section("XUẤT BÁO CÁO — logs/An_Latest_Report.md", C.GREEN)
    total = time.perf_counter() - t0
    rp    = write_report(shallow_all, luoi1_passed, deep_all, confirmed, failed_deep, total, args)
    log_ok(f"Report: {C.BOLD}{rp}{C.RESET}")

    banner("AUTO-SCANNER v3.0 HOÀN TẤT 🚀", C.GREEN)
    if confirmed:
        best = max(confirmed, key=lambda r: r.score)
        print(f"  {C.GREEN}🏆 Champion:{C.RESET}  {C.BOLD}{best.symbol} {best.tf}{C.RESET}  "
              f"score={C.GREEN}{C.BOLD}{best.score:.2f}{C.RESET}  WR={best.winrate:.1%}\n")

    ok_s = sum(1 for r in shallow_all if r.status == "ok")
    ok_d = sum(1 for r in deep_all    if r.status == "ok")
    print(f"  ⏱  Tổng          : {total:.0f}s ({total/60:.1f} phút)")
    print(f"  ⚡ Cấu hình       : {args.jobs} file song song × {args.workers} workers = {args.jobs*args.workers} luồng")
    print(f"  📊 Shallow scanned: {len(shallow_all)} mã  (OK: {ok_s})")
    print(f"  🔥 Pass Lưới 1    : {len(luoi1_passed)} mã  (WR > 50%)")
    print(f"  🎯 Deep Dive OK   : {ok_d}/{len(deep_all)}")
    print(f"  ✅ Xác nhận       : {len(confirmed)} mã")
    print(f"  ❌ Failed         : {len(failed_deep)} mã")
    print(f"  📄 Report         : {rp}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Auto-Scanner v3.0 — All-In Parallel Mode (Dual Xeon full utilization)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",       type=str, default=str(DATA_DIR),
                   help="Thư mục chứa history_*.csv")
    p.add_argument("--jobs",           type=int, default=PARALLEL_JOBS,
                   help="Số file CSV chạy song song (parallel jobs)")
    p.add_argument("--workers",        type=int, default=WORKERS_PER_JOB,
                   help="Số Optuna workers mỗi file (--workers cho ml_model.py)")
    p.add_argument("--shallow-trials", type=int, default=SHALLOW_TRIALS,
                   help="Trials mỗi file ở Phase 1 Shallow Scan")
    p.add_argument("--skip-deep",      action="store_true", default=False,
                   help="Chỉ Shallow Scan, không Deep Dive")
    return p.parse_args()


if __name__ == "__main__":
    run_pipeline(_parse_args())
