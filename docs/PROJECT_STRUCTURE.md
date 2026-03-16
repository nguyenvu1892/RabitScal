# 📁 RabitScal — Cây Thư Mục Chuẩn

> **Cập nhật lần cuối:** 2026-03-08 | **Branch:** `architecture/global-refactoring`
>
> ⚠️ **QUY TẮC:** Mỗi khi tạo file mới, phải đối chiếu với file này để đặt đúng thư mục.

---

## Sơ Đồ Thư Mục

```
RabitScal/
│
├── main.py                         🤖 Bot live trading — BotOrchestrator FSM
├── quant_main.py                   🚪 Cổng ML/Quant DUY NHẤT (train/backtest/report)
├── requirements.txt
├── .gitignore
├── README.md
├── PROJECT_STRUCTURE.md            ← File này
│
├── core/                           🏗️ CÁC MODULE CHẠY THỜI GIAN THỰC (BOT)
│   ├── __init__.py
│   ├── data_pipeline.py            Kéo OHLCV từ MT5, heartbeat, reconnect
│   ├── strategy_engine.py          5 vũ khí SMC+VSA: BOS/CHoCH/FVG/Pinbar/VSA
│   ├── feature_engine.py           73 features SMC/VSA/PA (vectorized)
│   ├── execution.py                OrderManager: gửi lệnh, Fill-or-Kill, spread gate
│   ├── risk_manager.py             Drawdown circuit breaker, position sizing
│   └── dashboard.py                FastAPI + WebSocket realtime dashboard
│
├── engine/                         🧠 ML & BACKTEST (OFFLINE — không dùng MT5)
│   ├── __init__.py
│   ├── ml_engine.py                Optuna TPE, SharedMemory, Ask-and-Tell batch
│   ├── backtest_engine.py          BacktestEnv: walk-forward, CSV + HTML report
│   ├── pipeline_phase1.py          Phase 1: Alpha Exploration (tìm weights tự do)
│   └── pipeline_phase2.py          Phase 2: KPI Shaping (ép khuôn Scalping $200)
│
├── utils/                          🔧 CODE DÙNG CHUNG — DRY (không lặp code ở nơi khác)
│   ├── __init__.py
│   ├── logger.py                   build_logger() — dùng ở MỌI module
│   └── data_loader.py              load_ohlcv_from_csv() — dùng ở engine + tools
│
├── tools/                          🛠️ SCRIPTS TIỆN ÍCH (chạy độc lập, không import nhau)
│   ├── export_mt5_data.py          Export MT5 data từ Windows → CSV (chỉ chạy trên Windows)
│   ├── auto_scanner.py             Quét nhiều symbol song song
│   ├── equity_simulator.py         Mô phỏng equity curve
│   ├── morning_report.py           Tạo báo cáo sáng (Top 10)
│   ├── phase1_morning_report.py    Report FANOVA/SHAP Phase 1
│   ├── optimize_all.py             Chạy optimize tất cả symbols
│   ├── rerun_failed.py             Rerun các trial bị failed
│   └── run_all_smart_dca.py        Chạy Smart DCA basket tất cả symbols
│
├── config/                         ⚙️ CẤU HÌNH (JSON — không commit .db)
│   ├── main_config.json            Config chính: symbols, sessions, MT5 params
│   ├── ml_config.json              Config ML: n_trials, n_workers, OOS hours
│   ├── risk_config.json            RiskManager: daily_dd_limit, balance_floor
│   ├── execution_config.json       OrderManager: spread, retry, magic_number
│   ├── pipeline_config.json        DataPipeline: timeframes, candle counts
│   ├── current_settings.json       Best params từ Optuna (promoted)
│   ├── state.json                  RiskManager persistent state (initial_balance)
│   └── versions/                   [GITIGNORED] Settings version history
│
├── data/                           📊 OHLCV DATA (TẤT CẢ GITIGNORED)
│   └── history_{SYMBOL}_{TF}.csv   Format: EURUSDm_M5, GBPUSDm_H1, ...
│
├── docs/                           📝 TÀI LIỆU DỰ ÁN
│   ├── walkthrough.md              Nhật ký kỹ thuật đầy đủ (append-only)
│   ├── walkthrough.md.resolved     Phân tích chi tiết task mới nhất
│   └── Proposal_V11.md             Kiến trúc V11.0 MTF Scalping
│
├── logs/                           📋 LOGS CHẠY (TẤT CẢ GITIGNORED)
│   ├── system.log                  Bot live log (RotatingFileHandler)
│   ├── ml_model.log                ML engine log
│   ├── An_Latest_Report.md         Report Optuna gần nhất (KHÔNG gitignored)
│   └── *.log                       Mọi file .log khác đều gitignored
│
├── static/                         🌐 DASHBOARD ASSETS
│   └── js/dashboard.js             Plotly + WebSocket client
│
├── templates/                      🌐 DASHBOARD HTML
│   └── index.html                  Dark theme, grid layout, Plotly CDN
│
└── archive_scripts/                🗃️ SCRIPTS ĐÃ RETIRE (chỉ để tham khảo)
    ├── ml_engine_phase1.py         Phase 1 legacy (trước khi có pipeline_phase1.py)
    └── ml_engine_v2.py             V2 legacy (trước khi có ml_engine.py)
```

---

## Quy Tắc Đặt File

| Loại file | Thư mục đúng | Ví dụ |
|-----------|-------------|-------|
| Module chạy cùng bot live | `core/` | `core/new_indicator.py` |
| ML training / backtest logic | `engine/` | `engine/pipeline_phase3.py` |
| Code dùng chung ≥ 2 modules | `utils/` | `utils/metrics.py` |
| Script tiện ích độc lập | `tools/` | `tools/export_report.py` |
| File cấu hình JSON | `config/` | `config/scanner_config.json` |
| Tài liệu / đề xuất | `docs/` | `docs/Proposal_V12.md` |
| Script cũ không dùng nữa | `archive_scripts/` | — |
| **TUYỆT ĐỐI KHÔNG để ở root** | ❌ | bất kỳ file logic nào ngoài `main.py` và `quant_main.py` |

---

## Entry Points

```bash
# Bot live trading
python main.py

# ML Training Phase 1 (Alpha Exploration)
python quant_main.py train --phase 1
python quant_main.py train --phase 1 --trials 3000 --workers 50

# ML Training Phase 2 (KPI Shaping / Scalping $200)
python quant_main.py train --phase 2
python quant_main.py train --phase 2 --mode evolution --trials 2000

# Backtest đầy đủ (CSV + HTML report)
python quant_main.py backtest --params config/current_settings.json

# Tạo report từ Optuna DB
python quant_main.py report --study data/optuna_phase1_EURUSDm.db --phase 1
```

---

## Gitignore Summary

| Pattern | Lý do |
|---------|-------|
| `data/` (`history_*.csv`) | Data lớn, local only |
| `*.db` / `config/versions/` | Optuna DB + version history |
| `logs/` | Toàn bộ log directory |
| `*.log` | Mọi file log ở bất kỳ đâu |
| `venv/` | Virtual environment |
| `__pycache__/` | Python bytecode |
