# 🧠 MASTER PLAN — Rabit_Exness AI (v1.1 — Revised & Hardened)

> **Version:** 1.1 | **Date:** 2026-03-05 | **Status:** Draft — Pending TechLead Approval
> **Revision Notes:** Bổ sung Safety Net Layer, State Machine, FVG TTL, ML Shadow Deployment, Execution Fill-or-Kill dựa trên phân tích phản biện Task 0.

---

## 🎯 MỤC TIÊU HỆ THỐNG

Xây dựng bot giao dịch Fully-Automated cho cặp Forex trên nền tảng **MetaTrader 5 — Exness Standard Cent**, sử dụng chiến lược kết hợp **SMC (Smart Money Concept) + VSA (Volume Spread Analysis)** trên đa khung thời gian (H1/M15/M5), có khả năng **tự tiến hóa (Self-Evolving)** thông qua module Machine Learning chạy ngầm.

---

## 📁 KIẾN TRÚC MODULE (Sau cải tiến)

```
rabit_exness/
├── main.py                    # State Machine Orchestrator
├── data_pipeline.py           # MT5 Connection + Data Quality + Heartbeat
├── strategy_engine.py         # 5 Weapons + FVG TTL + VSA Session-aware
├── risk_manager.py            # Per-trade + Daily DD + Safety Net Layer
├── execution.py               # Fill-or-Kill + Spread Filter + Retry Logic
├── backtest_env.py            # OHLC Worst-Case + Slippage Model + Walk-Forward
├── ml_model.py                # Bayesian Opt. + Shadow Deploy + Config Versioning
├── config/
│   ├── current_settings.json  # Active config (file-locked)
│   └── versions/              # Config version history (rollback)
├── data/
│   └── trade_log.csv          # Chi tiết mọi lệnh giao dịch
├── logs/
│   └── system.log             # Log toàn hệ thống
├── docs/
│   └── walkthrough.md         # Báo cáo hoạt động
└── History.txt                # Git-style project journal
```

---

## ⚙️ GIAI ĐOẠN 1: Nền tảng Kết nối & Xử lý Ma sát Thị trường

### `data_pipeline.py` — Data Ingestion Layer
- Kết nối MetaTrader5 (Standard Cent), kéo data H1, M15, M5 realtime
- **[NEW v1.1]** `mt5_reconnect()`: exponential backoff (1s → 2s → 4s → 8s → max 60s)
- **[NEW v1.1]** Heartbeat thread độc lập: ping MT5 mỗi 30 giây
- **[NEW v1.1]** Data Validator: kiểm tra candle `None`, time gap bất thường, timezone chuẩn hóa UTC
- Session Filters: tránh khung tin tức kinh tế, chọn khung giờ thanh khoản cao (London/NY session open)

### `execution.py` — Order Execution Layer
- **[REVISED v1.1]** Kiểm tra spread realtime trước khi gửi lệnh: nếu spread > `MAX_SPREAD_PIPS` → abort
- **[REVISED v1.1]** Implement **Fill-or-Kill**: retry tối đa `MAX_RETRY=3`, mỗi retry tính lại Entry/SL/TP từ giá mới
- Tính Commission và Dynamic Spread vào Entry/SL/TP
- Ghi log lý do từ chối lệnh: REQUOTE / TIMEOUT / SPREAD_TOO_HIGH

### `risk_manager.py` — Risk Control Layer
- Max Drawdown 3%/lệnh. Tính Lot Size tự động bằng ATR(14) và Balance
- **[NEW v1.1 — CRITICAL] Safety Net Layer:**
  - `daily_drawdown_limit = 6%` → auto-pause bot đến ngày hôm sau
  - `consecutive_loss_streak = 3` → cool-down 4 giờ
  - `min_balance_floor = 50% initial_balance` → hard stop toàn hệ thống

---

## ⚔️ GIAI ĐOẠN 2: Lõi Chiến lược — Strategy Engine

### H1 — Market Structure Analysis
- Dò Swing High/Low bằng thuật toán **confirmed-close only** (chỉ tính trên `data[:-1]`, bỏ candle đang mở)
- **[REVISED v1.1]** Lock swing level sau khi BOS/CHoCH confirmed — không tính lại để tránh repainting
- Nhận diện BOS (Break of Structure) và CHoCH (Change of Character)

### M15 — SMC Gap / FVG Scanner
- Scan 3 nến tìm Fair Value Gap (FVG). Lưu FVG pool
- Hủy FVG khi giá mitigate (lấp đầy)
- **[NEW v1.1]** FVG Time-To-Live (TTL): `max_age = 48 candles M15` (~12 giờ) → auto-expire
- **[NEW v1.1]** FVG pool limit: `max_size = 20`, FIFO khi đầy
- **[NEW v1.1]** FVG Quality Tag: FVG tạo trong vùng tin tức được tag `quality=LOW`, không dùng làm trigger

### M5 — Trigger & VSA Confirmation
- Chỉ kích hoạt khi giá chạm vùng FVG (M15)
- **Pinbar confirmation:** `wick_length > min(ATR(14) * 0.5, body_length * 2.0)` — adaptive threshold
- **[REVISED v1.1] VSA Session-aware:** Volume baseline tính riêng theo session (Asian/London/NY)
- Volume Climax (so với session baseline) hoặc No Demand/No Supply
- **[NEW v1.1]** VSA Quality Score (0–100) thay vì binary True/False
- SL động dựa vào ATR(14): `SL = wick_tip + spread + (ATR_multiplier * ATR(14))`

---

## 🧪 GIAI ĐOẠN 3: Môi trường Mô phỏng Realistic & Backtest

### `backtest_env.py`
- Test lịch sử với dữ liệu tick (hoặc OHLC với worst-case model)
- **[REVISED v1.1]** OHLC Worst-Case: nếu High/Low candle có thể chứa SL hit → SL triggered
- **[NEW v1.1]** Random Slippage Model: `slippage = gauss(mean=avg_spread, sigma=spread_std)`
- **[NEW v1.1]** Walk-Forward Validation: chia data thành rolling windows (train/validate/test)
- Bắt buộc áp dụng Spread/Commission giả định thực tế
- Ghi log toàn bộ vào `system.log` và chi tiết lệnh vào `data/trade_log.csv`

---

## 🤖 GIAI ĐOẠN 4: Tự tiến hóa bằng Machine Learning

### `ml_model.py`
- Dùng **Bayesian Optimization** hoặc **Genetic Algorithms** để tìm cấu hình siêu tham số tối ưu
  - Hệ số FVG quality threshold, ngưỡng VSA score, tỷ lệ Pinbar, ATR Multiplier SL/TP
- **[REVISED v1.1 — CRITICAL] File Lock:** threading.Lock khi đọc/ghi `current_settings.json`
- **[NEW v1.1 — CRITICAL] Shadow Deployment:**
  - Config mới chạy paper trade song song 24h trước khi deploy thật
  - Nếu paper performance > current performance → deploy
- **[NEW v1.1] Config Versioning:** lưu `settings_v{NNN}.json` → rollback tự động nếu performance giảm >10%
- **Điều kiện tối thiểu để ML optimize:** ≥ 500 lệnh lịch sử trong `trade_log.csv`
- Thu thập dữ liệu: ghi lệnh vào `data/trade_log.csv`, config vào `config/current_settings.json`

---

## 🚀 GIAI ĐOẠN 5: Tích hợp & Vận hành

### `main.py` — State Machine Orchestrator

```
[IDLE] → [SCANNING] → [SIGNAL_FOUND] → [PENDING_ORDER] → [IN_TRADE] → [CLOSING] → [IDLE]
```

- **[NEW v1.1 — CRITICAL]** Mỗi state có timeout và transition rule rõ ràng:
  - `IDLE`: chờ data pipeline ready
  - `SCANNING`: quét M5 trigger, timeout 60s
  - `SIGNAL_FOUND`: validate Risk Manager, timeout 30s
  - `PENDING_ORDER`: gửi lệnh, chờ fill confirmation, timeout 10s
  - `IN_TRADE`: chỉ monitor + manage SL/TP, KHÔNG scan signal mới
  - `CLOSING`: xử lý đóng lệnh, ghi log
- **Hard rule:** max 1 lệnh mở đồng thời (nới rộng sau khi hệ thống stable)
- Điều phối vòng lặp đầy đủ: **Data → Strategy → Risk → Execute → Log → ML Optimization**

---

## 📊 TIÊU CHÍ NGHIỆM THU (Acceptance Criteria)

| Giai đoạn | Tiêu chí tối thiểu |
|-----------|-------------------|
| Giai đoạn 1 | MT5 kết nối ổn định ≥ 99% uptime trong 48h test |
| Giai đoạn 2 | Signal accuracy ≥ 60% trên dữ liệu in-sample (không repainting verified) |
| Giai đoạn 3 | Backtest Profit Factor ≥ 1.3, Max Drawdown ≤ 10% |
| Giai đoạn 4 | ML cải thiện Winrate ≥ 5% sau 500 lệnh so với baseline rule-based |
| Giai đoạn 5 | Vòng lặp tự động chạy ổn định 72h không crash |

---

## 🗓️ LỘ TRÌNH TRIỂN KHAI ĐỀ XUẤT

```
Task 0: Review & Khởi tạo Plan            [DONE — Branch: task-0-init-plan]
Task 1: Giai đoạn 1 — data_pipeline.py    [Pending PROCEED]
Task 2: Giai đoạn 1 — execution.py        [Pending PROCEED]
Task 3: Giai đoạn 1 — risk_manager.py     [Pending PROCEED]
Task 4: Giai đoạn 2 — strategy_engine.py  [Pending PROCEED]
Task 5: Giai đoạn 3 — backtest_env.py     [Pending PROCEED]
Task 6: Giai đoạn 4 — ml_model.py         [Pending PROCEED]
Task 7: Giai đoạn 5 — main.py             [Pending PROCEED]
Task 8: Integration Test & Acceptance      [Pending PROCEED]
```

---

## 📜 QUY TẮC VÀNG (Golden Rules — Không thể thương lượng)

1. **No Repainting:** Mọi tín hiệu chỉ được tính trên candle đã đóng hoàn toàn.
2. **Safety First:** Safety Net Layer (Daily DD + Streak + Balance Floor) phải được implement TRƯỚC khi bất kỳ lệnh live nào được gửi.
3. **Log Everything:** Mọi quyết định, lỗi, config change phải được ghi log không ngoại lệ.
4. **Shadow Before Deploy:** Mọi thay đổi ML config phải qua shadow deployment 24h.
5. **One Trade at a Time:** Tối đa 1 lệnh mở đồng thời cho đến khi hệ thống được nghiệm thu ổn định.

---

*Plan v1.1 — Rabit_Exness AI | Được tạo và cải tiến bởi Antigravity — 2026-03-05*
