# 📋 WALKTHROUGH — Rabit_Exness AI

---

## Task 0: Review và Khởi tạo Master Plan

**Date:** 2026-03-05 | **Branch:** `task-0-init-plan` | **Author:** Antigravity (Senior AI Coder & Algo Trading Expert)

---

### Nội dung thay đổi/hoạt động

Phân tích phản biện toàn bộ Master Plan 5 giai đoạn của dự án Rabit_Exness AI nhằm tìm kiếm bottleneck hiệu năng, lỗ hổng logic và rủi ro tương tác với MT5/Exness Cent trước khi bắt tay vào code. Kết quả: **CÓ ĐỀ XUẤT CẢI TIẾN** — phát hiện 9 rủi ro nghiêm trọng và 6 đề xuất nâng cấp kiến trúc.

---

### Lý do (Căn cứ Trading thực chiến)

Kế hoạch ban đầu tuy vững về mặt chiến lược (SMC multi-timeframe + VSA confirmation) nhưng thiếu lớp bảo vệ kỹ thuật cần thiết để sống sót qua các điều kiện thị trường cực đoan trên MT5/Exness Cent. Các bot algorithmic trading thất bại không phải vì chiến lược sai — mà vì **bỏ qua ma sát hạ tầng**.

---

### 🔴 PHÂN TÍCH PHẢN BIỆN CHI TIẾT — 9 RỦI RO TIỀM ẨN

---

#### [RR-01] GIAI ĐOẠN 1 — `data_pipeline.py`: Thiếu cơ chế Reconnect & Heartbeat cho MT5

**Vấn đề:** Kế hoạch chỉ đề cập "kéo data H1, M15, M5 realtime" nhưng hoàn toàn im lặng về:
- Mất kết nối MT5 đột ngột (server Exness restart, mất mạng VPS)
- Giá trị `None` hoặc candle thiếu data trong stream realtime
- Timezone mismatch giữa MT5 server (UTC+2/UTC+3) và hệ thống

**Hậu quả nếu bỏ qua:** Bot tiếp tục chạy vòng lặp với data `None` → signal sai → vào lệnh tại thời điểm tồi tệ nhất (spread cao nhất, thanh khoản thấp nhất).

**Đề xuất bổ sung:**
```
data_pipeline.py:
- Thêm hàm mt5_reconnect() với exponential backoff (1s, 2s, 4s, 8s, max 60s)
- Validate từng candle: kiểm tra None, candle thừa/thiếu, time gap bất thường
- Thêm heartbeat thread độc lập ping MT5 mỗi 30 giây
- Chuẩn hóa timezone về UTC ngay khi nhận data
```

---

#### [RR-02] GIAI ĐOẠN 1 — `execution.py`: Max Deviation chưa đủ — thiếu Fill or Kill logic

**Vấn đề:** Kế hoạch ghi "Xử lý Slippage (Max Deviation)" nhưng không định nghĩa:
- Nếu lệnh bị từ chối (REQUOTE/REJECT) thì làm gì? Retry? Bỏ? Retry bao nhiêu lần?
- Exness Cent có spread floating, trong tin tức lớn spread có thể tăng 10x–50x bình thường
- Commission tính theo lot — với Cent lot (0.01) cần kiểm tra ngưỡng commission tối thiểu

**Hậu quả:** Retry vô han → vào lệnh trễ nhưng vẫn với signal cũ → lệnh sai hướng hoàn toàn.

**Đề xuất bổ sung:**
```
execution.py:
- Implement Fill-or-Kill: nếu lệnh bị reject quá MAX_RETRY (mặc định 3) lần → hủy lệnh, ghi log lý do
- Kiểm tra realtime spread trước khi gửi lệnh: nếu spread > MAX_SPREAD_PIPS → abort
- Tính lại Entry/SL/TP sau mỗi lần retry dựa trên bid/ask mới nhất, không dùng giá cũ
```

---

#### [RR-03] GIAI ĐOẠN 1 — `risk_manager.py`: Drawdown 3% chưa có Circuit Breaker toàn hệ thống

**Vấn đề:** Giới hạn "Max Drawdown 3%/lệnh" chỉ bảo vệ ở cấp độ lệnh đơn lẻ. Kế hoạch không đề cập:
- **Daily Drawdown Limit:** Nếu thua 5 lệnh liên tiếp (5 × 3% = 15% account) thì sao?
- **Consecutive Loss Streak:** Không có cơ chế dừng bot khi thua liên tục
- **Account Balance Floor:** Bot có thể tradetiếp dù account còn rất ít tiền

**Hậu quả:** Một ngày thị trường volatile bất thường (tin kinh tế đột xuất) có thể wipe toàn bộ tài khoản Cent.

**Đề xuất bổ sung:**
```
risk_manager.py — thêm 3 lớp bảo vệ:
1. daily_drawdown_limit = 6% → tự động dừng bot đến ngày hôm sau
2. consecutive_loss_streak = 3 → cool-down 4 giờ sau 3 lệnh thua liên tiếp
3. min_balance_floor = 50% initial_balance → hard stop toàn hệ thống
```

---

#### [RR-04] GIAI ĐOẠN 2 — `strategy_engine.py` (H1): Thuật toán Swing High/Low dễ bị Lookback Bias

**Vấn đề:** "Dò Swing High/Low bằng nến xác nhận" nghe có vẻ an toàn nhưng:
- Định nghĩa "nến xác nhận" cần số nến nhìn về trước (lookback). Trong realtime, candle cuối chưa đóng.
- Nếu dùng `close[i] > close[i-1]` trên candle đang mở → **Repainting** mặc dù kế hoạch nói không!
- BOS/CHoCH cần được validate bởi **candle đã đóng hoàn toàn** (confirmed close), không phải giá tick hiện tại.

**Đề xuất bổ sung:**
```
strategy_engine.py:
- Chỉ tính Swing H/L trên slice data[:-1] (bỏ candle cuối chưa đóng)
- Lock swing level khi confirmed: sau khi BOS/CHoCH được xác nhận → giữ nguyên cấp độ đó, không tính lại
- Thêm unit test: so sánh kết quả tính realtime vs tính offline → phải khớp nhau 100%
```

---

#### [RR-05] GIAI ĐOẠN 2 — `strategy_engine.py` (M15/M5): FVG không có TTL (Time-To-Live)

**Vấn đề:** Kế hoạch ghi "Hủy FVG khi giá mitigate". Nhưng:
- FVG tạo ra từ 2 tuần trước vẫn còn hiệu lực? Thị trường đã thay đổi cấu trúc nhưng bot vẫn chờ FVG cũ!
- Không giới hạn số lượng FVG lưu trữ → memory leak nếu chạy lâu
- FVG được tạo trong session tin tức (spread cao) thường là "fake FVG" — nhiễu loạn nhiều hơn signal

**Đề xuất bổ sung:**
```
strategy_engine.py:
- Thêm FVG.max_age = 48 candles M15 (tương đương 12 giờ) → auto-expire
- Giới hạn FVG_pool.max_size = 20 FVGs, FIFO khi đầy
- Tag nguồn gốc FVG: nếu tạo trong session filter blocked → set FVG.quality = LOW, không sử dụng làm entry trigger
```

---

#### [RR-06] GIAI ĐOẠN 2 — `strategy_engine.py` (M5): VSA Climax Volume thiếu baseline động

**Vấn đề:** "Tick Volume Climax so với MA20" — MA20 là trung bình trượt trên cùng timeframe M5. Nhưng:
- MA20 của M5 trong session London khác hoàn toàn MA20 trong session Sydney (volume thấp)
- So sánh volume M5 lúc 14:00 (London Open) với MA20 tính từ 12:00 (pre-session) → sẽ luôn "climax" — tín hiệu nhiễu liên tục
- Pinbar "râu/thân" ratio cần được normalize theo ATR để tránh sai lệch trong volatile vs calm market

**Đề xuất bổ sung:**
```
strategy_engine.py:
- Tính Volume baseline riêng theo session (Asian: 21:00-08:00, London: 08:00-17:00, NY: 13:00-22:00 UTC)
- Pinbar confirmation: wick_length > min(ATR(14) * 0.5, body_length * 2.0) — adaptive threshold
- Thêm VSA quality score (0-100) thay vì binary True/False
```

---

#### [RR-07] GIAI ĐOẠN 3 — `backtest_env.py`: Backtest trên close-price không phản ánh thực tế tick

**Vấn đề:** Kế hoạch ghi "test lịch sử với dữ liệu tick" nhưng không làm rõ:
- MT5 cung cấp tick data thực hay chỉ OHLC? (Với Cent account, tick data đầy đủ rất nặng)
- Nếu chỉ có OHLC: SL hit tính dựa vào Low của candle (unrealistic — thực tế có thể SL hit giữa candle)
- Wash trade vs real execution: backtest không có queue position (ta là taker không phải maker trên Cent)

**Đề xuất bổ sung:**
```
backtest_env.py:
- Implement "OHLC worst-case model": nếu High và Low của candle đều có thể chứa SL hit → SL triggered
- Thêm random slippage model: slippage = gauss(mean=avg_spread, sigma=spread_std) cho mỗi lệnh
- Ghi rõ trong log: đây là OHLC-based backtest (không phải tick-level), kết quả có độ lệch ±X%
```

---

#### [RR-08] GIAI ĐOẠN 4 — `ml_model.py`: ML Loop Race Condition và Overfitting Trap

**Vấn đề:** Bayesian Optimization/Genetic Algorithm chạy trên `system.log` từ live trade — vấn đề nghiêm trọng:
- **Race condition:** ML đang optimize, đồng thời bot đang live trade → ML ghi đè `config/current_settings.json` TRONG KHI bot đang đọc → config không nhất quán
- **Overfitting to recent data:** Nếu 2 tuần gần nhất là trending, ML sẽ optimize cho trending — rồi thị trường sideways → collapse
- **Không có rollback:** Nếu config mới tệ hơn, không có cơ chế nào để tự động revert về bộ cũ

**Đề xuất bổ sung:**
```
ml_model.py & config system:
- Thêm file lock (threading.Lock hoặc filelock library) khi ghi/đọc current_settings.json
- Implement "shadow deployment": config mới chạy song song với config cũ trong 24h (paper trade) trước khi deploy thật
- Versioning config: lưu config theo version (settings_v001.json, v002...), rollback tự động nếu performance drop >10%
- Minimum training data: ít nhất 500 lệnh lịch sử trước khi ML được phép optimize
```

---

#### [RR-09] GIAI ĐOẠN 5 — `main.py`: Thiếu State Machine — vòng lặp không có trạng thái rõ ràng

**Vấn đề:** "Điều phối tự động: Data → Strategy → Risk → Execute → Log → ML" là mô tả tuần tự. Nhưng thực tế:
- Nếu Execute thất bại → vòng lặp tiếp tục hay dừng? Strategy có chạy lại không?
- Nếu lệnh đang mở mà vòng lặp tiếp tục → bot có mở lệnh thứ 2 không?
- MT5 disconnect ở giữa Execute → lệnh có thể đã được gửi ở server nhưng không nhận được confirmation

**Đề xuất bổ sung:**
```
main.py — implement Bot State Machine:
States: IDLE → SCANNING → SIGNAL_FOUND → PENDING_ORDER → IN_TRADE → CLOSING → IDLE
- Mỗi state có timeout riêng
- PENDING_ORDER state: check MT5 order status, không gửi lệnh mới
- IN_TRADE state: chỉ monitor và manage SL/TP, không scan signal mới
- Hard rule: max 1 lệnh mở đồng thời (có thể nới rộng sau khi đã stable)
```

---

### ✅ ĐỀ XUẤT CẢI TIẾN KIẾN TRÚC — 6 NÂNG CẤP TỔNG QUÁT

| # | Module | Đề xuất | Ưu tiên |
|---|--------|---------|---------|
| A1 | `data_pipeline.py` | Thêm **Data Quality Score** cho mỗi candle batch (% candle valid, max gap) | HIGH |
| A2 | `risk_manager.py` | Thêm **Safety Net Layer**: Daily DD Limit + Streak Limit + Balance Floor | CRITICAL |
| A3 | `strategy_engine.py` | Thêm **Composite Signal Score** (0-100) thay vì binary True/False | HIGH |
| A4 | `backtest_env.py` | Implement **Walk-Forward Validation** thay vì single in-sample backtest | HIGH |
| A5 | `ml_model.py` | Thêm **Shadow Deployment** + **Config Versioning** + **Rollback** | CRITICAL |
| A6 | `main.py` | Implement **State Machine** thay vì vòng lặp tuyến tính đơn giản | HIGH |

---

### 📐 KIẾN TRÚC ĐỀ XUẤT SAU CẢI TIẾN

```
main.py (State Machine Orchestrator)
    │
    ├─ data_pipeline.py       [Reconnect + Heartbeat + Data Quality]
    │       └─ MT5 Connection Manager (exponential backoff)
    │
    ├─ strategy_engine.py     [5 Weapons + Session-aware VSA + FVG TTL]
    │       ├─ H1: Market Structure (confirmed-close only)
    │       ├─ M15: FVG Detection (TTL + max pool)
    │       └─ M5: Trigger (Pinbar + VSA Quality Score)
    │
    ├─ risk_manager.py        [Order Risk + Safety Net Layer + Circuit Breaker]
    │       ├─ Per-trade: 3% max loss + ATR lot sizing
    │       ├─ Daily: 6% DD limit
    │       └─ System: Balance floor + Streak limit
    │
    ├─ execution.py           [Fill-or-Kill + Spread Filter + Retry Logic]
    │       └─ Order State: PENDING → FILLED / REJECTED / TIMEOUT
    │
    ├─ backtest_env.py        [OHLC Worst-Case + Slippage Model + Walk-Forward]
    │
    └─ ml_model.py            [Bayesian/GA + Shadow Deploy + Config Versioning]
            ├─ config/current_settings.json (with file lock)
            └─ config/versions/ (rollback history)
```

---

### 📊 MA TRẬN RỦI RO (Risk Matrix)

| Rủi ro | Khả năng xảy ra | Mức độ thiệt hại | Ưu tiên xử lý |
|--------|-----------------|------------------|---------------|
| MT5 mất kết nối | CAO | THẤP (nếu có reconnect) | Giai đoạn 1 |
| FVG Stale (quá cũ) | TRUNG | THẤP | Giai đoạn 2 |
| Swing Repainting | THẤP (nếu dùng confirmed) | CAO | Giai đoạn 2 |
| Execution Retry Loop | TRUNG | TRUNG | Giai đoạn 1 |
| Daily DD Runaway | THẤP | RẤT CAO | Giai đoạn 1 |
| ML Race Condition | TRUNG | CAO | Giai đoạn 4 |
| Config Overwrite | THẤP | RẤT CAO | Giai đoạn 4 |
| OHLC Backtest Bias | CAO | TRUNG | Giai đoạn 3 |
| State Machine Missing | CAO | CAO | Giai đoạn 5 |

---

### 🏁 KẾT LUẬN

Bản Master Plan có nền tảng chiến lược vững chắc (SMC + VSA + Multi-Timeframe rất hợp lý cho Exness Cent scalping). Tuy nhiên, **9 rủi ro kỹ thuật & hạ tầng** trên đây cần được tích hợp vào Plan trước khi code. Hai rủi ro CRITICAL nhất:

1. **[RR-03] Safety Net Layer** — Không có là mất account, không thể thương lượng.
2. **[RR-08] ML Race Condition + Rollback** — Triển khai ML trực tiếp lên live config mà không có shadow deploy là hành động tự sát chiến lược.

**Khuyến nghị: Bổ sung các đề xuất trên vào Plan.md trước khi sang Giai đoạn 1.**

---

*Báo cáo được tạo bởi Antigravity — 2026-03-05 | Rabit_Exness AI Project*
