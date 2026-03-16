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

---

## Task 1.1: Thiết kế kiến trúc module Data Pipeline & Tối ưu Phần cứng

**Date:** 2026-03-05 | **Branch:** `task-1.1-data-pipeline` | **Author:** Antigravity

---

### Nội dung thay đổi/hoạt động

Phân tích và thiết kế toàn bộ kiến trúc module `data_pipeline.py` (Giai đoạn 1) — module nền tảng toàn hệ thống Rabit_Exness AI. Không viết code thực thi phần cứng cho đến khi TechLead phê duyệt thiết kế này.

---

### 🖥️ PHÂN TÍCH PHẦN CỨNG — Cơ sở để thiết kế kiến trúc

**Server hiện tại:**
- **CPU:** Dual Intel Xeon E5-2680 v4 @ 2.40GHz — 2 Sockets × 14 Cores × 2 HT = **56 Logical Threads**
- **RAM:** 96GB — đủ để cache toàn bộ tick history nhiều năm cặp tiền tệ in-memory (không cần HDD I/O)
- **Storage:** SSD — write latency thấp, phù hợp ghi log liên tục tốc độ cao
- **GPU:** NVIDIA — sẽ phát huy vai trò tại Giai đoạn 4 (`ml_model.py` — Bayesian/GA optimization)

**Bản đồ phân công Thread cho toàn hệ thống (tầm nhìn dài hạn):**

```
56 Total Logical Threads
│
├─ [Nhóm A] Giai đoạn 1 — Data Pipeline (Task 1.1 này)
│       ├─ Thread-1 : fetch_h1()          → kéo candle H1 từ MT5
│       ├─ Thread-2 : fetch_m15()         → kéo candle M15 từ MT5
│       ├─ Thread-3 : fetch_m5()          → kéo candle M5 từ MT5
│       └─ Thread-4 : _heartbeat_loop()   → Daemon, ping MT5/30s
│
├─ [Nhóm B] Giai đoạn 2—5 — Main Bot Loop (main.py State Machine)
│       └─ Thread-5 : Main Thread         → IDLE→SCANNING→SIGNAL→TRADE→CLOSE
│
└─ [Nhóm C] Giai đoạn 4 — ML Optimization (Tương lai: ProcessPoolExecutor)
        └─ 48–51 Workers                  → Bayesian/GA brute-force search
                                            Full 56 cores, bypass GIL bằng
                                            multiprocessing (subprocess per core)
```

> **Nhận định:** Với workload hiện tại (chỉ 1 cặp tiền, 3 timeframes), `data_pipeline.py` chỉ cần 4 threads. 52 threads còn lại là "dự trữ chiến lược" cho khi `ml_model.py` kích hoạt ProcessPoolExecutor — đây là thiết kế đúng: **không over-thread giai đoạn sớm, nhưng phải kiến trúc để scale được**.

---

### ⚙️ LÝ DO CHỌN `ThreadPoolExecutor` THAY VÌ `asyncio`

Đây là quyết định kiến trúc quan trọng nhất của module này.

| Tiêu chí | `asyncio` | `ThreadPoolExecutor` ✅ |
|----------|-----------|----------------------|
| Phù hợp với MT5 Python API | ❌ API là **blocking/synchronous** — `asyncio` không đạt concurrency thực sự mà không wrap thêm `loop.run_in_executor()` | ✅ Mỗi blocking call `mt5.copy_rates_from_pos()` chạy trên OS thread riêng biệt |
| Độ phức tạp implementation | Cao hơn — cần `async def`, `await`, event loop management | Thấp hơn — code quen thuộc, dễ debug |
| Phù hợp với I/O-bound workload | ✅ Tốt nhất cho native-async I/O | ✅ Tốt cho blocking I/O với thread pool |
| Thread Safety với dữ liệu shared | Tốt (single-threaded event loop) | Cần `RLock` — dễ implement |
| Hướng đến ProcessPoolExecutor tương lai | Khó bridge | ✅ Dễ dàng — cùng Executor interface |

**Kết luận:** `ThreadPoolExecutor(max_workers=3)` là lựa chọn tối ưu cho wrapping MT5 blocking API vào concurrent workers. `asyncio` sẽ được xem xét lại nếu hệ thống chuyển sang REST API async trong tương lai.

---

### 🏗️ THIẾT KẾ CLASS `DataPipeline` (OOP Architecture)

```
class DataPipeline
│
├─ __init__(config: dict, logger: logging.Logger)
│       Khởi tạo: connection params, executor, RLock,
│       internal data store (dict: tf → numpy array),
│       session config, validator thresholds
│
├─ start() → None
│       Spawn heartbeat daemon thread
│       Submit fetch_all() lần đầu vào executor
│       Set internal flag _running = True
│
├─ stop() → None
│       Graceful shutdown: _running = False
│       executor.shutdown(wait=True)
│       Ghi log "DataPipeline stopped"
│
├─ fetch_all() → dict[str, np.ndarray]
│       executor.map(_fetch_candles, ["H1","M15","M5"])
│       Trả về mapping tf → validated numpy array
│       Ghi metrics (latency, data quality score) vào system.log
│
├─ _fetch_candles(timeframe: str) → np.ndarray | None    [PRIVATE — chạy trong worker thread]
│       1. Gọi mt5.copy_rates_from_pos() với retry
│       2. Nếu None → gọi mt5_reconnect()
│       3. Gọi validate_candles() trên kết quả
│       4. Chuẩn hóa UTC: rates["time"] += tz_offset
│       5. Update _data_store[timeframe] dưới RLock
│
├─ mt5_reconnect() → bool
│       Vòng lặp exponential backoff:
│         attempt 1: sleep 1s → mt5.initialize()
│         attempt 2: sleep 2s → mt5.initialize()
│         attempt 3: sleep 4s → mt5.initialize()
│         ...max sleep 60s
│         max_attempts = 10 → nếu vượt: log CRITICAL, return False
│       Mỗi attempt ghi log: "Reconnect attempt N/10, wait Xs"
│
├─ _heartbeat_loop() → None                              [PRIVATE — Daemon Thread]
│       while _running:
│           sleep(30)
│           info = mt5.terminal_info()
│           if info is None hoặc info.connected == False:
│               log WARNING "Heartbeat lost → triggering reconnect"
│               mt5_reconnect()
│
├─ validate_candles(rates: np.ndarray, tf: str) → np.ndarray | None
│       Check 1: rates is None hoặc len(rates) == 0 → return None, log ERROR
│       Check 2: Phát hiện time gap bất thường
│                (delta > expected_interval[tf] * 1.5) → log WARNING, flag candle
│       Check 3: Giá trị OHLC hợp lệ (open/high/low/close > 0, high >= low)
│       Return: ndarray đã validated, hoặc None nếu fail critical check
│
├─ is_session_active() → bool
│       Đọc giờ UTC hiện tại
│       London Open:  08:00–12:00 UTC
│       NY Open:      13:00–17:00 UTC
│       Asian Avoid:  00:00–06:00 UTC (thanh khoản thấp)
│       Return True nếu trong giờ active, False nếu outside
│
└─ get_data(timeframe: str) → np.ndarray | None          [PUBLIC — Thread-safe]
        with self._lock:
            return self._data_store.get(timeframe, None)
```

---

### 🔌 CƠ CHẾ RECONNECT — Exponential Backoff Chi Tiết

```python
# Pseudocode minh họa cơ chế mt5_reconnect()

BACKOFF_SEQUENCE = [1, 2, 4, 8, 16, 32, 60, 60, 60, 60]  # seconds (max 60s)
MAX_ATTEMPTS = 10

for attempt, wait in enumerate(BACKOFF_SEQUENCE, start=1):
    log.warning(f"[MT5] Reconnect attempt {attempt}/{MAX_ATTEMPTS}, waiting {wait}s...")
    time.sleep(wait)

    mt5.shutdown()  # Reset connection trước
    if mt5.initialize(**connection_params):
        log.info(f"[MT5] Reconnected successfully on attempt {attempt}")
        return True

log.critical("[MT5] FAILED after 10 attempts. DataPipeline HALTED.")
# Trigger safety mechanism → notify main.py State Machine to enter IDLE
return False
```

---

### 📡 SESSION FILTER — Chỉ chạy giờ thanh khoản cao

Dựa trên nghiên cứu volume Forex thực tế (Exness Standard Cent liquidity patterns):

| Session | Giờ UTC | Đặc điểm | Hành động |
|---------|---------|----------|-----------|
| **London Open** | 07:00–12:00 | Volume cao nhất, spread thấp, SMC signal mạnh | ✅ ACTIVE — full scan |
| **NY Open** | 13:00–17:00 | Volume cao, overlap tốt, momentum mạnh | ✅ ACTIVE — full scan |
| **London-NY Overlap** | 12:00–13:00 | Cực kỳ volatile, spread tăng đột biến | ⚠️ CAUTION — scan nhưng tăng ngưỡng VSA |
| **Asian Session** | 21:00–05:00 | Volume thấp, false FVG nhiều | ❌ INACTIVE — skip signal |
| **News Window** | ±15 phút xung quanh tin | Spread tăng 10x-50x | ❌ BLOCKED — tự động skip |

> **Lưu ý WSL:** Server đang chạy Linux (WSL trên Windows). `datetime.utcnow()` hoạt động chuẩn trong Python. Không phụ thuộc vào timezone OS — mọi thứ xử lý UTC absolute.

---

### 📊 DATA VALIDATOR — Tiêu chí Quality Score

Module validator trả về kèm một **Data Quality Score (0.0–1.0)** cho mỗi batch fetch:

| Kiểm tra | Trọng số | Điều kiện Pass |
|----------|----------|----------------|
| None check | Critical (fail = skip batch) | `rates is not None and len > 0` |
| Time gap normality | 0.40 | `max_gap <= expected_interval * 1.5` |
| OHLC validity | 0.30 | `high >= low`, tất cả giá > 0 |
| Volume availability | 0.20 | `tick_volume > 0` ít nhất 95% candle |
| Candle count | 0.10 | Đủ số candle yêu cầu (H1:200, M15:500, M5:1000) |

Score < 0.60 → log WARNING + bỏ qua batch này, giữ nguyên data cũ.  
Score < 0.30 → log ERROR + trigger reconnect.

---

### 🔮 THIẾT KẾ HƯỚNG TỚI `ProcessPoolExecutor` (Giai đoạn 4)

`DataPipeline` được thiết kế để `ml_model.py` và `backtest_env.py` có thể consume data dễ dàng:

1. **Dữ liệu lưu dạng `numpy ndarray`** — serializable, zero-copy compatible với `multiprocessing.shared_memory`
2. **Public getter `get_data(tf)`** trả về numpy array — `ProcessPoolExecutor` workers chỉ cần gọi hàm này, không cần biết cơ chế bên trong
3. **Tách biệt hoàn toàn** IO thread pool (ThreadPoolExecutor) và CPU thread pool (ProcessPoolExecutor tương lai) — không share executor, không deadlock

```
Khi ml_model.py kích hoạt (Giai đoạn 4):

ProcessPoolExecutor(max_workers=48)   ← chiếm 48/56 cores cho Bayesian/GA search
    ├─ Worker-1:  evaluate_config(config_v001)
    ├─ Worker-2:  evaluate_config(config_v002)
    ├─ ...
    └─ Worker-48: evaluate_config(config_v048)

# Mỗi worker đọc data từ DataPipeline.get_data() → tính PnL độc lập
# Không conflict, không GIL throttle (subprocess riêng)
# 96GB RAM đủ để 48 workers mỗi worker load 1-2GB tick history
```

---

### 📐 CẤU TRÚC FILE ĐỀ XUẤT

```
/home/xeon-sever/RabitScal/
├── data_pipeline.py        ← Module này (Task 1.1)
├── config/
│   └── pipeline_config.json   ← MT5 credentials, symbols, candle counts
└── logs/
    └── system.log             ← Mọi event: connect, reconnect, validate, session
```

**`pipeline_config.json` dự kiến:**
```json
{
  "symbol": "EURUSDc",
  "timeframes": {
    "H1":  { "mt5_tf": "TIMEFRAME_H1",  "candles": 200 },
    "M15": { "mt5_tf": "TIMEFRAME_M15", "candles": 500 },
    "M5":  { "mt5_tf": "TIMEFRAME_M5",  "candles": 1000 }
  },
  "heartbeat_interval_sec": 30,
  "reconnect_max_attempts": 10,
  "data_quality_min_score": 0.60,
  "session_filters": {
    "london_open_utc":  [7, 12],
    "ny_open_utc":      [13, 17],
    "asian_avoid_utc":  [21, 5]
  }
}
```

---

### 📝 ĐỀ XUẤT CẢI TIẾN CHO KIẾN TRÚC NÀY

#### Đề xuất 1: Shared Memory Buffer (cho tương lai scale-out)
Thay vì `dict` in-memory thông thường, có thể dùng `multiprocessing.shared_memory` để chia sẻ numpy array trực tiếp giữa processes mà không serialize/deserialize — giảm latency từ ~5ms xuống ~0.1ms khi `ml_model.py` truy cập data.

#### Đề xuất 2: Adaptive Fetch Interval
Thay vì fetch cố định mỗi N giây, detect khi nào candle mới đóng (compare `rates[-1]["time"]` với lần fetch trước) và chỉ push update khi có candle mới — giảm MT5 API call không cần thiết trong giờ thanh khoản thấp.

#### Đề xuất 3: Redis Cache Layer (nếu scale múltiple bots)
Nếu tương lai cần chạy nhiều bot song song (nhiều cặp tiền), dùng Redis pub/sub để `DataPipeline` broadcast data một lần, nhiều consumers subscribe — tránh tình trạng 5 bots cùng gọi MT5 API tại cùng thời điểm.

---

### 🏁 KẾT LUẬN THIẾT KẾ TASK 1.1

| Yêu cầu TechLead | Giải pháp thiết kế |
|------------------|--------------------|
| Không block Main Thread | ThreadPoolExecutor(3) + Daemon heartbeat thread riêng biệt |
| Tận dụng 56 Threads Xeon | 4 threads giai đoạn này, 48+ cho ProcessPoolExecutor giai đoạn 4 |
| mt5_reconnect() với exponential backoff | Sequence: 1s→2s→4s→8s→16s→32s→60s×4, max 10 attempt |
| Session Filters | London(07-12 UTC) + NY(13-17 UTC), skip Asian + News window |
| Data Validator | Quality Score 0-1, None/gap/OHLC/volume/count checks |
| OOP Class structure | Class DataPipeline với public API rõ ràng, RLock thread-safe |
| Hướng tới ProcessPoolExecutor | Data store numpy array, getter public, tách biệt executor pools |

**Trạng thái:** ✅ Thiết kế hoàn chỉnh — TechLead APPROVED — TIẾN HÀNH IMPLEMENTATION

---

### 📦 KHỞI TẠO MÔI TRƯỜNG — `requirements.txt`

**Date:** 2026-03-05 21:45 UTC+7 | **Action:** Tạo file `requirements.txt` cho toàn dự án

Theo yêu cầu TechLead, khởi tạo file `requirements.txt` với 3 nhóm thư viện chia theo vai trò trong hệ thống:

#### Group 1 — Core Trading & Data
| Thư viện | Version | Vai trò |
|----------|---------|---------|
| `MetaTrader5` | ≥5.0.45 | MT5 Python API — kéo bar/tick, gửi lệnh |
| `pandas` | ≥2.2.0 | DataFrame xử lý OHLCV, `trade_log.csv` |
| `numpy` | ≥1.26.4 | Structured ndarray, shared memory (GIL bypass Giai đoạn 4) |
| `pytz` | ≥2024.1 | Timezone UTC chuẩn hoá cho Exness server offset |
| `filelock` | ≥3.13.1 | File lock cho `config/current_settings.json` — tránh ML race condition |
| `schedule` | ≥1.2.1 | Session window scheduling |

#### Group 2 — Web Dashboard (FastAPI + WebSocket)
| Thư viện | Version | Vai trò |
|----------|---------|---------|
| `fastapi` | ≥0.110.0 | Async REST + WebSocket API framework |
| `uvicorn[standard]` | ≥0.29.0 | ASGI server (uvloop + httptools) — throughput tối đa |
| `websockets` | ≥12.0 | Realtime chart data streaming đến browser |
| `plotly` | ≥5.20.0 | Interactive candlestick chart, equity curve |
| `pandas-ta` | ≥0.3.14b | TA indicators (ATR, EMA, RSI) trực tiếp trên pandas |
| `aiofiles` | ≥23.2.1 | Async file I/O — stream `system.log` lên dashboard realtime |

#### Group 3 — Machine Learning & Optimization
| Thư viện | Version | Vai trò |
|----------|---------|---------|
| `optuna` | ≥3.6.0 | Bayesian Hyperparameter Optimization (TPE) — dùng ProcessPoolExecutor 48 workers |
| `deap` | ≥1.4.1 | Genetic Algorithm / Evolutionary Strategy |
| `torch` | ≥2.2.2 | PyTorch — RL Agent (PPO/DQN) với GPU acceleration |
| `stable-baselines3` | ≥2.3.0 | RL algorithms chuẩn công nghiệp |
| `gymnasium` | ≥0.29.1 | Trading env interface cho `backtest_env.py` |
| `joblib` | ≥1.4.0 | Parallel CPU jobs, model serialization |

**Lý do pin version cụ thể:** Tránh breaking API changes (PyTorch, FastAPI, Gymnasium đều hay có breaking changes giữa major versions). Pin `>=` thay vì `==` để còn nhận patch security fixes.

---

### 💻 IMPLEMENTATION — `data_pipeline.py` CODE HOÀN TẤT

**Date:** 2026-03-05 22:00 UTC+7 | **Deliverable:** `data_pipeline.py` (production-ready)

#### Tóm lược những gì đã implement

| Method | Vai trò | Thread |
|--------|---------|--------|
| `__init__()` | Khởi tạo config, executor, RLock, data store, logger | Main |
| `start()` | Kết nối MT5, spawn heartbeat daemon, initial fetch | Main |
| `stop()` | Graceful shutdown executor + MT5, join heartbeat | Main |
| `fetch_all()` | Submit 3 futures song song vào ThreadPoolExecutor | Main → 3 Workers |
| `_fetch_candles(tf)` | MT5 call → retry → UTC normalize → validate → store | Worker Thread |
| `mt5_reconnect()` | Exponential backoff: `[1,2,4,8,16,32,60,60,60,60]`s, 10 attempts | Any Thread |
| `_heartbeat_loop()` | Daemon: ping MT5/30s → auto-reconnect → refresh data | Daemon Thread |
| `validate_candles()` | Quality Score 0–1: None/gap/OHLC/volume/count checks | Worker Thread |
| `is_session_active()` | London 07-12 / NY 13-17 UTC, skip Asian+News | Any Thread |
| `get_data(tf)` | Thread-safe getter (RLock), trả về `numpy ndarray` | Any Thread |
| `_detect_server_tz()` | Auto-detect Exness server UTC offset (UTC+2/+3) | Init |

#### Quy chuẩn Logging

Format áp dụng: `[TIME] - [LEVEL] - [MODULE] - [MESSAGE]`

```
[2026-03-05 14:30:00 UTC] - [INFO]    - [DataPipeline] - MT5 connected | terminal='MetaTrader 5' | server_tz_offset=UTC+2h
[2026-03-05 14:30:05 UTC] - [DEBUG]   - [DataPipeline] - _fetch_candles(H1): candles=200 | latency=23.5ms | quality=0.97 | passed=True
[2026-03-05 14:30:05 UTC] - [DEBUG]   - [DataPipeline] - _fetch_candles(M15): candles=500 | latency=31.2ms | quality=0.95 | passed=True
[2026-03-05 14:30:05 UTC] - [DEBUG]   - [DataPipeline] - _fetch_candles(M5): candles=1000 | latency=47.8ms | quality=0.93 | passed=True
[2026-03-05 15:00:30 UTC] - [WARNING] - [DataPipeline] - Heartbeat ✗ — MT5 terminal unreachable. Triggering reconnect...
[2026-03-05 15:00:31 UTC] - [WARNING] - [DataPipeline] - [MT5] Reconnect attempt 1/10 — waiting 1s before retry...
[2026-03-05 15:00:33 UTC] - [INFO]    - [DataPipeline] - [MT5] Reconnected successfully on attempt 2/10
```

#### Files đã tạo trong commit này

```
RabitScal/
├── data_pipeline.py            ← Module chính (600+ dòng, production-ready)
├── requirements.txt             ← 3 nhóm thư viện với version pin
├── config/
│   └── pipeline_config.json    ← Cấu hình symbol, timeframes, session filters
└── logs/                        ← auto-created bởi DataPipeline khi start()
    └── system.log
```

**Trạng thái:** ✅ Code hoàn tất — **PENDING TechLead review & PROCEED để merge vào `main`**

---

*Cập nhật bởi Antigravity — 2026-03-05 22:00 UTC+7 | Branch: task-1.1-data-pipeline*

---

> **📌 SOP UPDATE (TechLead — 2026-03-05 22:00):** Từ đây, `walkthrough.md` CHỈ ghi tóm tắt ngắn (≤7 bullets/task, append-only). Phân tích chi tiết viết vào `docs/walkthrough.md.resolved` (overwrite mỗi task mới).

---

## ✅ Task 1.1 HOÀN TẤT — `data_pipeline.py` (TechLead APPROVED)

**Date:** 2026-03-05 22:00 UTC+7 | **Branch:** `task-1.1-data-pipeline` → merged `main`

- Tạo `data_pipeline.py`: Class `DataPipeline` OOP, ~650 dòng, type hints + docstrings đầy đủ.
- Kiến trúc: `ThreadPoolExecutor(max_workers=3)` kéo H1/M15/M5 **song song hoàn toàn**; Daemon Thread heartbeat/30s độc lập — không block Main Thread.
- `mt5_reconnect()`: exponential backoff `[1,2,4,8,16,32,60]s` tối đa 10 lần, log mỗi attempt.
- `validate_candles()`: Data Quality Score 0–1 (None/gap/OHLC/volume/count), score <0.60 skip, <0.30 trigger reconnect.
- `is_session_active()`: London 07-12 / NY 13-17 UTC; skip Asian 21-05 UTC.
- `_detect_server_tz()`: auto-detect Exness UTC offset (UTC+2/+3), normalize timestamp về UTC tuyệt đối.
- Tạo `requirements.txt` (3 nhóm: Core Trading / Web Dashboard / ML) + `config/pipeline_config.json`.

---

## 🔄 Task 1.2: `execution.py` — Design Phase (TechLead APPROVED + 3 Fixes)

**Date:** 2026-03-05 22:11 UTC+7 | **Branch:** `task-1.2-execution`

- Thiết kế Class `OrderManager` OOP: `send_order()`, `check_spread()`, `calculate_order_params()`, `validate_order_params()`, `_handle_retcode()`, `_log_trade()`.
- Fill-or-Kill: retry tối đa 3 lần, mỗi lần fetch `bid/ask` realtime → tính lại `Entry/SL/TP` mới.
- TechLead phát hiện **3 lỗi chí mạng** và yêu cầu fix trước khi code: (1) thêm `magic_number` vào TradeRequest, (2) gắn `sl/tp` trực tiếp vào `order_send()` (không OrderModify sau), (3) làm tròn `lot` theo `volume_step` tránh `INVALID_VOLUME`.
- Log lý do từ chối: `SPREAD_TOO_HIGH` / `REQUOTE_MAX` / `TIMEOUT` / `BROKER_LIMIT`.
- Output: `data/trade_log.csv` (ticket, slippage_pips, spread, commission, reject_reason, attempts).

*Phân tích kiến trúc chi tiết: `docs/walkthrough.md.resolved`*

---

## ✅ Task 1.2: `execution.py` — Implementation HOÀN TẤT (TechLead APPROVED)

**Date:** 2026-03-05 22:27 UTC+7 | **Branch:** `task-1.2-execution` → merged `main`

- `execution.py` hoàn chỉnh ~650 dòng: Class `OrderManager`, type hints, docstrings đầy đủ.
- **FIX 1:** Thêm `magic_number=20250305` vào `MqlTradeRequest` — phân biệt lệnh bot vs tay.
- **FIX 2:** `sl/tp` gắn trực tiếp vào `order_send()` (không dùng `ORDER_MODIFY` sau) — ngăn race condition SL/TP.
- **FIX 3:** `_floor_lot()` làm tròn xuống theo `volume_step` bằng `round(..., 8)` — tránh `INVALID_VOLUME`.
- Fill-or-Kill: retry ≤3 lần, mỗi lần fetch `bid/ask` realtime, tính lại `Entry/SL/TP` mới.
- Spread Gate: kiểm tra `spread < MAX_SPREAD_PIPS` trước khi gửi lệnh.
- Output: `data/trade_log.csv` (ticket, slippage_pips, spread, commission, reject_reason, attempts).

---

## ✅ Task 1.3: `risk_manager.py` — Implementation HOÀN TẤT (TechLead APPROVED)

**Date:** 2026-03-05 22:55 UTC+7 | **Branch:** `task-1.3-risk-manager` → merged `main`

- `risk_manager.py` 686 dòng: Class `RiskManager` OOP, type hints, docstrings đầy đủ, `RLock` thread-safe.
- **Position Sizing:** `calculate_lot_size()` + `calculate_sl_distance()` — rủi ro 3%/lệnh, `pip_value_per_lot` qua MT5 API (không hardcode Exness Cent formula).
- **[APPROVED] `check_floating_drawdown(equity)`** — gọi ***liên tục*** từ main loop khi có lệnh mở. Equity < daily_start × (1−6%) → PAUSED ngay + signal Market Close All. Hàng rào TRƯỚC SL.
- **[APPROVED] Early Warning 80%:** Log WARNING khi floating DD ≥ 80% ngưỡng (4.8% của 6%) — cảnh báo sớm trước khi kích hoạt PAUSE.
- **Safety Net 3 lớp:** `_trigger_cooldown()` (4h/streak≥3) / `_trigger_pause()` (đến 00:00 UTC ngày sau) / `_trigger_halt()` (balance ≤ 50% init).
- **initial_balance persistence:** `config/state.json` lưu mốc bền vững; try/except bảo vệ JSON parse lỗi — không crash khi file corrupt.
- **Hot-reload unhalt:** Bot reload `risk_config.json` mỗi 10s; admin sửa `unhalt_timestamp` → HALTED → ACTIVE trong ≤10s, không cần restart.
- `validate_trade()`: 4 checks liên tiếp (bot ACTIVE / lot min / lot max / risk ≤ 3%). Hard Rule max 1 lệnh delegated sang `main.py` State Machine.

---

## ✅ Task 2.1: `strategy_engine.py` — Implementation HOÀN TẤT (TechLead APPROVED)

**Date:** 2026-03-05 23:18 UTC+7 | **Branch:** `task-2.1-strategy-engine` → merged `main`

- `strategy_engine.py` ~550 dòng: Class `StrategyEngine` OOP, type hints, docstrings đầy đủ. 5 vũ khí SMC+VSA trên H1/M15/M5.
- **Anti-Repainting tuyệt đối:** `h1[:-1]`, `m15[:-1]`, `m5[-2]`, `m5[-51:-1]` — không một dòng nào dùng `data[-1]` để tính signal.
- **[APPROVED] FVG Pool (Violated vs Mitigated):** Phân biệt 2 lý do xóa FVG: (1) *Violated* = close xuyên boundary → cấu trúc gãy; (2) *Mitigated* = đã trigger signal → tránh double entry. TTL=48 nến M15, max_size=20 (FIFO).
- **[APPROVED] Pinbar Validation:** wick_ratio ≥ 60%, body_ratio ≤ 35%. BẮt buộc wick phải chạm FVG (low ≤ fvg.top cho BUY, high ≥ fvg.bottom cho SELL).
- **[APPROVED] VSA 2-Layer Filter:** Layer 1 = vol ≥ session_mean × 1.5x (baseline riêng London/NY/Global); Layer 2 = vol > prev_candle.vol × 1.2x (để Climax nổi bật so với nến liền kề).
- **Composite Score:** 40%×pinbar + 35%×vsa + 25%×structure(BOS=1.0/CHoCH=0.7). Gate ≥ 0.55 mới ra signal.
- Entry = close Pinbar M5; SL = ngoài FVG boundary ± 2 pips buffer.

*Phân tích kiến trúc chi tiết (tinh chỉnh Violated/Mitigated + TechLead Q&A 5 câu): `docs/walkthrough.md.resolved`*

---

## 🔄 Task 3.1: `main.py` + `main_config.json` — Implementation (PENDING TechLead snippet review v2)

**Date:** 2026-03-05 23:57 UTC+7 | **Branch:** `task-3.1-main-orchestrator`

- `main.py` ~400 dòng: Class `BotOrchestrator` OOP, 6-state machine (IDLE→SCANNING→SIGNAL_FOUND→PENDING_ORDER→IN_TRADE→CLOSING), graceful SIGTERM shutdown.
- **Multi-symbol:** 7 cặp (`US100, US30, XAUUSD, USOIL, EURUSD, GBPUSD, USDJPY`); `StrategyEngine` một instance per symbol (FVG pool độc lập); `main_config.json` thay `"symbol"` → `"symbols"`.
- **Global Lock Rule:** `open_trade is not None` → không scan symbol mới. 1 lệnh tối đa toàn account, enforce trong `_state_idle()`.
- **[HOTFIX] Candle Sync:** Xóa `elapsed >= 300s`. Thay bằng `mt5.copy_rates_from_pos(sym, M5, 0, 2)` → so sánh `rates[-2]["time"] > last_candle_time[symbol]`. Bắt đúng từng giây khi nến M5 vừa đóng, không bao giờ lệch pha.
- **[HOTFIX] Ghost Order null-safe:** `if orders is None: continue` (thay vì `break`) — mạng lag 1s trả None không làm gián đoạn vòng chờ 30s.
- **[HOTFIX] magic_number từ config:** Xóa hardcode `20250305`. Lưu `self.magic_number = int(cfg.get("magic_number", 20260305))` từ `__init__`, dùng thống nhất khắp file.
- **IN_TRADE poll 0.5s** (`IN_TRADE_POLL_INTERVAL_SEC`); tất cả state khác 1s — bắt floating DD breach sớm hơn.
- **Daily reset Server Time:** `mt5.symbol_info_tick()` timestamp + `_server_tz_offset` (UTC+2/+3 Exness).
- **Ghost Order full flow:** Timeout 30s → poll position (fill muộn → IN_TRADE) → cancel lệnh treo → IDLE.
- `risk_config.json`: `daily_dd_limit 6%→15%`, `balance_floor 50%`.

---

## ✅ Task 3.1: `main.py` State Machine Orchestrator — HOÀN TẤT (TechLead APPROVED)

**Date:** 2026-03-06 00:23 UTC+7 | **Branch:** `task-3.1-main-orchestrator` → merged `main`

- `main.py` **787 dòng** — `BotOrchestrator` v1.0 hoàn chỉnh, 6-state machine (IDLE→SCANNING→SIGNAL_FOUND→PENDING_ORDER→IN_TRADE→CLOSING).
- **[HOTFIX v2] Candle Sync:** `mt5.copy_rates_from_pos(sym, M5, 0, 2)` → so sánh `rates[-2]["time"] > last_candle_time[sym]` — đồng bộ chính xác từng giây khi nến M5 vừa đóng, không bao giờ lệch pha hay re-scan cùng nến.
- **[HOTFIX v2] None API Guard:** `if result is None: continue` trong vòng lặp ghost-order poll — mạng lag trả None không crash vòng chờ 30s.
- **[HOTFIX v2] magic_number từ config:** Xóa hardcode, đọc từ `cfg["magic_number"]`, dùng thống nhất trong `OrderManager` call.
- Multi-symbol 7 cặp, Global Lock Rule (max 1 lệnh), `IN_TRADE_POLL_INTERVAL_SEC=0.5s`, daily reset theo Exness server time.
- Ghost Order full flow: timeout 30s → poll position (fill muộn → IN_TRADE) → cancel lệnh treo → IDLE.
- SIGTERM handler + graceful shutdown qua `stop()`.


---

## 🔄 Task 4.1: `ml_model.py` — Implementation (PENDING TechLead snippet review)

**Date:** 2026-03-06 00:43 UTC+7 | **Branch:** `task-4.1-ml-optimization`

- `ml_model.py` **1045 dòng** — Class `OptimizationEngine` OOP + `run_backtest_fast()` + `_shadow_deploy()` + CLI argparse.
- **Data fetch:** `mt5.copy_rates_range()` → `data/m5_historical.npy` (load 1 lần, cache, tránh disconnect bot live).
- **Shared Memory zero-copy:** `SharedNumpyArray` context manager — 1 numpy array 500MB → 48 worker processes đọc cùng, tiết kiệm 47× RAM; worker attach qua tên shm, không pickle.
- **ProcessPoolExecutor(max_workers=48):** 48 workers × 1 OS process = bypass GIL hoàn toàn. Thread 1 = Bot, Thread 2-5 = DataPipeline, Thread 6 = ML Orchestrator, Thread 7-55 = 48 Optuna Workers.
- **Optuna TPESampler + MedianPruner:** 500 trials, 9D search space (ATR×3, VSA×3, Pinbar×2, Gate×1). SQLite backend (`data/optuna_study.db`) → resume nếu crash.
- **Hàm Objective:** `WR^0.60 × PF^0.40 × dd_penalty` — TrialPruned ngay khi DD ≥ 15% hoặc trade_count < 200.
- **`run_backtest_fast()`:** Vectorized numpy: ATR(14) cumsum, Volume MA20 convolve, Pinbar mask, VSA 2-layer mask, FVG 3-candle vectorized, OHLC worst-case simulation. Ước tính ~25ms/trial.
- **`_shadow_deploy()` Walk-Forward OOS:** Sau khi Optuna hoàn tất, validate best_params trên 24h OOS data chưa thấy. `shadow_pf ≥ active_pf × 0.95` → **PROMOTE** vào `current_settings.json` (FileLock). Ngược lại → **RETIRE**.
- **Shadow versioning:** `config/versions/settings_v{NNN}.json` lưu đầy đủ: train_metrics, oos_metrics, status (promoted/retired), created_at.
- **CLI:** `python ml_model.py [--resume] [--trials N] [--workers N] [--fetch] [--log-level]`
- Tạo `config/ml_config.json` (n_trials=500, n_workers=48, oos=24h, promote=0.95).


---

## 🔄 Task 5.2 — Backend `dashboard.py` Complete (PENDING TechLead snippet review)

**Date:** 2026-03-06 01:23 UTC+7 | **Branch:** `task-5.2-dashboard`

- `dashboard.py` **581 dòng** — FastAPI app, DashboardHub (WebSocket manager), DashboardPublisher (asyncio.Queue bridge), REST endpoints, uvicorn launcher.
- **Architecture (Option B, 1 process):** uvicorn chạy trong daemon thread từ `main.py`. Communication qua `asyncio.Queue.put_nowait()` in-memory — zero blocking cho trading loop.
- **DashboardPublisher:** `call_soon_threadsafe()` → `put_nowait()` — thread-safe bridge từ sync `main.py` sang async FastAPI event loop. Queue đầy → discard silently, bot không bao giờ bị block.
- **DashboardHub:** WebSocket manager, auto-cleanup dead clients, gửi `snapshot` event ngay khi client mới connect.
- **REST endpoints:** `GET /api/candles` (MT5 → Plotly format), `GET /api/trades` (trade_log.csv), `GET /api/status` (shared_state), `GET /api/health`.
- **WS endpoint:** `/ws` — realtime event stream: state_change, candle_close, signal_found, order_filled, equity_update, trade_closed, safety_event.
- **main.py patch:** Import dashboard với try/except fallback (`_NullPub`), `_start_dashboard()` daemon thread, `dashboard_pub.publish()` tại `_transition`, `_state_scanning`, `_state_pending_order`, `_state_in_trade`, `_state_closing`.
- **Chart library:** Plotly.js (per TechLead directive) — FVG box dùng `shapes`, không cần hack.
- **Bind:** `127.0.0.1:8888` — SSH tunnel để access từ ngoài.


---

## 🔄 Task 5.2 — Frontend Complete + Backend Fixed (PENDING TechLead snippet review)

**Date:** 2026-03-06 01:31 UTC+7 | **Branch:** `task-5.2-dashboard`

- **Fix `/api/candles`:** Đọc từ `DataPipeline.get_data(tf)` trước (RAM, zero MT5 call). Convert numpy structured ndarray → Plotly format. Chỉ fallback `mt5.copy_rates_from_pos()` khi pipeline `None`. Response thêm `source: "pipeline_cache"|"mt5_direct"` để debug.
- **`set_pipeline()`:** Inject DataPipeline reference trước khi spawn uvicorn thread. `_pipeline_ref` global được set 1 lần tại `BotOrchestrator._start_dashboard()`.
- **`templates/index.html`** (262 dòng): Dark theme CSS variables, grid 3-row/2-col layout, Plotly.js v2 CDN, 5 side-panel cards (BotState, Equity, DD Gauge, Open Trade, Last Signal), trade history table.
- **`static/js/dashboard.js`** (625 dòng): Plotly candlestick init từ `/api/candles`, volume bars, 7 WS event handlers, `drawFVGBox()` dùng Plotly `layout.shapes` (rectangle), SL/TP horizontal lines (`xref:'paper'`), Entry arrow annotations, WS exponential backoff reconnect.


---

## ✅ Task 5.2 — Dashboard COMPLETE (TechLead fixes applied + MERGED)

**Date:** 2026-03-06 01:41 UTC+7 | **Branch:** `task-5.2-dashboard` → **MERGED main**

**Bug Fixes (TechLead directive):**
- `index.html`: `grid-template-columns: 1fr` → `minmax(0, 1fr)` — ngăn Plotly canvas overflow container.
- `main.py`: `signal_found` payload thêm `fvg_created_time` (ISO UTC string từ `fvg.created_time` hoặc `last_candle_time[symbol]`) — mốc gốc thực của FVG từ quá khứ.
- `dashboard.js`: `case signal_found` dùng `payload.fvg_created_time` làm `x0` của Plotly shape rect — FVG box vẽ đúng gốc, không cụt.

**Tổng hợp anh hàng đã deliver (Task 5.2):**
- `dashboard.py` (644 dòng): FastAPI + WS Hub + DashboardPublisher + REST endpoints
- `templates/index.html` (262 dòng): Grid layout dark theme, Plotly CDN, 5 panel cards
- `static/js/dashboard.js` (625 dòng): Plotly init, drawFVGBox() shapes API, WS 7-event handler, WS auto-reconnect exponential backoff

---

## 🔄 Task 4.2: `ml_model.py` — ML Training Loop Hoàn Thiện

**Date:** 2026-03-06 02:01 UTC+7 | **Branch:** `task-4.2-ml-training`

- **Bổ sung `objective(trial)`** top-level, picklable — không bị ẩn trong factory closure. Khai báo đầy đủ 9 chiều tham số:
  - ATR group: `atr_sl_multiplier` (0.8–3.0), `atr_lot_multiplier` (0.005–0.05), `atr_fvg_buffer` (0.3–1.5)
  - VSA group: `vsa_volume_ratio` (1.2–3.0), `vsa_neighbor_ratio` (1.1–2.0), `vsa_min_score` (0.3–0.6)
  - Pinbar group: `pinbar_wick_ratio` (0.50–0.75), `pinbar_body_ratio` (0.10–0.40)
  - Gate: `composite_score_gate` (0.45–0.75)
- **Score formula:** `WR^0.60 × PF^0.40 × dd_penalty` — `dd_penalty = 1 − (max_dd / max_dd_limit)²`. TrialPruned ngay nếu DD ≥ 15% hoặc trade_count < 200.
- **Bổ sung `run_optimization()`** standalone entry point: `optuna.create_study` SQLite backend (`data/optuna_study.db`, hỗ trợ `--resume`), khởi động `ProcessPoolExecutor(max_workers=48)` → bypass GIL hoàn toàn trên 56 luồng Xeon.
- **`_OPT_CONTEXT`** module-level dict: set bởi `run_optimization()` trước khi study chạy — giữ `objective` picklable cho subprocess workers.

---

## ✅ Task 4.2 FIX: Ask-and-Tell Pattern — Sửa Lỗi Đa Luồng SQLite

**Date:** 2026-03-06 02:15 UTC+7 | **Branch:** `task-4.2-ml-training` → **merged main**

- **Root cause:** `study.optimize()` + SQLite storage + 48 concurrent workers → `database is locked` fatal error.
- **Fix:** Thêm `_run_trial_worker(args: tuple)` — pure numpy worker, không bao giờ chạm SQLite. Pickle-safe hoàn toàn.
- **Ask-and-Tell loop** trong `run_optimization()`: (1) Main thread `study.ask() × batch_size` → (2) `executor.map(_run_trial_worker, worker_args)` → (3) Main thread `study.tell()` từng kết quả. Chỉ main thread chạm DB.
- Xóa dead code cũ (bản `run_optimization()` cũ bị sót sau `return study`).

---

## ✅ Task 6: `backtest_env.py` — Backtest Visualizer HOÀN TẤT

**Date:** 2026-03-06 02:28 UTC+7 | **Branch:** `task-6-backtest-env` → **merged main**

- `backtest_env.py` ~530 dòng: Class `BacktestEnv` OOP, `BacktestReport` + `TradeRecord` dataclasses.
- **`run(data, params)`:** OHLC worst-case simulation + Gaussian slippage model (`gauss(μ=avg_spread, σ=spread_std)`). Metrics: WR, PF, MaxDD, Sharpe, Calmar, avg_win/loss, avg_rrr.
- **`export_trade_log()`:** CSV đầy đủ — 17 cột (trade_id, open/close_time, direction, entry/sl/tp/close_price, close_reason, pnl_raw, commission, slippage, pnl_net, cumulative_pnl, atr_at_entry, winrate_running, pf_running).
- **`generate_html_report()`:** Plotly dark theme 3-row subplot: Row 1 = Equity Curve (line + markers color BUY/SELL); Row 2 = Drawdown % (area filled red + MaxDD annotation); Row 3 = PnL bar chart per trade (green/red). CDN inline, interactive zoom/pan.
- **[BUGFIX]** `path` → `out_file` (biến Path object nhất quán với param `out_path`).
- **[BUGFIX]** `winrate_running`: thay `t.trade_id + 1` bằng counter `trade_count` độc lập — tránh phụ thuộc vào giá trị trade_id khi dùng MT5 Ticket thực.
- **[CONFIRMED]** `name="Equity"`, `name="Drawdown %"`, `name="PnL/Trade"` trong mọi trace — Legend hiển thị đúng.
- CLI: `python backtest_env.py --params config/current_settings.json --out reports/`.




---

## ✅ Task 7: Nâng cấp Data Pipeline — Multi-Symbol 28 File CSV

**Date:** 2026-03-06 03:10 UTC+7 | **Branch:** `feature/parallel-scanner-v3`

### Mục tiêu
Mở rộng hệ thống từ 1 symbol (EURUSDc M5) sang **28 symbol đa dạng** để tránh Overfit vào một thị trường duy nhất — chiến lược "quét rộng, lọc sâu" của Sếp Vũ.

### Thay đổi tham số hút dữ liệu
- **Trước:** `DEFAULT_DAYS = 365` (1 năm) → số nến M15 (~26,272) quá ít cho Optuna 500 trials đủ rổ validate.
- **Sau:** Tăng lên **150,000 nến** (`--bars 150000`) cho mọi khung thời gian — tương đương ~2 năm M5, ~5 năm M15. Đảm bảo ≥ 1,000 rổ lệnh cho mọi asset class sau khi lọc.
- `export_mt5_data.py` — thêm `--bars` CLI flag; `fetch_by_bars()` gọi `mt5.copy_rates_from_pos(sym, TF, 0, n_bars)`.

### Danh sách 28 file CSV đã giải nén thành công trên Xeon

| # | Symbol | TF | Size | Asset Class |
|---|--------|----|------|-------------|
| 1-3 | XAUUSDm, XAGUSDm, XPTUSDm | M15 | 5.6–6.8 MB | METAL |
| 4-5 | BTCUSDm, ETHUSDm | M15 | 8.2–8.8 MB | BTC |
| 6-11 | ADAUSDm, DOGEUSDm, SOLUSDm, XRPUSDm, LINKUSDm, BNBUSDm | M15 | 5.4–8.3 MB | CRYPTO |
| 12-14 | US30m, US500m, USTECm | M5 | 8.1–8.7 MB | INDEX |
| 15 | USOILm | M5 | 7.5 MB | OIL |
| 16-17 | EURUSDm, GBPUSDm | M15 | 6.5–6.9 MB | FOREX-MAJOR |
| 18-28 | EURAUDm, EURGBPm, EURJPYm, GBPAUDm, GBPJPYm, AUDJPYm, AUDUSDm, NZDUSDm, USDCADm, USDCHFm, USDJPYm | M15/M5 | 5.9–8.1 MB | FOREX |

> **Tổng:** 28 file CSV | ~190 MB | Path: `data/history_{SYMBOL}_{TF}.csv`

### Files đã tạo/sửa
- `tools/export_mt5_data.py` — **386 dòng**, Windows-only bridge: `fetch_by_bars()`, `fetch_by_range()`, `save_to_csv()`, `run_export()`, full CLI. Guard `sys.platform != "win32"` + `try/except ImportError MetaTrader5`.
- `data/` — 28 file CSV, schema: `time (Unix int), open, high, low, close, volume`

---

## ✅ Task 8: Kiến trúc Auto-Scanner (Draft) — Phác thảo SOP + Đánh giá Rủi ro

**Date:** 2026-03-06 03:30 UTC+7 | **Branch:** `feature/parallel-scanner-v3`

### Quyết định chiến lược "Tất tay" (All-In)

| Phương án | Mô tả | Thời gian | Rủi ro |
|-----------|-------|-----------|--------|
| A — Tuần tự | 28 file × 500 trials, 1 file/lần | ~14h | Lãng phí 52 luồng |
| B — Soft Cap | Chỉ scan top 7 symbol | ~3.5h | Bỏ sót symbol tiềm năng |
| **C — All-In ✅** | **Tất cả 28 file, parallel tối đa** | **~2-3h** | RAM/CPU cao — chấp nhận được |

**Sếp Vũ quyết định: All-In.** Không Soft Cap. Bao nhiêu mã pass Lưới 1 → bê tất vào Lưới 2.

### SOP Pipeline
```
28 CSV files → [LƯỚI 1: 100 trials, WR > 50%] → ALL mã pass → [LƯỚI 2: 500 trials, WR > 65% + OOS] → An_Latest_Report.md
```

### Các Rủi ro đã phát hiện
- **Race Condition SQLite:** nhiều subprocess cùng 1 DB → `database is locked` → Giải pháp: DB riêng mỗi symbol.
- **Tràn số Score:** OIL/INDEX giá unit cao → PnL hàng tỷ → `pip_value` + `contract_size` cần gắn vào `ASSET_CLASS_CONFIG`.
- **File rỗng:** `len(data) < 50` hoặc `len(pnl_list) < 3` → crash → Guard `raise TrialPruned()`.

---

## ✅ Task 9: Phát triển `tools/auto_scanner.py` — Lưới Lọc 2 Lớp

**Date:** 2026-03-06 04:00 UTC+7 | **Branch:** `feature/parallel-scanner-v3`

`auto_scanner.py` **699 dòng** — Pipeline tự động "Chọn lọc tự nhiên 2 lớp lưới".

### Hằng số cấu hình chính
```python
PARALLEL_JOBS    = 5     # file chạy cùng lúc
WORKERS_PER_JOB  = 10    # workers mỗi file → 5 × 10 = 50 luồng thực

SHALLOW_TRIALS     = 100
SHALLOW_MIN_TRADES = 10
SHALLOW_TIMEOUT    = 600    # 10 phút/file
SHALLOW_GATE_WR    = 0.50   # Lưới 1: WR > 50%

DEEP_TRIALS      = 500
DEEP_TIMEOUT     = 2400   # 40 phút/file
CONFIRM_WR       = 0.65   # Lưới 2: WR > 65% + OOS Pass
```

### Dataclass `ScanResult`
Fields: `csv_path, symbol, tf, asset_class, phase, status, score, winrate, pf, max_dd, trades, duration_sec, oos_status, reject_reason, confirmed`

### Các hàm chính

| Hàm | Mô tả |
|-----|-------|
| `study_db_path(csv_path)` | DB riêng: `data/optuna_{symbol}.db` — tránh SQLite lock |
| `cleanup_shm()` | Dọn `/dev/shm/psm_*` orphans trước mỗi file |
| `parse_output(stdout, stderr)` | Ưu tiên `SCANNER_RESULT: {...}` JSON; fallback regex `Best Score:` |
| `run_one_file(...)` | Subprocess `ml_model.py` 1 file, timeout-safe → `ScanResult` |
| `run_batch_parallel(...)` | `ThreadPoolExecutor(max_workers=jobs)` + `as_completed()` → in real-time |
| `filter_luoi_1(results)` | WR > 50% AND trades ≥ 10 → pass; sort by score desc |
| `validate_luoi_2(deep_results)` | WR > 65% AND OOS ∈ {"✅ Live", "📦 Shadow"} → confirmed |
| `write_report(...)` | Xuất `logs/An_Latest_Report.md`: Master Table + Failed + Shallow + Config |
| `run_pipeline(args)` | Entry: Lưới 1 → Filter → Lưới 2 → Nghiệm thu → Report |

### Lưới 1 — Shallow Scan (WR > 50%)
- **Không giới hạn số lượng mã pass** — tất cả đủ điều kiện đều vào Lưới 2 (chiến lược All-In)
- In kết quả ngay khi từng file hoàn tất (`as_completed`)

### Lưới 2 — Deep Dive (WR > 65% + OOS Pass)
- 500 trials; OOS: `DEFAULT_OOS_HOURS = 720` (1 tháng — 30 ngày chưa thấy)
- Nghiệm thu gắt: `oos_status ∈ {"✅ Live", "📦 Shadow"}` mới được vinh danh

---

## ✅ Task 10: Ép xung Server 56 Luồng — Parallel Execution

**Date:** 2026-03-06 04:30 UTC+7 | **Branch:** `feature/parallel-scanner-v3`

### Vấn đề với kiến trúc tuần tự cũ
Scanner v1/v2 chạy 1 file CSV xong rồi mới đến file kế tiếp. 28 file × 500 trials = **~28 giờ**. Xeon 56 luồng chỉ ~5% công suất.

### Kiến trúc Parallel v3.0
```
ThreadPoolExecutor(max_workers=5)
├─ Thread-1: subprocess ml_model.py (XAUUSDm) → 10 Optuna workers → DB: optuna_XAUUSDm.db
├─ Thread-2: subprocess ml_model.py (BTCUSDm)  → 10 Optuna workers → DB: optuna_BTCUSDm.db
├─ Thread-3: subprocess ml_model.py (EURUSDm)  → 10 Optuna workers → DB: optuna_EURUSDm.db
├─ Thread-4: subprocess ml_model.py (US30m)    → 10 Optuna workers → DB: optuna_US30m.db
└─ Thread-5: subprocess ml_model.py (USOILm)   → 10 Optuna workers → DB: optuna_USOILm.db
     Tổng: 5 × 10 = 50 luồng hoạt động đồng thời (~90% CPU utilization)
```

**Lý do chọn `ThreadPoolExecutor` điều phối subprocess (không phải `ProcessPoolExecutor`):**
- Subprocess isolation — mỗi `ml_model.py` là 1 OS process riêng, GIL không ảnh hưởng
- DB isolation — mỗi subprocess nhận `--study-db riêng` → zero SQLite lock
- RAM efficient — thread chỉ giữ `subprocess.Popen`, không copy data array

### So sánh thời gian

| Phase | Tuần tự (cũ) | Parallel v3.0 |
|-------|-------------|--------------|
| Shallow 28 file × 100t | ~5h | **~1h** |
| Deep ≤28 file × 500t | ~28h | **~3-4h** |
| **Tổng** | **~33h** | **~4-5h** |

---

## ✅ Task 11: Fix Bug Hệ thống — 3 Critical Fixes

**Date:** 2026-03-06 05:00 UTC+7 | **Branch:** `feature/parallel-scanner-v3`

### Bug #1 — Race Condition: SQLite "database is locked"

**Triệu chứng:** 5 subprocess song song cùng `optuna.create_study(storage="sqlite:///data/optuna_study.db")` → `OperationalError: database is locked` → crash.

**Fix — `study_db_path(csv_path)` tách DB riêng mỗi symbol:**
```python
def study_db_path(csv_path: Path) -> Path:
    """DB riêng cho từng file — tuyệt đối không SQLite lock."""
    sym, _ = detect_symbol_tf(csv_path)
    return DATA_DIR / f"optuna_{sym}.db"   # e.g. optuna_XAUUSDm.db
```
Subprocess nhận `--study-db str(db_path)` → 28 DB file độc lập → zero lock.

---

### Bug #2 — Score Tràn Số (355 Tỷ): Thiếu `pip_value` + `contract_size`

**Triệu chứng:** `US30m` (~40,000 points) → PnL 1 lệnh = 400 USD → `profit_factor = 355,000,000,000` → Optuna chọn trial rác.

**Fix — bổ sung `pip_value` + `spread_cost` vào toàn bộ `ASSET_CLASS_CONFIG`:**
```python
"METAL":      { ..., "spread_cost": 0.40,     "pip_value": 1.0  }
"BTC":        { ..., "spread_cost": 5.00,     "pip_value": 1.0  }
"INDEX":      { ..., "spread_cost": 0.03,     "pip_value": 1.0  }  # US30 micro = $1/unit
"OIL":        { ..., "spread_cost": 0.025,    "pip_value": 1.0  }  # revert từ 0.01 → 1.0
"FOREX-MAJOR":{ ..., "spread_cost": 0.00015,  "pip_value": 1.0  }
"FOREX":      { ..., "spread_cost": 0.00015,  "pip_value": 1.0  }
"CRYPTO":     { ..., "spread_cost": 0.001,    "pip_value": 1.0  }
```

**Fix Score cap** trong `run_backtest_fast()`: `profit_factor = min(gross_profit / gross_loss, 999.0)` → tránh PF tỷ tỷ khi n_loss = 0.

**Fix `basket_pnl()`** normalize bằng `pip_value`:
```python
def basket_pnl(cur_price: float) -> float:
    return sum((cur_price - ep) * direction for ep, _ in entries) * pip_value
```

---

### Bug #3 — Ngoại lệ File Data Rỗng (Case XAUUSDm 0 rổ)

**Triệu chứng:** Symbol trả 0 rổ lệnh → `pnl_list = []` → `NaN`/`ZeroDivisionError` → crash subprocess.

**Fix — guard 2 tầng trong `run_backtest_fast()`:**
```python
if len(data) < 50:               # Tầng 1: file quá ít nến
    raise optuna.exceptions.TrialPruned()
# ... simulate ...
if len(pnl_list) < 3:            # Tầng 2: không đủ lệnh sau simulate
    raise optuna.exceptions.TrialPruned()
```

**Fix `run_one_file()`** — kiểm tra file tồn tại:
```python
if not csv_path.exists():
    return ScanResult(..., status="missing", reject_reason="File không tồn tại")
```

---

## ✅ Task 12: Git Flow Compliance — Branch `feature/parallel-scanner-v3`

**Date:** 2026-03-06 05:20 UTC+7

### Files thay đổi trong batch Tasks 7–12
```
RabitScal/
├── tools/
│   ├── auto_scanner.py        ← [NEW] 699 dòng — Auto-Scanner pipeline v3.0
│   └── export_mt5_data.py     ← [NEW] 386 dòng — MT5 Data Exporter (Windows only)
├── ml_model.py                ← [MODIFY] ASSET_CLASS_CONFIG (pip_value + spread_cost)
│                                          run_backtest_fast() guard + PF cap 999
│                                          _objective_worker() nhận spread_cost+pip_value
│                                          DEFAULT_OOS_HOURS = 720
├── data/history_*.csv (×28)  ← [NEW] 28 CSV files, ~190 MB
└── docs/walkthrough.md        ← [MODIFY] Append Tasks 7–12
```

### Tóm tắt commits
| Commit msg | Nội dung |
|-----------|----------|
| `feat(task-7)` | `export_mt5_data.py` — MT5 bridge, 28 symbols, 150K nến/symbol |
| `feat(task-9)` | `auto_scanner.py` v3.0 — 699 lines, ScanResult, Lưới 1+2, All-In |
| `feat(task-10)` | Parallel: `PARALLEL_JOBS=5`, `WORKERS_PER_JOB=10`, `run_batch_parallel()` |
| `fix(task-11-race)` | `study_db_path()`: DB riêng mỗi symbol → zero SQLite lock |
| `fix(task-11-pip)` | `ASSET_CLASS_CONFIG`: `pip_value` + `spread_cost` cho 7 asset classes |
| `fix(task-11-empty)` | Guard `len(data)<50` + `len(pnl)<3` + file-missing check |
| `docs(task-12)` | `walkthrough.md` Tasks 7–12 |

```bash
git checkout -b feature/parallel-scanner-v3
git add tools/auto_scanner.py tools/export_mt5_data.py ml_model.py data/ docs/walkthrough.md
git commit -m "feat: auto-scanner v3.0 + 28 CSV + parallel execution + 3 critical bugfixes"
git push origin feature/parallel-scanner-v3
```

**Trạng thái:** ✅ HOÀN TẤT — nhánh `feature/parallel-scanner-v3` đã push, chờ TechLead review & merge `main`.

---

*Cập nhật bởi Antigravity — 2026-03-06 22:44 UTC+7 | Tasks 7–12 | Branch: feature/parallel-scanner-v3*

---

## 🔄 Milestone 2026-03-07: Giai Đoạn 4 — AI Feature Engineering & Two-Phase Learning

**Date:** 2026-03-07 (cả ngày) | **Branch:** `feature/parallel-scanner-v3`

### Bối cảnh — Tại sao "Đập đi xây lại"?

Sếp Vũ phát hiện hệ thống đang bị drift từ "AI tự học" sang "EA chạy If-Else". Chiến dịch "Trở Về Bản Ngã AI" được phát động: xóa sạch hard constraints, chỉ truyền features (kiến thức), để ML tự quyết định.

### ✅ V12.0–V16.0 — Feature Engine (feature_engine.py — NEW 800 dòng)

- **73 Features** chia 4 nhóm: MTF Structure (F0-17), FVG/OB (F18-35), VSA Volume (F36-55), PA/Traps (F56-72), Session (F71-72)
- BOS/CHoCH bằng body-close (không wick), FVG Pool, EQL/EQH normalize by ATR, Pinbar/Engulfing/7 Market Traps (Ray Wan), Compression squeeze
- Performance: **3 giây/150,000 nến** trên Xeon

### ✅ ml_engine_v2.py — Phase 2 Training Engine (NEW)

- Composite Score: `features @ weights` BLAS vectorized (~1ms/150k)
- Shared Memory: 41.8MB, Walk-Forward IS 70% + OOS 30%
- Security Layers: [S1] Slippage Penalty, [S2] Cooldown Timer (cả hai là Optuna params), [S3] TPE fix
- Overnight run: PID=63906 | 5000 trials | 50 workers

### 🔄 Exness Scalping $200 Config

| Parameter | FTMO (cũ) | Scalping $200 (mới) |
|-----------|-----------|---------------------|
| max_dd | 15% | **28%** |
| min_trades IS | 50 | **1,050** |
| threshold range | [0.1,5.0] | **[0.05,2.0]** |
| rr_ratio | [0.3,5.0] ← giữ nguyên | AI TỰ QUYẾT |
| Fitness | EV×(1-DD×0.4) | **EV×(1+ln(n/1000))×(1-DD×0.3)** |

**Anti-pattern correction:** rr_ratio bị fix cứng [0.5,3.0] → rollback về [0.3,5.0] theo lệnh Sếp Vũ.

### 🔄 Two-Phase Learning Decision (Sếp Vũ — 2026-03-07 22:35)

> "7000 trials là đồ bỏ. Bot đang sinh tồn mù quáng, chưa học được bất kỳ bài học PA hay SMC nào."

- **Phase 1 — Free Exploration:** TẮT DD gate + Trade Count gate. Fitness = `PF × WR × sqrt(n/30) × log_bonus`. AI tự chứng minh Edge.
- **Phase 2 — Business Constraints:** Sau Phase 1, apply KPI (1500 lệnh/năm, DD 28%).

### 📁 Files Changed Today

| File | Action |
|------|--------|
| `feature_engine.py` | NEW — 73 features SMC/VSA/PA |
| `ml_engine_v2.py` | NEW — Training Engine V1.5 + 3 Security Layers |
| `ml_model.py` | MODIFY — Scalping $200 config, fitness formula |
| `tools/morning_report.py` | NEW — TOP 10 report generator |
| `.gitignore` | NEW — Block data/*.csv, *.db, logs/*.log, venv/ |

---

*Cập nhật bởi Antigravity — 2026-03-07 22:47 UTC+7 | Branch: feature/parallel-scanner-v3*

---

## 🧹 Chốt Sổ Phase 2 — Dọn Dẹp Tổng Thể (2026-03-08)

**Date:** 2026-03-08 21:21 UTC+7 | **Author:** Antigravity

### Kết quả Phase 2 Hard Constraint

Thử nghiệm Phase 2 với Hard Constraint (bắt buộc Net Profit > 0 làm điều kiện pruning) đã **THẤT BẠI** do Net Profit âm trên toàn bộ trial space. TPE sampler không học được vì bị pruning quá sớm, không có đủ data điểm dương để hội tụ. **Quyết định chiến lược: Chuyển sang Evolutionary Fitness** — cho phép fitness âm để TPE học gradient từ vùng tối, sau đó shape dần về KPI dương.

### Dọn Dẹp Cây Thư Mục

- **`archive_scripts/`** — Tạo mới, chứa các script version cũ không còn dùng:
  - `ml_engine_phase1.py` (phase 1 legacy)
  - `ml_engine_v2.py` (version cũ trước V3)
- **`logs/`** — Xóa 6 file log rác một lần chạy:
  - `training_overnight_v15.log` (1.1MB)
  - `training_scalp_v2_fresh.log` (550KB)
  - `training_v15_rr_free.log` (450KB)
  - `phase1_overnight.log` (138KB)
  - `phase1_pipeline.log` (54KB)
  - `morning_report_wait.log` (1KB)
- **`docs/Proposal_V11.md`** — Di chuyển từ `logs/` về đúng vị trí `docs/`
- **`.gitignore`** — Cập nhật: chặn hoàn toàn `logs/` directory, `history_*.csv` ở mọi cấp, `*.db` toàn bộ

---

*Cập nhật bởi Antigravity — 2026-03-08 21:21 UTC+7 | Chốt sổ Phase 2*

---

## 📋 SOP: Quy Tắc Ghi Nhật Ký Walkthrough (2026-03-08)

**Date:** 2026-03-08 22:54 UTC+7 | **Áp dụng từ:** ngay lập tức | **Author:** Sếp Vũ (directive) + Antigravity (ghi lại)

### Quy tắc bắt buộc

> Mỗi khi Antigravity **sửa, tạo mới, hoặc xóa** bất kỳ file nào trong dự án, **BẮT BUỘC** phải append một entry vào `docs/walkthrough.md` theo format sau:

```
## [Emoji] [Tên Task] — [Mô tả ngắn] ([YYYY-MM-DD])

**Date:** YYYY-MM-DD HH:MM UTC+7 | **Task:** Task X.Y | **Branch:** `branch-name`

### File thay đổi
- `đường/dẫn/file.py` — [MODIFY/CREATE/DELETE/MOVE] — Mô tả ngắn: sửa gì, vì sao

### Lý do
Giải thích ngắn gọn tại sao phải thay đổi — đủ để người đọc sau này hiểu context.
```

### Tiêu chí "đủ để người đọc hiểu"

Một entry walkthrough đạt yêu cầu khi trả lời được 3 câu hỏi:
1. **Sửa gì?** — File nào, dòng nào, logic gì thay đổi
2. **Vì sao?** — Bug, yêu cầu mới, refactor, hay lệnh từ Sếp
3. **Bối cảnh?** — Nằm trong Task nào, ngày giờ, branch nào

---

## 🏗️ Global Architecture Refactor (2026-03-08)

**Date:** 2026-03-08 22:20 UTC+7 | **Task:** Architecture Refactor | **Branch:** `architecture/global-refactoring`

### Lý do

Sếp Vũ phát hiện vi phạm kiến trúc nghiêm trọng: `run_phase1_pipeline.py`, `run_phase2_pipeline.py` nằm ở root — tư duy "đẻ file run script" không scale được. Lệnh: đập bỏ, chuẩn hóa thành Single Entry Point + phân tầng thư mục rõ ràng.

### Chuẩn bị

- Checkout `main`, tạo nhánh `architecture/global-refactoring`
- Merge `feature/parallel-scanner-v3` vào để có full codebase (21 files, 7127 insertions)

### Thư mục mới tạo

- `core/` — Chứa toàn bộ module chạy với bot live trading
- `engine/` — Chứa ML engine + backtest engine (offline)
- `utils/` — Code dùng chung theo nguyên tắc DRY

### Files di chuyển (git mv — giữ nguyên git history)

| File cũ (root) | File mới | Lý do |
|---------------|----------|-------|
| `data_pipeline.py` | `core/data_pipeline.py` | Phân tầng: core runtime |
| `strategy_engine.py` | `core/strategy_engine.py` | Phân tầng: core runtime |
| `feature_engine.py` | `core/feature_engine.py` | Phân tầng: core runtime |
| `execution.py` | `core/execution.py` | Phân tầng: core runtime |
| `risk_manager.py` | `core/risk_manager.py` | Phân tầng: core runtime |
| `dashboard.py` | `core/dashboard.py` | Phân tầng: core runtime |
| `ml_model.py` | `engine/ml_engine.py` | Tên rõ nghĩa hơn + phân tầng offline |
| `backtest_env.py` | `engine/backtest_engine.py` | Tên rõ nghĩa hơn + phân tầng offline |
| `run_phase1_pipeline.py` | `engine/pipeline_phase1.py` | Không còn file run_ ở root |
| `run_phase2_pipeline.py` | `engine/pipeline_phase2.py` | Không còn file run_ ở root |

### Files mới tạo

- `utils/logger.py` — [CREATE] — Gộp `_build_logger()` đang bị copy y hệt ở 3 file (`ml_model.py`, `backtest_env.py`, `main.py`). Vi phạm DRY.
- `utils/data_loader.py` — [CREATE] — Gộp `load_ohlcv_from_csv()` đang có 2 implementation khác nhau trong `backtest_engine.py` và `ml_engine.py`.
- `quant_main.py` — [CREATE] — Cổng ML/Quant duy nhất với 3 subcommands: `train`, `backtest`, `report`. Thay thế toàn bộ các file `run_*.py` bằng argparse dispatcher.
- `PROJECT_STRUCTURE.md` — [CREATE] — Bản đồ thư mục chuẩn để đối chiếu khi tạo file mới.

### Files sửa import

- `main.py` — [MODIFY] — Cập nhật 5 import paths sau khi core modules di chuyển vào `core/`:
  - `from data_pipeline` → `from core.data_pipeline`
  - `from execution` → `from core.execution`
  - `from risk_manager` → `from core.risk_manager`
  - `from strategy_engine` → `from core.strategy_engine`
  - `from dashboard` → `from core.dashboard`

### Git Commits

```
8d52108  refactor: global restructure, establish quant_main.py and resolve DRY issues
5739fd6  docs: add PROJECT_STRUCTURE.md — canonical folder map for reference
```

---

*Cập nhật bởi Antigravity — 2026-03-08 22:54 UTC+7 | Branch: architecture/global-refactoring*

---

## 📋 SOP: Phân Tích Trước Khi Thực Thi (2026-03-08)

**Date:** 2026-03-08 22:56 UTC+7 | **Áp dụng từ:** ngay lập tức | **Directive:** Sếp Vũ

### Quy tắc bắt buộc

> Mỗi khi nhận prompt nhiệm vụ, **TRƯỚC KHI BẤM CÒ**, Antigravity **BẮT BUỘC** phải:

**Bước 1 — Phân tích:**
- Đọc hiểu yêu cầu thực sự (không chỉ nghĩa đen)
- Xác định scope: files bị ảnh hưởng, dependencies ẩn, rủi ro
- Chỉ rõ điểm mơ hồ cần làm rõ (nếu có)

**Bước 2 — Đề xuất phương án tối ưu (nếu có):**
- Nếu có cách làm tốt hơn yêu cầu gốc (ít rủi ro hơn, sạch hơn, nhanh hơn) → trình bày rõ:
  - **Phương án A** (theo yêu cầu gốc): ưu/nhược
  - **Phương án B** (tối ưu đề xuất): ưu/nhược + lý do tốt hơn
- Nếu không có phương án tốt hơn → xác nhận sẽ thực hiện đúng yêu cầu

**Bước 3 — Xác nhận trước khi làm:**
- Nếu phương án tối ưu **khác đáng kể** so với yêu cầu gốc → hỏi Sếp chọn phương án nào
- Nếu phương án tối ưu chỉ là **cải tiến nhỏ trong cùng hướng** → tự làm và ghi chú lại

### Ví dụ đúng

```
Sếp giao: "Tạo run_phase3_pipeline.py"
Em phân tích: Tạo thêm file run_*.py ở root vi phạm Single Entry Point đã thỏa thuận.
Em đề xuất:
  - Phương án A (theo yêu cầu): tạo run_phase3_pipeline.py ở root
  - Phương án B (tối ưu): tạo engine/pipeline_phase3.py + đăng ký vào quant_main.py
Em hỏi: "Sếp chọn phương án nào?"
```

---

*Cập nhật bởi Antigravity — 2026-03-08 22:57 UTC+7 | Branch: architecture/global-refactoring*

---

## ✅ Ev05d: $200 Live Rehearsal — Hybrid Risk Engine (HOÀN TẤT)

**Date:** 2026-03-12 → 2026-03-13 | **Branch:** `ev05d` → **MERGED `main`** | **Author:** Antigravity

---

### 📋 Tóm tắt

Mẻ kiểm định cuối cùng "Ev05d — The $200 Live Rehearsal" theo chỉ đạo Sếp Vũ. Triển khai cơ chế Quản lý vốn Hybrid (Fixed lot XAU + Dynamic 3% còn lại) trên vốn $200, đội hình 5 mã tinh nhuệ (xóa XAGUSDm). Training 15 studies × 2000 trials = 30,000 kịch bản. **Kết quả: 15/15 studies ALL PASS OOS gate + ALL profitable.**

---

### 📁 Nhật ký thay đổi files

#### [NEW] `engine/pipeline_ev05d.py` (615 dòng)
- Pipeline Ev05d kế thừa Ev05c core (73 features, H1 trend filter, FVG Split Ticket)
- **Hybrid lot sizing:** XAUUSDm fixed 0.01 / Others dynamic 3% ($6/trade max loss)
- `calc_lot_size_ev05d()`: Margin check + risk-based sizing, chặn sàn `min_lot=0.01`
- `run_backtest_ev05d()`: Backtest engine + Inside Bar force close + Global Risk Gate 3 lệnh
- `_ev05d_worker()`: Subprocess worker cho multiprocessing Optuna
- `run_ev05d_queue()`: Orchestrator chạy 15 studies tuần tự (5 symbols × 3 TFs)
- Capital: `INITIAL_CAPITAL = 200.0`, `RISK_PER_TRADE = 0.03`, `MAX_MARGIN_PCT = 0.30`

#### [MODIFY] `core/symbol_registry.py`
- **Dòng 150:** Sửa BTCUSDm `min_lot` từ `0.001` → `0.01` (theo yêu cầu user)
- **Dòng 263-270:** Thêm `ELITE_5_SYMBOLS` list — 5 mã tinh nhuệ (xóa XAGUSDm):
  ```python
  ELITE_5_SYMBOLS = ["XAUUSDm", "US30m", "USTECm", "BTCUSDm", "ETHUSDm"]
  ```

#### [MODIFY] `quant_main.py`
- **Dòng 191-210:** Thêm `elif args.phase == 8:` block cho Ev05d pipeline
  - Import `run_ev05d_queue` from `engine.pipeline_ev05d`
  - Set `multiprocessing.set_start_method("spawn", force=True)`
- **Dòng ~330:** Cập nhật `choices` trong argparse: thêm `8` vào danh sách phases
- **Dòng ~335:** Cập nhật help text: `"8=Ev05d $200 Live Rehearsal"`

#### [NEW] `launch_ev05d.sh` (29 dòng)
- Script bash tự động:
  1. Dọn sạch DB cũ (`rm -f data/optuna_ev05c_*.db data/optuna_ev05d_*.db`)
  2. Launch Ev05d training nền (`nohup ... > /tmp/ev05d_train.log 2>&1 &`)
  3. Log PID + monitor instructions

#### [MODIFY] `logs/An_Latest_Report.md` (104 dòng — overwritten bởi report script)
- Report hoàn chỉnh 15 studies với:
  - Bảng Phong Thần (Total/CPL/PRN/WR/PF/DD/OOS PnL/Fitness/Status)
  - Top 5 Chiến Thần (US30_M5 #1, fitness=10983.94)
  - Phân tích theo Symbol (5 symbols, ALL 🟢)
  - Portfolio Summary + Nhận xét + Đề xuất Ev05e

#### [NEW] `logs/ev05d_fact_sheet.md` (36 dòng)
- Template fact sheet ban đầu (placeholder từ Ev05c era, chưa cập nhật)

#### [NEW — 15 files] `data/optuna_ev05d_*.db`
- 15 SQLite DBs (30-45MB mỗi file) chứa toàn bộ trials:
  - `optuna_ev05d_xauusd_{m1,m5,m15}.db`
  - `optuna_ev05d_us30_{m1,m5,m15}.db`
  - `optuna_ev05d_ustec_{m1,m5,m15}.db`
  - `optuna_ev05d_btcusd_{m1,m5,m15}.db`
  - `optuna_ev05d_ethusd_{m1,m5,m15}.db`

#### [TEMP — không commit] `/tmp/ev05d_report_v2.py`
- Script query 15 DBs, fix bug state='COMPLETE' (string) thay vì state=1 (int)
- Xuất console table + `logs/An_Latest_Report.md` + `/tmp/ev05d_results.json`

---

### 📊 Kết quả Training (15/15 PASS)

| Metric | Value |
|--------|-------|
| Total trials | 55,564 |
| Completed | 33,995 (61.2%) |
| Pruned | 21,569 |
| OOS PASS | **15/15** |
| OOS Profitable | **15/15** |
| Best fitness | 10,983.94 (US30_M5) |
| Avg WR (all) | 60.8% |
| Avg PF (all) | 2.42 |

**Top 3 Chiến Thần:**
1. 🥇 US30_M5 — Fitness: 10983.94 | WR: 58.5% | PF: 2.25 | OOS PnL: $+294,845
2. 🥈 US30_M15 — Fitness: 9598.26 | WR: 62.3% | PF: 2.56 | OOS PnL: $+203,183
3. 🥉 US30_M1 — Fitness: 7317.02 | WR: 58.2% | PF: 2.41 | OOS PnL: $+184,080

---

### 🐛 Bug Fix quan trọng

**Report Script State Bug:** Optuna v4 RDB storage lưu trial state dạng **string** `'COMPLETE'`/`'PRUNED'` thay vì integer `1`/`2` (Optuna v3). Report script cũ query `state=1` → 0 kết quả → hiển thị sai "0 completed trials". Fix: đổi thành `state='COMPLETE'`.

---

### ⚙️ Quyết định kỹ thuật quan trọng

1. **Hybrid Lot Sizing:** XAUUSDm fixed lot 0.01 vì pip value lớn ($0.10/pip/0.01lot × ATR ~300 pips = max loss ~$30 nếu dùng dynamic → quá nguy hiểm cho $200). Các mã khác dùng đ 3% risk = max $6/lệnh.
2. **XAGUSDm loại bỏ:** Hiệu suất kém trong Ev05c, Sếp Vũ quyết định loại khỏi đội hình.
3. **BTCUSDm min_lot:** Sửa từ 0.001 → 0.01 theo thông số sàn Exness thực tế.
4. **Multiprocessing spawn:** Bắt buộc dùng `set_start_method("spawn")` cho subprocess worker Optuna trên Linux (tránh CUDA/file lock issues với fork).

---

*Cập nhật bởi Antigravity — 2026-03-13 09:33 UTC+7 | Branch: ev05d → merged main*
