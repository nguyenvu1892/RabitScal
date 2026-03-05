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

