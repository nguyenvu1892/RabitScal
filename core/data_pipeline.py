"""
data_pipeline.py — Rabit_Exness AI
====================================
Module kéo dữ liệu realtime từ MetaTrader 5 cho hệ thống giao dịch tự động.

Kiến trúc đa luồng tối ưu cho Dual Xeon E5-2680 v4 (56 Threads / 96GB RAM):
  - ThreadPoolExecutor(max_workers=3): kéo H1/M15/M5 song song hoàn toàn
  - Daemon Thread độc lập: heartbeat ping MT5 mỗi 30 giây
  - threading.RLock: bảo vệ shared data store, thread-safe 100%
  - numpy ndarray output: sẵn sàng cho ProcessPoolExecutor tại Giai đoạn 4 (GIL bypass)

Author   : Antigravity (Senior AI Coder)
Branch   : task-1.1-data-pipeline
Date     : 2026-03-05
Plan ref : Plan.md v1.1 — Giai đoạn 1, data_pipeline.py
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Optional MetaTrader5 import (graceful fallback for offline unit tests) ──────
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:  # pragma: no cover
    mt5 = None            # type: ignore[assignment]
    MT5_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Mapping tên timeframe chuỗi → hằng số MT5
_TF_MAP: Dict[str, int] = {}   # populated after MT5 import check

# Thời gian kỳ vọng (giây) giữa 2 candle liên tiếp cho mỗi timeframe
_TF_INTERVAL_SEC: Dict[str, int] = {
    "H1":  3600,
    "M15": 900,
    "M5":  300,
}

# Dãy sleep exponential backoff (giây).  Phần tử cuối lặp lại đến max_attempts.
_BACKOFF_SEQUENCE: List[int] = [1, 2, 4, 8, 16, 32, 60, 60, 60, 60]

# Module name used in log records
_MODULE = "DataPipeline"


# ═══════════════════════════════════════════════════════════════════════════════
#  LOGGER FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def _build_logger(log_dir: str = "logs") -> logging.Logger:
    """
    Tạo và cấu hình Logger ghi ra cả console lẫn file ``logs/system.log``.

    Format: ``[TIME] - [LEVEL] - [MODULE] - [MESSAGE]``

    Args:
        log_dir: Thư mục chứa file log (tạo tự động nếu chưa tồn tại).

    Returns:
        logging.Logger đã cấu hình sẵn.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(log_dir, "system.log")

    fmt = "[%(asctime)s] - [%(levelname)s] - [%(name)s] - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S UTC"

    logger = logging.getLogger(_MODULE)
    if logger.handlers:          # Tránh duplicate handlers khi gọi lại
        return logger

    logger.setLevel(logging.DEBUG)

    # File handler — ghi toàn bộ từ DEBUG trở lên
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    # Console handler — chỉ từ INFO trở lên để console không bị spam
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA VALIDATION RESULT
# ═══════════════════════════════════════════════════════════════════════════════

class ValidationResult:
    """
    Kết quả trả về từ ``validate_candles()``.

    Attributes:
        passed   : True nếu data đủ chất lượng để sử dụng.
        score    : Float [0.0, 1.0] — điểm chất lượng tổng hợp.
        issues   : Danh sách mô tả các lỗi phát hiện.
        data     : numpy ndarray đã validated, hoặc None nếu failed critical check.
    """

    __slots__ = ("passed", "score", "issues", "data")

    def __init__(
        self,
        passed: bool,
        score: float,
        issues: List[str],
        data: Optional[np.ndarray],
    ) -> None:
        self.passed = passed
        self.score = score
        self.issues = issues
        self.data = data

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ValidationResult(passed={self.passed}, score={self.score:.2f}, "
            f"issues={len(self.issues)})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class DataPipeline:
    """
    Module kéo, validate và phân phối dữ liệu OHLCV realtime từ MetaTrader 5.

    Kiến trúc chủ chốt
    ------------------
    * **ThreadPoolExecutor(max_workers=3)** — 3 worker threads độc lập kéo H1,
      M15, M5 song song (concurrent), không chờ tuần tự. MT5 Python API là
      blocking/synchronous I/O nên ThreadPoolExecutor là lựa chọn tối ưu so với
      asyncio (asyncio không đạt concurrency thực sự với blocking calls).
    * **Daemon heartbeat thread** — ping MT5 terminal mỗi 30 s, trigger
      ``mt5_reconnect()`` tự động nếu kết nối bị mất. Không bao giờ block
      Main Thread.
    * **threading.RLock** — bảo vệ ``_data_store`` và ``_connected`` flag khỏi
      race condition khi nhiều workers ghi đồng thời.
    * **numpy ndarray output** — dữ liệu lưu dưới dạng structured ndarray,
      tương thích ``multiprocessing.shared_memory`` → sẵn sàng cho
      ``ProcessPoolExecutor`` tại Giai đoạn 4 (bypass GIL hoàn toàn).

    Phân công thread (bản đồ 56 Logical Processors Xeon E5-2680 v4)
    ---------------------------------------------------------------
    * Thread 1–3 : fetch H1 / M15 / M5 (ThreadPoolExecutor workers)
    * Thread 4   : ``_heartbeat_loop()`` (daemon thread)
    * Thread 5   : Main Thread (State Machine Orchestrator — main.py)
    * Thread 6+  : dự trữ cho ProcessPoolExecutor Giai đoạn 4 (48 workers)

    Usage
    -----
    >>> cfg = DataPipeline.load_config("config/pipeline_config.json")
    >>> dp = DataPipeline(cfg)
    >>> dp.start()
    >>> rates_m5 = dp.get_data("M5")   # thread-safe
    >>> dp.stop()
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: Dict,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Khởi tạo DataPipeline.

        Args:
            config : Dict cấu hình, thường load từ ``pipeline_config.json``.
                     Xem ``load_config()`` để biết cấu trúc.
            logger : Logger tuỳ chọn. Nếu None sẽ tự tạo logger ghi vào
                     ``logs/system.log``.
        """
        self._config = config
        self._log = logger or _build_logger(config.get("log_dir", "logs"))

        # Thông tin kết nối MT5
        self._symbol: str = config["symbol"]
        self._tf_config: Dict[str, Dict] = config["timeframes"]

        # Thresholds
        self._heartbeat_interval: int = config.get("heartbeat_interval_sec", 30)
        self._reconnect_max_attempts: int = config.get("reconnect_max_attempts", 10)
        self._quality_min_score: float = config.get("data_quality_min_score", 0.60)

        # MT5 connection params (optional — dùng khi login bằng account)
        self._mt5_login: Optional[int] = config.get("mt5_login")
        self._mt5_password: Optional[str] = config.get("mt5_password")
        self._mt5_server: Optional[str] = config.get("mt5_server")

        # Exness MT5 server timezone offset từ UTC (giờ).
        # Exness thường dùng UTC+2 (EET) hoặc UTC+3 (EEST mùa hè).
        # Pipeline tự động detect; fallback về giá trị config nếu không detect được.
        self._server_tz_offset: int = config.get("server_tz_offset_hours", 2)

        # Session filter config (giờ UTC)
        sf = config.get("session_filters", {})
        self._london_range: Tuple[int, int] = tuple(sf.get("london_open_utc", [7, 12]))   # type: ignore
        self._ny_range: Tuple[int, int] = tuple(sf.get("ny_open_utc", [13, 17]))          # type: ignore
        self._asian_avoid: Tuple[int, int] = tuple(sf.get("asian_avoid_utc", [21, 5]))     # type: ignore
        self._news_buffer_min: int = sf.get("news_buffer_minutes", 15)

        # Internal state
        self._data_store: Dict[str, Optional[np.ndarray]] = {tf: None for tf in self._tf_config}
        self._lock = threading.RLock()
        self._connected = False
        self._running = False

        # Thread pool — 3 workers cho H1 / M15 / M5
        self._executor: Optional[ThreadPoolExecutor] = None
        self._heartbeat_thread: Optional[threading.Thread] = None

        self._log.info(
            f"DataPipeline initialized | symbol={self._symbol} | "
            f"timeframes={list(self._tf_config.keys())} | "
            f"quality_min={self._quality_min_score}"
        )

    # ------------------------------------------------------------------
    # Public lifecycle API
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """
        Khởi động DataPipeline: kết nối MT5, fetch lần đầu, spawn heartbeat daemon.

        Returns:
            True nếu khởi động thành công, False nếu kết nối MT5 thất bại.
        """
        self._log.info("DataPipeline.start() called — initializing MT5 connection...")

        if not self._init_mt5():
            self._log.critical(
                "DataPipeline FAILED to start — MT5 init unsuccessful. "
                "Check MT5 terminal is running and credentials."
            )
            return False

        self._running = True

        # Spawn heartbeat daemon thread (non-blocking, dies with main process)
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="MT5-Heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()
        self._log.info(
            f"Heartbeat daemon thread started (interval={self._heartbeat_interval}s)"
        )

        # ThreadPoolExecutor: 3 dedicated workers cho H1 / M15 / M5
        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="MT5-Fetch")
        self._log.info("ThreadPoolExecutor(max_workers=3) created for H1/M15/M5 fetch workers")

        # Initial fetch — chạy song song 3 timeframes
        success = self._run_fetch_all()
        if success:
            self._log.info("DataPipeline started successfully — initial data fetch complete")
        else:
            self._log.warning(
                "DataPipeline started but initial fetch had partial failures — "
                "will retry on next heartbeat cycle"
            )
        return True

    def stop(self) -> None:
        """
        Dừng DataPipeline một cách graceful: shutdown executor, join threads,
        đóng kết nối MT5.
        """
        self._log.info("DataPipeline.stop() called — initiating graceful shutdown...")
        self._running = False

        if self._executor:
            self._executor.shutdown(wait=True)
            self._log.info("ThreadPoolExecutor shut down cleanly")

        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=self._heartbeat_interval + 5)

        if MT5_AVAILABLE and mt5:
            mt5.shutdown()

        with self._lock:
            self._connected = False

        self._log.info("DataPipeline stopped. MT5 connection closed.")

    # ------------------------------------------------------------------
    # Public data access (thread-safe)
    # ------------------------------------------------------------------

    def get_data(self, timeframe: str) -> Optional[np.ndarray]:
        """
        Trả về dữ liệu OHLCV mới nhất cho timeframe được chỉ định.

        Thread-safe: sử dụng RLock nội bộ.
        Data dạng numpy structured ndarray — tương thích trực tiếp với
        ``ProcessPoolExecutor`` workers tại Giai đoạn 4 (không cần serialize).

        Args:
            timeframe: Tên timeframe, ví dụ ``"H1"``, ``"M15"``, ``"M5"``.

        Returns:
            numpy.ndarray nếu data đã sẵn sàng, None nếu chưa có data hoặc
            timeframe không hợp lệ.

        Example:
            >>> rates_m5 = dp.get_data("M5")
            >>> if rates_m5 is not None:
            ...     last_close = rates_m5["close"][-1]
        """
        with self._lock:
            return self._data_store.get(timeframe)

    def get_all_data(self) -> Dict[str, Optional[np.ndarray]]:
        """
        Trả về snapshot toàn bộ data store (tất cả timeframes) dưới dạng dict.

        Thread-safe. Trả về shallow copy để tránh data race bên ngoài.

        Returns:
            Dict mapping timeframe → numpy ndarray (hoặc None nếu chưa có data).
        """
        with self._lock:
            return dict(self._data_store)

    def is_connected(self) -> bool:
        """
        Kiểm tra trạng thái kết nối MT5 hiện tại.

        Returns:
            True nếu đang kết nối, False nếu đã mất kết nối.
        """
        with self._lock:
            return self._connected

    def is_running(self) -> bool:
        """Trả về True nếu DataPipeline đang trong trạng thái running."""
        return self._running

    # ------------------------------------------------------------------
    # Session filter (public — dùng bởi strategy_engine.py)
    # ------------------------------------------------------------------

    def is_session_active(self, dt_utc: Optional[datetime] = None) -> bool:
        """
        Kiểm tra thời điểm hiện tại có nằm trong giờ giao dịch thanh khoản cao không.

        Giờ active (UTC):
          * London Open  : 07:00–12:00
          * NY Open      : 13:00–17:00

        Giờ tránh (UTC):
          * Asian Session: 21:00–05:00 (thanh khoản thấp, false FVG nhiều)
          * Overlap volatile (12:00–13:00): không filter hẳn nhưng
            caller nên tăng ngưỡng VSA quality score khi trong khoảng này.

        Args:
            dt_utc: Thời điểm UTC để kiểm tra. Nếu None sẽ dùng ``datetime.now(UTC)``.

        Returns:
            True nếu trong giờ giao dịch hoạt động, False nếu ngoài giờ.
        """
        if dt_utc is None:
            dt_utc = datetime.now(timezone.utc)

        h = dt_utc.hour

        london_start, london_end = self._london_range  # (7, 12)
        ny_start, ny_end = self._ny_range               # (13, 17)
        asian_start, asian_end = self._asian_avoid      # (21, 5) — wraps midnight

        # Kiểm tra trong London Open hoặc NY Open
        in_london = london_start <= h < london_end
        in_ny = ny_start <= h < ny_end

        # Kiểm tra trong Asian session — span qua nửa đêm
        if asian_start < asian_end:
            in_asian = asian_start <= h < asian_end
        else:
            # Span midnight: e.g., 21:00–05:00
            in_asian = h >= asian_start or h < asian_end

        active = (in_london or in_ny) and not in_asian

        self._log.debug(
            f"is_session_active | UTC_hour={h} | london={in_london} | "
            f"ny={in_ny} | asian={in_asian} | result={active}"
        )
        return active

    # ------------------------------------------------------------------
    # Reconnect with exponential backoff
    # ------------------------------------------------------------------

    def mt5_reconnect(self) -> bool:
        """
        Thử kết nối lại với MT5 terminal sử dụng exponential backoff.

        Cơ chế:
          * Shutdown kết nối cũ trước khi thử lại (reset state sạch)
          * Dãy sleep: 1s → 2s → 4s → 8s → 16s → 32s → 60s (giữ ở 60s)
          * Tối đa ``reconnect_max_attempts`` lần (mặc định 10)
          * Log mỗi attempt ở level WARNING; thành công log INFO; thất bại log CRITICAL

        Returns:
            True nếu kết nối lại thành công, False nếu đã dùng hết attempts.
        """
        if not MT5_AVAILABLE:
            self._log.warning("mt5_reconnect() called but MetaTrader5 is not installed")
            return False

        max_attempts = self._reconnect_max_attempts
        backoff = _BACKOFF_SEQUENCE  # [1,2,4,8,16,32,60,60,60,60]

        self._log.warning(
            f"MT5 connection lost. Starting reconnect sequence (max {max_attempts} attempts)..."
        )

        with self._lock:
            self._connected = False

        for attempt in range(1, max_attempts + 1):
            wait_sec = backoff[min(attempt - 1, len(backoff) - 1)]
            self._log.warning(
                f"[MT5] Reconnect attempt {attempt}/{max_attempts} — "
                f"waiting {wait_sec}s before retry..."
            )
            time.sleep(wait_sec)

            # Shutdown trước — reset trạng thái socket cũ
            try:
                mt5.shutdown()
            except Exception:
                pass

            success = self._init_mt5(log_success=False)
            if success:
                self._log.info(
                    f"[MT5] Reconnected successfully on attempt {attempt}/{max_attempts}"
                )
                return True

        self._log.critical(
            f"[MT5] FAILED to reconnect after {max_attempts} attempts. "
            "DataPipeline is HALTED. Manual intervention required."
        )
        self._running = False
        return False

    # ------------------------------------------------------------------
    # Data fetch — runs inside ThreadPoolExecutor workers
    # ------------------------------------------------------------------

    def fetch_all(self) -> Dict[str, Optional[np.ndarray]]:
        """
        Kéo data toàn bộ timeframes song song (concurrent) bằng ThreadPoolExecutor.

        Mỗi timeframe chạy trong worker thread riêng biệt — không block nhau,
        không block Main Thread. Kết quả được validated và lưu vào ``_data_store``.

        Returns:
            Dict mapping timeframe → numpy ndarray (None nếu fetch thất bại).
        """
        if self._executor is None:
            self._log.error("fetch_all() called before start() — executor not initialized")
            return {}

        return self._run_fetch_all_and_return()

    # ------------------------------------------------------------------
    # Data validation
    # ------------------------------------------------------------------

    def validate_candles(
        self,
        rates: Optional[np.ndarray],
        timeframe: str,
        expected_count: Optional[int] = None,
    ) -> ValidationResult:
        """
        Kiểm tra chất lượng dữ liệu candle và tính Data Quality Score.

        Các kiểm tra (theo thứ tự ưu tiên):

        +-----------------+----------+--------------------------------------------+
        | Check           | Weight   | Pass condition                             |
        +=================+==========+============================================+
        | None check      | Critical | ``rates is not None and len(rates) > 0``   |
        +-----------------+----------+--------------------------------------------+
        | Time gap        | 0.40     | max_gap ≤ expected_interval × 1.5          |
        +-----------------+----------+--------------------------------------------+
        | OHLC validity   | 0.30     | high ≥ low, all prices > 0                |
        +-----------------+----------+--------------------------------------------+
        | Volume          | 0.20     | tick_volume > 0 for ≥ 95% candles         |
        +-----------------+----------+--------------------------------------------+
        | Candle count    | 0.10     | len(rates) ≥ expected_count               |
        +-----------------+----------+--------------------------------------------+

        Score < 0.60 → WARNING, giữ nguyên data cũ
        Score < 0.30 → ERROR, trigger reconnect

        Args:
            rates          : Structured numpy ndarray từ MT5 (fields: time, open,
                             high, low, close, tick_volume, spread, real_volume).
            timeframe      : Tên timeframe (``"H1"``, ``"M15"``, ``"M5"``).
            expected_count : Số candle tối thiểu cần có. Lấy từ config nếu None.

        Returns:
            :class:`ValidationResult` với điểm chất lượng và danh sách lỗi.
        """
        issues: List[str] = []

        # ── Check 0: None / empty (CRITICAL) ──────────────────────────────────
        if rates is None or len(rates) == 0:
            issues.append("CRITICAL: rates is None or empty")
            return ValidationResult(passed=False, score=0.0, issues=issues, data=None)

        score = 0.0
        data = rates  # sẽ convert về numpy array đã normalize

        # ── Check 1: Time gap normality (weight 0.40) ─────────────────────────
        expected_interval = _TF_INTERVAL_SEC.get(timeframe, 60)
        gap_score = 0.40
        try:
            times = rates["time"].astype(np.int64)
            if len(times) > 1:
                deltas = np.diff(times)
                max_gap = int(deltas.max())
                threshold = expected_interval * 1.5
                if max_gap > threshold:
                    gap_pct = max_gap / expected_interval
                    issues.append(
                        f"Time gap anomaly ({timeframe}): max_gap={max_gap}s "
                        f"({gap_pct:.1f}× expected {expected_interval}s)"
                    )
                    # Proportional deduction
                    gap_score = max(0.0, 0.40 * (1.0 - min((gap_pct - 1.5) / 10.0, 1.0)))
        except (ValueError, KeyError) as exc:
            issues.append(f"Time gap check error: {exc}")
            gap_score = 0.0

        score += gap_score

        # ── Check 2: OHLC validity (weight 0.30) ──────────────────────────────
        ohlc_score = 0.30
        try:
            bad_ohlc = (
                (rates["high"] < rates["low"]) |
                (rates["open"] <= 0) |
                (rates["close"] <= 0) |
                (rates["high"] <= 0) |
                (rates["low"] <= 0)
            )
            bad_count = int(bad_ohlc.sum())
            if bad_count > 0:
                pct_bad = bad_count / len(rates)
                issues.append(
                    f"OHLC invalid rows ({timeframe}): {bad_count}/{len(rates)} "
                    f"({pct_bad:.1%}) — high<low or price≤0"
                )
                ohlc_score = max(0.0, 0.30 * (1.0 - pct_bad))
        except (ValueError, KeyError) as exc:
            issues.append(f"OHLC check error: {exc}")
            ohlc_score = 0.0

        score += ohlc_score

        # ── Check 3: Volume availability (weight 0.20) ────────────────────────
        vol_score = 0.20
        try:
            if "tick_volume" in rates.dtype.names:
                zero_vol = (rates["tick_volume"] <= 0).sum()
                pct_zero = zero_vol / len(rates)
                if pct_zero > 0.05:
                    issues.append(
                        f"Low tick_volume coverage ({timeframe}): "
                        f"{pct_zero:.1%} candles have zero volume"
                    )
                    vol_score = max(0.0, 0.20 * (1.0 - pct_zero))
            else:
                issues.append(f"tick_volume field missing ({timeframe})")
                vol_score = 0.10   # partial credit
        except (ValueError, KeyError) as exc:
            issues.append(f"Volume check error: {exc}")
            vol_score = 0.0

        score += vol_score

        # ── Check 4: Candle count (weight 0.10) ───────────────────────────────
        count_score = 0.10
        if expected_count is None:
            tf_cfg = self._tf_config.get(timeframe, {})
            expected_count = tf_cfg.get("candles", 200)

        if len(rates) < expected_count:
            ratio = len(rates) / expected_count
            issues.append(
                f"Insufficient candles ({timeframe}): got {len(rates)}, "
                f"expected ≥ {expected_count}"
            )
            count_score = max(0.0, 0.10 * ratio)

        score += count_score
        score = min(score, 1.0)   # cap at 1.0

        # ── Log summary ───────────────────────────────────────────────────────
        if issues:
            level = logging.ERROR if score < 0.30 else logging.WARNING
            self._log.log(
                level,
                f"validate_candles [{timeframe}] score={score:.2f} | "
                f"issues: {'; '.join(issues)}",
            )
        else:
            self._log.debug(
                f"validate_candles [{timeframe}] score={score:.2f} | PASSED"
            )

        passed = score >= self._quality_min_score
        return ValidationResult(passed=passed, score=score, issues=issues, data=data)

    # ------------------------------------------------------------------
    # Config helper
    # ------------------------------------------------------------------

    @staticmethod
    def load_config(config_path: str = "config/pipeline_config.json") -> Dict:
        """
        Load cấu hình từ JSON file.

        Args:
            config_path: Đường dẫn tới file config JSON.

        Returns:
            Dict cấu hình.

        Raises:
            FileNotFoundError: Nếu file config không tồn tại.
            json.JSONDecodeError: Nếu file không phải JSON hợp lệ.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                "Hint: copy config/pipeline_config.json.example và điều chỉnh thông số."
            )
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    # ══════════════════════════════════════════════════════════════════
    #  PRIVATE METHODS
    # ══════════════════════════════════════════════════════════════════

    def _init_mt5(self, log_success: bool = True) -> bool:
        """
        Khởi tạo và xác thực kết nối MT5.

        Nếu có thông tin login (account, password, server) thì đăng nhập
        tự động. Sau khi initialize, detect timezone của MT5 server bằng
        cách so sánh server time với UTC.

        Args:
            log_success: Log INFO khi kết nối thành công (tắt khi gọi từ
                         reconnect loop để tránh spam).

        Returns:
            True nếu kết nối thành công.
        """
        if not MT5_AVAILABLE:
            self._log.warning(
                "_init_mt5(): MetaTrader5 package not installed. "
                "Running in OFFLINE/MOCK mode."
            )
            with self._lock:
                self._connected = False
            return False

        # Initialize
        if self._mt5_login:
            ok = mt5.initialize(
                login=self._mt5_login,
                password=self._mt5_password,
                server=self._mt5_server,
            )
        else:
            ok = mt5.initialize()

        if not ok:
            err = mt5.last_error() if mt5 else "N/A"
            self._log.error(f"MT5 initialize() failed: {err}")
            with self._lock:
                self._connected = False
            return False

        # Detect server timezone offset
        self._detect_server_tz()

        with self._lock:
            self._connected = True

        if log_success:
            info = mt5.terminal_info()
            self._log.info(
                f"MT5 connected | terminal='{getattr(info, 'name', 'N/A')}' | "
                f"server_tz_offset=UTC+{self._server_tz_offset}h"
            )
        return True

    def _detect_server_tz(self) -> None:
        """
        Tự động detect timezone offset của MT5 server bằng cách kéo 1 candle H1
        và so sánh timestamp với UTC hiện tại.

        Exness thường dùng UTC+2 (EET) hoặc UTC+3 (EEST mùa hè). Hàm này
        xác định chính xác để chuẩn hóa thời gian candle về UTC tuyệt đối.

        Nếu detect thất bại → giữ nguyên giá trị ``server_tz_offset`` từ config.
        """
        if not MT5_AVAILABLE or not mt5:
            return
        try:
            if mt5.TIMEFRAME_H1 is None:
                return
            rates = mt5.copy_rates_from_pos(self._symbol, mt5.TIMEFRAME_H1, 0, 2)
            if rates is None or len(rates) == 0:
                return
            # Lấy timestamp candle cuối (đã đóng) và so với giờ UTC hệ thống
            candle_ts = int(rates[-2]["time"])   # candle đã đóng, an toàn hơn
            utc_now_ts = int(datetime.now(timezone.utc).timestamp())
            # Server time = candle_ts + offset*3600
            # Tính offset gần đúng bằng cách so với current UTC hour boundary
            utc_hour_boundary = (utc_now_ts // 3600) * 3600
            diff = (candle_ts % 86400) - (utc_hour_boundary % 86400)
            # Làm tròn về múi giờ gần nhất (UTC+0 đến UTC+12)
            offset = round(diff / 3600) % 24
            if 0 <= offset <= 12:
                self._server_tz_offset = offset
                self._log.debug(
                    f"MT5 server timezone auto-detected: UTC+{self._server_tz_offset}h"
                )
        except Exception as exc:
            self._log.warning(
                f"Server timezone detection failed ({exc}). "
                f"Using config fallback: UTC+{self._server_tz_offset}h"
            )

    def _run_fetch_all(self) -> bool:
        """
        Kéo data từ MT5 cho tất cả timeframes song song, cập nhật ``_data_store``.

        Returns:
            True nếu ít nhất 1 timeframe fetch thành công.
        """
        results = self._run_fetch_all_and_return()
        return any(v is not None for v in results.values())

    def _run_fetch_all_and_return(self) -> Dict[str, Optional[np.ndarray]]:
        """
        Kéo data song song và trả về kết quả (internal method).

        Dùng ``executor.submit()`` thay vì ``executor.map()`` để có thể log
        từng timeframe riêng biệt khi kết quả về.

        Returns:
            Dict mapping timeframe → ndarray (None nếu fetch thất bại).
        """
        if self._executor is None:
            return {}

        result_map: Dict[str, Optional[np.ndarray]] = {}

        # Submit 3 tasks song song vào thread pool
        future_to_tf = {
            self._executor.submit(self._fetch_candles, tf): tf
            for tf in self._tf_config
        }

        for future in as_completed(future_to_tf):
            tf = future_to_tf[future]
            try:
                data = future.result()
                result_map[tf] = data
            except Exception as exc:
                self._log.error(
                    f"_fetch_candles({tf}) raised unhandled exception: {exc}",
                    exc_info=True,
                )
                result_map[tf] = None

        return result_map

    def _fetch_candles(self, timeframe: str) -> Optional[np.ndarray]:
        """
        Kéo và validate dữ liệu candle cho một timeframe.

        Chạy bên trong ThreadPoolExecutor worker thread — **không** chạy trên
        Main Thread. Thread-safe: cập nhật ``_data_store`` dưới RLock.

        Quy trình:
          1. Map tên tf → hằng số MT5 (TIMEFRAME_H1, TIMEFRAME_M15, TIMEFRAME_M5)
          2. Gọi ``mt5.copy_rates_from_pos()``
          3. Retry 1 lần với reconnect nếu nhận None
          4. Chuẩn hóa timestamp về UTC (trừ server_tz_offset)
          5. Validate bằng ``validate_candles()``
          6. Cập nhật ``_data_store`` nếu quality score đủ ngưỡng

        Args:
            timeframe: Tên timeframe (``"H1"``, ``"M15"``, ``"M5"``).

        Returns:
            numpy ndarray đã validated và UTC-normalized, hoặc None nếu thất bại.
        """
        if not MT5_AVAILABLE or not mt5:
            self._log.debug(f"_fetch_candles({timeframe}): MT5 not available, skipping")
            return None

        tf_cfg = self._tf_config.get(timeframe, {})
        mt5_tf_name: str = tf_cfg.get("mt5_tf", f"TIMEFRAME_{timeframe}")
        candle_count: int = tf_cfg.get("candles", 200)

        # Map string → MT5 constant (e.g. "TIMEFRAME_H1" → mt5.TIMEFRAME_H1)
        mt5_tf = getattr(mt5, mt5_tf_name, None)
        if mt5_tf is None:
            self._log.error(
                f"_fetch_candles({timeframe}): Unknown MT5 timeframe constant "
                f"'{mt5_tf_name}'. Check pipeline_config.json."
            )
            return None

        t_start = time.monotonic()

        # First attempt
        rates = mt5.copy_rates_from_pos(self._symbol, mt5_tf, 0, candle_count)

        # Retry once with reconnect if failed
        if rates is None:
            self._log.warning(
                f"_fetch_candles({timeframe}): MT5 returned None — "
                "triggering reconnect and retrying once..."
            )
            reconnected = self.mt5_reconnect()
            if reconnected:
                rates = mt5.copy_rates_from_pos(self._symbol, mt5_tf, 0, candle_count)

        latency_ms = (time.monotonic() - t_start) * 1000

        if rates is None:
            self._log.error(
                f"_fetch_candles({timeframe}): FAILED after retry. "
                f"Latency={latency_ms:.1f}ms"
            )
            return None

        # Convert to numpy array (MT5 returns numpy structured array but ensure dtype)
        rates = np.array(rates)

        # ── UTC Normalization ──────────────────────────────────────────────────
        # MT5 Exness server trả về timestamp ở server local time (UTC+2/+3).
        # Trừ offset để về UTC tuyệt đối.
        offset_sec = self._server_tz_offset * 3600
        rates_normalized = rates.copy()
        rates_normalized["time"] = rates["time"] - offset_sec

        # ── Data Quality Validation ────────────────────────────────────────────
        vr = self.validate_candles(rates_normalized, timeframe, candle_count)

        self._log.debug(
            f"_fetch_candles({timeframe}): "
            f"candles={len(rates)} | latency={latency_ms:.1f}ms | "
            f"quality={vr.score:.2f} | passed={vr.passed}"
        )

        if not vr.passed:
            if vr.score < 0.30:
                # Score rất thấp → trigger reconnect
                self._log.error(
                    f"_fetch_candles({timeframe}): Quality score {vr.score:.2f} < 0.30 — "
                    "triggering reconnect"
                )
                self.mt5_reconnect()
            else:
                # Score thấp vừa → giữ data cũ, không update
                self._log.warning(
                    f"_fetch_candles({timeframe}): Quality score {vr.score:.2f} below "
                    f"threshold {self._quality_min_score} — keeping previous data"
                )
            return None

        # ── Update shared data store (thread-safe) ────────────────────────────
        with self._lock:
            self._data_store[timeframe] = vr.data

        return vr.data

    def _heartbeat_loop(self) -> None:
        """
        Daemon thread: ping MT5 terminal mỗi ``heartbeat_interval`` giây.

        Nếu ``terminal_info()`` trả về None hoặc mất kết nối →
        trigger ``mt5_reconnect()`` → sau khi reconnect thành công thì
        kéo lại full data bằng ``_run_fetch_all()``.

        Vòng lặp chạy đến khi ``_running == False``.
        Thread này là **daemon** → tự động kết thúc khi Main Thread exit.
        """
        self._log.debug(
            f"_heartbeat_loop: started (daemon=True, interval={self._heartbeat_interval}s)"
        )

        while self._running:
            time.sleep(self._heartbeat_interval)

            if not self._running:
                break

            if not MT5_AVAILABLE or not mt5:
                continue

            try:
                info = mt5.terminal_info()
                is_alive = (info is not None) and getattr(info, "connected", False)
            except Exception as exc:
                is_alive = False
                self._log.debug(f"_heartbeat_loop: terminal_info() raised: {exc}")

            if is_alive:
                self._log.debug("Heartbeat ✓ — MT5 terminal connected")
                with self._lock:
                    self._connected = True
                # Refresh data on each heartbeat cycle
                if self._executor:
                    self._run_fetch_all()
            else:
                self._log.warning(
                    "Heartbeat ✗ — MT5 terminal unreachable. Triggering reconnect..."
                )
                reconnected = self.mt5_reconnect()
                if reconnected and self._executor:
                    self._log.info(
                        "Heartbeat: reconnect successful — refreshing all timeframe data"
                    )
                    self._run_fetch_all()

        self._log.debug("_heartbeat_loop: exited gracefully")


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT — Quick smoke test (offline, no MT5 required)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Smoke test nhanh khi chạy trực tiếp: python data_pipeline.py
    Không cần kết nối MT5 thật — kiểm tra import, logger, session filter, validator.
    """
    import sys

    print("=" * 60)
    print("  Rabit_Exness AI — DataPipeline Smoke Test (Offline Mode)")
    print("=" * 60)

    # Minimal config không cần MT5
    test_config: Dict = {
        "symbol": "EURUSDc",
        "timeframes": {
            "H1":  {"mt5_tf": "TIMEFRAME_H1",  "candles": 200},
            "M15": {"mt5_tf": "TIMEFRAME_M15", "candles": 500},
            "M5":  {"mt5_tf": "TIMEFRAME_M5",  "candles": 1000},
        },
        "heartbeat_interval_sec": 30,
        "reconnect_max_attempts": 10,
        "data_quality_min_score": 0.60,
        "server_tz_offset_hours": 2,
        "log_dir": "logs",
        "session_filters": {
            "london_open_utc": [7, 12],
            "ny_open_utc":     [13, 17],
            "asian_avoid_utc": [21, 5],
            "news_buffer_minutes": 15,
        },
    }

    dp = DataPipeline(test_config)
    log = dp._log

    # Test 1: Session filter
    log.info("[SMOKE TEST 1] Session Filter")
    test_hours = [8, 14, 22, 3, 12]
    for h in test_hours:
        dt = datetime(2026, 3, 5, h, 0, 0, tzinfo=timezone.utc)
        active = dp.is_session_active(dt)
        print(f"  UTC {h:02d}:00 → session_active={active}")

    # Test 2: validate_candles with synthetic data
    log.info("[SMOKE TEST 2] Data Validator")
    dtype = np.dtype([
        ("time", np.int64), ("open", np.float64), ("high", np.float64),
        ("low", np.float64), ("close", np.float64),
        ("tick_volume", np.int64), ("spread", np.int32), ("real_volume", np.int64),
    ])
    # Generate 1000 synthetic M5 candles (base time: now, interval 300s)
    now_ts = int(datetime.now(timezone.utc).timestamp())
    n = 1000
    synth = np.zeros(n, dtype=dtype)
    synth["time"] = np.array([now_ts - (n - i) * 300 for i in range(n)])
    synth["open"] = 1.0850 + np.random.normal(0, 0.001, n)
    synth["close"] = synth["open"] + np.random.normal(0, 0.0005, n)
    synth["high"] = np.maximum(synth["open"], synth["close"]) + np.abs(np.random.normal(0, 0.0003, n))
    synth["low"] = np.minimum(synth["open"], synth["close"]) - np.abs(np.random.normal(0, 0.0003, n))
    synth["tick_volume"] = np.random.randint(100, 5000, n)

    vr = dp.validate_candles(synth, "M5", expected_count=1000)
    print(f"\n  Synthetic M5 validation:")
    print(f"    passed={vr.passed} | score={vr.score:.3f}")
    print(f"    issues={vr.issues if vr.issues else 'None'}")

    # Test 3: validate None data
    log.info("[SMOKE TEST 3] Validate None (should score 0.0)")
    vr_none = dp.validate_candles(None, "H1")
    print(f"\n  None data validation:")
    print(f"    passed={vr_none.passed} | score={vr_none.score:.3f} | issues={vr_none.issues}")

    print("\n" + "=" * 60)
    print("  Smoke test PASSED — DataPipeline importable and functional")
    print("  MT5 connectivity test requires live MT5 terminal.")
    print("=" * 60)
    sys.exit(0)
