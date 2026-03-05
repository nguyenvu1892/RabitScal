"""
strategy_engine.py — Signal Analysis Layer v1.0
=================================================
Module: Rabit_Exness AI — Phase 2, Task 2.1
Branch: task-2.1-strategy-engine
Author: Antigravity
Date:   2026-03-05

Trái tim hệ thống: phân tích đa khung thời gian (H1/M15/M5), kết hợp 5 vũ khí
SMC + VSA để ra tín hiệu vào lệnh. KHÔNG gửi lệnh, KHÔNG quản lý rủi ro.

5 Vũ khí:
    [H1]  1. Swing High/Low detection  (confirmed-close only — anti-repainting)
          2. BOS/CHoCH Market Structure
    [M15] 3. FVG Scanner (Imbalance Pool — TTL 48 bars, max_size 20, Violated/Mitigated)
    [M5]  4. Price Touch FVG + Pinbar (wick BẮT BUỘC chạm FVG)
          5. VSA Session Baseline + adjacent candle volume filter

Anti-Repainting — Quy tắc bất di bất dịch:
    Tất cả tính toán dùng data[:-1] hoặc data[-2] (candle đã đóng).
    TUYỆT ĐỐI không dùng data[-1] (nến đang chạy) cho bất kỳ logic signal nào.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SESSION_LONDON = "LONDON"
SESSION_NY     = "NY"
SESSION_GLOBAL = "GLOBAL"

DIRECTION_BUY     = "BUY"
DIRECTION_SELL    = "SELL"
DIRECTION_NEUTRAL = "NEUTRAL"

BIAS_BULLISH = "BULLISH"
BIAS_BEARISH = "BEARISH"
BIAS_NEUTRAL = "NEUTRAL"

STRUCTURE_BOS   = "BOS"
STRUCTURE_CHOCH = "CHoCH"
STRUCTURE_NONE  = "NONE"


# ---------------------------------------------------------------------------
# Data Containers
# ---------------------------------------------------------------------------

@dataclass
class FVGZone:
    """Fair Value Gap — một vùng imbalance M15."""
    top:           float    # Mức giá trên của gap
    bottom:        float    # Mức giá dưới của gap
    direction:     str      # "BULLISH" | "BEARISH"
    created_at:    float    # Unix timestamp của nến giữa (thời điểm FVG hình thành)
    ttl_remaining: int      # Số nến M15 còn lại (đếm ngược)
    fvg_id:        int      # ID duy nhất để track mitigated state

    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def is_expired(self) -> bool:
        return self.ttl_remaining <= 0

    @property
    def size(self) -> float:
        return self.top - self.bottom


@dataclass
class PriceLevel:
    """Một điểm Swing High hoặc Swing Low đã confirmed."""
    price:        float
    bar_index:    int
    level_type:   str   # "HIGH" | "LOW"
    confirmed_at: int   # bar_index của nến xác nhận phải (đã đóng)


@dataclass
class SwingPoints:
    highs: list[PriceLevel] = field(default_factory=list)
    lows:  list[PriceLevel] = field(default_factory=list)

    @property
    def last_high(self) -> Optional[PriceLevel]:
        return self.highs[-1] if self.highs else None

    @property
    def last_low(self) -> Optional[PriceLevel]:
        return self.lows[-1] if self.lows else None


@dataclass
class MarketBias:
    direction:      str    # "BULLISH" | "BEARISH" | "NEUTRAL"
    last_bos_level: float  # Mức giá của BOS/CHoCH gần nhất
    structure_type: str    # "BOS" | "CHoCH" | "NONE"


@dataclass
class SignalResult:
    """Output của StrategyEngine.analyze() — truyền sang main.py."""
    has_signal:    bool
    direction:     str     # "BUY" | "SELL" | ""
    score:         float   # Composite score 0.0–1.0
    entry_price:   float
    sl_price:      float
    fvg_ref:       Optional[FVGZone] = None
    structure:     str    = ""   # "BOS" | "CHoCH"
    pinbar_ratio:  float  = 0.0
    vsa_score:     float  = 0.0

    @staticmethod
    def no_signal() -> "SignalResult":
        return SignalResult(
            has_signal=False, direction="", score=0.0,
            entry_price=0.0, sl_price=0.0
        )


# ---------------------------------------------------------------------------
# StrategyEngine
# ---------------------------------------------------------------------------

class StrategyEngine:
    """
    Core signal analysis module cho Rabit_Exness AI.

    Gọi analyze() mỗi khi có nến M5 mới đóng. Trả về SignalResult.
    KHÔNG side-effect nào ngoài việc maintain self.fvg_pool.
    """

    # Defaults — bị override bởi config dict
    DEFAULT_SWING_LOOKBACK         = 3
    DEFAULT_FVG_MAX_SIZE           = 20
    DEFAULT_FVG_TTL_CANDLES        = 48      # 48 nến M15 ≈ 12h
    DEFAULT_PINBAR_WICK_RATIO      = 0.60
    DEFAULT_PINBAR_BODY_RATIO_MAX  = 0.35
    DEFAULT_VSA_CLIMAX_MULTIPLIER  = 1.5
    DEFAULT_VSA_ADJ_CANDLE_MULT    = 1.2     # vol > prev_candle.vol * 1.2
    DEFAULT_MIN_SESSION_CANDLES    = 10      # ≥ 10 nến cùng session mới tính baseline
    DEFAULT_MIN_SIGNAL_SCORE       = 0.55
    DEFAULT_FVG_BUFFER_PIPS        = 2
    DEFAULT_MIN_CANDLE_RANGE_PIPS  = 2       # Tránh Doji / nến quá nhỏ

    def __init__(
        self,
        config: dict,
        logger: logging.Logger,
        symbol: str = "EURUSDc",
    ) -> None:
        self.logger = logger
        self.symbol = symbol

        cfg_fvg = config.get("fvg", {})
        cfg_pin = config.get("pinbar", {})
        cfg_vsa = config.get("vsa", {})
        cfg_sig = config.get("signal", {})
        cfg_ses = config.get("sessions", {})

        self.swing_lookback         = int(config.get("swing_lookback",            self.DEFAULT_SWING_LOOKBACK))
        self.fvg_max_size           = int(cfg_fvg.get("max_size",                 self.DEFAULT_FVG_MAX_SIZE))
        self.fvg_ttl_candles        = int(cfg_fvg.get("ttl_candles",              self.DEFAULT_FVG_TTL_CANDLES))
        self.fvg_buffer_pips        = float(cfg_fvg.get("buffer_pips",            self.DEFAULT_FVG_BUFFER_PIPS))
        self.pinbar_wick_ratio      = float(cfg_pin.get("wick_ratio_min",          self.DEFAULT_PINBAR_WICK_RATIO))
        self.pinbar_body_max        = float(cfg_pin.get("body_ratio_max",          self.DEFAULT_PINBAR_BODY_RATIO_MAX))
        self.min_candle_range_pips  = float(cfg_pin.get("min_candle_range_pips",   self.DEFAULT_MIN_CANDLE_RANGE_PIPS))
        self.vsa_climax_mult        = float(cfg_vsa.get("climax_multiplier",       self.DEFAULT_VSA_CLIMAX_MULTIPLIER))
        self.vsa_adj_candle_mult    = float(cfg_vsa.get("adj_candle_multiplier",   self.DEFAULT_VSA_ADJ_CANDLE_MULT))
        self.min_session_candles    = int(cfg_vsa.get("min_session_candles",       self.DEFAULT_MIN_SESSION_CANDLES))
        self.min_signal_score       = float(cfg_sig.get("min_composite_score",     self.DEFAULT_MIN_SIGNAL_SCORE))
        self.session_london         = tuple(cfg_ses.get("london", [7, 12]))
        self.session_ny             = tuple(cfg_ses.get("ny",     [13, 17]))

        # FVG Pool — maintain xuyên suốt vòng đời StrategyEngine
        self.fvg_pool:         list[FVGZone] = []
        self._fvg_id_counter:  int           = 0
        self._mitigated_ids:   set[int]      = set()  # FVG đã trigger signal → không dùng lại

        # Point value cache (pips per price unit) — lấy 1 lần từ config
        self._pip_size: float = float(config.get("pip_size", 0.0001))

        self.logger.info(
            f"[StrategyEngine] Initialized | symbol={symbol}"
            f" | swing_lookback={self.swing_lookback}"
            f" | fvg_ttl={self.fvg_ttl_candles} bars"
            f" | fvg_pool_max={self.fvg_max_size}"
            f" | min_score={self.min_signal_score}"
        )

    # ==========================================================================
    # ORCHESTRATOR — Điểm vào duy nhất từ main.py
    # ==========================================================================

    def analyze(
        self,
        h1_data:      np.ndarray,
        m15_data:     np.ndarray,
        m5_data:      np.ndarray,
        current_time: datetime,
    ) -> SignalResult:
        """
        Phân tích đa khung thời gian và trả về SignalResult.

        Gọi mỗi khi nến M5 mới đóng cửa.

        Args:
            h1_data:      Mảng OHLCV H1 (numpy structured array với fields: time,open,high,low,close,tick_volume)
            m15_data:     Mảng OHLCV M15
            m5_data:      Mảng OHLCV M5
            current_time: Thời điểm UTC hiện tại (datetime với tzinfo)

        Returns:
            SignalResult — has_signal=True nếu đủ điều kiện vào lệnh
        """
        # --- Guard: đủ data để tính toán không? ---
        if len(h1_data) < self.swing_lookback * 2 + 5:
            self.logger.debug("[StrategyEngine] Insufficient H1 data — skip")
            return SignalResult.no_signal()
        if len(m15_data) < 5:
            self.logger.debug("[StrategyEngine] Insufficient M15 data — skip")
            return SignalResult.no_signal()
        if len(m5_data) < 52:
            self.logger.debug("[StrategyEngine] Insufficient M5 data (need ≥52) — skip")
            return SignalResult.no_signal()

        # ── STEP 1 [H1]: Market Structure ──────────────────────────────────
        swings = self.detect_swing_points(h1_data)
        bias   = self.detect_market_structure(h1_data, swings)

        if bias.direction == BIAS_NEUTRAL:
            self.logger.debug("[StrategyEngine] H1 bias=NEUTRAL — no signal")
            return SignalResult.no_signal()

        self.logger.info(
            f"[StrategyEngine] H1 bias={bias.direction} | structure={bias.structure_type}"
            f" | level={bias.last_bos_level:.5f}"
            f" | swings_high={len(swings.highs)} low={len(swings.lows)}"
        )

        # ── STEP 2 [M15]: FVG Update & Active Zones ────────────────────────
        current_price = float(m5_data[-2]["close"])   # nến M5 vừa đóng
        new_fvgs      = self.scan_fvg(m15_data)
        self.update_fvg_pool(new_fvgs, current_price)
        active_fvgs = self.get_active_fvgs(bias)

        self.logger.debug(
            f"[StrategyEngine] M15 FVG pool: total={len(self.fvg_pool)}"
            f" | active_aligned={len(active_fvgs)} | new_found={len(new_fvgs)}"
        )

        if not active_fvgs:
            self.logger.debug("[StrategyEngine] No active FVG aligned with bias — no signal")
            return SignalResult.no_signal()

        # ── STEP 3 [M5]: Price Touch FVG ───────────────────────────────────
        triggered_fvg = self.check_price_in_fvg(m5_data, active_fvgs)
        if triggered_fvg is None:
            return SignalResult.no_signal()

        direction = DIRECTION_BUY if bias.direction == BIAS_BULLISH else DIRECTION_SELL

        self.logger.info(
            f"[StrategyEngine] M5 FVG Touch! zone={triggered_fvg.direction}"
            f"[{triggered_fvg.bottom:.5f}–{triggered_fvg.top:.5f}]"
            f" | fvg_id={triggered_fvg.fvg_id} | price={current_price:.5f}"
        )

        # ── STEP 4 [M5]: Pinbar Validation ─────────────────────────────────
        is_pin, pin_score = self.validate_pinbar(m5_data, direction, triggered_fvg)
        if not is_pin:
            self.logger.debug(
                f"[StrategyEngine] Pinbar rejected | pin_score={pin_score:.3f}"
                f" | min_wick_ratio={self.pinbar_wick_ratio}"
            )
            return SignalResult.no_signal()

        self.logger.info(
            f"[StrategyEngine] Pinbar validated | wick_ratio={pin_score:.3f}"
            f" | direction={direction} ✓"
        )

        # ── STEP 5 [M5]: VSA Validation ────────────────────────────────────
        is_climax, vsa_score = self.validate_vsa(m5_data, current_time)
        if not is_climax:
            self.logger.debug(
                f"[StrategyEngine] VSA rejected | vsa_score={vsa_score:.3f}"
            )
            return SignalResult.no_signal()

        self.logger.info(
            f"[StrategyEngine] VSA confirmed | vsa_score={vsa_score:.3f} ✓"
        )

        # ── STEP 6: Composite Score ─────────────────────────────────────────
        signal_score = self._composite_score(pin_score, vsa_score, bias)
        if signal_score < self.min_signal_score:
            self.logger.debug(
                f"[StrategyEngine] Score too low | score={signal_score:.4f}"
                f" < min={self.min_signal_score}"
            )
            return SignalResult.no_signal()

        # ── STEP 7: Entry / SL ──────────────────────────────────────────────
        entry, sl = self._calculate_entry_sl(m5_data[-2], direction, triggered_fvg)

        # Đánh dấu FVG này đã được dùng làm trigger → không double entry
        self._mitigated_ids.add(triggered_fvg.fvg_id)

        result = SignalResult(
            has_signal   = True,
            direction    = direction,
            score        = signal_score,
            entry_price  = entry,
            sl_price     = sl,
            fvg_ref      = triggered_fvg,
            structure    = bias.structure_type,
            pinbar_ratio = pin_score,
            vsa_score    = vsa_score,
        )

        self.logger.info(
            f"[StrategyEngine] *** SIGNAL GENERATED *** | dir={direction}"
            f" | score={signal_score:.4f} | entry={entry:.5f} | sl={sl:.5f}"
            f" | fvg=[{triggered_fvg.bottom:.5f}–{triggered_fvg.top:.5f}]"
            f" | structure={bias.structure_type} | pin={pin_score:.3f} | vsa={vsa_score:.3f}"
        )
        return result

    # ==========================================================================
    # [VŨ KHÍ 1] H1 — SWING HIGH / LOW DETECTION
    # ==========================================================================

    def detect_swing_points(self, h1_data: np.ndarray) -> SwingPoints:
        """
        Xác định Swing High và Swing Low trên H1 — chỉ dùng nến đã đóng.

        Rule: Swing tại index i được confirmed khi:
            - high[i] > high[i±1..±n] (lookback đối xứng n nến)
            - Đã có đủ n nến BÊN PHẢI đã đóng để xác nhận
              → chỉ xác nhận index <= len(data) - 1 - lookback

        Args:
            h1_data: numpy structured array H1, dùng h1_data[:-1] (anti-repainting)
        """
        data = h1_data[:-1]   # ← Anti-repainting: bỏ nến H1 đang chạy
        n    = self.swing_lookback
        swings = SwingPoints()

        # Cần ít nhất 2n+1 nến để có 1 swing point
        if len(data) < 2 * n + 1:
            return swings

        highs = data["high"]
        lows  = data["low"]
        times = data["time"]

        # Duyệt vùng [n, len-n) — đảm bảo có đủ nến cả 2 phía đã đóng
        for i in range(n, len(data) - n):
            left_highs  = highs[i - n : i]
            right_highs = highs[i + 1 : i + n + 1]
            left_lows   = lows[i - n : i]
            right_lows  = lows[i + 1 : i + n + 1]

            # Swing High: high[i] là cực đại cục bộ
            if highs[i] > np.max(left_highs) and highs[i] > np.max(right_highs):
                swings.highs.append(PriceLevel(
                    price        = float(highs[i]),
                    bar_index    = i,
                    level_type   = "HIGH",
                    confirmed_at = i + n,   # bar index khi đủ n nến phải đã đóng
                ))

            # Swing Low: low[i] là cực tiểu cục bộ
            if lows[i] < np.min(left_lows) and lows[i] < np.min(right_lows):
                swings.lows.append(PriceLevel(
                    price        = float(lows[i]),
                    bar_index    = i,
                    level_type   = "LOW",
                    confirmed_at = i + n,
                ))

        self.logger.debug(
            f"[StrategyEngine] Swing points: highs={len(swings.highs)}"
            f" lows={len(swings.lows)}"
            + (f" | last_high={swings.last_high.price:.5f}" if swings.last_high else "")
            + (f" | last_low={swings.last_low.price:.5f}"  if swings.last_low  else "")
        )
        return swings

    # ==========================================================================
    # [VŨ KHÍ 2] H1 — MARKET STRUCTURE (BOS / CHoCH)
    # ==========================================================================

    def detect_market_structure(
        self, h1_data: np.ndarray, swings: SwingPoints
    ) -> MarketBias:
        """
        Phân tích BOS (Break of Structure) và CHoCH (Change of Character) trên H1.

        BOS:   Close phá vỡ Swing High/Low theo đúng hướng xu hướng → tiếp diễn
        CHoCH: Close phá vỡ Swing High/Low ngược hướng xu hướng → đảo chiều

        Chỉ xác nhận bằng confirmed close — KHÔNG dùng intrabar tick.

        Args:
            h1_data: numpy structured array H1 (dùng [:-1] anti-repainting)
            swings:  SwingPoints đã detected
        """
        data  = h1_data[:-1]   # Anti-repainting
        if len(data) == 0 or (not swings.highs and not swings.lows):
            return MarketBias(direction=BIAS_NEUTRAL, last_bos_level=0.0, structure_type=STRUCTURE_NONE)

        last_close = float(data[-1]["close"])

        # Lấy 2 swing gần nhất để xác định xu hướng hiện tại
        last_sh = swings.last_high
        last_sl = swings.last_low

        if last_sh is None or last_sl is None:
            return MarketBias(direction=BIAS_NEUTRAL, last_bos_level=0.0, structure_type=STRUCTURE_NONE)

        # BOS Bullish: close vượt trên Swing High gần nhất → xu hướng tăng tiếp tục
        if last_close > last_sh.price:
            return MarketBias(
                direction      = BIAS_BULLISH,
                last_bos_level = last_sh.price,
                structure_type = STRUCTURE_BOS,
            )

        # BOS Bearish: close phá dưới Swing Low gần nhất → xu hướng giảm tiếp tục
        if last_close < last_sl.price:
            return MarketBias(
                direction      = BIAS_BEARISH,
                last_bos_level = last_sl.price,
                structure_type = STRUCTURE_BOS,
            )

        # CHoCH: close nằm giữa, kiểm tra đảo chiều cấu trúc
        # Nếu giá từng tăng cao (có nhiều swing high) nhưng giờ close dưới swing low gần nhất
        # → ưu tiên kiểm tra CHoCH Bearish
        if len(swings.highs) >= 2 and len(swings.lows) >= 2:
            prev_sl = swings.lows[-2]
            prev_sh = swings.highs[-2]

            # CHoCH Bearish: giá phá thủng swing low trước — cấu trúc bullish bị phá
            if last_close < prev_sl.price:
                return MarketBias(
                    direction      = BIAS_BEARISH,
                    last_bos_level = prev_sl.price,
                    structure_type = STRUCTURE_CHOCH,
                )

            # CHoCH Bullish: giá phá vượt swing high trước — cấu trúc bearish bị phá
            if last_close > prev_sh.price:
                return MarketBias(
                    direction      = BIAS_BULLISH,
                    last_bos_level = prev_sh.price,
                    structure_type = STRUCTURE_CHOCH,
                )

        return MarketBias(direction=BIAS_NEUTRAL, last_bos_level=0.0, structure_type=STRUCTURE_NONE)

    # ==========================================================================
    # [VŨ KHÍ 3] M15 — FVG SCANNER
    # ==========================================================================

    def scan_fvg(self, m15_data: np.ndarray) -> list[FVGZone]:
        """
        Quét FVG (Fair Value Gap / Imbalance) trên M15 — 3 nến pattern.

        FVG Bullish: low[i+1] > high[i-1]  → gap phía trên (hỗ trợ tiềm năng)
        FVG Bearish: high[i+1] < low[i-1]  → gap phía dưới (kháng cự tiềm năng)

        QUAN TRỌNG: Dùng m15_data[:-1] — bỏ nến M15 chưa đóng.

        Args:
            m15_data: numpy structured array M15
        Returns:
            list[FVGZone] — các FVG mới tìm được (chưa lọc duplicate)
        """
        data = m15_data[:-1]   # Anti-repainting
        new_fvgs: list[FVGZone] = []

        if len(data) < 3:
            return new_fvgs

        for i in range(1, len(data) - 1):
            left  = data[i - 1]
            mid   = data[i]
            right = data[i + 1]

            # FVG Bullish: right.low > left.high → gap lên phía trên
            if float(right["low"]) > float(left["high"]):
                fvg_id = self._next_fvg_id()
                new_fvgs.append(FVGZone(
                    top           = float(right["low"]),
                    bottom        = float(left["high"]),
                    direction     = "BULLISH",
                    created_at    = float(mid["time"]),
                    ttl_remaining = self.fvg_ttl_candles,
                    fvg_id        = fvg_id,
                ))

            # FVG Bearish: right.high < left.low → gap xuống phía dưới
            elif float(right["high"]) < float(left["low"]):
                fvg_id = self._next_fvg_id()
                new_fvgs.append(FVGZone(
                    top           = float(left["low"]),
                    bottom        = float(right["high"]),
                    direction     = "BEARISH",
                    created_at    = float(mid["time"]),
                    ttl_remaining = self.fvg_ttl_candles,
                    fvg_id        = fvg_id,
                ))

        return new_fvgs

    def update_fvg_pool(self, new_fvgs: list[FVGZone], current_price: float) -> None:
        """
        Cập nhật FVG Pool mỗi khi có nến M15 mới đóng.

        Logic phân biệt 2 trạng thái hủy FVG (quan trọng — bản chất SMC):

        [VIOLATED / INVALIDATED]:
            Xảy ra khi nến M15/M5 ĐÓNG CỬA (close) xuyên qua ranh giới FVG:
                BUY zone:  close < fvg.bottom → cấu trúc đã gãy, vùng hỗ trợ thủng
                SELL zone: close > fvg.top    → cấu trúc đã gãy, vùng kháng cự thủng
            Hành động: XÓA FVG ngay lập tức.

        [MITIGATED / USED]:
            Xảy ra khi FVG này đã trigger ra 1 SignalResult (dù thắng hay thua).
            Tracking qua self._mitigated_ids (set các fvg_id đã dùng).
            Hành động: XÓA để tránh double entry trên cùng 1 vùng.

        [EXPIRED (TTL)]:
            FVG quá cũ (> 48 nến M15 = 12h) → không còn giá trị cấu trúc.
            Hành động: XÓA.

        Thứ tự xử lý:
            1. Giảm TTL tất cả FVG hiện tại
            2. Xóa Expired (TTL <= 0)
            3. Xóa Violated (close xuyên boundary)
            4. Xóa Mitigated (đã trigger signal)
            5. Append FVG mới (tránh duplicate created_at)
            6. Trim pool đến max_size (FIFO — xóa FVG cũ nhất)

        Args:
            new_fvgs:      FVG mới tìm được từ scan_fvg()
            current_price: Giá close của nến M5 vừa đóng
        """
        # STEP 1: Giảm TTL toàn bộ FVG hiện tại
        for fvg in self.fvg_pool:
            fvg.ttl_remaining -= 1

        before_count = len(self.fvg_pool)

        # STEP 2: Xóa Expired (TTL hết hạn)
        self.fvg_pool = [f for f in self.fvg_pool if not f.is_expired]

        # STEP 3: Xóa Violated (close xuyên qua ranh giới FVG — cấu trúc gãy)
        #   BUY zone  violated: current_price (close nến) < fvg.bottom
        #   SELL zone violated: current_price (close nến) > fvg.top
        #   NOTE: Dùng close (current_price từ m5_data[-2].close), KHÔNG dùng low/high
        #         để tránh false-positive từ wick chạm boundary nhưng close vẫn bên trong
        violated_ids: set[int] = set()
        for fvg in self.fvg_pool:
            if fvg.direction == "BULLISH" and current_price < fvg.bottom:
                violated_ids.add(fvg.fvg_id)
                self.logger.debug(
                    f"[StrategyEngine] FVG VIOLATED (BULLISH) id={fvg.fvg_id}"
                    f" | close={current_price:.5f} < bottom={fvg.bottom:.5f} → removed"
                )
            elif fvg.direction == "BEARISH" and current_price > fvg.top:
                violated_ids.add(fvg.fvg_id)
                self.logger.debug(
                    f"[StrategyEngine] FVG VIOLATED (BEARISH) id={fvg.fvg_id}"
                    f" | close={current_price:.5f} > top={fvg.top:.5f} → removed"
                )

        self.fvg_pool = [f for f in self.fvg_pool if f.fvg_id not in violated_ids]

        # STEP 4: Xóa Mitigated (đã trigger signal — tránh double entry)
        self.fvg_pool = [f for f in self.fvg_pool if f.fvg_id not in self._mitigated_ids]
        # Dọn dẹp _mitigated_ids: chỉ giữ những id còn trong pool để không phình set
        active_ids = {f.fvg_id for f in self.fvg_pool}
        self._mitigated_ids = {mid for mid in self._mitigated_ids if mid in active_ids}

        # STEP 5: Append FVG mới — tránh duplicate (cùng created_at + direction)
        existing_keys = {(f.created_at, f.direction) for f in self.fvg_pool}
        added = 0
        for fvg in new_fvgs:
            key = (fvg.created_at, fvg.direction)
            if key not in existing_keys:
                self.fvg_pool.append(fvg)
                existing_keys.add(key)
                added += 1

        # STEP 6: Trim đến max_size (FIFO — pop FVG cũ nhất tạo trước)
        while len(self.fvg_pool) > self.fvg_max_size:
            removed = self.fvg_pool.pop(0)
            self.logger.debug(
                f"[StrategyEngine] FVG pool FULL — evicted oldest id={removed.fvg_id}"
            )

        after_count = len(self.fvg_pool)
        removed_count = before_count - after_count + added

        self.logger.debug(
            f"[StrategyEngine] FVG pool update: {before_count}→{after_count}"
            f" | added={added} | removed≈{removed_count}"
            f" | violated={len(violated_ids)} | expired_ttl=TTL"
        )

    def get_active_fvgs(self, bias: MarketBias) -> list[FVGZone]:
        """
        Trả về FVG còn hiệu lực, cùng hướng với bias H1.

        bias=BULLISH → chỉ FVG BULLISH (vùng support tiềm năng, giá bounce lên)
        bias=BEARISH → chỉ FVG BEARISH (vùng resistance tiềm năng, giá bounce xuống)
        bias=NEUTRAL → [] (không trade khi structure không rõ)
        """
        if bias.direction == BIAS_NEUTRAL:
            return []

        target_dir = "BULLISH" if bias.direction == BIAS_BULLISH else "BEARISH"
        return [
            f for f in self.fvg_pool
            if f.direction == target_dir and f.fvg_id not in self._mitigated_ids
        ]

    # ==========================================================================
    # [VŨ KHÍ 4] M5 — PRICE TOUCH FVG
    # ==========================================================================

    def check_price_in_fvg(
        self,
        m5_data:     np.ndarray,
        active_fvgs: list[FVGZone],
    ) -> Optional[FVGZone]:
        """
        Kiểm tra nến M5 cuối đã đóng có chạm vào vùng FVG không.

        Touch logic (theo TechLead Q4):
            BUY zone:  candle.LOW <= fvg.top  AND  candle.LOW >= fvg.bottom
            SELL zone: candle.HIGH >= fvg.bottom  AND  candle.HIGH <= fvg.top

        Ý nghĩa: CHỈ cần wick (râu nến) chạm vào zone, không cần close bên trong.
        Điều này cho phép bắt được "sniper Pinbar" — wick xuyên FVG rồi đóng cửa
        tuốt luốt phía trên/dưới (dạng nến hoàn hảo nhất).

        QUAN TRỌNG: Dùng m5_data[-2] — nến M5 đã đóng cuối cùng.
                    m5_data[-1] là nến đang chạy → KHÔNG DÙNG.

        Args:
            m5_data:      numpy structured array M5
            active_fvgs:  FVG đang hoạt động cùng hướng bias
        Returns:
            FVGZone đầu tiên bị kích hoạt, hoặc None
        """
        if len(m5_data) < 2:
            return None

        candle = m5_data[-2]   # Anti-repainting: nến M5 cuối đã đóng
        c_low  = float(candle["low"])
        c_high = float(candle["high"])

        for fvg in active_fvgs:
            if fvg.direction == "BULLISH":
                # Wick dưới (low) phải chạm vào zone
                if fvg.bottom <= c_low <= fvg.top:
                    return fvg

            elif fvg.direction == "BEARISH":
                # Wick trên (high) phải chạm vào zone
                if fvg.bottom <= c_high <= fvg.top:
                    return fvg

        return None

    # ==========================================================================
    # [VŨ KHÍ 4] M5 — PINBAR VALIDATION
    # ==========================================================================

    def validate_pinbar(
        self,
        m5_data:     np.ndarray,
        direction:   str,
        fvg:         FVGZone,
    ) -> tuple[bool, float]:
        """
        Xác nhận nến Pinbar trên M5 (nến đã đóng).

        Điều kiện Pinbar BUY (Bullish):
            1. lower_wick / total_range >= pinbar_wick_ratio (≥60%)
            2. body_size / total_range  <= pinbar_body_max   (≤35%)
            3. BẮT BUỘC: low của nến (wick dưới) phải ≤ fvg.top
               → xác nhận wick là phần chạm vào FVG

        Điều kiện Pinbar SELL (Bearish):
            1. upper_wick / total_range >= pinbar_wick_ratio (≥60%)
            2. body_size / total_range  <= pinbar_body_max   (≤35%)
            3. BẮT BUỘC: high của nến (wick trên) phải ≥ fvg.bottom
               → xác nhận wick là phần chạm vào FVG

        Args:
            m5_data:   numpy structured array M5 (dùng [-2])
            direction: "BUY" | "SELL"
            fvg:       FVGZone đã trigger

        Returns:
            (is_pinbar: bool, wick_ratio: float)
        """
        if len(m5_data) < 2:
            return False, 0.0

        candle = m5_data[-2]   # Anti-repainting
        o = float(candle["open"])
        h = float(candle["high"])
        l = float(candle["low"])
        c = float(candle["close"])

        total_range = h - l
        body_size   = abs(c - o)

        # Tránh chia 0 — nến quá nhỏ (Doji cực nhỏ, liquidity void)
        min_range = self.min_candle_range_pips * self._pip_size
        if total_range < min_range:
            self.logger.debug(
                f"[StrategyEngine] Pinbar: candle too small | range={total_range:.6f}"
                f" < min={min_range:.6f}"
            )
            return False, 0.0

        body_ratio  = body_size / total_range
        upper_wick  = h - max(o, c)
        lower_wick  = min(o, c) - l

        if direction == DIRECTION_BUY:
            wick_ratio = lower_wick / total_range
            wick_ok    = wick_ratio >= self.pinbar_wick_ratio
            body_ok    = body_ratio <= self.pinbar_body_max
            # BẮT BUỘC: wick dưới (low) phải CHẠM vào FVG
            fvg_touch  = l <= fvg.top

            is_pinbar  = wick_ok and body_ok and fvg_touch
            self.logger.debug(
                f"[StrategyEngine] Pinbar BUY | wick_lo={wick_ratio:.3f}"
                f" (need≥{self.pinbar_wick_ratio}) | body={body_ratio:.3f}"
                f" (need≤{self.pinbar_body_max}) | fvg_touch={fvg_touch} → {'✓' if is_pinbar else '✗'}"
            )
            return is_pinbar, wick_ratio

        elif direction == DIRECTION_SELL:
            wick_ratio = upper_wick / total_range
            wick_ok    = wick_ratio >= self.pinbar_wick_ratio
            body_ok    = body_ratio <= self.pinbar_body_max
            # BẮT BUỘC: wick trên (high) phải CHẠM vào FVG
            fvg_touch  = h >= fvg.bottom

            is_pinbar  = wick_ok and body_ok and fvg_touch
            self.logger.debug(
                f"[StrategyEngine] Pinbar SELL | wick_up={wick_ratio:.3f}"
                f" (need≥{self.pinbar_wick_ratio}) | body={body_ratio:.3f}"
                f" (need≤{self.pinbar_body_max}) | fvg_touch={fvg_touch} → {'✓' if is_pinbar else '✗'}"
            )
            return is_pinbar, wick_ratio

        return False, 0.0

    # ==========================================================================
    # [VŨ KHÍ 5] M5 — VSA SESSION BASELINE + ADJACENT CANDLE FILTER
    # ==========================================================================

    def validate_vsa(
        self,
        m5_data:      np.ndarray,
        current_time: datetime,
    ) -> tuple[bool, float]:
        """
        Xác nhận tín hiệu khối lượng (VSA) với 2 lớp filter:

        [LỚP 1 — Session Baseline]:
            Volume của nến trigger phải >= session_mean * vsa_climax_multiplier (1.5x)
            Session baseline tính riêng: London / NY / Global
            Tránh "Climax ảo" khi London Open cao hơn Asian baseline thường chuyên.

        [LỚP 2 — Adjacent Candle Filter (TechLead Q5)]:
            Volume của nến trigger phải > volume nến TRƯỚC nó * vsa_adj_candle_mult (1.2x)
            Đảm bảo Climax thực sự NỔI BẬT so với nến liền kề, không chỉ so với trung bình.

        Chống Climax ảo:
            Nếu phiên mới bắt đầu < min_session_candles (10 nến) → chưa đủ data
            → dùng GLOBAL baseline thay thế để không bỏ lỡ tín hiệu đầu phiên.

        QUAN TRỌNG: Dùng m5_data[-2] làm nến trigger.
                    Window baseline: m5_data[-51:-1] (50 nến đã đóng, không dùng [-1]).

        Args:
            m5_data:      numpy structured array M5 (cần ≥ 52 nến)
            current_time: datetime UTC

        Returns:
            (is_climax: bool, vsa_score: float 0.0–1.0)
        """
        if len(m5_data) < 52:
            self.logger.debug("[StrategyEngine] VSA: insufficient M5 data (<52 bars)")
            return False, 0.0

        trigger_candle  = m5_data[-2]   # Anti-repainting: nến trigger đã đóng
        prev_candle     = m5_data[-3]   # Nến liền trước trigger
        trigger_vol     = float(trigger_candle["tick_volume"])
        prev_vol        = float(prev_candle["tick_volume"])

        if trigger_vol <= 0:
            return False, 0.0

        # ── LỚP 1: Session Baseline ─────────────────────────────────────────
        utc_hour = current_time.hour
        session  = self._get_session(utc_hour)

        # Window 50 nến đã đóng (không bao gồm nến trigger [-2] và nến đang chạy [-1])
        window   = m5_data[-52:-2]   # index -52 đến -3 → 50 nến
        vol_window = window["tick_volume"].astype(float)

        if session != SESSION_GLOBAL:
            # Lọc nến cùng session — ước tính giờ từ timestamp (unix seconds)
            def _in_session(ts: float, lo: int, hi: int) -> bool:
                hr = datetime.fromtimestamp(ts, tz=timezone.utc).hour
                return lo <= hr < hi

            lo_h, hi_h = (
                (self.session_london[0], self.session_london[1])
                if session == SESSION_LONDON
                else (self.session_ny[0], self.session_ny[1])
            )
            mask = np.array([
                _in_session(float(window[i]["time"]), lo_h, hi_h)
                for i in range(len(window))
            ])
            session_vols = vol_window[mask]

            if len(session_vols) >= self.min_session_candles:
                session_mean = float(np.mean(session_vols))
                self.logger.debug(
                    f"[StrategyEngine] VSA baseline: session={session}"
                    f" | candles_in_session={len(session_vols)}"
                    f" | mean={session_mean:.1f}"
                )
            else:
                # Chưa đủ nến trong phiên → fallback Global baseline
                session_mean = float(np.mean(vol_window))
                self.logger.debug(
                    f"[StrategyEngine] VSA baseline: session={session} NEW (<{self.min_session_candles} bars)"
                    f" → fallback GLOBAL mean={session_mean:.1f}"
                )
        else:
            session_mean = float(np.mean(vol_window))
            self.logger.debug(
                f"[StrategyEngine] VSA baseline: session=GLOBAL | mean={session_mean:.1f}"
            )

        if session_mean <= 0:
            return False, 0.0

        # LỚP 1 check: vol >= session_mean * climax_multiplier
        climax_threshold   = session_mean * self.vsa_climax_mult
        layer1_pass        = trigger_vol >= climax_threshold

        # ── LỚP 2: Adjacent Candle Filter ───────────────────────────────────
        # vol trigger phải > vol nến liền trước * adj_candle_mult
        adj_threshold = prev_vol * self.vsa_adj_candle_mult
        layer2_pass   = trigger_vol > adj_threshold

        is_climax = layer1_pass and layer2_pass

        # VSA score: normalized so với threshold (cap tại 1.0)
        vol_ratio  = trigger_vol / climax_threshold if climax_threshold > 0 else 0.0
        vsa_score  = float(min(vol_ratio, 1.0))

        self.logger.info(
            f"[StrategyEngine] VSA | session={session}"
            f" | vol={trigger_vol:.0f} | prev_vol={prev_vol:.0f}"
            f" | session_mean={session_mean:.1f}"
            f" | L1_threshold={climax_threshold:.1f} ({layer1_pass})"
            f" | L2_threshold={adj_threshold:.1f} ({layer2_pass})"
            f" | climax={'✓' if is_climax else '✗'} | score={vsa_score:.3f}"
        )
        return is_climax, vsa_score

    # ==========================================================================
    # PRIVATE HELPERS
    # ==========================================================================

    def _composite_score(
        self,
        pin_score: float,
        vsa_score: float,
        bias:      MarketBias,
    ) -> float:
        """
        Tính composite signal score 0.0–1.0.

        Trọng số:
            40% — pinbar_ratio  (chất lượng hình dạng nến)
            35% — vsa_score     (xác nhận khối lượng)
            25% — structure_score (độ mạnh BOS/CHoCH)

        BOS mạnh hơn CHoCH → structure_score khác nhau.
        """
        structure_score = 1.0 if bias.structure_type == STRUCTURE_BOS else 0.7
        score = (
            0.40 * pin_score
            + 0.35 * vsa_score
            + 0.25 * structure_score
        )
        return round(min(score, 1.0), 4)

    def _calculate_entry_sl(
        self,
        candle:    np.void,
        direction: str,
        fvg:       FVGZone,
    ) -> tuple[float, float]:
        """
        Tính Entry và Stop Loss dựa trên nến Pinbar và FVG boundary.

        Entry: close của nến Pinbar M5 (không predict — chờ nến đóng)
        SL:    ngoài ranh giới FVG + buffer (2 pips mặc định)

        BUY:  SL dưới fvg.bottom − buffer  (cấu trúc gãy = sai signal)
        SELL: SL trên fvg.top    + buffer
        """
        entry  = float(candle["close"])
        buffer = self.fvg_buffer_pips * self._pip_size

        if direction == DIRECTION_BUY:
            sl = fvg.bottom - buffer
        else:
            sl = fvg.top + buffer

        return entry, round(sl, 6)

    def _get_session(self, utc_hour: int) -> str:
        """Xác định trading session từ giờ UTC."""
        lo_start, lo_end = self.session_london
        ny_start, ny_end = self.session_ny
        if lo_start <= utc_hour < lo_end:
            return SESSION_LONDON
        if ny_start <= utc_hour < ny_end:
            return SESSION_NY
        return SESSION_GLOBAL

    def _next_fvg_id(self) -> int:
        """Tạo FVG ID duy nhất tăng dần."""
        self._fvg_id_counter += 1
        return self._fvg_id_counter

    def get_fvg_pool_snapshot(self) -> list[dict]:
        """
        Trả về snapshot của FVG pool — dùng cho dashboard/logging định kỳ.
        Thread-safe nếu không có concurrent write (main loop single-threaded).
        """
        return [
            {
                "fvg_id":        f.fvg_id,
                "direction":     f.direction,
                "top":           round(f.top, 6),
                "bottom":        round(f.bottom, 6),
                "ttl_remaining": f.ttl_remaining,
                "mitigated":     f.fvg_id in self._mitigated_ids,
            }
            for f in self.fvg_pool
        ]
