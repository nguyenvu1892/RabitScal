"""
risk_manager.py — Risk Control Layer v1.0
==========================================
Module: Rabit_Exness AI — Phase 1, Task 1.3
Branch: task-1.3-risk-manager
Author: Antigravity
Date:   2026-03-05

Chịu trách nhiệm 2 nhiệm vụ chính:
    1. Position Sizing:  Tính lot size dựa trên Balance + ATR(14). Rủi ro tối đa 3%/lệnh.
    2. Safety Net:       Giám sát equity/balance, kích hoạt circuit breaker theo 3 ngưỡng.

Safety Net — 3 lớp bảo vệ:
    • COOLDOWN (4h)      — 3 lệnh thua liên tiếp (consecutive_loss_streak)
    • PAUSED  (đến sáng) — Drawdown ngày ≥ 6% (daily_drawdown_limit), kể cả floating equity
    • HALTED  (vĩnh viễn)— Balance ≤ 50% initial_balance, cần admin unhalt
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from math import floor
from pathlib import Path
from threading import RLock
from typing import Optional, Tuple

import MetaTrader5 as mt5


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class TradeResult:
    """Kết quả một lệnh đã đóng, truyền vào on_trade_closed()."""
    ticket:       int
    pnl:          float           # Lãi/lỗ realised (USD)
    close_reason: str             # SL / TP / MANUAL / FORCE_CLOSE
    timestamp:    datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------

STATUS_ACTIVE   = "ACTIVE"
STATUS_COOLDOWN = "COOLDOWN"
STATUS_PAUSED   = "PAUSED"
STATUS_HALTED   = "HALTED"


# ---------------------------------------------------------------------------
# RiskManager
# ---------------------------------------------------------------------------

class RiskManager:
    """
    Core risk control module cho Rabit_Exness AI.

    Khởi tạo:
        config         — dict đọc từ config/risk_config.json
        logger         — logging.Logger từ main.py
        state_path     — đường dẫn tới config/state.json (lưu initial_balance bền vững)
        config_path    — đường dẫn tới config/risk_config.json (hot-reload unhalt)
        symbol         — symbol giao dịch (VD: "EURUSDc")
    """

    # Defaults — bị override bởi config dict
    DEFAULT_DAILY_DD_LIMIT     = 0.06   # 6%
    DEFAULT_STREAK_LIMIT       = 3
    DEFAULT_BALANCE_FLOOR_PCT  = 0.50   # 50% initial_balance
    DEFAULT_MAX_RISK_PER_TRADE = 0.03   # 3%
    DEFAULT_ATR_MULTIPLIER     = 1.5
    DEFAULT_COOLDOWN_HOURS     = 4
    DEFAULT_MIN_LOT            = 0.01
    DEFAULT_MAX_LOT            = 100.0
    DEFAULT_VOLUME_STEP        = 0.01

    # Hot-reload config mỗi N giây (để detect unhalt_timestamp thay đổi)
    CONFIG_RELOAD_INTERVAL_SEC = 10

    def __init__(
        self,
        config:      dict,
        logger:      logging.Logger,
        state_path:  str = "config/state.json",
        config_path: str = "config/risk_config.json",
        symbol:      str = "EURUSDc",
    ) -> None:
        self._lock       = RLock()
        self.logger      = logger
        self.symbol      = symbol
        self.state_path  = Path(state_path)
        self.config_path = Path(config_path)

        # --- Đọc tham số từ config ---
        sn = config.get("safety_net", {})
        vol = config.get("volume", {})

        self.daily_dd_limit       = float(sn.get("daily_drawdown_limit_pct",  self.DEFAULT_DAILY_DD_LIMIT))
        self.streak_limit         = int(sn.get("consecutive_loss_streak",      self.DEFAULT_STREAK_LIMIT))
        self.balance_floor_pct    = float(sn.get("min_balance_floor_pct",      self.DEFAULT_BALANCE_FLOOR_PCT))
        self.cooldown_hours       = float(sn.get("cooldown_hours",             self.DEFAULT_COOLDOWN_HOURS))
        self.max_risk_per_trade   = float(config.get("risk_per_trade_pct",     self.DEFAULT_MAX_RISK_PER_TRADE))
        self.atr_multiplier       = float(config.get("atr_multiplier",         self.DEFAULT_ATR_MULTIPLIER))
        self.min_lot              = float(vol.get("min_lot",                   self.DEFAULT_MIN_LOT))
        self.max_lot              = float(vol.get("max_lot",                   self.DEFAULT_MAX_LOT))
        self.volume_step          = float(vol.get("volume_step",               self.DEFAULT_VOLUME_STEP))

        # --- initial_balance: đọc từ state.json bền vững ---
        self.initial_balance:     float   = self._load_or_init_balance()
        self.daily_start_balance: float   = self.initial_balance

        # --- Internal state ---
        self.trade_history:       list[TradeResult] = []
        self.consecutive_losses:  int               = 0
        self.bot_status:          str               = STATUS_ACTIVE
        self.cooldown_until:      Optional[datetime] = None

        # --- Hot-reload tracking ---
        self._last_config_reload:  float = time.monotonic()
        self._last_unhalt_ts_seen: Optional[str] = None

        self.logger.info(
            f"[RiskManager] Initialized | symbol={symbol} | initial_balance={self.initial_balance:.2f}"
            f" | daily_dd_limit={self.daily_dd_limit:.1%} | streak={self.streak_limit}"
            f" | floor={self.balance_floor_pct:.1%} | atr_mult={self.atr_multiplier}"
        )

    # ==========================================================================
    # LAYER 1 — POSITION SIZING
    # ==========================================================================

    def calculate_sl_distance(self, atr_value: float, direction: str = "BUY") -> float:
        """
        Tính SL distance tính bằng pips.
        sl_pips = ATR(14) * atr_multiplier
        direction không ảnh hưởng khoảng cách (chỉ ảnh hưởng phía đặt giá SL trong execution.py).

        Args:
            atr_value:  Giá trị ATR(14) hiện tại (dạng price, VD: 0.00150)
            direction:  "BUY" hoặc "SELL"
        Returns:
            sl_pips (float): khoảng cách SL tính bằng pips
        """
        # Chuyển ATR từ price units → pips (1 pip = 10 points với 5-digit broker)
        info = mt5.symbol_info(self.symbol)
        if info is None:
            self.logger.error(f"[RiskManager] Cannot get symbol_info for {self.symbol} — using default point=0.00001")
            point = 0.00001
        else:
            point = info.point

        pip_size  = point * 10              # 1 pip = 10 points
        sl_pips   = (atr_value / pip_size) * self.atr_multiplier
        self.logger.debug(
            f"[RiskManager] SL distance | atr={atr_value:.5f} | pip_size={pip_size:.5f}"
            f" | multiplier={self.atr_multiplier} | sl_pips={sl_pips:.2f}"
        )
        return sl_pips

    def calculate_lot_size(self, balance: float, atr_value: float, sl_pips: float) -> float:
        """
        Tính lot size động — rủi ro tối đa max_risk_per_trade (3%) balance/lệnh.

        Công thức:
            risk_amount      = balance * max_risk_per_trade
            pip_value_per_lot = _get_pip_value_per_lot(symbol)      ← MT5 API, không hardcode
            raw_lot          = risk_amount / (sl_pips * pip_value_per_lot)
            final_lot        = _floor_lot(raw_lot, volume_step)

        Args:
            balance:   Số dư tài khoản hiện tại (USD)
            atr_value: Giá trị ATR(14) (price)
            sl_pips:   Khoảng cách SL (pips) — từ calculate_sl_distance()
        Returns:
            Lot size đã làm tròn xuống theo volume_step (float)
        """
        risk_amount       = balance * self.max_risk_per_trade
        pip_value_per_lot = self._get_pip_value_per_lot(self.symbol)

        if pip_value_per_lot <= 0 or sl_pips <= 0:
            self.logger.error(
                f"[RiskManager] Invalid inputs — pip_val={pip_value_per_lot} sl_pips={sl_pips} → return min_lot"
            )
            return self.min_lot

        raw_lot   = risk_amount / (sl_pips * pip_value_per_lot)
        final_lot = self._floor_lot(raw_lot, self.volume_step)

        self.logger.info(
            f"[RiskManager] Lot calculated | balance={balance:.2f} | risk={risk_amount:.4f}USD"
            f" | sl_pips={sl_pips:.2f} | pip_val={pip_value_per_lot:.4f}"
            f" | raw_lot={raw_lot:.4f} | final_lot={final_lot:.2f}"
        )
        return final_lot

    def validate_trade(self, balance: float, lot_size: float, sl_pips: float) -> Tuple[bool, str]:
        """
        Kiểm tra điều kiện trước khi cho phép gửi lệnh.
        Hard Rule: Max 1 lệnh mở tại 1 thời điểm — kiểm tra bởi main.py State Machine,
                   không phải ở đây (RiskManager không biết số lệnh đang mở).

        Returns:
            (True, "OK") — cho phép giao dịch
            (False, reason) — từ chối, reason là mã string để log
        """
        # Check 1: bot phải là ACTIVE
        if not self.is_active():
            return False, f"BOT_NOT_ACTIVE:{self.bot_status}"

        # Check 2: lot >= min_volume
        if lot_size < self.min_lot:
            return False, f"LOT_BELOW_MINIMUM:{lot_size:.4f}<{self.min_lot}"

        # Check 3: lot <= max_volume
        if lot_size > self.max_lot:
            return False, f"LOT_ABOVE_MAXIMUM:{lot_size:.4f}>{self.max_lot}"

        # Check 4: risk không vượt quá giới hạn
        pip_val    = self._get_pip_value_per_lot(self.symbol)
        risk_usd   = lot_size * sl_pips * pip_val
        risk_pct   = risk_usd / balance if balance > 0 else 1.0
        if risk_pct > self.max_risk_per_trade * 1.05:      # cho phép sai số 5%
            return False, f"RISK_EXCEEDS_3PCT:{risk_pct:.2%}"

        self.logger.info(
            f"[RiskManager] Trade validated OK | lot={lot_size:.2f} | risk={risk_usd:.4f}USD"
            f" | risk_pct={risk_pct:.2%} | bot_status={self.bot_status}"
        )
        return True, "OK"

    # ==========================================================================
    # LAYER 2 — SAFETY NET: FLOATING DRAWDOWN (REALTIME)
    # ==========================================================================

    def check_floating_drawdown(self, current_equity: float) -> bool:
        """
        ⚡ CRITICAL: Gọi liên tục từ main.py trong khi có lệnh đang mở.
        
        Kiểm tra floating (unrealised) equity so với daily_start_balance.
        Nếu equity giảm quá daily_dd_limit → PAUSED ngay lập tức + signal đóng lệnh khẩn.

        Tại sao cần hàm này tách riêng với on_trade_closed():
            on_trade_closed() chỉ chạy SAU KHI lệnh đóng (realised PnL).
            Nhưng floating loss có thể làm equity về 0 TRƯỚC khi SL kích hoạt
            (VD: slippage mạnh, news, gap). Đây là hàng rào đầu tiên và quan trọng nhất.

        Args:
            current_equity: Giá trị equity hiện tại từ mt5.account_info().equity (USD)

        Returns:
            True  — equity an toàn, tiếp tục trade
            False — equity vượt ngưỡng → main.py phải đóng lệnh khẩn + về IDLE
        """
        with self._lock:
            if self.bot_status != STATUS_ACTIVE:
                return False   # Đã bị ngắt rồi, không cần check thêm

            floating_dd = (self.daily_start_balance - current_equity) / self.daily_start_balance

            if floating_dd >= self.daily_dd_limit:
                self.logger.critical(
                    f"[CIRCUIT_BREAKER] FLOATING_DD BREACH | equity={current_equity:.2f}"
                    f" | daily_start={self.daily_start_balance:.2f}"
                    f" | floating_dd={floating_dd:.2%} >= limit={self.daily_dd_limit:.2%}"
                    f" → PAUSED + MARKET_CLOSE_ALL"
                )
                # Kích hoạt PAUSED ngay — main.py sẽ nhận False và đóng lệnh khẩn
                self._trigger_pause(reason=f"FLOATING_DD_{floating_dd:.2%}")
                return False

            # Log cảnh báo khi floating DD đến 80% ngưỡng
            if floating_dd >= self.daily_dd_limit * 0.80:
                self.logger.warning(
                    f"[RiskManager] FLOATING_DD WARNING | equity={current_equity:.2f}"
                    f" | floating_dd={floating_dd:.2%} (80% of {self.daily_dd_limit:.1%} limit)"
                )

            return True

    # ==========================================================================
    # LAYER 2 — SAFETY NET: REALISED PnL (SAU KHI ĐÓNG LỆNH)
    # ==========================================================================

    def on_trade_closed(self, result: TradeResult, current_balance: float) -> None:
        """
        Điểm vào duy nhất khi một lệnh được đóng. Gọi từ [CLOSING] state trong main.py.

        Args:
            result:          TradeResult chứa ticket, pnl, close_reason, timestamp
            current_balance: Balance realised sau khi lệnh đóng (mt5.account_info().balance)
        """
        with self._lock:
            self.trade_history.append(result)

            self.logger.warning(
                f"[TRADE_CLOSE] ticket={result.ticket} | pnl={result.pnl:+.4f}"
                f" | reason={result.close_reason} | streak={self.consecutive_losses}"
                f" | balance={current_balance:.2f} | status={self.bot_status}"
            )

            if self.bot_status != STATUS_ACTIVE:
                return   # Đã bị ngắt, không process thêm

            # Thứ tự check: Balance Floor trước (nghiêm trọng nhất), rồi DD, rồi Streak
            self._check_balance_floor(current_balance)
            if self.bot_status != STATUS_ACTIVE:
                return

            self._check_daily_drawdown(current_balance)
            if self.bot_status != STATUS_ACTIVE:
                return

            self._update_streak(result.pnl)

    def _update_streak(self, pnl: float) -> None:
        """Đếm thua liên tiếp. Reset về 0 nếu thắng."""
        if pnl < 0:
            self.consecutive_losses += 1
            self.logger.info(
                f"[RiskManager] Loss streak: {self.consecutive_losses}/{self.streak_limit}"
            )
            if self.consecutive_losses >= self.streak_limit:
                self._trigger_cooldown(reason=f"STREAK_LIMIT_{self.streak_limit}")
        else:
            if self.consecutive_losses > 0:
                self.logger.info(
                    f"[RiskManager] Win detected — streak reset from {self.consecutive_losses} to 0"
                )
            self.consecutive_losses = 0

    def _check_daily_drawdown(self, current_balance: float) -> None:
        """Kiểm tra realised drawdown trong ngày."""
        if self.daily_start_balance <= 0:
            return
        daily_dd = (self.daily_start_balance - current_balance) / self.daily_start_balance
        if daily_dd >= self.daily_dd_limit:
            self._trigger_pause(reason=f"DAILY_DD_{daily_dd:.2%}")

    def _check_balance_floor(self, current_balance: float) -> None:
        """Kiểm tra balance floor — mức sàn tuyệt đối không được vượt qua."""
        floor_value = self.initial_balance * self.balance_floor_pct
        if current_balance <= floor_value:
            self._trigger_halt(
                reason=f"BALANCE_FLOOR_BREACH | balance={current_balance:.2f} <= floor={floor_value:.2f}"
            )

    # ==========================================================================
    # CIRCUIT BREAKER — STATE TRANSITIONS
    # ==========================================================================

    def _trigger_cooldown(self, reason: str) -> None:
        """COOLDOWN: Dừng bot 4 giờ sau N lệnh thua liên tiếp. Auto-resume."""
        self.bot_status    = STATUS_COOLDOWN
        self.cooldown_until = datetime.now(timezone.utc) + timedelta(hours=self.cooldown_hours)
        self.logger.critical(
            f"[CIRCUIT_BREAKER] COOLDOWN activated | reason={reason}"
            f" | resume={self.cooldown_until.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )

    def _trigger_pause(self, reason: str) -> None:
        """PAUSED: Dừng đến 00:00 UTC ngày hôm sau. Auto-resume khi reset_daily() gọi."""
        now = datetime.now(timezone.utc)
        # Tính 00:00 UTC ngày tiếp theo
        next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0)
        self.bot_status    = STATUS_PAUSED
        self.cooldown_until = next_midnight
        self.logger.critical(
            f"[CIRCUIT_BREAKER] DAILY_PAUSE | reason={reason}"
            f" | resume={next_midnight.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )

    def _trigger_halt(self, reason: str) -> None:
        """HALTED: Hard stop vĩnh viễn. Cần admin set unhalt_timestamp trong risk_config.json."""
        self.bot_status    = STATUS_HALTED
        self.cooldown_until = None
        self.logger.critical(
            f"[CIRCUIT_BREAKER] HARD_HALT | reason={reason}"
            f" | MANUAL_INTERVENTION_REQUIRED — set unhalt_timestamp in risk_config.json"
        )

    # ==========================================================================
    # BOT STATUS — CHECK & AUTO-RESUME
    # ==========================================================================

    def is_active(self) -> bool:
        """
        Kiểm tra bot có đang hoạt động không.
        • COOLDOWN:  Auto-resume nếu cooldown_until đã qua.
        • PAUSED:    Auto-resume do reset_daily() — KHÔNG tự resume ở đây.
        • HALTED:    Hot-reload config để check unhalt_timestamp do admin set.
        """
        with self._lock:
            now = datetime.now(timezone.utc)

            if self.bot_status == STATUS_ACTIVE:
                return True

            if self.bot_status == STATUS_COOLDOWN:
                if self.cooldown_until and now >= self.cooldown_until:
                    self.bot_status        = STATUS_ACTIVE
                    self.consecutive_losses = 0
                    self.logger.info(
                        f"[CIRCUIT_BREAKER] Cooldown expired — bot ACTIVE | time={now.strftime('%H:%M:%S UTC')}"
                    )
                    return True
                return False

            if self.bot_status == STATUS_PAUSED:
                return False

            if self.bot_status == STATUS_HALTED:
                # Hot-reload config để detect unhalt_timestamp do admin thay đổi
                self._maybe_reload_config_for_unhalt(now)
                return self.bot_status == STATUS_ACTIVE

            return False

    def _maybe_reload_config_for_unhalt(self, now: datetime) -> None:
        """
        Hot-reload risk_config.json mỗi CONFIG_RELOAD_INTERVAL_SEC giây.
        Nếu admin đặt unhalt_timestamp trong file và timestamp đó đã qua → unlock HALTED.

        Admin flow:
            1. Mở config/risk_config.json
            2. Thêm/sửa: "unhalt_timestamp": "2026-03-06 10:00:00"
            3. Lưu file → bot tự detect trong ≤10 giây → HALTED → ACTIVE

        Format unhalt_timestamp: "YYYY-MM-DD HH:MM:SS" (UTC)
        """
        elapsed = time.monotonic() - self._last_config_reload
        if elapsed < self.CONFIG_RELOAD_INTERVAL_SEC:
            return   # Chưa đến lúc reload

        self._last_config_reload = time.monotonic()

        try:
            raw = json.loads(self.config_path.read_text(encoding="utf-8"))
            unhalt_ts_str: Optional[str] = raw.get("unhalt_timestamp")

            if not unhalt_ts_str:
                return   # Không có key → admin chưa set

            if unhalt_ts_str == self._last_unhalt_ts_seen:
                return   # Đã check timestamp này rồi, không xử lý lại

            self._last_unhalt_ts_seen = unhalt_ts_str

            # Parse timestamp (UTC)
            try:
                unhalt_dt = datetime.strptime(unhalt_ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            except ValueError:
                self.logger.error(
                    f"[RiskManager] Invalid unhalt_timestamp format: '{unhalt_ts_str}'"
                    f" — expected 'YYYY-MM-DD HH:MM:SS'"
                )
                return

            if now >= unhalt_dt:
                self.bot_status = STATUS_ACTIVE
                self.consecutive_losses = 0
                self.logger.critical(
                    f"[CIRCUIT_BREAKER] HALTED → ACTIVE | admin_unhalt_ts={unhalt_ts_str}"
                    f" | detected_at={now.strftime('%Y-%m-%d %H:%M:%S UTC')}"
                )
            else:
                self.logger.info(
                    f"[RiskManager] unhalt_timestamp found but not yet: {unhalt_ts_str}"
                    f" (current UTC: {now.strftime('%Y-%m-%d %H:%M:%S')})"
                )

        except FileNotFoundError:
            self.logger.warning(f"[RiskManager] Config file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            self.logger.error(f"[RiskManager] Failed to parse config JSON: {e}")

    # ==========================================================================
    # DAILY RESET
    # ==========================================================================

    def reset_daily(self, new_balance: float) -> None:
        """
        Gọi mỗi đầu ngày UTC mới (00:00:05 UTC) từ main.py scheduler.
        • Reset daily_start_balance về balance hiện tại (realised).
        • Auto-resume nếu đang PAUSED (hết thời gian daily pause).
        • Không reset nếu đang HALTED (cần admin unhalt).
        """
        with self._lock:
            self.daily_start_balance = new_balance

            if self.bot_status == STATUS_PAUSED:
                self.bot_status = STATUS_ACTIVE
                self.logger.info(
                    f"[DAILY_RESET] daily_start_balance={new_balance:.2f}"
                    f" | status: PAUSED → ACTIVE"
                )
            else:
                self.logger.info(
                    f"[DAILY_RESET] daily_start_balance={new_balance:.2f}"
                    f" | status remains: {self.bot_status}"
                )

    # ==========================================================================
    # INITIAL BALANCE — PERSISTENCE (state.json)
    # ==========================================================================

    def _load_or_init_balance(self) -> float:
        """
        Đọc initial_balance từ config/state.json (bền vững qua restart).

        Logic:
            - Nếu state.json tồn tại và có initial_balance hợp lệ → dùng giá trị đó.
            - Nếu không (lần chạy đầu tiên hoặc file bị xóa) → fetch từ MT5, lưu vào file.

        Lý do KHÔNG reset initial_balance mỗi lần restart:
            Bot có thể crash khi đang lỗ (balance < initial). Nếu lần restart mới lấy
            balance hiện tại làm mốc 100% → Balance Floor và Daily DD limit hoàn toàn
            mất tác dụng (bot nghĩ mình đang ở 100% dù thực ra đã -20%).
        """
        # Try load from state.json (persistent across restarts)
        if self.state_path.exists():
            try:
                state = json.loads(self.state_path.read_text(encoding="utf-8"))
                saved_balance = float(state.get("initial_balance", 0))
                if saved_balance > 0:
                    self.logger.info(
                        f"[RiskManager] initial_balance loaded from state.json: {saved_balance:.2f}"
                    )
                    return saved_balance
                else:
                    self.logger.warning(
                        "[RiskManager] state.json exists but initial_balance invalid — refetching from MT5"
                    )
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                self.logger.warning(f"[RiskManager] Failed to read state.json: {e} — refetching from MT5")

        # First run: fetch from MT5 and persist
        return self._fetch_and_save_initial_balance()

    def _fetch_and_save_initial_balance(self) -> float:
        """
        Lần chạy đầu tiên: lấy balance từ mt5.account_info().balance, lưu vào state.json.
        Mọi lần restart sau đó sẽ đọc từ file này.
        """
        account_info = mt5.account_info()
        if account_info is None:
            error_desc = mt5.last_error()
            self.logger.critical(
                f"[RiskManager] FATAL: Cannot fetch account_info from MT5: {error_desc}"
                f" — using fallback 100.0 USD"
            )
            balance = 100.0   # Fallback an toàn — sẽ trigger halted sớm hơn bình thường
        else:
            balance = float(account_info.balance)
            self.logger.info(
                f"[RiskManager] First run: fetched initial_balance={balance:.2f} from MT5"
            )

        # Persist to state.json
        self._save_state({"initial_balance": balance})
        return balance

    def _save_state(self, data: dict) -> None:
        """Ghi dict vào state.json (tạo thư mục nếu chưa tồn tại)."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            existing: dict = {}
            if self.state_path.exists():
                try:
                    existing = json.loads(self.state_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            existing.update(data)
            self.state_path.write_text(
                json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            self.logger.debug(f"[RiskManager] state.json updated: {data}")
        except OSError as e:
            self.logger.error(f"[RiskManager] Cannot write state.json: {e}")

    # ==========================================================================
    # UTILITIES
    # ==========================================================================

    def _floor_lot(self, raw_lot: float, volume_step: float) -> float:
        """
        Làm tròn lot XUỐNG về bội số gần nhất của volume_step.
        Dùng round(..., 8) để tránh floating-point artifact trước khi gửi MT5.
        Tránh lỗi TRADE_RETCODE_INVALID_VOLUME (10014).

        Ví dụ: raw_lot=2.6700001, volume_step=0.01 → 2.67
                raw_lot=2.674999,  volume_step=0.01 → 2.67  (floor, không round)
                raw_lot=0.009,     volume_step=0.01 → 0.00  → sẽ bị validate_trade() reject
        """
        if volume_step <= 0:
            return raw_lot
        floored = floor(round(raw_lot / volume_step, 8)) * volume_step
        return round(floored, 8)

    def _get_pip_value_per_lot(self, symbol: str) -> float:
        """
        Tính pip value per lot (USD) từ MT5 API — không hardcode.

        Công thức:
            pip_value_per_lot = (pip_size / tick_size) * tick_value

        Cho Exness Cent EURUSDc:
            tick_size  = 0.00001
            tick_value ≈ 0.01 USD  (Cent account, lot nhỏ hơn Standard 100x)
            pip_size   = 0.0001    (= 10 * point)
            → pip_value_per_lot = (0.0001/0.00001) * 0.01 = 0.10 USD/pip/lot
        """
        info = mt5.symbol_info(symbol)
        if info is None:
            self.logger.error(
                f"[RiskManager] Cannot get symbol_info for {symbol} — returning fallback 0.10"
            )
            return 0.10   # Fallback Exness Cent EURUSDc known value

        pip_size = info.point * 10    # 1 pip = 10 points (5-digit broker)
        if info.trade_tick_size <= 0:
            return 0.10
        pip_value_per_lot = (pip_size / info.trade_tick_size) * info.trade_tick_value
        return pip_value_per_lot

    # ==========================================================================
    # MONITORING & REPORTING
    # ==========================================================================

    def get_status_report(self) -> dict:
        """Trả về dict tóm tắt trạng thái — dùng cho dashboard và logging định kỳ."""
        with self._lock:
            balance_floor = self.initial_balance * self.balance_floor_pct
            daily_dd_pct  = (
                (self.daily_start_balance - self.initial_balance) / self.daily_start_balance
                if self.daily_start_balance > 0 else 0.0
            )
            return {
                "bot_status":          self.bot_status,
                "consecutive_losses":  self.consecutive_losses,
                "daily_start_balance": round(self.daily_start_balance, 2),
                "initial_balance":     round(self.initial_balance, 2),
                "balance_floor_value": round(balance_floor, 2),
                "daily_dd_limit_pct":  self.daily_dd_limit,
                "cooldown_until":      (
                    self.cooldown_until.strftime("%Y-%m-%d %H:%M:%S UTC")
                    if self.cooldown_until else None
                ),
                "today_trade_count":   len([
                    t for t in self.trade_history
                    if t.timestamp.date() == datetime.now(timezone.utc).date()
                ]),
            }

    def manual_reset(self, confirmed: bool = False) -> bool:
        """
        Admin-only: Unlock HALTED state.
        
        Ưu tiên dùng hot-reload qua unhalt_timestamp trong risk_config.json.
        Hàm này dùng khi cần unlock ngay qua code (VD: từ CLI admin script).

        Args:
            confirmed: Phải truyền True để tránh gọi nhầm
        Returns:
            True nếu unlock thành công, False nếu không ở HALTED hoặc không confirm
        """
        if not confirmed:
            self.logger.error("[RiskManager] manual_reset() called without confirmed=True — ignored")
            return False
        with self._lock:
            if self.bot_status != STATUS_HALTED:
                self.logger.warning(
                    f"[RiskManager] manual_reset() called but status is {self.bot_status} (not HALTED)"
                )
                return False
            self.bot_status        = STATUS_ACTIVE
            self.consecutive_losses = 0
            self.logger.critical("[CIRCUIT_BREAKER] HALTED → ACTIVE via manual_reset() — Admin confirmed")
            return True
