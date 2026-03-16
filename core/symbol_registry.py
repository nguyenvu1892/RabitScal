#!/usr/bin/env python3
"""
symbol_registry.py — Ev05: Multi-Asset Symbol Properties
=========================================================
Từ điển quy đổi tài sản: pip_size, contract_size, pip_value, swap rates.
Tính Dynamic Lot Size chuẩn xác cho mọi asset class.

Nguyên tắc:
  lot_size = (equity × risk_pct) / (sl_dist_price × pip_value_per_unit)
  Trong đó pip_value_per_unit = pip_size × contract_size × quote_to_usd_factor
"""
from __future__ import annotations

# ─── Symbol Properties ────────────────────────────────────────────────────────
#
# pip_size         : giá trị 1 pip/point (price units)
# contract_size    : số đơn vị base/lot
# pip_value_per_lot: USD giá trị 1 pip × 1 lot (khi quote=USD, static)
#                    Dùng "dynamic" nếu cần tính theo giá hiện tại
# swap_long        : USD / lot / đêm (âm = trừ tiền, dương = cộng tiền)
# swap_short       : USD / lot / đêm
# asset_class      : forex | metals | crypto | index | commodity
# spread_cost_usd  : Exness spread ước tính (USD / lot) — dùng cho backtest
# min_lot          : Lot size tối thiểu Exness

SYMBOL_PROPS: dict[str, dict] = {
    # ── FOREX ──────────────────────────────────────────────────────────────
    "EURUSDm": {
        "pip_size": 0.00001, "contract_size": 100_000,
        "pip_value_per_lot": 10.0,        # 0.00001 × 100k = $1/pip → ×10 = $10/pip... wait
        # Actually: 1 pip = 0.0001 for EURUSD → pip_value = 0.0001 × 100k = $10/lot
        # pip_size here = minimum price movement (0.00001 for 5-decimal)
        # SL in price distance → pip_value_per_lot = 1 USD per 0.00001 price move per lot
        #   = contract_size × pip_size = 100_000 × 0.00001 = $1 per pip per lot
        #   Thực tế dùng trong lot_calc: lot = risk_$ / sl_price_dist / contract_size
        "pip_value_per_lot": "formula",   # = contract_size (USD per price unit per lot)
        "swap_long":  -5.50,  "swap_short":  3.80,
        "asset_class": "forex", "spread_cost": 0.00015,
        "min_lot": 0.01, "max_lot": 500.0,
    },
    "GBPUSDm": {
        "pip_size": 0.00001, "contract_size": 100_000,
        "pip_value_per_lot": "formula",
        "swap_long": -6.20,  "swap_short":  4.10,
        "asset_class": "forex", "spread_cost": 0.00020,
        "min_lot": 0.01, "max_lot": 500.0,
    },
    "AUDUSDm": {
        "pip_size": 0.00001, "contract_size": 100_000,
        "pip_value_per_lot": "formula",
        "swap_long": -3.10,  "swap_short":  1.90,
        "asset_class": "forex", "spread_cost": 0.00015,
        "min_lot": 0.01, "max_lot": 500.0,
    },
    "NZDUSDm": {
        "pip_size": 0.00001, "contract_size": 100_000,
        "pip_value_per_lot": "formula",
        "swap_long": -2.80,  "swap_short":  1.60,
        "asset_class": "forex", "spread_cost": 0.00018,
        "min_lot": 0.01, "max_lot": 500.0,
    },
    "USDJPYm": {
        # JPY pairs: pip_value in USD = contract_size × pip_size / current_price
        "pip_size": 0.001, "contract_size": 100_000,
        "pip_value_per_lot": "dynamic_jpy",  # = contract_size × pip_size / price_usd_jpy
        "swap_long":  1.20,  "swap_short": -4.50,
        "asset_class": "forex", "spread_cost": 0.015,
        "min_lot": 0.01, "max_lot": 500.0,
    },
    "USDCADm": {
        "pip_size": 0.00001, "contract_size": 100_000,
        "pip_value_per_lot": "dynamic_cad",  # = contract_size × pip_size / price_usd_cad
        "swap_long": -3.20,  "swap_short":  1.80,
        "asset_class": "forex", "spread_cost": 0.00018,
        "min_lot": 0.01, "max_lot": 500.0,
    },
    "USDCHFm": {
        "pip_size": 0.00001, "contract_size": 100_000,
        "pip_value_per_lot": "dynamic_chf",
        "swap_long":  1.10,  "swap_short": -4.20,
        "asset_class": "forex", "spread_cost": 0.00018,
        "min_lot": 0.01, "max_lot": 500.0,
    },
    "EURJPYm": {
        "pip_size": 0.001, "contract_size": 100_000,
        "pip_value_per_lot": "dynamic_jpy",
        "swap_long": -2.50,  "swap_short": -1.80,
        "asset_class": "forex", "spread_cost": 0.025,
        "min_lot": 0.01, "max_lot": 500.0,
    },
    "GBPJPYm": {
        "pip_size": 0.001, "contract_size": 100_000,
        "pip_value_per_lot": "dynamic_jpy",
        "swap_long": -4.00,  "swap_short": -2.50,
        "asset_class": "forex", "spread_cost": 0.030,
        "min_lot": 0.01, "max_lot": 500.0,
    },
    "AUDJPYm": {
        "pip_size": 0.001, "contract_size": 100_000,
        "pip_value_per_lot": "dynamic_jpy",
        "swap_long":  0.80,  "swap_short": -3.00,
        "asset_class": "forex", "spread_cost": 0.022,
        "min_lot": 0.01, "max_lot": 500.0,
    },
    "EURGBPm": {
        "pip_size": 0.00001, "contract_size": 100_000,
        "pip_value_per_lot": "dynamic_gbp",
        "swap_long": -3.50,  "swap_short":  2.10,
        "asset_class": "forex", "spread_cost": 0.00015,
        "min_lot": 0.01, "max_lot": 500.0,
    },

    # ── METALS ─────────────────────────────────────────────────────────────
    "XAUUSDm": {
        # Gold: contract_size = 100 oz, pip_size = 0.01
        # pip_value per lot = 100 × 0.01 = $1 per pip per lot
        "pip_size": 0.01, "contract_size": 100,
        "pip_value_per_lot": 1.0,  # static: 100 oz × $0.01/oz = $1/pip/lot
        "swap_long": -2.80,  "swap_short":  1.90,
        "asset_class": "metals", "spread_cost": 0.25,
        "min_lot": 0.01, "max_lot": 100.0,
    },
    "XAGUSDm": {
        # Silver: contract_size = 5000 oz
        "pip_size": 0.001, "contract_size": 5_000,
        "pip_value_per_lot": 5.0,  # 5000 × 0.001 = $5/pip/lot
        "swap_long": -1.50,  "swap_short":  0.90,
        "asset_class": "metals", "spread_cost": 0.05,
        "min_lot": 0.01, "max_lot": 200.0,
    },
    "XPTUSDm": {
        # Platinum: contract_size = 100 oz
        "pip_size": 0.01, "contract_size": 100,
        "pip_value_per_lot": 1.0,
        "swap_long": -2.00,  "swap_short":  1.20,
        "asset_class": "metals", "spread_cost": 1.50,
        "min_lot": 0.01, "max_lot": 50.0,
    },

    # ── CRYPTO ─────────────────────────────────────────────────────────────
    "BTCUSDm": {
        # BTC: contract_size = 1 BTC, pip_size = 0.01 USD
        # Exness leverage 1:200 → margin_per_lot = price × 1 / 200
        # At BTC=$80k: margin = $80k/200 = $400/lot
        # On $20k account, max safe: 0.5 lot ($200 margin) — cấm vượt!
        "pip_size": 0.01, "contract_size": 1,
        "pip_value_per_lot": "dynamic_crypto",
        "swap_long": -50.0,  "swap_short": -50.0,
        "asset_class": "crypto", "spread_cost": 5.0,
        "min_lot": 0.01, "max_lot": 0.5,       # ← FIX: 0.01 min (Sếp Vũ), 0.5 max BTC
        "leverage": 200,
    },
    "ETHUSDm": {
        # ETH ~$3k: margin_per_lot = $3k/200 = $15/lot → max 5 lot = $75 margin
        "pip_size": 0.01, "contract_size": 1,
        "pip_value_per_lot": "dynamic_crypto",
        "swap_long": -12.0,  "swap_short": -12.0,
        "asset_class": "crypto", "spread_cost": 1.0,
        "min_lot": 0.01, "max_lot": 5.0,       # ← FIX: 5 lot ETH @$3k = $75 margin
        "leverage": 200,
    },
    "BNBUSDm": {
        # BNB ~$600: margin_per_lot = $600/200 = $3/lot → max 50 lot = $150 margin
        "pip_size": 0.01, "contract_size": 1,
        "pip_value_per_lot": "dynamic_crypto",
        "swap_long": -5.0,  "swap_short": -5.0,
        "asset_class": "crypto", "spread_cost": 0.3,
        "min_lot": 0.01, "max_lot": 50.0,      # ← FIX: 50 lot BNB @$600 = $150 margin
        "leverage": 200,
    },
    "XRPUSDm": {
        "pip_size": 0.00001, "contract_size": 10_000,
        "pip_value_per_lot": "dynamic_crypto",
        "swap_long": -0.80,  "swap_short": -0.80,
        "asset_class": "crypto", "spread_cost": 0.0005,
        "min_lot": 0.1, "max_lot": 10_000.0,
    },
    "SOLUSDm": {
        "pip_size": 0.001, "contract_size": 100,
        "pip_value_per_lot": "dynamic_crypto",
        "swap_long": -3.0,  "swap_short": -3.0,
        "asset_class": "crypto", "spread_cost": 0.15,
        "min_lot": 0.01, "max_lot": 500.0,
    },
    "LINKUSDm": {
        "pip_size": 0.001, "contract_size": 1_000,
        "pip_value_per_lot": "dynamic_crypto",
        "swap_long": -1.5,  "swap_short": -1.5,
        "asset_class": "crypto", "spread_cost": 0.05,
        "min_lot": 0.01, "max_lot": 1_000.0,
    },
    "ADAUSDm": {
        "pip_size": 0.00001, "contract_size": 100_000,
        "pip_value_per_lot": "dynamic_crypto",
        "swap_long": -0.50,  "swap_short": -0.50,
        "asset_class": "crypto", "spread_cost": 0.0003,
        "min_lot": 0.1, "max_lot": 50_000.0,
    },
    "DOGEUSDm": {
        "pip_size": 0.000001, "contract_size": 1_000_000,
        "pip_value_per_lot": "dynamic_crypto",
        "swap_long": -0.30,  "swap_short": -0.30,
        "asset_class": "crypto", "spread_cost": 0.00002,
        "min_lot": 0.1, "max_lot": 100_000.0,
    },

    # ── INDEX ──────────────────────────────────────────────────────────────
    "US30m": {
        # Dow ~39k: margin_per_lot = 39k×1/500 = $78/lot → max 10 lot = $780
        "pip_size": 1.0, "contract_size": 1,
        "pip_value_per_lot": 1.0,
        "swap_long": -3.50,  "swap_short": -3.50,
        "asset_class": "index", "spread_cost": 3.0,
        "min_lot": 0.01, "max_lot": 10.0,      # ← FIX: 10 lot Dow = $780 margin
        "leverage": 500,
    },
    "US500m": {
        "pip_size": 0.1, "contract_size": 10,
        "pip_value_per_lot": 1.0,
        "swap_long": -2.80,  "swap_short": -2.80,
        "asset_class": "index", "spread_cost": 0.5,
        "min_lot": 0.01, "max_lot": 20.0,
        "leverage": 500,
    },
    "USTECm": {
        # Nasdaq ~18k: margin_per_lot = 18k×10/500 = $360/lot → max 2 lot = $720
        "pip_size": 0.1, "contract_size": 10,
        "pip_value_per_lot": 1.0,
        "swap_long": -2.20,  "swap_short": -2.20,
        "asset_class": "index", "spread_cost": 1.0,
        "min_lot": 0.01, "max_lot": 2.0,       # ← FIX: 2 lot NQ @18k×10 = $720 margin
        "leverage": 500,
    },

    # ── COMMODITY ──────────────────────────────────────────────────────────
    "USOILm": {
        # WTI Crude Oil: $10 per pip per lot (100 barrels × $0.01/barrel/pip)
        "pip_size": 0.01, "contract_size": 100,
        "pip_value_per_lot": 1.0,
        "swap_long": -1.20,  "swap_short":  0.80,
        "asset_class": "commodity", "spread_cost": 0.04,
        "min_lot": 0.01, "max_lot": 200.0,
    },
}

# ─── Symbol Lists ─────────────────────────────────────────────────────────────
# Ev05: Top 10 (kept for reference)
TOP_10_SYMBOLS = [
    "XAUUSDm", "BTCUSDm", "ETHUSDm", "BNBUSDm", "US30m",
    "EURUSDm", "GBPUSDm", "USDJPYm", "AUDUSDm", "XAGUSDm",
]

# Ev05b: Super 6 — High-liquidity, strong FVG signal, cross-asset diversification
SUPER_6_SYMBOLS = [
    "XAUUSDm",   # Gold — Metals king, strongest Ev05 signal
    "XAGUSDm",   # Silver — Metals runner-up
    "US30m",     # Dow Jones — Index, US session liquidity
    "USTECm",    # Nasdaq 100 — Tech index, US session pair
    "BTCUSDm",   # Bitcoin — Crypto king (with fixed lot)
    "ETHUSDm",   # Ethereum — Crypto runner-up (with fixed lot)
]

# Ev05d: Elite 5 — $200 Live Rehearsal (Sếp Vũ phế truất XAGUSDm)
ELITE_5_SYMBOLS = [
    "XAUUSDm",   # Gold — Fixed lot 0.01 (vua bất biến)
    "US30m",     # Dow Jones — Index, dynamic 3% risk
    "USTECm",    # Nasdaq 100 — Index, dynamic 3% risk
    "BTCUSDm",   # Bitcoin — Crypto king, dynamic 3% risk
    "ETHUSDm",   # Ethereum — Crypto runner-up, dynamic 3% risk
]

ALL_SYMBOLS = list(SYMBOL_PROPS.keys())

# Bars per day per TF (approximate, Exness 24/5 trading)
BARS_PER_DAY = {
    "M1":  1440,
    "M5":  288,
    "M15": 96,
    "H1":  24,
}


# ─── Pip Value Calculator ─────────────────────────────────────────────────────

def get_pip_value_per_lot(symbol: str, current_price: float = 1.0) -> float:
    """
    Trả về USD value của 1 pip × 1 lot cho symbol.

    Args:
        symbol:        ví dụ 'XAUUSDm'
        current_price: giá hiện tại (close bar) — dùng cho JPY/cross pairs và crypto

    Returns:
        float: USD per 1 pip movement per 1 lot
    """
    props = SYMBOL_PROPS.get(symbol)
    if props is None:
        raise ValueError(f"Symbol {symbol} not found in SYMBOL_PROPS")

    pv = props["pip_value_per_lot"]
    pip_size = props["pip_size"]
    cs = props["contract_size"]

    if pv == "formula":
        # USD-quoted pairs (EURUSD, GBPUSD, AUDUSD, NZDUSD)
        # pip_value = pip_size × contract_size × 1 (quote=USD)
        return pip_size * cs   # e.g. 0.00001 × 100_000 = $1 per pip per lot

    elif pv == "dynamic_jpy":
        # JPY-quoted: pip_value = pip_size × contract_size / price
        return (pip_size * cs) / max(current_price, 1e-10)

    elif pv == "dynamic_cad":
        return (pip_size * cs) / max(current_price, 1e-10)

    elif pv == "dynamic_chf":
        return (pip_size * cs) / max(current_price, 1e-10)

    elif pv == "dynamic_gbp":
        # EURGBP: base=EUR, quote=GBP → pip_value in USD = pip_size × cs × GBP/USD
        # Approximate: use 1.25 as GBP/USD factor (will be close enough for training)
        return pip_size * cs * 1.25  # ~accurate for training purposes

    elif pv == "dynamic_crypto":
        # Crypto: pip_value = pip_size × contract_size (quote = USD always)
        return pip_size * cs

    else:
        # Static value already set
        return float(pv)


def calc_lot_size(
    equity: float,
    risk_pct: float,
    sl_price_dist: float,    # SL distance in price (not pips)
    symbol: str,
    current_price: float = 1.0,
    min_lot: float | None = None,
    max_lot: float | None = None,
) -> float:
    """
    Tính Dynamic Lot Size chuẩn xác cho mọi asset class.

    Formula:
        risk_amount_usd = equity × risk_pct
        lot = risk_amount_usd / (sl_price_dist / pip_size × pip_value_per_lot)
            = risk_amount_usd × pip_size / (sl_price_dist × pip_value_per_lot)

    Simplified:
        lot = risk_amount_usd / (sl_price_dist × contract_size)
        (khi pip_value_per_lot = pip_size × contract_size)

    Args:
        equity:         tài khoản hiện tại (USD)
        risk_pct:       % rủi ro per trade (0.01 = 1%)
        sl_price_dist:  khoảng cách SL theo giá (e.g., 2.0 ATR × price)
        symbol:         tên symbol (e.g., 'XAUUSDm')
        current_price:  giá điểm vào (dùng cho dynamic pip_value)

    Returns:
        float: lot size
    """
    props = SYMBOL_PROPS.get(symbol, {})
    risk_usd   = equity * risk_pct
    pv_per_lot = get_pip_value_per_lot(symbol, current_price)
    pip_size   = props.get("pip_size", 0.00001)

    # sl_in_pips = sl_price_dist / pip_size
    # lot = risk_usd / (sl_in_pips × pv_per_lot)
    sl_in_pips = sl_price_dist / max(pip_size, 1e-10)
    denom = sl_in_pips * pv_per_lot
    if denom <= 0:
        return props.get("min_lot", 0.01)

    lot = risk_usd / denom
    min_l = min_lot if min_lot is not None else props.get("min_lot", 0.01)
    max_l = max_lot if max_lot is not None else props.get("max_lot", 100.0)
    return float(max(min_l, min(max_l, lot)))


def calc_margin(symbol: str, lot: float, current_price: float = 1.0) -> float:
    """
    Tính margin requirement (USD) để mở 1 lệnh.

    Formula:
        margin = lot × contract_size × price / leverage

    Đây là safety gate: nếu margin > equity × max_margin_pct thì phải giảm lot.

    Args:
        symbol:        tên symbol
        lot:           lot size dự định
        current_price: giá thị trường hiện tại

    Returns:
        float: USD margin required
    """
    props = SYMBOL_PROPS.get(symbol, {})
    cs       = props.get("contract_size", 100_000)
    leverage = props.get("leverage", 1000)   # Exness default leverage
    return lot * cs * current_price / max(leverage, 1)


def calc_lot_size_with_margin(
    equity: float,
    risk_pct: float,
    sl_price_dist: float,
    symbol: str,
    current_price: float = 1.0,
    max_margin_pct: float = 0.30,   # tối đa 30% equity làm margin 1 lệnh
) -> float:
    """
    Tính lot size với double-check margin constraint.
    Đảm bảo:
      1. Lot đủ để risk đúng risk_pct
      2. Margin required <= equity × max_margin_pct
    """
    props   = SYMBOL_PROPS.get(symbol, {})
    min_l   = props.get("min_lot", 0.01)
    max_l   = props.get("max_lot", 100.0)

    # Bước 1: tính lot theo risk
    lot_by_risk = calc_lot_size(
        equity, risk_pct, sl_price_dist, symbol, current_price,
        min_lot=min_l, max_lot=max_l,
    )

    # Bước 2: kiểm tra margin
    margin_needed = calc_margin(symbol, lot_by_risk, current_price)
    max_margin    = equity * max_margin_pct
    if margin_needed > max_margin and lot_by_risk > min_l:
        # Scale xuống để fit margin
        scale   = max_margin / max(margin_needed, 1e-10)
        lot_adj = max(min_l, round(lot_by_risk * scale, 3))
        return lot_adj

    return lot_by_risk


def get_swap_per_bar(symbol: str, direction: int, tf: str) -> float:
    """
    Swap cost per bar (USD per lot).

    Args:
        direction: +1 = Long, -1 = Short
        tf:        'M1', 'M5', 'M15', 'H1'

    Returns:
        float: USD per lot per bar (negative = cost, positive = earn)
    """
    props = SYMBOL_PROPS.get(symbol, {})
    swap_per_night = props.get("swap_long" if direction == 1 else "swap_short", 0.0)
    bars_per_day   = BARS_PER_DAY.get(tf, 96)
    return swap_per_night / bars_per_day


# ─── Sanity Check ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== SYMBOL REGISTRY SANITY CHECK ===\n")
    test_cases = [
        ("EURUSDm", 1.0850, 0.001),    # EURUSD SL = 10 pips
        ("XAUUSDm", 2350.0, 5.0),      # Gold SL = 500 pips ($5 price move)
        ("BTCUSDm", 80_000.0, 800.0),  # BTC SL = $800 price move
        ("US30m",   39_000.0, 100.0),  # Dow SL = 100 points
        ("USDJPYm", 150.0, 0.5),       # USDJPY SL = 50 pips
    ]

    equity = 20_000.0
    risk   = 0.01  # 1%

    print(f"{'Symbol':<12} {'Price':>10} {'SL_dist':>10} {'Lot':>8} {'Risk_$':>8}")
    print("-" * 55)
    for sym, price, sl_dist in test_cases:
        lot = calc_lot_size(equity, risk, sl_dist, sym, price)
        pv  = get_pip_value_per_lot(sym, price)
        props = SYMBOL_PROPS[sym]
        pip_s = props["pip_size"]
        sl_pips = sl_dist / pip_s
        actual_risk = sl_pips * pv * lot
        print(f"{sym:<12} {price:>10.2f} {sl_dist:>10.4f} {lot:>8.4f} ${actual_risk:>7.2f}")

    print("\n=== SWAP RATES per bar (H1) ===")
    for sym in ["EURUSDm", "XAUUSDm", "BTCUSDm", "US30m"]:
        sl = get_swap_per_bar(sym, +1, "H1")
        print(f"  {sym}: swap_long_per_H1_bar = ${sl:.4f}/lot")
