# -*- coding: utf-8 -*-
"""
evaluate_v17.py -- X-QUANG TU DUY cho XGBoost V17.0
=====================================================
Bao cao cho Sep Vu:
    1. Load model xgb_all_v17.pkl
    2. TOP 20 Feature Importance (Gain + Weight)
    3. Confusion Matrix + Precision/Recall tren Test set (20% cuoi)

Usage:
    python evaluate_v17.py
"""

from __future__ import annotations

import io
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ["PYTHONIOENCODING"] = "utf-8"

# ---------------------------------------------------------------------------
# Setup paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR   = PROJECT_ROOT / "data"
MODEL_PATH = DATA_DIR / "models" / "xgb_all_v17.pkl"

# Import from project
from core.feature_engine import build_feature_matrix, _atr, N_FEATURES
from core.xgb_classifier import build_labels

# Patch all StreamHandlers in FeatureEngine logger to use UTF-8 stdout
import logging
for handler in logging.getLogger("FeatureEngine").handlers:
    if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
        handler.stream = sys.stdout

# ---------------------------------------------------------------------------
# Constants (same as train_v17.py)
# ---------------------------------------------------------------------------

SYMBOLS = ["BTCUSD", "ETHUSD", "US30", "USTEC", "XAUUSD"]
SPREAD_COSTS = {
    "BTCUSD": 150.0,
    "ETHUSD": 8.0,
    "US30":   0.03,
    "USTEC":  0.03,
    "XAUUSD": 20.0,
}
LABEL_LOOKAHEAD = 3
LABEL_RR        = 1.5
WARMUP          = 200
TEST_RATIO      = 0.20   # 20% cuoi lam test set

LABEL_MAP = {0: "HOLD", 1: "BUY", 2: "SELL"}

# ---------------------------------------------------------------------------
# STEP 1: Load Model
# ---------------------------------------------------------------------------

def load_model():
    print("=" * 70)
    print("  STEP 1: LOAD MODEL -- xgb_all_v17.pkl")
    print("=" * 70)

    if not MODEL_PATH.exists():
        print(f"  [FATAL] Model not found: {MODEL_PATH}")
        sys.exit(1)

    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)

    print(f"  File:       {MODEL_PATH}")
    print(f"  File size:  {MODEL_PATH.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Bundle keys: {list(bundle.keys())}")
    print(f"  Version:    {bundle.get('version', 'N/A')}")
    print(f"  N Features: {bundle.get('n_features', 'N/A')}")
    print(f"  Samples:    {bundle.get('samples', 'N/A')}")
    print(f"  Trained at: {bundle.get('trained_at', 'N/A')}")
    if 'label_dist' in bundle:
        ld = bundle['label_dist']
        total = sum(ld.values())
        print(f"  Label Dist: BUY={ld.get('BUY',0)} ({ld.get('BUY',0)/total*100:.1f}%) | "
              f"SELL={ld.get('SELL',0)} ({ld.get('SELL',0)/total*100:.1f}%) | "
              f"HOLD={ld.get('HOLD',0)} ({ld.get('HOLD',0)/total*100:.1f}%)")
    if 'params' in bundle:
        p = bundle['params']
        print(f"  Params: max_depth={p.get('max_depth')}, n_estimators={p.get('n_estimators')}, "
              f"lr={p.get('learning_rate')}, subsample={p.get('subsample')}")
    print()

    model = bundle["model"]
    features = bundle.get("features", [])
    return model, features, bundle


# ---------------------------------------------------------------------------
# STEP 2: Feature Importance (Top 20)
# ---------------------------------------------------------------------------

def show_feature_importance(model, feature_names: list[str]):
    print("=" * 70)
    print("  STEP 2: BANG XEP HANG VU KHI -- TOP 20 FEATURE IMPORTANCE")
    print("=" * 70)

    # === GAIN (Total Gain) ===
    try:
        gain_dict = model.get_booster().get_score(importance_type="gain")
    except Exception:
        gain_dict = {}

    # === WEIGHT (Frequency -- so lan duoc dung de split) ===
    try:
        weight_dict = model.get_booster().get_score(importance_type="weight")
    except Exception:
        weight_dict = {}

    # Map f0, f1, ... -> actual feature names
    def map_names(score_dict):
        result = {}
        for key, val in score_dict.items():
            if key.startswith("f"):
                try:
                    idx = int(key[1:])
                    if idx < len(feature_names):
                        result[feature_names[idx]] = val
                    else:
                        result[key] = val
                except ValueError:
                    result[key] = val
            else:
                result[key] = val
        return result

    gain_mapped = map_names(gain_dict)
    weight_mapped = map_names(weight_dict)

    # Sort and display
    gain_sorted = sorted(gain_mapped.items(), key=lambda x: x[1], reverse=True)
    weight_sorted = sorted(weight_mapped.items(), key=lambda x: x[1], reverse=True)

    print()
    print("  --- TOP 20 by GAIN (tong gain khi feature duoc dung) ---")
    print(f"  {'#':>3} | {'Feature Name':<30} | {'Gain':>15}")
    print(f"  {'---':>3}-+-{'---':<30}-+-{'---':>15}")
    for i, (name, gain) in enumerate(gain_sorted[:20], 1):
        marker = " ***" if i <= 5 else ""
        print(f"  {i:>3} | {name:<30} | {gain:>15.2f}{marker}")
    print()

    print("  --- TOP 20 by WEIGHT (so lan duoc chon lam split) ---")
    print(f"  {'#':>3} | {'Feature Name':<30} | {'Weight':>15}")
    print(f"  {'---':>3}-+-{'---':<30}-+-{'---':>15}")
    for i, (name, weight) in enumerate(weight_sorted[:20], 1):
        marker = " ***" if i <= 5 else ""
        print(f"  {i:>3} | {name:<30} | {weight:>15.0f}{marker}")
    print()

    # === KEY WEAPONS CHECK ===
    print("  --- KIEM TRA VU KHI COT LOI ---")
    core_weapons = [
        ("fvg_size_bull_m5", "FVG Bull Size"),
        ("fvg_size_bear_m5", "FVG Bear Size"),
        ("fvg_bull_m5",      "FVG Bull Signal"),
        ("fvg_bear_m5",      "FVG Bear Signal"),
        ("volume_ratio_m5",  "Volume Ratio M5"),
        ("relative_vol",     "Relative Volume (VSA)"),
        ("pinbar_m5",        "Pinbar M5"),
        ("pinbar_m1",        "Pinbar M1"),
        ("session_vol_percentile", "Session Vol Percentile"),
        ("session_vol_rank", "Session Vol Rank"),
        ("vol_acceleration", "Vol Acceleration"),
        ("dist_to_bull_ob",  "Dist to Bull OB"),
        ("dist_to_bear_ob",  "Dist to Bear OB"),
        ("judas_swing",      "Judas Swing"),
        ("in_killzone",      "In Killzone"),
        ("fib_ote_level",    "Fib OTE Level"),
        ("dist_eql_norm",    "Dist EQL"),
        ("dist_eqh_norm",    "Dist EQH"),
    ]

    print(f"  {'Feature':<30} | {'Gain Rank':>10} | {'Weight Rank':>12} | Status")
    print(f"  {'-'*30}-+-{'-'*10}-+-{'-'*12}-+-{'-'*20}")

    gain_ranking = {name: i+1 for i, (name, _) in enumerate(gain_sorted)}
    weight_ranking = {name: i+1 for i, (name, _) in enumerate(weight_sorted)}

    for feat_key, feat_label in core_weapons:
        g_rank = gain_ranking.get(feat_key, "-")
        w_rank = weight_ranking.get(feat_key, "-")
        if isinstance(g_rank, int) and g_rank <= 20:
            status = "[*] TOP 20 GAIN"
        elif isinstance(w_rank, int) and w_rank <= 20:
            status = "[*] TOP 20 WEIGHT"
        elif isinstance(g_rank, int) and g_rank <= 40:
            status = "[+] TOP 40"
        elif isinstance(g_rank, int):
            status = f"[ ] Rank {g_rank}"
        else:
            status = "[-] NOT USED"
        print(f"  {feat_label:<30} | {str(g_rank):>10} | {str(w_rank):>12} | {status}")
    print()

    return gain_sorted, weight_sorted


# ---------------------------------------------------------------------------
# STEP 3: Test Set Evaluation
# ---------------------------------------------------------------------------

def build_dataset():
    """Build unified X, y from all symbols (same as train_v17.py)"""
    all_X = []
    all_y = []

    for sym in SYMBOLS:
        spread = SPREAD_COSTS.get(sym, 0.00015)
        try:
            features, raw_m5, h1_ib = build_feature_matrix(
                symbol=sym,
                data_dir=str(DATA_DIR),
                spread_cost=spread,
            )
        except Exception as e:
            print(f"  [SKIP] {sym}: {e}")
            continue

        if features.shape[1] != N_FEATURES:
            print(f"  [SKIP] {sym}: Feature count mismatch! "
                  f"Got {features.shape[1]}, expected {N_FEATURES}")
            continue

        atr = _atr(raw_m5[:, 2], raw_m5[:, 3], raw_m5[:, 4], period=14)
        labels = build_labels(raw_m5, atr, lookahead=LABEL_LOOKAHEAD, rr=LABEL_RR)

        if len(features) > WARMUP:
            features = features[WARMUP:]
            labels = labels[WARMUP:]

        n_buy  = (labels == 1).sum()
        n_sell = (labels == 2).sum()
        n_hold = (labels == 0).sum()
        print(f"  {sym}: {len(features)} samples | "
              f"BUY={n_buy} ({n_buy/len(labels)*100:.1f}%) | "
              f"SELL={n_sell} ({n_sell/len(labels)*100:.1f}%) | "
              f"HOLD={n_hold} ({n_hold/len(labels)*100:.1f}%)")

        all_X.append(features)
        all_y.append(labels)

    X = np.vstack(all_X).astype(np.float32)
    y = np.concatenate(all_y).astype(np.int32)
    return X, y


def evaluate_on_test(model, feature_names):
    print("=" * 70)
    print("  STEP 3: BAI TEST THUC CHIEN -- Confusion Matrix & Win Rate")
    print("=" * 70)
    print()
    print("  Dang build dataset tu tat ca symbols...")
    print()

    t0 = time.time()
    X, y = build_dataset()
    elapsed = time.time() - t0
    print(f"\n  Dataset built: {X.shape[0]} samples x {X.shape[1]} features in {elapsed:.1f}s")

    # Split: 80% train (da thay), 20% cuoi = TEST (chua thay)
    # Dung split THEO THU TU THOI GIAN (time-series) -- khong shuffle!
    n_total = len(X)
    n_test  = int(n_total * TEST_RATIO)
    n_train = n_total - n_test

    X_test = X[n_train:]
    y_test = y[n_train:]

    print(f"  Train samples: {n_train:,} (80% dau -- da thay)")
    print(f"  Test samples:  {n_test:,}  (20% cuoi -- CHUA TUNG THAY)")
    print()

    # Predict
    print("  Dang chay predict tren test set...")
    t0 = time.time()
    y_pred = model.predict(X_test.astype(np.float32))
    elapsed = time.time() - t0
    print(f"  Predict xong: {elapsed:.2f}s")
    print()

    # Probabilities
    y_proba = model.predict_proba(X_test.astype(np.float32))

    # === CONFUSION MATRIX (manual -- khong can sklearn) ===
    n_classes = 3
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true, pred in zip(y_test, y_pred):
        cm[true][pred] += 1

    print("  --- CONFUSION MATRIX (True \\ Pred) ---")
    print(f"  {'':>15} | {'Pred HOLD':>12} | {'Pred BUY':>12} | {'Pred SELL':>12}")
    print(f"  {'-'*15}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    for i, label in enumerate(["True HOLD", "True BUY", "True SELL"]):
        row = cm[i]
        print(f"  {label:>15} | {row[0]:>12,} | {row[1]:>12,} | {row[2]:>12,}")
    print()

    # === PRECISION / RECALL / F1 per class ===
    print("  --- PRECISION / RECALL / F1-SCORE ---")
    print(f"  {'Class':>10} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10} | {'Support':>10}")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    precisions = []
    recalls = []
    f1s = []
    supports = []

    for i, label in enumerate(["HOLD", "BUY", "SELL"]):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

        print(f"  {label:>10} | {precision:>10.4f} | {recall:>10.4f} | {f1:>10.4f} | {support:>10,}")

    # Weighted average
    total_support = sum(supports)
    w_precision = sum(p * s for p, s in zip(precisions, supports)) / total_support
    w_recall = sum(r * s for r, s in zip(recalls, supports)) / total_support
    w_f1 = sum(f * s for f, s in zip(f1s, supports)) / total_support

    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    print(f"  {'W.Average':>10} | {w_precision:>10.4f} | {w_recall:>10.4f} | {w_f1:>10.4f} | {total_support:>10,}")
    print()

    # === WIN RATE ANALYSIS ===
    print("  --- WIN RATE ANALYSIS (Thuc chien) ---")

    # BUY signals
    buy_mask = y_pred == 1
    n_buy_signals = buy_mask.sum()
    if n_buy_signals > 0:
        buy_correct = ((y_pred == 1) & (y_test == 1)).sum()
        buy_win_rate = buy_correct / n_buy_signals * 100
        print(f"  BUY  signals: {n_buy_signals:>8,} | Correct: {buy_correct:>8,} | Win Rate: {buy_win_rate:.2f}%")
    else:
        print(f"  BUY  signals: 0 (khong phat tin hieu BUY nao)")

    # SELL signals
    sell_mask = y_pred == 2
    n_sell_signals = sell_mask.sum()
    if n_sell_signals > 0:
        sell_correct = ((y_pred == 2) & (y_test == 2)).sum()
        sell_win_rate = sell_correct / n_sell_signals * 100
        print(f"  SELL signals: {n_sell_signals:>8,} | Correct: {sell_correct:>8,} | Win Rate: {sell_win_rate:.2f}%")
    else:
        print(f"  SELL signals: 0 (khong phat tin hieu SELL nao)")

    # Overall accuracy
    overall_correct = (y_pred == y_test).sum()
    overall_acc = overall_correct / len(y_test) * 100
    print(f"\n  Overall Accuracy: {overall_correct:,} / {len(y_test):,} = {overall_acc:.2f}%")

    # BUY+SELL combined win rate (chi tinh signal trading, bo HOLD)
    trade_mask = y_pred != 0
    n_trades = trade_mask.sum()
    if n_trades > 0:
        trade_correct = ((y_pred != 0) & (y_pred == y_test)).sum()
        trade_win_rate = trade_correct / n_trades * 100
        print(f"  Trade Win Rate (BUY+SELL only): {trade_correct:,} / {n_trades:,} = {trade_win_rate:.2f}%")
    print()

    # === Confidence distribution ===
    print("  --- CONFIDENCE DISTRIBUTION (Avg probability) ---")
    for i, label in enumerate(["HOLD", "BUY", "SELL"]):
        mask = y_pred == i
        if mask.sum() > 0:
            avg_conf = y_proba[mask, i].mean()
            max_conf = y_proba[mask, i].max()
            min_conf = y_proba[mask, i].min()
            print(f"  {label}: avg={avg_conf:.4f} | max={max_conf:.4f} | min={min_conf:.4f} | count={mask.sum():,}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("=" * 70)
    print("  X-QUANG TU DUY -- RabitScal AI V17.0 EVALUATION")
    print("  Bao cao cho Sep Vu -- Chup X-Ray Nao AI")
    print("=" * 70)
    print()

    # Step 1: Load
    model, features, bundle = load_model()

    # Step 2: Feature Importance
    show_feature_importance(model, features)

    # Step 3: Test Evaluation
    evaluate_on_test(model, features)

    print("=" * 70)
    print("  X-QUANG HOAN TAT! Sep Vu kiem tra bao cao phia tren.")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
