"""
core/xgb_classifier.py — RabitScal XGBoost Signal Classifier v1.0
===================================================================
Binary classification: predict BUY / SELL / HOLD from live candle features.

Hardware target: GTX 750 Ti 4GB VRAM
    tree_method='hist', device='cuda', max_bin=64  (conservative for 4GB)

OOM Safety:
    - max_bin=64 (halved from gpu_config default of 128)
    - Batch training: 1 symbol at a time (never load 5 × 4 TF simultaneously)
    - CUDA OOM fallback: auto-retry on CPU

Protocol: only called from socket_bridge._run_ai_analysis() via asyncio.to_thread()
    → runs in thread pool, NEVER blocks event loop.
"""
from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("XGBClassifier")

# ── Constants ─────────────────────────────────────────────────────────────────

N_FEATURES     = 54      # Must match core/feature_engine.py N_FEATURES
MODEL_DIR      = Path(__file__).resolve().parent.parent / "data" / "models"
MIN_TRAIN_ROWS = 500     # Minimum candles needed to train

# ── GPU params for GTX 750 Ti 4GB VRAM ───────────────────────────────────────
_GPU_PARAMS = {
    "tree_method":    "hist",
    "device":         "cuda",
    "max_bin":        64,       # 64 (conservative) — avoids OOM on 4GB VRAM
    "max_depth":      6,
    "n_estimators":   300,
    "learning_rate":  0.05,
    "subsample":      0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma":          0.1,
    "reg_alpha":      0.1,
    "reg_lambda":     1.0,
    "objective":      "multi:softprob",
    "num_class":      3,        # 0=HOLD, 1=BUY, 2=SELL
    "eval_metric":    "mlogloss",
    "verbosity":      0,
    "n_jobs":         1,
}

_CPU_PARAMS = {**_GPU_PARAMS, "device": "cpu", "n_jobs": -1}

# Label map: model output → action string
LABEL_MAP = {0: "HOLD", 1: "BUY", 2: "SELL"}

# Confidence threshold: only emit BUY/SELL if confidence ≥ this
MIN_CONFIDENCE = 0.55


# ── Classifier ────────────────────────────────────────────────────────────────

class RabitScalClassifier:
    """
    XGBoost 3-class classifier: HOLD=0, BUY=1, SELL=2.

    Usage (training, typically offline):
        clf = RabitScalClassifier()
        clf.fit(X, y, symbol="XAUUSD")
        clf.save(symbol="XAUUSD")

    Usage (inference, from socket_bridge thread):
        clf = RabitScalClassifier.load(symbol="XAUUSD")
        result = clf.predict_single(feature_row)
        # returns {"action": "BUY", "confidence": 0.73}
    """

    def __init__(self):
        self._models: dict[str, object] = {}   # symbol → xgb.XGBClassifier
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        symbol: str,
        use_gpu: bool = True,
    ) -> "RabitScalClassifier":
        """
        Train XGBoost classifier for one symbol.

        Args:
            X: Feature matrix, shape (N, N_FEATURES) — from feature_engine.compute_features()
            y: Labels, shape (N,) — values in {0, 1, 2} (HOLD/BUY/SELL)
            symbol: Symbol name e.g. "XAUUSD"
            use_gpu: Try CUDA first, fall back to CPU on OOM
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost not installed. Run: pip install xgboost")

        if len(X) < MIN_TRAIN_ROWS:
            raise ValueError(
                f"[XGB] {symbol}: Only {len(X)} rows — need {MIN_TRAIN_ROWS}+ to train."
            )

        log.info(f"[XGB] Training {symbol}: {len(X)} rows, {X.shape[1]} features")
        t0 = time.time()

        params = _GPU_PARAMS.copy() if use_gpu else _CPU_PARAMS.copy()

        # Try GPU first; fall back to CPU on OOM
        for attempt, p in enumerate([params, _CPU_PARAMS]):
            mode = "CUDA" if attempt == 0 and use_gpu else "CPU"
            try:
                model = xgb.XGBClassifier(**p)
                model.fit(X.astype(np.float32), y.astype(np.int32))
                elapsed = time.time() - t0
                log.info(
                    f"[XGB] {symbol} trained ({mode}) in {elapsed:.1f}s | "
                    f"params: max_bin={p['max_bin']}, depth={p['max_depth']}"
                )
                self._models[symbol] = model
                return self
            except Exception as e:
                err_str = str(e).lower()
                if attempt == 0 and use_gpu and (
                    "cuda" in err_str or "oom" in err_str or "out of memory" in err_str
                    or "device" in err_str
                ):
                    log.warning(
                        f"[XGB] {symbol} GPU OOM/error: {e}. "
                        f"Retrying on CPU with n_jobs=-1..."
                    )
                    continue
                else:
                    raise

        raise RuntimeError(f"[XGB] {symbol}: Training failed on both GPU and CPU.")

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_single(
        self,
        feature_row: np.ndarray,
        symbol: str,
    ) -> dict:
        """
        Predict action for a single feature vector.

        Args:
            feature_row: shape (N_FEATURES,) or (1, N_FEATURES)
            symbol: Symbol name

        Returns:
            {"action": "BUY"|"SELL"|"HOLD", "confidence": float, "proba": list}
        """
        model = self._models.get(symbol)
        if model is None:
            log.debug(f"[XGB] No model for {symbol} — returning HOLD")
            return {"action": "HOLD", "confidence": 0.0, "proba": [1.0, 0.0, 0.0]}

        x = feature_row.reshape(1, -1).astype(np.float32)
        try:
            proba = model.predict_proba(x)[0]   # shape (3,): [P_HOLD, P_BUY, P_SELL]
        except Exception as e:
            log.error(f"[XGB] predict_single error for {symbol}: {e}")
            return {"action": "HOLD", "confidence": 0.0, "proba": [1.0, 0.0, 0.0]}

        label_idx   = int(np.argmax(proba))
        confidence  = float(proba[label_idx])
        action      = LABEL_MAP[label_idx]

        # Suppress low-confidence signals
        if action != "HOLD" and confidence < MIN_CONFIDENCE:
            log.debug(
                f"[XGB] {symbol}: {action} suppressed (confidence={confidence:.2f} "
                f"< {MIN_CONFIDENCE})"
            )
            action = "HOLD"

        return {
            "action":     action,
            "confidence": round(confidence, 4),
            "proba":      [round(float(p), 4) for p in proba],
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, symbol: str) -> Path:
        """Save model to disk as pickle."""
        model = self._models.get(symbol)
        if model is None:
            raise ValueError(f"[XGB] No model for {symbol} to save.")
        path = MODEL_DIR / f"xgb_{symbol}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        log.info(f"[XGB] Model saved: {path}")
        return path

    def load_symbol(self, symbol: str) -> bool:
        """Load model from disk for one symbol. Returns True if found."""
        path = MODEL_DIR / f"xgb_{symbol}.pkl"
        if not path.exists():
            log.warning(f"[XGB] Model file not found: {path}")
            return False
        try:
            with open(path, "rb") as f:
                self._models[symbol] = pickle.load(f)
            log.info(f"[XGB] Model loaded: {path}")
            return True
        except Exception as e:
            log.error(f"[XGB] Failed to load model {path}: {e}")
            return False

    def load_all(self, symbols: list[str]) -> int:
        """Load models for all symbols. Returns count loaded."""
        loaded = sum(self.load_symbol(s) for s in symbols)
        log.info(f"[XGB] Loaded {loaded}/{len(symbols)} models")
        return loaded

    @property
    def loaded_symbols(self) -> list[str]:
        return list(self._models.keys())


# ── Label Builder Helper ──────────────────────────────────────────────────────

def build_labels(
    data: np.ndarray,
    atr: np.ndarray,
    lookahead: int = 3,
    rr: float = 1.5,
) -> np.ndarray:
    """
    Generate training labels from OHLCV data using TP/SL outcome.

    Simple labeling:
        Look ahead `lookahead` candles.
        BUY=1  if price rises by > 1.0 × ATR before dropping by SL = ATR
        SELL=2 if price drops by > 1.0 × ATR before rising by SL = ATR
        HOLD=0 otherwise

    Args:
        data: np.ndarray shape (N, 6) — [time, O, H, L, C, V]
        atr:  np.ndarray shape (N,)   — ATR values aligned to data
        lookahead: candles to look forward
        rr: reward/risk ratio

    Returns:
        labels: np.ndarray shape (N,) dtype int32, values in {0,1,2}
    """
    N      = len(data)
    closes = data[:, 4]
    highs  = data[:, 2]
    lows   = data[:, 3]
    labels = np.zeros(N, dtype=np.int32)   # default HOLD

    for i in range(N - lookahead):
        sl_dist = atr[i]
        tp_dist = sl_dist * rr

        future_highs = highs[i + 1: i + 1 + lookahead]
        future_lows  = lows[i + 1: i + 1 + lookahead]

        buy_tp_hit  = np.any(future_highs >= closes[i] + tp_dist)
        buy_sl_hit  = np.any(future_lows  <= closes[i] - sl_dist)
        sell_tp_hit = np.any(future_lows  <= closes[i] - tp_dist)
        sell_sl_hit = np.any(future_highs >= closes[i] + sl_dist)

        if buy_tp_hit and not buy_sl_hit:
            labels[i] = 1   # BUY won
        elif sell_tp_hit and not sell_sl_hit:
            labels[i] = 2   # SELL won
        # else HOLD (0)

    return labels


# ── Singleton instance (shared across threads) ────────────────────────────────

_classifier: Optional[RabitScalClassifier] = None


def get_classifier() -> RabitScalClassifier:
    """Get or create the global classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = RabitScalClassifier()
    return _classifier
