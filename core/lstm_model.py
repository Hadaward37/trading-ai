"""LSTM model for next-candle direction prediction.

Architecture  : LSTM(64) → Dropout → LSTM(32) → Dropout → Dense(1, sigmoid)
Features (8)  : RSI, MACD_hist, BB_pct, ATR_norm, ADX, Stoch_K, Volume_norm, Sentiment_conf
Target        : close[t+1] > close[t]  → 1 (UP) | 0 (DOWN)
Lookback      : 60 candles
Output        : predict(df) → float 0-100  (probability of UP move × 100)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # silence TF noise
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

LOOKBACK   = 60
MODEL_PATH  = _ROOT / "models" / "lstm_eurusd_1h.h5"
ONNX_PATH   = _ROOT / "models" / "lstm_eurusd_1h.onnx"   # backend leve p/ VM sem TF
SCALER_PATH = _ROOT / "models" / "lstm_scaler.pkl"

FEATURES = [
    "rsi",               # momentum oscillator 0-100
    "macd_hist",         # MACD histogram (direction & strength)
    "bb_pct",            # Bollinger Band % position
    "atr_norm",          # ATR / close  (volatility, price-normalised)
    "adx",               # trend strength
    "stoch_k",           # stochastic %K
    "volume_norm",       # volume z-score
    "sentiment_confidence",  # Groq/GPT sentiment confidence 0-1
]

N_FEATURES = len(FEATURES)


# ── Feature engineering ───────────────────────────────────────────────────────

def _build_features(df: pd.DataFrame) -> Optional[np.ndarray]:
    """Return float32 array (n_rows, N_FEATURES) or None if data is insufficient."""
    required = {"rsi", "macd_hist", "bb_pct", "atr", "adx", "stoch_k", "close", "volume"}
    missing  = required - set(df.columns)
    if missing:
        logger.warning("LSTM: missing columns %s", missing)
        return None
    if len(df) < LOOKBACK + 1:
        logger.warning("LSTM: need at least %d rows, got %d", LOOKBACK + 1, len(df))
        return None

    out = df.copy()
    out["atr_norm"]    = (out["atr"] / out["close"]).fillna(0)
    out["volume_norm"] = (
        (out["volume"] - out["volume"].rolling(60, min_periods=1).mean())
        / (out["volume"].rolling(60, min_periods=1).std().replace(0, 1))
    ).fillna(0)
    out["sentiment_confidence"] = out.get("sentiment_confidence", pd.Series(0.0, index=out.index))

    return out[FEATURES].fillna(0).astype(np.float32).values


# ── Model builder ─────────────────────────────────────────────────────────────

def build_model() -> "tf.keras.Model":  # noqa: F821
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(LOOKBACK, N_FEATURES)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1,  activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Persistence ───────────────────────────────────────────────────────────────

def load_model():
    """Load trained model from disk. Returns None if not found."""
    if not MODEL_PATH.exists():
        logger.info("LSTM model not found at %s — train first.", MODEL_PATH)
        return None
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(str(MODEL_PATH))
        logger.info("LSTM model loaded from %s", MODEL_PATH)
        return model
    except Exception as exc:
        logger.error("Failed to load LSTM model: %s", exc)
        return None


def load_scaler():
    """Load fitted StandardScaler from disk. Returns None if not found."""
    if not SCALER_PATH.exists():
        return None
    try:
        import joblib
        return joblib.load(SCALER_PATH)
    except Exception as exc:
        logger.error("Failed to load LSTM scaler: %s", exc)
        return None


def load_onnx_session():
    """Load ONNX inference session (lightweight, no tensorflow). None if absent."""
    if not ONNX_PATH.exists():
        return None
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
        logger.info("LSTM ONNX session loaded from %s", ONNX_PATH)
        return sess
    except Exception as exc:
        logger.warning("Failed to load LSTM ONNX session: %s", exc)
        return None


# ── Public API ────────────────────────────────────────────────────────────────

# Module-level cache (loaded once per process)
_model   = None      # Keras model (quando TF disponível)
_session = None      # ONNX session (quando TF ausente — ex: VM 1GB)
_in_name = None      # nome do input ONNX
_scaler  = None
_loaded  = False
_backend = None      # "keras" | "onnx" | None


def _ensure_loaded() -> bool:
    """Carrega o backend: Keras se houver tensorflow (fonte de verdade local),
    senão ONNX (leve, p/ VM sem TF). Cacheado por processo."""
    global _model, _session, _in_name, _scaler, _loaded, _backend
    if _loaded:
        return _backend is not None

    _scaler = load_scaler()

    tf_available = True
    try:
        import tensorflow  # noqa: F401
    except Exception:
        tf_available = False

    if tf_available:
        _model = load_model()
        if _model is not None:
            _backend = "keras"

    if _backend is None:   # sem TF ou Keras falhou → tenta ONNX
        _session = load_onnx_session()
        if _session is not None:
            _in_name = _session.get_inputs()[0].name
            _backend = "onnx"

    _loaded = True
    if _backend:
        logger.info("LSTM backend: %s", _backend)
    return _backend is not None


def predict(df: pd.DataFrame) -> float:
    """Return probability of next-candle UP move as 0-100.

    Usa Keras ou ONNX (o que estiver disponível). Falls back to 50.0 (neutral)
    if no backend is available or data is insufficient.
    """
    if not _ensure_loaded():
        return 50.0

    feats = _build_features(df)
    if feats is None:
        return 50.0

    try:
        if _scaler is not None:
            feats = _scaler.transform(feats)

        # Take the last LOOKBACK rows
        window = feats[-LOOKBACK:].astype(np.float32)
        X = window.reshape(1, LOOKBACK, N_FEATURES)

        if _backend == "onnx":
            prob = float(_session.run(None, {_in_name: X})[0].ravel()[0])
        else:
            prob = float(_model.predict(X, verbose=0)[0][0])
        return round(prob * 100, 1)

    except Exception as exc:
        logger.error("LSTM predict error: %s", exc)
        return 50.0


def invalidate_cache() -> None:
    """Force reload on next predict() call (used after retraining)."""
    global _loaded, _backend, _model, _session
    _loaded = False
    _backend = None
    _model = None
    _session = None
