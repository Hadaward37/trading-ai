"""LSTM trainer — loads historical data from SQLite, trains the model, saves artifacts.

Usage:
    python -m core.lstm_trainer          # train if not trained yet
    python -m core.lstm_trainer --force  # force retrain
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import config
from core.lstm_model import (
    FEATURES, LOOKBACK, MODEL_PATH, N_FEATURES, SCALER_PATH,
    build_model, _build_features, invalidate_cache,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

METRICS_PATH = _ROOT / "models" / "lstm_metrics.json"
MIN_ROWS     = 500
EPOCHS       = 30
BATCH_SIZE   = 64
VAL_SPLIT    = 0.15


# ── Data loading ──────────────────────────────────────────────────────────────

def load_training_data() -> tuple[np.ndarray, np.ndarray]:
    """Load and prepare (X, y) from signals_1h SQLite table.

    X shape: (n_samples, LOOKBACK, N_FEATURES)
    y shape: (n_samples,)  — 1 if close[t+1] > close[t], else 0
    """
    import sqlite3
    db_path = config.DB_PATH
    conn    = sqlite3.connect(db_path)

    query = """
        SELECT s.datetime, s.close, s.rsi, s.macd_hist, s.bb_pct, s.atr,
               s.adx, s.stoch_k, o.volume,
               COALESCE(s.sentiment_confidence, 0.0) AS sentiment_confidence
        FROM signals_1h s
        JOIN ohlcv_1h o ON s.datetime = o.datetime
        ORDER BY s.datetime ASC
    """
    try:
        df = pd.read_sql_query(query, conn, index_col="datetime", parse_dates=["datetime"])
    except Exception as exc:
        conn.close()
        raise RuntimeError(f"Failed to load training data: {exc}") from exc
    conn.close()

    if len(df) < MIN_ROWS:
        raise RuntimeError(
            f"Insufficient data: {len(df)} rows (need >= {MIN_ROWS}). "
            "Run the scheduler to collect more candles."
        )

    logger.info("Loaded %d rows from signals_1h", len(df))

    # Build raw feature matrix (unscaled)
    feats_raw = _build_features(df)
    if feats_raw is None:
        raise RuntimeError("Feature extraction failed — check column names.")

    # Target: close[t+1] > close[t]
    close = df["close"].values
    y_raw = (close[1:] > close[:-1]).astype(np.float32)

    # Build sliding windows BEFORE scaling
    n_samples = len(feats_raw) - LOOKBACK - 1
    X_raw = np.zeros((n_samples, LOOKBACK, N_FEATURES), dtype=np.float32)
    y     = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        X_raw[i] = feats_raw[i : i + LOOKBACK]
        y[i]     = y_raw[i + LOOKBACK]

    logger.info("Dataset: %d samples | UP=%.1f%% DOWN=%.1f%%",
                n_samples, y.mean() * 100, (1 - y.mean()) * 100)

    # ── Chronological split BEFORE fitting scaler (prevents leakage) ──────────
    n_val   = int(n_samples * VAL_SPLIT)
    n_train = n_samples - n_val
    X_train_raw = X_raw[:n_train]
    X_val_raw   = X_raw[n_train:]
    y_train_raw = y[:n_train]
    y_val_raw   = y[n_train:]

    # Fit scaler ONLY on training windows (flattened to 2D)
    from sklearn.preprocessing import StandardScaler
    import joblib

    X_tr_2d = X_train_raw.reshape(-1, N_FEATURES)
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_2d).reshape(n_train, LOOKBACK, N_FEATURES).astype(np.float32)
    X_va_sc = scaler.transform(X_val_raw.reshape(-1, N_FEATURES)).reshape(n_val, LOOKBACK, N_FEATURES).astype(np.float32)

    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    logger.info("Scaler (fit on train only) saved to %s", SCALER_PATH)

    return X_tr_sc, y_train_raw, X_va_sc, y_val_raw


# ── Training ──────────────────────────────────────────────────────────────────

def train(force: bool = False) -> dict:
    """Train the LSTM model and return metrics dict.

    Skips training if the model already exists and force=False.
    """
    if MODEL_PATH.exists() and not force:
        logger.info("Model already exists at %s. Use --force to retrain.", MODEL_PATH)
        if METRICS_PATH.exists():
            with open(METRICS_PATH) as f:
                return json.load(f)
        return {}

    logger.info("Starting LSTM training...")
    X_train, y_train, X_val, y_val = load_training_data()
    n_train, n_val = len(X_train), len(X_val)
    logger.info("Train: %d | Val: %d", n_train, n_val)

    import tensorflow as tf

    model = build_model()
    model.summary(print_fn=logger.info)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH), save_format="h5")
    logger.info("Model saved to %s", MODEL_PATH)

    # Evaluate on validation set
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    best_epoch = int(np.argmax(history.history["val_accuracy"])) + 1

    metrics = {
        "trained_at":  datetime.now(timezone.utc).isoformat(),
        "n_train":     n_train,
        "n_val":       n_val,
        "val_accuracy": round(float(val_acc),  4),
        "val_loss":     round(float(val_loss), 4),
        "epochs_run":   len(history.history["loss"]),
        "best_epoch":   best_epoch,
        "accuracy_history": [round(v, 4) for v in history.history["val_accuracy"]],
        "loss_history":     [round(v, 4) for v in history.history["val_loss"]],
        "lookback":     LOOKBACK,
        "features":     FEATURES,
        "symbol":       config.MT5_SYMBOL,
        "timeframe":    "1h",
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", METRICS_PATH)

    invalidate_cache()
    return metrics


def should_retrain() -> bool:
    """Return True if model should be retrained (older than 7 days or missing)."""
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        return True
    try:
        with open(METRICS_PATH) as f:
            m = json.load(f)
        trained_at = datetime.fromisoformat(m["trained_at"])
        age_days   = (datetime.now(timezone.utc) - trained_at).days
        return age_days >= 7
    except Exception:
        return True


def load_metrics() -> dict:
    """Return metrics dict from disk, or empty dict if not available."""
    if not METRICS_PATH.exists():
        return {}
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    force = "--force" in sys.argv
    metrics = train(force=force)
    print("\n=== LSTM Training Complete ===")
    print(f"  Val Accuracy : {metrics.get('val_accuracy', '?'):.2%}")
    print(f"  Val Loss     : {metrics.get('val_loss', '?'):.4f}")
    print(f"  Best Epoch   : {metrics.get('best_epoch', '?')} / {metrics.get('epochs_run', '?')}")
    print(f"  Train samples: {metrics.get('n_train', '?')}")
    print(f"  Val samples  : {metrics.get('n_val', '?')}")
    print(f"  Model saved  : {MODEL_PATH}")
