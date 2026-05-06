"""XGBoost model for next-candle direction prediction.

Features (24) : current + rolling-5 mean + rolling-5 std of the 8 LSTM features
Target        : close[t+1] > close[t]  → 1 (UP) | 0 (DOWN)
Output        : predict(df) → float 0-100  (probability of UP move × 100)

Training      : run via  python -m core.xgboost_model  [--force]
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config

logger = logging.getLogger(__name__)

MODEL_PATH   = _ROOT / "models" / "xgboost_eurusd.pkl"
METRICS_PATH = _ROOT / "models" / "xgboost_metrics.json"
SCALER_PATH  = _ROOT / "models" / "xgboost_scaler.pkl"

BASE_FEATURES = [
    "rsi",
    "macd_hist",
    "bb_pct",
    "atr_norm",
    "adx",
    "stoch_k",
    "volume_norm",
    "sentiment_confidence",
]
ROLL_WINDOW = 5
MIN_ROWS    = 300


# ── Feature engineering ───────────────────────────────────────────────────────

def _build_feature_matrix(df: pd.DataFrame) -> Optional[np.ndarray]:
    """Return (n_rows, 24) float32 array or None if insufficient data."""
    required = {"rsi", "macd_hist", "bb_pct", "atr", "adx", "stoch_k", "close", "volume"}
    missing  = required - set(df.columns)
    if missing:
        logger.warning("XGBoost: missing columns %s", missing)
        return None
    if len(df) < ROLL_WINDOW + 2:
        logger.warning("XGBoost: need at least %d rows, got %d", ROLL_WINDOW + 2, len(df))
        return None

    out = df.copy()
    out["atr_norm"]    = (out["atr"] / out["close"]).fillna(0)
    out["volume_norm"] = (
        (out["volume"] - out["volume"].rolling(60, min_periods=1).mean())
        / (out["volume"].rolling(60, min_periods=1).std().replace(0, 1))
    ).fillna(0)
    out["sentiment_confidence"] = out.get(
        "sentiment_confidence", pd.Series(0.0, index=out.index)
    )

    base = out[BASE_FEATURES].fillna(0).astype(np.float32)

    roll_mean = base.rolling(ROLL_WINDOW, min_periods=1).mean()
    roll_std  = base.rolling(ROLL_WINDOW, min_periods=1).std().fillna(0)

    combined = np.concatenate(
        [base.values, roll_mean.values, roll_std.values], axis=1
    )
    return combined.astype(np.float32)


# ── Persistence ───────────────────────────────────────────────────────────────

def _load_model():
    if not MODEL_PATH.exists():
        return None
    try:
        import joblib
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        logger.error("XGBoost load error: %s", exc)
        return None


def _load_scaler():
    if not SCALER_PATH.exists():
        return None
    try:
        import joblib
        return joblib.load(SCALER_PATH)
    except Exception as exc:
        logger.error("XGBoost scaler load error: %s", exc)
        return None


# ── Module-level cache ─────────────────────────────────────────────────────────

_model  = None
_scaler = None
_loaded = False


def _ensure_loaded() -> bool:
    global _model, _scaler, _loaded
    if _loaded:
        return _model is not None
    _model  = _load_model()
    _scaler = _load_scaler()
    _loaded = True
    return _model is not None


# ── Public predict API ────────────────────────────────────────────────────────

def predict(df: pd.DataFrame) -> float:
    """Return probability of next-candle UP move as 0-100. Falls back to 50.0."""
    if not getattr(config, "XGBOOST_ENABLED", False):
        return 50.0
    if not _ensure_loaded():
        return 50.0

    feats = _build_feature_matrix(df)
    if feats is None:
        return 50.0

    try:
        last = feats[-1:, :]  # shape (1, 24)
        if _scaler is not None:
            last = _scaler.transform(last)

        prob = float(_model.predict_proba(last)[0][1])  # class 1 = UP
        return round(prob * 100, 1)

    except Exception as exc:
        logger.error("XGBoost predict error: %s", exc)
        return 50.0


def invalidate_cache() -> None:
    global _loaded
    _loaded = False


# ── Training ──────────────────────────────────────────────────────────────────

def train(force: bool = False) -> dict:
    """Train XGBoost on signals_1h data and return metrics dict."""
    if MODEL_PATH.exists() and not force:
        logger.info("XGBoost model already exists. Use --force to retrain.")
        if METRICS_PATH.exists():
            with open(METRICS_PATH) as f:
                return json.load(f)
        return {}

    logger.info("Starting XGBoost training …")

    import sqlite3
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import joblib
    import xgboost as xgb

    conn  = sqlite3.connect(config.DB_PATH)
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
            "Collect more candles first."
        )

    logger.info("Loaded %d rows from signals_1h", len(df))

    feats = _build_feature_matrix(df)
    if feats is None:
        raise RuntimeError("Feature extraction failed.")

    # Target: close[t+1] > close[t]
    close = df["close"].values
    y = (close[1:] > close[:-1]).astype(int)
    X = feats[:-1]  # drop last row (no label for it)

    # Chronological split (no shuffle to avoid lookahead bias)
    n_val   = max(50, int(len(X) * 0.15))
    X_train, X_val = X[:-n_val], X[-n_val:]
    y_train, y_val = y[:-n_val], y[-n_val:]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)

    model = xgb.XGBClassifier(
        n_estimators     = 400,
        max_depth        = 5,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        use_label_encoder= False,
        eval_metric      = "logloss",
        early_stopping_rounds = 30,
        random_state     = 42,
        verbosity        = 1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    joblib.dump(model, MODEL_PATH)
    logger.info("XGBoost model saved to %s", MODEL_PATH)

    val_probs = model.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)
    accuracy  = float((val_preds == y_val).mean())
    up_rate   = float(y.mean())

    metrics = {
        "trained_at":    datetime.now(timezone.utc).isoformat(),
        "n_train":       int(len(X_train)),
        "n_val":         int(len(X_val)),
        "val_accuracy":  round(accuracy, 4),
        "up_rate":       round(up_rate, 4),
        "best_iteration": int(model.best_iteration),
        "features":      BASE_FEATURES,
        "n_features":    int(X_train.shape[1]),
        "symbol":        config.MT5_SYMBOL,
        "timeframe":     "1h",
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("XGBoost metrics saved to %s", METRICS_PATH)

    invalidate_cache()
    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    force   = "--force" in sys.argv
    metrics = train(force=force)
    print("\n=== XGBoost Training Complete ===")
    print(f"  Val Accuracy  : {metrics.get('val_accuracy', '?'):.2%}")
    print(f"  Best Iteration: {metrics.get('best_iteration', '?')}")
    print(f"  Train samples : {metrics.get('n_train', '?')}")
    print(f"  Val samples   : {metrics.get('n_val', '?')}")
    print(f"  Model saved   : {MODEL_PATH}")
