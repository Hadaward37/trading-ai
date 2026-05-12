from __future__ import annotations

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────

ATR_PERIOD = 14
ATR_ROLL_WINDOW = 20
EMA_FAST = 20
EMA_SLOW = 50
EMA_TREND = 200
SLOPE_SHORT_LAG = 5
SLOPE_MID_LAG = 20
VOL_PERIOD = 20
BB_PERIOD = 20
RSI_PERIOD = 14


# ── Internal helpers ───────────────────────────────────────────────────────────

def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names to title case."""
    rename = {c: c.title() for c in df.columns if c.lower() in ("open", "high", "low", "close", "volume")}
    return df.rename(columns=rename)


def _true_range(df: pd.DataFrame) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift(1)).abs()
    lc = (df["Low"] - df["Close"].shift(1)).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


# ── Feature computations ───────────────────────────────────────────────────────

def _add_atr(df: pd.DataFrame) -> None:
    tr = _true_range(df)
    df["atr"] = tr.rolling(ATR_PERIOD, min_periods=1).mean()
    df["atr14"] = df["atr"]


def _add_atr_ratio(df: pd.DataFrame) -> None:
    if "atr" not in df.columns:
        _add_atr(df)
    atr_mean = df["atr"].rolling(ATR_ROLL_WINDOW, min_periods=5).mean()
    df["atr_ratio"] = (df["atr"] / atr_mean.replace(0, np.nan)).clip(0.2, 4.0)


def _add_ema_distance(df: pd.DataFrame) -> None:
    """Signed distance from EMA50 expressed in ATR units."""
    if "atr" not in df.columns:
        _add_atr(df)
    ema50 = df["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
    df["ema50"] = ema50
    df["ema_distance"] = ((df["Close"] - ema50) / df["atr"].replace(0, np.nan)).clip(-10, 10)
    df["signed_distance_to_mean"] = df["ema_distance"]
    df["stretch_abs"] = df["ema_distance"].abs()


def _add_ema_slope(df: pd.DataFrame) -> None:
    """Dual-slope market bias using EMA200 (causal)."""
    ema200 = df["Close"].ewm(span=EMA_TREND, adjust=False).mean()
    df["ema200"] = ema200
    df["slope_short"] = ema200.shift(1) - ema200.shift(1 + SLOPE_SHORT_LAG)
    df["slope_mid"] = ema200.shift(1) - ema200.shift(1 + SLOPE_MID_LAG)

    def _bias(row) -> int:
        if pd.isna(row["slope_short"]) or pd.isna(row["slope_mid"]):
            return 0
        if row["slope_short"] > 0 and row["slope_mid"] > 0:
            return 1
        if row["slope_short"] < 0 and row["slope_mid"] < 0:
            return -1
        return 0

    df["market_bias"] = df.apply(_bias, axis=1)
    df["ema_slope"] = df["slope_short"]


def _add_return_rate(df: pd.DataFrame) -> None:
    df["return_rate"] = df["Close"].pct_change(1)
    df["return_5"] = df["Close"].pct_change(5)
    df["return_20"] = df["Close"].pct_change(20)


def _add_volatility(df: pd.DataFrame) -> None:
    if "return_rate" not in df.columns:
        _add_return_rate(df)
    df["volatility"] = df["return_rate"].rolling(VOL_PERIOD, min_periods=5).std()
    df["realized_vol"] = df["volatility"]


def _add_rolling_mean_deviation(df: pd.DataFrame) -> None:
    roll_mean = df["Close"].rolling(VOL_PERIOD, min_periods=5).mean()
    df["rolling_mean_deviation"] = (df["Close"] - roll_mean).abs()


def _add_momentum(df: pd.DataFrame) -> None:
    df["momentum"] = df["Close"] - df["Close"].shift(10)


def _add_bb_features(df: pd.DataFrame) -> None:
    mid = df["Close"].rolling(BB_PERIOD, min_periods=5).mean()
    std = df["Close"].rolling(BB_PERIOD, min_periods=5).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    df["bb_width"] = (upper - lower) / mid.replace(0, np.nan)
    df["bb_position"] = ((df["Close"] - lower) / (upper - lower).replace(0, np.nan)).clip(0, 1)


def _add_rsi_proxy(df: pd.DataFrame) -> None:
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(RSI_PERIOD, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(RSI_PERIOD, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_proxy"] = 100 - (100 / (1 + rs))


def _add_body_ratio(df: pd.DataFrame) -> None:
    body = (df["Close"] - df["Open"]).abs()
    range_ = (df["High"] - df["Low"]).replace(0, np.nan)
    df["body_ratio"] = (body / range_).clip(0, 1)


def _add_ema20_features(df: pd.DataFrame) -> None:
    """EMA20 slope and distance — for trend-following entry signals."""
    if "atr" not in df.columns:
        _add_atr(df)
    ema20 = df["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema20"] = ema20
    df["ema20_slope"] = ema20 - ema20.shift(5)   # 5-bar slope
    df["ema20_distance"] = (
        (df["Close"] - ema20) / df["atr"].replace(0, np.nan)
    ).clip(-10, 10)


def _add_disorder_proxy(df: pd.DataFrame) -> None:
    """Rolling direction-flip rate (bar-by-bar chaos measure)."""
    c2c = df["Close"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    flips = (c2c != c2c.shift(1)) & (c2c != 0) & (c2c.shift(1) != 0)
    df["disorder_proxy"] = flips.rolling(12, min_periods=4).mean()


# ── Feature registry ───────────────────────────────────────────────────────────

_FEATURE_BUILDERS = [
    _add_atr,
    _add_atr_ratio,
    _add_ema_distance,
    _add_ema_slope,
    _add_ema20_features,
    _add_return_rate,
    _add_volatility,
    _add_rolling_mean_deviation,
    _add_momentum,
    _add_bb_features,
    _add_rsi_proxy,
    _add_body_ratio,
    _add_disorder_proxy,
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all standard research features to an OHLCV DataFrame.

    Input: DataFrame with Open/High/Low/Close columns (title or lower case).
    Output: DataFrame with original columns plus computed features.
    All calculations are causal (no look-ahead).
    """
    df = _ensure_ohlcv(df).copy()
    for builder in _FEATURE_BUILDERS:
        builder(df)
    return df
