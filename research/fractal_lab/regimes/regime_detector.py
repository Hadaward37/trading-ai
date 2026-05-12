from __future__ import annotations

import numpy as np
import pandas as pd

# ── Regime labels ──────────────────────────────────────────────────────────────

TRENDING = "TRENDING"
MEAN_REVERTING = "MEAN_REVERTING"
HIGH_VOLATILITY = "HIGH_VOLATILITY"
LOW_VOLATILITY = "LOW_VOLATILITY"

ALL_REGIMES = [TRENDING, MEAN_REVERTING, HIGH_VOLATILITY, LOW_VOLATILITY]

# ── Classification thresholds ──────────────────────────────────────────────────

ATR_RATIO_HIGH_VOL = 1.4        # ATR > 1.4× rolling mean → high volatility
ATR_RATIO_LOW_VOL = 0.75        # ATR < 0.75× rolling mean → low volatility
SLOPE_TREND_THRESHOLD = 0.0002  # |ema_slope| > this → trending (normalised)
STRETCH_MR_THRESHOLD = 1.0      # |ema_distance| > 1 ATR → mean reversion candidate
DISORDER_HIGH = 0.55            # flip rate > 55% → chaotic (high vol override)


def classify_regime(row: pd.Series) -> str:
    """
    Classify a single bar into one of four regimes.

    Priority order: HIGH_VOLATILITY → LOW_VOLATILITY → TRENDING → MEAN_REVERTING

    Uses: atr_ratio, slope_short, slope_mid, ema_distance, disorder_proxy.
    Falls back to safe defaults when features are NaN.
    """
    atr_ratio = float(row.get("atr_ratio", 1.0))
    slope_short = float(row.get("slope_short", 0.0) or 0.0)
    slope_mid = float(row.get("slope_mid", 0.0) or 0.0)
    ema_dist_abs = abs(float(row.get("ema_distance", 0.0) or 0.0))
    disorder = float(row.get("disorder_proxy", 0.4) or 0.4)

    # HIGH_VOLATILITY: extreme ATR expansion or chaotic price action
    if atr_ratio > ATR_RATIO_HIGH_VOL or disorder > DISORDER_HIGH:
        return HIGH_VOLATILITY

    # LOW_VOLATILITY: compressed ATR
    if atr_ratio < ATR_RATIO_LOW_VOL:
        return LOW_VOLATILITY

    # Slope alignment in same direction → trending
    slopes_aligned = (slope_short > 0 and slope_mid > 0) or (slope_short < 0 and slope_mid < 0)
    slope_magnitude = max(abs(slope_short), abs(slope_mid))
    if slopes_aligned and slope_magnitude > SLOPE_TREND_THRESHOLD:
        return TRENDING

    # Price displaced from mean → mean reversion setup
    if ema_dist_abs > STRETCH_MR_THRESHOLD:
        return MEAN_REVERTING

    # Default: mean reverting (normal operating conditions)
    return MEAN_REVERTING


def tag_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'regime' column to a DataFrame that already has base features.
    Requires: atr_ratio, slope_short, slope_mid, ema_distance, disorder_proxy.
    """
    df = df.copy()
    df["regime"] = df.apply(classify_regime, axis=1)
    return df


def regime_counts(df: pd.DataFrame) -> dict:
    """Return regime distribution as percentage dict."""
    if "regime" not in df.columns:
        return {}
    counts = df["regime"].value_counts()
    total = len(df)
    return {r: round(counts.get(r, 0) / total, 4) for r in ALL_REGIMES}
