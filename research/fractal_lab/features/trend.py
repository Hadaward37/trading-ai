"""Trend-specific features — thin wrappers over base_features."""
from __future__ import annotations
import pandas as pd
from .base_features import compute_features


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add only trend-family features.
    Returns df with: ema200, slope_short, slope_mid, market_bias, ema_slope, momentum.
    """
    df = compute_features(df)
    return df[list(df.columns)]
