"""Volatility-specific features — thin wrappers over base_features."""
from __future__ import annotations
import pandas as pd
from .base_features import compute_features


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add only volatility-family features.
    Returns df with: atr, atr14, atr_ratio, volatility, realized_vol, bb_width.
    """
    df = compute_features(df)
    return df[list(df.columns)]
