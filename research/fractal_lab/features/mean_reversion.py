"""Mean-reversion-specific features — thin wrappers over base_features."""
from __future__ import annotations
import pandas as pd
from .base_features import compute_features


def add_mr_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add mean-reversion features.
    Returns df with: ema50, ema_distance, signed_distance_to_mean, stretch_abs,
    rolling_mean_deviation, rsi_proxy, bb_position.
    """
    df = compute_features(df)
    return df[list(df.columns)]
