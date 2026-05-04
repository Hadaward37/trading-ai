"""Unit tests for the indicator and signal pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.indicators import add_all_indicators
from core.signals import generate_signals


def _synthetic_ohlcv(n: int = 200) -> pd.DataFrame:
    """Deterministic synthetic EUR/USD-like OHLCV data."""
    rng = np.random.default_rng(42)
    close = 1.08 + np.cumsum(rng.standard_normal(n) * 0.001)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    noise = rng.uniform(0, 0.002, n)
    return pd.DataFrame(
        {
            "open":   close * (1 - noise / 2),
            "high":   close * (1 + noise),
            "low":    close * (1 - noise),
            "close":  close,
            "volume": rng.integers(1_000, 10_000, n).astype(float),
        },
        index=idx,
    )


class TestIndicators:
    def test_all_columns_present(self):
        df = add_all_indicators(_synthetic_ohlcv())
        expected = ["rsi", "macd", "macd_signal", "macd_hist",
                    "bb_upper", "bb_mid", "bb_lower", "bb_pct", "atr"]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_rsi_bounds(self):
        df = add_all_indicators(_synthetic_ohlcv())
        rsi = df["rsi"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_bollinger_ordering(self):
        df = add_all_indicators(_synthetic_ohlcv()).dropna()
        assert (df["bb_upper"] >= df["bb_mid"]).all()
        assert (df["bb_mid"]   >= df["bb_lower"]).all()

    def test_atr_non_negative(self):
        df = add_all_indicators(_synthetic_ohlcv()).dropna()
        assert (df["atr"] >= 0).all()


class TestSignals:
    def test_signal_values(self):
        df = add_all_indicators(_synthetic_ohlcv())
        df = generate_signals(df)
        assert set(df["signal"].unique()).issubset({-1, 0, 1})

    def test_no_nan_in_signal(self):
        df = add_all_indicators(_synthetic_ohlcv())
        df = generate_signals(df)
        assert df["signal"].isna().sum() == 0
