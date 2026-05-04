"""Signal generation — combines RSI, MACD crossovers, and Bollinger Bands."""

from __future__ import annotations

import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# ── Thresholds (pulled from centralised config) ───────────────────────────────
BB_TOLERANCE: float = config.BB_TOLERANCE


def _compute_votes(
    df: pd.DataFrame, rsi_buy: int, rsi_sell: int
) -> tuple[pd.Series, pd.Series]:
    """Return (buy_votes, sell_votes) Series for each bar."""
    # RSI
    v_rsi_buy  = (df["rsi"] < rsi_buy).astype(int)
    v_rsi_sell = (df["rsi"] > rsi_sell).astype(int)

    # MACD crossover
    prev_macd = df["macd"].shift(1)
    prev_sig  = df["macd_signal"].shift(1)
    v_macd_up = ((df["macd"] > df["macd_signal"]) & (prev_macd <= prev_sig)).astype(int)
    v_macd_dn = ((df["macd"] < df["macd_signal"]) & (prev_macd >= prev_sig)).astype(int)

    # Bollinger Bands
    v_bb_buy  = (df["close"] <= df["bb_lower"] * (1 + BB_TOLERANCE)).astype(int)
    v_bb_sell = (df["close"] >= df["bb_upper"] * (1 - BB_TOLERANCE)).astype(int)

    buy_votes  = v_rsi_buy  + v_macd_up + v_bb_buy
    sell_votes = v_rsi_sell + v_macd_dn + v_bb_sell
    return buy_votes, sell_votes


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``signal`` column using optimised defaults from ``config``.

    Thresholds: RSI Buy < :data:`config.RSI_BUY`, Sell > :data:`config.RSI_SELL`.
    A signal fires when **at least 2 of 3** indicators agree.

    Signal values: ``1`` buy | ``-1`` sell | ``0`` neutral.

    Args:
        df: DataFrame produced by :func:`core.indicators.add_all_indicators`.

    Returns:
        Copy of *df* with the ``signal`` column appended.
    """
    return generate_signals_custom(df, config.RSI_BUY, config.RSI_SELL)


def generate_signals_custom(
    df: pd.DataFrame, rsi_buy: int, rsi_sell: int
) -> pd.DataFrame:
    """Like :func:`generate_signals` but with explicit RSI thresholds.

    Used by the dashboard (interactive sliders) and the optimizer (grid search).

    Args:
        df: DataFrame with indicator columns.
        rsi_buy:  Buy  threshold — signal when RSI < *rsi_buy*.
        rsi_sell: Sell threshold — signal when RSI > *rsi_sell*.

    Returns:
        Copy of *df* with the ``signal`` column appended.
    """
    df = df.copy()
    buy_votes, sell_votes = _compute_votes(df, rsi_buy, rsi_sell)
    df["signal"] = 0
    df.loc[buy_votes  >= 2, "signal"] =  1
    df.loc[sell_votes >= 2, "signal"] = -1
    return df
