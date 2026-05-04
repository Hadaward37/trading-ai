"""Signal generation — combines RSI, MACD crossovers, and Bollinger Bands."""

from __future__ import annotations

import pandas as pd

# ── Thresholds ────────────────────────────────────────────────────────────────
RSI_OVERSOLD: int = 35
RSI_OVERBOUGHT: int = 65
# Price within 0.2% of a band edge counts as a "touch"
BB_TOLERANCE: float = 0.002


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``signal`` column using a 3-indicator voting strategy.

    Each indicator casts a +1 (buy) or -1 (sell) vote.  A signal fires
    when **at least 2 of 3** indicators agree.

    Signal values:
    * ``1``  — buy
    * ``-1`` — sell
    * ``0``  — neutral / no consensus

    Args:
        df: DataFrame produced by :func:`core.indicators.add_all_indicators`.

    Returns:
        Copy of *df* with the ``signal`` column appended.
    """
    df = df.copy()

    # ── RSI votes ─────────────────────────────────────────────────────────────
    rsi_buy  = df["rsi"] < RSI_OVERSOLD
    rsi_sell = df["rsi"] > RSI_OVERBOUGHT

    # ── MACD crossover votes ──────────────────────────────────────────────────
    prev_macd = df["macd"].shift(1)
    prev_sig  = df["macd_signal"].shift(1)
    macd_cross_up = (df["macd"] > df["macd_signal"]) & (prev_macd <= prev_sig)
    macd_cross_dn = (df["macd"] < df["macd_signal"]) & (prev_macd >= prev_sig)

    # ── Bollinger Bands votes ─────────────────────────────────────────────────
    bb_buy  = df["close"] <= df["bb_lower"] * (1 + BB_TOLERANCE)
    bb_sell = df["close"] >= df["bb_upper"] * (1 - BB_TOLERANCE)

    # ── Tally ─────────────────────────────────────────────────────────────────
    buy_votes  = rsi_buy.astype(int)  + macd_cross_up.astype(int) + bb_buy.astype(int)
    sell_votes = rsi_sell.astype(int) + macd_cross_dn.astype(int) + bb_sell.astype(int)

    df["signal"] = 0
    df.loc[buy_votes  >= 2, "signal"] =  1
    df.loc[sell_votes >= 2, "signal"] = -1

    return df
