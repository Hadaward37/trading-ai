"""Technical indicator calculations using the *ta* library."""

from __future__ import annotations

import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD as MACDIndicator, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Append RSI column to *df*.

    Column added: ``rsi``
    """
    df = df.copy()
    df["rsi"] = RSIIndicator(close=df["close"], window=window).rsi()
    return df


def add_macd(
    df: pd.DataFrame,
    window_fast: int = 12,
    window_slow: int = 26,
    window_sign: int = 9,
) -> pd.DataFrame:
    """Append MACD columns to *df*.

    Columns added: ``macd``, ``macd_signal``, ``macd_hist``
    """
    df = df.copy()
    macd = MACDIndicator(
        close=df["close"],
        window_fast=window_fast,
        window_slow=window_slow,
        window_sign=window_sign,
    )
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    return df


def add_bollinger_bands(
    df: pd.DataFrame, window: int = 20, window_dev: float = 2.0
) -> pd.DataFrame:
    """Append Bollinger Bands columns to *df*.

    Columns added: ``bb_upper``, ``bb_mid``, ``bb_lower``, ``bb_pct``
    """
    df = df.copy()
    bb = BollingerBands(close=df["close"], window=window, window_dev=window_dev)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_pct"] = bb.bollinger_pband()
    return df


def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Append ATR column to *df*.

    Column added: ``atr``
    """
    df = df.copy()
    df["atr"] = AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=window
    ).average_true_range()
    return df


def add_adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Append ADX and directional indicator columns to *df*.

    Columns added: ``adx`` (trend strength 0-100), ``adx_pos`` (+DI),
    ``adx_neg`` (-DI).
    """
    df = df.copy()
    adx = ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=window
    )
    df["adx"]     = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()
    return df


def add_stochastic(
    df: pd.DataFrame,
    k_window: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> pd.DataFrame:
    """Append Stochastic Oscillator columns to *df*.

    Columns added: ``stoch_k`` (%K smoothed), ``stoch_d`` (%D — signal line).
    """
    df = df.copy()
    stoch = StochasticOscillator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=k_window,
        smooth_window=smooth_k,
    )
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    return df


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all indicators to *df* in one call.

    Computes: RSI, MACD, Bollinger Bands, ATR, ADX, Stochastic.
    """
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_adx(df)
    df = add_stochastic(df)
    return df
