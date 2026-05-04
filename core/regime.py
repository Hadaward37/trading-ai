"""Market regime detection using ADX and relative ATR."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import pandas as pd

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config

MarketRegime = Literal["Tendencia Alta", "Tendencia Baixa", "Range", "Alta Volatilidade"]

_REGIME_COLORS: dict[str, str] = {
    "Tendencia Alta":    "#26a69a",
    "Tendencia Baixa":   "#ef5350",
    "Range":             "#ff9800",
    "Alta Volatilidade": "#9c27b0",
}

_REGIME_EMOJI: dict[str, str] = {
    "Tendencia Alta":    "📈",
    "Tendencia Baixa":   "📉",
    "Range":             "↔️",
    "Alta Volatilidade": "⚡",
}


def detect_regime(df: pd.DataFrame) -> MarketRegime:
    """Identify the current market regime from the last complete bar.

    Priority order:
    1. **Alta Volatilidade** — ATR > ``REGIME_VOLATILITY_THRESHOLD`` × ATR_50ma
    2. **Tendencia Alta**    — ADX >= ``REGIME_ADX_TREND`` and +DI > -DI
    3. **Tendencia Baixa**   — ADX >= ``REGIME_ADX_TREND`` and -DI > +DI
    4. **Range**             — ADX < ``REGIME_ADX_RANGE`` (or between thresholds)

    Args:
        df: DataFrame with ``adx``, ``adx_pos``, ``adx_neg``, ``atr`` columns.

    Returns:
        One of the four :data:`MarketRegime` literals.
    """
    clean = df.dropna(subset=["adx", "atr"])
    if clean.empty:
        return "Range"

    last    = clean.iloc[-1]
    adx     = float(last["adx"])
    adx_pos = float(last["adx_pos"])
    adx_neg = float(last["adx_neg"])

    # Relative ATR: current vs 50-bar rolling mean
    atr_mean = clean["atr"].rolling(50, min_periods=10).mean().iloc[-1]
    atr_ratio = float(last["atr"]) / atr_mean if atr_mean > 0 else 1.0

    if atr_ratio >= config.REGIME_VOLATILITY_THRESHOLD:
        return "Alta Volatilidade"
    if adx >= config.REGIME_ADX_TREND:
        return "Tendencia Alta" if adx_pos >= adx_neg else "Tendencia Baixa"
    return "Range"


def regime_color(regime: MarketRegime) -> str:
    """Hex colour string for the given regime."""
    return _REGIME_COLORS.get(regime, "#ffffff")


def regime_emoji(regime: MarketRegime) -> str:
    """Emoji for the given regime."""
    return _REGIME_EMOJI.get(regime, "")


def regime_blocks_signal(regime: MarketRegime) -> bool:
    """Return True if this regime should suppress new trade entries.

    Controlled by :data:`config.REGIME_BLOCK_RANGE_SIGNALS`.
    """
    if not config.REGIME_BLOCK_RANGE_SIGNALS:
        return False
    return regime == "Range"
