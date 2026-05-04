"""Signal generation — RSI + MACD + BB voting with ADX/Stoch confidence score.

Signal column  : 1=BUY | -1=SELL | 0=HOLD
Score column   : 0-100 weighted confidence
                 RSI(25%) + MACD(25%) + BB(20%) + ADX(15%) + Stoch(15%)
Regime filter  : if regime=="Range" and REGIME_BLOCK_RANGE_SIGNALS, signal=0
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config

# Re-exported so optimizer.py can still do `from core.signals import BB_TOLERANCE`
BB_TOLERANCE: float = config.BB_TOLERANCE


# ── Internal helpers ──────────────────────────────────────────────────────────

def _vote_series(
    df: pd.DataFrame, rsi_buy: int, rsi_sell: int
) -> tuple[pd.Series, pd.Series]:
    """Return (buy_votes, sell_votes) per bar — RSI + MACD crossover + BB."""
    v_rsi_buy  = (df["rsi"] < rsi_buy).astype(int)
    v_rsi_sell = (df["rsi"] > rsi_sell).astype(int)

    prev_macd = df["macd"].shift(1)
    prev_sig  = df["macd_signal"].shift(1)
    v_macd_up = ((df["macd"] > df["macd_signal"]) & (prev_macd <= prev_sig)).astype(int)
    v_macd_dn = ((df["macd"] < df["macd_signal"]) & (prev_macd >= prev_sig)).astype(int)

    v_bb_buy  = (df["close"] <= df["bb_lower"] * (1 + BB_TOLERANCE)).astype(int)
    v_bb_sell = (df["close"] >= df["bb_upper"] * (1 - BB_TOLERANCE)).astype(int)

    return (
        v_rsi_buy  + v_macd_up + v_bb_buy,
        v_rsi_sell + v_macd_dn + v_bb_sell,
    )


def _score_series(
    df: pd.DataFrame, rsi_buy: int, rsi_sell: int
) -> tuple[pd.Series, pd.Series]:
    """Return (buy_score, sell_score) 0-100 per bar using all 5 indicators."""

    # ── RSI (SCORE_WEIGHT_RSI pts) ────────────────────────────────────────────
    rsi_buy_pts  = (rsi_buy  - df["rsi"]).clip(lower=0) / max(rsi_buy,          1) * config.SCORE_WEIGHT_RSI
    rsi_sell_pts = (df["rsi"] - rsi_sell).clip(lower=0) / max(100 - rsi_sell,   1) * config.SCORE_WEIGHT_RSI

    # ── MACD (SCORE_WEIGHT_MACD pts) ──────────────────────────────────────────
    # Crossover = full weight, aligned (no cross) = half weight
    prev_macd   = df["macd"].shift(1)
    prev_sig_   = df["macd_signal"].shift(1)
    cross_up    = (df["macd"] > df["macd_signal"]) & (prev_macd <= prev_sig_)
    cross_dn    = (df["macd"] < df["macd_signal"]) & (prev_macd >= prev_sig_)
    aligned_up  = df["macd"] > df["macd_signal"]
    aligned_dn  = df["macd"] < df["macd_signal"]
    half_w      = config.SCORE_WEIGHT_MACD / 2

    macd_buy_pts  = cross_up * config.SCORE_WEIGHT_MACD + (~cross_up & aligned_up) * half_w
    macd_sell_pts = cross_dn * config.SCORE_WEIGHT_MACD + (~cross_dn & aligned_dn) * half_w

    # ── BB (SCORE_WEIGHT_BB pts) ──────────────────────────────────────────────
    # bb_pct: 0=lower band, 1=upper band
    bb_pct        = df["bb_pct"].fillna(0.5)
    bb_buy_pts    = (1 - bb_pct.clip(0, 1)) * config.SCORE_WEIGHT_BB
    bb_sell_pts   = bb_pct.clip(0, 1) * config.SCORE_WEIGHT_BB

    # ── ADX (SCORE_WEIGHT_ADX pts — same for both directions) ─────────────────
    adx_pts = df["adx"].fillna(0).clip(0, 50) / 50 * config.SCORE_WEIGHT_ADX

    # ── Stochastic (SCORE_WEIGHT_STOCH pts) ───────────────────────────────────
    stoch_k        = df["stoch_k"].fillna(50)
    stoch_buy_pts  = (config.STOCH_OVERSOLD  - stoch_k).clip(lower=0) / max(config.STOCH_OVERSOLD,       1) * config.SCORE_WEIGHT_STOCH
    stoch_sell_pts = (stoch_k - config.STOCH_OVERBOUGHT).clip(lower=0) / max(100 - config.STOCH_OVERBOUGHT, 1) * config.SCORE_WEIGHT_STOCH

    buy_total  = (rsi_buy_pts  + macd_buy_pts  + bb_buy_pts  + adx_pts + stoch_buy_pts ).clip(0, 100)
    sell_total = (rsi_sell_pts + macd_sell_pts + bb_sell_pts + adx_pts + stoch_sell_pts).clip(0, 100)

    return buy_total, sell_total


# ── Public API ────────────────────────────────────────────────────────────────

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate signals using defaults from ``config``.

    Convenience wrapper around :func:`generate_signals_custom`.
    """
    return generate_signals_custom(df, config.RSI_BUY, config.RSI_SELL)


def generate_signals_custom(
    df: pd.DataFrame,
    rsi_buy: int,
    rsi_sell: int,
    regime: Optional[str] = None,
) -> pd.DataFrame:
    """Add ``signal`` (1/-1/0) and ``score`` (0-100) columns to *df*.

    Signal logic — 2-of-3 vote (RSI + MACD crossover + BB touch):
    * ``1``  — BUY
    * ``-1`` — SELL
    * ``0``  — HOLD / no consensus

    Score weights:
    RSI {rsi_w}% + MACD {macd_w}% + BB {bb_w}% + ADX {adx_w}% + Stoch {stoch_w}%

    Args:
        df: DataFrame with all indicator columns (from
            :func:`core.indicators.add_all_indicators`).
        rsi_buy:  Buy threshold (RSI below this counts as oversold vote).
        rsi_sell: Sell threshold (RSI above this counts as overbought vote).
        regime:   Current market regime string. When
            :data:`config.REGIME_BLOCK_RANGE_SIGNALS` is True and regime is
            ``"Range"``, all signals are suppressed.

    Returns:
        Copy of *df* with ``signal`` and ``score`` columns appended.
    """.format(
        rsi_w=config.SCORE_WEIGHT_RSI, macd_w=config.SCORE_WEIGHT_MACD,
        bb_w=config.SCORE_WEIGHT_BB,   adx_w=config.SCORE_WEIGHT_ADX,
        stoch_w=config.SCORE_WEIGHT_STOCH,
    )
    df = df.copy()

    buy_votes, sell_votes = _vote_series(df, rsi_buy, rsi_sell)
    buy_score, sell_score = _score_series(df, rsi_buy, rsi_sell)

    df["signal"] = 0
    df.loc[buy_votes  >= 2, "signal"] =  1
    df.loc[sell_votes >= 2, "signal"] = -1

    # Regime filter: suppress signals during ranging market
    if regime == "Range" and config.REGIME_BLOCK_RANGE_SIGNALS:
        df["signal"] = 0

    # Score column — non-zero only where signal fires
    score = pd.Series(0.0, index=df.index)
    score[df["signal"] ==  1] = buy_score[ df["signal"] ==  1]
    score[df["signal"] == -1] = sell_score[df["signal"] == -1]
    df["score"] = score.round(1)

    return df
