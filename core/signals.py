"""Signal generation — RSI + MACD + BB voting with ADX/Stoch confidence score.

Signal column      : 1=BUY | -1=SELL | 0=HOLD/NEWS_BLOCK
Score column       : 0-100 weighted confidence
                     RSI(25%) + MACD(25%) + BB(20%) + ADX(15%) + Stoch(15%)
Regime filter      : if regime=="Range" and REGIME_BLOCK_RANGE_SIGNALS, signal=0
news_blocked column: True when a high-impact economic event suppressed the signal
news_reason column : name of the blocking event (empty string when not blocked)
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

    # ── LSTM (SCORE_WEIGHT_LSTM pts) ──────────────────────────────────────────
    # lstm_prob is a scalar 0-100 for the LAST bar; broadcast to all rows.
    # Returns neutral (50) when model not available — contributes 0 net bias.
    lstm_prob = _get_lstm_prob(df)
    lstm_buy_pts  = pd.Series(
        max(lstm_prob - 50, 0) / 50 * config.SCORE_WEIGHT_LSTM, index=df.index
    )
    lstm_sell_pts = pd.Series(
        max(50 - lstm_prob, 0) / 50 * config.SCORE_WEIGHT_LSTM, index=df.index
    )

    buy_total  = (rsi_buy_pts  + macd_buy_pts  + bb_buy_pts  + adx_pts + stoch_buy_pts  + lstm_buy_pts ).clip(0, 100)
    sell_total = (rsi_sell_pts + macd_sell_pts + bb_sell_pts + adx_pts + stoch_sell_pts + lstm_sell_pts).clip(0, 100)

    return buy_total, sell_total


def _get_lstm_prob(df: pd.DataFrame) -> float:
    """Return LSTM directional probability (0-100). Returns 50 if disabled/unavailable."""
    if not getattr(config, "LSTM_ENABLED", False):
        return 50.0
    try:
        from core.lstm_model import predict as lstm_predict
        return lstm_predict(df)
    except Exception:
        return 50.0


# ── Public API ────────────────────────────────────────────────────────────────

def generate_signals(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """Generate signals using defaults from ``config``.

    Convenience wrapper around :func:`generate_signals_custom` with news
    filtering and sentiment analysis enabled (used by the live scheduler and
    dashboard).  Pass *ticker* (e.g. ``"EUR/USD"``) to enable sentiment.
    """
    return generate_signals_custom(
        df, config.RSI_BUY, config.RSI_SELL,
        check_news=True, check_sentiment=True, ticker=ticker,
    )


def generate_signals_custom(
    df: pd.DataFrame,
    rsi_buy: int,
    rsi_sell: int,
    regime: Optional[str] = None,
    check_news: bool = False,
    check_sentiment: bool = False,
    ticker: str = "",
) -> pd.DataFrame:
    """Add ``signal`` (1/-1/0) and ``score`` (0-100) columns to *df*.

    Signal logic — 2-of-3 vote (RSI + MACD crossover + BB touch):
    * ``1``  — BUY
    * ``-1`` — SELL
    * ``0``  — HOLD / no consensus

    Score weights:
    RSI {rsi_w}% + MACD {macd_w}% + BB {bb_w}% + ADX {adx_w}% + Stoch {stoch_w}%

    Args:
        df:               DataFrame with all indicator columns (from
                          :func:`core.indicators.add_all_indicators`).
        rsi_buy:          Buy threshold (RSI below this counts as oversold vote).
        rsi_sell:         Sell threshold (RSI above this counts as overbought vote).
        regime:           Current market regime string. When
                          :data:`config.REGIME_BLOCK_RANGE_SIGNALS` is True and
                          regime is ``"Range"``, all signals are suppressed.
        check_news:       When True, call :func:`core.news_filter.is_danger_zone`
                          and suppress signals near high-impact economic events.
                          Disabled by default so the optimizer/backtest are unaffected.
        check_sentiment:  When True, call
                          :func:`core.news_intelligence.get_market_sentiment` and
                          adjust ``final_score`` based on Gemini+GPT-4o analysis.
                          Disabled by default so the optimizer/backtest are unaffected.
        ticker:           Asset name passed to the sentiment pipeline (e.g. ``"EUR/USD"``).

    Returns:
        Copy of *df* with ``signal``, ``score``, ``news_blocked``, ``news_reason``,
        ``sentiment``, ``sentiment_confidence``, ``sentiment_impact``, and
        ``final_score`` columns appended.
    """.format(
        rsi_w=config.SCORE_WEIGHT_RSI, macd_w=config.SCORE_WEIGHT_MACD,
        bb_w=config.SCORE_WEIGHT_BB,   adx_w=config.SCORE_WEIGHT_ADX,
        stoch_w=config.SCORE_WEIGHT_STOCH,
    )
    df = df.copy()

    buy_votes, sell_votes = _vote_series(df, rsi_buy, rsi_sell)
    buy_score, sell_score = _score_series(df, rsi_buy, rsi_sell)

    # LSTM probability column (scalar broadcast to all rows)
    lstm_prob = _get_lstm_prob(df)
    df["lstm_prob"] = round(lstm_prob, 1)

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

    # News filter — always add columns; populate when check_news=True
    df["news_blocked"] = False
    df["news_reason"]  = ""

    if check_news and config.NEWS_FILTER_ENABLED:
        from core.news_filter import is_danger_zone  # lazy import avoids circular deps
        blocked, reason = is_danger_zone()
        if blocked:
            df["signal"]       = 0
            df["score"]        = 0.0
            df["news_blocked"] = True
            df["news_reason"]  = reason

    # ── Sentiment analysis (Phase 2) ──────────────────────────────────────────
    df["sentiment"]            = "NEUTRAL"
    df["sentiment_confidence"] = 0.0
    df["sentiment_impact"]     = "LOW"
    df["final_score"]          = df["score"]

    if check_sentiment and config.NEWS_INTELLIGENCE_ENABLED:
        from core.news_intelligence import get_market_sentiment  # lazy import
        analysis = get_market_sentiment(ticker)
        df["sentiment"]            = analysis.sentiment
        df["sentiment_confidence"] = analysis.confidence
        df["sentiment_impact"]     = analysis.impact

        # Adjust final_score for bars where a signal fired
        signal_mask = df["signal"] != 0
        if signal_mask.any():
            adjustments = df.loc[signal_mask, "signal"].apply(
                lambda s: analysis.score_adjustment(int(s))
            )
            df.loc[signal_mask, "final_score"] = (
                df.loc[signal_mask, "score"] + adjustments
            ).clip(0, 100).round(1)

    # ── Copom macro context (Brazilian assets only) ───────────────────────────
    df["copom_tone"]       = "neutral"
    df["copom_adjustment"] = 0.0

    if getattr(config, "COPOM_ENABLED", False) and ticker:
        _apply_copom_adjustment(df, ticker)

    return df


def _apply_copom_adjustment(df: "pd.DataFrame", ticker: str) -> None:
    """In-place: apply Copom score adjustment to final_score for BR assets."""
    try:
        from core.macro_context import copom_score_adjustment, get_copom_context, BR_SYMBOLS
        import config as _cfg

        # Map ticker name to yfinance symbol
        symbol = _cfg.ASSETS.get(ticker, ticker)
        if symbol not in BR_SYMBOLS:
            return   # EUR/USD and other non-BR assets unaffected

        ctx  = get_copom_context()
        tone = ctx.get("copom_tone", "neutral")
        df["copom_tone"] = tone

        signal_mask = df["signal"] != 0
        if signal_mask.any():
            adj = df.loc[signal_mask, "signal"].apply(
                lambda s: copom_score_adjustment(symbol, int(s))
            )
            df.loc[signal_mask, "copom_adjustment"] = adj
            df.loc[signal_mask, "final_score"] = (
                df.loc[signal_mask, "final_score"] + adj
            ).clip(0, 100).round(1)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Copom adjustment failed: %s", exc)
