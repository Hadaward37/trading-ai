"""Data collection module — fetches OHLCV data from yfinance.

Supports any yfinance ticker (Forex, B3 stocks, indices).
Timeframes 15m and 1h are fetched directly; 4h is produced by resampling 1h.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Literal

import pandas as pd
import yfinance as yf

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config

logger = logging.getLogger(__name__)

# 4h is handled by resampling 1h — not fetched directly (yfinance limitation)
_TIMEFRAME_CONFIG: dict[str, dict[str, str]] = {
    "15m": {"interval": "15m", "period": "60d"},
    "1h":  {"interval": "1h",  "period": "2y"},
}


def _resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Resample a 1h OHLCV DataFrame to 4h bars."""
    resampled = df.resample("4h").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna()
    resampled.index.name = "datetime"
    return resampled


def fetch_ohlcv(
    timeframe: Literal["15m", "1h", "4h"] = "1h",
    symbol: str | None = None,
    period: str | None = None,
) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance for any supported ticker.

    ``"4h"`` is produced by fetching 1h data and resampling; all other
    timeframes are fetched directly.

    Args:
        timeframe: Candle interval — ``"15m"``, ``"1h"``, or ``"4h"``.
        symbol:    yfinance ticker (e.g. ``"EURUSD=X"``, ``"VALE3.SA"``,
                   ``"^BVSP"``). Defaults to ``config.SYMBOL``.
        period:    Optional override for the lookback window (e.g. ``"30d"``).

    Returns:
        DataFrame indexed by timezone-naive UTC datetime with columns
        ``open``, ``high``, ``low``, ``close``, ``volume``.

    Raises:
        ValueError:   For unsupported timeframe values.
        RuntimeError: When yfinance returns no data.
    """
    _symbol = symbol or config.SYMBOL

    if timeframe == "4h":
        df_1h = fetch_ohlcv("1h", symbol=_symbol, period=period)
        return _resample_4h(df_1h)

    if timeframe not in _TIMEFRAME_CONFIG:
        raise ValueError(
            f"Unsupported timeframe {timeframe!r}. "
            f"Choose from {list(_TIMEFRAME_CONFIG) + ['4h']}"
        )

    cfg = _TIMEFRAME_CONFIG[timeframe]
    effective_period = period or cfg["period"]

    logger.info("Fetching %s | interval=%s period=%s", _symbol, cfg["interval"], effective_period)

    raw = yf.Ticker(_symbol).history(
        period=effective_period, interval=cfg["interval"], auto_adjust=True
    )

    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {_symbol!r}")

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = pd.Index(["open", "high", "low", "close", "volume"])
    df.index.name = "datetime"

    if df.index.tzinfo is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)

    logger.info("Received %d candles (%s to %s)", len(df), df.index[0], df.index[-1])
    return df


def get_mtf_confluence(symbol: str | None = None) -> dict:
    """Compute signal confluence across 15m, 1h, and 4h timeframes.

    Fetches live data for each timeframe, computes indicators and signals,
    and returns whether a majority (>= :data:`config.MTF_MIN_AGREEMENTS`) agree.

    Args:
        symbol: yfinance ticker. Defaults to ``config.SYMBOL``.

    Returns:
        dict with keys ``"15m"``, ``"1h"``, ``"4h"`` (signal int 1/-1/0),
        ``"confluence"`` (consolidated signal), ``"buy_count"``, ``"sell_count"``.
    """
    from core.indicators import add_all_indicators
    from core.signals import generate_signals_custom

    _symbol = symbol or config.SYMBOL
    results: dict[str, int] = {}

    df_1h_raw  = fetch_ohlcv("1h",  symbol=_symbol)
    df_15m_raw = fetch_ohlcv("15m", symbol=_symbol)
    df_4h_raw  = fetch_ohlcv("4h",  symbol=_symbol)  # resampled from 1h internally

    for tf, df_raw in [("15m", df_15m_raw), ("1h", df_1h_raw), ("4h", df_4h_raw)]:
        try:
            df_ind = add_all_indicators(df_raw)
            # check_news=False: MTF confluence is used for display only, no blocking needed
            df_sig = generate_signals_custom(
                df_ind, config.RSI_BUY, config.RSI_SELL, check_news=False
            )
            results[tf] = int(df_sig.iloc[-2]["signal"]) if len(df_sig) >= 2 else 0
        except Exception as exc:
            logger.warning("MTF %s %s failed: %s", tf, _symbol, exc)
            results[tf] = 0

    signals    = list(results.values())
    buy_count  = sum(1 for s in signals if s ==  1)
    sell_count = sum(1 for s in signals if s == -1)
    confluence = (
         1 if buy_count  >= config.MTF_MIN_AGREEMENTS else
        -1 if sell_count >= config.MTF_MIN_AGREEMENTS else 0
    )

    return {
        **results,
        "confluence":  confluence,
        "buy_count":   buy_count,
        "sell_count":  sell_count,
    }
