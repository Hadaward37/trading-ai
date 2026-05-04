"""Data collection module — fetches OHLCV data from yfinance."""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

SYMBOL = "EURUSD=X"

_TIMEFRAME_CONFIG: dict[str, dict[str, str]] = {
    "15m": {"interval": "15m", "period": "60d"},
    "1h":  {"interval": "1h",  "period": "2y"},
}


def fetch_ohlcv(
    timeframe: Literal["15m", "1h"] = "1h",
    period: str | None = None,
) -> pd.DataFrame:
    """Download EUR/USD OHLCV data from Yahoo Finance.

    Args:
        timeframe: Candle interval — ``"15m"`` or ``"1h"``.
        period: Optional override for the lookback window (e.g. ``"30d"``).
            yfinance caps 15 m data at 60 days regardless of this value.

    Returns:
        DataFrame indexed by timezone-naive UTC datetime with columns
        ``open``, ``high``, ``low``, ``close``, ``volume``.

    Raises:
        ValueError: For unsupported timeframe values.
        RuntimeError: When yfinance returns no data.
    """
    if timeframe not in _TIMEFRAME_CONFIG:
        raise ValueError(
            f"Unsupported timeframe {timeframe!r}. Choose from {list(_TIMEFRAME_CONFIG)}"
        )

    cfg = _TIMEFRAME_CONFIG[timeframe]
    effective_period = period or cfg["period"]

    logger.info("Fetching %s | interval=%s period=%s", SYMBOL, cfg["interval"], effective_period)

    raw = yf.Ticker(SYMBOL).history(
        period=effective_period, interval=cfg["interval"], auto_adjust=True
    )

    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {SYMBOL!r}")

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = pd.Index(["open", "high", "low", "close", "volume"])
    df.index.name = "datetime"

    # Strip timezone so SQLite / pandas comparisons stay consistent
    if df.index.tzinfo is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)

    logger.info("Received %d candles (%s to %s)", len(df), df.index[0], df.index[-1])
    return df
