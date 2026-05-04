"""Data collection module — fetches OHLCV data from yfinance.

Supports 15m, 1h natively, and 4h via resampling of 1h data.
"""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

SYMBOL = "EURUSD=X"

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
    period: str | None = None,
) -> pd.DataFrame:
    """Download EUR/USD OHLCV data from Yahoo Finance.

    ``"4h"`` is produced by fetching 1h data and resampling; all other
    timeframes are fetched directly.

    Args:
        timeframe: Candle interval — ``"15m"``, ``"1h"``, or ``"4h"``.
        period: Optional override for the lookback window (e.g. ``"30d"``).

    Returns:
        DataFrame indexed by timezone-naive UTC datetime with columns
        ``open``, ``high``, ``low``, ``close``, ``volume``.

    Raises:
        ValueError: For unsupported timeframe values.
        RuntimeError: When yfinance returns no data.
    """
    if timeframe == "4h":
        df_1h = fetch_ohlcv("1h", period=period)
        return _resample_4h(df_1h)

    if timeframe not in _TIMEFRAME_CONFIG:
        raise ValueError(
            f"Unsupported timeframe {timeframe!r}. Choose from {list(_TIMEFRAME_CONFIG) + ['4h']}"
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

    if df.index.tzinfo is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)

    logger.info("Received %d candles (%s to %s)", len(df), df.index[0], df.index[-1])
    return df


def get_mtf_confluence(symbol: str = SYMBOL) -> dict:
    """Compute signal confluence across 15m, 1h, and 4h timeframes.

    Fetches live data for each timeframe, computes indicators and signals,
    and returns whether a majority (>= :data:`config.MTF_MIN_AGREEMENTS`) agree.

    Returns:
        dict with keys ``"15m"``, ``"1h"``, ``"4h"`` (signal int 1/-1/0),
        ``"confluence"`` (consolidated signal), ``"buy_count"``, ``"sell_count"``.
    """
    import sys
    from pathlib import Path
    _root = Path(__file__).parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    import config
    from core.indicators import add_all_indicators
    from core.signals import generate_signals

    results: dict[str, int] = {}

    # Fetch each timeframe
    df_1h_raw  = fetch_ohlcv("1h")
    df_15m_raw = fetch_ohlcv("15m")
    df_4h_raw  = fetch_ohlcv("4h")   # resampled from 1h internally

    for tf, df_raw in [("15m", df_15m_raw), ("1h", df_1h_raw), ("4h", df_4h_raw)]:
        try:
            df_ind = add_all_indicators(df_raw)
            df_sig = generate_signals(df_ind)
            # Use last COMPLETED bar (index -2); -1 is the forming candle
            results[tf] = int(df_sig.iloc[-2]["signal"]) if len(df_sig) >= 2 else 0
        except Exception as exc:
            logger.warning("MTF %s failed: %s", tf, exc)
            results[tf] = 0

    signals     = list(results.values())
    buy_count   = sum(1 for s in signals if s ==  1)
    sell_count  = sum(1 for s in signals if s == -1)
    confluence  = (
         1 if buy_count  >= config.MTF_MIN_AGREEMENTS else
        -1 if sell_count >= config.MTF_MIN_AGREEMENTS else 0
    )

    return {
        **results,
        "confluence":  confluence,
        "buy_count":   buy_count,
        "sell_count":  sell_count,
    }
