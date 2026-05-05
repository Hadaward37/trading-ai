"""MetaTrader 5 data connector — fetches live OHLCV in the same format as collector.py.

Supports M15, H1, H4 timeframes for EUR/USD (and any MT5 symbol).
Falls back gracefully when MT5 is unavailable (USE_MT5=False or terminal not running).
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config

logger = logging.getLogger(__name__)

# MT5 timeframe constants (populated on first successful import)
_MT5_TF: dict[str, int] = {}

_TF_MAP: dict[str, str] = {
    "15m": "TIMEFRAME_M15",
    "1h":  "TIMEFRAME_H1",
    "4h":  "TIMEFRAME_H4",
}

# How many bars to fetch per timeframe
_BAR_COUNT: dict[str, int] = {
    "15m": 5000,
    "1h":  2000,
    "4h":  1000,
}


def _get_mt5():
    """Import MetaTrader5 module, raising ImportError when not installed."""
    try:
        import MetaTrader5 as mt5
        return mt5
    except ImportError as exc:
        raise ImportError("MetaTrader5 not installed — run: pip install MetaTrader5") from exc


def initialize() -> bool:
    """Open connection to the running MT5 terminal.

    Returns True on success, False otherwise.
    MT5 terminal must already be open and logged in.
    """
    mt5 = _get_mt5()
    if not mt5.initialize():
        err = mt5.last_error()
        logger.error("MT5 initialize() failed: %s", err)
        return False
    info = mt5.terminal_info()
    logger.info("MT5 connected — build=%s community=%s", info.build, info.community_account)
    return True


def shutdown() -> None:
    """Release MT5 connection."""
    mt5 = _get_mt5()
    mt5.shutdown()
    logger.info("MT5 connection closed")


def fetch_ohlcv(
    timeframe: Literal["15m", "1h", "4h"] = "1h",
    symbol: str = "EURUSD",
    count: int | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV bars from MT5 and return a DataFrame identical to collector.py output.

    Args:
        timeframe: ``"15m"``, ``"1h"``, or ``"4h"``.
        symbol:    MT5 symbol name (e.g. ``"EURUSD"``).
        count:     Number of bars to fetch. Defaults to per-TF presets in _BAR_COUNT.

    Returns:
        DataFrame indexed by timezone-naive UTC datetime with columns
        ``open``, ``high``, ``low``, ``close``, ``volume``.

    Raises:
        ValueError:   Unsupported timeframe.
        RuntimeError: MT5 not connected or no data returned.
    """
    if timeframe not in _TF_MAP:
        raise ValueError(
            f"Unsupported timeframe {timeframe!r}. Choose from {list(_TF_MAP)}"
        )

    mt5 = _get_mt5()

    tf_attr = _TF_MAP[timeframe]
    tf_const = getattr(mt5, tf_attr, None)
    if tf_const is None:
        raise RuntimeError(f"MT5 has no attribute {tf_attr!r} — check your MT5 version")

    n_bars = count or _BAR_COUNT[timeframe]
    logger.info("Fetching %s bars from MT5: symbol=%s tf=%s", n_bars, symbol, timeframe)

    rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, n_bars)
    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        raise RuntimeError(f"MT5 returned no data for {symbol!r} {timeframe} — error: {err}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None)
    df = df.set_index("time")
    df.index.name = "datetime"

    df = df.rename(columns={
        "open":      "open",
        "high":      "high",
        "low":       "low",
        "close":     "close",
        "tick_volume": "volume",
    })[["open", "high", "low", "close", "volume"]]

    logger.info(
        "MT5 received %d candles (%s → %s)",
        len(df), df.index[0], df.index[-1],
    )
    return df


def get_mtf_confluence(symbol: str = "EURUSD") -> dict:
    """Compute signal confluence across M15, H1, H4 using MT5 data.

    Drop-in replacement for collector.get_mtf_confluence() when USE_MT5=True.

    Returns:
        dict with keys ``"15m"``, ``"1h"``, ``"4h"``,
        ``"confluence"``, ``"buy_count"``, ``"sell_count"``.
    """
    from core.indicators import add_all_indicators
    from core.signals import generate_signals_custom

    results: dict[str, int] = {}

    for tf in ("15m", "1h", "4h"):
        try:
            df_raw = fetch_ohlcv(tf, symbol=symbol)
            df_ind = add_all_indicators(df_raw)
            df_sig = generate_signals_custom(
                df_ind, config.RSI_BUY, config.RSI_SELL, check_news=False
            )
            results[tf] = int(df_sig.iloc[-2]["signal"]) if len(df_sig) >= 2 else 0
        except Exception as exc:
            logger.warning("MT5 MTF %s %s failed: %s", tf, symbol, exc)
            results[tf] = 0

    signals    = list(results.values())
    buy_count  = sum(1 for s in signals if s ==  1)
    sell_count = sum(1 for s in signals if s == -1)
    confluence = (
         1 if buy_count  >= config.MTF_MIN_AGREEMENTS else
        -1 if sell_count >= config.MTF_MIN_AGREEMENTS else 0
    )

    return {**results, "confluence": confluence, "buy_count": buy_count, "sell_count": sell_count}


def print_last_candles(symbol: str = "EURUSD", timeframe: str = "1h", n: int = 10) -> None:
    """Pretty-print the last N candles from MT5 to stdout."""
    df = fetch_ohlcv(timeframe, symbol=symbol)
    last = df.tail(n)
    print(f"\n{'='*60}")
    print(f"  MT5 | {symbol} | {timeframe.upper()} — last {n} candles")
    print(f"{'='*60}")
    print(last.to_string(float_format="{:.5f}".format))
    print(f"{'='*60}\n")
