import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

SYMBOL = "EURUSD=X"
CACHE_DIR = Path("data/fractal_cache")

FOLD1_START = "2024-09-01"
FOLD1_END = "2025-02-28"
HOLDOUT_START = "2025-12-01"
HOLDOUT_END = "2026-05-07"

# yfinance hard limits (days back from today)
_LOOKBACK_LIMIT = {"1m": 7, "5m": 60}


def _cache_path(symbol: str, interval: str, start: str, end: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = f"{symbol}_{interval}_{start}_{end}".replace("=", "").replace("/", "-")
    return CACHE_DIR / f"{key}.parquet"


def download_data(symbol: str, interval: str, start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Download OHLCV data via yfinance.
    Clips requested range to yfinance availability; returns empty DataFrame if out of range.
    """
    today = datetime.utcnow()
    start_dt = pd.to_datetime(start).tz_localize(None)
    end_dt = min(pd.to_datetime(end).tz_localize(None), today)

    limit_days = _LOOKBACK_LIMIT.get(interval)
    if limit_days:
        earliest_available = today - timedelta(days=limit_days - 1)
        if start_dt < earliest_available:
            start_dt = earliest_available
        if start_dt >= end_dt:
            return pd.DataFrame()

    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")
    cache_file = _cache_path(symbol, interval, start_str, end_str)

    if use_cache and cache_file.exists():
        df = pd.read_parquet(cache_file)
        return df

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start_str,
            end=end_str,
            interval=interval,
            auto_adjust=True,
        )
    except Exception as e:
        print(f"[data_loader] Download failed ({symbol} {interval}): {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    if use_cache:
        df.to_parquet(cache_file)

    return df


def get_period_data(period: str, use_cache: bool = True):
    """
    Returns (df_1m, df_5m) for a named period.
    Periods: 'fold1' | 'holdout'
    """
    periods = {
        "fold1": (FOLD1_START, FOLD1_END),
        "holdout": (HOLDOUT_START, HOLDOUT_END),
    }
    if period not in periods:
        raise ValueError(f"Unknown period '{period}'. Use: {list(periods)}")

    start, end = periods[period]
    df_1m = download_data(SYMBOL, "1m", start, end, use_cache)
    df_5m = download_data(SYMBOL, "5m", start, end, use_cache)
    return df_1m, df_5m
