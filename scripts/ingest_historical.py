"""Historical data ingestion — EUR/USD + macro indicators.

Sources:
  - MT5 terminal  : EUR/USD 1H (10+ years if available)
  - yfinance      : EUR/USD daily fallback + B3 equities
  - FRED API      : macro indicators (optional — requires FRED_API_KEY in .env)

Usage:
  python scripts/ingest_historical.py [--no-mt5] [--no-fred] [--retrain]

Flags:
  --no-mt5    skip MT5 download (use yfinance only)
  --no-fred   skip FRED macro download
  --retrain   retrain LSTM and XGBoost after ingestion

Output:
  - Populates ohlcv_1h and signals_1h tables in data/trading.db
  - Saves macro data to data/macro_fred.csv (if FRED enabled)
  - Prints a summary with row counts and date ranges
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ── MT5 historical download ───────────────────────────────────────────────────

def _fetch_from_mt5(years: int = 12) -> pd.DataFrame | None:
    """Download EUR/USD 1H from MT5 terminal going back `years` years."""
    try:
        import MetaTrader5 as mt5  # type: ignore[import]
    except ImportError:
        logger.warning("MetaTrader5 package not installed — skipping MT5 download.")
        return None

    if not mt5.initialize(timeout=config.MT5_TIMEOUT_MS):
        logger.warning("MT5 initialize failed: %s", mt5.last_error())
        return None

    from datetime import timedelta
    import pytz  # type: ignore[import]

    utc_tz    = pytz.utc
    date_to   = datetime.now(utc_tz)
    date_from = date_to - timedelta(days=years * 365)

    rates = mt5.copy_rates_range(
        config.MT5_SYMBOL, mt5.TIMEFRAME_H1,
        date_from, date_to,
    )
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        logger.warning("MT5 returned no rates for %s.", config.MT5_SYMBOL)
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={
        "time": "datetime", "open": "open", "high": "high",
        "low": "low", "close": "close", "tick_volume": "volume",
    })
    df = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df.set_index("datetime", inplace=True)
    df = df.sort_index()

    logger.info(
        "MT5: %d rows | %s → %s",
        len(df), df.index[0].date(), df.index[-1].date(),
    )
    return df


# ── yfinance fallback download ────────────────────────────────────────────────

def _fetch_from_yfinance_hourly(symbol: str = "EURUSD=X", days: int = 729) -> pd.DataFrame | None:
    """Download up to 729 days of 1H data from yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed.")
        return None

    ticker = yf.Ticker(symbol)
    df = ticker.history(period=f"{days}d", interval="1h", auto_adjust=True)

    if df.empty:
        logger.warning("yfinance returned empty for %s 1H.", symbol)
        return None

    df.index = pd.to_datetime(df.index, utc=True)
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index.name = "datetime"
    df = df.sort_index()

    logger.info(
        "yfinance 1H: %d rows | %s → %s",
        len(df), df.index[0].date(), df.index[-1].date(),
    )
    return df


def _fetch_from_yfinance_daily(symbol: str = "EURUSD=X", years: int = 15) -> pd.DataFrame | None:
    """Download daily OHLCV from yfinance (up to 15 years for USD pairs)."""
    try:
        import yfinance as yf
    except ImportError:
        return None

    ticker = yf.Ticker(symbol)
    df = ticker.history(period=f"{years}y", interval="1d", auto_adjust=True)

    if df.empty:
        return None

    df.index = pd.to_datetime(df.index, utc=True)
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index.name = "datetime"
    df = df.sort_index()

    logger.info(
        "yfinance Daily: %d rows | %s → %s",
        len(df), df.index[0].date(), df.index[-1].date(),
    )
    return df


# ── FRED macro data ───────────────────────────────────────────────────────────

_FRED_SERIES = {
    "fed_funds_rate":  "FEDFUNDS",    # US Federal Funds Rate (monthly)
    "us_cpi":          "CPIAUCSL",    # US CPI (monthly)
    "vix":             "VIXCLS",      # CBOE VIX (daily)
    "dxy":             "DTWEXBGS",    # USD Broad Index (daily)
    "eurusd_daily":    "DEXUSEU",     # EUR/USD spot rate (daily)
}


def _fetch_fred_data(start_year: int = 2010) -> pd.DataFrame | None:
    """Download macro series from FRED API. Requires FRED_API_KEY in .env."""
    api_key = os.getenv("FRED_API_KEY", "")
    if not api_key:
        logger.info(
            "FRED_API_KEY not set in .env — skipping macro download.\n"
            "  Free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
        )
        return None

    try:
        from fredapi import Fred  # type: ignore[import]
    except ImportError:
        logger.warning("fredapi not installed. Run: pip install fredapi")
        return None

    fred       = Fred(api_key=api_key)
    start_date = f"{start_year}-01-01"
    frames: list[pd.DataFrame] = []

    for name, series_id in _FRED_SERIES.items():
        try:
            s = fred.get_series(series_id, observation_start=start_date)
            s.name = name
            frames.append(s.to_frame())
            logger.info("FRED [%s]: %d observations", series_id, len(s))
        except Exception as exc:
            logger.warning("FRED [%s] failed: %s", series_id, exc)

    if not frames:
        return None

    macro = pd.concat(frames, axis=1)
    macro.index = pd.to_datetime(macro.index, utc=True)
    macro.index.name = "date"
    macro = macro.sort_index()

    out_path = _ROOT / "data" / "macro_fred.csv"
    macro.to_csv(out_path)
    logger.info("FRED macro saved to %s (%d rows, %d cols)", out_path, len(macro), len(macro.columns))
    return macro


# ── SQLite storage ────────────────────────────────────────────────────────────

def _save_to_sqlite(df: pd.DataFrame, table: str) -> int:
    """Upsert OHLCV rows into SQLite. Returns number of new rows inserted."""
    import sqlite3

    conn = sqlite3.connect(config.DB_PATH)
    cur  = conn.cursor()

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            datetime TEXT PRIMARY KEY,
            open     REAL,
            high     REAL,
            low      REAL,
            close    REAL,
            volume   REAL
        )
    """)
    conn.commit()

    # Convert index to string for SQLite
    df_out = df.copy()
    df_out.index = df_out.index.strftime("%Y-%m-%d %H:%M:%S+00:00")

    inserted = 0
    for dt, row in df_out.iterrows():
        cur.execute(
            f"INSERT OR IGNORE INTO {table} (datetime,open,high,low,close,volume) VALUES (?,?,?,?,?,?)",
            (dt, row["open"], row["high"], row["low"], row["close"], row["volume"]),
        )
        inserted += cur.rowcount

    conn.commit()
    conn.close()
    return inserted


def _get_existing_row_count(table: str) -> int:
    import sqlite3
    try:
        conn = sqlite3.connect(config.DB_PATH)
        cur  = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        n = cur.fetchone()[0]
        conn.close()
        return n
    except Exception:
        return 0


# ── Retraining ────────────────────────────────────────────────────────────────

def _retrain_models() -> None:
    """Force-retrain LSTM and XGBoost with all available data."""
    print("\n--- Retraining LSTM ---")
    try:
        from core.lstm_trainer import train as lstm_train
        m = lstm_train(force=True)
        print(f"  LSTM val_accuracy : {m.get('val_accuracy', '?'):.2%}")
        print(f"  LSTM best_epoch   : {m.get('best_epoch', '?')}")
    except Exception as exc:
        print(f"  LSTM training FAILED: {exc}")

    print("\n--- Retraining XGBoost ---")
    try:
        from core.xgboost_model import train as xgb_train
        m = xgb_train(force=True)
        print(f"  XGBoost val_accuracy  : {m.get('val_accuracy', '?'):.2%}")
        print(f"  XGBoost best_iteration: {m.get('best_iteration', '?')}")
    except Exception as exc:
        print(f"  XGBoost training FAILED: {exc}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    use_mt5   = "--no-mt5"   not in sys.argv
    use_fred  = "--no-fred"  not in sys.argv
    do_retrain= "--retrain"  in sys.argv

    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Trading AI — Historical Data Ingestion")
    print(f"  Target DB: {config.DB_PATH}")
    print("=" * 60)

    # ── EUR/USD 1H ────────────────────────────────────────────────────────────
    df_eurusd: pd.DataFrame | None = None

    if use_mt5:
        print("\n[1/4] Downloading EUR/USD 1H from MT5 (10 years)…")
        df_eurusd = _fetch_from_mt5(years=10)

    if df_eurusd is None:
        print("\n[1/4] Downloading EUR/USD 1H from yfinance (729 days)…")
        df_eurusd = _fetch_from_yfinance_hourly()

    if df_eurusd is None:
        print("  WARN: Could not fetch EUR/USD 1H data.")
    else:
        before = _get_existing_row_count("ohlcv_1h")
        new    = _save_to_sqlite(df_eurusd, "ohlcv_1h")
        after  = _get_existing_row_count("ohlcv_1h")
        print(f"  ohlcv_1h: {before} → {after} rows (+{new} new)")

    # ── EUR/USD Daily (for long-term context) ─────────────────────────────────
    print("\n[2/4] Downloading EUR/USD daily from yfinance (15 years)…")
    df_daily = _fetch_from_yfinance_daily("EURUSD=X", years=15)
    if df_daily is None:
        print("  WARN: Could not fetch EUR/USD daily data.")
    else:
        before = _get_existing_row_count("ohlcv_1d")
        new    = _save_to_sqlite(df_daily, "ohlcv_1d")
        after  = _get_existing_row_count("ohlcv_1d")
        print(f"  ohlcv_1d: {before} → {after} rows (+{new} new)")

    # ── B3 equities daily ─────────────────────────────────────────────────────
    print("\n[3/4] Downloading B3 equities from yfinance…")
    b3_symbols = {k: v for k, v in config.ASSETS.items() if v != "EURUSD=X"}
    for name, symbol in b3_symbols.items():
        df_b3 = _fetch_from_yfinance_daily(symbol, years=10)
        if df_b3 is None:
            print(f"  WARN: {name} ({symbol}) — no data.")
            continue
        tbl  = f"ohlcv_daily_{symbol.replace('.', '_').replace('^', '')}"
        new  = _save_to_sqlite(df_b3, tbl)
        print(f"  {name}: {len(df_b3)} rows (+{new} new) → {tbl}")

    # ── FRED macro ────────────────────────────────────────────────────────────
    if use_fred:
        print("\n[4/4] Downloading FRED macro indicators…")
        _fetch_fred_data(start_year=2010)
    else:
        print("\n[4/4] FRED skipped (--no-fred).")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Ingestion complete!")
    total_1h = _get_existing_row_count("ohlcv_1h")
    print(f"  ohlcv_1h total rows : {total_1h:,}")
    if total_1h < 500:
        print("  WARN: < 500 rows — models need more data before retraining.")
    print("=" * 60)

    if do_retrain:
        if total_1h < 500:
            print("Skipping retrain — insufficient data.")
        else:
            _retrain_models()
            print("\nModels retrained successfully.")


if __name__ == "__main__":
    main()
