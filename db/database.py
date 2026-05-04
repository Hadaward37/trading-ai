"""SQLite persistence layer — stores OHLCV and signal/indicator data."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, Engine

logger = logging.getLogger(__name__)

_DEFAULT_DB: Path = Path(__file__).parent.parent / "data" / "trading.db"


def _engine(db_path: Optional[Path] = None) -> Engine:
    path = db_path or _DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path}", echo=False)


# ── Write ─────────────────────────────────────────────────────────────────────

def save_ohlcv(
    df: pd.DataFrame, timeframe: str, db_path: Optional[Path] = None
) -> None:
    """Persist raw OHLCV data to the ``ohlcv_<timeframe>`` table.

    Replaces existing data for the same timeframe.
    """
    eng = _engine(db_path)
    table = f"ohlcv_{timeframe}"
    df.to_sql(table, con=eng, if_exists="replace", index=True, index_label="datetime")
    logger.info("Saved %d rows -> '%s'", len(df), table)


def save_signals(
    df: pd.DataFrame, timeframe: str, db_path: Optional[Path] = None
) -> None:
    """Persist full indicator + signal DataFrame to ``signals_<timeframe>``."""
    eng = _engine(db_path)
    table = f"signals_{timeframe}"
    df.to_sql(table, con=eng, if_exists="replace", index=True, index_label="datetime")
    logger.info("Saved %d rows -> '%s'", len(df), table)


# ── Read ──────────────────────────────────────────────────────────────────────

def load_ohlcv(
    timeframe: str, db_path: Optional[Path] = None
) -> Optional[pd.DataFrame]:
    """Load OHLCV data from SQLite; returns ``None`` when the table is absent."""
    eng = _engine(db_path)
    table = f"ohlcv_{timeframe}"
    try:
        df = pd.read_sql(
            f'SELECT * FROM "{table}"',
            con=eng,
            index_col="datetime",
            parse_dates=["datetime"],
        )
        logger.info("Loaded %d rows ← '%s'", len(df), table)
        return df
    except Exception as exc:
        logger.warning("Table '%s' not found: %s", table, exc)
        return None


def load_signals(
    timeframe: str, db_path: Optional[Path] = None
) -> Optional[pd.DataFrame]:
    """Load indicator + signal data from SQLite; returns ``None`` when absent."""
    eng = _engine(db_path)
    table = f"signals_{timeframe}"
    try:
        df = pd.read_sql(
            f'SELECT * FROM "{table}"',
            con=eng,
            index_col="datetime",
            parse_dates=["datetime"],
        )
        return df
    except Exception as exc:
        logger.warning("Table '%s' not found: %s", table, exc)
        return None
