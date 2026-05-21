"""Cron job — fill outcome windows for signals that have aged past each threshold.

Run this every 5 minutes via Windows Task Scheduler or APScheduler:
    python scripts/fill_outcomes_job.py

Uses yfinance (same data source as core/signals.py) to fetch current prices.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yfinance as yf

from core.outcome_filler import fill_outcomes


def get_price(symbol: str) -> float:
    """Fetch current price via yfinance. symbol must be a valid yfinance ticker
    (e.g. 'EURUSD=X', 'VALE3.SA', '^BVSP').
    """
    ticker = yf.Ticker(symbol)
    price = ticker.fast_info.get("last_price")
    if price is None:
        raise ValueError(f"No price returned for {symbol!r}")
    return float(price)


if __name__ == "__main__":
    count = fill_outcomes(get_price)
    print(f"Outcomes filled: {count} cells updated.")
