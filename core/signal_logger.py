"""Structured signal logger — appends one JSON line per signal to logs/signals.jsonl.

Outcomes (price return after N minutes/hours) are filled asynchronously by
core/outcome_filler.py, keeping this module as fast as possible.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Optional

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
SIGNALS_FILE = LOG_DIR / "signals.jsonl"

_lock = Lock()


def log_signal(
    symbol: str,
    signal: str,
    price_at_signal: float,
    features: dict,
    regime: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> str:
    """Append a signal entry to logs/signals.jsonl.

    Returns the signal_id so callers can correlate with outcome updates later.
    """
    signal_id = f"{symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
    entry = {
        "signal_id": signal_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "signal": signal,
        "price_at_signal": price_at_signal,
        "features": features,
        "regime": regime,
        "metadata": metadata or {},
        "outcome_5min": None,
        "outcome_30min": None,
        "outcome_1h": None,
        "outcome_4h": None,
        "outcome_1d": None,
    }
    with _lock:
        with open(SIGNALS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return signal_id
