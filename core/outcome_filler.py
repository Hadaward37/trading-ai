"""Outcome filler — reads signals.jsonl and fills price-return columns.

Designed to run as a periodic job (e.g., every 5 minutes via cron or APScheduler).
For each signal entry, computes % return against current price once the window
has elapsed. Idempotent: already-filled windows are skipped.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

SIGNALS_FILE = Path("logs/signals.jsonl")

WINDOWS: dict[str, timedelta] = {
    "outcome_5min":  timedelta(minutes=5),
    "outcome_30min": timedelta(minutes=30),
    "outcome_1h":    timedelta(hours=1),
    "outcome_4h":    timedelta(hours=4),
    "outcome_1d":    timedelta(days=1),
}


def fill_outcomes(get_price_fn: Callable[[str], float]) -> int:
    """Fill outcome windows for all signals whose time has come.

    Args:
        get_price_fn: callable(symbol) -> float returning current price.
                      Use the same data source as core/signals.py.

    Returns:
        Number of outcome cells filled in this run.
    """
    if not SIGNALS_FILE.exists():
        return 0

    lines = SIGNALS_FILE.read_text(encoding="utf-8").splitlines()
    now = datetime.now(timezone.utc)
    updated: list[str] = []
    filled_count = 0

    for line in lines:
        if not line.strip():
            continue
        entry = json.loads(line)
        signal_time = datetime.fromisoformat(entry["timestamp_utc"])
        symbol = entry["symbol"]
        price_initial = entry["price_at_signal"]

        for window_key, delta in WINDOWS.items():
            if entry.get(window_key) is not None:
                continue
            if now - signal_time >= delta:
                try:
                    current_price = get_price_fn(symbol)
                    ret_pct = (current_price - price_initial) / price_initial * 100
                    entry[window_key] = round(ret_pct, 4)
                    filled_count += 1
                except Exception as exc:
                    entry[window_key] = f"ERROR: {str(exc)[:50]}"
                    filled_count += 1

        updated.append(json.dumps(entry, ensure_ascii=False))

    SIGNALS_FILE.write_text("\n".join(updated) + "\n", encoding="utf-8")
    return filled_count
