"""Economic news filter — blocks signals during high-impact event windows.

Primary source : ForexFactory CDN JSON feed (thisweek + nextweek).
Fallback        : BeautifulSoup HTML scraper (may fail if FF uses JS rendering).
Cache           : 1-hour TTL so the server is not hammered on every cycle.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import NamedTuple, Optional

import requests
from bs4 import BeautifulSoup

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config

logger = logging.getLogger(__name__)


# ── Data model ────────────────────────────────────────────────────────────────

class NewsEvent(NamedTuple):
    title: str
    currency: str
    event_time: datetime  # always UTC-aware
    impact: str           # "High" | "Medium" | "Low"


# ── In-process cache ─────────────────────────────────────────────────────────

class _Cache:
    def __init__(self) -> None:
        self._events: list[NewsEvent] = []
        self._fetched_at: Optional[float] = None

    def stale(self) -> bool:
        if self._fetched_at is None:
            return True
        return (time.monotonic() - self._fetched_at) > config.NEWS_CACHE_TTL_SECONDS

    def set(self, events: list[NewsEvent]) -> None:
        self._events = events
        self._fetched_at = time.monotonic()

    def get(self) -> list[NewsEvent]:
        return list(self._events)


_cache = _Cache()

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# ── Fetchers ──────────────────────────────────────────────────────────────────

def _fetch_json_feed() -> list[NewsEvent]:
    """ForexFactory CDN JSON — returns this-week and next-week calendars."""
    events: list[NewsEvent] = []
    for week in ("thisweek", "nextweek"):
        url = f"https://nfs.faireconomy.media/ff_calendar_{week}.json"
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=10)
            resp.raise_for_status()
            for item in resp.json():
                if item.get("impact", "").lower() != "high":
                    continue
                try:
                    dt = datetime.fromisoformat(item["date"])
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                except (KeyError, ValueError):
                    continue
                events.append(NewsEvent(
                    title=item.get("title", "Unknown"),
                    currency=item.get("country", ""),
                    event_time=dt,
                    impact="High",
                ))
        except Exception as exc:
            logger.warning("FF JSON feed (%s) failed: %s", week, exc)
    return events


def _fetch_html_scrape() -> list[NewsEvent]:
    """BeautifulSoup fallback — may fail when ForexFactory returns a JS shell."""
    events: list[NewsEvent] = []
    try:
        resp = requests.get(
            "https://www.forexfactory.com/calendar",
            headers=_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        table = soup.find("table", class_="calendar__table")
        if not table:
            logger.warning("FF HTML: calendar__table not found (likely JS-rendered page)")
            return events

        now_utc = datetime.now(tz=timezone.utc)

        for row in table.find_all("tr", class_=lambda c: c and "calendar__row" in c):
            impact_td = row.find("td", class_="calendar__impact")
            if not impact_td:
                continue
            # High-impact events have a red icon
            red_span = impact_td.find(
                "span", class_=lambda c: c and "icon--red" in c
            )
            if not red_span:
                continue

            time_td  = row.find("td", class_="calendar__time")
            event_td = row.find("td", class_="calendar__event")
            curr_td  = row.find("td", class_="calendar__currency")
            if not (time_td and event_td):
                continue

            title    = event_td.get_text(strip=True)
            currency = curr_td.get_text(strip=True) if curr_td else ""
            time_str = time_td.get_text(strip=True).upper()

            try:
                t  = datetime.strptime(time_str, "%I:%M%p")
                dt = now_utc.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
            except ValueError:
                continue

            events.append(NewsEvent(
                title=title, currency=currency,
                event_time=dt, impact="High",
            ))

    except Exception as exc:
        logger.warning("FF HTML scrape failed: %s", exc)

    return events


# ── Cache refresh ─────────────────────────────────────────────────────────────

def _refresh() -> None:
    events = _fetch_json_feed()
    if not events:
        logger.info("JSON feed returned nothing — falling back to HTML scrape")
        events = _fetch_html_scrape()
    _cache.set(events)
    logger.info("News cache refreshed: %d high-impact events loaded", len(events))


# ── Public API ────────────────────────────────────────────────────────────────

def get_upcoming_events(n: int = 3) -> list[NewsEvent]:
    """Return the next *n* upcoming high-impact USD/EUR events, sorted by time."""
    if _cache.stale():
        _refresh()
    now = datetime.now(tz=timezone.utc)
    relevant = [
        e for e in _cache.get()
        if e.event_time >= now and e.currency in config.NEWS_FILTER_CURRENCIES
    ]
    relevant.sort(key=lambda e: e.event_time)
    return relevant[:n]


def is_danger_zone() -> tuple[bool, str]:
    """Return ``(True, event_title)`` when *now* is within the configured danger window.

    Returns ``(False, "")`` when it is safe to trade.
    Always returns ``(False, "")`` when ``config.NEWS_FILTER_ENABLED`` is False.
    """
    if not config.NEWS_FILTER_ENABLED:
        return False, ""

    if _cache.stale():
        _refresh()

    now    = datetime.now(tz=timezone.utc)
    window = timedelta(minutes=config.NEWS_FILTER_WINDOW_MINUTES)

    for event in _cache.get():
        if event.currency not in config.NEWS_FILTER_CURRENCIES:
            continue
        if abs(now - event.event_time) <= window:
            return True, event.title

    return False, ""
