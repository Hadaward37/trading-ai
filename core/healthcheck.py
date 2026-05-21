"""Healthcheck — background thread that alerts via Telegram if the system goes silent.

Usage in scheduler:
    from core.healthcheck import start_healthcheck, heartbeat
    start_healthcheck()   # once at startup
    heartbeat()           # inside every scheduler cycle
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Thread
from typing import Optional

logger = logging.getLogger(__name__)

HEARTBEAT_FILE = Path("logs/heartbeat.txt")
ALERT_THRESHOLD_MIN = 10
CHECK_INTERVAL_SEC = 60
_ALERT_COOLDOWN_MIN = 30

_last_alert_sent: Optional[datetime] = None


def heartbeat() -> None:
    """Record that the system is alive. Call once per scheduler cycle."""
    Path("logs").mkdir(exist_ok=True)
    HEARTBEAT_FILE.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")


def _get_notifier():
    import config
    from core.notifier import TelegramNotifier
    return TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)


def _check_loop() -> None:
    global _last_alert_sent
    while True:
        try:
            if HEARTBEAT_FILE.exists():
                raw = HEARTBEAT_FILE.read_text(encoding="utf-8").strip()
                last = datetime.fromisoformat(raw)
                elapsed = datetime.now(timezone.utc) - last
                if elapsed > timedelta(minutes=ALERT_THRESHOLD_MIN):
                    cooldown_ok = (
                        _last_alert_sent is None
                        or datetime.now(timezone.utc) - _last_alert_sent
                        > timedelta(minutes=_ALERT_COOLDOWN_MIN)
                    )
                    if cooldown_ok:
                        elapsed_min = int(elapsed.total_seconds() / 60)
                        try:
                            _get_notifier().send_text(
                                f"ALERTA: Sistema 1 sem heartbeat há {elapsed_min}min\n"
                                f"Último sinal de vida: {last.isoformat()}"
                            )
                        except Exception as exc:
                            logger.error("[healthcheck] failed to send alert: %s", exc)
                        _last_alert_sent = datetime.now(timezone.utc)
        except Exception as exc:
            logger.warning("[healthcheck] check error: %s", exc)
        time.sleep(CHECK_INTERVAL_SEC)


def start_healthcheck() -> Thread:
    """Start the background monitoring thread. Safe to call multiple times."""
    t = Thread(target=_check_loop, daemon=True, name="healthcheck")
    t.start()
    logger.info(
        "[healthcheck] started | threshold=%dmin | cooldown=%dmin",
        ALERT_THRESHOLD_MIN, _ALERT_COOLDOWN_MIN,
    )
    return t
