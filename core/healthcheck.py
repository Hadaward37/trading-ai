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


_last_freshness_alert: Optional[datetime] = None

# 72h cobre fim de semana de forex (sexta→segunda ~65h) sem falso positivo,
# e ainda pega o modo de falha catastrófico: tabela congelada por dias/semanas.
SIGNAL_STALE_HOURS = 72


def check_signal_freshness(timeframe: str = "1h") -> None:
    """Alerta no Telegram se ``signals_<tf>`` parou de ser gravado.

    Pega a falha de 2026-05-21: scheduler vivo (heartbeat OK) mas ``save_signals``
    nunca executava, deixando a tabela congelada silenciosamente. Chamar uma vez
    por dia (bloco diário do scheduler). Cooldown de 24h entre alertas.
    """
    global _last_freshness_alert
    try:
        from db.database import last_signal_timestamp

        last = last_signal_timestamp(timeframe)
        now = datetime.now(timezone.utc)

        if last is None:
            msg = f"⚠️ ALERTA: tabela signals_{timeframe} ausente ou vazia — persistência não está gravando."
        else:
            last_utc = last.tz_localize("UTC") if last.tzinfo is None else last.tz_convert("UTC")
            age_h = (now - last_utc).total_seconds() / 3600
            if age_h <= SIGNAL_STALE_HOURS:
                return  # fresco — nada a fazer
            msg = (
                f"⚠️ ALERTA: signals_{timeframe} congelado há {age_h:.0f}h\n"
                f"Último registro: {last_utc.isoformat()}\n"
                f"Persistência pode ter parado (vivo mas sem gravar)."
            )

        cooldown_ok = (
            _last_freshness_alert is None
            or now - _last_freshness_alert > timedelta(hours=24)
        )
        if cooldown_ok:
            try:
                _get_notifier().send_text(msg)
            except Exception as exc:
                logger.error("[healthcheck] freshness alert failed: %s", exc)
            _last_freshness_alert = now
            logger.warning("[healthcheck] %s", msg.replace("\n", " | "))
    except Exception as exc:
        logger.warning("[healthcheck] freshness check error: %s", exc)


def start_healthcheck() -> Thread:
    """Start the background monitoring thread. Safe to call multiple times."""
    t = Thread(target=_check_loop, daemon=True, name="healthcheck")
    t.start()
    logger.info(
        "[healthcheck] started | threshold=%dmin | cooldown=%dmin",
        ALERT_THRESHOLD_MIN, _ALERT_COOLDOWN_MIN,
    )
    return t
