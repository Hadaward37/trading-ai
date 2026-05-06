"""Kill Switch dinâmico baseado em ATR — protege contra drawdown anormal.

Lógica:
  Trigger: drawdown diário > 2 × ATR médio dos últimos 20 dias
  Persistência: data/kill_switch_state.json
  Regra: não reativar no mesmo dia após trigger
  Reset: automático às 00:00 UTC

API pública:
  ks = KillSwitch()
  ks.is_active()                          → bool
  ks.check_and_trigger(dd, atr20)         → bool  (True = recém-ativado)
  ks.reset()                              → None  (reset manual)
  ks.get_status()                         → dict
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

_STATE_FILE = _ROOT / "data" / "kill_switch_state.json"

_DEFAULT_STATE: dict = {
    "active":        False,
    "trigger_date":  None,      # YYYY-MM-DD UTC
    "trigger_time":  None,      # ISO 8601 UTC
    "trigger_drawdown": None,
    "trigger_atr":   None,
    "trigger_threshold": None,
    "trigger_count_today": 0,
    "last_reset_time": None,
    "history":       [],        # list of past trigger events
}


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class KillSwitch:
    """Kill switch persistente baseado em drawdown diário × ATR(20)."""

    def __init__(self) -> None:
        self._state = self._load_state()

    # ── Public API ────────────────────────────────────────────────────────────

    def is_active(self) -> bool:
        """True se o kill switch está ativo para o dia atual."""
        self._auto_reset_if_new_day()
        return bool(self._state.get("active", False))

    def check_and_trigger(
        self,
        daily_drawdown: float,
        atr_20: float,
    ) -> bool:
        """Verifica se o kill switch deve ser ativado.

        Args:
            daily_drawdown: Drawdown diário atual (negativo, e.g. -0.0025).
            atr_20: ATR médio dos últimos 20 dias (em preço, e.g. 0.0012).

        Returns:
            True se foi recém-ativado nesta chamada, False caso contrário.
        """
        self._auto_reset_if_new_day()

        if self._state["active"]:
            logger.debug("Kill switch já ativo — ignorando check.")
            return False

        threshold = 2.0 * atr_20
        if daily_drawdown < -threshold:
            self._trigger(daily_drawdown, atr_20, threshold)
            return True

        return False

    def reset(self) -> None:
        """Reset manual do kill switch (uso administrativo)."""
        logger.warning("Kill switch resetado manualmente.")
        self._state["active"]          = False
        self._state["last_reset_time"] = _now_iso()
        self._save_state()

    def get_status(self) -> dict:
        """Retorna estado atual sem side effects."""
        self._auto_reset_if_new_day()
        return {
            "active":           self._state["active"],
            "trigger_date":     self._state["trigger_date"],
            "trigger_drawdown": self._state["trigger_drawdown"],
            "trigger_threshold":self._state["trigger_threshold"],
            "trigger_count_today": self._state["trigger_count_today"],
            "last_reset":       self._state["last_reset_time"],
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _trigger(self, drawdown: float, atr_20: float, threshold: float) -> None:
        today = _today_utc()
        now   = _now_iso()

        event = {
            "triggered_at":  now,
            "daily_drawdown": round(drawdown, 6),
            "atr_20":        round(atr_20, 6),
            "threshold":     round(threshold, 6),
            "date":          today,
        }

        self._state["active"]             = True
        self._state["trigger_date"]       = today
        self._state["trigger_time"]       = now
        self._state["trigger_drawdown"]   = round(drawdown, 6)
        self._state["trigger_atr"]        = round(atr_20, 6)
        self._state["trigger_threshold"]  = round(threshold, 6)
        self._state["trigger_count_today"]= self._state.get("trigger_count_today", 0) + 1

        history = self._state.setdefault("history", [])
        history.append(event)
        if len(history) > 500:
            self._state["history"] = history[-500:]

        self._save_state()

        logger.warning(
            "⛔ KILL SWITCH ATIVADO | drawdown=%.5f | threshold=%.5f (2×ATR=%.5f)",
            drawdown, threshold, atr_20,
        )

        self._send_telegram_alert(drawdown, threshold, atr_20)

    def _auto_reset_if_new_day(self) -> None:
        """Reset automático às 00:00 UTC quando o dia muda."""
        if not self._state.get("active"):
            return
        trigger_date = self._state.get("trigger_date", "")
        today        = _today_utc()
        if trigger_date and trigger_date != today:
            logger.info(
                "Kill switch: reset automático (dia anterior=%s, hoje=%s)",
                trigger_date, today,
            )
            self._state["active"]             = False
            self._state["trigger_count_today"] = 0
            self._state["last_reset_time"]    = _now_iso()
            self._save_state()

    def _load_state(self) -> dict:
        if _STATE_FILE.exists():
            try:
                with open(_STATE_FILE) as f:
                    return {**_DEFAULT_STATE, **json.load(f)}
            except Exception as exc:
                logger.warning("Erro ao ler kill switch state: %s — usando default.", exc)
        return dict(_DEFAULT_STATE)

    def _save_state(self) -> None:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(_STATE_FILE, "w") as f:
                json.dump(self._state, f, indent=2)
        except Exception as exc:
            logger.error("Erro ao salvar kill switch state: %s", exc)

    def _send_telegram_alert(
        self,
        drawdown: float,
        threshold: float,
        atr_20: float,
    ) -> None:
        try:
            import config
            if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
                return
            from core.notifier import TelegramNotifier
            msg = (
                f"⛔ *Kill Switch Ativado*\n\n"
                f"📉 Drawdown diário: `{drawdown*100:+.3f}%`\n"
                f"🔴 Threshold (2×ATR): `{threshold*100:.3f}%`\n"
                f"📊 ATR(20): `{atr_20:.5f}`\n\n"
                f"⚠️ Novos sinais bloqueados até 00:00 UTC"
            )
            TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)._send(msg)
        except Exception as exc:
            logger.warning("Telegram alert falhou: %s", exc)


# ── Module-level singleton ────────────────────────────────────────────────────

_instance: Optional[KillSwitch] = None


def get_kill_switch() -> KillSwitch:
    global _instance
    if _instance is None:
        _instance = KillSwitch()
    return _instance
