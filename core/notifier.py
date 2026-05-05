"""Telegram notification module — sends formatted trading signal alerts."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"

_SIGNAL_EMOJI = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}


@dataclass
class SignalAlert:
    """All data needed to render one trading alert message."""

    signal:      str    # "BUY" | "SELL" | "HOLD"
    asset:       str    # e.g. "EUR/USD" or "VALE3"
    timeframe:   str    # e.g. "1h"
    price:       float
    stop_loss:   float
    take_profit: float
    win_rate:    float  # historical win rate in percent
    confidence:  int    # 1..3 — how many indicators agreed
    rsi:         float
    atr:         float
    flag:        str = "📊"   # flag emoji from ASSET_META
    decimals:    int = 5      # price decimal places (5=forex, 2=stocks, 0=index)


class TelegramNotifier:
    """Sends Markdown-formatted signal alerts to a Telegram chat."""

    def __init__(self, token: str, chat_id: str) -> None:
        self._token   = token
        self._chat_id = chat_id
        self._url     = _TELEGRAM_API.format(token=token)

    @property
    def is_configured(self) -> bool:
        """True when both token and chat_id are non-empty."""
        return bool(self._token and self._chat_id)

    # ── Public ────────────────────────────────────────────────────────────────

    def send_signal_alert(self, alert: SignalAlert) -> bool:
        """Format and dispatch *alert* to Telegram.

        Returns:
            ``True`` on success, ``False`` on any error or missing config.
        """
        if not self.is_configured:
            logger.warning(
                "Telegram not configured — set TELEGRAM_BOT_TOKEN and "
                "TELEGRAM_CHAT_ID in your .env file. Alert logged only:\n%s",
                self._format_message(alert),
            )
            return False

        return self._send(self._format_message(alert))

    def send_text(self, text: str) -> bool:
        """Send a raw text message (no formatting). Useful for startup/error notices."""
        if not self.is_configured:
            logger.warning("Telegram not configured. Message: %s", text)
            return False
        return self._send(text, parse_mode=None)

    # ── Private ───────────────────────────────────────────────────────────────

    def _format_message(self, a: SignalAlert) -> str:
        emoji    = _SIGNAL_EMOJI.get(a.signal, "⚪")
        conf_bar = "█" * a.confidence + "░" * (3 - a.confidence)
        conf_pct = int(a.confidence / 3 * 100)

        d   = a.decimals
        fmt = f".{d}f"

        # Percent distances from current price
        sl_pct = abs(a.price - a.stop_loss)   / a.price * 100
        tp_pct = abs(a.take_profit - a.price) / a.price * 100
        rr     = tp_pct / sl_pct if sl_pct > 0 else 0.0

        sl_sign = "-" if a.signal == "BUY" else "+"
        tp_sign = "+" if a.signal == "BUY" else "-"

        lines = [
            f"{emoji} {a.flag} *{a.signal} — {a.asset}*",
            "",
            f"📊 *Timeframe:*    `{a.timeframe}`",
            f"💰 *Preco atual:*  `{a.price:{fmt}}`",
            f"🛑 *Stop Loss:*    `{a.stop_loss:{fmt}}` ({sl_sign}{sl_pct:.2f}%)",
            f"🎯 *Take Profit:*  `{a.take_profit:{fmt}}` ({tp_sign}{tp_pct:.2f}%)",
            f"⚖️ *Risco/Retorno:* `1 : {rr:.1f}`",
            "",
            f"📈 *RSI (14):*     `{a.rsi:.1f}`",
            f"📉 *ATR:*          `{a.atr:{fmt}}`",
            "",
            f"🏆 *Win Rate hist:* `{a.win_rate:.1f}%`",
            f"💡 *Confianca:*    `{conf_bar}` {conf_pct}% ({a.confidence}/3 indicadores)",
        ]
        return "\n".join(lines)

    def _send(self, text: str, parse_mode: str | None = "Markdown") -> bool:
        payload: dict = {"chat_id": self._chat_id, "text": text}
        if parse_mode:
            payload["parse_mode"] = parse_mode

        try:
            resp = requests.post(self._url, json=payload, timeout=10)
            resp.raise_for_status()
            logger.info("Telegram alert sent OK (chat_id=%s)", self._chat_id)
            return True
        except requests.exceptions.HTTPError as exc:
            logger.error("Telegram HTTP error: %s — %s", exc, exc.response.text)
        except requests.exceptions.RequestException as exc:
            logger.error("Telegram request failed: %s", exc)
        return False
