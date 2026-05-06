"""Macro context module — Copom tone and its impact on Brazilian assets.

Reads data/copom_tone.json (written by scripts/ingest_copom.py).
Provides score adjustments for BR equities based on Copom stance.
EUR/USD is not affected by the Copom (different monetary policy).

Usage:
    from core.macro_context import get_copom_context, copom_score_adjustment
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config

logger = logging.getLogger(__name__)

TONE_PATH    = _ROOT / "data" / "copom_tone.json"
CACHE_TTL_H  = 6          # hours before re-fetching
_cache: dict = {}


# ── Tone labels ───────────────────────────────────────────────────────────────

TONE_LABEL = {
    "hawkish": "Hawkish — juros em alta, impacto negativo em acoes BR",
    "dovish":  "Dovish  — juros em queda, impacto positivo em acoes BR",
    "neutral": "Neutro  — sem viés monetário claro",
}

TONE_EMOJI = {
    "hawkish": "🦅",
    "dovish":  "🕊",
    "neutral": "⚖️",
}

# Brazilian assets affected by Copom
BR_SYMBOLS = {"VALE3.SA", "PETR4.SA", "ITUB4.SA", "BBDC4.SA", "^BVSP"}


# ── Public API ────────────────────────────────────────────────────────────────

def get_copom_context(refresh: bool = False) -> dict:
    """Return cached Copom macro context, refreshing if stale.

    Returns:
        dict with keys: copom_tone, impact_brl, impact_stocks, selic_rate,
        selic_delta_3m, confidence, updated_at.
        Falls back to neutral if data unavailable.
    """
    global _cache

    if _cache and not refresh:
        age_h = (datetime.now(timezone.utc).timestamp() - _cache.get("updated_at_ts", 0)) / 3600
        if age_h < CACHE_TTL_H:
            return _cache

    if TONE_PATH.exists():
        try:
            data   = json.loads(TONE_PATH.read_text(encoding="utf-8"))
            age_h  = (datetime.now(timezone.utc).timestamp() - data.get("updated_at_ts", 0)) / 3600
            _cache = data
            if age_h > CACHE_TTL_H:
                logger.info("Copom tone cache stale (%.1fh). Run scripts/ingest_copom.py to refresh.", age_h)
            return _cache
        except Exception as exc:
            logger.warning("Failed to read copom_tone.json: %s", exc)

    logger.warning("copom_tone.json not found — returning neutral context. Run scripts/ingest_copom.py.")
    return {
        "copom_tone":     "neutral",
        "impact_brl":     "neutral",
        "impact_stocks":  "neutral",
        "selic_rate":     None,
        "selic_delta_3m": 0.0,
        "confidence":     "low",
        "updated_at":     None,
        "updated_at_ts":  0,
        "chromadb_ok":    False,
    }


def copom_score_adjustment(symbol: str, signal: int) -> float:
    """Return score adjustment (pts) for a given symbol and signal direction.

    Rules:
      - EUR/USD: no adjustment (Copom doesn't affect EUR/USD)
      - BR assets + dovish Copom + BUY  →  +COPOM_DOVISH_BONUS  pts
      - BR assets + dovish Copom + SELL → -COPOM_DOVISH_BONUS  pts (suppress sells)
      - BR assets + hawkish Copom + BUY → -COPOM_HAWKISH_PENALTY pts
      - BR assets + hawkish Copom + SELL→ +COPOM_HAWKISH_PENALTY pts (reinforce sells)

    Args:
        symbol: yfinance ticker (e.g. "VALE3.SA", "EURUSD=X")
        signal: 1=BUY | -1=SELL | 0=HOLD

    Returns:
        Float score adjustment in range [-PENALTY, +BONUS]. 0.0 for EUR/USD.
    """
    if symbol not in BR_SYMBOLS or signal == 0:
        return 0.0

    ctx  = get_copom_context()
    tone = ctx.get("copom_tone", "neutral")

    bonus   = getattr(config, "COPOM_DOVISH_BONUS",    10.0)
    penalty = getattr(config, "COPOM_HAWKISH_PENALTY", 10.0)

    if tone == "dovish":
        return signal * bonus      #  BUY +bonus | SELL -bonus
    if tone == "hawkish":
        return -signal * penalty   #  BUY -penalty | SELL +penalty
    return 0.0


def format_telegram_line() -> str:
    """Return a formatted one-liner for Telegram alerts."""
    ctx   = get_copom_context()
    tone  = ctx.get("copom_tone", "neutral")
    rate  = ctx.get("selic_rate")
    emoji = TONE_EMOJI.get(tone, "⚖️")
    label = TONE_LABEL.get(tone, "Neutro")

    rate_str = f" | Selic {rate}% aa" if rate else ""
    return f"{emoji} Copom: {label}{rate_str}"


def query_pythex(question: str) -> str:
    """Ask the Pythex ChromaDB RAG about Copom macro context."""
    try:
        pythex_root = Path(r"C:\Users\dudut\pythex-ia-engine")
        import os
        orig_dir = os.getcwd()
        os.chdir(str(pythex_root))
        sys.path.insert(0, str(pythex_root))

        from dotenv import load_dotenv
        load_dotenv(pythex_root / ".env")

        from app.agent import ask
        result = ask(cliente="copom", question=question, session_id="macro")
        os.chdir(orig_dir)
        return result.get("answer", "")
    except Exception as exc:
        logger.warning("Pythex query failed: %s", exc)
        return ""
