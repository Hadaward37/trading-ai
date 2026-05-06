"""Pythex bridge — explica sinais de trading usando OpenAI gpt-4o-mini.

Arquitetura: lê dados locais (CSV + SQLite) como contexto e envia ao LLM,
retornando uma explicação curta e executiva do sinal atual.

Fluxo:
  load_context() → monta snapshot dos últimos sinais + journal
  query(pergunta) → chama gpt-4o-mini com esse contexto
"""

from __future__ import annotations

import csv
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

import config

logger = logging.getLogger(__name__)

_SIGNALS_CSV = _ROOT / "data" / "mt5_signals.csv"
_DB_PATH     = _ROOT / "data" / "trading.db"

_SYSTEM_PROMPT = """\
Você é o Pythex Trading Analyst, especialista em análise técnica de Forex.
Responda de forma executiva, em até 2 frases, SEM markdown, em português.
Base sua análise APENAS no contexto fornecido — indicadores, histórico de sinais e journal de trades.
Não invente dados. Se não souber, diga "dados insuficientes".
"""


# ── Context builders ──────────────────────────────────────────────────────────

def _load_last_signals(n: int = 10) -> str:
    """Return the last N signal rows from mt5_signals.csv as text."""
    if not _SIGNALS_CSV.exists():
        return "Nenhum sinal disponível."
    try:
        with open(_SIGNALS_CSV, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return "Arquivo de sinais vazio."
        tail = rows[-n:]
        lines = ["Últimos sinais (data | direção | entry | SL | TP | score | ATR):"]
        for r in tail:
            sig = "BUY" if r.get("signal") == "1" else "SELL"
            lines.append(
                f"  {r.get('datetime','?')} | {sig} | "
                f"entry={r.get('entry','?')} SL={r.get('sl','?')} "
                f"TP={r.get('tp','?')} score={r.get('score','?')} ATR={r.get('atr','?')}"
            )
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("Falha ao ler signals CSV: %s", exc)
        return "Erro ao ler sinais."


def _load_journal_summary() -> str:
    """Return a short stats summary from the trade_journal table."""
    if not _DB_PATH.exists():
        return "Journal não disponível."
    try:
        conn = sqlite3.connect(_DB_PATH)
        cur  = conn.cursor()

        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trade_journal'")
        if not cur.fetchone():
            conn.close()
            return "Tabela trade_journal não encontrada."

        cur.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                ROUND(AVG(pnl), 4) as avg_pnl,
                ROUND(SUM(pnl), 4) as total_pnl,
                MIN(entry_time) as first_trade,
                MAX(entry_time) as last_trade
            FROM trade_journal
        """)
        row = cur.fetchone()
        conn.close()

        if not row or row[0] == 0:
            return "Journal sem registros ainda."

        total, wins, avg_pnl, total_pnl, first, last = row
        win_rate = wins / total * 100 if total else 0
        return (
            f"Journal de trades: {total} trades | "
            f"Win rate: {win_rate:.1f}% | "
            f"PnL médio: {avg_pnl} | "
            f"PnL total: {total_pnl} | "
            f"Período: {first} → {last}"
        )
    except Exception as exc:
        logger.warning("Falha ao ler journal: %s", exc)
        return "Erro ao ler journal."


def load_context() -> str:
    """Build the full context string sent to the LLM."""
    signals = _load_last_signals(10)
    journal = _load_journal_summary()
    params  = (
        f"Parâmetros ativos: RSI(14) Buy<{config.RSI_BUY} Sell>{config.RSI_SELL} | "
        f"SL {config.SL_ATR_MULT}x ATR | TP {config.TP_ATR_MULT}x ATR | "
        f"Ativo principal: {config.MT5_SYMBOL}"
    )
    return f"{params}\n\n{signals}\n\n{journal}"


# ── Public API ────────────────────────────────────────────────────────────────

def query(question: str, max_chars: int = 280) -> str:
    """Ask the Pythex LLM a question about the current trading context.

    Args:
        question:  Natural language question (e.g. "Por que o sistema gerou BUY?").
        max_chars: Soft cap on response length for Telegram compatibility.

    Returns:
        Plain-text explanation from gpt-4o-mini, or an error string.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return "OPENAI_API_KEY não configurada."

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        context = load_context()
        messages = [
            {"role": "system",  "content": _SYSTEM_PROMPT},
            {"role": "user",    "content": f"CONTEXTO:\n{context}\n\nPERGUNTA: {question}"},
        ]

        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            max_tokens=120,
            temperature=0,
        )
        answer = resp.choices[0].message.content.strip()
        # Trim to max_chars without cutting mid-word
        if len(answer) > max_chars:
            answer = answer[:max_chars].rsplit(" ", 1)[0] + "..."
        return answer

    except Exception as exc:
        logger.error("Pythex bridge error: %s", exc)
        return f"Análise indisponível ({type(exc).__name__})."


def explain_signal(signal: str, entry: float, sl: float, tp: float, score: float) -> str:
    """Convenience wrapper: generate a one-line explanation for a specific signal."""
    q = (
        f"O sistema gerou sinal {signal} em {entry:.5f} "
        f"(SL={sl:.5f}, TP={tp:.5f}, score={score:.0f}/100). "
        f"Explique em 1-2 frases por que esse sinal faz sentido com base nos indicadores e histórico."
    )
    return query(q)
