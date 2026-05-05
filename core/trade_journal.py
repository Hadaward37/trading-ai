"""Trade Journal — diário automático de operações de paper trading.

Recursos:
  - Export CSV diário para data/journal/YYYY-MM-DD.csv
  - Resumo do dia: total de sinais, execuções, P&L, win rate
  - Lê diretamente do SQLite (tabela paper_trades)
"""

from __future__ import annotations

import csv
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from core.paper_trading import VirtualPortfolio, _conn

logger = logging.getLogger(__name__)

_JOURNAL_DIR: Path = config.ROOT_DIR / "data" / "journal"

_CSV_FIELDS = [
    "id", "asset_name", "direction", "entry_price", "stop_loss", "take_profit",
    "quantity", "position_value", "entry_time", "exit_price", "exit_time",
    "exit_reason", "pnl", "pnl_pct", "status", "signal_score", "final_score",
    "sentiment", "timeframe",
]


def export_daily_csv(target_date: Optional[date] = None) -> Path:
    """Exporta todas as operações do dia para CSV.

    Args:
        target_date: Data alvo (padrão: hoje UTC).

    Returns:
        Caminho do arquivo CSV gerado.
    """
    if target_date is None:
        target_date = datetime.now(tz=timezone.utc).date()

    date_str  = target_date.strftime("%Y-%m-%d")
    _JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _JOURNAL_DIR / f"{date_str}.csv"

    db = config.ROOT_DIR / "data" / "trading.db"
    with _conn(db) as con:
        rows = con.execute(
            """SELECT * FROM paper_trades
               WHERE date(entry_time) = ? OR date(exit_time) = ?
               ORDER BY entry_time""",
            (date_str, date_str),
        ).fetchall()

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))

    logger.info("Journal exportado: %s (%d operações)", csv_path, len(rows))
    return csv_path


def daily_summary(target_date: Optional[date] = None) -> dict:
    """Retorna resumo do dia de paper trading.

    Returns:
        dict com: date, total_trades, open_trades, closed_trades,
                  wins, losses, win_rate, total_pnl, best_trade, worst_trade
    """
    if target_date is None:
        target_date = datetime.now(tz=timezone.utc).date()

    date_str = target_date.strftime("%Y-%m-%d")
    db = config.ROOT_DIR / "data" / "trading.db"

    with _conn(db) as con:
        all_rows = con.execute(
            "SELECT * FROM paper_trades WHERE date(entry_time) = ?",
            (date_str,),
        ).fetchall()

    trades   = [dict(r) for r in all_rows]
    closed   = [t for t in trades if t["status"] == "CLOSED"]
    wins     = [t for t in closed if (t["pnl"] or 0) > 0]
    losses   = [t for t in closed if (t["pnl"] or 0) <= 0]
    total_pnl = sum(t["pnl"] or 0 for t in closed)

    best  = max(closed, key=lambda t: t["pnl"] or 0, default=None)
    worst = min(closed, key=lambda t: t["pnl"] or 0, default=None)

    return {
        "date":          date_str,
        "total_trades":  len(trades),
        "open_trades":   len(trades) - len(closed),
        "closed_trades": len(closed),
        "wins":          len(wins),
        "losses":        len(losses),
        "win_rate":      round(len(wins) / len(closed) * 100, 1) if closed else 0.0,
        "total_pnl":     round(total_pnl, 2),
        "best_trade":    round(best["pnl"], 2) if best else 0.0,
        "worst_trade":   round(worst["pnl"], 2) if worst else 0.0,
    }


def print_daily_summary(target_date: Optional[date] = None) -> None:
    """Imprime resumo formatado do dia no log."""
    s = daily_summary(target_date)
    logger.info(
        "=== Resumo Paper Trading %s === | Trades: %d | Fechados: %d | "
        "Wins: %d | Losses: %d | WinRate: %.1f%% | P&L: %.2f",
        s["date"], s["total_trades"], s["closed_trades"],
        s["wins"], s["losses"], s["win_rate"], s["total_pnl"],
    )
