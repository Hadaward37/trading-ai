"""Paper Trading — carteira virtual com persistência SQLite.

Recursos:
  - Capital inicial configurável (padrão: config.PAPER_TRADING_CAPITAL)
  - Uma posição aberta por ativo (troca de direção fecha e reabre)
  - SL/TP checados contra high/low dos últimos candles (sem tick data)
  - Métricas em tempo real: saldo, P&L, win rate, drawdown
  - WAL mode para leitura segura pelo dashboard em paralelo

Tabelas SQLite criadas/gerenciadas por este módulo:
  paper_trades, portfolio_state, balance_history
"""

from __future__ import annotations

import logging
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional

import pandas as pd

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config

logger = logging.getLogger(__name__)

_DB: Path = config.ROOT_DIR / "data" / "trading.db"

# ── DDL ───────────────────────────────────────────────────────────────────────

_DDL = [
    """
    CREATE TABLE IF NOT EXISTS paper_trades (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol         TEXT    NOT NULL,
        asset_name     TEXT    NOT NULL,
        direction      TEXT    NOT NULL,
        entry_price    REAL    NOT NULL,
        stop_loss      REAL    NOT NULL,
        take_profit    REAL    NOT NULL,
        quantity       REAL    NOT NULL,
        position_value REAL    NOT NULL,
        entry_time     TEXT    NOT NULL,
        exit_price     REAL,
        exit_time      TEXT,
        exit_reason    TEXT,
        pnl            REAL,
        pnl_pct        REAL,
        status         TEXT    NOT NULL DEFAULT 'OPEN',
        signal_score   REAL    DEFAULT 0,
        final_score    REAL    DEFAULT 0,
        sentiment      TEXT    DEFAULT 'NEUTRAL',
        timeframe      TEXT    DEFAULT '1h'
    )""",
    """
    CREATE TABLE IF NOT EXISTS portfolio_state (
        id              INTEGER PRIMARY KEY CHECK (id = 1),
        balance         REAL    NOT NULL,
        initial_capital REAL    NOT NULL,
        updated_at      TEXT    NOT NULL
    )""",
    """
    CREATE TABLE IF NOT EXISTS balance_history (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        balance   REAL    NOT NULL,
        equity    REAL    NOT NULL,
        timestamp TEXT    NOT NULL
    )""",
]


# ── Connection helper ─────────────────────────────────────────────────────────

@contextmanager
def _conn(db: Path = _DB) -> Generator[sqlite3.Connection, None, None]:
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db), timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


# ── Main class ────────────────────────────────────────────────────────────────

class VirtualPortfolio:
    """Carteira virtual de paper trading com persistência SQLite."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        initial_capital: Optional[float] = None,
    ) -> None:
        self._db = db_path or _DB
        self._initial = initial_capital or config.PAPER_TRADING_CAPITAL
        self._init_db()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with _conn(self._db) as con:
            for ddl in _DDL:
                con.execute(ddl)
            # Initialize portfolio_state only if not present
            existing = con.execute("SELECT id FROM portfolio_state WHERE id=1").fetchone()
            if existing is None:
                now = _now()
                con.execute(
                    "INSERT INTO portfolio_state (id, balance, initial_capital, updated_at) VALUES (1,?,?,?)",
                    (self._initial, self._initial, now),
                )
                con.execute(
                    "INSERT INTO balance_history (balance, equity, timestamp) VALUES (?,?,?)",
                    (self._initial, self._initial, now),
                )
                logger.info("Portfolio inicializado: capital=%.2f", self._initial)

    # ── Public API ────────────────────────────────────────────────────────────

    def open_trade(
        self,
        symbol: str,
        asset_name: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        signal_score: float = 0.0,
        final_score: float = 0.0,
        sentiment: str = "NEUTRAL",
        timeframe: str = "1h",
    ) -> Optional[int]:
        """Abre nova posição. Retorna trade_id ou None se não houver capital.

        Se já existir posição aberta neste ativo, encerra antes de abrir nova.
        """
        balance = self._get_balance()
        position_value = round(balance * config.POSITION_SIZE_PCT, 2)

        if position_value < 1.0:
            logger.warning("[PT] Saldo insuficiente para abrir trade: %.2f", balance)
            return None

        # Fecha posição contrária existente (sem saída de mercado real)
        open_now = self._open_for_symbol(symbol)
        if open_now:
            if open_now["direction"] == direction:
                logger.info("[PT] Já em %s %s — ignorando", direction, asset_name)
                return None
            self._close_trade(open_now["id"], entry_price, "REVERSED")

        quantity = position_value / entry_price
        now = _now()

        with _conn(self._db) as con:
            cur = con.execute(
                """INSERT INTO paper_trades
                   (symbol, asset_name, direction, entry_price, stop_loss, take_profit,
                    quantity, position_value, entry_time, status,
                    signal_score, final_score, sentiment, timeframe)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (symbol, asset_name, direction, entry_price, stop_loss, take_profit,
                 quantity, position_value, now, "OPEN",
                 signal_score, final_score, sentiment, timeframe),
            )
            trade_id = cur.lastrowid

        logger.info(
            "[PT] Abriu #%d %s %s @ %.5f | SL=%.5f TP=%.5f | val=%.2f",
            trade_id, direction, asset_name, entry_price, stop_loss, take_profit, position_value,
        )
        return trade_id

    def check_and_close(self, symbol: str, df: pd.DataFrame) -> list[dict]:
        """Verifica posições abertas contra os últimos candles do DataFrame.

        Checa high/low dos últimos 3 candles (conservador — SL tem prioridade).
        Retorna lista de operações fechadas para alertas.
        """
        trade = self._open_for_symbol(symbol)
        if trade is None:
            return []

        closed: list[dict] = []
        recent = df.iloc[-3:]  # últimos 3 candles (2 completos + 1 em formação)

        for _, bar in recent.iterrows():
            high = float(bar["high"])
            low  = float(bar["low"])
            direction = trade["direction"]

            hit_sl = (direction == "BUY"  and low  <= trade["stop_loss"]) or \
                     (direction == "SELL" and high >= trade["stop_loss"])
            hit_tp = (direction == "BUY"  and high >= trade["take_profit"]) or \
                     (direction == "SELL" and low  <= trade["take_profit"])

            # SL tem prioridade quando ambos atingidos no mesmo candle
            if hit_sl:
                result = self._close_trade(trade["id"], trade["stop_loss"], "SL_HIT")
                closed.append(result)
                break
            if hit_tp:
                result = self._close_trade(trade["id"], trade["take_profit"], "TP_HIT")
                closed.append(result)
                break

        return closed

    def get_metrics(self) -> dict:
        """Retorna métricas completas da carteira virtual."""
        balance  = self._get_balance()
        initial  = self._get_initial_capital()
        total_pnl     = balance - initial
        total_pnl_pct = total_pnl / initial * 100 if initial > 0 else 0.0

        closed = self.get_closed_trades()
        total  = len(closed)
        wins   = int((closed["pnl"] > 0).sum()) if total > 0 else 0
        win_rate = wins / total * 100 if total > 0 else 0.0

        history = self.get_balance_history()
        if len(history) > 1:
            peak     = history["balance"].cummax()
            drawdown = float(((history["balance"] - peak) / peak * 100).min())
        else:
            drawdown = 0.0

        open_trades = self.get_open_trades()

        return {
            "balance":        round(balance, 2),
            "initial_capital": round(initial, 2),
            "total_pnl":      round(total_pnl, 2),
            "total_pnl_pct":  round(total_pnl_pct, 2),
            "total_trades":   total,
            "wins":           wins,
            "losses":         total - wins,
            "win_rate":       round(win_rate, 1),
            "max_drawdown":   round(drawdown, 2),
            "open_positions": len(open_trades),
        }

    def get_open_trades(self) -> pd.DataFrame:
        with _conn(self._db) as con:
            rows = con.execute(
                "SELECT * FROM paper_trades WHERE status='OPEN' ORDER BY entry_time DESC"
            ).fetchall()
        return _rows_to_df(rows)

    def get_closed_trades(self, n: int = 50) -> pd.DataFrame:
        with _conn(self._db) as con:
            rows = con.execute(
                "SELECT * FROM paper_trades WHERE status='CLOSED' ORDER BY exit_time DESC LIMIT ?",
                (n,),
            ).fetchall()
        return _rows_to_df(rows)

    def get_balance_history(self) -> pd.DataFrame:
        with _conn(self._db) as con:
            rows = con.execute(
                "SELECT timestamp, balance, equity FROM balance_history ORDER BY timestamp"
            ).fetchall()
        if not rows:
            return pd.DataFrame(columns=["timestamp", "balance", "equity"])
        df = pd.DataFrame([dict(r) for r in rows])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def reset(self) -> None:
        """Reinicia a carteira com o capital inicial. Apaga todo o histórico."""
        with _conn(self._db) as con:
            con.execute("DELETE FROM paper_trades")
            con.execute("DELETE FROM balance_history")
            now = _now()
            con.execute(
                "UPDATE portfolio_state SET balance=?, initial_capital=?, updated_at=? WHERE id=1",
                (self._initial, self._initial, now),
            )
            con.execute(
                "INSERT INTO balance_history (balance, equity, timestamp) VALUES (?,?,?)",
                (self._initial, self._initial, now),
            )
        logger.info("[PT] Carteira resetada: capital=%.2f", self._initial)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _open_for_symbol(self, symbol: str) -> Optional[sqlite3.Row]:
        with _conn(self._db) as con:
            return con.execute(
                "SELECT * FROM paper_trades WHERE symbol=? AND status='OPEN' LIMIT 1",
                (symbol,),
            ).fetchone()

    def _get_balance(self) -> float:
        with _conn(self._db) as con:
            row = con.execute("SELECT balance FROM portfolio_state WHERE id=1").fetchone()
        return float(row["balance"]) if row else self._initial

    def _get_initial_capital(self) -> float:
        with _conn(self._db) as con:
            row = con.execute("SELECT initial_capital FROM portfolio_state WHERE id=1").fetchone()
        return float(row["initial_capital"]) if row else self._initial

    def _close_trade(self, trade_id: int, exit_price: float, reason: str) -> dict:
        now = _now()
        with _conn(self._db) as con:
            trade = con.execute(
                "SELECT * FROM paper_trades WHERE id=?", (trade_id,)
            ).fetchone()
            if trade is None:
                return {}

            direction      = trade["direction"]
            entry_price    = trade["entry_price"]
            quantity       = trade["quantity"]
            position_value = trade["position_value"]

            if direction == "BUY":
                pnl = (exit_price - entry_price) * quantity
            else:
                pnl = (entry_price - exit_price) * quantity

            pnl_pct = pnl / position_value * 100 if position_value > 0 else 0.0

            con.execute(
                """UPDATE paper_trades SET
                   exit_price=?, exit_time=?, exit_reason=?,
                   pnl=?, pnl_pct=?, status='CLOSED'
                   WHERE id=?""",
                (exit_price, now, reason, round(pnl, 4), round(pnl_pct, 2), trade_id),
            )

            # Update portfolio balance
            con.execute(
                "UPDATE portfolio_state SET balance=balance+?, updated_at=? WHERE id=1",
                (round(pnl, 4), now),
            )

        # Save balance snapshot after close
        new_balance = self._get_balance()
        with _conn(self._db) as con:
            con.execute(
                "INSERT INTO balance_history (balance, equity, timestamp) VALUES (?,?,?)",
                (new_balance, new_balance, now),
            )

        result = {
            "trade_id":     trade_id,
            "asset_name":   trade["asset_name"],
            "direction":    direction,
            "entry_price":  entry_price,
            "exit_price":   exit_price,
            "exit_reason":  reason,
            "pnl":          round(pnl, 4),
            "pnl_pct":      round(pnl_pct, 2),
            "new_balance":  new_balance,
            "symbol":       trade["symbol"],
        }
        logger.info(
            "[PT] Fechou #%d %s %s | %s | P&L=%.2f (%.2f%%) | Saldo=%.2f",
            trade_id, direction, trade["asset_name"],
            reason, pnl, pnl_pct, new_balance,
        )
        return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _rows_to_df(rows: list) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])
