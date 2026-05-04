"""Vectorized backtesting engine with ATR-based stop-loss / take-profit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """Aggregated performance metrics from a single backtest run."""

    total_return: float    # percent, e.g. 12.5 means +12.5 %
    sharpe_ratio: float    # annualised (uses ~6 500 trading hours/year)
    max_drawdown: float    # percent (negative), e.g. -8.3
    win_rate: float        # percent of profitable trades
    total_trades: int
    equity_curve: pd.Series
    trades: pd.DataFrame


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = 10_000.0,
    position_size_pct: float = 0.10,
    sl_atr_mult: float = 1.5,
    tp_atr_mult: float = 3.0,
) -> BacktestResult:
    """Simulate trades driven by the ``signal`` column in *df*.

    Entry is taken on the bar *after* a signal appears.
    Exit happens when: stop-loss hit, take-profit hit, or an opposing signal
    appears on the previous bar.

    Args:
        df: DataFrame with OHLCV + all indicators + ``signal`` column.
        initial_capital: Starting cash in account currency.
        position_size_pct: Fraction of current capital committed per trade.
        sl_atr_mult: ATR multiplier for stop-loss distance.
        tp_atr_mult: ATR multiplier for take-profit distance.

    Returns:
        :class:`BacktestResult` with metrics and per-trade log.
    """
    data = df.dropna().reset_index()

    capital: float = initial_capital
    position: int = 0       # 1 = long, -1 = short, 0 = flat
    entry_price: float = 0.0
    entry_idx: int = 0
    stop_loss: float = 0.0
    take_profit: float = 0.0

    equity: list[float] = [capital]
    trades: list[dict] = []

    for i in range(1, len(data)):
        row  = data.iloc[i]
        prev = data.iloc[i - 1]
        price: float = float(row["close"])
        atr: float   = float(row["atr"])

        # ── Check exit ────────────────────────────────────────────────────────
        if position != 0:
            exit_triggered = (
                (position ==  1 and (price <= stop_loss or price >= take_profit or prev["signal"] == -1)) or
                (position == -1 and (price >= stop_loss or price <= take_profit or prev["signal"] ==  1))
            )
            if exit_triggered:
                pnl = (price - entry_price) * position / entry_price * capital * position_size_pct
                capital += pnl
                trades.append({
                    "entry_time":  data.iloc[entry_idx]["datetime"],
                    "exit_time":   row["datetime"],
                    "direction":   "long" if position == 1 else "short",
                    "entry_price": round(entry_price, 5),
                    "exit_price":  round(price, 5),
                    "pnl":         round(pnl, 4),
                })
                position = 0

        # ── Check entry ───────────────────────────────────────────────────────
        if position == 0 and atr > 0:
            sig = int(prev["signal"])
            if sig == 1:
                position, entry_price, entry_idx = 1, price, i
                stop_loss  = price - atr * sl_atr_mult
                take_profit = price + atr * tp_atr_mult
            elif sig == -1:
                position, entry_price, entry_idx = -1, price, i
                stop_loss  = price + atr * sl_atr_mult
                take_profit = price - atr * tp_atr_mult

        equity.append(capital)

    equity_series = pd.Series(equity, index=data["datetime"][: len(equity)])
    trades_df = (
        pd.DataFrame(trades)
        if trades
        else pd.DataFrame(columns=["entry_time", "exit_time", "direction",
                                   "entry_price", "exit_price", "pnl"])
    )

    # ── Metrics ───────────────────────────────────────────────────────────────
    total_return = (capital - initial_capital) / initial_capital * 100

    bar_returns = equity_series.pct_change().dropna()
    # ~6 500 trading hours per year (24h × 5 days × 54 weeks)
    sharpe = (
        bar_returns.mean() / bar_returns.std() * np.sqrt(6_500)
        if bar_returns.std() > 0
        else 0.0
    )

    rolling_max = equity_series.cummax()
    max_dd = ((equity_series - rolling_max) / rolling_max).min() * 100

    win_rate = (
        (trades_df["pnl"] > 0).mean() * 100 if not trades_df.empty else 0.0
    )

    return BacktestResult(
        total_return=round(total_return, 2),
        sharpe_ratio=round(float(sharpe), 3),
        max_drawdown=round(float(max_dd), 2),
        win_rate=round(float(win_rate), 2),
        total_trades=len(trades_df),
        equity_curve=equity_series,
        trades=trades_df,
    )
