from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..storage.models import Trade


def compute_metrics(trades: List[Trade]) -> Dict[str, float]:
    """
    Compute all standard performance metrics from a list of trades.

    Returns empty-safe dict with NaN for metrics that require more data.
    """
    if not trades:
        return _empty_metrics()

    pnls = np.array([t.pnl_pips for t in trades], dtype=float)
    n = len(pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    # Profit Factor
    total_win = wins.sum() if len(wins) > 0 else 0.0
    total_loss = abs(losses.sum()) if len(losses) > 0 else 1e-9
    pf = round(total_win / total_loss, 4)

    # Win rate
    win_rate = round(len(wins) / n, 4)

    # Expectancy
    expectancy = round(float(np.mean(pnls)), 4)

    # Sharpe (simplified: E / std, not annualised — use for relative comparison)
    std = float(np.std(pnls, ddof=1)) if n > 1 else 1e-9
    sharpe = round(expectancy / std, 4) if std > 1e-10 else 0.0

    # Max drawdown on cumulative PnL curve
    cum = np.cumsum(pnls)
    roll_max = np.maximum.accumulate(cum)
    drawdown_series = roll_max - cum
    max_dd = round(float(drawdown_series.max()), 4)

    # Average bars held
    avg_bars = round(float(np.mean([t.bars_held for t in trades])), 2)

    # Payoff ratio (avg win / avg loss)
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = abs(float(losses.mean())) if len(losses) > 0 else 1e-9
    payoff = round(avg_win / avg_loss, 4)

    # Recovery factor (total pnl / max drawdown)
    total_pnl = round(float(pnls.sum()), 4)
    recovery = round(total_pnl / max_dd, 4) if max_dd > 0 else float("inf")

    # Outcome breakdown
    outcomes = [t.outcome for t in trades]
    tp_count = outcomes.count("TP_HIT")
    sl_count = outcomes.count("SL_HIT")
    to_count = outcomes.count("TIMEOUT")

    return {
        "n_trades": n,
        "win_rate": win_rate,
        "profit_factor": pf,
        "expectancy_pips": expectancy,
        "sharpe": sharpe,
        "max_drawdown_pips": max_dd,
        "total_pnl_pips": total_pnl,
        "avg_bars_held": avg_bars,
        "payoff_ratio": payoff,
        "recovery_factor": recovery,
        "tp_count": tp_count,
        "sl_count": sl_count,
        "timeout_count": to_count,
        "pnl_std": round(std, 4),
        "pnl_mean": expectancy,
    }


def compute_regime_breakdown(trades: List[Trade]) -> Dict[str, Dict]:
    """Compute metrics per regime for the trade list."""
    regimes = set(t.regime for t in trades if t.regime)
    result = {}
    for regime in sorted(regimes):
        group = [t for t in trades if t.regime == regime]
        result[regime] = compute_metrics(group)
    return result


def _empty_metrics() -> Dict[str, float]:
    keys = [
        "n_trades", "win_rate", "profit_factor", "expectancy_pips",
        "sharpe", "max_drawdown_pips", "total_pnl_pips", "avg_bars_held",
        "payoff_ratio", "recovery_factor", "tp_count", "sl_count",
        "timeout_count", "pnl_std", "pnl_mean",
    ]
    return {k: 0.0 for k in keys}
