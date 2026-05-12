from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..storage.models import Trade


def run_monte_carlo(
    trades: List[Trade],
    n_simulations: int = 1000,
    seed: int = 42,
) -> Dict:
    """
    Monte Carlo simulation by bootstrapping trade outcomes.

    Resamples the trade PnL sequence N times (with replacement) to build a
    distribution of possible equity curves. Quantifies path-dependency risk.

    Returns a dict with summary statistics over N simulations.
    """
    if len(trades) < 5:
        return {
            "n_simulations": 0,
            "note": "insufficient trades (< 5)",
            "final_pnl_mean": 0.0,
            "final_pnl_std": 0.0,
            "max_dd_mean": 0.0,
            "max_dd_std": 0.0,
            "prob_profitable": 0.0,
            "pnl_5th_pct": 0.0,
            "pnl_95th_pct": 0.0,
        }

    rng = np.random.default_rng(seed)
    pnls = np.array([t.pnl_pips for t in trades], dtype=float)
    n = len(pnls)

    final_pnls = np.empty(n_simulations)
    max_drawdowns = np.empty(n_simulations)

    for i in range(n_simulations):
        sample = rng.choice(pnls, size=n, replace=True)
        cum = np.cumsum(sample)
        final_pnls[i] = cum[-1]
        roll_max = np.maximum.accumulate(cum)
        max_drawdowns[i] = (roll_max - cum).max()

    return {
        "n_simulations": n_simulations,
        "n_trades_base": n,
        "final_pnl_mean": round(float(final_pnls.mean()), 2),
        "final_pnl_std": round(float(final_pnls.std()), 2),
        "final_pnl_median": round(float(np.median(final_pnls)), 2),
        "max_dd_mean": round(float(max_drawdowns.mean()), 2),
        "max_dd_std": round(float(max_drawdowns.std()), 2),
        "max_dd_95th_pct": round(float(np.percentile(max_drawdowns, 95)), 2),
        "prob_profitable": round(float((final_pnls > 0).mean()), 4),
        "pnl_5th_pct": round(float(np.percentile(final_pnls, 5)), 2),
        "pnl_95th_pct": round(float(np.percentile(final_pnls, 95)), 2),
        "pnl_25th_pct": round(float(np.percentile(final_pnls, 25)), 2),
        "pnl_75th_pct": round(float(np.percentile(final_pnls, 75)), 2),
    }
