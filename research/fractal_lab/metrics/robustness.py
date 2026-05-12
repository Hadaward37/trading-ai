from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..storage.models import PeriodResult


def compute_robustness(
    wf_results: List[PeriodResult],
    monte_carlo: Dict,
) -> Dict:
    """
    Robustness score (0.0 → 1.0) from walk-forward consistency + MC confidence.

    Components:
      consistency_score  — fraction of WF folds that are profitable (PF > 1.0)
      stability_score    — 1 - CV(expectancy across folds);  low variance = stable
      mc_prob_score      — MC probability of positive final PnL
      dd_risk_score      — 1 - (mc_max_dd_95pct / |total_pnl|); reward vs tail risk

    Final = 0.35*consistency + 0.30*stability + 0.20*mc_prob + 0.15*dd_risk
    """
    if not wf_results:
        return {"robustness_score": 0.0, "verdict": "NO_DATA"}

    pfs = [r.metrics.get("profit_factor", 0.0) for r in wf_results]
    expectations = [r.metrics.get("expectancy_pips", 0.0) for r in wf_results]
    trades_counts = [r.n_trades for r in wf_results]

    # Only evaluate folds with at least 1 trade
    valid_folds = [(pf, e) for pf, e, n in zip(pfs, expectations, trades_counts) if n > 0]

    if not valid_folds:
        return {"robustness_score": 0.0, "verdict": "NO_TRADES"}

    valid_pfs, valid_es = zip(*valid_folds)

    # Consistency: fraction of profitable folds
    consistency_score = float(np.mean([pf > 1.0 for pf in valid_pfs]))

    # Stability: coefficient of variation of expectancy
    e_arr = np.array(valid_es, dtype=float)
    if len(e_arr) > 1 and e_arr.mean() != 0:
        cv = float(np.std(e_arr, ddof=1)) / abs(float(e_arr.mean()))
        stability_score = max(0.0, min(1.0, 1.0 - cv))
    else:
        stability_score = 0.5  # single fold, neutral

    # Monte Carlo: probability of profit
    mc_prob_score = float(monte_carlo.get("prob_profitable", 0.5))

    # Drawdown risk: tail drawdown vs mean final PnL
    mc_dd_95 = float(monte_carlo.get("max_dd_95th_pct", 0.0))
    mc_pnl_mean = float(monte_carlo.get("final_pnl_mean", 1.0))
    if mc_pnl_mean > 0 and mc_dd_95 > 0:
        dd_risk_score = max(0.0, min(1.0, mc_pnl_mean / (mc_pnl_mean + mc_dd_95)))
    else:
        dd_risk_score = 0.5

    final = (
        0.35 * consistency_score
        + 0.30 * stability_score
        + 0.20 * mc_prob_score
        + 0.15 * dd_risk_score
    )
    final = round(min(1.0, max(0.0, final)), 4)

    if final >= 0.70:
        verdict = "ROBUST"
    elif final >= 0.50:
        verdict = "MODERATE"
    elif final >= 0.30:
        verdict = "WEAK"
    else:
        verdict = "FRAGILE"

    return {
        "robustness_score": final,
        "verdict": verdict,
        "consistency_score": round(consistency_score, 4),
        "stability_score": round(stability_score, 4),
        "mc_prob_score": round(mc_prob_score, 4),
        "dd_risk_score": round(dd_risk_score, 4),
        "n_wf_folds": len(wf_results),
        "n_valid_folds": len(valid_folds),
        "pf_per_fold": [round(pf, 3) for pf in valid_pfs],
        "expectancy_per_fold": [round(e, 2) for e in valid_es],
    }


def stability_score(wf_results: List[PeriodResult]) -> float:
    """Rolling variance of expectancy across WF folds. Lower = more stable."""
    if len(wf_results) < 2:
        return 0.5
    expectations = np.array([r.metrics.get("expectancy_pips", 0.0) for r in wf_results])
    cv = float(np.std(expectations, ddof=1)) / (abs(float(expectations.mean())) + 1e-9)
    return round(max(0.0, min(1.0, 1.0 - cv)), 4)
