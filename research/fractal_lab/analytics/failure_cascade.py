"""
Failure Cascade Analysis.

Pergunta central: "losses sao isolados ou em cascata?"

Metricas por combinacao ativo x hipotese:
  - Cascades identificadas (sequencias consecutivas de perdas)
  - Profundidade media da cascata (n de losses consecutivos)
  - Drawdown medio por cascata (em pips)
  - Tempo medio de recuperacao pos-cascata (em trades)
  - P(perda | N losses consecutivos) — funcao "calor" da cascata
  - Max cascade depth observada

Base para:
  - Kill-switch automatico baseado em cascade depth
  - Exposure throttling apos N losses consecutivos

Causalidade estrita: todas as metricas usam apenas sequencia temporal de trades.
Nenhum trade futuro e consultado ao calcular metricas de trades passados.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._shared import BEHAVIORAL_MAP_DIR, pf_from_pnls, trades_to_df

OUTPUT_DIR = BEHAVIORAL_MAP_DIR

# Minimum streak length to qualify as a "cascade" (not just random noise)
MIN_CASCADE_DEPTH = 2
# Max streak length for conditional probability table
MAX_COND_STREAK = 7
# Laplace smoothing alpha for conditional probability estimation
LAPLACE_ALPHA = 0.5
# Min trades required for meaningful cascade analysis
MIN_TRADES = 20


# ── Cascade identification ─────────────────────────────────────────────────────

def extract_cascades(trades_df: pd.DataFrame) -> List[Dict]:
    """
    Identify all consecutive losing sequences (cascades) in temporal order.

    A cascade starts when a losing trade follows a win (or is the first trade)
    and ends when the sequence is broken by a winning trade.

    Returns list of cascade dicts with: start_idx, end_idx, depth, cumulative_pnl.
    """
    if trades_df.empty:
        return []

    wins   = trades_df["win"].values
    pnls   = trades_df["pnl_pips"].values
    n      = len(wins)
    cascades = []

    i = 0
    while i < n:
        if wins[i] == 0:   # losing trade
            start = i
            cum_pnl = pnls[i]
            j = i + 1
            while j < n and wins[j] == 0:
                cum_pnl += pnls[j]
                j += 1
            depth = j - start
            if depth >= MIN_CASCADE_DEPTH:
                cascades.append({
                    "start_idx":  start,
                    "end_idx":    j - 1,
                    "depth":      depth,
                    "cum_pnl":    round(float(cum_pnl), 2),
                    "entry_time": trades_df.iloc[start]["entry_time"],
                    "exit_time":  trades_df.iloc[j - 1]["entry_time"],
                })
            i = j
        else:
            i += 1

    return cascades


def compute_recovery_times(cascades: List[Dict], trades_df: pd.DataFrame) -> List[float]:
    """
    For each cascade, compute how many trades until cumulative PnL recovers to 0.
    Strictly causal: starts counting from the trade AFTER the cascade ends.

    Returns list of recovery times (in trades). If never recovered, returns inf.
    """
    recovery_times = []
    pnls = trades_df["pnl_pips"].values
    n    = len(pnls)

    for cascade in cascades:
        end_idx    = cascade["end_idx"]
        cascade_dd = cascade["cum_pnl"]  # negative

        if end_idx + 1 >= n:
            recovery_times.append(float("inf"))
            continue

        # Walk forward and find when cumulative pnl from end_idx+1 >= |cascade_dd|
        cum = 0.0
        recovered = False
        for k in range(end_idx + 1, n):
            cum += pnls[k]
            if cum >= abs(cascade_dd):
                recovery_times.append(float(k - end_idx))
                recovered = True
                break

        if not recovered:
            recovery_times.append(float("inf"))

    return recovery_times


def compute_conditional_loss_prob(wins: np.ndarray) -> Dict[int, Dict]:
    """
    P(next_trade_is_loss | last N trades were all losses) for N = 1..MAX_COND_STREAK.

    Uses Laplace smoothing to handle sparse counts:
    P = (count_loss_after_N + alpha) / (count_any_after_N + 2*alpha)

    Causal: we never look at future trades to compute this probability.
    """
    n = len(wins)
    result = {}

    for streak_len in range(1, MAX_COND_STREAK + 1):
        count_after_streak = 0    # times we had N consecutive losses
        count_loss_follows = 0    # times the next trade was also a loss

        for i in range(streak_len, n):
            # Check if the previous streak_len trades were all losses
            window = wins[i - streak_len:i]
            if np.all(window == 0):
                count_after_streak += 1
                if wins[i] == 0:
                    count_loss_follows += 1

        prob = (count_loss_follows + LAPLACE_ALPHA) / (count_after_streak + 2 * LAPLACE_ALPHA)

        result[streak_len] = {
            "streak_length":    streak_len,
            "n_observations":   count_after_streak,
            "n_loss_follows":   count_loss_follows,
            "p_loss":           round(float(prob), 4),
            "p_win":            round(1.0 - float(prob), 4),
            "risk_elevated":    bool(prob > 0.65),   # >65% loss rate after N losses
        }

    return result


# ── Summary metrics ────────────────────────────────────────────────────────────

def compute_cascade_summary(
    cascades: List[Dict],
    recovery_times: List[float],
    cond_probs: Dict[int, Dict],
    n_trades: int,
) -> Dict:
    """Aggregate cascade statistics for one combination."""
    if not cascades:
        return {
            "n_cascades":         0,
            "cascade_rate":       0.0,
            "note":               "no cascades detected",
        }

    depths      = np.array([c["depth"] for c in cascades], dtype=float)
    pnls        = np.array([c["cum_pnl"] for c in cascades], dtype=float)
    recoveries  = [r for r in recovery_times if r != float("inf")]
    n_unrecovered = len(recovery_times) - len(recoveries)

    # Throttle trigger recommendation
    trigger_streak = next(
        (n for n, d in cond_probs.items() if d.get("risk_elevated", False)),
        None,
    )

    return {
        "n_cascades":              len(cascades),
        "cascade_rate":            round(len(cascades) / max(n_trades, 1), 4),
        "avg_depth":               round(float(depths.mean()), 2),
        "median_depth":            float(np.median(depths)),
        "max_depth":               int(depths.max()),
        "p90_depth":               float(np.percentile(depths, 90)),
        "avg_cascade_dd_pips":     round(float(pnls.mean()), 2),
        "worst_cascade_dd_pips":   round(float(pnls.min()), 2),
        "avg_recovery_trades":     round(float(np.mean(recoveries)), 1) if recoveries else None,
        "median_recovery_trades":  round(float(np.median(recoveries)), 1) if recoveries else None,
        "pct_unrecovered":         round(n_unrecovered / len(recovery_times) * 100, 1),
        "throttle_trigger_at_N":   trigger_streak,
        "p_loss_after_1":          cond_probs.get(1, {}).get("p_loss"),
        "p_loss_after_2":          cond_probs.get(2, {}).get("p_loss"),
        "p_loss_after_3":          cond_probs.get(3, {}).get("p_loss"),
        "p_loss_after_5":          cond_probs.get(5, {}).get("p_loss"),
    }


# ── Public interface ───────────────────────────────────────────────────────────

class FailureCascadeAnalyzer:
    """
    Failure cascade analysis: identifies loss streaks, recovery times,
    and conditional loss probabilities.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_combination(
        self,
        trades: List,
        asset_key: str,
        hyp_name: str,
    ) -> Dict:
        """Full cascade analysis for one combination."""
        df = trades_to_df(trades)
        key = f"{asset_key}_{hyp_name}"

        if df.empty or len(df) < MIN_TRADES:
            return {
                "key":        key,
                "asset":      asset_key,
                "hypothesis": hyp_name,
                "n_trades":   len(df),
                "note":       f"Need >= {MIN_TRADES} trades",
            }

        cascades     = extract_cascades(df)
        recoveries   = compute_recovery_times(cascades, df)
        cond_probs   = compute_conditional_loss_prob(df["win"].values)
        summary      = compute_cascade_summary(cascades, recoveries, cond_probs, len(df))

        return {
            "key":          key,
            "asset":        asset_key,
            "hypothesis":   hyp_name,
            "n_trades":     len(df),
            **summary,
            "_cascades":    cascades,        # internal
            "_cond_probs":  cond_probs,      # internal
        }

    def run(self, combinations: List[Tuple]) -> Dict:
        """Run cascade analysis for all combinations and save outputs."""
        summaries  = []
        cond_rows  = []
        all_cascades = []

        for asset_key, hyp_name, _, trades in combinations:
            if not trades:
                continue
            res = self.analyze_combination(trades, asset_key, hyp_name)

            # Expand cascade detail
            if "_cascades" in res:
                for c in res["_cascades"]:
                    all_cascades.append({
                        "key":        res["key"],
                        "asset":      asset_key,
                        "hypothesis": hyp_name,
                        **{k: v for k, v in c.items() if k not in ("entry_time", "exit_time")},
                        "entry_time": str(c.get("entry_time", "")),
                        "exit_time":  str(c.get("exit_time", "")),
                    })
                del res["_cascades"]

            # Expand conditional probs
            if "_cond_probs" in res:
                for streak, d in res["_cond_probs"].items():
                    cond_rows.append({
                        "key":        res["key"],
                        "asset":      asset_key,
                        "hypothesis": hyp_name,
                        **d,
                    })
                del res["_cond_probs"]

            summaries.append(res)

        # Save
        if summaries:
            pd.DataFrame(summaries).to_csv(
                self.output_dir / "cascade_stats.csv", index=False, encoding="utf-8"
            )
        if cond_rows:
            pd.DataFrame(cond_rows).to_csv(
                self.output_dir / "conditional_loss_prob.csv", index=False, encoding="utf-8"
            )
        if all_cascades:
            pd.DataFrame(all_cascades).to_csv(
                self.output_dir / "cascade_detail.csv", index=False, encoding="utf-8"
            )

        report = self._format_report(summaries, cond_rows)
        (self.output_dir / "failure_cascade_report.txt").write_text(report, encoding="utf-8")
        print(report)

        return {
            "summaries":    summaries,
            "n_cond_rows":  len(cond_rows),
            "n_cascades":   len(all_cascades),
        }

    @staticmethod
    def _format_report(summaries: List[Dict], cond_rows: List[Dict]) -> str:
        lines = [
            "=" * 72,
            "  FAILURE CASCADE ANALYSIS",
            f"  Generated: {datetime.utcnow().isoformat()}",
            f"  Combinations: {len(summaries)}",
            "=" * 72,
            "",
            f"  {'Combination':<30} {'N':>5} {'Casc':>5} {'AvgDep':>7} {'MaxDep':>7} "
            f"{'AvgDD':>9} {'Recov':>7} {'Unrecov%':>9} {'Trigger':>8}",
            "  " + "-" * 100,
        ]

        valid = [s for s in summaries if s.get("n_cascades", 0) > 0]
        for s in sorted(valid, key=lambda x: -(x.get("avg_depth") or 0)):
            key    = s.get("key", "?")
            n      = s.get("n_trades", 0)
            nc     = s.get("n_cascades", 0)
            ad     = s.get("avg_depth") or 0
            md     = s.get("max_depth") or 0
            dd     = s.get("avg_cascade_dd_pips") or 0
            rec    = s.get("avg_recovery_trades")
            unrec  = s.get("pct_unrecovered") or 0
            trigger = s.get("throttle_trigger_at_N")
            rec_str = f"{rec:.1f}" if rec is not None else "N/A"
            trig_str = f"N={trigger}" if trigger else "none"

            lines.append(
                f"  {key:<30} {n:>5} {nc:>5} {ad:>7.1f} {md:>7} "
                f"{dd:>9.1f} {rec_str:>7} {unrec:>8.1f}% {trig_str:>8}"
            )

        # No-cascade combinations
        no_cas = [s for s in summaries if s.get("n_cascades", 0) == 0]
        if no_cas:
            lines += ["", f"  {len(no_cas)} combination(s) had zero multi-loss cascades (robust series)."]

        # Conditional probability heat table
        if cond_rows:
            lines += [
                "",
                "  CONDITIONAL LOSS PROBABILITY  P(loss | N consecutive losses)",
                "  (> 0.65 = elevated risk, recommended throttle point)",
                "",
            ]

            # Aggregate across all combinations
            df_cond = pd.DataFrame(cond_rows)
            agg = df_cond.groupby("streak_length")["p_loss"].agg(["mean", "min", "max"])
            header = f"  {'N':<5} {'Mean P(loss)':>14} {'Min':>8} {'Max':>8}  {'Heat'}"
            lines.append(header)
            lines.append("  " + "-" * 60)
            for streak, row in agg.iterrows():
                bar  = ">" * int(row["mean"] * 20)
                flag = " ** ELEVATED **" if row["mean"] > 0.65 else ""
                lines.append(
                    f"  {int(streak):<5} {row['mean']:>14.4f} {row['min']:>8.4f} {row['max']:>8.4f}"
                    f"  {bar:<20}{flag}"
                )

            # Aggregate recommendation
            elevated = [int(s) for s in agg.index if agg.loc[s, "mean"] > 0.65]
            if elevated:
                lines += [
                    "",
                    f"  KILL-SWITCH RECOMMENDATION: pause trading after {min(elevated)} consecutive losses.",
                    f"  Conditional loss probability exceeds 65% at N={min(elevated)}.",
                ]
            else:
                lines += [
                    "",
                    "  No elevated conditional loss probability detected.",
                    "  Losses appear independent — cascade risk is low.",
                ]

        lines.append("\n" + "=" * 72)
        return "\n".join(lines)
