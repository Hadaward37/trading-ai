"""
Edge Decay Analysis.

Calcula para cada combinacao ativo x hipotese:
  - Rolling PF (janelas 30, 60, 90 trades)
  - Rolling Sharpe e Rolling Win Rate
  - Edge half-life: numero de trades ate PF decair 50% do valor inicial
  - Decay type: STABLE | IMPROVING | DECAY | SATURATION | STRUCTURAL_BREAK
  - Rolling Robustness proxy (PF consistency)

Causalidade: rolling calculado apenas com trades passados (sem look-ahead).
Todos os indices sao baseados em posicao na sequencia temporal de trades.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._shared import BEHAVIORAL_MAP_DIR, pf_from_pnls, trades_to_df

OUTPUT_DIR = BEHAVIORAL_MAP_DIR

WINDOWS = [30, 60, 90]
HALF_LIFE_THRESHOLD = 0.5    # PF decays to 50% of initial value
MIN_TRADES_FOR_DECAY = 60    # need at least this many trades to fit decay
BREAK_SIGMA = 2.0            # structural break: deviation > 2 sigma


# ── Rolling metrics computation ────────────────────────────────────────────────

def _rolling_pf(pnls: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling Profit Factor over past `window` trades.
    Strictly causal: at index i, uses trades [max(0, i-window+1) .. i].
    """
    n = len(pnls)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        result[i] = pf_from_pnls(pnls[max(0, i - window + 1):i + 1])
    return result


def _rolling_sharpe(pnls: np.ndarray, window: int) -> np.ndarray:
    """Rolling Sharpe (E/std, simplified). Causal."""
    n = len(pnls)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        w = pnls[max(0, i - window + 1):i + 1]
        std = w.std()
        result[i] = (w.mean() / std) if std > 1e-9 else 0.0
    return result


def _rolling_winrate(wins: np.ndarray, window: int) -> np.ndarray:
    """Rolling win rate. Causal."""
    n = len(wins)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        w = wins[max(0, i - window + 1):i + 1]
        result[i] = float(w.mean())
    return result


# ── Edge half-life ─────────────────────────────────────────────────────────────

def _compute_half_life(rolling_pf: np.ndarray, initial_pf: float) -> float:
    """
    Number of trades until rolling_pf drops to initial_pf * HALF_LIFE_THRESHOLD.
    Returns inf if never drops, or nan if insufficient data.
    """
    target = initial_pf * HALF_LIFE_THRESHOLD
    valid = ~np.isnan(rolling_pf)
    if valid.sum() < 5:
        return float("nan")
    first_valid = np.where(valid)[0][0]
    for i in range(first_valid, len(rolling_pf)):
        if not np.isnan(rolling_pf[i]) and rolling_pf[i] <= target:
            return float(i - first_valid)
    return float("inf")


# ── Decay type classification ──────────────────────────────────────────────────

def _classify_decay(rolling_pf: np.ndarray) -> str:
    """
    Classify the shape of the rolling PF curve.

    STABLE            — low variance, no clear trend
    IMPROVING         — positive slope, PF growing
    DECAY             — monotonically decreasing trend
    SATURATION        — drops and stabilises below 1.0
    STRUCTURAL_BREAK  — sudden step-change detected
    INSUFFICIENT_DATA — too few points
    """
    valid = rolling_pf[~np.isnan(rolling_pf)]
    if len(valid) < 10:
        return "INSUFFICIENT_DATA"

    # Detect structural break: look for a point where value shifts > BREAK_SIGMA * std
    global_std = float(valid.std())
    if global_std > 1e-9:
        diffs = np.abs(np.diff(valid))
        if diffs.max() > BREAK_SIGMA * global_std * 3:
            return "STRUCTURAL_BREAK"

    # Fit linear trend
    x = np.arange(len(valid), dtype=float)
    slope = float(np.polyfit(x, valid, 1)[0])

    pf_early = float(valid[:len(valid)//3].mean())
    pf_late  = float(valid[-len(valid)//3:].mean())
    last_val = float(valid[-1])

    if abs(slope) < 0.002 and valid.std() / max(abs(valid.mean()), 1e-9) < 0.15:
        return "STABLE"
    if slope > 0.003:
        return "IMPROVING"
    if slope < -0.003 and last_val < 1.0:
        # Further distinguish: saturation (flat end) vs ongoing decay
        if abs(valid[-5:].mean() - valid[-15:-5].mean()) < 0.05 and last_val < 1.0:
            return "SATURATION"
        return "DECAY"
    if slope < -0.001:
        return "DECAY"
    return "STABLE"


# ── Public interface ───────────────────────────────────────────────────────────

class EdgeDecayAnalyzer:
    """
    Edge decay analysis: rolling PF, half-life, and decay classification.
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
        """Full edge decay analysis for one combination."""
        df = trades_to_df(trades)
        if df.empty or len(df) < WINDOWS[0]:
            return {
                "key": f"{asset_key}_{hyp_name}",
                "asset": asset_key,
                "hypothesis": hyp_name,
                "n_trades": len(df),
                "decay_type": "INSUFFICIENT_DATA",
                "note": f"Need >= {WINDOWS[0]} trades, got {len(df)}",
            }

        pnls = df["pnl_pips"].values
        wins = df["win"].values
        n    = len(pnls)
        key  = f"{asset_key}_{hyp_name}"

        # Baseline PF (first 20% of trades — truly in-sample baseline)
        baseline_n  = max(10, n // 5)
        baseline_pf = pf_from_pnls(pnls[:baseline_n])

        # Rolling metrics for each window
        rolling_results: Dict[int, Dict] = {}
        for w in WINDOWS:
            rpf    = _rolling_pf(pnls, w)
            rshp   = _rolling_sharpe(pnls, w)
            rwr    = _rolling_winrate(wins, w)
            hl     = _compute_half_life(rpf, baseline_pf)
            decay  = _classify_decay(rpf)

            rolling_results[w] = {
                "rolling_pf":    rpf,
                "rolling_sharpe": rshp,
                "rolling_wr":    rwr,
                "half_life_trades": hl,
                "decay_type":    decay,
            }

        # Final PF (last 30% of trades)
        final_n  = max(10, n * 3 // 10)
        final_pf = pf_from_pnls(pnls[-final_n:])

        # Decay magnitude: baseline vs final
        decay_magnitude = round((final_pf - baseline_pf) / max(baseline_pf, 1e-9), 4)

        # Consensus decay type (majority vote across windows)
        decay_types = [rolling_results[w]["decay_type"] for w in WINDOWS]
        consensus = max(set(decay_types), key=decay_types.count)

        return {
            "key":              key,
            "asset":            asset_key,
            "hypothesis":       hyp_name,
            "n_trades":         n,
            "baseline_pf":      round(baseline_pf, 3),
            "final_pf":         round(final_pf, 3),
            "decay_magnitude":  decay_magnitude,
            "decay_type":       consensus,
            "half_life_30":     round(rolling_results[30]["half_life_trades"], 1) if not np.isnan(rolling_results[30]["half_life_trades"]) else None,
            "half_life_60":     round(rolling_results[60]["half_life_trades"], 1) if not np.isnan(rolling_results[60]["half_life_trades"]) else None,
            "half_life_90":     round(rolling_results[90]["half_life_trades"], 1) if not np.isnan(rolling_results[90]["half_life_trades"]) else None,
            "decay_type_30":    rolling_results[30]["decay_type"],
            "decay_type_60":    rolling_results[60]["decay_type"],
            "decay_type_90":    rolling_results[90]["decay_type"],
            "_rolling":         rolling_results,  # internal, excluded from CSV
        }

    def run(self, combinations: List[Tuple]) -> Dict:
        """Analyze all combinations and save outputs."""
        summaries  = []
        all_series = []

        for asset_key, hyp_name, _, trades in combinations:
            if not trades:
                continue
            res = self.analyze_combination(trades, asset_key, hyp_name)
            # Extract series data for CSV
            if "_rolling" in res:
                for w in WINDOWS:
                    r = res["_rolling"][w]
                    n = len(r["rolling_pf"])
                    for i in range(n):
                        if not np.isnan(r["rolling_pf"][i]):
                            all_series.append({
                                "key":        res["key"],
                                "asset":      asset_key,
                                "hypothesis": hyp_name,
                                "window":     w,
                                "trade_idx":  i,
                                "rolling_pf": round(float(r["rolling_pf"][i]), 4),
                                "rolling_sharpe": round(float(r["rolling_sharpe"][i]), 4) if not np.isnan(r["rolling_sharpe"][i]) else None,
                                "rolling_wr": round(float(r["rolling_wr"][i]), 4) if not np.isnan(r["rolling_wr"][i]) else None,
                            })
                del res["_rolling"]
            summaries.append(res)

        # Save
        if summaries:
            pd.DataFrame(summaries).to_csv(
                self.output_dir / "edge_decay_summary.csv", index=False, encoding="utf-8"
            )
        if all_series:
            pd.DataFrame(all_series).to_csv(
                self.output_dir / "rolling_pf_series.csv", index=False, encoding="utf-8"
            )

        report = self._format_report(summaries)
        (self.output_dir / "edge_decay_report.txt").write_text(report, encoding="utf-8")
        print(report)

        return {"summaries": summaries, "n_series_rows": len(all_series)}

    @staticmethod
    def _format_report(summaries: List[Dict]) -> str:
        lines = [
            "=" * 72,
            "  EDGE DECAY ANALYSIS",
            f"  Generated: {datetime.utcnow().isoformat()}",
            f"  Combinations: {len(summaries)}",
            "=" * 72,
            "",
            f"  {'Combination':<30} {'N':>5} {'Base PF':>8} {'Final PF':>9} {'Decay%':>8} {'Type':<22} {'HL-30':>7}",
            "  " + "-" * 96,
        ]

        for s in sorted(summaries, key=lambda x: x.get("decay_magnitude", 0)):
            key    = s.get("key", "?")
            n      = s.get("n_trades", 0)
            bpf    = s.get("baseline_pf", 0) or 0
            fpf    = s.get("final_pf", 0) or 0
            dm     = s.get("decay_magnitude", 0) or 0
            dtype  = s.get("decay_type", "?")
            hl30   = s.get("half_life_30")
            hl_str = f"{hl30:.0f}" if hl30 is not None and hl30 != float("inf") else ("inf" if hl30 == float("inf") else "N/A")

            lines.append(
                f"  {key:<30} {n:>5} {bpf:>8.3f} {fpf:>9.3f} "
                f"{dm*100:>+7.1f}% {dtype:<22} {hl_str:>7}"
            )

        # Summary
        types = [s.get("decay_type", "") for s in summaries if s.get("decay_type") not in ("INSUFFICIENT_DATA", None)]
        lines += ["", "  DECAY TYPE DISTRIBUTION:"]
        for dtype in ["STABLE", "IMPROVING", "DECAY", "SATURATION", "STRUCTURAL_BREAK"]:
            count = types.count(dtype)
            pct   = count / len(types) * 100 if types else 0
            bar   = "=" * int(pct / 5)
            lines.append(f"    {dtype:<22} {count:>3}  [{bar:<20}] {pct:5.1f}%")

        decay_combos = [s for s in summaries if s.get("decay_type") in ("DECAY", "SATURATION")]
        if decay_combos:
            lines += ["", "  WARNING: The following combinations show edge decay:", ""]
            for s in decay_combos:
                hl = s.get("half_life_30")
                hl_str = f"HL={hl:.0f} trades" if hl and hl != float("inf") else "HL=never"
                lines.append(f"    {s['key']}: {s['decay_type']} | {hl_str} | decay={s.get('decay_magnitude',0)*100:+.1f}%")

        lines.append("\n" + "=" * 72)
        return "\n".join(lines)
