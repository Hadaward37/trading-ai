"""
Regime Analytics -- analise de performance por regime para combinacoes passantes.

Pergunta central: "qual hipotese mantem estabilidade estrutural em qual regime?"

Inputs : batch_summary.json de qualquer run do batch_runner
Outputs: per_regime_metrics.csv, stability_ranking.csv,
         behavioral_market_map.csv, regime_analytics_report.txt
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────

ALL_REGIMES = ["TRENDING", "MEAN_REVERTING", "HIGH_VOLATILITY", "LOW_VOLATILITY"]
PASS_VERDICTS = {"PASS", "OOS_PASS"}
MIN_REGIME_TRADES = 10   # minimum trades to consider a regime reading reliable

OUTPUT_DIR = Path("research/results/regime_analytics")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_all_results(results_root: Optional[Path] = None) -> List[Dict]:
    """
    Load all rows from results_full.json files (primary) or batch_summary.json (fallback).
    Deduplicates by (asset, hypothesis) keeping the latest run.
    """
    if results_root is None:
        results_root = Path("research/results/batch_runs")

    all_rows: List[Dict] = []
    seen: Dict[Tuple, str] = {}  # (asset, hyp) -> run_timestamp

    # Primary: results_full.json (has regime_detail)
    for full_path in sorted(results_root.glob("*/results_full.json")):
        try:
            ts = full_path.parent.name   # timestamp is the directory name
            rows = json.loads(full_path.read_text(encoding="utf-8"))
            for row in rows:
                if not row.get("asset") or not row.get("hypothesis"):
                    continue
                key = (row["asset"], row["hypothesis"])
                if key not in seen or ts > seen[key]:
                    seen[key] = ts
                    all_rows.append(row)
        except Exception:
            continue

    # Fallback: results_matrix.csv (older runs without regime_detail)
    if not all_rows:
        for csv_path in sorted(results_root.glob("*/results_matrix.csv")):
            try:
                ts = csv_path.parent.name
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    rdict = row.to_dict()
                    if not rdict.get("asset") or not rdict.get("hypothesis"):
                        continue
                    key = (rdict["asset"], rdict["hypothesis"])
                    if key not in seen or ts > seen[key]:
                        seen[key] = ts
                        all_rows.append(rdict)
            except Exception:
                continue

    return all_rows


# ── Per-regime metrics extraction ──────────────────────────────────────────────

def extract_per_regime_rows(all_rows: List[Dict]) -> List[Dict]:
    """
    Flatten regime_detail from each result row into one record per
    (asset, hypothesis, period, regime).
    """
    records = []
    for row in all_rows:
        asset   = row.get("asset", "")
        hyp     = row.get("hypothesis", "")
        group   = row.get("group", "")
        verdict = row.get("oos_verdict", "")
        rob     = float(row.get("robustness", 0.0))
        pf_deg  = float(row.get("pf_degradation", 0.0))
        category = row.get("category", "")

        detail = row.get("regime_detail", {})
        if not detail:
            continue

        for period in ("test", "oos"):
            period_detail = detail.get(period, {})
            for regime, metrics in period_detail.items():
                n = int(metrics.get("n_trades", 0))
                if n == 0:
                    continue
                records.append({
                    "asset":       asset,
                    "category":    category,
                    "hypothesis":  hyp,
                    "group":       group,
                    "oos_verdict": verdict,
                    "robustness":  rob,
                    "pf_degradation": pf_deg,
                    "period":      period,
                    "regime":      regime,
                    "n_trades":    n,
                    "pf":          float(metrics.get("profit_factor", 0.0)),
                    "sharpe":      float(metrics.get("sharpe", 0.0)),
                    "win_rate":    float(metrics.get("win_rate", 0.0)),
                    "expectancy":  float(metrics.get("expectancy_pips", 0.0)),
                    "max_dd":      float(metrics.get("max_drawdown_pips", 0.0)),
                })
    return records


# ── Stability scoring ──────────────────────────────────────────────────────────

def compute_regime_stability(regime_rows: List[Dict]) -> pd.DataFrame:
    """
    Stability score per hypothesis per period:
      - For each hypothesis, collect PF across regimes (where N >= MIN_REGIME_TRADES)
      - Stability = 1 - CV(PF across regimes)   [CV = std/|mean|]
      - Higher = more consistent across market conditions

    Returns DataFrame indexed by (hypothesis, period) with stability metrics.
    """
    if not regime_rows:
        return pd.DataFrame()

    df = pd.DataFrame(regime_rows)
    records = []

    for (hyp, grp, period), sub in df.groupby(["hypothesis", "group", "period"]):
        reliable = sub[sub["n_trades"] >= MIN_REGIME_TRADES]
        if len(reliable) < 2:
            continue

        pf_vals   = reliable["pf"].values
        e_vals    = reliable["expectancy"].values
        pf_mean   = float(np.mean(pf_vals))
        pf_std    = float(np.std(pf_vals, ddof=1)) if len(pf_vals) > 1 else 0.0
        cv        = pf_std / max(abs(pf_mean), 1e-9)
        stability = max(0.0, min(1.0, 1.0 - cv))

        n_positive_regimes = int((pf_vals > 1.0).sum())
        n_regimes_covered  = len(reliable)

        best_regime  = reliable.loc[reliable["pf"].idxmax(), "regime"] if len(reliable) else "N/A"
        worst_regime = reliable.loc[reliable["pf"].idxmin(), "regime"] if len(reliable) else "N/A"

        records.append({
            "hypothesis":          hyp,
            "group":               grp,
            "period":              period,
            "stability_score":     round(stability, 4),
            "pf_mean":             round(pf_mean, 3),
            "pf_std":              round(pf_std, 3),
            "pf_cv":               round(cv, 4),
            "n_regimes_covered":   n_regimes_covered,
            "n_positive_regimes":  n_positive_regimes,
            "best_regime":         best_regime,
            "worst_regime":        worst_regime,
            "e_mean":              round(float(np.mean(e_vals)), 2),
        })

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records)
    result = result.sort_values("stability_score", ascending=False)
    return result


# ── OOS degradation by regime ──────────────────────────────────────────────────

def compute_oos_degradation_by_regime(regime_rows: List[Dict]) -> pd.DataFrame:
    """
    For each (hypothesis, asset, regime), compare test PF vs OOS PF.
    Identifies which regimes are most prone to data-mining degradation.
    """
    if not regime_rows:
        return pd.DataFrame()

    df = pd.DataFrame(regime_rows)
    records = []

    for (hyp, asset, regime), sub in df.groupby(["hypothesis", "asset", "regime"]):
        test_row = sub[sub["period"] == "test"]
        oos_row  = sub[sub["period"] == "oos"]

        if test_row.empty or oos_row.empty:
            continue

        n_test = int(test_row["n_trades"].iloc[0])
        n_oos  = int(oos_row["n_trades"].iloc[0])
        if n_test < MIN_REGIME_TRADES or n_oos < MIN_REGIME_TRADES:
            continue

        pf_test = float(test_row["pf"].iloc[0])
        pf_oos  = float(oos_row["pf"].iloc[0])
        deg     = round((pf_oos - pf_test) / max(pf_test, 1e-9), 4)

        records.append({
            "hypothesis": hyp,
            "asset":      asset,
            "regime":     regime,
            "n_test":     n_test,
            "n_oos":      n_oos,
            "pf_test":    round(pf_test, 3),
            "pf_oos":     round(pf_oos, 3),
            "degradation":deg,
            "stable":     abs(deg) < 0.15,   # within 15% = stable
        })

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).sort_values("degradation")


# ── Behavioral market map ──────────────────────────────────────────────────────

def compute_behavioral_market_map(regime_rows: List[Dict]) -> pd.DataFrame:
    """
    For each (asset, regime), find which strategy GROUP has the highest average PF.
    Produces an asset x regime matrix of "dominant strategy type".

    Also records the average PF per group and the best single hypothesis.
    """
    if not regime_rows:
        return pd.DataFrame()

    df = pd.DataFrame(regime_rows)
    test_df = df[df["period"] == "test"].copy()
    reliable = test_df[test_df["n_trades"] >= MIN_REGIME_TRADES]

    records = []
    for (asset, regime), sub in reliable.groupby(["asset", "regime"]):
        group_pf = sub.groupby("group")["pf"].mean()
        if group_pf.empty:
            continue

        best_group    = group_pf.idxmax()
        best_group_pf = round(float(group_pf.max()), 3)
        best_hyp_row  = sub.loc[sub["pf"].idxmax()]

        all_group_pf = {g: round(float(v), 3) for g, v in group_pf.items()}

        records.append({
            "asset":           asset,
            "regime":          regime,
            "best_group":      best_group,
            "best_group_pf":   best_group_pf,
            "best_hypothesis": best_hyp_row["hypothesis"],
            "best_hyp_pf":     round(float(best_hyp_row["pf"]), 3),
            "n_total_trades":  int(sub["n_trades"].sum()),
            **{f"pf_{g.lower()[:8]}": all_group_pf.get(g, float("nan"))
               for g in ["mean_reversion", "trend_follow", "volatility"]},
        })

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).sort_values(["asset", "regime"])


# ── Report generation ──────────────────────────────────────────────────────────

def _format_regime_stability_report(
    stability_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    behavioral_map: pd.DataFrame,
    all_rows: List[Dict],
) -> str:
    lines = [
        "=" * 72,
        "  REGIME ANALYTICS REPORT",
        f"  Generated: {datetime.utcnow().isoformat()}",
        f"  Total combinations loaded: {len(all_rows)}",
        "=" * 72,
    ]

    # ── Stability ranking ──────────────────────────────────────────────────────
    lines += ["", "  HYPOTHESIS STABILITY RANKING (test period, higher = more consistent across regimes)", ""]
    lines.append(
        f"  {'Hypothesis':<18} {'Group':<14} {'Stability':>10} {'PF mean':>8} "
        f"{'PF std':>8} {'Regimes':>8} {'Best regime':<18} {'Worst regime':<18}"
    )
    lines.append("  " + "-" * 100)

    test_stab = stability_df[stability_df["period"] == "test"] if not stability_df.empty else pd.DataFrame()
    for _, row in test_stab.iterrows():
        lines.append(
            f"  {row['hypothesis']:<18} {row['group']:<14} {row['stability_score']:>10.4f} "
            f"{row['pf_mean']:>8.3f} {row['pf_std']:>8.3f} "
            f"{row['n_regimes_covered']:>8} {row['best_regime']:<18} {row['worst_regime']:<18}"
        )

    # ── OOS degradation by regime ──────────────────────────────────────────────
    if not degradation_df.empty:
        lines += ["", "  OOS DEGRADATION BY REGIME (worst first)", ""]
        lines.append(
            f"  {'Hypothesis':<18} {'Asset':<8} {'Regime':<20} "
            f"{'PF Test':>8} {'PF OOS':>8} {'Degr%':>8} {'Stable':>8}"
        )
        lines.append("  " + "-" * 80)
        for _, row in degradation_df.sort_values("degradation").head(20).iterrows():
            sign = "+" if row["degradation"] >= 0 else ""
            stable = "YES" if row["stable"] else "NO"
            lines.append(
                f"  {row['hypothesis']:<18} {row['asset']:<8} {row['regime']:<20} "
                f"{row['pf_test']:>8.3f} {row['pf_oos']:>8.3f} "
                f"{sign}{row['degradation']*100:>7.1f}% {stable:>8}"
            )

    # ── Behavioral market map ──────────────────────────────────────────────────
    if not behavioral_map.empty:
        lines += ["", "  BEHAVIORAL MARKET MAP (which strategy dominates per asset x regime)", ""]
        lines.append(f"  {'Asset':<8} {'Regime':<22} {'Best Group':<16} {'PF':>6} {'Best Hypothesis':<20}")
        lines.append("  " + "-" * 76)

        for asset in sorted(behavioral_map["asset"].unique()):
            sub = behavioral_map[behavioral_map["asset"] == asset].sort_values("regime")
            for _, row in sub.iterrows():
                lines.append(
                    f"  {row['asset']:<8} {row['regime']:<22} {row['best_group']:<16} "
                    f"{row['best_group_pf']:>6.3f} {row['best_hypothesis']:<20}"
                )
            lines.append("")

    # ── Summary answer ─────────────────────────────────────────────────────────
    lines += [
        "=" * 72,
        "  SUMMARY: HIPOTESES COM ESTABILIDADE ESTRUTURAL",
        "=" * 72,
    ]

    if not test_stab.empty:
        top3 = test_stab.head(3)
        for _, row in top3.iterrows():
            lines.append(
                f"  {row['hypothesis']}: stability={row['stability_score']:.3f} | "
                f"best_in={row['best_regime']} (PF avg {row['pf_mean']:.3f}) | "
                f"fragile_in={row['worst_regime']}"
            )
    lines.append("=" * 72)

    return "\n".join(lines)


# ── Main analytics runner ──────────────────────────────────────────────────────

class RegimeAnalytics:
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, results_root: Optional[Path] = None) -> Dict[str, Any]:
        print(f"[RegimeAnalytics] Loading batch results...")
        all_rows = load_all_results(results_root)
        print(f"  {len(all_rows)} total combinations loaded")

        if not all_rows:
            print("  No results found.")
            return {}

        regime_rows = extract_per_regime_rows(all_rows)
        print(f"  {len(regime_rows)} per-regime records extracted")

        # ── Compute analytics ──────────────────────────────────────────────────
        stability_df   = compute_regime_stability(regime_rows)
        degradation_df = compute_oos_degradation_by_regime(regime_rows)
        behavioral_map = compute_behavioral_market_map(regime_rows)

        # ── Save outputs ───────────────────────────────────────────────────────
        pd.DataFrame(regime_rows).to_csv(
            self.output_dir / "per_regime_metrics.csv", index=False, encoding="utf-8"
        )
        if not stability_df.empty:
            stability_df.to_csv(
                self.output_dir / "stability_ranking.csv", index=False, encoding="utf-8"
            )
        if not degradation_df.empty:
            degradation_df.to_csv(
                self.output_dir / "oos_degradation_by_regime.csv", index=False, encoding="utf-8"
            )
        if not behavioral_map.empty:
            behavioral_map.to_csv(
                self.output_dir / "behavioral_market_map.csv", index=False, encoding="utf-8"
            )

        report = _format_regime_stability_report(
            stability_df, degradation_df, behavioral_map, all_rows
        )
        (self.output_dir / "regime_analytics_report.txt").write_text(
            report, encoding="utf-8"
        )
        print(report)

        return {
            "n_rows":       len(all_rows),
            "n_regime_rows": len(regime_rows),
            "stability_df":  stability_df,
            "degradation_df": degradation_df,
            "behavioral_map": behavioral_map,
        }


def run_regime_analytics(results_root: Optional[Path] = None) -> Dict[str, Any]:
    return RegimeAnalytics().run(results_root)


if __name__ == "__main__":
    run_regime_analytics()
