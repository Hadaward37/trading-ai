"""
Behavioral Map Orchestrator.

Roda os 4 modulos de analise profunda sobre as combinacoes OOS_PASS/PASS:
  1. regime_transition  -- "morrem na transicao ou dentro do regime?"
  2. edge_decay         -- rolling PF, half-life, decay classification
  3. regime_persistence -- duracao de regime, Markov Matrix, steady-state
  4. failure_cascade    -- cascades de perda, P(loss|N losses), kill-switch threshold

Usage:
    python -X utf8 -m research.fractal_lab.research.run_behavioral_map
    python -X utf8 -m research.fractal_lab.research.run_behavioral_map --assets EURUSD BTC
    python -X utf8 -m research.fractal_lab.research.run_behavioral_map --n-top 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from research.fractal_lab.analytics._shared import (
    BEHAVIORAL_MAP_DIR,
    TRAIN_PERIOD, TEST_PERIOD, OOS_PERIOD,
    load_batch_rows, select_analysis_targets,
    load_asset_data, prepare_df, run_backtest, trades_to_df,
)
from research.fractal_lab.analytics.regime_transition  import RegimeTransitionAnalyzer
from research.fractal_lab.analytics.edge_decay         import EdgeDecayAnalyzer
from research.fractal_lab.analytics.regime_persistence import RegimePersistenceAnalyzer
from research.fractal_lab.analytics.failure_cascade    import FailureCascadeAnalyzer

OUTPUT_DIR = BEHAVIORAL_MAP_DIR


# ── Data acquisition ───────────────────────────────────────────────────────────

def build_combination_data(
    targets: List[Tuple[str, str]],
    verbose: bool = True,
) -> Tuple[List[Tuple], List[Tuple]]:
    """
    For each target (asset_key, hyp_name):
      1. Load OHLCV data
      2. Prepare (features + regimes)
      3. Run backtest (full period: TRAIN_PERIOD start -> OOS_PERIOD end)

    Returns:
      combinations: [(asset_key, hyp_name, df_prepared, trades)] for analytics modules
      asset_dfs:    [(asset_key, df_prepared, trades)] deduplicated per asset
    """
    combinations: List[Tuple] = []
    asset_dfs:    List[Tuple] = []
    asset_df_cache: Dict[str, pd.DataFrame] = {}

    for asset_key, hyp_name in targets:
        # Load and prepare data (cached per asset)
        if asset_key not in asset_df_cache:
            if verbose:
                print(f"  [data] Loading {asset_key}...", end=" ", flush=True)
            raw = load_asset_data(asset_key)
            if raw.empty:
                if verbose:
                    print("EMPTY — skipping")
                asset_df_cache[asset_key] = pd.DataFrame()
                continue
            df_prep = prepare_df(raw)
            asset_df_cache[asset_key] = df_prep
            if verbose:
                print(f"{len(df_prep)} bars OK")
        else:
            df_prep = asset_df_cache[asset_key]

        if df_prep.empty:
            continue

        # Run backtest
        if verbose:
            print(f"  [bt]   {asset_key} x {hyp_name}...", end=" ", flush=True)
        t0 = time.time()
        trades = run_backtest(df_prep, asset_key, hyp_name)
        elapsed = time.time() - t0
        if verbose:
            print(f"{len(trades)} trades ({elapsed:.1f}s)")

        if len(trades) < 15:
            if verbose:
                print(f"         => too few trades, skipping")
            continue

        combinations.append((asset_key, hyp_name, df_prep, trades))

    # Build asset_dfs: one entry per unique asset (using all its trades merged)
    seen_assets = set()
    for asset_key, hyp_name, df_prep, trades in combinations:
        if asset_key not in seen_assets:
            seen_assets.add(asset_key)
            # Collect trades for all hypotheses on this asset
            all_trades = [
                t for ak, _, _, tt in combinations
                if ak == asset_key
                for t in tt
            ]
            asset_dfs.append((asset_key, df_prep, all_trades))

    return combinations, asset_dfs


# ── Consolidated JSON ──────────────────────────────────────────────────────────

def _make_json_safe(obj: Any) -> Any:
    """Recursively convert numpy types to Python natives for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_safe(x) for x in obj]
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict("records")
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


# ── Behavioral insights synthesis ─────────────────────────────────────────────

def synthesize_insights(
    transition_result: Dict,
    decay_result: Dict,
    persistence_result: Dict,
    cascade_result: Dict,
) -> List[str]:
    """
    Generate actionable behavioral insights from the 4 analysis modules.
    Returns list of insight strings.
    """
    insights = []

    # Transition insights
    zone_stats = transition_result.get("zone_stats", [])
    if zone_stats:
        die_trans = sum(1 for z in zone_stats if z.get("verdict") == "DIE_AT_TRANSITIONS")
        die_within = sum(1 for z in zone_stats if z.get("verdict") == "DIE_WITHIN_REGIMES")
        resilient  = sum(1 for z in zone_stats if z.get("verdict") == "TRANSITION_RESILIENT")
        total = len(zone_stats)

        if total > 0:
            if die_trans / total > 0.4:
                insights.append(
                    f"TRANSITION_RISK: {die_trans}/{total} combinations lose edge at regime transitions. "
                    "Consider a {TRANSITION_WINDOW_BARS}-bar entry moratorium after any regime change."
                )
            elif resilient / total > 0.5:
                insights.append(
                    f"TRANSITION_ROBUST: {resilient}/{total} combinations maintain edge through transitions. "
                    "Strategy quality is regime-agnostic — focus on regime selection, not transition timing."
                )

    # Decay insights
    decay_summaries = decay_result.get("summaries", [])
    if decay_summaries:
        decaying = [s for s in decay_summaries if s.get("decay_type") in ("DECAY", "SATURATION")]
        stable   = [s for s in decay_summaries if s.get("decay_type") == "STABLE"]
        improving = [s for s in decay_summaries if s.get("decay_type") == "IMPROVING"]
        if decaying:
            names = ", ".join(s["key"] for s in decaying[:3])
            insights.append(
                f"EDGE_DECAY: {len(decaying)} combinations show decaying edge: {names}. "
                "Likely regime saturation or delayed mean-reversion. Monitor rolling PF monthly."
            )
        if improving:
            names = ", ".join(s["key"] for s in improving[:3])
            insights.append(
                f"EDGE_IMPROVING: {len(improving)} combinations show improving edge over time: {names}. "
                "This could indicate the strategy adapts to market structure."
            )
        if stable:
            insights.append(
                f"EDGE_STABLE: {len(stable)} combinations maintain consistent edge (low PF variance). "
                "These are the most reliable for capital allocation."
            )

    # Persistence insights
    steady = persistence_result.get("aggregate_steady", {})
    if steady:
        dominant = max(steady, key=steady.get) if steady else None
        if dominant:
            insights.append(
                f"REGIME_DISTRIBUTION: {dominant} is the dominant long-run regime "
                f"({steady.get(dominant, 0):.1%} of time in steady state). "
                "Strategy allocation should weight toward hypotheses that excel in this regime."
            )

    # Markov insights
    markov = persistence_result.get("aggregate_markov", pd.DataFrame())
    if isinstance(markov, pd.DataFrame) and not markov.empty:
        # Find highest exit probability (most volatile transition)
        for r in markov.index:
            diag = markov.loc[r, r] if r in markov.columns else 0
            if diag == 0:  # no self-transitions by construction
                max_exit_col = markov.loc[r].idxmax()
                max_prob = markov.loc[r].max()
                if max_prob > 0.5:
                    insights.append(
                        f"MARKOV_PATH: {r} most likely transitions to {max_exit_col} "
                        f"(P={max_prob:.2f}). Pre-position strategies for this succession."
                    )

    # Cascade insights
    cascade_summaries = cascade_result.get("summaries", [])
    if cascade_summaries:
        trigger_counts = [
            s.get("throttle_trigger_at_N") for s in cascade_summaries
            if s.get("throttle_trigger_at_N") is not None
        ]
        if trigger_counts:
            median_trigger = int(np.median(trigger_counts))
            insights.append(
                f"CASCADE_RISK: Conditional loss probability exceeds 65% after "
                f"{median_trigger} consecutive losses (median across combinations). "
                f"Recommended kill-switch: pause after N={median_trigger} consecutive losses."
            )
        no_cascade = [s for s in cascade_summaries if s.get("n_cascades", 0) == 0]
        if no_cascade:
            names = ", ".join(s["key"] for s in no_cascade[:3])
            insights.append(
                f"CASCADE_RESISTANT: {len(no_cascade)} combinations had no multi-loss cascades: {names}. "
                "These are candidates for higher position sizing."
            )

    return insights


# ── Main runner ────────────────────────────────────────────────────────────────

def run_behavioral_map(
    assets: Optional[List[str]] = None,
    n_top: int = 12,
    output_dir: Optional[Path] = None,
) -> Dict:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 66}")
    print(f"  FRACTAL LAB BEHAVIORAL MAP -- {ts}")
    print(f"  Output: {out}")
    print(f"{'=' * 66}\n")

    # ── Select targets ─────────────────────────────────────────────────────────
    print("[1/6] Loading batch results and selecting targets...")
    rows    = load_batch_rows()
    if assets:
        rows = [r for r in rows if r.get("asset") in assets]
    targets = select_analysis_targets(rows, n_top=n_top)
    print(f"  {len(targets)} combinations selected:")
    for a, h in targets:
        print(f"    {a} x {h}")

    if not targets:
        print("  No passing combinations found. Run batch_runner first.")
        return {}

    # ── Build data ─────────────────────────────────────────────────────────────
    print("\n[2/6] Loading data and running backtests...")
    combinations, asset_dfs = build_combination_data(targets, verbose=True)
    print(f"  {len(combinations)} combinations ready for analysis")

    if not combinations:
        print("  No valid combinations. Exiting.")
        return {}

    # ── Run analytics ──────────────────────────────────────────────────────────
    print("\n[3/6] Regime Transition Analysis...")
    trans_analyzer = RegimeTransitionAnalyzer(output_dir=out)
    transition_result = trans_analyzer.run(combinations)

    print("\n[4/6] Edge Decay Analysis...")
    decay_analyzer = EdgeDecayAnalyzer(output_dir=out)
    decay_result = decay_analyzer.run(combinations)

    print("\n[5/6] Regime Persistence Analysis...")
    persist_analyzer = RegimePersistenceAnalyzer(output_dir=out)
    persistence_result = persist_analyzer.run(asset_dfs)

    print("\n[6/6] Failure Cascade Analysis...")
    cascade_analyzer = FailureCascadeAnalyzer(output_dir=out)
    cascade_result = cascade_analyzer.run(combinations)

    # ── Synthesize insights ────────────────────────────────────────────────────
    insights = synthesize_insights(
        transition_result, decay_result, persistence_result, cascade_result
    )

    # ── Save consolidated JSON ─────────────────────────────────────────────────
    consolidated = {
        "run_timestamp": ts,
        "n_combinations": len(combinations),
        "targets": [{"asset": a, "hypothesis": h} for a, h in targets],
        "insights": insights,
        "regime_transition": {
            "zone_stats": transition_result.get("zone_stats", []),
            "per_pair":   transition_result.get("per_pair", [])[:20],
        },
        "edge_decay": {
            "summaries": decay_result.get("summaries", []),
        },
        "regime_persistence": {
            "aggregate_steady": persistence_result.get("aggregate_steady", {}),
        },
        "failure_cascade": {
            "summaries": cascade_result.get("summaries", []),
        },
    }
    (out / "behavioral_map_consolidated.json").write_text(
        json.dumps(_make_json_safe(consolidated), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    # ── Print synthesis ────────────────────────────────────────────────────────
    print(f"\n{'=' * 66}")
    print(f"  BEHAVIORAL INSIGHTS SYNTHESIS")
    print(f"{'=' * 66}")
    for i, insight in enumerate(insights, 1):
        print(f"\n  [{i}] {insight}")
    print(f"\n{'=' * 66}")
    print(f"  Output saved: {out}")
    print(f"{'=' * 66}\n")

    return consolidated


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Behavioral Map -- 4 deep analysis modules")
    parser.add_argument("--assets", nargs="+", help="Filter to these assets only")
    parser.add_argument("--n-top", type=int, default=12, help="Number of top combinations to analyze")
    parser.add_argument("--output", type=Path, default=None, help="Output directory override")
    args = parser.parse_args()

    run_behavioral_map(
        assets=args.assets,
        n_top=args.n_top,
        output_dir=args.output,
    )
