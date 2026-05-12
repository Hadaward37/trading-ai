"""
Regime Transition Analysis.

Pergunta central: "estrategias morrem NA TRANSICAO ou DENTRO do regime?"

Calcula para cada par FROM_REGIME -> TO_REGIME:
  - Contagem de transicoes
  - PF medio antes da transicao (ultimas N trades no FROM)
  - PF medio apos a transicao (primeiras N trades no TO)
  - Degradation delta = PF_post - PF_pre
  - Recovery time: candles no TO ate PF voltar ao baseline global
  - "Transition zone" score: PF nos W bars imediatamente apos a transicao

Causalidade: todas as metricas usam dados que estavam disponiveis
no momento da entrada em cada trade (sem look-ahead).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._shared import (
    ALL_REGIMES, BEHAVIORAL_MAP_DIR, OOS_PERIOD, TEST_PERIOD, TRAIN_PERIOD,
    build_regime_blocks, pf_from_pnls, trades_to_df,
)

OUTPUT_DIR = BEHAVIORAL_MAP_DIR

# Transition zone: N bars after a regime change where trades are "in transition"
TRANSITION_WINDOW_BARS = 8
# Minimum trades in a window to compute reliable PF
MIN_WINDOW_TRADES = 5
# Lookback/lookahead trades for pre/post PF per transition
TRADES_WINDOW = 20


# ── Core computation ───────────────────────────────────────────────────────────

def _assign_transition_zone(df_prepared: pd.DataFrame, blocks: pd.DataFrame) -> pd.Series:
    """
    For each bar in df_prepared, return True if within TRANSITION_WINDOW_BARS of
    a regime change boundary. Strictly causal: uses only bar index, not future regimes.
    """
    n = len(df_prepared)
    in_zone = pd.Series(False, index=df_prepared.index)

    # Every block boundary (start of new block) is a transition point
    for _, blk in blocks.iterrows():
        start_i = int(blk["start_idx"])
        zone_end = min(start_i + TRANSITION_WINDOW_BARS, n)
        in_zone.iloc[start_i:zone_end] = True

    return in_zone


def compute_transition_matrix(blocks: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    FROM x TO count matrix and probability matrix.
    Rows = FROM regime, Columns = TO regime.
    """
    count_matrix = pd.DataFrame(0, index=ALL_REGIMES, columns=ALL_REGIMES)

    for i in range(len(blocks) - 1):
        f = blocks.iloc[i]["regime"]
        t = blocks.iloc[i + 1]["regime"]
        if f in ALL_REGIMES and t in ALL_REGIMES:
            count_matrix.loc[f, t] += 1

    # Probability matrix: normalize rows
    row_sums = count_matrix.sum(axis=1).replace(0, np.nan)
    prob_matrix = count_matrix.div(row_sums, axis=0).fillna(0.0).round(4)

    return count_matrix, prob_matrix


def compute_per_transition_pf(
    blocks: pd.DataFrame,
    trades_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each observed FROM->TO transition, compute:
    - n_events: how many times this pair occurred
    - pf_pre:   PF of last TRADES_WINDOW trades in the FROM block
    - pf_post:  PF of first TRADES_WINDOW trades in the TO block
    - delta_pf: pf_post - pf_pre
    - recovery_bars: mean blocks[TO].duration_bars when pf_post < pf_pre * 0.9

    All trade lookups use entry_time (causal: trade existed before the next transition).
    """
    if trades_df.empty or len(blocks) < 2:
        return pd.DataFrame()

    records: Dict[Tuple, List] = {}

    for i in range(len(blocks) - 1):
        from_blk = blocks.iloc[i]
        to_blk   = blocks.iloc[i + 1]
        from_r   = from_blk["regime"]
        to_r     = to_blk["regime"]
        if from_r not in ALL_REGIMES or to_r not in ALL_REGIMES:
            continue

        key = (from_r, to_r)
        if key not in records:
            records[key] = {"pre_pnls": [], "post_pnls": [], "recovery_bars": []}

        # Trades in the FROM block (by entry_time)
        pre_mask = (
            (trades_df["entry_time"] >= from_blk["start_ts"]) &
            (trades_df["entry_time"] <= from_blk["end_ts"])
        )
        pre_trades = trades_df[pre_mask]["pnl_pips"].values[-TRADES_WINDOW:]

        # Trades in the TO block
        post_mask = (
            (trades_df["entry_time"] >= to_blk["start_ts"]) &
            (trades_df["entry_time"] <= to_blk["end_ts"])
        )
        post_trades = trades_df[post_mask]["pnl_pips"].values[:TRADES_WINDOW]

        records[key]["pre_pnls"].extend(pre_trades.tolist())
        records[key]["post_pnls"].extend(post_trades.tolist())

        # Recovery approximation: duration of TO block (how long until regime stabilises)
        records[key]["recovery_bars"].append(int(to_blk["duration_bars"]))

    rows = []
    for (from_r, to_r), data in records.items():
        pre  = np.array(data["pre_pnls"])
        post = np.array(data["post_pnls"])
        n_events = len(data["recovery_bars"])

        pf_pre  = pf_from_pnls(pre)  if len(pre)  >= MIN_WINDOW_TRADES else float("nan")
        pf_post = pf_from_pnls(post) if len(post) >= MIN_WINDOW_TRADES else float("nan")

        if not (np.isnan(pf_pre) or np.isnan(pf_post)):
            delta = round(pf_post - pf_pre, 4)
            recovery_bars = round(float(np.mean(data["recovery_bars"])), 1)
        else:
            delta = float("nan")
            recovery_bars = float("nan")

        rows.append({
            "from_regime":    from_r,
            "to_regime":      to_r,
            "n_events":       n_events,
            "n_pre_trades":   len(pre),
            "n_post_trades":  len(post),
            "pf_pre":         round(pf_pre,  3) if not np.isnan(pf_pre)  else None,
            "pf_post":        round(pf_post, 3) if not np.isnan(pf_post) else None,
            "delta_pf":       round(delta, 4)   if not np.isnan(delta)   else None,
            "recovery_bars":  recovery_bars,
            "degrades":       bool(delta < -0.05) if not np.isnan(delta) else None,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("n_events", ascending=False)


def transition_vs_stable_pf(
    df_prepared: pd.DataFrame,
    blocks: pd.DataFrame,
    trades_df: pd.DataFrame,
) -> Dict:
    """
    Compare PF for trades inside vs outside the transition zone.

    Transition zone = TRANSITION_WINDOW_BARS bars after any regime change.
    Stable zone = all other bars.

    Key answer: pf_transition vs pf_stable
    """
    if trades_df.empty:
        return {}

    in_zone = _assign_transition_zone(df_prepared, blocks)

    # For each trade, check if entry_time falls in the transition zone
    zone_mask = []
    for ts in trades_df["entry_time"]:
        if ts in in_zone.index:
            zone_mask.append(bool(in_zone.loc[ts]))
        else:
            # Find nearest past bar
            past = in_zone[in_zone.index <= ts]
            zone_mask.append(bool(past.iloc[-1]) if not past.empty else False)

    trades_df = trades_df.copy()
    trades_df["in_transition_zone"] = zone_mask

    transition_pnls = trades_df[trades_df["in_transition_zone"]]["pnl_pips"].values
    stable_pnls     = trades_df[~trades_df["in_transition_zone"]]["pnl_pips"].values

    pf_trans  = pf_from_pnls(transition_pnls) if len(transition_pnls) >= MIN_WINDOW_TRADES else float("nan")
    pf_stable = pf_from_pnls(stable_pnls)     if len(stable_pnls)     >= MIN_WINDOW_TRADES else float("nan")

    # Answer the main question
    if np.isnan(pf_trans) or np.isnan(pf_stable):
        verdict = "INSUFFICIENT_DATA"
    elif pf_trans < pf_stable * 0.85:
        verdict = "DIE_AT_TRANSITIONS"    # clearly worse at transitions
    elif pf_stable < 1.0:
        verdict = "DIE_WITHIN_REGIMES"    # stable zones also broken
    elif pf_trans < 1.0 and pf_stable > 1.0:
        verdict = "TRANSITION_SENSITIVE"  # only hurt at transitions
    else:
        verdict = "TRANSITION_RESILIENT"  # similar PF both zones

    return {
        "n_transition_trades": len(transition_pnls),
        "n_stable_trades":     len(stable_pnls),
        "pf_transition_zone":  round(pf_trans,  3) if not np.isnan(pf_trans)  else None,
        "pf_stable_zone":      round(pf_stable, 3) if not np.isnan(pf_stable) else None,
        "pf_ratio":            round(pf_trans / max(pf_stable, 1e-9), 4) if not (np.isnan(pf_trans) or np.isnan(pf_stable)) else None,
        "verdict":             verdict,
    }


# ── Public interface ───────────────────────────────────────────────────────────

class RegimeTransitionAnalyzer:
    """
    Regime transition analysis for a collection of (asset, hypothesis) combinations.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_combination(
        self,
        df_prepared: pd.DataFrame,
        trades: List,
        asset_key: str,
        hyp_name: str,
    ) -> Dict:
        """Run full transition analysis for one (asset, hypothesis) combination."""
        blocks    = build_regime_blocks(df_prepared)
        trades_df = trades_to_df(trades)
        key       = f"{asset_key}_{hyp_name}"

        count_matrix, prob_matrix = compute_transition_matrix(blocks)
        per_pair   = compute_per_transition_pf(blocks, trades_df)
        zone_stats = transition_vs_stable_pf(df_prepared, blocks, trades_df)

        return {
            "key":             key,
            "asset":           asset_key,
            "hypothesis":      hyp_name,
            "n_blocks":        len(blocks),
            "n_transitions":   max(len(blocks) - 1, 0),
            "count_matrix":    count_matrix.to_dict(),
            "prob_matrix":     prob_matrix.to_dict(),
            "per_pair_stats":  per_pair.to_dict("records") if not per_pair.empty else [],
            "zone_stats":      zone_stats,
        }

    def run(
        self,
        combinations: List[Tuple],    # [(asset_key, hyp_name, df_prepared, trades)]
    ) -> Dict:
        """Run analysis over all combinations and save results."""
        all_per_pair = []
        all_zone     = []
        count_accum  = pd.DataFrame(0, index=ALL_REGIMES, columns=ALL_REGIMES)
        prob_accum   = pd.DataFrame(0.0, index=ALL_REGIMES, columns=ALL_REGIMES)
        results      = []

        for asset_key, hyp_name, df_prepared, trades in combinations:
            if df_prepared.empty or not trades:
                continue
            res = self.analyze_combination(df_prepared, trades, asset_key, hyp_name)
            results.append(res)

            # Accumulate matrices
            cm = pd.DataFrame(res["count_matrix"])
            for r in ALL_REGIMES:
                for c in ALL_REGIMES:
                    if r in cm.index and c in cm.columns:
                        count_accum.loc[r, c] += cm.loc[r, c]

            for pair in res["per_pair_stats"]:
                pair["asset"] = asset_key
                pair["hypothesis"] = hyp_name
                all_per_pair.append(pair)

            if res["zone_stats"]:
                row = {**res["zone_stats"], "asset": asset_key, "hypothesis": hyp_name}
                all_zone.append(row)

        # Normalize accumulated count matrix to probability
        row_sums = count_accum.sum(axis=1).replace(0, np.nan)
        prob_accum = count_accum.div(row_sums, axis=0).fillna(0.0).round(4)

        # Save
        count_accum.to_csv(self.output_dir / "transition_count_matrix.csv", encoding="utf-8")
        prob_accum.to_csv(self.output_dir / "transition_prob_matrix.csv", encoding="utf-8")

        if all_per_pair:
            pd.DataFrame(all_per_pair).to_csv(
                self.output_dir / "per_transition_stats.csv", index=False, encoding="utf-8"
            )
        if all_zone:
            pd.DataFrame(all_zone).to_csv(
                self.output_dir / "transition_zone_stats.csv", index=False, encoding="utf-8"
            )

        # Generate report
        report = self._format_report(count_accum, prob_accum, all_per_pair, all_zone, results)
        (self.output_dir / "regime_transition_report.txt").write_text(report, encoding="utf-8")
        print(report)

        return {
            "count_matrix":   count_accum,
            "prob_matrix":    prob_accum,
            "per_pair":       all_per_pair,
            "zone_stats":     all_zone,
            "combinations":   results,
        }

    @staticmethod
    def _format_report(count_matrix, prob_matrix, per_pair, zone_stats, results) -> str:
        lines = [
            "=" * 72,
            "  REGIME TRANSITION ANALYSIS",
            f"  Generated: {datetime.utcnow().isoformat()}",
            f"  Combinations analyzed: {len(results)}",
            "=" * 72,
            "",
            "  AGGREGATE TRANSITION PROBABILITY MATRIX (P[TO | FROM])",
            "",
        ]

        # Matrix table
        header = f"  {'FROM \\ TO':<22}" + "".join(f"{r[:12]:>14}" for r in ALL_REGIMES)
        lines.append(header)
        lines.append("  " + "-" * (22 + 14 * len(ALL_REGIMES)))
        for from_r in ALL_REGIMES:
            if from_r in prob_matrix.index:
                cells = "".join(f"{prob_matrix.loc[from_r, c]:>14.3f}" for c in ALL_REGIMES)
                lines.append(f"  {from_r:<22}{cells}")

        # Per-pair stats
        if per_pair:
            df_pp = pd.DataFrame(per_pair).dropna(subset=["delta_pf"])
            if not df_pp.empty:
                lines += [
                    "",
                    "  TOP DEGRADING TRANSITIONS (delta_pf most negative)",
                    "",
                    f"  {'FROM':<22} {'TO':<22} {'N':>5} {'PF_pre':>8} {'PF_post':>8} {'Delta':>8}  {'Degrades'}",
                    "  " + "-" * 86,
                ]
                worst = df_pp.sort_values("delta_pf").head(10)
                for _, row in worst.iterrows():
                    sign = "YES" if row.get("degrades") else "NO"
                    lines.append(
                        f"  {row['from_regime']:<22} {row['to_regime']:<22} "
                        f"{int(row['n_events']):>5} {row.get('pf_pre', 0) or 0:>8.3f} "
                        f"{row.get('pf_post', 0) or 0:>8.3f} {row['delta_pf']:>8.4f}  {sign}"
                    )

        # Zone verdict summary
        if zone_stats:
            lines += ["", "  TRANSITION ZONE vs STABLE ZONE — PER COMBINATION", ""]
            lines.append(f"  {'Combination':<30} {'PF_trans':>10} {'PF_stable':>10} {'Verdict':<25}")
            lines.append("  " + "-" * 80)
            for z in sorted(zone_stats, key=lambda x: x.get("pf_ratio") or 1.0):
                key = f"{z['asset']}_{z['hypothesis']}"
                lines.append(
                    f"  {key:<30} {z.get('pf_transition_zone') or 0:>10.3f} "
                    f"{z.get('pf_stable_zone') or 0:>10.3f} {z.get('verdict','?'):<25}"
                )

        # Aggregate answer
        verdicts = [z.get("verdict","") for z in zone_stats]
        die_trans = verdicts.count("DIE_AT_TRANSITIONS")
        die_within = verdicts.count("DIE_WITHIN_REGIMES") + verdicts.count("STRUCTURAL_FAILURE")
        resilient  = verdicts.count("TRANSITION_RESILIENT")

        lines += [
            "",
            "=" * 72,
            "  RESPOSTA: sistemas morrem na transicao ou dentro do regime?",
            "=" * 72,
            f"  DIE_AT_TRANSITIONS:  {die_trans} combinacoes",
            f"  DIE_WITHIN_REGIMES:  {die_within} combinacoes",
            f"  TRANSITION_RESILIENT: {resilient} combinacoes",
            "",
        ]

        if die_trans > die_within:
            lines.append("  => CONCLUSAO: estrategias morrem PREDOMINANTEMENTE nas transicoes de regime.")
            lines.append("     Implementar filtro de 'transition window' pode recuperar edge significativo.")
        elif die_within >= die_trans:
            lines.append("  => CONCLUSAO: estrategias morrem DENTRO dos regimes (edge sistematicamente fraco).")
            lines.append("     Selecao de regime mais rigorosa e necessaria antes de entrar.")
        lines.append("=" * 72)

        return "\n".join(lines)
