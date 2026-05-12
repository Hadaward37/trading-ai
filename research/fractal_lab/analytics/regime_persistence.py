"""
Regime Persistence Analysis.

Calcula sobre a sequencia temporal de regimes (df_prepared com coluna 'regime'):
  1. Duracao media de cada regime (candles e dias estimados)
  2. Distribuicao de duracao: min, p25, median, p75, p95, max
  3. Markov Transition Matrix: probabilidade empirica P(TO | FROM)
  4. Steady-state distribution: distribuicao de longo prazo via autovetor da Markov
  5. Impacto da duracao no PF: correlacao entre duracao do bloco e PF naquele bloco

Causalidade: toda classificacao de regime em bar T usa apenas dados <= T.
A Markov Matrix e estimada empiricamente sem assumir dados futuros.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._shared import (
    ALL_REGIMES, BEHAVIORAL_MAP_DIR,
    build_regime_blocks, pf_from_pnls, trades_to_df,
)

OUTPUT_DIR = BEHAVIORAL_MAP_DIR


# ── Duration statistics ────────────────────────────────────────────────────────

def compute_duration_stats(blocks: pd.DataFrame) -> pd.DataFrame:
    """
    Per-regime statistics of block duration (in bars).
    Assumes 1H bars -> duration_hours = duration_bars.
    """
    if blocks.empty:
        return pd.DataFrame()

    records = []
    for regime in ALL_REGIMES:
        sub = blocks[blocks["regime"] == regime]["duration_bars"].values
        if len(sub) == 0:
            records.append({
                "regime": regime,
                "n_blocks": 0,
                "mean_bars": 0,
                "median_bars": 0,
                "p25_bars": 0,
                "p75_bars": 0,
                "p95_bars": 0,
                "max_bars": 0,
                "mean_hours": 0.0,
                "mean_days": 0.0,
                "total_bars": 0,
                "pct_of_time": 0.0,
            })
            continue

        total_bars = int(blocks["duration_bars"].sum())
        records.append({
            "regime":       regime,
            "n_blocks":     len(sub),
            "mean_bars":    round(float(sub.mean()), 1),
            "median_bars":  float(np.median(sub)),
            "p25_bars":     float(np.percentile(sub, 25)),
            "p75_bars":     float(np.percentile(sub, 75)),
            "p95_bars":     float(np.percentile(sub, 95)),
            "max_bars":     int(sub.max()),
            "mean_hours":   round(float(sub.mean()), 1),          # 1H bars
            "mean_days":    round(float(sub.mean()) / 24, 2),     # trading hours
            "total_bars":   int(sub.sum()),
            "pct_of_time":  round(float(sub.sum()) / max(total_bars, 1) * 100, 2),
        })

    return pd.DataFrame(records)


# ── Markov Transition Matrix ───────────────────────────────────────────────────

def build_markov_matrix(blocks: pd.DataFrame) -> pd.DataFrame:
    """
    Empirical Markov Transition Matrix: P[TO | FROM].
    Rows = FROM, Columns = TO. Each row sums to 1.

    Includes self-transitions (staying in same regime across adjacent blocks
    is impossible by construction, but the matrix is well-defined as the
    probability of transitioning to each other regime given you just left the current one).
    """
    count = pd.DataFrame(0, index=ALL_REGIMES, columns=ALL_REGIMES)

    for i in range(len(blocks) - 1):
        f = blocks.iloc[i]["regime"]
        t = blocks.iloc[i + 1]["regime"]
        if f in ALL_REGIMES and t in ALL_REGIMES:
            count.loc[f, t] += 1

    row_sums = count.sum(axis=1).replace(0, np.nan)
    prob = count.div(row_sums, axis=0).fillna(0.0).round(4)
    return prob


def compute_steady_state(markov: pd.DataFrame) -> Dict[str, float]:
    """
    Steady-state distribution pi such that pi = pi * P.
    Solved via left eigenvector of the transition matrix.
    Returns normalised probability vector.
    """
    P = markov.values
    if P.shape[0] == 0 or np.allclose(P, 0):
        return {r: 0.0 for r in ALL_REGIMES}

    # Left eigenvector for eigenvalue ~1
    try:
        eigenvalues, eigenvectors = np.linalg.eig(P.T)
        # Find the eigenvector closest to eigenvalue 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()
        stationary = np.clip(stationary, 0, None)
        stationary = stationary / stationary.sum()
        return {r: round(float(stationary[i]), 4) for i, r in enumerate(ALL_REGIMES)}
    except Exception:
        return {r: 0.0 for r in ALL_REGIMES}


# ── Duration-PF correlation ────────────────────────────────────────────────────

def duration_pf_correlation(
    blocks: pd.DataFrame,
    trades_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each regime, correlate block duration with block PF.
    Blocks with fewer than 5 trades are excluded (unreliable PF estimate).

    Returns: DataFrame with (regime, pearson_r, n_blocks, interpretation).
    """
    if trades_df.empty or blocks.empty:
        return pd.DataFrame()

    records = []
    for regime in ALL_REGIMES:
        regime_blocks = blocks[blocks["regime"] == regime]
        durations, pfs = [], []

        for _, blk in regime_blocks.iterrows():
            mask = (
                (trades_df["entry_time"] >= blk["start_ts"]) &
                (trades_df["entry_time"] <= blk["end_ts"])
            )
            block_pnls = trades_df[mask]["pnl_pips"].values
            if len(block_pnls) < 5:
                continue
            durations.append(blk["duration_bars"])
            pfs.append(pf_from_pnls(block_pnls))

        if len(durations) < 3:
            r, interpretation = float("nan"), "INSUFFICIENT_DATA"
        else:
            d_arr = np.array(durations, dtype=float)
            p_arr = np.array(pfs, dtype=float)
            # Clip extreme PF values before correlation
            p_arr = np.clip(p_arr, 0, 5.0)
            r = float(np.corrcoef(d_arr, p_arr)[0, 1])
            if np.isnan(r):
                interpretation = "INSUFFICIENT_DATA"
            elif r > 0.3:
                interpretation = "LONGER_DURATION_BETTER_PF"
            elif r < -0.3:
                interpretation = "SHORTER_DURATION_BETTER_PF"
            else:
                interpretation = "NO_DURATION_EFFECT"

        records.append({
            "regime":        regime,
            "n_blocks_used": len(durations),
            "pearson_r":     round(r, 4) if not np.isnan(r) else None,
            "interpretation": interpretation,
        })

    return pd.DataFrame(records)


# ── Public interface ───────────────────────────────────────────────────────────

class RegimePersistenceAnalyzer:
    """
    Regime persistence: duration statistics, Markov matrix, steady-state, duration-PF correlation.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze(
        self,
        df_prepared: pd.DataFrame,
        trades: Optional[List] = None,
        asset_key: str = "UNKNOWN",
    ) -> Dict:
        """Full persistence analysis for one asset's regime sequence."""
        blocks   = build_regime_blocks(df_prepared)
        dur_stats = compute_duration_stats(blocks)
        markov   = build_markov_matrix(blocks)
        steady   = compute_steady_state(markov)

        trades_df = trades_to_df(trades) if trades else pd.DataFrame()
        dur_pf    = duration_pf_correlation(blocks, trades_df)

        return {
            "asset":       asset_key,
            "n_blocks":    len(blocks),
            "n_regimes":   blocks["regime"].nunique(),
            "dur_stats":   dur_stats,
            "markov":      markov,
            "steady_state": steady,
            "dur_pf":      dur_pf,
        }

    def run(
        self,
        asset_dfs: List[Tuple],    # [(asset_key, df_prepared, trades)]
    ) -> Dict:
        """
        Aggregate Markov matrix and duration stats across all assets.
        Saves per-asset and aggregate outputs.
        """
        all_dur_stats = []
        all_dur_pf    = []
        accum_count   = pd.DataFrame(0, index=ALL_REGIMES, columns=ALL_REGIMES)
        all_steady    = {}

        for asset_key, df_prepared, trades in asset_dfs:
            if df_prepared.empty:
                continue
            res = self.analyze(df_prepared, trades, asset_key)

            ds = res["dur_stats"].copy()
            ds["asset"] = asset_key
            all_dur_stats.append(ds)

            dp = res["dur_pf"].copy()
            dp["asset"] = asset_key
            all_dur_pf.append(dp)

            all_steady[asset_key] = res["steady_state"]

            # Accumulate transition counts for aggregate Markov matrix
            m = res["markov"]
            for r in ALL_REGIMES:
                for c in ALL_REGIMES:
                    if r in m.index and c in m.columns:
                        accum_count.loc[r, c] += int(
                            round(m.loc[r, c] * max(1, int(res["n_blocks"])))
                        )

        # Normalise aggregate Markov
        row_sums = accum_count.sum(axis=1).replace(0, np.nan)
        agg_markov = accum_count.div(row_sums, axis=0).fillna(0.0).round(4)
        agg_steady = compute_steady_state(agg_markov)

        # Save
        if all_dur_stats:
            pd.concat(all_dur_stats, ignore_index=True).to_csv(
                self.output_dir / "regime_duration_stats.csv", index=False, encoding="utf-8"
            )
        if all_dur_pf:
            pd.concat(all_dur_pf, ignore_index=True).to_csv(
                self.output_dir / "duration_pf_correlation.csv", index=False, encoding="utf-8"
            )
        agg_markov.to_csv(self.output_dir / "markov_matrix.csv", encoding="utf-8")

        steady_df = pd.DataFrame.from_dict(
            {**all_steady, "AGGREGATE": agg_steady}, orient="index"
        )
        steady_df.to_csv(self.output_dir / "steady_state_distribution.csv", encoding="utf-8")

        report = self._format_report(all_dur_stats, agg_markov, agg_steady, all_dur_pf, all_steady)
        (self.output_dir / "regime_persistence_report.txt").write_text(report, encoding="utf-8")
        print(report)

        return {
            "aggregate_markov": agg_markov,
            "aggregate_steady": agg_steady,
            "per_asset_steady": all_steady,
        }

    @staticmethod
    def _format_report(dur_stats_list, markov, steady, dur_pf_list, per_asset_steady) -> str:
        lines = [
            "=" * 72,
            "  REGIME PERSISTENCE ANALYSIS",
            f"  Generated: {datetime.utcnow().isoformat()}",
            "=" * 72,
        ]

        # Aggregate duration stats
        if dur_stats_list:
            agg = pd.concat(dur_stats_list, ignore_index=True)
            by_regime = agg.groupby("regime")[["mean_bars", "median_bars", "p95_bars", "max_bars", "n_blocks"]].mean()

            lines += ["", "  REGIME DURATION STATS (aggregate across assets, 1H bars)", ""]
            lines.append(f"  {'Regime':<22} {'Mean':>7} {'Median':>8} {'P95':>7} {'Max':>7} {'Blocks':>8} {'~Days':>8}")
            lines.append("  " + "-" * 70)
            for regime in ALL_REGIMES:
                if regime in by_regime.index:
                    row = by_regime.loc[regime]
                    lines.append(
                        f"  {regime:<22} {row['mean_bars']:>7.1f} {row['median_bars']:>8.1f} "
                        f"{row['p95_bars']:>7.1f} {row['max_bars']:>7.1f} "
                        f"{row['n_blocks']:>8.0f} {row['mean_bars']/24:>8.2f}"
                    )

        # Markov matrix
        lines += ["", "  MARKOV TRANSITION MATRIX P[TO | FROM] (aggregate)", ""]
        header = f"  {'FROM \\ TO':<22}" + "".join(f"{r[:12]:>14}" for r in ALL_REGIMES)
        lines.append(header)
        lines.append("  " + "-" * (22 + 14 * len(ALL_REGIMES)))
        for from_r in ALL_REGIMES:
            if from_r in markov.index:
                cells = "".join(f"{markov.loc[from_r, c]:>14.3f}" for c in ALL_REGIMES)
                lines.append(f"  {from_r:<22}{cells}")

        # Steady-state
        lines += ["", "  STEADY-STATE DISTRIBUTION (long-run regime probability)", ""]
        for r, p in steady.items():
            bar = "=" * int(p * 40)
            lines.append(f"  {r:<22} {p:6.3f}  [{bar:<40}]")

        # Duration-PF insight
        if dur_pf_list:
            all_dp = pd.concat(dur_pf_list, ignore_index=True)
            lines += ["", "  DURATION -> PF CORRELATION (does regime duration affect edge?)", ""]
            lines.append(f"  {'Regime':<22} {'n_assets':>8} {'Mean r':>8} {'Interpretation'}")
            lines.append("  " + "-" * 70)
            for regime in ALL_REGIMES:
                sub = all_dp[all_dp["regime"] == regime].dropna(subset=["pearson_r"])
                if sub.empty:
                    continue
                mean_r = float(sub["pearson_r"].mean())
                interp = sub["interpretation"].mode().iloc[0] if not sub.empty else "?"
                lines.append(f"  {regime:<22} {len(sub):>8} {mean_r:>8.3f}  {interp}")

        lines.append("\n" + "=" * 72)
        return "\n".join(lines)
