"""
Regime Analysis — Sistema 1 performance by market regime.

Loads 1H OHLCV + Sistema 1 signals, classifies each bar into one of 3 regimes,
tags each trade with its entry regime, resolves TP/SL outcomes, computes
per-regime statistics, and runs KS separability tests.

Key question:
  "Are EUR/USD market regimes statistically different enough to justify
   an execution filter for Sistema 1?"

Usage:
    .\\venv\\Scripts\\python scripts\\run_regime_analysis.py
    .\\venv\\Scripts\\python scripts\\run_regime_analysis.py --period fold1
    .\\venv\\Scripts\\python scripts\\run_regime_analysis.py --period holdout
    .\\venv\\Scripts\\python scripts\\run_regime_analysis.py --period all
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import sqlite3

from research.fractal_lab.regime_classifier import (
    REGIME_STRESS, REGIME_COMPRESSED, REGIME_STRUCTURED, REGIMES,
    compute_features, tag_regimes, resolve_outcomes,
    tag_trades_with_regime, regime_stats, ks_test_regimes, separability_score,
)

DB_PATH       = _ROOT / "data" / "trading.db"
SIGNALS_CSV   = _ROOT / "data" / "mt5_signals.csv"
REPORT_PATH   = _ROOT / "data" / "regime_analysis.json"

FOLD1_START   = "2024-09-01"
FOLD1_END     = "2025-02-28"
HOLDOUT_START = "2025-12-01"


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_ohlcv_1h() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT datetime, open, high, low, close, volume FROM ohlcv_1h", conn)
    conn.close()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    df.columns = [c.title() for c in df.columns]
    print(f"[ohlcv_1h] {len(df):,} rows  {df.index[0].date()} -> {df.index[-1].date()}")
    return df


def load_adx_from_db() -> pd.Series:
    """Load pre-computed ADX from signals_1h table."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT datetime, adx FROM signals_1h", conn)
    conn.close()
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.floor("h")
    df = df.dropna(subset=["adx"]).set_index("datetime")["adx"]
    return df


def load_signals() -> pd.DataFrame:
    """Load mt5_signals.csv (310 trades with SL/TP defined)."""
    df = pd.read_csv(SIGNALS_CSV)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"[signals]  {len(df):,} trades  {df.datetime.min().date()} -> {df.datetime.max().date()}")
    print(f"           BUY: {(df.signal==1).sum()}  SELL: {(df.signal==-1).sum()}")
    return df


def filter_period(df: pd.DataFrame, period: str, dt_col: str = "datetime") -> pd.DataFrame:
    if period == "fold1":
        return df[(df[dt_col] >= FOLD1_START) & (df[dt_col] <= FOLD1_END)]
    if period == "holdout":
        return df[df[dt_col] >= HOLDOUT_START]
    return df  # "all"


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run(period: str = "all") -> dict:
    print("=" * 64)
    print(f"  Regime Analysis v1  |  period={period}")
    print("=" * 64)

    # 1. Load OHLCV and compute features
    df_ohlcv = load_ohlcv_1h()
    adx_series = load_adx_from_db()

    print("\n[features] Computing ATR ratio, disorder_proxy, coherence_proxy ...")
    df_feat = compute_features(df_ohlcv)

    # Merge pre-computed ADX (more accurate than manual calculation)
    df_feat["adx"] = adx_series.reindex(df_feat.index).fillna(df_feat.get("adx", np.nan))

    # 2. Classify regimes for every bar
    df_feat = tag_regimes(df_feat)
    regime_counts = df_feat["regime"].value_counts()
    total = len(df_feat)
    print(f"\n[regimes]  Distribution over {total:,} 1H bars:")
    for r in REGIMES:
        n = regime_counts.get(r, 0)
        print(f"  {r:<28} {n:5,}  ({100*n/total:.1f}%)")

    # 3. Load and filter signals
    df_signals = load_signals()
    df_signals = filter_period(df_signals, period)
    print(f"\n[signals]  After period filter ({period}): {len(df_signals):,} trades")

    if len(df_signals) < 10:
        print(f"[WARN] Too few trades for statistical analysis: {len(df_signals)}")

    # 4. Resolve TP/SL outcomes
    print("\n[outcomes] Resolving TP/SL via forward OHLCV simulation ...")
    df_signals = resolve_outcomes(df_signals, df_ohlcv)
    outcome_counts = df_signals["outcome"].value_counts()
    print(f"  TP_HIT : {outcome_counts.get('TP_HIT', 0):3d}")
    print(f"  SL_HIT : {outcome_counts.get('SL_HIT', 0):3d}")
    print(f"  TIMEOUT: {outcome_counts.get('TIMEOUT', 0):3d}")
    print(f"  Overall win rate: {(df_signals['outcome']=='TP_HIT').mean():.1%}")
    print(f"  Mean pnl: {df_signals['pnl_pips'].mean():.2f} pips")

    # 5. Tag each trade with regime at entry
    df_feat_for_tag = filter_period(
        df_feat.reset_index().rename(columns={"index": "datetime"}),
        period, "datetime"
    ).set_index("datetime")

    df_tagged = tag_trades_with_regime(df_signals, df_feat)
    tagged_counts = df_tagged["regime"].value_counts()
    print(f"\n[tagging]  Trades per regime:")
    for r in REGIMES:
        n = tagged_counts.get(r, 0)
        print(f"  {r:<28} {n:3d}")

    # 6. Per-regime statistics
    print("\n[stats]    Per-regime statistics:")
    stats_df = regime_stats(df_tagged)

    header = f"  {'Regime':<28} {'N':>4}  {'WinRate':>8}  {'PF':>6}  {'E(pips)':>8}  {'Sharpe':>7}  {'MaxDD':>8}"
    print(header)
    print("  " + "-" * 75)
    for regime, row in stats_df.iterrows():
        n = int(row["n_trades"]) if not np.isnan(row["n_trades"]) else 0
        wr  = f"{row['win_rate']:.1%}"   if not np.isnan(row['win_rate'])            else "  N/A"
        pf  = f"{row['profit_factor']:.2f}" if not np.isnan(row['profit_factor'])    else "  N/A"
        e   = f"{row['expectancy_pips']:+.2f}" if not np.isnan(row['expectancy_pips']) else "  N/A"
        sh  = f"{row['sharpe_proxy']:.3f}"  if not np.isnan(row['sharpe_proxy'])     else "  N/A"
        dd  = f"{row['max_drawdown_pips']:.1f}" if not np.isnan(row['max_drawdown_pips']) else "  N/A"
        print(f"  {regime:<28} {n:>4}  {wr:>8}  {pf:>6}  {e:>8}  {sh:>7}  {dd:>8}")

    # 7. KS separability test
    print("\n[ks-test]  Kolmogorov-Smirnov pairwise separability:")
    ks_results = ks_test_regimes(df_tagged)
    for pair, res in ks_results.items():
        sig_str = "* p<0.05" if res.get("significant") else "  n.s."
        ks_s = f"{res['ks_stat']:.4f}" if not np.isnan(res.get("ks_stat", np.nan)) else " N/A "
        pv   = f"{res['p_value']:.4f}" if not np.isnan(res.get("p_value", np.nan)) else " N/A "
        n1, n2 = res.get("n", (0, 0))
        print(f"  {pair.replace('REGIME_', 'R').replace('_', '.'):<45} "
              f"KS={ks_s}  p={pv}  n=({n1},{n2})  {sig_str}")

    # 8. Separability score
    sep = separability_score(ks_results, stats_df)
    print(f"\n[score]    REGIME_SEPARABILITY_SCORE = {sep['REGIME_SEPARABILITY_SCORE']:.4f}")
    print(f"           {sep['verdict']}")

    # 9. Final answer
    answer = _build_answer(sep, stats_df, df_tagged)
    print(f"\n{'='*64}")
    print("PERGUNTA: Os regimes sao estatisticamente diferentes o suficiente")
    print("          para justificar um filtro de execucao no Sistema 1?")
    print(f"RESPOSTA: {answer}")
    print(f"{'='*64}\n")

    # 10. Save report
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "period": period,
        "regime_definitions": {
            REGIME_STRESS:     {"atr_ratio": f"> 1.5", "adx": f"> 35.0",
                                "disorder_proxy": f"> 0.55", "priority": 1},
            REGIME_COMPRESSED: {"atr_ratio": f"< 0.8", "adx": f"< 18.0",
                                "coherence_proxy": f"< 0.42", "priority": 2},
            REGIME_STRUCTURED: {"description": "default (not STRESS or COMPRESSED)",
                                "priority": 3},
        },
        "distribution": {r: int(regime_counts.get(r, 0)) for r in REGIMES},
        "outcome_summary": {
            "tp_hit": int(outcome_counts.get("TP_HIT", 0)),
            "sl_hit": int(outcome_counts.get("SL_HIT", 0)),
            "timeout": int(outcome_counts.get("TIMEOUT", 0)),
            "overall_win_rate": round(float((df_tagged["outcome"] == "TP_HIT").mean()), 4),
            "mean_pnl_pips": round(float(df_tagged["pnl_pips"].mean()), 2),
        },
        "regime_stats": stats_df.reset_index().to_dict(orient="records"),
        "ks_results": ks_results,
        "separability": sep,
        "answer": answer,
        "tagged_trades_sample": df_tagged[
            ["datetime", "signal", "entry", "outcome", "pnl_pips", "regime"]
        ].head(20).to_dict(orient="records"),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"[report] Saved to {REPORT_PATH}")

    return report


def _build_answer(sep: dict, stats_df: pd.DataFrame, df_tagged: pd.DataFrame) -> str:
    score = sep["REGIME_SEPARABILITY_SCORE"]

    # Identify best and worst regime by expectancy (with min 10 trades)
    valid = stats_df[stats_df["n_trades"] >= 10]["expectancy_pips"].dropna()
    if len(valid) >= 2:
        best_r  = valid.idxmax()
        worst_r = valid.idxmin()
        best_e  = valid.max()
        worst_e = valid.min()
        e_spread = f"Melhor: {best_r} E={best_e:+.2f}pips  |  Pior: {worst_r} E={worst_e:+.2f}pips."
        filter_rec = (
            f"Filtrar '{worst_r}' aumentaria expectativa. "
            f"ATENCAO: resultado e period-specific (bullish Jan-Mai2026); "
            f"validar em Fold1 antes de implementar."
        )
    else:
        e_spread = "Poucos trades para comparar expectativas."
        filter_rec = "Mais dados necessarios."

    if score > 0.6:
        verdict = "SIM"
        detail = f"Score={score:.2f}. Regimes separaveis (KS p<0.05). {e_spread} {filter_rec}"
    elif score > 0.4:
        verdict = "PARCIALMENTE"
        detail = (
            f"Score={score:.2f}. Evidencia fraca a moderada. {e_spread} "
            f"Filtro pode ajudar mas precisa mais trades para confirmar."
        )
    else:
        verdict = "NAO (INCONCLUSIVO)"
        detail = (
            f"Score={score:.2f}. Regimes NAO separaveis com dados disponiveis. "
            f"Nao implementar filtro sem mais evidencia."
        )
    return f"{verdict} — {detail}"


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", default="all",
                        choices=["all", "fold1", "holdout"],
                        help="Period to analyze (default: all)")
    args = parser.parse_args()
    run(period=args.period)
