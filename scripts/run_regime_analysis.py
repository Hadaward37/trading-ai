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
    BIAS_BULL, BIAS_BEAR, BIAS_NEUTRAL,
    STRETCH_BUCKETS, STRETCH_LOW_LABEL, STRETCH_HIGH_LABEL,
    compute_features, tag_regimes, resolve_outcomes,
    tag_trades_with_regime, tag_alignment,
    regime_stats, ks_test_regimes, separability_score,
    alignment_stats, ks_test_alignment,
    tag_stretch, stretch_stats, stretch_regime_stats,
    ks_test_stretch, answer_stretch_questions,
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

    # 9. Final answer (v1)
    answer = _build_answer(sep, stats_df, df_tagged)
    print(f"\n{'='*64}")
    print("PERGUNTA v1: Regimes sao separaveis o suficiente para filtro?")
    print(f"RESPOSTA:    {answer}")
    print(f"{'='*64}\n")

    # ── v2: Dual-Slope Alignment ───────────────────────────────────────────────
    print(f"{'='*64}")
    print("  v2 — Dual-Slope Alignment Filter")
    print(f"{'='*64}")

    # Tag alignment
    df_tagged = tag_alignment(df_tagged, df_feat)

    bias_counts = df_tagged["market_bias"].value_counts()
    aligned_n   = df_tagged["is_aligned"].sum()
    total_n     = len(df_tagged)
    print(f"\n[bias]   Market bias distribution at trade entries:")
    print(f"  BULL    : {bias_counts.get(BIAS_BULL,    0):3d}")
    print(f"  BEAR    : {bias_counts.get(BIAS_BEAR,    0):3d}")
    print(f"  NEUTRAL : {bias_counts.get(BIAS_NEUTRAL, 0):3d}")
    print(f"  is_aligned=True: {aligned_n} / {total_n} ({100*aligned_n/total_n:.1f}%)")

    # Regime x Alignment stats
    print(f"\n[v2-stats] Regime x Alignment performance:")
    align_df = alignment_stats(df_tagged)

    header = f"  {'Combination':<30} {'N':>4}  {'WinRate':>8}  {'PF':>6}  {'E(pips)':>8}  {'Sharpe':>7}"
    print(header)
    print("  " + "-" * 68)
    for label, row in align_df.iterrows():
        n   = int(row["n_trades"]) if not np.isnan(row["n_trades"]) else 0
        wr  = f"{row['win_rate']:.1%}"        if not np.isnan(row["win_rate"])         else "  N/A"
        pf  = f"{row['profit_factor']:.2f}"  if not np.isnan(row["profit_factor"])    else "  N/A"
        e   = f"{row['expectancy_pips']:+.2f}" if not np.isnan(row["expectancy_pips"]) else "  N/A"
        sh  = f"{row['sharpe_proxy']:.3f}"   if not np.isnan(row["sharpe_proxy"])     else "  N/A"
        print(f"  {label:<30} {n:>4}  {wr:>8}  {pf:>6}  {e:>8}  {sh:>7}")

    # KS-test: STRESS_ALIGNED vs STRESS_NOT_ALIGNED
    ks_align = ks_test_alignment(df_tagged)
    print(f"\n[ks-v2]  STRESS ALIGNED vs NOT_ALIGNED:")
    ks_s = f"{ks_align['ks_stat']:.4f}" if not np.isnan(ks_align.get("ks_stat", np.nan)) else " N/A"
    pv   = f"{ks_align['p_value']:.4f}" if not np.isnan(ks_align.get("p_value", np.nan)) else " N/A"
    sig  = "* p<0.05 (SIGNIFICANT)" if ks_align.get("significant") else "  n.s."
    n1, n2 = ks_align.get("n", (0, 0))
    print(f"  KS={ks_s}  p={pv}  n=({n1},{n2})  {sig}")
    if "note" in ks_align:
        print(f"  Note: {ks_align['note']}")

    # Answer the 3 specific questions
    v2_answers = _answer_v2_questions(df_tagged, align_df, ks_align, stats_df)
    print(f"\n{'='*64}")
    print("PERGUNTAS v2:")
    for q, a in v2_answers.items():
        print(f"  Q: {q}")
        print(f"  A: {a}")
        print()
    print(f"{'='*64}\n")

    # ── v3: Distance-to-mean stretch analysis ─────────────────────────────────
    print(f"{'='*64}")
    print("  v3 — Distance-to-Mean Stretch Analysis")
    print(f"{'='*64}")

    df_tagged = tag_stretch(df_tagged, df_feat)

    # Distribution of stretch at trade entries
    bkt_dist = df_tagged["stretch_bucket"].value_counts().sort_index()
    mr_pct   = df_tagged["mr_setup"].mean()
    print(f"\n[stretch] Distribution at trade entries (|stretch| = |close-EMA50|/ATR14):")
    for bkt, n in bkt_dist.items():
        print(f"  {bkt:<20} {n:3d}  ({100*n/len(df_tagged):.1f}%)")
    print(f"  mr_setup=True (BUY below / SELL above EMA50): "
          f"{df_tagged['mr_setup'].sum()} ({100*mr_pct:.1f}%)")

    # Per-bucket stats
    print(f"\n[v3-stats] Per-bucket performance:")
    bkt_stats = stretch_stats(df_tagged)
    hdr = f"  {'Bucket':<20} {'N':>4}  {'WinRate':>8}  {'PF':>6}  {'E(pips)':>8}  {'Sharpe':>7}  {'MeanStr':>8}"
    print(hdr)
    print("  " + "-" * 72)
    for label, row in bkt_stats.iterrows():
        n   = int(row["n_trades"]) if not np.isnan(row["n_trades"]) else 0
        wr  = f"{row['win_rate']:.1%}"         if not np.isnan(row["win_rate"])          else "   N/A"
        pf  = f"{row['profit_factor']:.2f}"   if not np.isnan(row["profit_factor"])     else "  N/A"
        e   = f"{row['expectancy_pips']:+.2f}" if not np.isnan(row["expectancy_pips"])   else "   N/A"
        sh  = f"{row['sharpe_proxy']:.3f}"    if not np.isnan(row["sharpe_proxy"])      else "   N/A"
        ms  = f"{row['mean_stretch']:.2f}"    if not np.isnan(row["mean_stretch"])      else "  N/A"
        print(f"  {label:<20} {n:>4}  {wr:>8}  {pf:>6}  {e:>8}  {sh:>7}  {ms:>8}")

    # Bucket × regime stats (STRESS and STRUCTURED)
    print(f"\n[v3-regime] Bucket x Regime (STRESS / STRUCTURED):")
    reg_bkt = stretch_regime_stats(df_tagged)
    hdr2 = f"  {'Bucket':<20} {'Regime':<26} {'N':>4}  {'PF':>6}  {'E(pips)':>8}"
    print(hdr2)
    print("  " + "-" * 68)
    for (bkt, reg), row in reg_bkt.iterrows():
        if reg == REGIME_COMPRESSED:
            continue
        n  = int(row["n_trades"]) if not np.isnan(row["n_trades"]) else 0
        pf = f"{row['profit_factor']:.2f}"   if not np.isnan(row["profit_factor"])   else "  N/A"
        e  = f"{row['expectancy_pips']:+.2f}" if not np.isnan(row["expectancy_pips"]) else "   N/A"
        print(f"  {bkt:<20} {reg:<26} {n:>4}  {pf:>6}  {e:>8}")

    # KS-test: low vs high stretch
    ks_str = ks_test_stretch(df_tagged)
    print(f"\n[ks-v3]  {STRETCH_LOW_LABEL} vs {STRETCH_HIGH_LABEL}:")
    ks_s = f"{ks_str['ks_stat']:.4f}" if not np.isnan(ks_str.get("ks_stat", np.nan)) else " N/A"
    pv   = f"{ks_str['p_value']:.4f}" if not np.isnan(ks_str.get("p_value", np.nan)) else " N/A"
    sig  = "* SIGNIFICANT (p<0.05)" if ks_str.get("significant") else "  n.s."
    print(f"  KS={ks_s}  p={pv}  n=({ks_str.get('n_low',0)},{ks_str.get('n_high',0)})  {sig}")
    if "e_low" in ks_str:
        print(f"  E_low={ks_str['e_low']:+.2f}pips  E_high={ks_str['e_high']:+.2f}pips  "
              f"delta={ks_str['e_high']-ks_str['e_low']:+.2f}")
    if "note" in ks_str:
        print(f"  Note: {ks_str['note']}")

    # Answer the 4 questions
    stretch_answers = answer_stretch_questions(df_tagged, bkt_stats)
    print(f"\n{'='*64}")
    print("PERGUNTAS v3 — Stretch Analysis:")
    for q, a in stretch_answers.items():
        print(f"  Q: {q}")
        print(f"  A: {a}")
        print()
    print(f"{'='*64}\n")

    # 10. Save report
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "period": period,
        "regime_definitions": {
            REGIME_STRESS:     {"atr_ratio": "> 1.5", "adx": "> 35.0",
                                "disorder_proxy": "> 0.55", "priority": 1},
            REGIME_COMPRESSED: {"atr_ratio": "< 0.8", "adx": "< 18.0",
                                "coherence_proxy": "< 0.42", "priority": 2},
            REGIME_STRUCTURED: {"description": "default (not STRESS or COMPRESSED)",
                                "priority": 3},
        },
        "dual_slope_definition": {
            "slope_short": "EMA200[t-1] - EMA200[t-6]",
            "slope_mid":   "EMA200[t-1] - EMA200[t-21]",
            "BULL":    "slope_short > 0 AND slope_mid > 0",
            "BEAR":    "slope_short < 0 AND slope_mid < 0",
            "NEUTRAL": "slopes diverge",
        },
        "distribution": {r: int(regime_counts.get(r, 0)) for r in REGIMES},
        "outcome_summary": {
            "tp_hit": int(outcome_counts.get("TP_HIT", 0)),
            "sl_hit": int(outcome_counts.get("SL_HIT", 0)),
            "timeout": int(outcome_counts.get("TIMEOUT", 0)),
            "overall_win_rate": round(float((df_tagged["outcome"] == "TP_HIT").mean()), 4),
            "mean_pnl_pips": round(float(df_tagged["pnl_pips"].mean()), 2),
        },
        "v1_regime_stats": stats_df.reset_index().to_dict(orient="records"),
        "v1_ks_results": ks_results,
        "v1_separability": sep,
        "v1_answer": answer,
        "v2_alignment": {
            "bias_distribution": {
                "BULL":    int(bias_counts.get(BIAS_BULL,    0)),
                "BEAR":    int(bias_counts.get(BIAS_BEAR,    0)),
                "NEUTRAL": int(bias_counts.get(BIAS_NEUTRAL, 0)),
            },
            "aligned_trades": int(aligned_n),
            "aligned_pct": round(float(aligned_n / total_n), 4),
            "alignment_stats": align_df.reset_index().to_dict(orient="records"),
            "ks_stress_aligned_vs_not": ks_align,
            "answers": v2_answers,
        },
        "v3_stretch": {
            "bucket_stats": bkt_stats.reset_index().to_dict(orient="records"),
            "ks_low_vs_high": ks_str,
            "answers": stretch_answers,
        },
        "tagged_trades_sample": df_tagged[
            ["datetime", "signal", "entry", "outcome", "pnl_pips",
             "regime", "market_bias", "is_aligned",
             "signed_distance_to_mean", "stretch_bucket", "mr_setup"]
        ].head(20).to_dict(orient="records"),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"[report] Saved to {REPORT_PATH}")

    return report


def _answer_v2_questions(df_tagged: pd.DataFrame, align_df: pd.DataFrame,
                         ks_align: dict, stats_v1: pd.DataFrame) -> dict:
    """Answer the 3 specific v2 questions."""
    answers = {}

    # Q1: STRESS_ALIGNED E vs STRESS isolado (v1)
    stress_v1_e = float(stats_v1.loc[REGIME_STRESS, "expectancy_pips"]) \
        if REGIME_STRESS in stats_v1.index else np.nan
    sa_row = align_df.loc["STRESS + ALIGNED"] if "STRESS + ALIGNED" in align_df.index else None
    stress_aligned_e = float(sa_row["expectancy_pips"]) if sa_row is not None else np.nan
    stress_aligned_n = int(sa_row["n_trades"])           if sa_row is not None else 0

    ks_sig  = ks_align.get("significant", False)
    ks_p    = ks_align.get("p_value", np.nan)
    ks_note = ks_align.get("note", "")

    if np.isnan(stress_aligned_e):
        q1 = "INCONCLUSIVO: STRESS_ALIGNED sem trades suficientes."
    elif np.isnan(stress_v1_e):
        q1 = f"STRESS_ALIGNED E={stress_aligned_e:+.2f}pips (n={stress_aligned_n}). " \
             "Sem baseline v1 para comparar."
    else:
        diff = stress_aligned_e - stress_v1_e
        direction = "MAIOR" if diff > 1 else ("SIMILAR" if abs(diff) <= 1 else "MENOR")
        sig_text  = f"KS p={ks_p:.4f} ({('SIGNIFICATIVO' if ks_sig else 'n.s.')})" \
                    if not np.isnan(ks_p) else ks_note
        q1 = (f"STRESS_ALIGNED E={stress_aligned_e:+.2f}pips (n={stress_aligned_n}) "
              f"vs STRESS_v1 E={stress_v1_e:+.2f}pips. "
              f"Delta={diff:+.2f}pips -> {direction}. {sig_text}")
    answers["STRESS_ALIGNED tem E maior que STRESS isolado da v1?"] = q1

    # Q2: PF do STRESS_NOT_ALIGNED
    sna_row = align_df.loc["STRESS + NOT_ALIGNED"] if "STRESS + NOT_ALIGNED" in align_df.index else None
    if sna_row is not None and not np.isnan(sna_row["profit_factor"]):
        pf   = float(sna_row["profit_factor"])
        n    = int(sna_row["n_trades"])
        e    = float(sna_row["expectancy_pips"])
        verdict = "positivo (edge presente)" if pf > 1 else "negativo (sem edge)"
        q2 = f"PF={pf:.2f} (n={n}, E={e:+.2f}pips) -> {verdict}"
    else:
        q2 = "INCONCLUSIVO: dados insuficientes para STRESS_NOT_ALIGNED."
    answers["Qual o PF do STRESS_NOT_ALIGNED?"] = q2

    # Q3: Quantos trades sobram com is_aligned=TRUE
    aligned_n = int(df_tagged["is_aligned"].sum())
    total_n   = len(df_tagged)
    pct       = 100 * aligned_n / total_n if total_n > 0 else 0
    aligned_e = float(df_tagged[df_tagged["is_aligned"]]["pnl_pips"].mean()) \
        if aligned_n > 0 else np.nan
    overall_e = float(df_tagged["pnl_pips"].mean())
    e_delta   = f"E_aligned={aligned_e:+.2f}pips vs E_all={overall_e:+.2f}pips" \
        if not np.isnan(aligned_e) else ""
    q3 = (f"{aligned_n} / {total_n} trades ({pct:.1f}%) passam pelo filtro is_aligned=TRUE. "
          f"{e_delta}")
    answers["Quantos trades sobram com filtro is_aligned=TRUE?"] = q3

    return answers


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
