"""Fold1 Diagnosis — why did the Fold1 regime fail?

Compares market characteristics between:
  TOXIC   : Fold1 test period (B2: ~2024-09 to ~2025-02)  — E<0 in all combos
  POSITIVE: Holdout period    (B5: ~2025-12 to ~2026-05) — E>0, Sharpe>6

Metrics per period:
  1. Average trend duration  (consecutive same-direction candles)
  2. % time ADX > 35         (strong-trend filter trigger rate)
  3. EMA50 slope             (directional bias, pips/candle)
  4. EMA200 slope            (long-term bias, pips/candle)
  5. Max unidirectional move (max consecutive pip gain one-way)
  6. Volume by session       (Asia / London / NY)
  7. Mean atr_ratio          (volatility regime vs 20-bar norm)
  8. Price range / day       (daily H-L spread)
  9. Reversal frequency      (fraction of candles that reverse direction)

Output: comparative table + root cause analysis.

Usage: python scripts/fold1_diagnosis.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from core.indicators import add_all_indicators

LINE = "=" * 72
SEP  = "-" * 72


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    conn = sqlite3.connect(config.DB_PATH)
    df   = pd.read_sql_query(
        "SELECT * FROM ohlcv_1h ORDER BY datetime ASC",
        conn, index_col="datetime", parse_dates=["datetime"],
    )
    conn.close()
    df.index = pd.to_datetime(df.index, utc=True)
    df = add_all_indicators(df)

    # EMA features
    df["ema50"]  = df["close"].ewm(span=50,  adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    # atr_ratio
    atr_ma20        = df["atr"].rolling(20, min_periods=1).mean().replace(0, 1e-9)
    df["atr_ratio"] = df["atr"] / atr_ma20

    # pip-level change
    df["chg_pips"] = (df["close"] - df["close"].shift(1)) * 10_000  # EUR/USD in pips
    df["direction"] = np.sign(df["chg_pips"])   # +1, -1, 0

    return df.dropna()


def get_blocks(df: pd.DataFrame) -> dict[str, tuple]:
    """Reproduce the 5-block structure from experiment_definitive."""
    d_min, d_max = df.index.min(), df.index.max()
    total_td = d_max - d_min
    delta    = total_td / 5
    cuts     = [d_min + delta * i for i in range(6)]
    return {
        "B1_toxic_train": (cuts[0], cuts[1]),   # 2024-05 → 2024-09
        "B2_toxic_test":  (cuts[1], cuts[2]),   # 2024-09 → 2025-02  ← Fold1 TEST
        "B3":             (cuts[2], cuts[3]),
        "B4":             (cuts[3], cuts[4]),
        "B5_holdout":     (cuts[4], cuts[5]),   # 2025-12 → 2026-05  ← Holdout
    }


# ── Metric computation ────────────────────────────────────────────────────────

def avg_trend_duration(df: pd.DataFrame) -> float:
    """Average consecutive candle count in the same direction (trend run length)."""
    dirs = df["direction"].values
    streaks: list[int] = []
    streak = 1
    for i in range(1, len(dirs)):
        if dirs[i] == dirs[i - 1] and dirs[i] != 0:
            streak += 1
        else:
            if streak > 1:
                streaks.append(streak)
            streak = 1
    return float(np.mean(streaks)) if streaks else 1.0


def pct_adx_above(df: pd.DataFrame, threshold: float = 35.0) -> float:
    return float((df["adx"] > threshold).mean() * 100)


def ema_slope(df: pd.DataFrame, col: str = "ema50") -> float:
    """Average per-candle slope in pips."""
    return float(df[col].diff().mean() * 10_000)


def max_unidirectional_move(df: pd.DataFrame) -> float:
    """Maximum consecutive same-direction cumulative move in pips."""
    cumulative = 0.0
    best       = 0.0
    prev_dir   = 0
    for chg, d in zip(df["chg_pips"].values, df["direction"].values):
        if d == prev_dir and d != 0:
            cumulative += abs(chg)
            best = max(best, cumulative)
        else:
            cumulative = abs(chg) if d != 0 else 0.0
            prev_dir   = d
    return float(best)


def volume_by_session(df: pd.DataFrame) -> dict[str, float]:
    hour = df.index.hour
    return {
        "Asia":    float(df.loc[(hour >= 0)  & (hour <  8),  "volume"].mean()),
        "London":  float(df.loc[(hour >= 8)  & (hour < 17),  "volume"].mean()),
        "NY":      float(df.loc[(hour >= 13) & (hour < 21),  "volume"].mean()),
        "Overlap": float(df.loc[(hour >= 13) & (hour < 17),  "volume"].mean()),
    }


def daily_range_pips(df: pd.DataFrame) -> float:
    """Mean daily high-low range in pips."""
    daily = df["high"].resample("1D").max() - df["low"].resample("1D").min()
    return float(daily.mean() * 10_000)


def reversal_frequency(df: pd.DataFrame) -> float:
    """Fraction of candles that reverse the previous direction."""
    dirs = df["direction"].values
    reversals = sum(
        1 for i in range(1, len(dirs))
        if dirs[i] != 0 and dirs[i - 1] != 0 and dirs[i] != dirs[i - 1]
    )
    total = sum(1 for d in dirs if d != 0)
    return float(reversals / max(total, 1) * 100)


def compute_all_metrics(df: pd.DataFrame) -> dict:
    """Compute all 9 diagnostic metrics for a period."""
    vol = volume_by_session(df)
    return {
        "n_candles":         len(df),
        "avg_trend_dur":     avg_trend_duration(df),
        "pct_adx_above_35":  pct_adx_above(df, 35),
        "ema50_slope_pips":  ema_slope(df, "ema50"),
        "ema200_slope_pips": ema_slope(df, "ema200"),
        "max_uni_move_pips": max_unidirectional_move(df),
        "vol_asia":          vol["Asia"],
        "vol_london":        vol["London"],
        "vol_ny":            vol["NY"],
        "mean_atr_ratio":    float(df["atr_ratio"].mean()),
        "daily_range_pips":  daily_range_pips(df),
        "reversal_freq_pct": reversal_frequency(df),
    }


# ── Report ────────────────────────────────────────────────────────────────────

def _flag(toxic, holdout, higher_is_bad: bool = True) -> str:
    """Return a symbol showing which direction is the diagnosis."""
    if higher_is_bad:
        return "TOXIC" if toxic > holdout * 1.10 else "similar"
    else:
        return "TOXIC" if toxic < holdout * 0.90 else "similar"


def print_report(
    m_fold1:   dict,
    m_holdout: dict,
    fold1_range: str,
    holdout_range: str,
) -> None:
    print(f"\n{LINE}")
    print("  FOLD1 DIAGNOSIS — Why Did Fold1 Fail?")
    print(f"  Toxic period  : {fold1_range}  (Fold1 test = B2)")
    print(f"  Positive period: {holdout_range}  (Holdout = B5)")
    print(LINE)

    rows = [
        ("n_candles",         "Bars",           "",      "{:.0f}",  False),
        ("avg_trend_dur",     "Trend duration", "candles", "{:.2f}", True),
        ("pct_adx_above_35",  "ADX>35 freq",    "%",     "{:.1f}",  True),
        ("ema50_slope_pips",  "EMA50 slope",    "pips/c","{:+.3f}", False),
        ("ema200_slope_pips", "EMA200 slope",   "pips/c","{:+.3f}", False),
        ("max_uni_move_pips", "Max uni-move",   "pips",  "{:.1f}",  True),
        ("vol_asia",          "Volume Asia",    "avg",   "{:.0f}",  False),
        ("vol_london",        "Volume London",  "avg",   "{:.0f}",  False),
        ("vol_ny",            "Volume NY",      "avg",   "{:.0f}",  False),
        ("mean_atr_ratio",    "ATR ratio mean", "x",     "{:.3f}",  True),
        ("daily_range_pips",  "Daily range",    "pips",  "{:.1f}",  True),
        ("reversal_freq_pct", "Reversal freq",  "%",     "{:.1f}",  False),
    ]

    print(f"\n  {'Metric':<24} {'Unit':<8} {'Fold1':>10} {'Holdout':>10}  {'Delta':>9}  Diagnosis")
    print(f"  {'-'*80}")

    for key, label, unit, fmt, higher_is_bad in rows:
        t = m_fold1[key]
        h = m_holdout[key]
        delta = t - h
        diag  = _flag(t, h, higher_is_bad)
        print(
            f"  {label:<24} {unit:<8} "
            f"{fmt.format(t):>10} {fmt.format(h):>10}  {delta:>+9.2f}  "
            f"{'** ' + diag if diag == 'TOXIC' else diag}"
        )

    # Root-cause summary
    print(f"\n{SEP}")
    print("  ROOT CAUSE ANALYSIS")
    print(SEP)

    findings = []

    if m_fold1["pct_adx_above_35"] > m_holdout["pct_adx_above_35"] * 1.2:
        diff = m_fold1["pct_adx_above_35"] - m_holdout["pct_adx_above_35"]
        findings.append(
            f"  [1] STRONGER TREND (ADX>35): Fold1 had {m_fold1['pct_adx_above_35']:.1f}% "
            f"vs Holdout {m_holdout['pct_adx_above_35']:.1f}% (+{diff:.1f}pp).\n"
            f"      Model trained on range/oscillation patterns — struggles in strong trends.\n"
            f"      -> This is what ADX<35 filter is designed to block."
        )

    if m_fold1["avg_trend_dur"] > m_holdout["avg_trend_dur"] * 1.15:
        findings.append(
            f"  [2] LONGER TREND RUNS: Fold1 avg {m_fold1['avg_trend_dur']:.1f} consecutive "
            f"candles vs Holdout {m_holdout['avg_trend_dur']:.1f}.\n"
            f"      Reversal-based signals (RSI oversold + BB touch) fail in persistent trends."
        )

    if abs(m_fold1["ema200_slope_pips"]) > abs(m_holdout["ema200_slope_pips"]) * 1.3:
        dir_fold1   = "upward" if m_fold1["ema200_slope_pips"]   > 0 else "downward"
        dir_holdout = "upward" if m_holdout["ema200_slope_pips"] > 0 else "downward"
        findings.append(
            f"  [3] DIRECTIONAL BIAS: EMA200 slope Fold1={m_fold1['ema200_slope_pips']:+.3f} pips "
            f"({dir_fold1}) vs Holdout={m_holdout['ema200_slope_pips']:+.3f} ({dir_holdout}).\n"
            f"      Strong EMA bias penalizes counter-trend signals."
        )

    if m_fold1["max_uni_move_pips"] > m_holdout["max_uni_move_pips"] * 1.2:
        findings.append(
            f"  [4] LARGER DIRECTIONAL SWINGS: Fold1 max {m_fold1['max_uni_move_pips']:.0f} pips "
            f"vs Holdout {m_holdout['max_uni_move_pips']:.0f} pips.\n"
            f"      Large moves create adverse entries and stop-outs."
        )

    if m_fold1["reversal_freq_pct"] < m_holdout["reversal_freq_pct"] * 0.9:
        findings.append(
            f"  [5] LOWER REVERSAL FREQUENCY: Fold1 {m_fold1['reversal_freq_pct']:.1f}% "
            f"vs Holdout {m_holdout['reversal_freq_pct']:.1f}%.\n"
            f"      Mean-reversion model needs frequent reversals to generate profitable signals."
        )

    atr_diff = m_fold1["mean_atr_ratio"] - m_holdout["mean_atr_ratio"]
    if abs(atr_diff) > 0.1:
        direction = "elevated" if atr_diff > 0 else "depressed"
        findings.append(
            f"  [6] ATR RATIO {'ELEVATED' if atr_diff > 0 else 'DEPRESSED'}: "
            f"Fold1={m_fold1['mean_atr_ratio']:.3f} vs Holdout={m_holdout['mean_atr_ratio']:.3f} "
            f"(diff={atr_diff:+.3f}).\n"
            f"      {'Elevated ATR inflates thresholds, reducing labeled samples.' if atr_diff > 0 else 'Depressed ATR = low volatility, wide neutral zone.'}"
        )

    if findings:
        for f in findings:
            print(f"\n{f}")
    else:
        print("\n  No statistically significant differences found.")
        print("  The toxic period may be due to regime shift not captured by these metrics.")

    print(f"\n{SEP}")
    print("  CONCLUSION")
    print(SEP)
    top_issues = [f.split('[')[1].split(']')[0] for f in findings[:3]]
    if top_issues:
        print(f"\n  Primary failure modes in Fold1:")
        for issue in top_issues:
            labels = {
                "1": "Strong persistent trends (ADX>35) — model trained for oscillation",
                "2": "Long trend runs — mean-reversion signals fail in momentum",
                "3": "Strong directional EMA bias — counter-trend entries punished",
                "4": "Large directional swings — stop-outs before target",
                "5": "Low reversal frequency — fewer valid entry opportunities",
                "6": "Abnormal ATR ratio — volatility regime mismatch",
            }
            print(f"    [{issue}] {labels.get(issue, '')}")
        print(f"\n  Action: the ADX<35 + atr_ratio=(0.8,1.5) filter directly addresses "
              f"issues [{', '.join(top_issues)}].")
    else:
        print("\n  No clear structural differences — further analysis needed.")

    print(f"\n{LINE}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\nLoading data...")
    df = load_data()
    blocks = get_blocks(df)

    # Fold1 test period = B2 (the period where Fold1 model was evaluated)
    f1_start, f1_end = blocks["B2_toxic_test"]
    ho_start, ho_end = blocks["B5_holdout"]

    df_fold1   = df.loc[(df.index >= f1_start) & (df.index < f1_end)]
    df_holdout = df.loc[(df.index >= ho_start) & (df.index < ho_end)]

    fold1_range   = f"{f1_start.strftime('%Y-%m-%d')} to {f1_end.strftime('%Y-%m-%d')}"
    holdout_range = f"{ho_start.strftime('%Y-%m-%d')} to {ho_end.strftime('%Y-%m-%d')}"

    print(f"  Fold1 test period : {fold1_range} ({len(df_fold1):,} bars)")
    print(f"  Holdout period    : {holdout_range} ({len(df_holdout):,} bars)")

    print("  Computing metrics...")
    m_fold1   = compute_all_metrics(df_fold1)
    m_holdout = compute_all_metrics(df_holdout)

    print_report(m_fold1, m_holdout, fold1_range, holdout_range)


if __name__ == "__main__":
    main()
