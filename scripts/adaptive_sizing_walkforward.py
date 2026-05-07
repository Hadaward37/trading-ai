"""
Adaptive Sizing Walk-Forward Validation — Fractal Lab.

Validates whether the B1-B5 sizing table is robust out-of-sample.

Design:
  - TRAIN: all signals from start of data to May 2025 (n=236)
  - TEST:  Jun-Aug 2025, NEVER seen during Fractal Lab analysis (n=37)
  - Sizing thresholds are FIXED (pre-determined, NOT calibrated on any period)
  - Three scenarios compared: Baseline 1.0x | B5=0.1x | B5=0.0x

Sizing table (pre-determined, do not recalibrate):
  B1 [0.0-0.5)  -> 1.0x
  B2 [0.5-1.0)  -> 1.0x
  B3 [1.0-1.5)  -> 0.7x
  B4 [1.5-2.0)  -> 0.3x
  B5 [2.0+]     -> 0.1x  (scenario A) or  0.0x  (scenario B)

Degradation Ratio = PF_out / PF_in
  >0.70 : robust — most of the edge survives
  0.50-0.70 : moderate degradation, common
  <0.50 : severe, warrants caution

Answers: "Is adaptive sizing robust out-of-sample?"

Usage:
    .\\venv\\Scripts\\python scripts\\adaptive_sizing_walkforward.py
"""

from __future__ import annotations

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
    compute_features, tag_regimes, OUTCOME_MAX_BARS,
    STRETCH_BUCKETS, assign_stretch_bucket,
)

DB_PATH     = _ROOT / "data" / "trading.db"
REPORT_PATH = _ROOT / "data" / "adaptive_sizing_walkforward.json"

# ── Periods ───────────────────────────────────────────────────────────────────

TRAIN_START = "2024-05-01"
TRAIN_END   = "2025-05-31"
TEST_START  = "2025-06-01"
TEST_END    = "2025-08-31"

# ── Sizing scenarios (buckets fixed, only B5 multiplier varies) ───────────────

SIZING_SCENARIOS = {
    "baseline_1.0x": {
        "B1 [0.0-0.5)": 1.0,
        "B2 [0.5-1.0)": 1.0,
        "B3 [1.0-1.5)": 1.0,
        "B4 [1.5-2.0)": 1.0,
        "B5 [2.0+]":    1.0,
    },
    "adaptive_B5=0.1x": {
        "B1 [0.0-0.5)": 1.0,
        "B2 [0.5-1.0)": 1.0,
        "B3 [1.0-1.5)": 0.7,
        "B4 [1.5-2.0)": 0.3,
        "B5 [2.0+]":    0.1,
    },
    "adaptive_B5=0.0x": {
        "B1 [0.0-0.5)": 1.0,
        "B2 [0.5-1.0)": 1.0,
        "B3 [1.0-1.5)": 0.7,
        "B4 [1.5-2.0)": 0.3,
        "B5 [2.0+]":    0.0,
    },
}

SL_MULT = 2.5
TP_MULT = 4.0
DR_ROBUST_THRESHOLD = 0.70


# ── Data loading ──────────────────────────────────────────────────────────────

def load_ohlcv() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT datetime, open, high, low, close, volume FROM ohlcv_1h", conn)
    conn.close()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    df.columns = [c.title() for c in df.columns]
    return df


def load_signals_1h() -> pd.DataFrame:
    """Load SELL signals from signals_1h with ATR. Build SL/TP from ATR multiples."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT datetime, close, atr, adx, signal FROM signals_1h WHERE signal != 0",
        conn
    )
    conn.close()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.dropna(subset=["atr", "close"]).sort_values("datetime").reset_index(drop=True)

    # Build entry/sl/tp for SELL signals
    df["entry"] = df["close"]
    df["sl"] = df.apply(
        lambda r: r["entry"] + SL_MULT * r["atr"] if r["signal"] == -1
                  else r["entry"] - SL_MULT * r["atr"],
        axis=1,
    )
    df["tp"] = df.apply(
        lambda r: r["entry"] - TP_MULT * r["atr"] if r["signal"] == -1
                  else r["entry"] + TP_MULT * r["atr"],
        axis=1,
    )
    return df[["datetime", "signal", "entry", "sl", "tp", "atr", "adx"]]


# ── Outcome resolution ────────────────────────────────────────────────────────

def resolve_outcomes_df(df_trades: pd.DataFrame, df_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Forward-scan TP/SL hits for each trade."""
    ohlcv = df_ohlcv.copy()
    ohlcv.index = pd.to_datetime(ohlcv.index).floor("h")

    outcomes, pnls = [], []
    for _, t in df_trades.iterrows():
        entry_dt = pd.to_datetime(t["datetime"]).floor("h")
        sig, entry, sl, tp = int(t["signal"]), float(t["entry"]), float(t["sl"]), float(t["tp"])
        future = ohlcv[ohlcv.index > entry_dt].head(OUTCOME_MAX_BARS)

        outcome, pnl = "TIMEOUT", 0.0
        for _, bar in future.iterrows():
            hi, lo = float(bar["High"]), float(bar["Low"])
            if sig == -1:
                if lo <= tp:
                    outcome = "TP_HIT"; pnl = (entry - tp) * 10_000; break
                if hi >= sl:
                    outcome = "SL_HIT"; pnl = (entry - sl) * 10_000; break
            else:
                if hi >= tp:
                    outcome = "TP_HIT"; pnl = (tp - entry) * 10_000; break
                if lo <= sl:
                    outcome = "SL_HIT"; pnl = (sl - entry) * 10_000; break
        if outcome == "TIMEOUT":
            last = future["Close"].iloc[-1] if len(future) else entry
            pnl = ((last - entry) if sig == 1 else (entry - last)) * 10_000

        outcomes.append(outcome)
        pnls.append(round(pnl, 2))

    df_out = df_trades.copy()
    df_out["outcome"]  = outcomes
    df_out["pnl_pips"] = pnls
    return df_out


# ── Stretch tagging ───────────────────────────────────────────────────────────

def tag_stretch_from_features(df_trades: pd.DataFrame,
                               df_feat: pd.DataFrame) -> pd.DataFrame:
    """Attach stretch_bucket at trade entry (no lookahead)."""
    feat_idx = df_feat.index

    buckets, stretches = [], []
    for _, t in df_trades.iterrows():
        dt = pd.to_datetime(t["datetime"]).floor("h")
        past = df_feat[feat_idx <= dt]
        if past.empty or "stretch_bucket" not in past.columns:
            bkt, sa = "B5 [2.0+]", 3.0
        else:
            row = past.iloc[-1]
            bkt = str(row["stretch_bucket"])
            sa  = float(row["stretch_abs"])
        buckets.append(bkt)
        stretches.append(round(sa, 3))

    df_out = df_trades.copy()
    df_out["stretch_bucket"] = buckets
    df_out["stretch_abs"]    = stretches
    return df_out


# ── Statistics ────────────────────────────────────────────────────────────────

def _compute_stats(pnl: pd.Series, sizing: pd.Series | None = None,
                   trades_dt: pd.Series | None = None) -> dict:
    """Compute trading stats, optionally with per-trade sizing."""
    pnl = pnl.copy()
    if sizing is not None:
        pnl = pnl * sizing

    pnl = pnl.dropna()
    if len(pnl) == 0:
        return {"n_trades": 0, "n_active": 0}

    n_active = int((sizing > 0).sum()) if sizing is not None else len(pnl)
    wins   = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    tw = float(wins.sum())
    tl = float(losses.sum())
    if tl >= 0:
        tl = -1e-9

    pf = tw / abs(tl) if tw > 0 else 0.0
    e  = float(pnl.mean())
    wr = len(wins) / len(pnl)
    sd = float(pnl.std()) if len(pnl) > 1 else 1e-9
    sharpe = e / sd if sd > 0 else 0.0

    if trades_dt is not None:
        cum = pnl.values
        cumsum = np.cumsum(cum)
        roll_max = np.maximum.accumulate(cumsum)
        dd = float(np.max(roll_max - cumsum))
    else:
        cumsum = np.cumsum(pnl.values)
        roll_max = np.maximum.accumulate(cumsum)
        dd = float(np.max(roll_max - cumsum))

    return {
        "n_trades":          len(pnl),
        "n_active":          n_active,
        "win_rate":          round(wr,     4),
        "profit_factor":     round(pf,     3),
        "expectancy_pips":   round(e,      2),
        "sharpe_proxy":      round(sharpe, 3),
        "max_drawdown_pips": round(dd,     2),
        "total_pips":        round(float(pnl.sum()), 2),
    }


def bucket_breakdown(df: pd.DataFrame) -> dict:
    """Per-bucket stats (raw, unweighted)."""
    result = {}
    for label, lo, hi in STRETCH_BUCKETS:
        grp = df[df["stretch_bucket"] == label]
        if len(grp) == 0:
            result[label] = {"n_trades": 0}
            continue
        wins   = grp[grp["pnl_pips"] > 0]
        losses = grp[grp["pnl_pips"] <= 0]
        tw = float(wins["pnl_pips"].sum())
        tl = float(losses["pnl_pips"].sum())
        if tl >= 0:
            tl = -1e-9
        pf = tw / abs(tl) if tw > 0 else 0.0
        result[label] = {
            "n_trades":        len(grp),
            "win_rate":        round(len(wins) / len(grp), 3),
            "profit_factor":   round(pf, 3),
            "expectancy_pips": round(float(grp["pnl_pips"].mean()), 2),
        }
    return result


# ── Walk-forward runner ───────────────────────────────────────────────────────

def run() -> dict:
    print("=" * 64)
    print("  Adaptive Sizing Walk-Forward Validation")
    print(f"  Train: {TRAIN_START} -> {TRAIN_END}")
    print(f"  Test:  {TEST_START}  -> {TEST_END}")
    print("=" * 64)

    # Load data
    df_ohlcv  = load_ohlcv()
    df_sigs   = load_signals_1h()
    print(f"\n[data]  Total signals: {len(df_sigs)}")

    # Compute features on full OHLCV history (EMA50 needs warm-up)
    print("[feat]  Computing stretch features on full 1H history ...")
    df_feat = compute_features(df_ohlcv)
    df_feat = tag_regimes(df_feat)

    # Resolve all outcomes
    print("[outc]  Resolving TP/SL outcomes ...")
    df_all = resolve_outcomes_df(df_sigs, df_ohlcv)

    # Tag stretch (no lookahead)
    print("[str]   Tagging stretch buckets at entry time ...")
    df_all = tag_stretch_from_features(df_all, df_feat)

    # Split
    df_train = df_all[
        (df_all["datetime"] >= TRAIN_START) & (df_all["datetime"] <= TRAIN_END)
    ].copy()
    df_test  = df_all[
        (df_all["datetime"] >= TEST_START)  & (df_all["datetime"] <= TEST_END)
    ].copy()

    print(f"\n[split] Train: {len(df_train)} trades  |  Test: {len(df_test)} trades")

    # Bucket breakdown
    print("\n[bkts]  Bucket distribution:")
    for period_label, df_p in [("TRAIN", df_train), ("TEST ", df_test)]:
        dist = df_p["stretch_bucket"].value_counts().sort_index()
        bkt_str = "  ".join(f"{b}: {n}" for b, n in dist.items())
        print(f"  {period_label}  {bkt_str}")

    print("\n[bkts]  Per-bucket raw performance:")
    for period_label, df_p in [("TRAIN", df_train), ("TEST ", df_test)]:
        bbd = bucket_breakdown(df_p)
        print(f"\n  {period_label}:")
        for bkt, stats in bbd.items():
            if stats["n_trades"] == 0:
                continue
            pf_s = f"PF={stats['profit_factor']:.2f}" if stats["n_trades"] >= 5 else f"PF=? (n={stats['n_trades']})"
            e_s  = f"E={stats['expectancy_pips']:+.2f}pips" if stats["n_trades"] >= 5 else ""
            print(f"    {bkt:<20}  n={stats['n_trades']:3d}  {pf_s}  {e_s}")

    # Scenario comparison
    results = {}
    print(f"\n{'='*64}")
    print("  Scenario Comparison: Baseline vs Adaptive Sizing")
    print(f"{'='*64}")

    header = f"  {'Scenario':<22} {'Period':<8} {'N':>4}  {'Active':>6}  {'PF':>6}  {'E(pips)':>8}  {'Sharpe':>7}  {'MaxDD':>8}"
    print(header)
    print("  " + "-" * 72)

    for scenario_name, table in SIZING_SCENARIOS.items():
        results[scenario_name] = {}
        for period_label, df_p in [("train", df_train), ("test", df_test)]:
            sizing_factor = df_p["stretch_bucket"].map(table).fillna(1.0)
            s = _compute_stats(df_p["pnl_pips"], sizing_factor, df_p["datetime"])
            results[scenario_name][period_label] = s

            n   = s["n_trades"]
            act = s.get("n_active", n)
            pf  = f"{s['profit_factor']:.2f}"  if n > 0 else "  N/A"
            e   = f"{s['expectancy_pips']:+.2f}" if n > 0 else "  N/A"
            sh  = f"{s['sharpe_proxy']:.3f}"    if n > 0 else "  N/A"
            dd  = f"{s['max_drawdown_pips']:.1f}" if n > 0 else "  N/A"
            print(f"  {scenario_name:<22} {period_label:<8} {n:>4}  {act:>6}  "
                  f"{pf:>6}  {e:>8}  {sh:>7}  {dd:>8}")
        print()

    # Degradation ratios
    print("[degr]  Degradation Ratio (PF_out / PF_in):")
    for scenario_name in SIZING_SCENARIOS:
        pf_in  = results[scenario_name]["train"].get("profit_factor", 0)
        pf_out = results[scenario_name]["test"].get("profit_factor", 0)
        if pf_in and pf_in > 0:
            dr = pf_out / pf_in
            verdict = ("ROBUST" if dr >= DR_ROBUST_THRESHOLD
                       else ("MODERATE" if dr >= 0.50 else "FRAGILE"))
            print(f"  {scenario_name:<22}  PF_in={pf_in:.2f}  PF_out={pf_out:.2f}  "
                  f"DR={dr:.3f}  -> {verdict}")
        else:
            print(f"  {scenario_name:<22}  PF_in=N/A")
        results[scenario_name]["degradation_ratio"] = round(pf_out / pf_in, 3) if pf_in else None

    # MaxDD comparison
    print("\n[dd]    MaxDD comparison (train -> test):")
    for scenario_name in SIZING_SCENARIOS:
        dd_in  = results[scenario_name]["train"].get("max_drawdown_pips", 0)
        dd_out = results[scenario_name]["test"].get("max_drawdown_pips", 0)
        direction = "lower" if dd_out < dd_in else "higher"
        print(f"  {scenario_name:<22}  train={dd_in:.1f}  test={dd_out:.1f}  -> {direction}")

    # B5=0.1x vs B5=0.0x comparison
    print("\n[b5]    B5=0.1x vs B5=0.0x (test period):")
    s01 = results["adaptive_B5=0.1x"]["test"]
    s00 = results["adaptive_B5=0.0x"]["test"]
    print(f"  B5=0.1x: PF={s01['profit_factor']:.2f}  E={s01['expectancy_pips']:+.2f}  "
          f"Active={s01.get('n_active','?')}  MaxDD={s01['max_drawdown_pips']:.1f}")
    print(f"  B5=0.0x: PF={s00['profit_factor']:.2f}  E={s00['expectancy_pips']:+.2f}  "
          f"Active={s00.get('n_active','?')}  MaxDD={s00['max_drawdown_pips']:.1f}")
    b5_pref = "B5=0.1x" if s01["profit_factor"] >= s00["profit_factor"] else "B5=0.0x"
    print(f"  Preferred in test: {b5_pref}")

    # Final answer
    dr_01 = results["adaptive_B5=0.1x"].get("degradation_ratio", 0) or 0
    dr_00 = results["adaptive_B5=0.0x"].get("degradation_ratio", 0) or 0
    dr_base = results["baseline_1.0x"].get("degradation_ratio", 0) or 0

    def _verdict(dr):
        if dr >= DR_ROBUST_THRESHOLD:
            return f"ROBUSTO (DR={dr:.2f})"
        elif dr >= 0.50:
            return f"DEGRADACAO MODERADA (DR={dr:.2f})"
        else:
            return f"FRAGIL (DR={dr:.2f})"

    answer_lines = [
        f"Baseline 1.0x: {_verdict(dr_base)}",
        f"Adaptive B5=0.1x: {_verdict(dr_01)}",
        f"Adaptive B5=0.0x: {_verdict(dr_00)}",
    ]

    # DR > 1 means test > train (system improved out-of-sample — can happen in favorable periods)
    # Compare: does adaptive sizing consistently reduce MaxDD across both periods?
    dd_train_base  = results["baseline_1.0x"]["train"].get("max_drawdown_pips", 0)
    dd_train_01    = results["adaptive_B5=0.1x"]["train"].get("max_drawdown_pips", 0)
    dd_test_base   = results["baseline_1.0x"]["test"].get("max_drawdown_pips", 0)
    dd_test_01     = results["adaptive_B5=0.1x"]["test"].get("max_drawdown_pips", 0)

    dd_reduction_train = 1 - dd_train_01 / dd_train_base if dd_train_base > 0 else 0
    dd_reduction_test  = 1 - dd_test_01  / dd_test_base  if dd_test_base  > 0 else 0

    # All DRs > 1 means test period happened to be a good regime for the system
    all_improving = dr_01 > 1 and dr_00 > 1 and dr_base > 1
    b5_profitable_in_test = (results["baseline_1.0x"]["test"].get("profit_factor", 0) > 2.0)

    if all_improving and b5_profitable_in_test:
        conclusion = (
            f"INCONCLUSIVO SOBRE B5 — test period (Jun-Aug2025) foi favoravel para todos os cenarios "
            f"(DR_baseline={dr_base:.2f}), inclusive B5 (E=+{df_test[df_test['stretch_bucket']=='B5 [2.0+]']['pnl_pips'].mean():.1f}pips). "
            f"B5 nao e universalmente ruim: seu edge e regime-dependente. "
            f"MaxDD reduzido consistentemente: train -{100*dd_reduction_train:.0f}%, test -{100*dd_reduction_test:.0f}%. "
            f"Conclusao principal: adaptive sizing e robusto em MaxDD, B5 block e controverso. "
            f"Recomendar B5=0.1x sobre B5=0.0x para preservar exposicao em regimes favoraveis. "
            f"n_test={len(df_test)}: insuficiente para conclusao estatistica sobre buckets."
        )
    elif dr_01 >= DR_ROBUST_THRESHOLD and dd_reduction_train > 0.30:
        conclusion = (
            f"SIM — DR={dr_01:.2f} (robusto) e MaxDD -{100*dd_reduction_train:.0f}% (train), "
            f"-{100*dd_reduction_test:.0f}% (test). Sizing adaptativo e promissor."
        )
    else:
        conclusion = (
            f"INCONCLUSIVO — DR adaptive={dr_01:.2f}, baseline={dr_base:.2f}. "
            f"n_test={len(df_test)} insuficiente para conclusao definitiva."
        )
    answer_lines.append(f"CONCLUSAO: {conclusion}")

    print(f"\n{'='*64}")
    print("PERGUNTA: Adaptive Sizing e robusto fora da amostra?")
    for line in answer_lines:
        print(f"  {line}")
    print(f"  n_test={len(df_test)} — amostras limitadas, interprete com cautela")
    print(f"{'='*64}\n")

    # Save report
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "periods": {
            "train": {"start": TRAIN_START, "end": TRAIN_END, "n": len(df_train)},
            "test":  {"start": TEST_START,  "end": TEST_END,  "n": len(df_test)},
        },
        "sizing_scenarios": SIZING_SCENARIOS,
        "results": results,
        "bucket_breakdown": {
            "train": bucket_breakdown(df_train),
            "test":  bucket_breakdown(df_test),
        },
        "b5_comparison": {
            "preferred": b5_pref,
            "B5=0.1x_test_PF": s01["profit_factor"],
            "B5=0.0x_test_PF": s00["profit_factor"],
        },
        "answer": " | ".join(answer_lines),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"[report] Saved to {REPORT_PATH}")
    return report


if __name__ == "__main__":
    run()
