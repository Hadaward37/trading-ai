"""
Early Entry Simulation — Fractal Lab v3 research.

Hypothetical: what if Sistema 1 entered when stretch was in B2 (0.5-1.0 ATR)
instead of waiting for BB touch confirmation at B5 (>2 ATR)?

Method (no lookahead, no modification to Sistema 1):
  1. For each actual trade, look back up to LOOKBACK_BARS hours
  2. Find the most recent bar where stretch was in B2 range
  3. Simulate the same trade direction from that earlier price
     using the same SL/TP ATR multiples (2.5x SL, 4.0x TP)
  4. Resolve outcome via forward OHLCV
  5. Compare: early_pnl vs actual_pnl

Key questions:
  - Does early entry preserve Win Rate or increase Drawdown?
  - What is the PnL improvement per trade?
  - How many trades have a valid B2 precursor within the lookback?

Output: data/early_entry_simulation.json
"""

from __future__ import annotations

import json
import sys
from datetime import timezone, datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import sqlite3

from research.fractal_lab.regime_classifier import (
    compute_features, tag_regimes, tag_stretch,
    resolve_outcomes, OUTCOME_MAX_BARS,
    STRETCH_BUCKETS,
)

SIGNALS_CSV   = _ROOT / "data" / "mt5_signals.csv"
REPORT_PATH   = _ROOT / "data" / "early_entry_simulation.json"
DB_PATH       = _ROOT / "data" / "trading.db"

SL_MULT       = 2.5
TP_MULT       = 4.0
LOOKBACK_BARS = 12    # how many hours back to search for a B2 entry point
TARGET_BUCKET = "B2 [0.5-1.0)"
SIZING_TABLE  = {
    "B1 [0.0-0.5)": 1.0,
    "B2 [0.5-1.0)": 1.0,
    "B3 [1.0-1.5)": 0.7,
    "B4 [1.5-2.0)": 0.3,
    "B5 [2.0+]":    0.0,
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_ohlcv() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT datetime, open, high, low, close, volume FROM ohlcv_1h", conn)
    conn.close()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    df.columns = [c.title() for c in df.columns]
    return df


def load_adx() -> pd.Series:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT datetime, adx FROM signals_1h", conn)
    conn.close()
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.floor("h")
    return df.dropna().set_index("datetime")["adx"]


def load_signals() -> pd.DataFrame:
    df = pd.read_csv(SIGNALS_CSV)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


# ── Early entry finder ────────────────────────────────────────────────────────

def find_early_entry(entry_dt: pd.Timestamp, signal: int,
                     df_feat: pd.DataFrame,
                     lookback: int = LOOKBACK_BARS) -> dict | None:
    """
    Look back up to `lookback` bars before entry_dt for the most recent bar
    in TARGET_BUCKET. Returns the bar info or None if not found.

    No lookahead: only considers bars strictly before entry_dt.
    """
    window_start = entry_dt - pd.Timedelta(hours=lookback)
    candidates = df_feat[
        (df_feat.index >= window_start) &
        (df_feat.index < entry_dt) &
        (df_feat["stretch_bucket"] == TARGET_BUCKET)
    ]
    if candidates.empty:
        return None

    # Use the most recent B2 bar before entry
    bar = candidates.iloc[-1]
    atr14 = float(bar["atr14"]) if bar["atr14"] > 0 else 0.0001

    early_entry_price = float(bar["Close"])
    if signal == 1:
        early_sl = early_entry_price - SL_MULT * atr14
        early_tp = early_entry_price + TP_MULT * atr14
    else:
        early_sl = early_entry_price + SL_MULT * atr14
        early_tp = early_entry_price - TP_MULT * atr14

    return {
        "early_dt":    bar.name,
        "early_entry": round(early_entry_price, 5),
        "early_sl":    round(early_sl, 5),
        "early_tp":    round(early_tp, 5),
        "early_atr":   round(atr14, 6),
        "early_stretch": round(float(bar["stretch_abs"]), 3),
        "hours_early": round((entry_dt - bar.name).total_seconds() / 3600, 1),
    }


# ── Outcome resolver (single trade) ──────────────────────────────────────────

def resolve_one(entry_dt, signal, entry, sl, tp, df_ohlcv) -> dict:
    """Forward-scan to find TP or SL hit."""
    future = df_ohlcv[df_ohlcv.index > entry_dt].head(OUTCOME_MAX_BARS)
    for bar_dt, bar in future.iterrows():
        hi, lo = float(bar["High"]), float(bar["Low"])
        if signal == 1:
            if hi >= tp:
                return {"outcome": "TP_HIT", "pnl_pips": round((tp - entry) * 10_000, 2), "at": bar_dt}
            if lo <= sl:
                return {"outcome": "SL_HIT", "pnl_pips": round((sl - entry) * 10_000, 2), "at": bar_dt}
        else:
            if lo <= tp:
                return {"outcome": "TP_HIT", "pnl_pips": round((entry - tp) * 10_000, 2), "at": bar_dt}
            if hi >= sl:
                return {"outcome": "SL_HIT", "pnl_pips": round((entry - sl) * 10_000, 2), "at": bar_dt}
    last_close = future["Close"].iloc[-1] if len(future) else entry
    pnl = ((last_close - entry) if signal == 1 else (entry - last_close)) * 10_000
    return {"outcome": "TIMEOUT", "pnl_pips": round(pnl, 2), "at": None}


# ── Per-trade sizing ──────────────────────────────────────────────────────────

def get_sizing(stretch_bucket: str) -> float:
    return SIZING_TABLE.get(stretch_bucket, 0.0)


# ── Main simulation ───────────────────────────────────────────────────────────

def run() -> dict:
    print("=" * 64)
    print("  Early Entry Simulation — Fractal Lab v3")
    print("=" * 64)

    # Load and prepare features
    df_ohlcv = load_ohlcv()
    adx      = load_adx()
    df_feat  = compute_features(df_ohlcv)
    df_feat["adx"] = adx.reindex(df_feat.index).fillna(df_feat.get("adx", np.nan))
    df_feat  = tag_regimes(df_feat)
    # tag_stretch needs a trades df; use the features index instead
    # We'll access stretch directly from df_feat since it's already computed
    print(f"[features] {len(df_feat):,} 1H bars with stretch computed")

    signals = load_signals()
    print(f"[signals]  {len(signals):,} trades to simulate")

    # First: resolve ACTUAL outcomes (re-run so we have them for comparison)
    print("\n[actual]   Resolving actual trade outcomes ...")
    actual_results = []
    for _, t in signals.iterrows():
        res = resolve_one(
            pd.to_datetime(t["datetime"]).floor("h"),
            int(t["signal"]), float(t["entry"]),
            float(t["sl"]), float(t["tp"]), df_ohlcv,
        )
        actual_results.append(res)

    signals["actual_outcome"]  = [r["outcome"]  for r in actual_results]
    signals["actual_pnl_pips"] = [r["pnl_pips"] for r in actual_results]

    # Attach stretch at actual entry
    feat_idx = df_feat.index
    def _get_feat(dt, col, default):
        dt = pd.to_datetime(dt).floor("h")
        past = df_feat[feat_idx <= dt]
        if past.empty or col not in past.columns:
            return default
        return past.iloc[-1][col]

    signals["entry_stretch_bucket"] = signals["datetime"].apply(
        lambda d: _get_feat(d, "stretch_bucket", "B5 [2.0+]")
    )
    signals["entry_stretch_abs"] = signals["datetime"].apply(
        lambda d: float(_get_feat(d, "stretch_abs", 3.0))
    )
    signals["sizing_factor"] = signals["entry_stretch_bucket"].map(SIZING_TABLE).fillna(0.0)

    # Second: find early entry for each trade
    print("[early]    Searching for B2 precursors within lookback window ...")
    early_records = []
    found, not_found = 0, 0
    for _, t in signals.iterrows():
        entry_dt = pd.to_datetime(t["datetime"]).floor("h")
        ee = find_early_entry(entry_dt, int(t["signal"]), df_feat, LOOKBACK_BARS)
        if ee:
            found += 1
            # Resolve early entry outcome
            res = resolve_one(
                pd.to_datetime(ee["early_dt"]).floor("h"),
                int(t["signal"]),
                ee["early_entry"], ee["early_sl"], ee["early_tp"],
                df_ohlcv,
            )
            early_records.append({**ee, **res, "trade_idx": t.name})
        else:
            not_found += 1
            early_records.append({"trade_idx": t.name, "found": False})

    print(f"  B2 precursor found:     {found:3d} / {len(signals)} ({100*found/len(signals):.1f}%)")
    print(f"  No B2 within {LOOKBACK_BARS}h:      {not_found:3d} / {len(signals)} ({100*not_found/len(signals):.1f}%)")

    # Build comparison table
    early_df = pd.DataFrame(early_records)
    has_early = early_df[early_df.get("found", True) != False].copy() if "found" in early_df.columns \
        else early_df[early_df["early_entry"].notna()].copy()
    has_early = early_df[early_df["early_entry"].notna()].copy() if "early_entry" in early_df.columns \
        else pd.DataFrame()

    # Merge
    signals_with_early = signals.copy()
    signals_with_early["early_pnl_pips"] = np.nan
    signals_with_early["early_outcome"]  = "NO_PRECURSOR"
    signals_with_early["hours_early"]    = np.nan
    signals_with_early["early_stretch"]  = np.nan

    for rec in early_records:
        if rec.get("found") is False or "early_entry" not in rec:
            continue
        idx = rec["trade_idx"]
        signals_with_early.at[idx, "early_pnl_pips"] = rec.get("pnl_pips", np.nan)
        signals_with_early.at[idx, "early_outcome"]  = rec.get("outcome", "N/A")
        signals_with_early.at[idx, "hours_early"]    = rec.get("hours_early", np.nan)
        signals_with_early.at[idx, "early_stretch"]  = rec.get("early_stretch", np.nan)

    # Stats: actual vs early (for trades that have a valid early entry)
    valid = signals_with_early[signals_with_early["early_pnl_pips"].notna()].copy()

    print(f"\n[compare]  Valid comparison trades: {len(valid)}")
    print()

    def _stats(pnl_series, label):
        pnl = pnl_series.dropna()
        wins = pnl[pnl > 0]
        losses = pnl[pnl <= 0]
        pf = wins.sum() / max(abs(losses.sum()), 1e-9)
        e  = pnl.mean()
        wr = len(wins) / len(pnl) if len(pnl) else 0
        cum = pnl.cumsum()
        dd = float((cum.cummax() - cum).max())
        sharpe = e / pnl.std() if pnl.std() > 0 else 0
        print(f"  {label:<22} N={len(pnl):3d}  WR={wr:.1%}  PF={pf:.2f}  "
              f"E={e:+.2f}pips  Sharpe={sharpe:.3f}  MaxDD={dd:.1f}pips")
        return {"n": len(pnl), "win_rate": round(wr, 4), "profit_factor": round(pf, 3),
                "expectancy_pips": round(e, 2), "sharpe": round(sharpe, 3),
                "max_drawdown_pips": round(dd, 2), "total_pips": round(float(pnl.sum()), 2)}

    actual_stats = _stats(valid["actual_pnl_pips"],   "Actual (B5 entry)")
    early_stats  = _stats(valid["early_pnl_pips"],    f"Early  (B2 ~{TARGET_BUCKET})")

    # Adaptive sizing stats
    print()
    sized_pnl = valid["actual_pnl_pips"] * valid["sizing_factor"]
    sizing_stats = _stats(sized_pnl, "Adaptive sizing")

    # Delta analysis
    delta_e  = early_stats["expectancy_pips"]  - actual_stats["expectancy_pips"]
    delta_wr = early_stats["win_rate"]          - actual_stats["win_rate"]
    delta_dd = early_stats["max_drawdown_pips"] - actual_stats["max_drawdown_pips"]
    delta_pf = early_stats["profit_factor"]     - actual_stats["profit_factor"]

    print(f"\n[delta]    Early vs Actual (for {len(valid)} matched trades):")
    print(f"  E delta    : {delta_e:+.2f} pips/trade")
    print(f"  WinRate d  : {delta_wr:+.1%}")
    print(f"  PF delta   : {delta_pf:+.2f}")
    print(f"  MaxDD delta: {delta_dd:+.1f} pips  ({'INCREASES' if delta_dd > 0 else 'DECREASES'})")
    print(f"  Hours early: {valid['hours_early'].mean():.1f}h avg")

    # Hours early distribution
    print(f"\n[timing]   Hours before actual entry (early entry):")
    for h_range, lo, hi in [("0-2h", 0, 2), ("2-5h", 2, 5), ("5-10h", 5, 10), (">10h", 10, 999)]:
        n = ((valid["hours_early"] >= lo) & (valid["hours_early"] < hi)).sum()
        print(f"  {h_range:6s}: {n:3d} ({100*n/len(valid):.1f}%)")

    # Adaptive sizing: current bucket distribution + PnL
    print(f"\n[sizing]   Adaptive sizing impact (all {len(signals)} trades):")
    print(f"  Table: B2=1.0x B3=0.7x B4=0.3x B5=0.0x")
    for bkt, factor in SIZING_TABLE.items():
        n = (signals["entry_stretch_bucket"] == bkt).sum()
        grp = signals[signals["entry_stretch_bucket"] == bkt]
        e_raw  = grp["actual_pnl_pips"].mean()
        e_sized = (grp["actual_pnl_pips"] * factor).mean()
        print(f"  {bkt:<20} n={n:3d}  sizing={factor:.1f}x  "
              f"E_raw={e_raw:+.2f}  E_sized={e_sized:+.2f}")

    gross_raw   = signals["actual_pnl_pips"].sum()
    gross_sized = (signals["actual_pnl_pips"] * signals["sizing_factor"]).sum()
    print(f"\n  Total PnL (raw):   {gross_raw:+.1f} pips  ({len(signals)} trades, all 1.0x)")
    print(f"  Total PnL (sized): {gross_sized:+.1f} pips  (B5 blocked, others reduced)")
    print(f"  Trades blocked (B5): {(signals['entry_stretch_bucket']=='B5 [2.0+]').sum()}")

    # Answer the key question
    print(f"\n{'='*64}")
    print("PERGUNTA: Antecipar entrada preserva Win Rate ou aumenta Drawdown?")
    wrc = "PRESERVA" if abs(delta_wr) < 0.05 else ("AUMENTA" if delta_wr > 0 else "REDUZ")
    ddc = "AUMENTA" if delta_dd > 50 else ("REDUZ" if delta_dd < -50 else "NEUTRO")
    print(f"Win Rate: {wrc} ({delta_wr:+.1%}  actual={actual_stats['win_rate']:.1%} "
          f"early={early_stats['win_rate']:.1%})")
    print(f"Drawdown: {ddc} ({delta_dd:+.1f}pips  actual={actual_stats['max_drawdown_pips']:.1f} "
          f"early={early_stats['max_drawdown_pips']:.1f})")
    print(f"E delta : {delta_e:+.2f} pips/trade  PF delta: {delta_pf:+.2f}")
    print(f"{'='*64}\n")

    # Build report
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "simulation_params": {
            "target_bucket": TARGET_BUCKET,
            "lookback_bars": LOOKBACK_BARS,
            "sl_mult": SL_MULT,
            "tp_mult": TP_MULT,
        },
        "coverage": {
            "total_trades": len(signals),
            "with_b2_precursor": found,
            "without_precursor": not_found,
            "coverage_pct": round(found / len(signals), 4),
        },
        "actual_stats":  actual_stats,
        "early_stats":   early_stats,
        "sizing_stats":  sizing_stats,
        "deltas": {
            "expectancy_pips": round(delta_e,  2),
            "win_rate":        round(delta_wr, 4),
            "profit_factor":   round(delta_pf, 3),
            "max_drawdown":    round(delta_dd, 2),
            "avg_hours_early": round(float(valid["hours_early"].mean()), 1),
        },
        "adaptive_sizing": {
            "table": SIZING_TABLE,
            "total_pnl_raw":   round(float(gross_raw),   2),
            "total_pnl_sized": round(float(gross_sized), 2),
            "trades_blocked":  int((signals["entry_stretch_bucket"] == "B5 [2.0+]").sum()),
        },
        "answer": (
            f"Win Rate {wrc} ({delta_wr:+.1%}), "
            f"Drawdown {ddc} ({delta_dd:+.1f}pips), "
            f"E delta {delta_e:+.2f}pips/trade. "
            f"Coverage: {found}/{len(signals)} trades ({100*found/len(signals):.1f}%) "
            f"have a B2 precursor within {LOOKBACK_BARS}h."
        ),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"[report] Saved to {REPORT_PATH}")
    return report


if __name__ == "__main__":
    run()
