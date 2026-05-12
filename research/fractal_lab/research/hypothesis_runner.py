"""
Hypothesis Runner — executes one or more hypotheses and prints a summary.

Usage (from project root):
  python -m research.fractal_lab.research.hypothesis_runner

Or import and call directly:
  from research.fractal_lab.research.hypothesis_runner import run_example
  run_example()
"""
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

# Ensure project root is in path when run as __main__
_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from research.fractal_lab.research.hypothesis import build_hypothesis
from research.fractal_lab.core.orchestrator import Orchestrator
from research.fractal_lab.storage.models import ResearchReport


def print_report(report: ResearchReport) -> None:
    print("\n" + "=" * 60)
    print(f"RESEARCH REPORT — {report.hypothesis_name}")
    print(f"Asset: {report.asset}  |  TF: {report.timeframe}")
    print(f"Created: {report.created_at}")
    print("=" * 60)

    for period_result in [report.train, report.test, report.oos]:
        if period_result is None:
            continue
        m = period_result.metrics
        print(f"\n[{period_result.period_type.upper()}] {period_result.start} -> {period_result.end}")
        print(f"  Trades:   {period_result.n_trades}")
        print(f"  PF:       {m.get('profit_factor', 0):.3f}")
        print(f"  Win Rate: {m.get('win_rate', 0):.1%}")
        print(f"  E[pips]:  {m.get('expectancy_pips', 0):+.2f}")
        print(f"  Sharpe:   {m.get('sharpe', 0):.3f}")
        print(f"  Max DD:   {m.get('max_drawdown_pips', 0):.1f} pips")

        if period_result.regime_breakdown:
            print("  Regime breakdown:")
            for regime, rm in period_result.regime_breakdown.items():
                print(
                    f"    {regime}: n={rm['n_trades']} | "
                    f"PF={rm['profit_factor']:.2f} | "
                    f"E={rm['expectancy_pips']:+.2f}"
                )

    if report.walk_forward:
        print(f"\n[WALK-FORWARD] {len(report.walk_forward)} folds")
        for fold in report.walk_forward:
            m = fold.metrics
            print(
                f"  {fold.period_type}: n={fold.n_trades} | "
                f"PF={m.get('profit_factor', 0):.2f} | "
                f"E={m.get('expectancy_pips', 0):+.2f}"
            )

    mc = report.monte_carlo
    print(f"\n[MONTE CARLO] n_sim={mc.get('n_simulations', 0)}")
    print(f"  P(profitable):  {mc.get('prob_profitable', 0):.1%}")
    print(f"  PnL mean:       {mc.get('final_pnl_mean', 0):+.1f} pips")
    print(f"  PnL 5th-95th:   [{mc.get('pnl_5th_pct', 0):+.1f}, {mc.get('pnl_95th_pct', 0):+.1f}]")
    print(f"  Max DD 95th:    {mc.get('max_dd_95th_pct', 0):.1f} pips")
    print(f"  Robustness:     {report.robustness_score:.3f} ({mc.get('verdict', '?')})")

    print(f"\n{'━' * 60}")
    print(f"  VERDICT: {report.verdict}")
    print("=" * 60 + "\n")


def run_example() -> ResearchReport:
    """
    Run a mean-reversion hypothesis on EUR/USD 1H data.

    Requires data to exist in data/fractal_cache/ (use data_loader.py to download).
    Falls back to synthetic data if no real data is available.
    """
    df = _load_data()
    if df is None or df.empty:
        print("[Runner] No real data found — generating synthetic OHLCV data for demo.")
        df = _synthetic_ohlcv(n_bars=5000)

    hypothesis = build_hypothesis(
        name="MR_ATR1.0_RR2.0",
        asset="EURUSD",
        timeframe="1H",
        entry_logic={
            "type": "mean_reversion",
            "threshold": 1.0,
        },
        exit_logic={
            "type": "fixed_rr",
            "sl_atr_multiplier": 1.0,
            "tp_atr_multiplier": 2.0,
            "max_bars": 48,
        },
        features=["atr", "ema_distance", "atr_ratio", "market_bias"],
        test_period=("2024-09-01", "2025-05-31"),
        oos_period=("2025-06-01", "2026-04-30"),
        regime_filter=None,
        description="Mean reversion entry when price > 1 ATR from EMA50, 1:2 RR.",
    )

    orch = Orchestrator(persist=True)
    report = orch.run_hypothesis(df, hypothesis)
    print_report(report)
    return report


def run_trend_example() -> ResearchReport:
    """Secondary example: trend-following hypothesis filtered to TRENDING regime."""
    df = _load_data()
    if df is None or df.empty:
        df = _synthetic_ohlcv(n_bars=5000)

    hypothesis = build_hypothesis(
        name="TREND_FOLLOW_REGIME_FILTER",
        asset="EURUSD",
        timeframe="1H",
        entry_logic={
            "type": "trend_follow",
        },
        exit_logic={
            "type": "fixed_rr",
            "sl_atr_multiplier": 1.5,
            "tp_atr_multiplier": 2.5,
            "max_bars": 72,
        },
        features=["atr", "ema_slope", "market_bias"],
        test_period=("2024-09-01", "2025-05-31"),
        oos_period=("2025-06-01", "2026-04-30"),
        regime_filter="TRENDING",
        description="Trend follow filtered to TRENDING regime only.",
    )

    orch = Orchestrator(persist=True)
    report = orch.run_hypothesis(df, hypothesis)
    print_report(report)
    return report


# ── Data helpers ───────────────────────────────────────────────────────────────

def _load_data() -> pd.DataFrame | None:
    """Try to load cached EUR/USD 1H data from standard cache paths."""
    cache_dir = Path("data/fractal_cache")
    candidates = sorted(cache_dir.glob("EURUSDX_*_1h_*.csv"), reverse=True)
    if not candidates:
        candidates = sorted(cache_dir.glob("EURUSD*1h*.csv"), reverse=True)
    if not candidates:
        candidates = sorted(cache_dir.glob("*.csv"), reverse=True)

    for path in candidates[:3]:
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if {"Open", "High", "Low", "Close"}.issubset({c.title() for c in df.columns}):
                df.columns = [c.title() if c.lower() in ("open","high","low","close","volume") else c
                              for c in df.columns]
                df.index = pd.to_datetime(df.index, utc=True)
                print(f"[Runner] Loaded data: {path.name}  ({len(df)} bars)")
                return df
        except Exception:
            continue

    # Try parquet
    for path in sorted(cache_dir.glob("EURUSD*.parquet"), reverse=True)[:3]:
        try:
            df = pd.read_parquet(path)
            df.columns = [c.title() if c.lower() in ("open","high","low","close","volume") else c
                          for c in df.columns]
            df.index = pd.to_datetime(df.index, utc=True)
            print(f"[Runner] Loaded data: {path.name}  ({len(df)} bars)")
            return df
        except Exception:
            continue

    return None


def _synthetic_ohlcv(n_bars: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing the pipeline."""
    import numpy as np

    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="1h", tz="UTC")

    returns = rng.normal(0, 0.0005, n_bars)
    close = 1.10 * np.exp(returns.cumsum())

    noise = rng.uniform(0.0001, 0.0008, n_bars)
    high = close + noise
    low = close - noise
    open_ = close * (1 + rng.normal(0, 0.0002, n_bars))

    volume = rng.integers(1000, 5000, n_bars).astype(float)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="runpy")
    run_example()
