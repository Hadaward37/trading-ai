"""
Shared utilities for behavioral map analytics modules.

Provides: data loading, backtest re-execution, regime block extraction,
and causal trade-regime assignment.
All calculations are strictly causal (no look-ahead).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────

BATCH_RESULTS_ROOT = Path("research/results/batch_runs")
BEHAVIORAL_MAP_DIR = Path("research/results/behavioral_map")

ALL_REGIMES = ["TRENDING", "MEAN_REVERTING", "HIGH_VOLATILITY", "LOW_VOLATILITY"]

TRAIN_PERIOD = ("2024-01-01", "2024-08-31")
TEST_PERIOD  = ("2024-09-01", "2025-05-31")
OOS_PERIOD   = ("2025-06-01", "2026-04-30")

# ── Hypothesis catalogue (subset used in batch runner) ─────────────────────────

HYPOTHESES: Dict[str, Dict] = {
    "MR_BB_Stretch": {
        "entry": {"type": "mean_reversion", "threshold": 2.0},
        "exit":  {"type": "fixed_rr", "sl_atr_multiplier": 1.0, "tp_atr_multiplier": 2.0, "max_bars": 48},
    },
    "MR_EMA_Dist": {
        "entry": {"type": "mean_reversion", "threshold": 1.5},
        "exit":  {"type": "fixed_rr", "sl_atr_multiplier": 1.0, "tp_atr_multiplier": 2.0, "max_bars": 48},
    },
    "MR_RSI_Exhaust": {
        "entry": {"type": "oscillator_extremes", "feature": "rsi_proxy", "low_threshold": 30.0, "high_threshold": 70.0},
        "exit":  {"type": "fixed_rr", "sl_atr_multiplier": 1.0, "tp_atr_multiplier": 2.0, "max_bars": 48},
    },
    "TF_EMA_Slope": {
        "entry": {"type": "ema_slope", "slope_threshold": 0.0},
        "exit":  {"type": "fixed_rr", "sl_atr_multiplier": 1.5, "tp_atr_multiplier": 2.5, "max_bars": 72},
    },
    "TF_Momentum": {
        "entry": {"type": "momentum_atr", "atr_multiplier": 0.5},
        "exit":  {"type": "fixed_rr", "sl_atr_multiplier": 1.5, "tp_atr_multiplier": 2.5, "max_bars": 72},
    },
    "TF_Breakout_20": {
        "entry": {"type": "breakout", "period": 20},
        "exit":  {"type": "fixed_rr", "sl_atr_multiplier": 1.5, "tp_atr_multiplier": 2.5, "max_bars": 72},
    },
    "VOL_ATR_Expand": {
        "entry": {"type": "volatility_breakout", "threshold": 1.5},
        "exit":  {"type": "fixed_rr", "sl_atr_multiplier": 1.0, "tp_atr_multiplier": 2.0, "max_bars": 24},
    },
    "VOL_Range_Bkout": {
        "entry": {"type": "range_breakout", "threshold": 0.7},
        "exit":  {"type": "fixed_rr", "sl_atr_multiplier": 1.0, "tp_atr_multiplier": 2.5, "max_bars": 48},
    },
}

ASSETS: Dict[str, Dict] = {
    "EURUSD": {"ticker": "EURUSD=X", "pip_multiplier": 10000.0},
    "GOLD":   {"ticker": "GC=F",      "pip_multiplier": 10.0},
    "NASDAQ": {"ticker": "NQ=F",      "pip_multiplier": 1.0},
    "ITUB4":  {"ticker": "ITUB4.SA",  "pip_multiplier": 100.0},
    "PETR4":  {"ticker": "PETR4.SA",  "pip_multiplier": 100.0},
    "IBOV":   {"ticker": "^BVSP",     "pip_multiplier": 1.0},
    "SPY":    {"ticker": "SPY",        "pip_multiplier": 100.0},
    "QQQ":    {"ticker": "QQQ",        "pip_multiplier": 100.0},
    "BTC":    {"ticker": "BTC-USD",    "pip_multiplier": 1.0},
    "DXY":    {"ticker": "DX-Y.NYB",  "pip_multiplier": 1000.0},
}


# ── Batch results loading ──────────────────────────────────────────────────────

def load_batch_rows(results_root: Optional[Path] = None) -> List[Dict]:
    """Load all rows from results_full.json files, deduplicating by (asset, hypothesis)."""
    root = results_root or BATCH_RESULTS_ROOT
    seen: Dict[Tuple, str] = {}
    rows: List[Dict] = []

    for p in sorted(root.glob("*/results_full.json")):
        try:
            ts = p.parent.name
            for row in json.loads(p.read_text(encoding="utf-8")):
                if not row.get("asset") or not row.get("hypothesis"):
                    continue
                key = (row["asset"], row["hypothesis"])
                if key not in seen or ts > seen[key]:
                    seen[key] = ts
                    rows.append(row)
        except Exception:
            continue
    return rows


def select_analysis_targets(rows: List[Dict], n_top: int = 12) -> List[Tuple[str, str]]:
    """
    Select top-N combinations for deep analysis.
    Criteria: oos_verdict in {PASS, OOS_PASS}, test_trades >= 30.
    Ranked by robustness descending.
    """
    candidates = [
        r for r in rows
        if r.get("oos_verdict") in {"PASS", "OOS_PASS"}
        and int(r.get("test_trades", 0)) >= 30
    ]
    candidates.sort(key=lambda r: -float(r.get("robustness", 0)))
    seen_assets: Dict[str, int] = {}
    targets: List[Tuple[str, str]] = []
    for r in candidates:
        asset = r["asset"]
        seen_assets[asset] = seen_assets.get(asset, 0) + 1
        if seen_assets[asset] <= 2:   # max 2 per asset for diversity
            targets.append((asset, r["hypothesis"]))
        if len(targets) >= n_top:
            break
    return targets


# ── Data loading ───────────────────────────────────────────────────────────────

def load_asset_data(asset_key: str) -> pd.DataFrame:
    """Load 1H OHLCV data — tries local cache first, then yfinance."""
    # Cache
    cache_dir = Path("data/fractal_cache")
    if cache_dir.exists():
        ticker_hint = ASSETS[asset_key]["ticker"].replace("=X","X").replace("^","").replace("-","").replace(".","")
        for ext, reader in [(".parquet", pd.read_parquet), (".csv", None)]:
            for p in sorted(cache_dir.glob(f"*{ticker_hint[:4]}*{ext}"), reverse=True)[:2]:
                try:
                    df = reader(p) if ext == ".parquet" else pd.read_csv(p, index_col=0, parse_dates=True)
                    df.columns = [c.title() if c.lower() in ("open","high","low","close","volume") else c for c in df.columns]
                    if not {"Open","High","Low","Close"}.issubset(df.columns):
                        continue
                    df.index = pd.to_datetime(df.index, utc=True)
                    df = df.dropna(subset=["Open","High","Low","Close"])
                    if len(df) > 200:
                        return df
                except Exception:
                    continue

    # yfinance
    try:
        import yfinance as yf
        ticker = ASSETS[asset_key]["ticker"]
        df = yf.download(ticker, period="max", interval="1h", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df.columns = [c.title() if c.lower() in ("open","high","low","close","volume") else c for c in df.columns]
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        return df.dropna(subset=["Open","High","Low","Close"])
    except Exception as exc:
        print(f"    [data] Failed to load {asset_key}: {exc}")
        return pd.DataFrame()


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature computation + regime tagging. Returns full prepared DataFrame."""
    from ..features.base_features import compute_features
    from ..regimes.regime_detector import tag_regimes
    df_feat = compute_features(df)
    return tag_regimes(df_feat)


# ── Backtest re-execution ──────────────────────────────────────────────────────

def run_backtest(
    df_prepared: pd.DataFrame,
    asset_key: str,
    hyp_name: str,
    period_start: Optional[str] = None,
    period_end: Optional[str] = None,
) -> List:
    """Re-run backtest for a combination and return Trade list."""
    from ..research.hypothesis import build_hypothesis
    from ..validation.backtest_engine import BacktestEngine

    hyp_cfg = HYPOTHESES.get(hyp_name)
    asset_cfg = ASSETS.get(asset_key)
    if hyp_cfg is None or asset_cfg is None:
        return []

    start = period_start or TRAIN_PERIOD[0]
    end   = period_end   or OOS_PERIOD[1]

    hypothesis = build_hypothesis(
        name=f"{asset_key}_{hyp_name}",
        asset=asset_key,
        timeframe="1H",
        entry_logic=hyp_cfg["entry"],
        exit_logic=hyp_cfg["exit"],
        features=[],
        test_period=TEST_PERIOD,
        oos_period=OOS_PERIOD,
        pip_multiplier=asset_cfg["pip_multiplier"],
    )
    return BacktestEngine().run(df_prepared, hypothesis, period_start=start, period_end=end)


# ── Regime block extraction ────────────────────────────────────────────────────

def build_regime_blocks(df_prepared: pd.DataFrame) -> pd.DataFrame:
    """
    Extract contiguous regime blocks from a regime-labeled DataFrame.
    Returns DataFrame with: regime, start_ts, end_ts, duration_bars, start_idx, end_idx.
    """
    if "regime" not in df_prepared.columns or df_prepared.empty:
        return pd.DataFrame()

    regimes = df_prepared["regime"].values
    index   = df_prepared.index
    blocks  = []
    i = 0
    while i < len(regimes):
        j = i
        while j < len(regimes) and regimes[j] == regimes[i]:
            j += 1
        blocks.append({
            "regime":        regimes[i],
            "start_idx":     i,
            "end_idx":       j - 1,
            "start_ts":      index[i],
            "end_ts":        index[j - 1],
            "duration_bars": j - i,
        })
        i = j
    return pd.DataFrame(blocks)


# ── Trade utilities ────────────────────────────────────────────────────────────

def trades_to_df(trades: List) -> pd.DataFrame:
    """Convert Trade list to DataFrame with UTC-aware timestamps."""
    if not trades:
        return pd.DataFrame()
    rows = []
    for t in trades:
        rows.append({
            "entry_time": pd.Timestamp(t.entry_time, tz="UTC") if not isinstance(t.entry_time, pd.Timestamp) else t.entry_time,
            "exit_time":  pd.Timestamp(t.exit_time,  tz="UTC") if not isinstance(t.exit_time,  pd.Timestamp) else t.exit_time,
            "direction":  t.direction,
            "pnl_pips":   t.pnl_pips,
            "outcome":    t.outcome,
            "bars_held":  t.bars_held,
            "regime":     t.regime,
            "win":        1 if t.pnl_pips > 0 else 0,
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("entry_time").reset_index(drop=True)
    return df


def pf_from_pnls(pnls: np.ndarray) -> float:
    """Profit Factor from array of signed PnL values."""
    wins   = pnls[pnls > 0].sum()
    losses = abs(pnls[pnls <= 0].sum())
    return round(float(wins / losses), 4) if losses > 1e-9 else (float("inf") if wins > 0 else 0.0)
