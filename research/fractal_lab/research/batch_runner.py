"""
Batch Runner -- stress test do Fractal Lab Core Engine.

Roda automaticamente 8 hipoteses x 6 ativos = 48 combinacoes.
Gera matriz consolidada de resultados, ranking por Robustness Score e heatmaps.

Objetivo: identificar hipoteses que sobrevivem em multiplos ativos (consistencia),
nao as de maior PF em um ativo especifico (overfitting).

Usage:
    python -X utf8 -m research.fractal_lab.research.batch_runner
    python -X utf8 -m research.fractal_lab.research.batch_runner --dry-run
    python -X utf8 -m research.fractal_lab.research.batch_runner --assets EURUSD GOLD
    python -X utf8 -m research.fractal_lab.research.batch_runner --hypotheses MR_BB_Stretch MR_RSI_Exhaust
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from research.fractal_lab.core.orchestrator import Orchestrator
from research.fractal_lab.research.hypothesis import build_hypothesis
from research.fractal_lab.storage.models import ResearchReport

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Asset catalogue ────────────────────────────────────────────────────────────

ASSETS: Dict[str, Dict] = {
    "EURUSD": {
        "ticker": "EURUSD=X",
        "name": "EUR/USD",
        "pip_multiplier": 10000.0,
        "category": "forex",
    },
    "GOLD": {
        "ticker": "GC=F",
        "name": "Gold Futures",
        "pip_multiplier": 10.0,
        "category": "commodity",
    },
    "NASDAQ": {
        "ticker": "NQ=F",
        "name": "NASDAQ Futures",
        "pip_multiplier": 1.0,
        "category": "index",
    },
    "ITUB4": {
        "ticker": "ITUB4.SA",
        "name": "Itau Unibanco",
        "pip_multiplier": 100.0,
        "category": "brazil_equity",
    },
    "PETR4": {
        "ticker": "PETR4.SA",
        "name": "Petrobras",
        "pip_multiplier": 100.0,
        "category": "brazil_equity",
    },
    "IBOV": {
        "ticker": "^BVSP",
        "name": "Ibovespa",
        "pip_multiplier": 1.0,
        "category": "index",
    },
}

# ── Period definitions ─────────────────────────────────────────────────────────

TRAIN_PERIOD = ("2024-01-01", "2024-08-31")
TEST_PERIOD  = ("2024-09-01", "2025-05-31")
OOS_PERIOD   = ("2025-06-01", "2026-04-30")

# ── Hypothesis catalogue ───────────────────────────────────────────────────────

HYPOTHESES: Dict[str, Dict] = {

    # Group 1: Mean Reversion
    "MR_BB_Stretch": {
        "group": "mean_reversion",
        "entry": {"type": "mean_reversion", "threshold": 2.0},
        "exit": {"type": "fixed_rr", "sl_atr_multiplier": 1.0, "tp_atr_multiplier": 2.0, "max_bars": 48},
        "features": ["atr", "ema_distance"],
        "description": "MR entry when price > 2.0 ATR from EMA50",
    },
    "MR_EMA_Dist": {
        "group": "mean_reversion",
        "entry": {"type": "mean_reversion", "threshold": 1.5},
        "exit": {"type": "fixed_rr", "sl_atr_multiplier": 1.0, "tp_atr_multiplier": 2.0, "max_bars": 48},
        "features": ["atr", "ema_distance"],
        "description": "MR entry when price > 1.5 ATR from EMA50",
    },
    "MR_RSI_Exhaust": {
        "group": "mean_reversion",
        "entry": {
            "type": "oscillator_extremes",
            "feature": "rsi_proxy",
            "low_threshold": 30.0,
            "high_threshold": 70.0,
        },
        "exit": {"type": "fixed_rr", "sl_atr_multiplier": 1.0, "tp_atr_multiplier": 2.0, "max_bars": 48},
        "features": ["atr", "rsi_proxy"],
        "description": "MR entry at RSI < 30 (buy) or RSI > 70 (sell)",
    },

    # Group 2: Trend Following
    "TF_EMA_Slope": {
        "group": "trend_follow",
        "entry": {"type": "ema_slope", "slope_threshold": 0.0},
        "exit": {"type": "fixed_rr", "sl_atr_multiplier": 1.5, "tp_atr_multiplier": 2.5, "max_bars": 72},
        "features": ["atr", "ema20_slope", "ema20_distance"],
        "description": "Trend follow: EMA20 slope direction + price above/below EMA20",
    },
    "TF_Momentum": {
        "group": "trend_follow",
        "entry": {"type": "momentum_atr", "atr_multiplier": 0.5},
        "exit": {"type": "fixed_rr", "sl_atr_multiplier": 1.5, "tp_atr_multiplier": 2.5, "max_bars": 72},
        "features": ["atr", "momentum"],
        "description": "Trend follow: 10-bar momentum > 0.5x ATR",
    },
    "TF_Breakout_20": {
        "group": "trend_follow",
        "entry": {"type": "breakout", "period": 20},
        "exit": {"type": "fixed_rr", "sl_atr_multiplier": 1.5, "tp_atr_multiplier": 2.5, "max_bars": 72},
        "features": ["atr"],
        "description": "Breakout: close > max or < min of last 20 candles",
    },

    # Group 3: Volatility
    "VOL_ATR_Expand": {
        "group": "volatility",
        "entry": {"type": "volatility_breakout", "threshold": 1.5},
        "exit": {"type": "fixed_rr", "sl_atr_multiplier": 1.0, "tp_atr_multiplier": 2.0, "max_bars": 24},
        "features": ["atr", "atr_ratio", "market_bias"],
        "description": "Enter in bias direction when ATR > 1.5x rolling mean",
    },
    "VOL_Range_Bkout": {
        "group": "volatility",
        "entry": {"type": "range_breakout", "threshold": 0.7},
        "exit": {"type": "fixed_rr", "sl_atr_multiplier": 1.0, "tp_atr_multiplier": 2.5, "max_bars": 48},
        "features": ["atr", "atr_ratio", "market_bias"],
        "description": "Enter in bias direction after ATR compression (< 0.7x mean)",
    },
}


# ── Data loading ───────────────────────────────────────────────────────────────

def _download_yfinance(ticker: str, interval: str = "1h") -> pd.DataFrame:
    """Download data via yfinance, compatible with 0.1.x and 0.2.x API."""
    try:
        import yfinance as yf
        df = yf.download(
            ticker,
            period="max",
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            return pd.DataFrame()

        # yfinance >= 0.2 may return MultiIndex columns for single ticker
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        df.columns = [
            c.title() if c.lower() in ("open", "high", "low", "close", "volume") else c
            for c in df.columns
        ]
        # Ensure required columns exist
        required = {"Open", "High", "Low", "Close"}
        if not required.issubset(df.columns):
            return pd.DataFrame()

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        return df

    except Exception as exc:
        print(f"    [download] yfinance error: {exc}")
        return pd.DataFrame()


def _load_from_cache(asset_key: str) -> pd.DataFrame:
    """Try to load data from the local fractal_cache directory."""
    cache_dir = Path("data/fractal_cache")
    if not cache_dir.exists():
        return pd.DataFrame()

    ticker_hint = ASSETS[asset_key]["ticker"].replace("=X", "X").replace("^", "").replace(".", "")
    # Remove futures suffix for file matching
    ticker_hint = ticker_hint.replace("=F", "")

    # Search for matching files (parquet first, then CSV)
    for ext, reader in [(".parquet", pd.read_parquet), (".csv", pd.read_csv)]:
        candidates = sorted(cache_dir.glob(f"*{ticker_hint[:4]}*{ext}"), reverse=True)
        for path in candidates[:3]:
            try:
                if ext == ".csv":
                    df = pd.read_csv(path, index_col=0, parse_dates=True)
                else:
                    df = reader(path)
                df.columns = [
                    c.title() if c.lower() in ("open","high","low","close","volume") else c
                    for c in df.columns
                ]
                if not {"Open", "High", "Low", "Close"}.issubset(df.columns):
                    continue
                df.index = pd.to_datetime(df.index, utc=True)
                df = df.dropna(subset=["Open", "High", "Low", "Close"])
                if len(df) > 100:
                    print(f"    [cache] Loaded {path.name} ({len(df)} bars)")
                    return df
            except Exception:
                continue
    return pd.DataFrame()


def _check_period_coverage(df: pd.DataFrame, start: str, end: str) -> int:
    if df.empty:
        return 0
    ts_start = pd.Timestamp(start).tz_localize("UTC")
    ts_end   = pd.Timestamp(end).tz_localize("UTC")
    return int(((df.index >= ts_start) & (df.index <= ts_end)).sum())


# ── Single combination ─────────────────────────────────────────────────────────

def _run_combination(
    asset_key: str,
    asset_cfg: Dict,
    hyp_name: str,
    hyp_cfg: Dict,
    df: pd.DataFrame,
    orchestrator: Orchestrator,
    dry_run: bool = False,
) -> Dict:
    min_bars = _check_period_coverage(df, *TEST_PERIOD)
    if min_bars < 30:
        return _empty_row(asset_key, hyp_name, hyp_cfg, "SKIP_NO_DATA")

    if dry_run:
        print(f"    [DRY-RUN] {asset_key} x {hyp_name}: {min_bars} test bars")
        return _empty_row(asset_key, hyp_name, hyp_cfg, "DRY_RUN")

    try:
        hypothesis = build_hypothesis(
            name=f"{asset_key}_{hyp_name}",
            asset=asset_key,
            timeframe="1H",
            entry_logic=hyp_cfg["entry"],
            exit_logic=hyp_cfg["exit"],
            features=hyp_cfg["features"],
            test_period=TEST_PERIOD,
            oos_period=OOS_PERIOD,
            pip_multiplier=asset_cfg["pip_multiplier"],
            description=hyp_cfg.get("description", ""),
        )
        report = orchestrator.run_hypothesis(df, hypothesis)
        return _report_to_row(asset_key, hyp_name, hyp_cfg, asset_cfg, report)

    except Exception as exc:
        print(f"    [ERROR] {asset_key} x {hyp_name}: {exc}")
        return _empty_row(asset_key, hyp_name, hyp_cfg, f"ERROR")


def _report_to_row(
    asset_key: str,
    hyp_name: str,
    hyp_cfg: Dict,
    asset_cfg: Dict,
    report: ResearchReport,
) -> Dict:
    test = report.test
    oos  = report.oos
    mc   = report.monte_carlo

    tm = test.metrics if test else {}
    om = oos.metrics  if oos  else {}

    dominant_regime, regime_pcts = _compute_regime_dist(test)

    return {
        "asset":           asset_key,
        "asset_name":      asset_cfg["name"],
        "category":        asset_cfg["category"],
        "hypothesis":      hyp_name,
        "group":           hyp_cfg["group"],
        "description":     hyp_cfg.get("description", ""),
        "test_pf":         round(tm.get("profit_factor", 0.0), 3),
        "test_sharpe":     round(tm.get("sharpe", 0.0), 3),
        "test_max_dd":     round(tm.get("max_drawdown_pips", 0.0), 1),
        "test_trades":     int(tm.get("n_trades", 0)),
        "test_wr":         round(tm.get("win_rate", 0.0), 4),
        "test_expectancy": round(tm.get("expectancy_pips", 0.0), 2),
        "oos_pf":          round(om.get("profit_factor", 0.0), 3),
        "oos_sharpe":      round(om.get("sharpe", 0.0), 3),
        "oos_max_dd":      round(om.get("max_drawdown_pips", 0.0), 1),
        "oos_trades":      int(om.get("n_trades", 0)),
        "oos_wr":          round(om.get("win_rate", 0.0), 4),
        "oos_expectancy":  round(om.get("expectancy_pips", 0.0), 2),
        "oos_verdict":     report.verdict,
        "robustness":      round(report.robustness_score, 4),
        "rob_verdict":     mc.get("verdict", ""),
        "mc_prob_profit":  round(mc.get("prob_profitable", 0.0), 4),
        "mc_pnl_mean":     round(mc.get("final_pnl_mean", 0.0), 1),
        "mc_dd_95pct":     round(mc.get("max_dd_95th_pct", 0.0), 1),
        "dominant_regime": dominant_regime,
        "pct_trending":    regime_pcts.get("TRENDING", 0.0),
        "pct_mean_rev":    regime_pcts.get("MEAN_REVERTING", 0.0),
        "pct_high_vol":    regime_pcts.get("HIGH_VOLATILITY", 0.0),
        "pct_low_vol":     regime_pcts.get("LOW_VOLATILITY", 0.0),
    }


def _compute_regime_dist(period_result) -> Tuple[str, Dict[str, float]]:
    if period_result is None or not period_result.trades:
        return "N/A", {}
    counts: Dict[str, int] = {}
    for t in period_result.trades:
        r = t.regime or "UNKNOWN"
        counts[r] = counts.get(r, 0) + 1
    total = len(period_result.trades)
    pcts = {r: round(n / total, 4) for r, n in counts.items()}
    dominant = max(counts, key=counts.get) if counts else "N/A"
    return dominant, pcts


def _empty_row(asset_key: str, hyp_name: str, hyp_cfg: Dict, verdict: str) -> Dict:
    row: Dict[str, Any] = {
        "asset": asset_key,
        "asset_name": ASSETS[asset_key]["name"],
        "category": ASSETS[asset_key]["category"],
        "hypothesis": hyp_name,
        "group": hyp_cfg["group"],
        "description": hyp_cfg.get("description", ""),
        "oos_verdict": verdict,
        "robustness": 0.0,
        "rob_verdict": verdict,
        "dominant_regime": "N/A",
    }
    for col in [
        "test_pf", "test_sharpe", "test_max_dd", "test_trades", "test_wr", "test_expectancy",
        "oos_pf", "oos_sharpe", "oos_max_dd", "oos_trades", "oos_wr", "oos_expectancy",
        "mc_prob_profit", "mc_pnl_mean", "mc_dd_95pct",
        "pct_trending", "pct_mean_rev", "pct_high_vol", "pct_low_vol",
    ]:
        row.setdefault(col, 0.0)
    return row


# ── CSV / heatmap generators ───────────────────────────────────────────────────

def _save_matrix_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.sort_values(["robustness", "test_pf"], ascending=False, inplace=True)
    df.to_csv(path, index=False, float_format="%.4f", encoding="utf-8")


def _save_ranking_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    ranked = (
        df.groupby("hypothesis")
        .agg(
            group=("group", "first"),
            avg_robustness=("robustness", "mean"),
            avg_pf_test=("test_pf", "mean"),
            avg_pf_oos=("oos_pf", "mean"),
            avg_sharpe=("test_sharpe", "mean"),
            n_pass=("oos_verdict", lambda x: x.isin(["PASS", "OOS_PASS"]).sum()),
            n_assets=("asset", "count"),
        )
        .reset_index()
        .sort_values("avg_robustness", ascending=False)
    )
    ranked.to_csv(path, index=False, float_format="%.4f", encoding="utf-8")


def _save_heatmap_csv(rows: List[Dict], metric: str, path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(
        index="asset", columns="hypothesis", values=metric, aggfunc="first"
    )
    pivot.to_csv(path, float_format="%.4f", encoding="utf-8")


def _ascii_heatmap(rows: List[Dict], metric: str = "robustness") -> str:
    """Render an asset x hypothesis heatmap as an ASCII string."""
    if not rows:
        return ""

    # active assets/hyps in the current batch (preserving catalogue order)
    active_assets = [a for a in ASSETS if any(r["asset"] == a for r in rows)]
    active_hyps   = [h for h in HYPOTHESES if any(r["hypothesis"] == h for r in rows)]

    pivot: Dict[str, Dict[str, float]] = {a: {} for a in active_assets}
    for row in rows:
        pivot[row["asset"]][row["hypothesis"]] = float(row.get(metric, 0.0))

    # Column header abbreviations
    abbr = {h: h[:9] for h in active_hyps}
    col_w   = 10
    label_w = 8

    header = f"{'Asset':<{label_w}} |" + "".join(f"{abbr[h]:>{col_w}}" for h in active_hyps)
    sep    = "-" * len(header)

    lines = [
        "",
        "=" * len(header),
        f"  HEATMAP -- {metric.upper()}",
        "=" * len(header),
        header,
        sep,
    ]

    for asset in active_assets:
        cells = []
        for h in active_hyps:
            v = pivot[asset].get(h, float("nan"))
            if np.isnan(v):
                cell = "N/A"
            else:
                if metric == "robustness":
                    mark = "*" if v >= 0.50 else ("~" if v >= 0.35 else ".")
                elif metric in ("test_pf", "oos_pf"):
                    mark = "+" if v > 1.0 else ("-" if v < 1.0 else "=")
                else:
                    mark = ""
                cell = f"{v:.3f}{mark}"
            cells.append(f"{cell:>{col_w}}")
        lines.append(f"{asset:<{label_w}} |" + "".join(cells))

    lines.append(sep)
    if metric == "robustness":
        lines.append("  * >= 0.50 (moderate+)   ~ >= 0.35 (weak)   . < 0.35 (fragile)")
    elif metric in ("test_pf", "oos_pf"):
        lines.append("  + > 1.0 (edge)   = 1.0 (breakeven)   - < 1.0 (no edge)")
    lines.append("=" * len(header))
    return "\n".join(lines)


# ── Console print helpers ──────────────────────────────────────────────────────

def _print_results_table(rows: List[Dict]) -> None:
    if not rows:
        print("No results.")
        return

    COL = 128
    print("\n" + "=" * COL)
    print("  BATCH RESULTS MATRIX  (sorted by Robustness)")
    print("=" * COL)
    print(
        f"{'Asset':<8} {'Hypothesis':<16} {'Group':<14} "
        f"{'PF':>6} {'Shp':>6} "
        f"{'Rob':>6} {'Trd':>5} {'WR':>6} "
        f"{'OOS_PF':>7} {'OOS_T':>6} {'Verdict':<10} "
        f"{'Dom.Regime':<18}"
    )
    print("-" * COL)

    for row in sorted(rows, key=lambda r: -r.get("robustness", 0)):
        v = row.get("oos_verdict", "?")
        print(
            f"{row.get('asset',''):<8} {row.get('hypothesis',''):<16} {row.get('group',''):<14} "
            f"{row.get('test_pf',0):>6.3f} {row.get('test_sharpe',0):>6.3f} "
            f"{row.get('robustness',0):>6.3f} {row.get('test_trades',0):>5} "
            f"{row.get('test_wr',0):>6.1%} "
            f"{row.get('oos_pf',0):>7.3f} {row.get('oos_trades',0):>6} {v:<10} "
            f"{row.get('dominant_regime','N/A'):<18}"
        )

    print("=" * COL)


def _print_survival_summary(rows: List[Dict]) -> None:
    print("\n" + "=" * 72)
    print("  HYPOTHESIS SURVIVAL  (PASS or OOS_PASS across assets)")
    print("=" * 72)

    survival: Dict[str, Dict] = {}
    for row in rows:
        h = row["hypothesis"]
        if h not in survival:
            survival[h] = {"group": row["group"], "passing": [], "rob": [], "oos_pf": []}
        if row.get("oos_verdict") in ("PASS", "OOS_PASS"):
            survival[h]["passing"].append(row["asset"])
        survival[h]["rob"].append(row.get("robustness", 0.0))
        survival[h]["oos_pf"].append(row.get("oos_pf", 0.0))

    print(f"  {'Hypothesis':<18} {'Group':<14} {'Passing Assets':<32} {'Avg Rob':>8} {'Avg OOS PF':>10}")
    print("  " + "-" * 82)
    for hyp, d in sorted(survival.items(), key=lambda x: -len(x[1]["passing"])):
        passing = ", ".join(d["passing"]) if d["passing"] else "none"
        avg_rob = float(np.mean(d["rob"])) if d["rob"] else 0.0
        avg_pf  = float(np.mean(d["oos_pf"])) if d["oos_pf"] else 0.0
        print(f"  {hyp:<18} {d['group']:<14} {passing:<32} {avg_rob:>8.3f} {avg_pf:>10.3f}")

    print("=" * 72)


def _print_regime_analysis(rows: List[Dict]) -> None:
    print("\n" + "=" * 66)
    print("  REGIME DISTRIBUTION by strategy group (avg % of trades)")
    print("=" * 66)

    grp: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        g = row.get("group", "?")
        if g not in grp:
            grp[g] = {"TRENDING": [], "MEAN_REVERTING": [], "HIGH_VOLATILITY": [], "LOW_VOLATILITY": []}
        grp[g]["TRENDING"].append(row.get("pct_trending", 0.0))
        grp[g]["MEAN_REVERTING"].append(row.get("pct_mean_rev", 0.0))
        grp[g]["HIGH_VOLATILITY"].append(row.get("pct_high_vol", 0.0))
        grp[g]["LOW_VOLATILITY"].append(row.get("pct_low_vol", 0.0))

    print(f"  {'Group':<16} {'TRENDING':>10} {'MEAN_REV':>10} {'HIGH_VOL':>10} {'LOW_VOL':>10}")
    print("  " + "-" * 56)
    for g, d in sorted(grp.items()):
        print(
            f"  {g:<16} {np.mean(d['TRENDING']):>10.1%} "
            f"{np.mean(d['MEAN_REVERTING']):>10.1%} "
            f"{np.mean(d['HIGH_VOLATILITY']):>10.1%} "
            f"{np.mean(d['LOW_VOLATILITY']):>10.1%}"
        )
    print("=" * 66)


def _print_degradation_analysis(rows: List[Dict]) -> None:
    """Show how much each strategy degrades from test to OOS."""
    print("\n" + "=" * 66)
    print("  OOS DEGRADATION by strategy group  (test PF -> OOS PF)")
    print("=" * 66)

    grp: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        g = row.get("group", "?")
        if g not in grp:
            grp[g] = {"test_pf": [], "oos_pf": []}
        grp[g]["test_pf"].append(row.get("test_pf", 0.0))
        grp[g]["oos_pf"].append(row.get("oos_pf", 0.0))

    print(f"  {'Group':<16} {'Avg Test PF':>12} {'Avg OOS PF':>11} {'Degradation':>12}")
    print("  " + "-" * 52)
    for g, d in sorted(grp.items()):
        t  = float(np.mean(d["test_pf"]))
        o  = float(np.mean(d["oos_pf"]))
        dg = round(((o - t) / max(t, 1e-9)) * 100, 1)
        sign = "+" if dg >= 0 else ""
        print(f"  {g:<16} {t:>12.3f} {o:>11.3f} {sign}{dg:>11.1f}%")
    print("=" * 66)


# ── Main batch runner ──────────────────────────────────────────────────────────

def run_batch(
    output_dir: Optional[Path] = None,
    dry_run: bool = False,
    assets: Optional[List[str]] = None,
    hypotheses: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Run the full stress test batch.

    Parameters
    ----------
    output_dir  : path for results (default: research/results/batch_runs/<timestamp>/)
    dry_run     : skip computation, only verify setup and data availability
    assets      : subset of ASSETS to run (default: all)
    hypotheses  : subset of HYPOTHESES to run (default: all)
    """
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = Path("research/results/batch_runs") / ts
    output_dir.mkdir(parents=True, exist_ok=True)

    active_assets = {k: v for k, v in ASSETS.items() if assets is None or k in assets}
    active_hyps   = {k: v for k, v in HYPOTHESES.items() if hypotheses is None or k in hypotheses}
    total         = len(active_assets) * len(active_hyps)

    print(f"\n{'=' * 60}")
    print(f"  FRACTAL LAB BATCH RUNNER -- {ts}")
    print(f"  Assets: {len(active_assets)} | Hypotheses: {len(active_hyps)} | Total: {total}")
    print(f"  Train: {TRAIN_PERIOD[0]} -> {TRAIN_PERIOD[1]}")
    print(f"  Test:  {TEST_PERIOD[0]}  -> {TEST_PERIOD[1]}")
    print(f"  OOS:   {OOS_PERIOD[0]}   -> {OOS_PERIOD[1]}")
    print(f"  Out:   {output_dir}")
    if dry_run:
        print("  MODE: DRY-RUN (no computation)")
    print("=" * 60 + "\n")

    orchestrator = Orchestrator(persist=False, n_mc_simulations=500, verbose=False)
    all_rows: List[Dict] = []

    # ── Download data once per asset ───────────────────────────────────────────
    asset_data: Dict[str, pd.DataFrame] = {}
    for i, (asset_key, asset_cfg) in enumerate(active_assets.items(), 1):
        print(f"[{i}/{len(active_assets)}] Loading {asset_key} ({asset_cfg['ticker']})...")
        if dry_run:
            asset_data[asset_key] = pd.DataFrame()
        else:
            # Try yfinance first, fall back to cache
            df = _download_yfinance(asset_cfg["ticker"])
            if df.empty:
                print(f"    yfinance empty — trying local cache...")
                df = _load_from_cache(asset_key)
            asset_data[asset_key] = df
            n_train = _check_period_coverage(df, *TRAIN_PERIOD)
            n_test  = _check_period_coverage(df, *TEST_PERIOD)
            n_oos   = _check_period_coverage(df, *OOS_PERIOD)
            status  = "OK" if n_test >= 50 else ("WARN: low test data" if n_test > 0 else "NO DATA")
            print(f"    {status} | train={n_train} test={n_test} oos={n_oos} total={len(df)}")
        time.sleep(0.3)

    # ── Run all combinations ───────────────────────────────────────────────────
    done = 0
    for asset_key, asset_cfg in active_assets.items():
        df = asset_data.get(asset_key, pd.DataFrame())
        print(f"\n--- {asset_key}: {asset_cfg['name']} ---")

        for hyp_name, hyp_cfg in active_hyps.items():
            done += 1
            print(f"  [{done:>2}/{total}] {hyp_name:<20}", end=" ", flush=True)
            t0 = time.time()

            row = _run_combination(
                asset_key, asset_cfg, hyp_name, hyp_cfg, df, orchestrator, dry_run
            )
            all_rows.append(row)
            elapsed = time.time() - t0
            print(
                f"Verdict={row.get('oos_verdict','?'):<12} "
                f"Rob={row.get('robustness',0):.3f}  "
                f"OOS_PF={row.get('oos_pf',0):.3f}  "
                f"({elapsed:.1f}s)"
            )

    if not all_rows:
        print("\nNo results generated.")
        return []

    # ── Save results ───────────────────────────────────────────────────────────
    print(f"\nSaving results to {output_dir}...")
    _save_matrix_csv(all_rows, output_dir / "results_matrix.csv")
    _save_ranking_csv(all_rows, output_dir / "ranking.csv")
    _save_heatmap_csv(all_rows, "robustness", output_dir / "heatmap_robustness.csv")
    _save_heatmap_csv(all_rows, "test_pf",    output_dir / "heatmap_pf_test.csv")
    _save_heatmap_csv(all_rows, "oos_pf",     output_dir / "heatmap_pf_oos.csv")
    _save_heatmap_csv(all_rows, "oos_verdict", output_dir / "heatmap_verdict.csv")

    rob_heatmap = _ascii_heatmap(all_rows, "robustness")
    pf_heatmap  = _ascii_heatmap(all_rows, "oos_pf")
    oos_heatmap = _ascii_heatmap(all_rows, "test_pf")
    (output_dir / "heatmap_ascii.txt").write_text(
        "\n\n".join([rob_heatmap, pf_heatmap, oos_heatmap]), encoding="utf-8"
    )

    n_pass = sum(1 for r in all_rows if r.get("oos_verdict") in ("PASS", "OOS_PASS"))
    summary = {
        "run_timestamp": ts,
        "n_assets":        len(active_assets),
        "n_hypotheses":    len(active_hyps),
        "n_combinations":  len(all_rows),
        "n_pass":          n_pass,
        "periods":         {"train": TRAIN_PERIOD, "test": TEST_PERIOD, "oos": OOS_PERIOD},
        "top_by_robustness": sorted(all_rows, key=lambda r: -r.get("robustness", 0))[:5],
    }
    (output_dir / "batch_summary.json").write_text(
        json.dumps(summary, indent=2, default=str, ensure_ascii=True),
        encoding="utf-8",
    )

    # ── Print full report ──────────────────────────────────────────────────────
    _print_results_table(all_rows)
    print(rob_heatmap)
    print(pf_heatmap)
    _print_survival_summary(all_rows)
    _print_regime_analysis(all_rows)
    _print_degradation_analysis(all_rows)

    print(f"\n  BATCH COMPLETE: {n_pass}/{len(all_rows)} passed OOS")
    print(f"  Output directory: {output_dir}\n")

    return all_rows


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="Fractal Lab Batch Runner -- stress test")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip computation, only verify data availability")
    parser.add_argument("--assets", nargs="+", choices=list(ASSETS.keys()),
                        help="Run only these assets (default: all)")
    parser.add_argument("--hypotheses", nargs="+", choices=list(HYPOTHESES.keys()),
                        help="Run only these hypotheses (default: all)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory override")
    args = parser.parse_args()

    run_batch(
        output_dir=args.output,
        dry_run=args.dry_run,
        assets=args.assets,
        hypotheses=args.hypotheses,
    )
