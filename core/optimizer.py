"""Grid-search optimizer — finds the best strategy parameters by Sharpe Ratio.

Searches 768 combinations (3×4×4×4×4) using data already in SQLite.
RSI is cached per period to avoid redundant computation.
"""

from __future__ import annotations

import itertools
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from ta.momentum import RSIIndicator

from core.backtest import run_backtest
from core.indicators import add_atr, add_bollinger_bands, add_macd
from core.signals import BB_TOLERANCE
from db.database import load_ohlcv

logger = logging.getLogger(__name__)

RESULTS_PATH = Path(__file__).parent.parent / "data" / "optimization_results.csv"

# ── Parameter grid (768 total combinations) ───────────────────────────────────
PARAM_GRID: dict[str, list] = {
    "rsi_period":  [10, 14, 21],
    "rsi_buy":     [25, 30, 35, 40],
    "rsi_sell":    [60, 65, 70, 75],
    "sl_atr_mult": [1.0, 1.5, 2.0, 2.5],
    "tp_atr_mult": [2.0, 3.0, 4.0, 5.0],
}

# ── Quality filters ───────────────────────────────────────────────────────────
MIN_WIN_RATE: float = 45.0
MIN_TRADES:   int   = 100


@dataclass
class OptimResult:
    rsi_period:   int
    rsi_buy:      int
    rsi_sell:     int
    sl_atr_mult:  float
    tp_atr_mult:  float
    sharpe:       float
    total_return: float
    max_drawdown: float
    win_rate:     float
    total_trades: int


# ── Internal helpers ──────────────────────────────────────────────────────────

def _recalc_rsi(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Return a copy of *df* with the ``rsi`` column recalculated for *period*."""
    out = df.copy()
    out["rsi"] = RSIIndicator(close=out["close"], window=period).rsi()
    return out


def _apply_signal_thresholds(
    df: pd.DataFrame, rsi_buy: int, rsi_sell: int
) -> pd.DataFrame:
    """Generate signals using custom RSI thresholds; MACD and BB logic unchanged."""
    out = df.copy()

    # RSI votes
    vote_rsi_buy  = out["rsi"] < rsi_buy
    vote_rsi_sell = out["rsi"] > rsi_sell

    # MACD crossover votes
    prev_macd = out["macd"].shift(1)
    prev_sig  = out["macd_signal"].shift(1)
    vote_macd_up = (out["macd"] > out["macd_signal"]) & (prev_macd <= prev_sig)
    vote_macd_dn = (out["macd"] < out["macd_signal"]) & (prev_macd >= prev_sig)

    # Bollinger Band votes
    vote_bb_buy  = out["close"] <= out["bb_lower"] * (1 + BB_TOLERANCE)
    vote_bb_sell = out["close"] >= out["bb_upper"] * (1 - BB_TOLERANCE)

    buy_votes  = vote_rsi_buy.astype(int)  + vote_macd_up.astype(int) + vote_bb_buy.astype(int)
    sell_votes = vote_rsi_sell.astype(int) + vote_macd_dn.astype(int) + vote_bb_sell.astype(int)

    out["signal"] = 0
    out.loc[buy_votes  >= 2, "signal"] =  1
    out.loc[sell_votes >= 2, "signal"] = -1
    return out


# ── Public API ────────────────────────────────────────────────────────────────

def run_optimization(timeframe: str = "1h") -> list[OptimResult]:
    """Run grid search over :data:`PARAM_GRID` using SQLite data.

    Args:
        timeframe: Table to load — ``"1h"`` (default) or ``"15m"``.

    Returns:
        Filtered and Sharpe-ranked list of :class:`OptimResult`.

    Raises:
        RuntimeError: If no data exists for the requested timeframe.
    """
    logger.info("Loading OHLCV data from SQLite (timeframe=%s)...", timeframe)
    base_df = load_ohlcv(timeframe)
    if base_df is None:
        raise RuntimeError(
            f"No data for timeframe={timeframe!r}. Run `python main.py --timeframe {timeframe}` first."
        )

    # Pre-compute stable indicators (MACD, BB, ATR never change across the grid)
    logger.info("Pre-computing MACD, Bollinger Bands, ATR...")
    base_df = add_macd(base_df)
    base_df = add_bollinger_bands(base_df)
    base_df = add_atr(base_df)

    keys   = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))
    total  = len(combos)
    logger.info("Grid size: %d combinations. Filters: WinRate>%.0f%%, Trades>%d",
                total, MIN_WIN_RATE, MIN_TRADES)

    results: list[OptimResult] = []
    # Cache RSI DataFrames keyed by period (avoids recomputing for same period)
    rsi_cache: dict[int, pd.DataFrame] = {}

    for idx, combo in enumerate(combos, 1):
        p = dict(zip(keys, combo))

        period = p["rsi_period"]
        if period not in rsi_cache:
            rsi_cache[period] = _recalc_rsi(base_df, period)

        df_sig = _apply_signal_thresholds(
            rsi_cache[period], rsi_buy=p["rsi_buy"], rsi_sell=p["rsi_sell"]
        )

        bt = run_backtest(
            df_sig,
            sl_atr_mult=p["sl_atr_mult"],
            tp_atr_mult=p["tp_atr_mult"],
        )

        if idx % 100 == 0 or idx == total:
            pct = idx / total * 100
            logger.info("[%d/%d] %.0f%% — passed so far: %d", idx, total, pct, len(results))

        if bt.win_rate >= MIN_WIN_RATE and bt.total_trades >= MIN_TRADES:
            results.append(OptimResult(
                rsi_period=period,
                rsi_buy=p["rsi_buy"],
                rsi_sell=p["rsi_sell"],
                sl_atr_mult=p["sl_atr_mult"],
                tp_atr_mult=p["tp_atr_mult"],
                sharpe=bt.sharpe_ratio,
                total_return=bt.total_return,
                max_drawdown=bt.max_drawdown,
                win_rate=bt.win_rate,
                total_trades=bt.total_trades,
            ))

    results.sort(key=lambda r: r.sharpe, reverse=True)
    logger.info(
        "Optimization done. %d/%d combinations passed filters.",
        len(results), total,
    )
    return results


def save_top10(results: list[OptimResult]) -> Path:
    """Save the top-10 results to :data:`RESULTS_PATH` as CSV."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    top10 = results[:10]
    pd.DataFrame([asdict(r) for r in top10]).to_csv(RESULTS_PATH, index=False)
    logger.info("Top 10 saved -> %s", RESULTS_PATH)
    return RESULTS_PATH


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    results = run_optimization(timeframe="1h")

    if not results:
        print("\nNo combinations passed the filters.")
        print(f"  Win Rate > {MIN_WIN_RATE}%  AND  Total Trades > {MIN_TRADES}")
        print("Try relaxing the thresholds at the top of optimizer.py.")
        sys.exit(0)

    csv_path = save_top10(results)

    best = results[0]
    sep = "-" * 49

    print(f"\n{sep}")
    print("  BEST PARAMETER SET  (ranked by Sharpe Ratio)")
    print(sep)
    print(f"  RSI Period      : {best.rsi_period}")
    print(f"  RSI Buy  <      : {best.rsi_buy}")
    print(f"  RSI Sell >      : {best.rsi_sell}")
    print(f"  Stop-Loss  ATR  : {best.sl_atr_mult}x")
    print(f"  Take-Profit ATR : {best.tp_atr_mult}x")
    print(sep)
    print(f"  Sharpe Ratio    : {best.sharpe:+.3f}")
    print(f"  Total Return    : {best.total_return:+.2f}%")
    print(f"  Max Drawdown    : {best.max_drawdown:.2f}%")
    print(f"  Win Rate        : {best.win_rate:.1f}%")
    print(f"  Total Trades    : {best.total_trades}")
    print(sep)

    print(f"\nTop 10 results -> {csv_path}\n")

    print("TOP 10 (Sharpe descending):")
    df_top = pd.DataFrame([asdict(r) for r in results[:10]])
    df_top.index = range(1, 11)
    df_top.columns = [
        "RSI_P", "RSI_Buy", "RSI_Sell", "SL", "TP",
        "Sharpe", "Return%", "MaxDD%", "WinRate%", "Trades",
    ]
    print(df_top.to_string())
