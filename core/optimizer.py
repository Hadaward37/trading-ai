"""Grid-search optimizer — finds the best strategy parameters by Sharpe Ratio.

v2: adds ADX regime-blocking threshold and Stochastic confirmation filter.

Grid: rsi_buy(4) x rsi_sell(2) x sl(3) x tp(3) x adx_thresh(3) x stoch_os(2) = 432 combinations
RSI period fixed at 14 (winner of previous run).
Signals (base): 2-of-3 vote  RSI + MACD crossover + BB
Filters added : ADX < adx_threshold -> Range -> signal blocked
                stoch_k >= stoch_oversold for BUY  -> blocked
                stoch_k <= stoch_overbought for SELL -> blocked
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
from core.indicators import add_atr, add_bollinger_bands, add_macd, add_adx, add_stochastic
from core.signals import BB_TOLERANCE
from db.database import load_ohlcv

logger = logging.getLogger(__name__)

RESULTS_PATH = Path(__file__).parent.parent / "data" / "optimization_results.csv"

RSI_PERIOD = 14   # fixed — won previous optimization decisively

# ── Parameter grid (432 combinations) ────────────────────────────────────────
PARAM_GRID: dict[str, list] = {
    "rsi_buy":        [25, 30, 35, 40],
    "rsi_sell":       [70, 75],
    "sl_atr_mult":    [2.0, 2.5, 3.0],
    "tp_atr_mult":    [3.0, 4.0, 5.0],
    "adx_threshold":  [20, 25, 30],   # ADX below this -> Range -> signal blocked
    "stoch_oversold": [20, 25],        # stoch_k threshold for buy confirmation
}

# ── Quality filters ───────────────────────────────────────────────────────────
MIN_WIN_RATE: float = 45.0
MIN_TRADES:   int   = 50    # lower than v1 (filters reduce trade count)


@dataclass
class OptimResult:
    rsi_buy:         int
    rsi_sell:        int
    sl_atr_mult:     float
    tp_atr_mult:     float
    adx_threshold:   int
    stoch_oversold:  int
    sharpe:          float
    total_return:    float
    max_drawdown:    float
    win_rate:        float
    total_trades:    int


# ── Internal helpers ──────────────────────────────────────────────────────────

def _base_signals(df: pd.DataFrame, rsi_buy: int, rsi_sell: int) -> pd.DataFrame:
    """2-of-3 vote: RSI + MACD crossover + BB. Returns df with 'signal' column."""
    out = df.copy()

    v_rsi_buy  = (out["rsi"] < rsi_buy).astype(int)
    v_rsi_sell = (out["rsi"] > rsi_sell).astype(int)

    prev_macd = out["macd"].shift(1)
    prev_sig  = out["macd_signal"].shift(1)
    v_macd_up = ((out["macd"] > out["macd_signal"]) & (prev_macd <= prev_sig)).astype(int)
    v_macd_dn = ((out["macd"] < out["macd_signal"]) & (prev_macd >= prev_sig)).astype(int)

    v_bb_buy  = (out["close"] <= out["bb_lower"] * (1 + BB_TOLERANCE)).astype(int)
    v_bb_sell = (out["close"] >= out["bb_upper"] * (1 - BB_TOLERANCE)).astype(int)

    buy_votes  = v_rsi_buy  + v_macd_up + v_bb_buy
    sell_votes = v_rsi_sell + v_macd_dn + v_bb_sell

    out["signal"] = 0
    out.loc[buy_votes  >= 2, "signal"] =  1
    out.loc[sell_votes >= 2, "signal"] = -1
    return out


def _apply_filters(
    df: pd.DataFrame, adx_threshold: int, stoch_oversold: int
) -> pd.DataFrame:
    """Apply ADX regime-block and Stochastic confirmation filter in-place copy."""
    out = df.copy()
    stoch_overbought = 100 - stoch_oversold

    # Regime filter: ADX below threshold = Range, block entries
    out.loc[out["adx"] < adx_threshold, "signal"] = 0

    # Stochastic filter: buy only when stoch_k is oversold
    out.loc[(out["signal"] ==  1) & (out["stoch_k"] >= stoch_oversold),   "signal"] = 0
    # Stochastic filter: sell only when stoch_k is overbought
    out.loc[(out["signal"] == -1) & (out["stoch_k"] <= stoch_overbought), "signal"] = 0

    return out


# ── Public API ────────────────────────────────────────────────────────────────

def run_optimization(timeframe: str = "1h") -> list[OptimResult]:
    """Run grid search over :data:`PARAM_GRID` using SQLite data.

    Caching strategy:
    * MACD / BB / ATR / ADX / Stochastic: pre-computed once (stable).
    * RSI: pre-computed once at RSI_PERIOD=14.
    * Base signals: cached per (rsi_buy, rsi_sell) — 8 unique DFs.
    * Filtered signals: computed per (adx_threshold, stoch_oversold) — fast mask.
    * Backtest: one run per full combo.

    Returns:
        Filtered (Win Rate > :data:`MIN_WIN_RATE`, Trades > :data:`MIN_TRADES`)
        and Sharpe-ranked list of :class:`OptimResult`.
    """
    logger.info("Loading OHLCV from SQLite (timeframe=%s)...", timeframe)
    base_df = load_ohlcv(timeframe)
    if base_df is None:
        raise RuntimeError(
            f"No data for timeframe={timeframe!r}. "
            f"Run `python main.py --timeframe {timeframe}` first."
        )

    logger.info("Pre-computing stable indicators (MACD, BB, ATR, ADX, Stochastic)...")
    base_df = add_macd(base_df)
    base_df = add_bollinger_bands(base_df)
    base_df = add_atr(base_df)
    base_df = add_adx(base_df)
    base_df = add_stochastic(base_df)

    # RSI fixed at RSI_PERIOD
    base_df["rsi"] = RSIIndicator(close=base_df["close"], window=RSI_PERIOD).rsi()

    keys   = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))
    total  = len(combos)
    logger.info(
        "Grid: %d combos | RSI_PERIOD=%d fixed | WinRate>%.0f%% Trades>%d",
        total, RSI_PERIOD, MIN_WIN_RATE, MIN_TRADES,
    )

    results: list[OptimResult] = []

    # Cache base signals per (rsi_buy, rsi_sell)
    sig_cache: dict[tuple, pd.DataFrame] = {}

    for idx, combo in enumerate(combos, 1):
        p = dict(zip(keys, combo))

        sig_key = (p["rsi_buy"], p["rsi_sell"])
        if sig_key not in sig_cache:
            sig_cache[sig_key] = _base_signals(base_df, *sig_key)

        df_filtered = _apply_filters(
            sig_cache[sig_key],
            adx_threshold=p["adx_threshold"],
            stoch_oversold=p["stoch_oversold"],
        )

        bt = run_backtest(
            df_filtered,
            sl_atr_mult=p["sl_atr_mult"],
            tp_atr_mult=p["tp_atr_mult"],
        )

        if idx % 50 == 0 or idx == total:
            logger.info("[%d/%d] %.0f%% — passed: %d", idx, total,
                        idx / total * 100, len(results))

        if bt.win_rate >= MIN_WIN_RATE and bt.total_trades >= MIN_TRADES:
            results.append(OptimResult(
                rsi_buy=p["rsi_buy"],
                rsi_sell=p["rsi_sell"],
                sl_atr_mult=p["sl_atr_mult"],
                tp_atr_mult=p["tp_atr_mult"],
                adx_threshold=p["adx_threshold"],
                stoch_oversold=p["stoch_oversold"],
                sharpe=bt.sharpe_ratio,
                total_return=bt.total_return,
                max_drawdown=bt.max_drawdown,
                win_rate=bt.win_rate,
                total_trades=bt.total_trades,
            ))

    results.sort(key=lambda r: r.sharpe, reverse=True)
    logger.info("Optimization done. %d/%d passed filters.", len(results), total)
    return results


def save_top10(results: list[OptimResult]) -> Path:
    """Overwrite :data:`RESULTS_PATH` with the top-10 results."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(r) for r in results[:10]]).to_csv(RESULTS_PATH, index=False)
    logger.info("Top 10 saved -> %s", RESULTS_PATH)
    return RESULTS_PATH


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    results = run_optimization(timeframe="1h")

    if not results:
        print(f"\nNo combinations passed WinRate>{MIN_WIN_RATE}% and Trades>{MIN_TRADES}.")
        sys.exit(0)

    csv_path = save_top10(results)
    best     = results[0]
    sep      = "-" * 52

    print(f"\n{sep}")
    print("  BEST PARAMETER SET  (ranked by Sharpe Ratio)")
    print(sep)
    print(f"  RSI Period        : {RSI_PERIOD} (fixed)")
    print(f"  RSI Buy  <        : {best.rsi_buy}")
    print(f"  RSI Sell >        : {best.rsi_sell}")
    print(f"  Stop-Loss  ATR    : {best.sl_atr_mult}x")
    print(f"  Take-Profit ATR   : {best.tp_atr_mult}x")
    print(f"  ADX Trend Thresh  : {best.adx_threshold}")
    print(f"  Stoch Oversold <  : {best.stoch_oversold}")
    print(sep)
    print(f"  Sharpe Ratio      : {best.sharpe:+.3f}")
    print(f"  Total Return      : {best.total_return:+.2f}%")
    print(f"  Max Drawdown      : {best.max_drawdown:.2f}%")
    print(f"  Win Rate          : {best.win_rate:.1f}%")
    print(f"  Total Trades      : {best.total_trades}")
    print(sep)
    print(f"\nTop 10 -> {csv_path}\n")

    df_top = pd.DataFrame([asdict(r) for r in results[:10]])
    df_top.index = range(1, min(11, len(df_top) + 1))
    df_top.columns = [
        "Buy", "Sell", "SL", "TP", "ADX_T", "Stoch_OS",
        "Sharpe", "Return%", "MaxDD%", "WinRate%", "Trades",
    ]
    print(df_top.to_string())
