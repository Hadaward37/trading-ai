"""CLI entry point — runs the full data pipeline and prints backtest results.

Usage:
    python main.py                   # 1-hour timeframe (default)
    python main.py --timeframe 15m   # 15-minute timeframe
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


def run_pipeline(timeframe: str) -> None:
    """Fetch → indicators → signals → backtest → persist."""
    from core.backtest import run_backtest
    from core.collector import fetch_ohlcv
    from core.indicators import add_all_indicators
    from core.signals import generate_signals
    from db.database import save_ohlcv, save_signals

    logger.info("Starting pipeline | timeframe=%s", timeframe)

    df = fetch_ohlcv(timeframe=timeframe)
    df = add_all_indicators(df)
    df = generate_signals(df)

    save_ohlcv(df[["open", "high", "low", "close", "volume"]], timeframe)
    save_signals(df, timeframe)

    result = run_backtest(df)

    logger.info(
        "Pipeline done | return=%.2f%% sharpe=%.3f max_dd=%.2f%% "
        "trades=%d win_rate=%.1f%%",
        result.total_return, result.sharpe_ratio, result.max_drawdown,
        result.total_trades, result.win_rate,
    )

    print("\n--- Backtest Results ----------------------------")
    print(f"  Timeframe    : {timeframe}")
    print(f"  Total Return : {result.total_return:+.2f}%")
    print(f"  Sharpe Ratio : {result.sharpe_ratio:.3f}")
    print(f"  Max Drawdown : {result.max_drawdown:.2f}%")
    print(f"  Win Rate     : {result.win_rate:.1f}%  ({result.total_trades} trades)")
    print("-------------------------------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading AI — EUR/USD Pipeline")
    parser.add_argument(
        "--timeframe",
        choices=["15m", "1h"],
        default="1h",
        help="Candle interval (default: 1h)",
    )
    args = parser.parse_args()
    run_pipeline(args.timeframe)
