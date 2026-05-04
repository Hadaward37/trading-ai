"""Signal scheduler — polls market data every N minutes and fires Telegram
alerts only when the signal changes (BUY <-> SELL / HOLD -> BUY|SELL).

Spam prevention: HOLD signals and repeated identical signals are suppressed.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure project root is importable regardless of how this module is invoked
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from core.backtest import run_backtest
from core.collector import fetch_ohlcv
from core.indicators import add_all_indicators
from core.notifier import SignalAlert, TelegramNotifier
from core.signals import generate_signals
from db.database import save_ohlcv, save_signals

logger = logging.getLogger(__name__)

_SIGNAL_NAMES = {1: "BUY", -1: "SELL", 0: "HOLD"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _count_votes(row, prev_row) -> int:
    """Count how many of the 3 indicators voted for row's signal."""
    signal = int(row["signal"])
    if signal == 0:
        return 0

    rsi_vote = (
        (row["rsi"] < config.RSI_BUY)  if signal == 1 else
        (row["rsi"] > config.RSI_SELL)
    )

    macd_cross_up = (row["macd"] > row["macd_signal"]) and (prev_row["macd"] <= prev_row["macd_signal"])
    macd_cross_dn = (row["macd"] < row["macd_signal"]) and (prev_row["macd"] >= prev_row["macd_signal"])
    macd_vote = macd_cross_up if signal == 1 else macd_cross_dn

    bb_vote = (
        (row["close"] <= row["bb_lower"] * (1 + config.BB_TOLERANCE)) if signal == 1 else
        (row["close"] >= row["bb_upper"] * (1 - config.BB_TOLERANCE))
    )

    return int(rsi_vote) + int(macd_vote) + int(bb_vote)


def _build_alert(df, win_rate: float) -> SignalAlert:
    """Construct a :class:`SignalAlert` from the last completed bar."""
    # -2 = last *completed* bar; -1 = current forming bar (not yet closed)
    row      = df.iloc[-2]
    prev_row = df.iloc[-3]

    signal_val  = int(row["signal"])
    signal_name = _SIGNAL_NAMES.get(signal_val, "HOLD")
    price       = float(row["close"])
    atr         = float(row["atr"])

    if signal_val == 1:      # BUY
        stop_loss   = price - config.SL_ATR_MULT * atr
        take_profit = price + config.TP_ATR_MULT * atr
    elif signal_val == -1:   # SELL
        stop_loss   = price + config.SL_ATR_MULT * atr
        take_profit = price - config.TP_ATR_MULT * atr
    else:                    # HOLD — theoretical levels for context
        stop_loss   = price - config.SL_ATR_MULT * atr
        take_profit = price + config.TP_ATR_MULT * atr

    return SignalAlert(
        signal      = signal_name,
        asset       = config.SYMBOL,
        timeframe   = config.SCHEDULER_TIMEFRAME,
        price       = price,
        stop_loss   = round(stop_loss, 5),
        take_profit = round(take_profit, 5),
        win_rate    = win_rate,
        confidence  = _count_votes(row, prev_row),
        rsi         = round(float(row["rsi"]), 2),
        atr         = round(atr, 5),
    )


# ── Core cycle ────────────────────────────────────────────────────────────────

def run_once(
    notifier: TelegramNotifier,
    last_signal: Optional[int],
) -> int:
    """Execute one pipeline cycle.

    Returns:
        The signal value of the last completed bar (1, -1, or 0).
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("--- Cycle start | %s ---", now)

    df = fetch_ohlcv(timeframe=config.SCHEDULER_TIMEFRAME)
    df = add_all_indicators(df)
    df = generate_signals(df)

    save_ohlcv(df[["open", "high", "low", "close", "volume"]], config.SCHEDULER_TIMEFRAME)
    save_signals(df, config.SCHEDULER_TIMEFRAME)

    bt = run_backtest(df)

    last_row   = df.iloc[-2]
    cur_signal = int(last_row["signal"])
    cur_name   = _SIGNAL_NAMES.get(cur_signal, "HOLD")

    logger.info(
        "Signal=%-4s | RSI=%5.1f | Price=%.5f | WinRate=%.1f%%",
        cur_name, float(last_row["rsi"]), float(last_row["close"]), bt.win_rate,
    )

    # ── Alert logic: fire only on meaningful signal changes ───────────────────
    if cur_signal != 0 and cur_signal != last_signal:
        alert = _build_alert(df, bt.win_rate)
        logger.info("Signal changed %s -> %s — sending alert",
                    _SIGNAL_NAMES.get(last_signal, "NONE"), cur_name)
        notifier.send_signal_alert(alert)

    elif cur_signal == 0 and last_signal not in (0, None):
        logger.info("Signal cleared -> HOLD (no alert sent)")

    else:
        logger.info("No signal change — skipping alert")

    return cur_signal


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_scheduler() -> None:
    """Infinite polling loop. Runs :func:`run_once` every
    :data:`config.SCHEDULER_INTERVAL_MIN` minutes."""
    interval_sec = config.SCHEDULER_INTERVAL_MIN * 60

    notifier = TelegramNotifier(
        token   = config.TELEGRAM_BOT_TOKEN,
        chat_id = config.TELEGRAM_CHAT_ID,
    )

    if not notifier.is_configured:
        logger.warning(
            "Telegram credentials not configured.\n"
            "  1. Copy .env.example to .env\n"
            "  2. Fill in TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID\n"
            "Alerts will be printed to the log only."
        )
    else:
        notifier.send_text(
            f"Trading AI iniciado\n"
            f"Ativo: {config.SYMBOL} | TF: {config.SCHEDULER_TIMEFRAME} | "
            f"Intervalo: {config.SCHEDULER_INTERVAL_MIN} min"
        )

    logger.info(
        "Scheduler running | interval=%d min | timeframe=%s | asset=%s",
        config.SCHEDULER_INTERVAL_MIN, config.SCHEDULER_TIMEFRAME, config.SYMBOL,
    )

    last_signal: Optional[int] = None

    while True:
        try:
            last_signal = run_once(notifier, last_signal)
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user.")
            break
        except Exception:
            logger.exception("Unhandled error in cycle — will retry next interval")

        next_at = datetime.now().strftime("%H:%M:%S")
        logger.info("Sleeping %d min (next check after %s)...",
                    config.SCHEDULER_INTERVAL_MIN, next_at)
        time.sleep(interval_sec)
