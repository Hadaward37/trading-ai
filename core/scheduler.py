"""Signal scheduler — polls all assets every N minutes and fires Telegram alerts
only when the signal changes (BUY <-> SELL / HOLD -> BUY|SELL).

Spam prevention: HOLD signals and repeated identical signals are suppressed.
News filter: high-impact economic events block signal alerts with a warning.
Multi-asset: monitors EUR/USD, VALE3, PETR4, ITUB4, BBDC4, IBOV simultaneously.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from core.backtest import run_backtest
from core.collector import fetch_ohlcv
from core.indicators import add_all_indicators
from core.news_filter import is_danger_zone
from core.notifier import SignalAlert, TelegramNotifier
from core.signals import generate_signals_custom
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


def _build_alert(df, win_rate: float, symbol: str) -> SignalAlert:
    """Construct a :class:`SignalAlert` from the last completed bar."""
    meta = config.ASSET_META.get(symbol, config.ASSET_META[config.SYMBOL])

    row      = df.iloc[-2]
    prev_row = df.iloc[-3]

    signal_val  = int(row["signal"])
    signal_name = _SIGNAL_NAMES.get(signal_val, "HOLD")
    price       = float(row["close"])
    atr         = float(row["atr"])

    if signal_val == 1:
        stop_loss   = price - config.SL_ATR_MULT * atr
        take_profit = price + config.TP_ATR_MULT * atr
    elif signal_val == -1:
        stop_loss   = price + config.SL_ATR_MULT * atr
        take_profit = price - config.TP_ATR_MULT * atr
    else:
        stop_loss   = price - config.SL_ATR_MULT * atr
        take_profit = price + config.TP_ATR_MULT * atr

    d = meta["decimals"]
    return SignalAlert(
        signal      = signal_name,
        asset       = meta["name"],
        timeframe   = config.SCHEDULER_TIMEFRAME,
        price       = price,
        stop_loss   = round(stop_loss,   d),
        take_profit = round(take_profit, d),
        win_rate    = win_rate,
        confidence  = _count_votes(row, prev_row),
        rsi         = round(float(row["rsi"]), 2),
        atr         = round(atr, d),
        flag        = meta["flag"],
        decimals    = d,
    )


# ── Per-asset cycle ───────────────────────────────────────────────────────────

def _run_asset(
    symbol: str,
    notifier: TelegramNotifier,
    last_signal: Optional[int],
) -> int:
    """Run one full pipeline cycle for a single asset.

    Returns the signal value of the last completed bar (1, -1, or 0).
    """
    meta = config.ASSET_META.get(symbol, config.ASSET_META[config.SYMBOL])
    name = meta["name"]

    try:
        df = fetch_ohlcv(timeframe=config.SCHEDULER_TIMEFRAME, symbol=symbol)
        df = add_all_indicators(df)
        # check_news=False here — we do the news check manually below so we can
        # send a meaningful blocked message rather than just silently returning 0
        df = generate_signals_custom(df, config.RSI_BUY, config.RSI_SELL, check_news=False)

        # Only save DB records for the default asset to avoid schema complexity
        if symbol == config.SYMBOL:
            save_ohlcv(df[["open", "high", "low", "close", "volume"]], config.SCHEDULER_TIMEFRAME)
            save_signals(df, config.SCHEDULER_TIMEFRAME)

        bt = run_backtest(df)

        last_row   = df.iloc[-2]
        cur_signal = int(last_row["signal"])
        cur_name   = _SIGNAL_NAMES.get(cur_signal, "HOLD")

        logger.info(
            "%s %s | Signal=%-4s RSI=%5.1f Price=%s WinRate=%.1f%%",
            meta["flag"], name, cur_name, float(last_row["rsi"]),
            f"{float(last_row['close']):.{meta['decimals']}f}", bt.win_rate,
        )

        # ── Alert logic ───────────────────────────────────────────────────────
        if cur_signal != 0 and cur_signal != last_signal:
            news_blocked, news_reason = is_danger_zone()
            if news_blocked:
                logger.info("[%s] News filter blocked — %s", name, news_reason)
                notifier.send_text(
                    f"⚠️ {meta['flag']} {name} — sinal bloqueado\n"
                    f"Evento de alto impacto: {news_reason}\n"
                    f"Sinal técnico: {cur_name} | Aguardando janela segura "
                    f"(±{config.NEWS_FILTER_WINDOW_MINUTES} min)"
                )
            else:
                alert = _build_alert(df, bt.win_rate, symbol)
                logger.info("[%s] Signal changed %s -> %s — sending alert",
                            name, _SIGNAL_NAMES.get(last_signal, "NONE"), cur_name)
                notifier.send_signal_alert(alert)

        elif cur_signal == 0 and last_signal not in (0, None):
            logger.info("[%s] Signal cleared -> HOLD", name)
        else:
            logger.info("[%s] No signal change — skipping", name)

        return cur_signal

    except Exception:
        logger.exception("[%s] Unhandled error in asset cycle", name)
        return last_signal if last_signal is not None else 0


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_once(
    notifier: TelegramNotifier,
    last_signals: dict[str, Optional[int]],
) -> dict[str, int]:
    """Execute one pipeline cycle for ALL configured assets.

    Returns:
        Updated mapping of symbol -> current signal.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("=== Cycle start | %s | %d assets ===", now, len(config.ASSETS))

    updated: dict[str, int] = {}
    for name, symbol in config.ASSETS.items():
        updated[symbol] = _run_asset(symbol, notifier, last_signals.get(symbol))

    return updated


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
        asset_list = " | ".join(config.ASSETS.keys())
        notifier.send_text(
            f"Trading AI iniciado — multi-asset\n"
            f"Ativos: {asset_list}\n"
            f"TF: {config.SCHEDULER_TIMEFRAME} | "
            f"Intervalo: {config.SCHEDULER_INTERVAL_MIN} min"
        )

    logger.info(
        "Scheduler running | interval=%d min | timeframe=%s | assets=%s",
        config.SCHEDULER_INTERVAL_MIN,
        config.SCHEDULER_TIMEFRAME,
        list(config.ASSETS.keys()),
    )

    last_signals: dict[str, Optional[int]] = {}

    while True:
        try:
            last_signals = run_once(notifier, last_signals)
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user.")
            break
        except Exception:
            logger.exception("Unhandled error in cycle — will retry next interval")

        next_at = datetime.now().strftime("%H:%M:%S")
        logger.info("Sleeping %d min (next check after %s)...",
                    config.SCHEDULER_INTERVAL_MIN, next_at)
        time.sleep(interval_sec)
