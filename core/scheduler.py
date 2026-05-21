"""Signal scheduler — polls all assets every N minutes and fires Telegram alerts
only when the signal changes (BUY <-> SELL / HOLD -> BUY|SELL).

Spam prevention: HOLD signals and repeated identical signals are suppressed.
News filter: high-impact economic events block signal alerts with a warning.
Multi-asset: monitors EUR/USD, VALE3, PETR4, ITUB4, BBDC4, IBOV simultaneously.

Observer integration (non-blocking):
  The Fractal Lab Observer runs every 1H alongside Sistema 1.
  REGRA: observer APENAS observa, registra e alerta.
  Nunca bloqueia sinais, nunca altera parametros, nunca interfere na execucao.
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
from core.healthcheck import heartbeat, start_healthcheck
from core.indicators import add_all_indicators
from core.news_filter import is_danger_zone
from core.notifier import SignalAlert, TelegramNotifier
from core.paper_trading import VirtualPortfolio
from core.signals import generate_signals_custom
from core.telemetry import get_telemetry
from core.trade_journal import export_daily_csv, print_daily_summary
from db.database import save_ohlcv, save_signals

logger = logging.getLogger(__name__)

_SIGNAL_NAMES = {1: "BUY", -1: "SELL", 0: "HOLD"}

# ── Observer Agent (optional, non-blocking) ───────────────────────────────────
# Loaded once at module level. Import errors are silently ignored — Sistema 1
# continues normally without the observer.

_OBSERVER_AVAILABLE = False
try:
    from research.fractal_lab.observer.agent import ObserverAgent as _ObserverAgent
    _OBSERVER_AVAILABLE = True
except Exception:
    pass

OBSERVER_INTERVAL_S = 3600          # observer runs every 1H
SUMMARY_INTERVAL_S  = 6 * 3600     # 6H Telegram summary


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
        sentiment             = str(row.get("sentiment", "NEUTRAL")),
        sentiment_confidence  = float(row.get("sentiment_confidence", 0.0)),
        sentiment_impact      = str(row.get("sentiment_impact", "LOW")),
        final_score           = float(row.get("final_score", row.get("score", 0.0))),
        lstm_prob             = float(row.get("lstm_prob", 50.0)),
        xgboost_prob          = float(row.get("xgboost_prob", 50.0)),
        ensemble_prob         = float(row.get("ensemble_prob", 0.0)),
    )


# ── Per-asset cycle ───────────────────────────────────────────────────────────

def _send_close_alert(notifier: TelegramNotifier, closed: dict) -> None:
    """Envia alerta Telegram quando uma operação de paper trading fecha."""
    reason_label = {"TP_HIT": "🎯 TP ATINGIDO", "SL_HIT": "🛑 SL ATINGIDO",
                    "REVERSED": "↩️ SINAL INVERTIDO"}.get(closed.get("exit_reason", ""), "✅ FECHOU")
    pnl   = closed.get("pnl", 0)
    emoji = "✅" if pnl >= 0 else "❌"
    meta  = config.ASSET_META.get(closed.get("symbol", ""), {})
    flag  = meta.get("flag", "📊")
    curr  = meta.get("currency", "")
    d     = meta.get("decimals", 2)

    text = (
        f"{emoji} {flag} *Paper Trade Fechado — {closed['asset_name']}*\n\n"
        f"{reason_label}\n"
        f"Direção: `{closed['direction']}`\n"
        f"Entrada: `{closed['entry_price']:.{d}f}` → Saída: `{closed['exit_price']:.{d}f}`\n"
        f"P&L: `{curr} {pnl:+.2f}` ({closed.get('pnl_pct', 0):+.2f}%)\n"
        f"Saldo atual: `{curr} {closed.get('new_balance', 0):,.2f}`"
    )
    notifier._send(text)


def _run_asset(
    symbol: str,
    notifier: TelegramNotifier,
    last_signal: Optional[int],
    portfolio: Optional[VirtualPortfolio] = None,
) -> int:
    """Run one full pipeline cycle for a single asset.

    Returns the signal value of the last completed bar (1, -1, or 0).
    """
    meta = config.ASSET_META.get(symbol, config.ASSET_META[config.SYMBOL])
    name = meta["name"]

    try:
        ticker_name = config.ASSET_META.get(symbol, {}).get("name", symbol)

        df = fetch_ohlcv(timeframe=config.SCHEDULER_TIMEFRAME, symbol=symbol)
        df = add_all_indicators(df)
        # check_news=False here — we do the news check manually below so we can
        # send a meaningful blocked message rather than just silently returning 0
        df = generate_signals_custom(
            df, config.RSI_BUY, config.RSI_SELL,
            check_news=False, check_sentiment=True, ticker=ticker_name,
        )

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

        # ── Telemetry: record every signal + regime filter observation ────────
        if cur_signal != 0:
            try:
                atr_val   = float(last_row.get("atr", 0.001))
                atr_ma    = float(df["atr"].iloc[-22:-2].mean()) if len(df) > 22 else atr_val
                atr_ratio = atr_val / max(atr_ma, 1e-9)
                get_telemetry().record_signal(
                    symbol    = symbol,
                    name      = name,
                    signal    = cur_name,
                    price     = float(last_row["close"]),
                    adx       = float(last_row.get("adx", 0)),
                    atr_ratio = atr_ratio,
                    hour_utc  = last_row.name.hour if hasattr(last_row.name, "hour") else 12,
                )
            except Exception:
                pass

        # ── Paper trading — checar SL/TP e abrir novas posições ──────────────
        if portfolio and config.PAPER_TRADING_ENABLED:
            # 1. Verificar se posição aberta atingiu SL ou TP
            closed_trades = portfolio.check_and_close(symbol, df)
            for ct in closed_trades:
                _send_close_alert(notifier, ct)
                # Telemetry: record trade outcome
                try:
                    direction = ct.get("direction", "BUY")
                    pf_alert  = get_telemetry().record_trade_outcome(
                        symbol      = symbol,
                        direction   = direction,
                        entry_price = float(ct.get("entry_price", 0)),
                        exit_price  = float(ct.get("exit_price", 0)),
                        pnl         = float(ct.get("pnl", 0)),
                        pnl_pct     = float(ct.get("pnl_pct", 0)),
                        hour_utc    = datetime.now().hour,
                    )
                    if pf_alert:
                        notifier.send_text(pf_alert)
                except Exception:
                    pass

            # 2. Abrir nova posição se sinal mudou (e não bloqueado por news)
            if cur_signal != 0 and cur_signal != last_signal:
                news_blocked, _ = is_danger_zone()
                if not news_blocked:
                    price = float(last_row["close"])
                    atr   = float(last_row["atr"])
                    if cur_signal == 1:   # BUY
                        sl = price - config.SL_ATR_MULT * atr
                        tp = price + config.TP_ATR_MULT * atr
                    else:                  # SELL
                        sl = price + config.SL_ATR_MULT * atr
                        tp = price - config.TP_ATR_MULT * atr

                    portfolio.open_trade(
                        symbol      = symbol,
                        asset_name  = name,
                        direction   = cur_name,
                        entry_price = price,
                        stop_loss   = round(sl, meta["decimals"]),
                        take_profit = round(tp, meta["decimals"]),
                        signal_score  = float(last_row.get("score", 0)),
                        final_score   = float(last_row.get("final_score", 0)),
                        sentiment     = str(last_row.get("sentiment", "NEUTRAL")),
                        timeframe     = config.SCHEDULER_TIMEFRAME,
                    )

        # ── Alert logic ───────────────────────────────────────────────────────
        if cur_signal != 0 and cur_signal != last_signal:
            try:
                from core.signal_logger import log_signal
                log_signal(
                    symbol=symbol,
                    signal=cur_name,
                    price_at_signal=float(last_row["close"]),
                    features={
                        "rsi":          round(float(last_row.get("rsi",          0.0)), 2),
                        "adx":          round(float(last_row.get("adx",          0.0)), 2),
                        "atr":          round(float(last_row.get("atr",          0.0)), 5),
                        "bb_pct":       round(float(last_row.get("bb_pct",       0.5)), 4),
                        "stoch_k":      round(float(last_row.get("stoch_k",     50.0)), 2),
                        "macd":         round(float(last_row.get("macd",         0.0)), 6),
                        "score":        round(float(last_row.get("score",        0.0)), 1),
                        "final_score":  round(float(last_row.get("final_score",  0.0)), 1),
                        "lstm_prob":    round(float(last_row.get("lstm_prob",   50.0)), 1),
                        "xgboost_prob": round(float(last_row.get("xgboost_prob",50.0)), 1),
                        "ensemble_prob":round(float(last_row.get("ensemble_prob", 0.0)),1),
                        "sentiment":    str(last_row.get("sentiment", "NEUTRAL")),
                    },
                    regime=None,
                    metadata={"source": "system_1", "timeframe": config.SCHEDULER_TIMEFRAME},
                )
            except Exception:
                pass

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

        # ── Telemetry: resolve blocked signal hypothetical outcomes ──────────
        if symbol == config.SYMBOL:   # EUR/USD only — regime filter tested on EUR/USD
            try:
                blocked_alerts = get_telemetry().resolve_blocked_outcomes(df)
                for balert in blocked_alerts:
                    notifier.send_text(balert)
            except Exception:
                pass

        return cur_signal

    except Exception:
        logger.exception("[%s] Unhandled error in asset cycle", name)
        return last_signal if last_signal is not None else 0


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_once(
    notifier: TelegramNotifier,
    last_signals: dict[str, Optional[int]],
    portfolio: Optional[VirtualPortfolio] = None,
) -> dict[str, int]:
    """Execute one pipeline cycle for ALL configured assets."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("=== Cycle start | %s | %d assets ===", now, len(config.ASSETS))
    heartbeat()

    updated: dict[str, int] = {}
    for name, symbol in config.ASSETS.items():
        updated[symbol] = _run_asset(symbol, notifier, last_signals.get(symbol), portfolio)

    return updated


def run_scheduler() -> None:
    """Infinite polling loop. Runs :func:`run_once` every
    :data:`config.SCHEDULER_INTERVAL_MIN` minutes."""
    interval_sec = config.SCHEDULER_INTERVAL_MIN * 60
    start_healthcheck()

    notifier = TelegramNotifier(
        token   = config.TELEGRAM_BOT_TOKEN,
        chat_id = config.TELEGRAM_CHAT_ID,
    )

    # Paper trading portfolio
    portfolio: Optional[VirtualPortfolio] = None
    if config.PAPER_TRADING_ENABLED:
        portfolio = VirtualPortfolio()
        metrics   = portfolio.get_metrics()
        logger.info(
            "Paper trading ativo | Saldo=%.2f | Capital=%.2f | Trades=%d",
            metrics["balance"], metrics["initial_capital"], metrics["total_trades"],
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
        pt_status  = f"Paper Trading: {'ON' if config.PAPER_TRADING_ENABLED else 'OFF'}"
        notifier.send_text(
            f"Trading AI iniciado — multi-asset\n"
            f"Ativos: {asset_list}\n"
            f"TF: {config.SCHEDULER_TIMEFRAME} | "
            f"Intervalo: {config.SCHEDULER_INTERVAL_MIN} min\n"
            f"{pt_status}"
        )

    logger.info(
        "Scheduler running | interval=%d min | timeframe=%s | assets=%s | paper_trading=%s",
        config.SCHEDULER_INTERVAL_MIN,
        config.SCHEDULER_TIMEFRAME,
        list(config.ASSETS.keys()),
        config.PAPER_TRADING_ENABLED,
    )

    # ── Observer Agent initialization (non-blocking) ──────────────────────────
    _observer = None
    _last_observer_ts: float = 0.0
    _observer_cycle:   int   = 0

    if _OBSERVER_AVAILABLE:
        try:
            _observer = _ObserverAgent()
            logger.info(
                "[Observer] Initialized | interval=1H | targets=%d | alert_fatigue=ON",
                len(_observer.targets),
            )
            notifier.send_text(
                "🔬 *Fractal Lab Observer* iniciado\n"
                "Monitoramento cientifico ativo (read-only)\n"
                "Alertas: WARNING a cada 4H | CRITICAL a cada 2H\n"
                "Health summary: a cada 6H"
            )
        except Exception as exc:
            logger.warning("[Observer] Could not initialize: %s — continuing without observer", exc)

    last_signals: dict[str, Optional[int]] = {}
    _last_journal_date: Optional[str] = None

    while True:
        try:
            last_signals = run_once(notifier, last_signals, portfolio)

            # ── Observer cycle (every 1H, non-blocking) ───────────────────────
            if _observer is not None:
                now_mono = time.monotonic()
                if (now_mono - _last_observer_ts) >= OBSERVER_INTERVAL_S:
                    try:
                        _observer.run_once()
                        _observer_cycle += 1
                        _last_observer_ts = now_mono
                        logger.info(
                            "[Observer] Cycle #%d complete | Health=%s/100 (%s)",
                            _observer_cycle,
                            _observer._aggregate_health.score if _observer._aggregate_health else "?",
                            _observer._aggregate_health.label  if _observer._aggregate_health else "?",
                        )
                    except Exception as exc:
                        logger.warning("[Observer] Cycle error (non-fatal): %s", exc)

            # Export journal CSV uma vez por dia (à meia-noite UTC)
            today = datetime.now().strftime("%Y-%m-%d")
            if _last_journal_date != today and portfolio:
                print_daily_summary()
                export_daily_csv()
                _last_journal_date = today

        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user.")
            if portfolio:
                print_daily_summary()
                export_daily_csv()
            if _observer:
                # Send final summary before exiting
                try:
                    summary = _observer.get_health_summary()
                    notifier.send_text(summary)
                except Exception:
                    pass
            break
        except Exception:
            logger.exception("Unhandled error in cycle — will retry next interval")

        next_at = datetime.now().strftime("%H:%M:%S")
        logger.info("Sleeping %d min (next check after %s)...",
                    config.SCHEDULER_INTERVAL_MIN, next_at)
        time.sleep(interval_sec)
