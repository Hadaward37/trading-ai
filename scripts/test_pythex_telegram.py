"""Test: Pythex bridge + Telegram alert with analysis."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from core.pythex_bridge import explain_signal, load_context
from core.notifier import TelegramNotifier, SignalAlert

# --- dados do ultimo sinal BUY do EURUSD (2026-05-05 09:00)
SIGNAL    = "BUY"
ENTRY     = 1.16858
SL        = 1.16579
TP        = 1.17304
SCORE     = 46.5
RSI       = 40.9
ATR       = 0.001116
WIN_RATE  = 47.2   # historico backtest

print("[ 1/3 ] Carregando contexto Pythex...")
ctx = load_context()
print(ctx[:300], "...\n")

print("[ 2/3 ] Consultando gpt-4o-mini para explicacao do sinal...")
analysis = explain_signal(SIGNAL, ENTRY, SL, TP, SCORE)
print(f"  Analise: {analysis}\n")

print("[ 3/3 ] Enviando alerta Telegram com analise...")
notifier = TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)

alert = SignalAlert(
    signal      = SIGNAL,
    asset       = "EUR/USD",
    timeframe   = "H1",
    price       = ENTRY,
    stop_loss   = SL,
    take_profit = TP,
    win_rate    = WIN_RATE,
    confidence  = 2,
    rsi         = RSI,
    atr         = ATR,
    flag        = "🌍",
    decimals    = 5,
    sentiment   = "NEUTRAL",
    sentiment_confidence = 0.0,
    sentiment_impact     = "LOW",
    final_score = SCORE,
    analysis    = analysis,
)

ok = notifier.send_signal_alert(alert)
print(f"  Telegram: {'OK' if ok else 'FALHOU'}")
