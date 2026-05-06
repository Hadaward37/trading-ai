"""Test: Copom macro context + Telegram alert with Copom line."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from core.macro_context import get_copom_context, format_telegram_line, copom_score_adjustment
from core.notifier import TelegramNotifier, SignalAlert

ctx = get_copom_context()
print("=== Copom Context ===")
print(f"  Tom      : {ctx['copom_tone'].upper()}")
print(f"  Selic    : {ctx['selic_rate']}% aa  (delta 3m: {ctx['selic_delta_3m']:+.2f}pp)")
print(f"  BRL      : {ctx['impact_brl']}")
print(f"  Acoes    : {ctx['impact_stocks']}")
print(f"  Telegram : {format_telegram_line()}".encode("ascii", "replace").decode())
print()

# Score adjustment for VALE3 BUY signal
adj = copom_score_adjustment("VALE3.SA", 1)
print(f"  VALE3 BUY adjustment : {adj:+.0f} pts")

adj_eur = copom_score_adjustment("EURUSD=X", 1)
print(f"  EUR/USD BUY adjustment: {adj_eur:+.0f} pts (nao afetado)")
print()

# Send Telegram with Copom line
print("Enviando alerta Telegram com linha Copom...")
notifier = TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)

alert = SignalAlert(
    signal      = "BUY",
    asset       = "VALE3",
    timeframe   = "H1",
    price       = 62.50,
    stop_loss   = 60.80,
    take_profit = 65.30,
    win_rate    = 47.2,
    confidence  = 2,
    rsi         = 38.5,
    atr         = 1.20,
    flag        = "🇧🇷",
    decimals    = 2,
    sentiment   = "BULLISH",
    sentiment_confidence = 0.7,
    sentiment_impact     = "MEDIUM",
    final_score = 62.0,
    analysis    = "Selic em queda favorece ativo alavancado com P/L atrativo.",
    copom_tone  = ctx["copom_tone"],
)
ok = notifier.send_signal_alert(alert)
print(f"  Telegram: {'OK' if ok else 'FALHOU'}")
