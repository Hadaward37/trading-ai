"""Centralised project configuration.

All optimised parameters sourced from grid search (2026-05-04):
  RSI(14) | Buy < 30 | Sell > 75 | SL 2.5x ATR | TP 4.0x ATR
  -> Sharpe 1.494 | Win Rate 45.2% | Max DD -0.42% (timeframe 1h)
"""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).parent

# ── Data source ───────────────────────────────────────────────────────────────
SYMBOL      = "EURUSD=X"
TIMEFRAMES  = ("15m", "1h")
DEFAULT_TF  = "1h"
DB_PATH     = ROOT_DIR / "data" / "trading.db"

# ── RSI  (optimised — grid search 2026-05-04) ─────────────────────────────────
RSI_PERIOD = 14
RSI_BUY    = 30   # buy  signal when RSI < RSI_BUY
RSI_SELL   = 75   # sell signal when RSI > RSI_SELL

# ── MACD ──────────────────────────────────────────────────────────────────────
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGN = 9

# ── Bollinger Bands ───────────────────────────────────────────────────────────
BB_WINDOW    = 20
BB_DEV       = 2.0
BB_TOLERANCE = 0.002   # price within 0.2% of a band edge counts as a "touch"

# ── ATR ───────────────────────────────────────────────────────────────────────
ATR_WINDOW = 14

# ── Backtest defaults (optimised) ─────────────────────────────────────────────
INITIAL_CAPITAL   = 10_000.0
POSITION_SIZE_PCT = 0.10
SL_ATR_MULT       = 2.5   # stop-loss  distance in ATR multiples
TP_ATR_MULT       = 4.0   # take-profit distance in ATR multiples

# ── Dashboard ────────────────────────────────────────────────────────────────
CACHE_TTL_SECONDS = 900   # market-data cache lifetime (15 min)
