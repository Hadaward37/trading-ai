"""Centralised project configuration.

Re-optimised parameters (2026-05-04, v2 — ADX + Stochastic filters):
  RSI(14) | Buy < 35 | Sell > 75 | SL 2.5x ATR | TP 4.0x ATR
  ADX Trend threshold = 25 | Stoch Oversold < 25
  -> Sharpe 1.299 | Win Rate 47.2% | Max DD -0.50% | 212 trades (1h)
  (v1 no-filter baseline: Sharpe 1.494, Win Rate 45.2%, 425 trades)
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent

# ── Data source ───────────────────────────────────────────────────────────────
SYMBOL      = "EURUSD=X"
TIMEFRAMES  = ("15m", "1h", "4h")   # 4h is resampled from 1h internally
DEFAULT_TF  = "1h"
DB_PATH     = ROOT_DIR / "data" / "trading.db"

# ── RSI  (re-optimised — v2 with ADX + Stoch filters) ────────────────────────
RSI_PERIOD = 14
RSI_BUY    = 35   # raised from 30 (v1) — better Win Rate with stoch filter
RSI_SELL   = 75

# ── MACD ──────────────────────────────────────────────────────────────────────
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGN = 9

# ── Bollinger Bands ───────────────────────────────────────────────────────────
BB_WINDOW    = 20
BB_DEV       = 2.0
BB_TOLERANCE = 0.002

# ── ATR ───────────────────────────────────────────────────────────────────────
ATR_WINDOW = 14

# ── ADX ───────────────────────────────────────────────────────────────────────
ADX_WINDOW = 14

# ── Stochastic ────────────────────────────────────────────────────────────────
STOCH_K_WINDOW   = 14
STOCH_SMOOTH_K   = 3
STOCH_SMOOTH_D   = 3
STOCH_OVERSOLD   = 25   # raised from 20 (v1) — allows slightly more signals
STOCH_OVERBOUGHT = 75   # symmetric with oversold

# ── Market Regime ─────────────────────────────────────────────────────────────
REGIME_ADX_TREND            = 25    # ADX >= this -> trending
REGIME_ADX_RANGE            = 20    # ADX <  this -> range
REGIME_VOLATILITY_THRESHOLD = 1.5   # ATR / ATR_50ma above this -> high vol
REGIME_BLOCK_RANGE_SIGNALS  = True  # suppress BUY/SELL entries during Range

# ── Multi-Timeframe ───────────────────────────────────────────────────────────
MTF_MIN_AGREEMENTS = 2   # number of TFs that must agree for a valid MTF signal

# ── Signal score weights (must sum to 100) ────────────────────────────────────
SCORE_WEIGHT_RSI   = 25
SCORE_WEIGHT_MACD  = 25
SCORE_WEIGHT_BB    = 20
SCORE_WEIGHT_ADX   = 15
SCORE_WEIGHT_STOCH = 15

# ── Backtest defaults (optimised) ─────────────────────────────────────────────
INITIAL_CAPITAL   = 10_000.0
POSITION_SIZE_PCT = 0.10
SL_ATR_MULT       = 2.5
TP_ATR_MULT       = 4.0

# ── Dashboard ─────────────────────────────────────────────────────────────────
CACHE_TTL_SECONDS = 900

# ── Telegram alerts ───────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID:   str = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Scheduler ─────────────────────────────────────────────────────────────────
SCHEDULER_INTERVAL_MIN = 15
SCHEDULER_TIMEFRAME    = "1h"
