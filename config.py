"""Centralised project configuration.

Re-optimised parameters (2026-05-04, v2 — ADX + Stochastic filters):
  RSI(14) | Buy < 35 | Sell > 75 | SL 2.5x ATR | TP 4.0x ATR
  ADX Trend threshold = 25 | Stoch Oversold < 25
  -> Sharpe 1.299 | Win Rate 47.2% | Max DD -0.50% | 212 trades (1h)
  (v1 no-filter baseline: Sharpe 1.494, Win Rate 45.2%, 425 trades)

Multi-asset support (2026-05-04):
  EUR/USD (Forex) + VALE3, PETR4, ITUB4, BBDC4, IBOV (B3)
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent

# ── MetaTrader 5 integration ──────────────────────────────────────────────────
USE_MT5        = False             # desativado na incubação (EUR/USD removido do universo)
MT5_SYMBOL     = "EURUSD"          # MT5 symbol name (no "=X" suffix)
MT5_TIMEOUT_MS = 60_000            # connection timeout in milliseconds

# ── Data source ───────────────────────────────────────────────────────────────
SYMBOL      = "EURUSD=X"            # default / backward-compat (yfinance)
TIMEFRAMES  = ("15m", "1h", "4h")   # 4h is resampled from 1h internally
DEFAULT_TF  = "1h"
DB_PATH     = ROOT_DIR / "data" / "trading.db"

# ── Multi-asset universe ──────────────────────────────────────────────────────
# 2026-06-13: voltado para EUR/USD para VALIDAÇÃO DE EDGE.
# Motivo: a incubação B3 (22/05-13/06) nunca gerou dados — o cérebro
# (LSTM+XGBoost) e o edge validado (PF 1.432) são de EUR/USD, e o backtest
# mostrou B3 frágil (PF~1.03, robustez FRÁGIL). Ver knowledge-base/incident_20260613.md.
# Religar EUR/USD destrava a persistência (save_signals é gated em config.SYMBOL).
ASSETS: dict[str, str] = {
    "EURUSD": "EURUSD=X",
}

# Universo B3 — DESATIVADO em 2026-06-13 (sem modelo B3 + edge frágil no backtest).
# Restaurar só após treinar modelos B3 + migração de schema (signals_1h precisa de coluna symbol):
# ASSETS = {"VALE3": "VALE3.SA", "PETR4": "PETR4.SA", "ITUB4": "ITUB4.SA",
#           "BBDC4": "BBDC4.SA", "IBOV": "^BVSP"}

# Ativos usados apenas como benchmark (sinais são logados, mas NÃO geram trades no paper portfolio)
BENCHMARK_ASSETS: set[str] = {"IBOV"}

# Per-asset display metadata
ASSET_META: dict[str, dict] = {
    "EURUSD=X": {"name": "EUR/USD", "currency": "USD", "flag": "🌍", "decimals": 5, "exchange": "Forex"},
    "VALE3.SA": {"name": "VALE3",   "currency": "BRL", "flag": "🇧🇷", "decimals": 2, "exchange": "B3"},
    "PETR4.SA": {"name": "PETR4",   "currency": "BRL", "flag": "🇧🇷", "decimals": 2, "exchange": "B3"},
    "ITUB4.SA": {"name": "ITUB4",   "currency": "BRL", "flag": "🇧🇷", "decimals": 2, "exchange": "B3"},
    "BBDC4.SA": {"name": "BBDC4",   "currency": "BRL", "flag": "🇧🇷", "decimals": 2, "exchange": "B3"},
    "^BVSP":    {"name": "IBOV",    "currency": "BRL", "flag": "🇧🇷", "decimals": 0, "exchange": "B3"},
}

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
# v4 weights (2026-05-06): XGBoost ensemble added alongside LSTM
SCORE_WEIGHT_RSI      = 14
SCORE_WEIGHT_MACD     = 14
SCORE_WEIGHT_BB       = 11
SCORE_WEIGHT_ADX      = 10
SCORE_WEIGHT_STOCH    = 10
SCORE_WEIGHT_LSTM     = 20   # LSTM neural net — directional probability
SCORE_WEIGHT_XGBOOST  = 21   # XGBoost ensemble — directional probability

# ── LSTM model ────────────────────────────────────────────────────────────────
LSTM_ENABLED     = True
LSTM_LOOKBACK    = 60    # candles fed to the network
LSTM_MODEL_PATH  = "models/lstm_eurusd_1h.h5"
LSTM_SCALER_PATH = "models/lstm_scaler.pkl"

# ── XGBoost ensemble ──────────────────────────────────────────────────────────
XGBOOST_ENABLED      = True
XGBOOST_MODEL_PATH   = "models/xgboost_eurusd.pkl"
XGBOOST_REQUIRE_CONSENSUS = True   # cancel signal if LSTM and XGBoost disagree

# ── FinBERT NLP sentiment ─────────────────────────────────────────────────────
FINBERT_ENABLED = True
FINBERT_WEIGHT  = 0.70   # FinBERT 70% + Groq 30% when both available

# ── Copom macro context ───────────────────────────────────────────────────────
COPOM_ENABLED          = True
COPOM_DOVISH_BONUS     = 10.0  # pts added to buy score for BR assets when dovish
COPOM_HAWKISH_PENALTY  = 10.0  # pts removed from buy score for BR assets when hawkish
COPOM_TONE_TTL_HOURS   = 6     # refresh interval for copom_tone.json

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

# ── News filter ───────────────────────────────────────────────────────────────
NEWS_FILTER_ENABLED        = True
NEWS_FILTER_WINDOW_MINUTES = 30          # block N min before AND after event
NEWS_CACHE_TTL_SECONDS     = 3600        # refresh calendar at most once per hour
NEWS_FILTER_CURRENCIES     = ("USD", "EUR", "BRL")  # currencies whose events block signals

# ── Paper Trading ─────────────────────────────────────────────────────────────
PAPER_TRADING_ENABLED = True
PAPER_TRADING_CAPITAL = 10_000.0   # capital inicial em R$/USD

# ── Incubação 2026-05-22 → 2026-06-06 (strategy v1.0) ────────────────────────

# Custos de transação aplicados no paper trading
# 0.05% por lado = 0.10% round-trip (cobre emolumentos B3 + ISS + slippage estimado)
PAPER_TRADING_COMMISSION_PCT = 0.0005  # 0.05% por lado

# Threshold mínimo de confiança (escala 0-100, mesmo do final_score)
# Sinais com final_score >= threshold geram trade
# Sinais com final_score < threshold são LOGADOS mas NÃO executam trade
SIGNAL_CONFIDENCE_THRESHOLD = 55

# Versionamento da estratégia para análise pós-06/06
STRATEGY_VERSION = "v1.0"

# ── News Intelligence — Phase 2 (Gemini + GPT-4o) ────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY:   str = os.getenv("GROQ_API_KEY", "")
GEMINI_MODEL              = "gemini-2.0-flash"          # ajuste conforme quota disponível
GROQ_MODEL                = "llama-3.3-70b-versatile"   # free tier: 14400 req/dia
NEWS_INTELLIGENCE_ENABLED     = False   # desativado na incubação (rate limit invalida qualidade)
NEWS_INTELLIGENCE_TTL_SECONDS = 300     # cache per-ticker — 5 minutes
SENTIMENT_SCORE_WEIGHT        = 15      # max pts added/removed from technical score
