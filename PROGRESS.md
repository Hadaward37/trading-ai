# Trading AI — Log de Progresso

## Sistema
- **Ativo:** EUR/USD
- **Timeframes:** 15m, 1H, 4H (4H resampled do 1H)
- **Storage:** SQLite local (`data/trading.db`)
- **Dashboard:** Streamlit + Plotly (`streamlit run dashboard/app.py`)
- **Alertas:** Telegram (`@tradingai_dudu_bot`, chat_id em `.env`)
- **Repositório:** https://github.com/Hadaward37/trading-ai

## Parâmetros Otimizados (v2 — com ADX + Stoch filters)
| Parâmetro | Valor |
|-----------|-------|
| RSI período | 14 |
| RSI Buy < | 35 |
| RSI Sell > | 75 |
| Stop Loss | 2.5× ATR |
| Take Profit | 4.0× ATR |
| ADX Threshold | 25 (bloqueia Range) |
| Stoch Oversold | 25 |
| **Sharpe** | **1.299** |
| **Win Rate** | **47.2%** |
| **Trades** | **212** |

> v1 baseline (sem filtros): Sharpe 1.494 · Win Rate 45.2% · 425 trades

## Módulos Implementados
- [x] Coleta OHLCV — `core/collector.py` (yfinance, 15m/1H/4H)
- [x] Indicadores — `core/indicators.py` (RSI, MACD, BB, ATR, ADX, Estocástico)
- [x] Regime de mercado — `core/regime.py` (Tendência Alta/Baixa, Range, Alta Vol)
- [x] Geração de sinais — `core/signals.py` (score 0-100, filtro de regime)
- [x] Multi-timeframe — `core/collector.get_mtf_confluence()` (2-de-3 consenso)
- [x] Backtesting — `core/backtest.py` (ATR-based SL/TP, equity curve)
- [x] Optimizer v1 — `core/optimizer.py` (768 combos, Sharpe 1.494)
- [x] Optimizer v2 — `core/optimizer.py` (432 combos + ADX/Stoch, Sharpe 1.299)
- [x] Dashboard Streamlit — `dashboard/app.py` (5 painéis + regime + MTF)
- [x] Alertas Telegram — `core/notifier.py` + `core/scheduler.py`
- [x] Config centralizado — `config.py`
- [ ] Filtro de notícias/eventos econômicos
- [ ] Ações brasileiras (Ibovespa, VALE3, PETR4)
- [ ] Integração TradingView / Pine Script export

## Estrutura do Projeto
```
trading-ai/
├── config.py               # Todos os parâmetros centralizados
├── main.py                 # CLI: python main.py --timeframe 1h
├── run_scheduler.py        # Bot Telegram: python run_scheduler.py
├── core/
│   ├── collector.py        # yfinance + resample 4H + MTF confluence
│   ├── indicators.py       # RSI, MACD, BB, ATR, ADX, Stoch
│   ├── signals.py          # Votação 2/3 + score 0-100 + regime filter
│   ├── regime.py           # Detecção de regime de mercado
│   ├── backtest.py         # Engine de backtest vetorizado
│   ├── optimizer.py        # Grid search v2 (ADX + Stoch)
│   ├── notifier.py         # TelegramNotifier
│   └── scheduler.py        # Loop 15min, anti-spam
├── db/
│   └── database.py         # SQLAlchemy + SQLite
├── dashboard/
│   └── app.py              # Streamlit 5-painéis + regime + MTF
├── tests/
│   └── test_indicators.py  # pytest (6/6 passing)
├── data/
│   ├── trading.db          # SQLite (gitignored)
│   └── optimization_results.csv
├── .env                    # Telegram credentials (gitignored)
└── .env.example
```

## Comandos Rápidos
```powershell
cd C:\Users\dudut\trading-ai

# Pipeline completo
.\venv\Scripts\python main.py --timeframe 1h

# Dashboard
.\venv\Scripts\streamlit run dashboard\app.py

# Bot Telegram (background)
start /B .\venv\Scripts\python run_scheduler.py > scheduler.log 2>&1

# Optimizer
.\venv\Scripts\python -m core.optimizer

# Testes
.\venv\Scripts\pytest tests/ -v
```

## Próximos Passos
1. **Filtro de notícias** — desativar sinais em janelas de high-impact (FOMC, NFP, CPI)
2. **Ações brasileiras** — adicionar VALE3.SA, PETR4.SA, IBOV via yfinance
3. **TradingView** — exportar estratégia em Pine Script para backtesting visual

## Como Retomar
Ao iniciar nova conversa com Claude, cole:

> "Estamos desenvolvendo o projeto trading-ai. Repositório: github.com/Hadaward37/trading-ai. Consulte o PROGRESS.md e continue de onde paramos."
