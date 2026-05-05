# Trading AI — Log de Progresso

## Sistema
- **Ativos:** EUR/USD (Forex) | VALE3, PETR4, ITUB4, BBDC4, IBOV (B3)
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

### Alternativa Interessante — #3 (maior Win Rate)
| Parâmetro | Valor |
|-----------|-------|
| RSI Buy < | 30 |
| RSI Sell > | 75 |
| Stop Loss | 2.5× ATR |
| Take Profit | 3.0× ATR |
| ADX Threshold | 20 |
| Stoch Oversold | 25 |
| **Win Rate** | **52.7%** |
| **Trades** | **205** |

> Opção conservadora: menor TP (3.0×) e ADX mais permissivo (20) resultam em win rate mais alto às custas de menor Sharpe. Considerar para perfis avessos a drawdown.

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
- [x] Filtro de notícias/eventos econômicos — `core/news_filter.py` (FF JSON + BS4, cache 1h, ±30 min)
- [x] Ações brasileiras — `config.ASSETS` (VALE3, PETR4, ITUB4, BBDC4, IBOV via yfinance)
- [x] Integração TradingView / Pine Script — `tradingview/strategy.pine` + `tradingview/indicator.pine`

### ✅ Fase 2 — Inteligência de Notícias (concluída 2026-05-05)
- [x] **Groq API (Llama 3.3 70B)** — análise de sentimento gratuita (14.400 req/dia)
- [x] **Pipeline completo**: Yahoo Finance RSS coleta → Groq classifica → score integrado
- [x] **Output estruturado**: `sentiment` (BULLISH/BEARISH/NEUTRAL) · `confidence` · `impact` · `key_factors`
- [x] **final_score**: score técnico ±15 pts conforme alinhamento sentimento × sinal
- [x] **Fallback inteligente**: keyword-based quando API indisponível
- [x] **Sentimento no Telegram**: alerta inclui sentimento + confiança + impacto
- [x] **Cache por ticker**: TTL 5 min (evita chamadas excessivas de API)

> **Nota:** GPT-4o e Gemini pendentes de créditos — Groq substituindo gratuitamente com qualidade equivalente.
> Prioridade de engines: GPT-4o → Groq → Gemini → keyword fallback

## Estrutura do Projeto
```
trading-ai/
├── config.py               # Todos os parâmetros centralizados
├── main.py                 # CLI: python main.py --timeframe 1h
├── run_scheduler.py        # Bot Telegram: python run_scheduler.py
├── core/
│   ├── collector.py        # yfinance + resample 4H + MTF confluence
│   ├── indicators.py       # RSI, MACD, BB, ATR, ADX, Stoch
│   ├── signals.py          # Votação 2/3 + score 0-100 + sentiment final_score
│   ├── regime.py           # Detecção de regime de mercado
│   ├── backtest.py         # Engine de backtest vetorizado
│   ├── optimizer.py        # Grid search v2 (ADX + Stoch)
│   ├── news_filter.py      # Filtro ForexFactory (bloqueia FOMC/NFP/CPI)
│   ├── news_intelligence.py# Fase 2: RSS + Groq/GPT/Gemini → sentimento
│   ├── notifier.py         # TelegramNotifier (inclui sentimento)
│   └── scheduler.py        # Loop 15min, anti-spam, multi-asset
├── db/
│   └── database.py         # SQLAlchemy + SQLite
├── dashboard/
│   └── app.py              # Streamlit 5-painéis + regime + MTF
├── tests/
│   └── test_indicators.py  # pytest (6/6 passing)
├── data/
│   ├── trading.db          # SQLite (gitignored)
│   └── optimization_results.csv
├── .env                    # Credenciais (gitignored)
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

### Fase 3 — Validação de Risco com Claude API
- [ ] Claude API como validador antes de cada sinal
- [ ] Checagem: "dado o contexto macro, essa operação é segura?"
- [ ] Relatório diário automático de performance
- [ ] Sugestão de ajustes na estratégia via Claude

### Fase 4 — Arquitetura Multi-Agente Completa
- [ ] Groq/GPT-4o → sentimento estruturado em JSON
- [ ] Sistema Python → cruza técnico + sentimento
- [ ] Claude → valida risco final
- [ ] Sinal final → Telegram + TradingView

### Fase 5 — Paper Trading e Validação
- [ ] Diário de operações automático
- [ ] Comparar sinais vs resultado real do mercado
- [ ] Meta: 2-3 meses de paper trading consistente

### Fase 6 — Servidor 24/7
- [ ] Migrar para Oracle Cloud ou Railway
- [ ] Sistema rodando sem depender do Samsung

## Como Retomar
Ao iniciar nova conversa com Claude, cole:

> "Estamos desenvolvendo o projeto trading-ai. Repositório: github.com/Hadaward37/trading-ai. Consulte o PROGRESS.md e continue de onde paramos."
