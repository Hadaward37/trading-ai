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
- [x] **MetaTrader 5** — `core/mt5_connector.py` (OHLCV M15/H1/H4 ao vivo) + `mql5/TradingAI_Signals.mq5` (EA anota sinais no gráfico)
- [x] **Pythex Bridge** — `core/pythex_bridge.py` (GPT-4o-mini, análise executiva por sinal, 1 frase ≤200 chars)
- [x] **LSTM Neural Network** — `core/lstm_model.py` + `core/lstm_trainer.py` (55.41% acurácia, 12.260 amostras, retreino semanal automático)

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

### ✅ Fase 7 — MetaTrader 5 + Pythex IA (concluída 2026-05-06)
- [x] **MT5 conectado** — `core/mt5_connector.py` busca OHLCV ao vivo (M15/H1/H4) via API Python
- [x] **Sinais exportados** — `data/mt5_signals.csv` + `data/mt5_signals.json` (310 sinais gerados)
- [x] **EA MQL5** — `mql5/TradingAI_Signals.mq5` lê CSV a cada 15s e desenha setas de entrada, linhas SL/TP e score no gráfico MT5
- [x] **Pythex integrado** — `core/pythex_bridge.py` como base de conhecimento de trading
- [x] **GPT-4o-mini** — análise executiva gerada por sinal (max 55 tokens, 1 frase ≤200 chars)
- [x] **Telegram com análise** — campo `analysis` em `SignalAlert`, bloco "Analise Pythex" no alerta
- [x] **Conta demo MetaQuotes** — MT5 instalado e conectado localmente

> Configuração: `USE_MT5 = True` em `config.py` · EA em `MQL5/Experts/TradingAI_Signals.mq5`

### ✅ Fase 8 — LSTM Neural Network (concluída 2026-05-06)
- [x] **Modelo LSTM** — arquitetura LSTM(64)→Dropout→LSTM(32)→Dropout→Dense(1,sigmoid)
- [x] **Features (8)** — RSI, MACD_hist, BB_pct, ATR_norm, ADX, Stoch_K, Volume_norm, Sentiment_conf
- [x] **Treinamento** — 12.260 amostras de signals_1h SQLite | split cronológico (sem lookahead)
- [x] **Acurácia inicial: 55.41%** — +5.4% acima do baseline aleatório de 50%
- [x] **Pesos v3** — RSI(20%) + MACD(20%) + BB(15%) + ADX(12%) + Stoch(12%) + **LSTM(21%)** = 100%
- [x] **Retreino automático semanal** — `should_retrain()` detecta modelo com >7 dias
- [x] **Dashboard aba LSTM** — gauge de probabilidade de alta, acurácia por epoch, pizza de pesos, botão de retreino
- [x] **Fallback seguro** — retorna 50% (neutro) quando modelo indisponível; não bloqueia sinais

> Acurácia melhora com mais dados acumulados pelo scheduler (meta: >58% com 6+ meses de candles)
> Retreinar manualmente: `.\venv\Scripts\python.exe -m core.lstm_trainer --force`

### 🚀 Fase 5 — Paper Trading e Validação (iniciada 2026-05-05)
- [x] **Paper trading implementado** — `core/paper_trading.py` com SQLite WAL
- [x] **Carteira virtual R$ 10.000** — posição 10% do capital por trade
- [x] **SL/TP automático** — monitorado a cada ciclo do scheduler (15 min)
- [x] **Diário automático** — `core/trade_journal.py` + CSV diário em `data/journal/`
- [x] **Dashboard aba Paper Trading** — KPIs, posições abertas, gráfico saldo, histórico
- [x] **Alertas Telegram** — notifica quando TP ou SL é atingido com P&L
- [ ] **Meta: 50–100 trades** para validação estatística
- [ ] Comparar sinais vs resultado real do mercado
- [ ] Ajuste de parâmetros baseado em performance real

## Estrutura do Projeto
```
trading-ai/
├── config.py                # Todos os parâmetros centralizados
├── main.py                  # CLI: python main.py --timeframe 1h
├── run_scheduler.py         # Bot Telegram: python run_scheduler.py
├── core/
│   ├── collector.py         # yfinance + resample 4H + MTF confluence
│   ├── indicators.py        # RSI, MACD, BB, ATR, ADX, Stoch
│   ├── signals.py           # Votação 2/3 + score 0-100 + sentiment final_score
│   ├── regime.py            # Detecção de regime de mercado
│   ├── backtest.py          # Engine de backtest vetorizado
│   ├── optimizer.py         # Grid search v2 (ADX + Stoch)
│   ├── news_filter.py       # Filtro ForexFactory (bloqueia FOMC/NFP/CPI)
│   ├── news_intelligence.py # Fase 2: RSS + Groq/GPT/Gemini → sentimento
│   ├── paper_trading.py     # Fase 5: carteira virtual, SL/TP, métricas
│   ├── trade_journal.py     # Fase 5: CSV diário, resumo do dia
│   ├── notifier.py          # TelegramNotifier (sentimento + análise Pythex)
│   ├── scheduler.py         # Loop 15min, paper trading integrado
│   ├── mt5_connector.py     # Fase 7: MT5 OHLCV ao vivo (M15/H1/H4)
│   ├── mt5_signals.py       # Fase 7: exporta sinais → CSV/JSON para o EA
│   └── pythex_bridge.py     # Fase 7: GPT-4o-mini análise executiva por sinal
├── mql5/
│   └── TradingAI_Signals.mq5  # EA MT5: lê CSV e anota sinais no gráfico
├── scripts/
│   ├── run_pipeline.py      # Pipeline MT5 completo (dados → sinais → CSV)
│   └── test_pythex_telegram.py  # Teste: bridge + Telegram
├── db/
│   └── database.py          # SQLAlchemy + SQLite
├── dashboard/
│   └── app.py               # Streamlit — Sinais + Paper Trading (2 abas)
├── tests/
│   └── test_indicators.py   # pytest (6/6 passing)
├── data/
│   ├── trading.db           # SQLite (gitignored)
│   ├── journal/             # CSVs diários (gitignored)
│   └── optimization_results.csv
├── .env                     # Credenciais (gitignored)
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

# LSTM — treinar / retreinar
.\venv\Scripts\python.exe -m core.lstm_trainer          # treina se ainda não existir
.\venv\Scripts\python.exe -m core.lstm_trainer --force  # força retreino

# Pipeline MT5 (dados ao vivo → sinais → CSV)
.\venv\Scripts\python.exe scripts\run_pipeline.py

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

### ✅ Fase 6 — Servidor 24/7 (concluída 2026-05-05)
- [x] **VM Oracle Cloud Always Free** — Ubuntu 22.04 LTS, ARM A1.Flex
- [x] **IP fixo:** `137.131.228.166` — Brazil East (São Paulo)
- [x] **Systemd** — restart automático, inicia no boot
- [x] **Sistema rodando 24/7** — Samsung pode ser desligado
- [x] **Primeiro ciclo confirmado** — 6 ativos, Groq sentimento, Telegram OK

> Deploy: `ssh -i C:\Users\dudut\.ssh\ssh-key-2026-05-05.key ubuntu@137.131.228.166`  
> Log: `tail -f ~/trading-ai/scheduler.log`  
> Atualizar: `cd ~/trading-ai && git pull && sudo systemctl restart trading-ai`

## Como Retomar
Ao iniciar nova conversa com Claude, cole:

> "Estamos desenvolvendo o projeto trading-ai. Repositório: github.com/Hadaward37/trading-ai. Consulte o PROGRESS.md e continue de onde paramos."
