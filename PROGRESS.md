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
- [x] **RAG Copom** — `core/macro_context.py` + `scripts/ingest_copom.py` (BCB API, tom DOVISH detectado, +10/-10 pts em ações BR)

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

### ✅ Fase 9 — RAG Copom / Contexto Macro BR (concluída 2026-05-06)
- [x] **Fonte de dados** — BCB API oficial (`api.bcb.gov.br/dados/serie/bcdata.sgs.4189`) — SELIC meta, sempre disponível
- [x] **Tom detectado: DOVISH** — Selic caindo de 14.90% → 14.40% (delta -0.29pp em 3 meses)
- [x] **GPT-4o-mini** confirma tom em 1 token a partir da trajetória da taxa
- [x] **Score BR ajustado** — `+10 pts` dovish | `-10 pts` hawkish nos ativos VALE3/PETR4/ITUB4/BBDC4/IBOV
- [x] **EUR/USD isolado** — Copom não afeta Forex (verificado via `BR_SYMBOLS`)
- [x] **Cache 6h** — `data/copom_tone.json` evita chamadas excessivas à BCB API
- [x] **ChromaDB Pythex** — contexto salvo em `pythex-ia-engine/clientes/copom/pdfs/` para RAG queries
- [x] **Telegram com linha Copom** — emoji 🕊/🦅 + tom + Selic em cada alerta de ativo BR
- [x] **PDFs automáticos** — slot preparado para quando o BCB disponibilizar URLs diretas (SPA Angular atual bloqueia scraping)

> Tom atual: DOVISH | Selic 14.40% aa | Ações BR: impacto positivo
> Atualizar: `.\venv\Scripts\python.exe scripts\ingest_copom.py --force`

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

# Copom — atualizar tom macro
.\venv\Scripts\python.exe scripts\ingest_copom.py        # usa cache se fresco (<6h)
.\venv\Scripts\python.exe scripts\ingest_copom.py --force  # força re-fetch BCB + LLM

# Pipeline MT5 (dados ao vivo → sinais → CSV)
.\venv\Scripts\python.exe scripts\run_pipeline.py

# Testes
.\venv\Scripts\pytest tests/ -v
```

---

## 🔒 FASE DE INCUBAÇÃO — Sistema Congelado (2026-05-06 a 2026-06-06)

**Status:** Sistema CONGELADO por 30 dias para observação ao vivo.
**Regra absoluta:** nenhuma alteração em modelo, features, target, thresholds ou filtros.

### O que foi validado (experimentos quantitativos)

| Módulo | Resultado |
|--------|-----------|
| **Auditoria leakage** | LSTM: scaler bug corrigido (fit em treino apenas). XGBoost: limpo. |
| **Walk-forward 3 folds** | N=2H k=0.7: E=+0.0001 a +0.0002, PF=1.16–1.25, Sharpe>3 |
| **Anti-snooping holdout** | 3/3 combos sobrevivem (100%), Sharpe 3.37–4.87 |
| **Filtro de regime** | ADX<35 + atr_ratio=(0.8,1.5) → Holdout: PF 1.16→1.43, Sharpe 3.37→7.15 |
| **SHAP consistência** | Features estáveis: `atr_ratio`, `adx`, `bb_pct`, `momentum_3h`, `atr`, `macd_hist` |
| **IBOV paralelo** | N=2H k=0.7: E=+0.0006→+0.0012, PF 1.17→1.32 com filtro |

### Diagnóstico Fold1 (período tóxico set2024–fev2025)

| Causa | Fold1 | Holdout | Status |
|-------|-------|---------|--------|
| EMA200 slope | -0.289 pips/candle (downtrend) | +0.026 (flat) | **TOXIC** |
| Max unidirectional | 191 pips | 159 pips | **TOXIC** |
| ADX>35 freq | 14.3% | 19.0% | similar |

**Conclusão:** Fold1 falhou por forte viés direcional (EUR/USD em tendência de baixa). Sinais de reversão (RSI oversold + BB touch) foram penalizados pela tendência persistente. O filtro ADX<35 é uma medida paliativa — o root cause é o regime macro direcional.

### Fractal Lab — Diagnóstico de Microestrutura (Sistema 2, 2026-05-07)

Análise 5m vs 15m em janelas de 1h usando `research/fractal_lab/` (Sistema 1 não tocado).

| Métrica | Fold1 (dez2024–fev2025) | Holdout (mar–mai2026) | Delta |
|---------|------------------------|-----------------------|-------|
| **disorder_score** | **34.9** | 25.2 | **+9.7 (+38%)** |
| **coherence_score** | **36.4** | 43.3 | **−6.9 (−16%)** |
| Janelas analisadas | 980–1045 | 912–1014 | — |

**Hipótese confirmada:** A microestrutura do Fold1 era estruturalmente mais caótica.
- O 5m invertia direção com 38% mais frequência que no Holdout
- O 5m estava 16% menos alinhado com a tendência do 15m
- Mercado caótico + EMA200 em downtrend = sinais de reversão sistematicamente penalizados

**Implicação:** disorder_score alto não causou as perdas isoladamente, mas é um indicador antecipado de regime adverso para a estratégia do Sistema 1. Em mercados com disorder > ~30, a taxa de acerto de reversões cai.

> Script: `.\venv\Scripts\python scripts\export_mt5_fractal.py`
> Dados: `data/fractal_cache/EURUSD_5m_fold1.csv` (12.539 candles MT5) + yfinance holdout
> Relatório: `data/fractal_report.json`

### Configuração congelada

```python
# Melhores combos validados (NÃO ALTERAR)
# N=2H, k=0.7, balanced=True  → E=+0.0001 sel | E=+0.0003 holdout | PF=1.432 | Sharpe=7.15
# N=2H, k=0.7, balanced=False → E=+0.0002 sel | E=+0.0003 holdout | PF=1.402 | Sharpe=6.72

# Filtro de regime (observacional — não bloqueia trades ainda)
REGIME_ADX_THRESHOLD  = 35.0
REGIME_ATR_RATIO_MIN  = 0.8
REGIME_ATR_RATIO_MAX  = 1.5
```

### O que a telemetria monitora (data/telemetry.json)

- Todo sinal gerado + se regime filter teria bloqueado
- Para sinais bloqueados: resultado hipotético 2H depois (ganho ou perda)
- PF rolling dos últimos 20 trades (alerta se < 1.0)
- Drawdown diário + breakdown por sessão (Ásia/Londres/NY)
- Slippage real vs backtest

### Alertas Telegram automáticos

| Condição | Mensagem |
|----------|----------|
| PF rolling < 1.0 | ⚠️ PF Rolling caiu abaixo de 1.0 — monitorar |
| Trade bloqueado resolvido | ✅ Trade bloqueado pelo filtro — seria GANHO/PERDA (+X pips) |

### Decisão pós-incubação (2026-06-06)

- **Se filtro acertou ≥65% dos bloqueios** → Ativar filtro em produção
- **Se PF rolling > 1.2 sem filtro** → Manter sistema atual, filtro optional
- **Se PF rolling < 0.8** → Investigar regime shift antes de qualquer mudança
- **Nunca retreinar** sem nova auditoria walk-forward completa
- **Fractal Lab (novo):** Testar disorder_score como filtro adicional — bloquear sinais quando disorder_5m_vs_15m > 30 (limiar baseado na distribuição do Fold1: p50=35.0)

---

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
