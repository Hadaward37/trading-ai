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

---

## Fractal Lab — Laboratório Quantitativo (research/)

### ✅ Fractal Lab Core Engine MVP (2026-05-11)
- [x] **Orchestrator** — `research/fractal_lab/core/orchestrator.py` · interface: `run_hypothesis(df, hypothesis) → ResearchReport`
- [x] **Pipeline** — features → regimes → backtest → walk-forward → Monte Carlo → robustness
- [x] **Feature Engine** — `base_features.py` · ATR, EMA20/50/200, slope, volatility, RSI, BB, momentum
- [x] **4 regimes** — `regime_detector.py` · TRENDING | MEAN_REVERTING | HIGH_VOLATILITY | LOW_VOLATILITY
- [x] **Backtest Engine** — candle-by-candle, sem look-ahead, signal t-1 / entrada t
- [x] **Walk-Forward** — janela expansiva, N folds temporais, sem random split
- [x] **Monte Carlo** — bootstrap de trades, distribuição de equity curve (1.000 simulações)
- [x] **Robustness Score** — 0–1: consistência × estabilidade × MC × DD risk
- [x] **SQLite Storage** — `data/fractal_lab.db`, queries por hipótese e timestamp
- [x] **12 entry types** — mean_reversion, trend_follow, breakout, oscillator_extremes, ema_slope, momentum_atr, volatility_breakout, range_breakout, e mais

### ✅ Batch Runner — Stress Test Multi-Asset (2026-05-12)
- [x] **10 ativos** — EURUSD, Gold, NASDAQ, ITUB4, PETR4, IBOV, SPY, QQQ, BTC, DXY
- [x] **8 hipóteses** — 3 grupos: Mean Reversion (3) + Trend Following (3) + Volatility (2)
- [x] **80 combinações** rodadas com train/test/OOS separados deterministicamente
- [x] **Output por run** — `results_matrix.csv`, `results_full.json`, `ranking.csv`, `heatmap_ascii.txt`
- [x] **CLI flexível** — `--assets`, `--hypotheses`, `--dry-run`

**Resultados chave (80 combinações):**
- 39/80 passaram OOS (PASS ou OOS_PASS)
- Robustez máxima: BTC×MR_RSI_Exhaust (0.890), GOLD×TF_Momentum (0.870), EURUSD×MR_RSI_Exhaust (0.828)
- MR_RSI_Exhaust e MR_EMA_Dist: sobrevivem em 7+ ativos diferentes

### ✅ Regime Analytics + Failure Clusters (2026-05-12)
- [x] **RegimeAnalytics** — `analytics/regime_analytics.py` · per-regime PF/Sharpe/N, stability ranking, behavioral market map, OOS degradation by regime
- [x] **FailureClusterer** — `analytics/failure_clusters.py` · KMeans (k automático) + DBSCAN outliers, auto-naming de clusters, structural insights
- [x] **Output** — `research/results/regime_analytics/` · per_regime_metrics.csv, stability_ranking.csv, behavioral_market_map.csv, cluster_assignments.csv

**Behavioral Market Map emergente (10 ativos × 4 regimes):**

| Asset | TRENDING | MEAN_REVERTING | HIGH_VOL | LOW_VOL |
|-------|---------|----------------|----------|---------|
| EURUSD | MR (RSI) | MR (BB) | MR (RSI) | MR (RSI) |
| DXY | MR (BB) | MR (BB) | MR (Range) | MR (EMA) |
| BTC | MR (RSI) | TF (Slope) | MR (RSI) | MR (Slope) |
| GOLD | TF (Mom) | TF (Mom★) | MR (Brk) | MR (BB) |
| NASDAQ | TF (Mom) | TF (RSI) | VOL (ATR) | TF (Brk) |
| SPY | MR (RSI) | — | VOL (Range) | — |
| QQQ | TF (Mom) | MR (EMA) | MR (RSI) | — |
| ITUB4 | TF (Mom) | MR (RSI) | MR (RSI) | — |
| PETR4 | TF (Slope) | — | TF (Mom) | — |
| IBOV | TF (Brk) | — | MR (RSI) | — |

★ GOLD MEAN_REVERTING: TF_Momentum PF 2.635 — exceção notável

**Stability Ranking (consistência de PF cross-regime, não maior PF):**
1. TF_Breakout_20: 0.774 — estável mas PF médio 0.94 (edge fraco)
2. MR_EMA_Dist: 0.681 — melhor equilíbrio estabilidade × edge
3. MR_BB_Stretch: 0.669 — consistente, edge positivo
4. MR_RSI_Exhaust: 0.662 — alta robustez absoluta, menor estabilidade cross-regime

**Clusters de Falha identificados (41 falhas):**
- **DATA_MINING_OVERFIT** (8): test PF bom, OOS colapsou −35.8% — não confiar em PF test isolado
- **STRUCTURAL_FAILURE** (27): hipótese incompatível com o ativo — rejeitar
- **VOLATILITY_TRAP** (6): execução em HIGH_VOL com spread real eliminaria o edge
- **DBSCAN outliers** (9): VOL_ATR_Expand em maioria — poucos trades, estatisticamente inválido

**Regime mais perigoso para degradação OOS:** MEAN_REVERTING (até −71% em alguns casos)
**Regime mais estável:** TRENDING (menor variância test→OOS)

---

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

### ✅ Fractal Lab (Sistema 2) — Pesquisa Concluída (2026-05-07)

**Achado principal:** Sistema 1 é um "capturador de reversão confirmada" por design.
O `BB_DEV=2.0` é filtro de exaustão implícito — aguarda 2σ (~2.4 ATR) para confirmar
reversão. Não é delay acidental: é proteção estrutural contra entradas prematuras.

**Candidato identificado para pós-incubação:** Adaptive Sizing baseado em stretch.

| Métrica | Baseline 1.0x | Adaptive B5=0.1x |
|---------|---------------|-----------------|
| MaxDD (train) | 1446 pips | **254 pips (−82%)** |
| MaxDD (test OOS) | 239 pips | **44 pips (−81%)** |
| Sharpe proxy | 0.027 (train) | 0.004 → consistente |
| B5 edge | PF=1.06 (Jan-Mai2026) | Regime-dependente |

**Status da hipótese:**
- MaxDD robustamente reduzido out-of-sample ✅
- B5=0.0x vs B5=0.1x: inconclusivo (n_test=37 insuficiente) ⚠️
- Adaptive sizing não altera lógica de sinal nem thresholds do Sistema 1 ✅

> Detalhes: `research/fractal_lab/findings.md`
> Walk-forward: `.\venv\Scripts\python scripts\adaptive_sizing_walkforward.py`

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

Análise usando `research/fractal_lab/` — v1 (5m/15m, 60min) e v2 (1m/5m, 15min). Sistema 1 não tocado.

#### Resultado v1 inicial — CORRIGIDO (artifact de fonte de dados)

A primeira versão comparou Fold1 com fonte MT5 e Holdout com fonte yfinance. Valores inválidos para comparação.

| Métrica | Fold1 (MT5) | Holdout (yfinance) | Delta | Status |
|---------|-------------|-------------------|-------|--------|
| disorder_score | 34.9 | 25.2 | +38% | ~~confirmado~~ **ARTIFACT** |
| coherence_score | 36.4 | 43.3 | −16% | ~~confirmado~~ **ARTIFACT** |

**Causa do artifact:** yfinance suaviza os dados 5m de forma diferente do feed real do MT5, produzindo disorder artificialmente mais baixo no Holdout.

#### Resultado v2 corrigido — mesma fonte (MT5) para ambos os períodos

| Métrica | Fold1 (MT5 5m) | Holdout (MT5 5m) | Delta |
|---------|---------------|-----------------|-------|
| disorder_score v1 (5m/15m) | **34.9** | **35.8** | −0.9 ≈ igual |
| coherence_score v1 (5m/15m) | 36.4 | 36.1 | +0.3 ≈ igual |
| disorder_score v2 (1m/5m) | N/A* | **33.4** | — |
| coherence_score v2 (1m/5m) | N/A* | 36.4 | — |

*Fold1 1m indisponível na Clear (retenção ~60 dias; Fold1 é 15+ meses atrás).

**Hipótese REVISADA:** disorder_score **não discrimina** Fold1 do Holdout quando usada mesma fonte de dados. Os níveis de microestrutura são similares entre os dois períodos.

**Robustez entre granularidades (Holdout):** delta disorder entre 1m/5m e 5m/15m = −2.4. O sinal é estável independentemente da granularidade usada.

**Discriminador real confirmado: EMA200 slope**
O Fractal Lab validou que a microestrutura não é o diferencial — reforçando o diagnóstico original: o root cause do Fold1 foi o **viés direcional macro** (EMA200 slope −0.289 pips/candle). Sinais de reversão falham sistematicamente em tendências fortes, independentemente da ordem do 5m.

> Scripts: `.\venv\Scripts\python scripts\export_mt5_fractal.py [--skip-export] [--fold1-attempt]`
> Dados: `data/fractal_cache/EURUSD_{5m_fold1,1m_holdout,5m_holdout}.csv`
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
- **Fractal Lab (revisado):** disorder_score descartado como filtro — não discrimina Fold1 do Holdout em fonte consistente (MT5). disorder robusto entre granularidades (delta 1m/5m vs 5m/15m = −2.4).
- **EMA200 slope (novo):** Investigar EMA200 slope como filtro adicional — bloquear sinais de reversão quando slope < −0.10 pips/candle (abaixo da metade do valor tóxico do Fold1: −0.289)
- **Adaptive Sizing (PRIORITÁRIO — candidato validado):** Walk-forward confirmou MaxDD −82% com B5=0.1x. Implementar tabela de sizing baseada em stretch (ver seção Achado crítico). Usar B5=0.1x (não 0.0x) para manter exposição em regimes favoráveis.

### Achado crítico — Fractal Lab v3: Stretch Analysis (2026-05-07)

**Causa do timing de entrada:** `BB_DEV=2.0` em `config.py` + `BB_WINDOW=20` cria banda de 2.41 ATR de largura média. Quando `v_bb_buy` (em `core/signals.py` linha 43) dispara, o preço já está em B5 (>2 ATR da média). Este comportamento é **estruturalmente correto** — não é um bug, é um filtro implícito de exaustão.

**Evidência quantitativa (310 trades, jan–mai2026):**

| Bucket | N | % | PF | E (pips) | Win Rate |
|--------|---|---|----|----------|----------|
| B1–B2 [0–1.0 ATR] | 33 | 10.6% | 5.57 | +28.4 | 76% |
| B3–B4 [1–2.0 ATR] | 42 | 13.6% | 3.25 | +21.2 | 69% |
| **B5 [>2 ATR]** | **235** | **75.8%** | **1.06** | **+1.3** | **43%** |

KS-test B1 vs B5: p=0.016 — **estatisticamente significativo**.

**Simulação: entrada antecipada em B2 PIORA o sistema:**
- Win Rate: 51.8% → 23.5% (−28.2%), MaxDD: +680 pips
- O preço tipicamente continua de B2 para B5 (~3 ATR adicionais), derrubando o SL (2.5 ATR) antes de reverter.

**Solução: Adaptive Sizing** — não mudar timing, mudar tamanho de posição:

| Bucket | Sizing | PF atual |
|--------|--------|----------|
| B1 [0.0–0.5) | 1.0x | 5.53 |
| B2 [0.5–1.0) | 1.0x | 5.61 |
| B3 [1.0–1.5) | 0.7x | 3.59 |
| B4 [1.5–2.0) | 0.3x | 3.02 |
| **B5 [2.0+]** | **0.0x** | 1.06 |

**Impacto simulado (310 trades):** MaxDD: 487 → **30 pips (−94%)**, Sharpe: 0.21 → **0.50 (+142%)**.

> Detalhes: `research/fractal_lab/findings.md`
> Simulação: `.\venv\Scripts\python scripts\simulate_early_entry.py`
> Relatórios: `data/regime_analysis.json`, `data/early_entry_simulation.json`

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
