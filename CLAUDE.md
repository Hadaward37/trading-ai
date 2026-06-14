# ⚠️ INCUBAÇÃO B3 INVALIDADA — VALIDAÇÃO EUR/USD EM CURSO (desde 13/06/2026)

> **LEIA `knowledge-base/incident_20260613.md` ANTES DE QUALQUER AÇÃO.**
>
> A incubação B3 (22/05→13/06) **nunca coletou dados**: o scheduler rodava mas
> produzia HOLD puro em todos os ativos B3, e a persistência (`save_signals`)
> estava gated em EUR/USD (`config.SYMBOL`), que tinha sido removido do universo.
> 23 dias = zero sinais/trades B3. **A premissa "esperar até 06/07" morreu — não
> havia experimento vivo.** O freeze antigo abaixo está OBSOLETO.

## Estado atual (desde 13/06/2026)

| Item | Valor |
|---|---|
| Status | VALIDAÇÃO DE EDGE — EUR/USD — **stopgap v1.1 NO AR (14/06)** |
| Estratégia rodando | **v1.1**: XGBoost-only (LSTM off, consenso off). VM de 1GB não roda tensorflow |
| Decisão | Validar onde o cérebro e o edge existem (EUR/USD); B3 não tem modelo e o backtest deu edge FRÁGIL (PF~1.03) |
| Ativo ativo | **EUR/USD** (`EURUSD=X`) |
| B3 | DESATIVADO em `config.py` (preservado em comentário p/ restauro) |
| Cérebro na VM | XGBoost ✅ (xgb_prob real, ex: 44.9) \| LSTM ❌ (precisa TF → A1.Flex) |
| Próximo p/ v1.0 puro | migrar A1.Flex (Python 3.11+, TF) → reativar LSTM + `XGBOOST_REQUIRE_CONSENSUS` |
| Threshold de confiança | 55 (sobre `final_score`, escala 0-100) |
| Comissão paper | 0.05%/lado = 0.10% round-trip |
| Plano de capital | Demo só **depois** de validar o edge; B3 era preferência operacional, não restrição |
| Próximo passo se validar | port para B3 exige treinar modelos B3 + migração de schema (`signals_1h` sem coluna `symbol`) |

> **Healthcheck novo (13/06):** `check_signal_freshness()` alerta no Telegram se
> `signals_1h` congelar >72h — a instrumentação que teria pego o bug em 22/05.

---

<details><summary>Histórico: bloco de incubação B3 (OBSOLETO — mantido para contexto)</summary>

**Estado congelado (22/05/2026 → invalidado em 13/06/2026):** INCUBAÇÃO v1.0,
ativos VALE3/PETR4/ITUB4/BBDC4, IBOV benchmark, EUR/USD removido, revisão
agendada p/ 06/07. Janela estendida de 16→45 dias em 22/05. Tudo isso caiu com
o diagnóstico de 13/06 (ver incident_20260613.md).

</details>

## O que está rodando na Oracle (137.131.228.166)

- `trading-ai.service` → **ATIVO** (systemd, Restart=always)
- `polymarket-bot.service` → PAUSADO (disabled até 06/07)
- `polymarket-dashboard.service` → PAUSADO (disabled até 06/07)
- `trading-ai-observator` → PAUSADO (nohup, não reinicia no boot)
- Swap: 1 GB em `/swapfile` (ativado em 22/05/2026)
- Cron: `*/5 * * * *` → `fill_outcomes_job.py`

## Operações PERMITIDAS até 06/07

```bash
# Único SSH autorizado — healthcheck read-only
ssh -i ~/.ssh/ssh-key-2026-05-05.key ubuntu@137.131.228.166 \
  'tail -3 ~/trading-ai/logs/heartbeat.txt && \
   free -h && \
   wc -l ~/trading-ai/logs/signals.jsonl 2>/dev/null || echo "0 sinais"'
```

## Próxima ação pós-incubação (06/07/2026)

1. `wc -l logs/signals.jsonl` — quantos sinais coletados?
2. Abrir `notebooks/analise_06_06.ipynb` e rodar todas as células em ordem
3. Consultar `knowledge-base/edge_thesis.md` — aplicar critérios de validação
4. Consultar `knowledge-base/post_incubation_plan.md` — escolher cenário A-F
5. Consultar `knowledge-base/ops_runbook.md` — comandos de reativação

## Arquivos críticos desta incubação

| Arquivo | Função |
|---|---|
| `knowledge-base/edge_thesis.md` | Tese + critérios validação/invalidação |
| `knowledge-base/post_incubation_plan.md` | Decisão por cenário (A-F) |
| `knowledge-base/phase_2_proposal.md` | Proposta de research engine com LLM local (condicional, +90 dias pós-validação Sistema 1) |
| `knowledge-base/ops_runbook.md` | Comandos de reativação SSH |
| `logs/signals.jsonl` | Dataset em acúmulo (na Oracle) |
| `notebooks/analise_06_06.ipynb` | Notebook de análise pronto para rodar |

---

*(conteúdo original do CLAUDE.md abaixo — não alterar durante incubação)*

---

# Trading-AI (Sistema 1) — CLAUDE.md

## Status atual
**SISTEMA CONGELADO EM INCUBAÇÃO** desde 2026-05-06.
Revisão em **2026-07-06**. Até lá: zero alterações em modelo, features, target, thresholds ou filtros.

## O que o sistema faz
Bot de trading algorítmico com ML. Gera sinais exportados para MT5 via EA MQL5.

**Ativos**: EUR/USD (foco principal) + VALE3, PETR4, ITUB4, BBDC4, IBOV
**Timeframes**: 15m / 1H / 4H (4H resampled do 1H)

### Pipeline de sinal
```
OHLCV (MT5 + yfinance)
    → Indicadores (RSI, MACD, BB, ATR, ADX, Estocástico)
    → Score 0–100 + consenso multi-timeframe (2-de-3)
    → Filtro de regime de mercado (4 regimes)
    → News filter (ForexFactory ±30min)
    → RAG Copom (BCB API → DOVISH/HAWKISH → ±10pts em ativos BR)
    → LSTM + XGBoost ensemble
    → FinBERT (NLP notícias)
    → Kill switch (drawdown diário > 2×ATR(20))
    → MT5 via EA MQL5
    → GPT-4o-mini gera análise executiva (1 frase ≤200 chars, via Pythex Bridge)
    → Telegram
```

### Edge validado (holdout)
- E = +0.0003 | PF = 1.432 | Sharpe = 7.15
- Filtros ativos: ADX < 35 + atr_ratio ∈ (0.8, 1.5)

## Stack
- **Python 3.12** + venv (Windows 11 / PowerShell)
- **ML**: TensorFlow/Keras (LSTM 55.41% acc), XGBoost (55.19% acc), FinBERT (HuggingFace)
- **Dados**: MetaTrader5, yfinance
- **Infra**: Streamlit dashboard, Telegram Bot API, SQLite, Groq/GPT-4o-mini

## Estrutura de pastas
```
trading-ai/
├── core/              → indicadores, coleta, sinais, LSTM, XGBoost, MT5, telemetria
├── research/
│   └── fractal_lab/
│       ├── core/      → orchestrator, pipeline
│       ├── features/  → ATR, EMA, volatility, momentum, RSI, BB
│       ├── regimes/   → RegimeDetector (4 regimes) + HMMRegimeDetector
│       ├── validation/→ backtest candle-a-candle, walk-forward, Monte Carlo
│       ├── metrics/   → PF, WR, Expectancy, Sharpe, MaxDD, Robustness
│       ├── analytics/ → failure clusters, edge decay, regime transitions
│       ├── observer/  → Observer Agent (5 monitores, event bus, health score)
│       └── storage/   → SQLite (fractal_lab.db)
├── dashboard/         → Streamlit + Plotly
├── scripts/           → auditoria, WF, experimentos, ingestão histórica
├── mql5/              → EA MetaTrader 5
└── data/              → telemetry.json, observer_state.json, fractal_lab.db
```

## Arquivos críticos (não tocar)
- `core/regime_filter.py` — substituição pelo HMM só pós-06/07
- `core/kill_switch.py` — proteção de capital, não alterar
- `core/signals.py` — thresholds congelados até revisão
- `data/telemetry.json` — dados ao vivo em acúmulo, não corromper
- `research/fractal_lab/observer/` — Observer Agent monitorando ao vivo

## Onde roda
- **Local**: Windows 11 via `.\venv\Scripts\python` (execução principal)
- **Oracle VM** (`137.131.228.166`): reservada para Dashboard Mobile (FastAPI porta 8000)
  - Porta 8000 ainda **não aberta** no firewall OCI — roadmap pós-06/07
  - Endpoints planejados: `/candles/recent`, `/features/recent`, `/events/recent`
- **MT5**: Clear corretora conectada

## Roadmap pós-06/07
1. **Validar incubação** — checar critérios do `PROGRESS.md` com 30 dias de dados ao vivo
2. **Integrar HMM** — substituir `regime_filter.py` rule-based pelo `HMMRegimeDetector` (holdout conf=95.8%)
3. **Adaptive Sizing** — B2=1.0x, B3=0.7x, B4=0.3x, B5=0.1x (só ativar se PF rolling > 1.0)
4. **Dashboard Mobile** — abrir firewall OCI → FastAPI → PWA com Lightweight Charts + regimes coloridos ao vivo

## Regras absolutas até 06/07
- ❌ Não alterar modelo (LSTM, XGBoost, FinBERT)
- ❌ Não alterar features ou target
- ❌ Não alterar thresholds de sinal
- ❌ Não alterar filtros (ADX, atr_ratio, news_filter, kill_switch)
- ✅ Pode: ler logs, consultar telemetria, rodar scripts de análise read-only
- ✅ Pode: adicionar instrumentação (logging, healthcheck, outcome filler) — sem tocar lógica de sinal

## Instrumentação de incubação (adicionada 2026-05-21)
Módulos de observação — **não alteram lógica de sinal**:
- `core/signal_logger.py` — appender JSONL thread-safe: 1 linha por sinal em `logs/signals.jsonl`
- `core/outcome_filler.py` — preenche retorno % por janela (5min/30min/1h/4h/1d) de forma idempotente
- `core/healthcheck.py` — thread daemon: alerta Telegram se sem heartbeat >10min (cooldown 30min)
- `scripts/fill_outcomes_job.py` — cron a cada 5min para preencher outcomes via yfinance
- `core/scheduler.py` — integrado: `start_healthcheck()` no startup, `heartbeat()` a cada ciclo, `log_signal()` em cada novo sinal não-HOLD (antes do notifier, wrappado em try/except)
- `.gitignore` — criado; exclui `logs/`, `*.jsonl`, WAL SQLite, modelos ML, `.ex5`

## Comandos úteis
```powershell
# Ativar venv (sempre primeiro)
.\venv\Scripts\Activate.ps1

# Rodar sistema principal
python main.py

# Dashboard Streamlit
streamlit run dashboard/app.py

# Checar telemetria
python scripts/check_telemetry.py

# Rodar Observer Agent
python -m research.fractal_lab.observer.main

# Preencher outcomes de sinais (rodar a cada 5min via Task Scheduler)
python scripts/fill_outcomes_job.py
```

## Divisão Claude.ai vs Claude Code
- **Claude.ai**: análise de telemetria 30 dias, decisões de arquitetura pós-06/07, design do HMM integration, estratégia de Dashboard Mobile
- **Claude Code**: implementação pós-aprovação, scripts de análise read-only, ajustes de infra
