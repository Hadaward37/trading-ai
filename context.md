# context.md — Sistema 1 (trading-ai) — Documento de Referência Completo

> **Gerado em:** 2026-06-04  
> **Objetivo:** este arquivo é autocontido. Qualquer pessoa ou IA lendo apenas este documento deve entender o sistema completo, o que está rodando, o que não pode ser tocado e o que fazer em cada cenário pós-incubação.

---

## 1. O que é o sistema

**Trading-AI Sistema 1** é um bot de trading algorítmico com ML para large caps brasileiras na B3. Gera sinais de BUY/SELL/HOLD que são executados em paper trading e futuramente em capital real via MetaTrader 5.

### Hipótese central

> "RSI(14) combinado com XGBoost em timeframe 15min, aplicado a large caps brasileiras (PETR4, VALE3, ITUB4, BBDC4), captura reversões e tendências de curto prazo que geram Profit Factor > 1.0 líquido de custos em janelas de 5min a 4h de holding period."

Hipótese auxiliar: o edge, se existir, **não é uniforme** entre os 4 ativos — a análise deve segmentar por ativo.

### Pipeline de sinal (congelado em v1.0)

```
OHLCV (yfinance)
    → Indicadores (RSI, MACD, BB, ATR, ADX, Estocástico)
    → Score 0–100 + consenso multi-timeframe (2-de-3: 15m/1H/4H)
    → Filtro de regime de mercado (4 regimes: TRENDING / MEAN_REVERTING / HIGH_VOL / LOW_VOL)
    → News filter (ForexFactory ±30min — bloqueia FOMC/NFP/CPI/Copom)
    → RAG Copom (BCB API → DOVISH/HAWKISH → ±10pts em ativos BR)
    → LSTM + XGBoost ensemble
    → FinBERT (NLP notícias — DESATIVADO na incubação)
    → Kill switch (drawdown diário > 2×ATR(20))
    → Paper Trading (R$ 10.000 capital virtual)
    → GPT-4o-mini análise executiva (via Pythex Bridge)
    → Telegram
```

### Edge validado em holdout (pré-incubação)

| Métrica | Valor |
|---|---|
| Expectancy | +0.0003 |
| Profit Factor | 1.432 |
| Sharpe | 7.15 |
| Filtros ativos | ADX < 35 + atr_ratio ∈ (0.8, 1.5) |
| Win Rate (optimizer v2) | 47.2% |
| Trades (optimizer v2) | 212 |

### Stack técnica

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.12 + venv |
| ML | TensorFlow/Keras (LSTM 55.41% acc), XGBoost (55.19% acc), FinBERT |
| Dados | yfinance (B3), MetaTrader5 (EUR/USD — removido da incubação) |
| Infra | Systemd (Oracle VM), Streamlit dashboard, Telegram Bot API |
| Storage | SQLite (`data/trading.db`, `data/fractal_lab.db`) |
| NLP | Groq API (Llama 3.3 70B), GPT-4o-mini, FinBERT |
| Análise | Fractal Lab (walk-forward, Monte Carlo, Observer Agent) |

---

## 2. Estado atual — Incubação ativa

### Tabela de estado

| Item | Valor |
|---|---|
| **Status** | **INCUBAÇÃO v1.0 — SISTEMA CONGELADO** |
| Início | 2026-05-22 |
| **Revisão obrigatória** | **2026-07-06** |
| Commit de congelamento | `daaa5ca` |
| Strategy version | `v1.0` |
| Ativos ativos | VALE3, PETR4, ITUB4, BBDC4 |
| Benchmark (sem trades) | IBOV |
| EUR/USD | REMOVIDO do universo durante incubação |
| Sentimento LLM | DESATIVADO (peso = 0) |
| Threshold de confiança | 55 (sobre `final_score`) |
| Comissão paper | 0.05%/lado = 0.10% round-trip |
| Capital paper | R$ 10.000 |

### Por que 45 dias (e não 16)

Janela original era 16 dias (até 06/06/2026). Estendida para 45 dias por dois motivos:
1. Usuário pausou acesso ao Claude por questão de assinatura até 06/07/2026
2. Janela maior gera amostra estatística mais robusta (estimativa: 30+ trades vs 15 mínimos) — elimina risco do Cenário C (promissor mas poucos trades)

### Regras absolutas até 06/07/2026

| Proibido | Permitido |
|---|---|
| ❌ Alterar modelo (LSTM, XGBoost, FinBERT) | ✅ Ler logs e telemetria |
| ❌ Alterar features ou target | ✅ Consultar `scripts/check_telemetry.py` |
| ❌ Alterar thresholds de sinal | ✅ Adicionar instrumentação (sem tocar lógica de sinal) |
| ❌ Alterar filtros (ADX, atr_ratio, news_filter, kill_switch) | ✅ Rodar scripts de análise read-only |
| ❌ Mexer em systemd, crontab ou config da VM | ✅ Healthcheck SSH read-only (comando exato abaixo) |

---

## 3. Configuração congelada v1.0

Valores em `config.py` — **não alterar** até revisão de 06/07:

```python
# Ativos
ASSETS = {"VALE3": "VALE3.SA", "PETR4": "PETR4.SA", "ITUB4": "ITUB4.SA", "BBDC4": "BBDC4.SA", "IBOV": "^BVSP"}
BENCHMARK_ASSETS = {"IBOV"}  # logado, sem trades
USE_MT5 = False              # EUR/USD fora do universo

# Indicadores
RSI_PERIOD = 14 | RSI_BUY = 35 | RSI_SELL = 75
MACD_FAST = 12 | MACD_SLOW = 26 | MACD_SIGN = 9
BB_WINDOW = 20 | BB_DEV = 2.0
ATR_WINDOW = 14 | ADX_WINDOW = 14
STOCH_K_WINDOW = 14 | STOCH_OVERSOLD = 25 | STOCH_OVERBOUGHT = 75

# Regime
REGIME_ADX_TREND = 25 | REGIME_ADX_RANGE = 20
REGIME_VOLATILITY_THRESHOLD = 1.5
REGIME_BLOCK_RANGE_SIGNALS = True

# Score (pesos somam 100)
RSI=14 | MACD=14 | BB=11 | ADX=10 | STOCH=10 | LSTM=20 | XGBOOST=21

# Sinal
SIGNAL_CONFIDENCE_THRESHOLD = 55  # final_score >= 55 → trade
SCHEDULER_INTERVAL_MIN = 15
PAPER_TRADING_COMMISSION_PCT = 0.0005  # 0.05% por lado
PAPER_TRADING_CAPITAL = 10_000.0
STRATEGY_VERSION = "v1.0"

# Desativados na incubação
NEWS_INTELLIGENCE_ENABLED = False  # Groq/GPT sentimento desligado
COPOM_ENABLED = True               # Copom macro ATIVO (não LLM de sentimento)
LSTM_ENABLED = True | XGBOOST_ENABLED = True | FINBERT_ENABLED = True
```

---

## 4. Critérios de validação pós-incubação

Todos os critérios devem ser atendidos para declarar hipótese **VÁLIDA**:

| Critério | Meta | Notas |
|---|---|---|
| Profit Factor líquido | PF >= 1.2 | Após 0.10% round-trip de custo |
| PF sem melhor trade | PF > 1.0 | Proteção anti-outlier |
| Win Rate | >= 45% | Com avg_win > avg_loss |
| Edge distribuído | >= 2 dos 4 ativos | Não pode ser concentrado em 1 |
| Drawdown máximo | <= 5% | = R$ 500 sobre R$ 10.000 |
| Mínimo operacional | >= 15 trades | Abaixo: amostra inutilizável |
| Mínimo estatístico | >= 30 trades | Abaixo: "promissor", não "edge robusto" |

**Critérios de INVALIDAÇÃO** (qualquer um dispara pivot):
- PF < 0.9 líquido
- Win Rate < 35% com PF < 1.0
- Edge concentrado em 1 ativo apenas
- Drawdown > 8% (= R$ 800)
- Menos de 8 trades
- Healthcheck disparou > 3 vezes (instabilidade operacional)

**Critério de sobrevivência operacional** (independente dos números):
- Exige intervenção manual > 1x/semana → arquivo
- Instabilidade > 1 disparo healthcheck/semana → arquivo
- Consome > tempo do que o edge justifica para um solo founder → arquivo

---

## 5. Cenários de decisão (A-F) — pré-registrados, vinculantes

Após rodar `notebooks/analise_06_06.ipynb`, identificar o cenário e seguir o protocolo:

### CENÁRIO A — Edge forte (PF >= 1.5, n >= 30)
**Decisão:** live trading.
1. Migrar VM para A1.Flex → reativar observator (systemd + MemoryLimit=200M) → reativar polymarket
2. Live na MT5 (Clear) com size MÍNIMO R$ 100/trade por 30 dias
3. Após 30 dias estáveis → Adaptive Sizing: B2=1.0x, B3=0.7x, B4=0.3x, B5=0.1x (só se PF rolling 30d > 1.0)
4. Iniciar Sistema 2 (Historical Similarity Engine com Pythex)
5. **Após 90 dias de PF > 1.3 estável em produção** → avaliar Fase 2 (LLM local)

### CENÁRIO B — Validado conservador (1.2 <= PF < 1.5, n >= 30)
**Decisão:** mais paper trading. Estender +30 dias. Não ir live ainda.
- Migrar VM A1.Flex (infra OK)
- Em 06/08: reavaliar. Se mantiver PF >= 1.2 → Cenário A. Se cair → Cenário D.

### CENÁRIO C — Promissor, amostra pequena (PF >= 1.2, n < 30)
**Decisão:** estender incubação por +16 dias (até 22/06/2026). MESMA config v1.0.
- Criar tag: `git tag -a "incubation-v1.0-extended" -m "extended to 22/06"`
- Investigação read-only: distribuição de `final_score` — se >80% entre 40-55, threshold pode estar restringindo

### CENÁRIO D — Zona cinza (0.9 <= PF < 1.2, qualquer n)
**Decisão:** investigar antes de qualquer ação.
- Calcular PF por bucket de score (55-60, 60-70, 70+) e por ativo (PETR4, VALE3, ITUB4, BBDC4)
- Opções (escolher UMA): subir threshold para 65 / cortar ativo(s) perdedor(es) / pular para Sistema 2
- **Proibido:** adicionar features, mudar timeframe, mudar RSI/MACD/BB

### CENÁRIO E — Invalidado (PF < 0.9, qualquer n)
**Decisão:** Sistema 1 v1.0 morre. Postmortem obrigatório.
1. Escrever `knowledge-base/postmortem_v1.md` (hipótese original, o que aconteceu, 3 lições, 3 coisas diferentes)
2. `git tag -a "archived-v1.0" -m "Sistema 1 v1.0 encerrado — PF < 0.9"`
3. Não consertar v1.0 com patches — ir direto para Sistema 2

### CENÁRIO F — Inviável operacionalmente
**Sinais:** VM travou 3+ vezes / >1h/semana manutenção / >5 gaps de >30min no heartbeat
**Decisão:** migrar VM A1.Flex, avaliar substituição do yfinance (Alpha Vantage / IEX Cloud / API broker), repetir incubação com MESMA config v1.0 na nova infra.

---

## 6. Healthcheck semanal — comando exato

```bash
# SSH read-only (único autorizado durante incubação)
ssh -i ~/.ssh/ssh-key-2026-05-05.key ubuntu@137.131.228.166 \
  'tail -3 ~/trading-ai/logs/heartbeat.txt && \
   free -h && \
   wc -l ~/trading-ai/logs/signals.jsonl 2>/dev/null || echo "0 sinais"'
```

**Diagnóstico completo (quando necessário):**

```bash
ssh oracle '
  echo "=== HEARTBEAT ===" && tail -1 ~/trading-ai/logs/heartbeat.txt
  echo "=== SINAIS ===" && wc -l ~/trading-ai/logs/signals.jsonl 2>/dev/null || echo "0"
  echo "=== RAM ===" && free -h | grep Mem
  echo "=== SWAP ===" && free -h | grep Swap
  echo "=== SERVICE ===" && systemctl is-active trading-ai.service
'
```

**O que monitorar:**
- `heartbeat.txt` → última linha deve ser recente (< 30min atrás): se > 10min, healthcheck dispara Telegram
- `signals.jsonl` → contagem deve crescer ao longo dos dias
- `free -h` → swap idealmente zero; se Used > 0, investigar pressão de memória
- `systemctl is-active trading-ai.service` → deve retornar `active`

**Se SSH falhar / VM travada:**
Usar Oracle Cloud Console: `cloud.oracle.com → Compute → Instances → trading-ai → Console Connection`

---

## 7. Mapa dos arquivos importantes

### Arquivos de decisão (leitura obrigatória na revisão de 06/07)

| Arquivo | Função |
|---|---|
| `CLAUDE.md` / `AGENTS.md` | Instruções para IAs — estado de incubação, regras |
| `knowledge-base/edge_thesis.md` | Tese + critérios validação/invalidação/zona cinza + pacto pessoal |
| `knowledge-base/post_incubation_plan.md` | Mapa de decisão cenários A-F (este documento) |
| `knowledge-base/ops_runbook.md` | Comandos SSH de reativação, troubleshooting, upgrade VM |
| `knowledge-base/phase_2_proposal.md` | Proposta arquivada: research engine LLM local (gate 90 dias) |
| `notebooks/analise_06_06.ipynb` | Notebook de análise pronto para rodar em 06/07 |
| `PROGRESS.md` | Log completo de progresso, achados quantitativos, módulos |

### Arquivos críticos de código (não tocar até 06/07)

| Arquivo | Função | Por que não tocar |
|---|---|---|
| `core/signals.py` | Geração de sinais, thresholds congelados | Alterar invalida amostra da incubação |
| `core/regime_filter.py` | Filtro de regime rule-based | Substituição pelo HMM só pós-06/07 |
| `core/kill_switch.py` | Proteção de capital (drawdown > 2×ATR(20)) | Risco operacional |
| `core/scheduler.py` | Loop 15min, integra heartbeat + signal_logger | Alterar quebra instrumentação |
| `data/telemetry.json` | Dados ao vivo em acúmulo | Corromper perde histórico |
| `config.py` | Todos os parâmetros centralizados | Thresholds congelados |

### Arquivos de instrumentação de incubação (adicionados 2026-05-21, read-only OK)

| Arquivo | Função |
|---|---|
| `core/signal_logger.py` | Appender JSONL thread-safe → `logs/signals.jsonl` |
| `core/outcome_filler.py` | Preenche retorno % por janela (5/30/60/240/1440min), idempotente |
| `core/healthcheck.py` | Thread daemon → alerta Telegram se heartbeat > 10min antigo |
| `scripts/fill_outcomes_job.py` | Cron a cada 5min: `*/5 * * * *` — preenche outcomes via yfinance |

### Dataset de incubação (na Oracle VM)

| Arquivo | Conteúdo |
|---|---|
| `logs/signals.jsonl` | 1 linha por sinal não-HOLD: symbol, timestamp, price, features, outcome_* |
| `logs/heartbeat.txt` | Timestamp UTC do último ciclo do scheduler |
| `logs/outcome_filler.log` | Log do cron de preenchimento de outcomes |
| `data/trading.db` | SQLite com candles, paper trades, histórico |

### Fractal Lab (pesquisa quantitativa — research/)

| Módulo | Função |
|---|---|
| `research/fractal_lab/core/` | Orchestrator: `run_hypothesis(df, hypothesis) → ResearchReport` |
| `research/fractal_lab/regimes/` | RegimeDetector (4 regimes) + **HMMRegimeDetector** (candidato pós-06/07) |
| `research/fractal_lab/validation/` | Backtest candle-a-candle, walk-forward, Monte Carlo |
| `research/fractal_lab/observer/` | Observer Agent: 5 monitores, event bus, health score 0-100 |
| `research/fractal_lab/analytics/` | Failure clusters, edge decay, regime transitions |

---

## 8. As duas VMs Oracle (não confundir)

### VM Atual — VM.Standard.E2.1.Micro

| Atributo | Valor |
|---|---|
| IP | `137.131.228.166` |
| Shape | VM.Standard.E2.1.Micro |
| RAM | 1 GB física + 1 GB swap (`/swapfile`) |
| CPU | 1 OCPU (x86) |
| Chave SSH | `~/.ssh/ssh-key-2026-05-05.key` (Windows: `$env:USERPROFILE\.ssh\...`) |
| OS | Ubuntu 22.04 LTS |
| Região | Brazil East (São Paulo) |

**Serviços ativos:**
- `trading-ai.service` → ATIVO (systemd, Restart=always) — o Sistema 1
- `polymarket-bot.service` → PAUSADO (disabled até 06/07)
- `polymarket-dashboard.service` → PAUSADO (disabled até 06/07)
- `trading-ai-observator` → PAUSADO (era nohup, não reinicia no boot — débito técnico)
- Cron: `*/5 * * * *` → `fill_outcomes_job.py`

**Limitação:** 1 GB RAM é apertado para 3 serviços Python. VM travou em 21/05 com memory pressure. Swap é rede de segurança, não solução de performance. Se swap Used > 0: sinal de que precisa de upgrade.

### VM Planejada — VM.Standard.A1.Flex ARM (pós-06/07)

| Atributo | Valor |
|---|---|
| Shape | VM.Standard.A1.Flex |
| RAM | 12–24 GB (grátis no Always Free tier) |
| CPU | 2–4 OCPU (ARM) |
| Custo | Zero (Oracle Always Free) |
| Status | NÃO migrar durante incubação |

**Como migrar (pós-06/07, Cenários A/B/F):**
```bash
# Oracle Cloud Console:
# Compute → Instances → Stop → Change shape → A1.Flex → 2 OCPU / 12 GB RAM → Start
# Verificar IP (pode mudar) → atualizar DNS se necessário
# SSH → free -h → confirmar ~12 GB
# sudo swapon --show → manter swap (não prejudica)
# Reativar polymarket + observator com MemoryLimit no unit file
```

**Débito técnico a resolver na migração:**
1. Criar `trading-ai-observator.service` (hoje roda como nohup solto)
2. Definir `Restart=on-failure`, `MemoryLimit=200M` no unit file
3. `systemctl enable trading-ai-observator.service`

---

## 9. Fase 2 arquivada — gates obrigatórios

A Fase 2 (Research Engine com LLM local) está arquivada até todos os gates serem vencidos. **Não iniciar antes.**

### O que é a Fase 2

Pipeline de research automatizada:
- DuckDB + Parquet + Polars (storage/queries)
- vectorbt (backtests paramétricos em paralelo)
- LLM local: `qwen2.5-coder:3b` via Ollama (~2GB RAM, CPU-only)
- Pythex como vector base para trades históricos similares
- O LLM **nunca prevê preço** — apenas interpreta relatórios de backtest e gera hipóteses estruturadas para teste

### Gates obrigatórios (TODOS devem ser verdadeiros)

1. ✅ Incubação 06/07/2026 concluída e analisada
2. ✅ Adaptive Sizing implementado e validado em paper trading
3. ✅ Historical Similarity Engine (Sistema 2) funcionando com Pythex
4. ✅ Sistema 1 em produção real com **PF rolling > 1.3 estável por 90 dias consecutivos**
5. ✅ Ollama reinstalado localmente (foi removido pós-Fase 1)

### Critério de falsificação da Fase 2

Após 30 dias de paper trading da Fase 2:
- **Válida:** PF Fase 2 > PF Sistema 1 + 0.2, Sharpe > 1.0, drawdown <= 8%
- **Inválida:** PF não bate Sistema 1 ou drawdown > 10% → arquivar todo o pipeline Fase 2
- **Zona cinza:** melhora < 0.2 de PF → estender 30 dias

**Por que não construir agora:** Sistema 1 ainda não provou edge ao vivo. Construir pesquisa sobre sistema não-validado é otimização prematura. Gate de 90 dias protege contra impulsividade.

---

## 10. Última leitura de saúde conhecida

| Item | Valor | Quando |
|---|---|---|
| Último heartbeat local | `2026-05-31T20:48:36.452698+00:00` | 31/05/2026 ~20:48 UTC |
| `signals.jsonl` local | 1 linha (cópia local — arquivo real está na Oracle VM) | — |
| `trading-ai.service` | ATIVO (confirmado na última sessão) | 2026-05-22 |
| RAM livre (pós-hardening) | 447 MB livre com trading-ai rodando | 2026-05-22 |
| Swap | 1 GB disponível, zero uso esperado em operação normal | 2026-05-22 |
| Tom Copom | DOVISH — Selic 14.40% aa, trajetória de queda | 2026-05-06 |
| Observer health score | 83/100 GOOD (EURUSD saudável, referência pré-incubação) | 2026-05-12 |

**Atenção:** o último heartbeat registrado localmente é de 31/05. Para verificar saúde atual, rodar o comando SSH de healthcheck da seção 6. O gap entre 31/05 e hoje (04/06) pode indicar:
1. Sistema rodando normalmente na Oracle, mas arquivo local não foi sincronizado (mais provável)
2. Problema de conectividade ou crash → verificar via SSH antes da revisão de 06/07

---

## Checklist de revisão — 06/07/2026

Executar na ordem:

- [ ] `ssh oracle 'wc -l ~/trading-ai/logs/signals.jsonl'` — quantos sinais?
- [ ] Backup: `cp signals.jsonl ~/backups/signals_incubation_final_$(date +%Y%m%d).jsonl`
- [ ] Backup completo: `tar -czf ~/backups/incubation-v1.0-$(date +%Y%m%d).tar.gz ~/trading-ai/logs/ ~/trading-ai/data/trading.db ~/trading-ai/scheduler.log`
- [ ] Rodar `notebooks/analise_06_06.ipynb` completamente (todas as células em ordem)
- [ ] Consultar `knowledge-base/edge_thesis.md` — aplicar critérios
- [ ] Consultar `knowledge-base/post_incubation_plan.md` — identificar cenário (A-F)
- [ ] Registrar resultado em `edge_thesis.md` (seção "Resultado da incubação")
- [ ] Consultar `knowledge-base/ops_runbook.md` — executar passos de reativação
- [ ] Atualizar `CLAUDE.md` / `AGENTS.md` com próximo milestone
- [ ] Se mudou qualquer parâmetro: criar nova tag `pre-incubation-vN.N` antes de retomar

---

## Protocolo de zona cinza — análise obrigatória

Se PF cair em 0.9-1.2 (Cenário D), calcular antes de qualquer decisão:

| Análise | Comando/Query |
|---|---|
| PF por bucket de score | bucket 55-60 / 60-70 / 70+ |
| PF por ativo | PETR4 / VALE3 / ITUB4 / BBDC4 separados |
| PF sem melhor trade | remover o trade com maior retorno positivo |
| Regime do período | VOL IBOV vs média histórica 30d/90d, ADX, slope EMA200 |
| Eventos macro | Copom, payroll EUA, balanços, eventos políticos BR no período |

Se score 70+ tem PF > 1.5 e 55-60 tem PF < 1.0 → threshold deveria ser 65 → nova incubação v1.1 (NÃO adicionar features).

---

## Template para registrar resultado em 06/07

```markdown
## Resultado da incubação — 06/07/2026

**Período:** 2026-05-22 → 2026-07-06 (45 dias)
**n de trades:** ___
**PF líquido:** ___
**PF sem melhor trade:** ___
**Win rate:** ___%
**Drawdown máximo:** ___%
**Regime do período:** [ALTA/LATERAL/BAIXA] + [VOL NORMAL/ALTA/BAIXA]
**Cenário:** [A / B / C / D / E / F]
**Decisão:** ___
**Próxima revisão:** ___
```
