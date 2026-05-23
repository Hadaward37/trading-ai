# Fase 2 — Research Engine com LLM Local (Proposta Condicional)

> **Status:** proposta arquivada para avaliação futura
> **Data da proposta:** 22/05/2026
> **Avaliação:** pós-validação completa do Sistema 1
> **Gate de entrada:** PF > 1.3 estável por 90 dias em produção real

---

## Quando ler este documento

Ler **apenas** se TODAS as condições abaixo forem verdadeiras:

1. ✅ Milestone 06/06/2026 concluído
2. ✅ Telemetria de 30 dias da incubação analisada
3. ✅ Adaptive Sizing implementado e validado em paper trading
4. ✅ Historical Similarity Engine (Sistema 2) funcionando com Pythex como vector base
5. ✅ Sistema 1 em produção real com PF rolling > 1.3 estável por 90 dias consecutivos

**Se qualquer condição falhar: NÃO iniciar Fase 2. Voltar ao
`post_incubation_plan.md` e seguir o cenário aplicável.**

---

## Hipótese central

LLM local pode atuar como **pesquisador de hipóteses** sobre telemetria
do Sistema 1 — interpretando padrões em backtests e sugerindo filtros
de execução — sem nunca ser usado como preditor de preço.

A hipótese a ser validada é: "uma camada de research automatizada com
LLM acelera descoberta de filtros úteis vs análise manual humana, sem
introduzir overfitting estrutural."

## Princípios não-negociáveis

1. **LLM nunca prevê preço, candle ou direção de mercado.** Apenas
   interpreta resultados de backtests e sugere hipóteses para teste.
2. **Toda hipótese gerada passa por validação estatística** antes de
   virar regra: walk-forward + Out-of-Sample + Monte Carlo.
3. **Memória de pesquisa evolutiva.** Cada hipótese testada é salva com
   contexto, regime de mercado, resultado e condição de falha. Vira
   base de aprendizado.
4. **Foco em filtragem e contexto**, nunca em "indicador mágico". Edge
   real vem de: regime detection, execution control, risk allocation,
   evitar operações ruins.
5. **Paper trading obrigatório de 30 dias** antes de qualquer produção.
6. **Produção gradual** com sizing reduzido nos primeiros 30 dias live.

## Arquitetura proposta

```
[Telemetria real do Sistema 1]
↓
[DuckDB + Parquet + Polars]
Armazenamento eficiente, queries vetorizadas
↓
[vectorbt]
Backtests de variações de estratégia em paralelo
↓
[Relatório estruturado em texto]
Métricas, drawdowns, regimes detectados, falhas observadas
↓
[LLM local: qwen2.5-coder:3b via Ollama]
Interpreta relatório e gera hipóteses sobre o que filtrar
↓
[Validação estatística obrigatória]
Walk-forward + OOS + Monte Carlo
↓
[Paper trading 30 dias]
Shadow mode, sem capital real
↓
[Produção gradual]
Sizing reduzido, escalonamento condicional
```

## Stack técnica

| Componente | Tecnologia | Justificativa |
|---|---|---|
| Storage | DuckDB + Parquet | Queries OLAP rápidas em laptop, sem servidor |
| Processing | Polars | Memory-efficient vs pandas em dataset grande |
| Backtesting | vectorbt | Paralelização nativa, integra com numpy |
| LLM local | qwen2.5-coder:3b via Ollama | ~2GB RAM, CPU-only, suficiente pra raciocínio sobre métricas |
| Vector base | Pythex (já existente como Sistema 2) | Reaproveita Historical Similarity Engine |
| Hardware | Samsung i5 11ª, 8GB RAM | Roda localmente, sem cloud cost |

**Nota:** Ollama foi removido do setup local pós-Fase 1. Será
reinstalado apenas quando Fase 2 for iniciada — não antes.

## Roadmap de execução (4 semanas)

### Semana 1 — Pipeline de telemetria
- Export do `paper_trading.db` e `signals.jsonl` para Parquet
- Schema DuckDB com tabelas: signals, trades, regimes, heartbeats
- Queries de baseline (PF por bucket, distribuição de scores, gaps)
- Notebook: `analysis/phase_2_baseline.ipynb`

### Semana 2 — Motor de backtests com vectorbt
- Importar dataset histórico para vectorbt
- Implementar variações paramétricas do v1.0 (grid de thresholds, ativos, regimes)
- Output: relatório estruturado em markdown com métricas comparativas

### Semana 3 — Integração Pythex + LLM local
- Reinstalar Ollama, baixar qwen2.5-coder:3b
- Pipeline: relatório markdown → LLM → hipóteses estruturadas em JSON
- Conectar Pythex como vector base para "trades históricos similares"
- Loop: LLM lê relatório + contexto Pythex → propõe filtro testável

### Semana 4 — Validação estatística + paper trading
- Implementar walk-forward com janela móvel
- Out-of-sample em últimos 20% dos dados
- Monte Carlo bootstrap (1000+ runs) por hipótese aprovada
- Hipóteses sobreviventes vão pra paper trading 30 dias
- Após paper: shadow mode em produção com sizing 10% antes de aumentar

## Riscos mapeados (justificativa do gate de 90 dias)

| Risco | Mitigação |
|---|---|
| **Overfitting automatizado em massa** (LLM gera 1000 hipóteses, encontra ruído como sinal) | Validação estatística obrigatória por hipótese + Bonferroni correction em múltiplos testes |
| **Leakage temporal** em features (info futura entra no treino) | Walk-forward com gap explícito + checklist anti-leakage em cada feature |
| **Regime dependency** (estratégia funciona só em mercado lateral, ex.) | Backtest segmentado por regime + filtro de regime obrigatório nas hipóteses |
| **Survivorship bias** (testar só em ativos que sobreviveram) | Dataset histórico inclui ativos delistados quando possível |
| **Complexidade mascarando ausência de edge** | Princípio: se Fase 2 não melhora PF em pelo menos 0.2 vs Sistema 1 puro, descartar |

## Critério de falsificação da Fase 2

Após 30 dias de paper trading da Fase 2:

- **Validada:** PF de hipóteses Fase 2 > PF Sistema 1 puro + 0.2, com Sharpe > 1.0 e drawdown <= 8%
- **Inválida:** PF não bate Sistema 1 puro ou drawdown > 10%
- **Zona cinza:** melhora marginal (< 0.2 de PF) → estender 30 dias

Se inválida: arquivar todo o pipeline Fase 2, manter Sistema 1 puro
rodando, postmortem em `knowledge-base/postmortem_phase_2.md`.

## Por que NÃO construir agora

1. **Sistema 1 ainda não provou edge.** Construir camada de research
   sobre sistema não-validado é otimização prematura.
2. **Distração de DuduStudio + outras prioridades.** Solo founder não
   tem horas pra duas frentes técnicas profundas simultâneas.
3. **Risco de overfitting do próprio pipeline.** Quanto mais cedo a
   complexidade entra, mais difícil isolar onde o edge real vem.
4. **Gate de 90 dias é proteção contra impulsividade.** Se em 90 dias
   pós-validação ainda fizer sentido, faz. Se não, foi proteção valiosa.

## Referência cruzada

- Trigger para reler este documento: `post_incubation_plan.md` Cenário A
  + 90 dias de produção estável
- Não substitui: `edge_thesis.md` (que é a tese do Sistema 1 v1.0)
- Complementa: Sistema 2 (Historical Similarity Engine com Pythex)
  como pré-requisito #3
