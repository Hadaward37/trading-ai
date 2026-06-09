# Plano Pós-Incubação — Sistema 1 v1.0

**Data de criação:** 22/05/2026  
**Válido para revisão de:** 06/07/2026  
**Critérios completos em:** `knowledge-base/edge_thesis.md`

Este documento é o mapa de decisão. Após rodar `notebooks/analise_06_06.ipynb`,
identificar o cenário abaixo e seguir o protocolo correspondente.

---

## Como usar este documento

1. Rodar notebook `analise_06_06.ipynb` completamente
2. Anotar: PF líquido, win rate, n de trades, drawdown máximo
3. Localizar o cenário correto abaixo
4. Seguir os passos na ordem — sem pular, sem improvisar

---

## CENÁRIO A — Edge validado forte (PF >= 1.5, n >= 30)

**Decisão:** Sistema 1 v1.0 → live trading.

1. Migrar VM para A1.Flex (ver `ops_runbook.md`)
2. Reativar observator como systemd unit com MemoryLimit=200M
3. Ajustar threshold heartbeat (10min → 18min se ciclo do scheduler for 15min)
4. Live na MT5 (Clear) com size MÍNIMO: R$ 100/trade por 30 dias
5. Após 30 dias estáveis → escalonar via Adaptive Sizing:
   - B2: 1.0x
   - B3: 0.7x
   - B4: 0.3x
   - B5: 0.1x
   (apenas se PF rolling 30d > 1.0)
6. Iniciar Sistema 2 (Historical Similarity Engine com Pythex)
7. **Após 90 dias de PF > 1.3 estável em produção:** avaliar Fase 2
   (research engine com LLM local). Ver `knowledge-base/phase_2_proposal.md`.
   NÃO iniciar antes do gate de 90 dias.

---

## CENÁRIO B — Validado conservador (1.2 <= PF < 1.5, n >= 30)

**Decisão:** Mais paper trading antes do live.

1. Estender paper trading por +30 dias (até 06/07/2026)
2. NÃO ir live ainda
3. Migrar VM A1.Flex (infra é seguro fazer agora)
4. Reativar observator filtrado (monitoramento, sem trade)
5. Em 06/07: reavaliar com 60+ trades acumulados
6. Se mantiver PF >= 1.2 → seguir CENÁRIO A
7. Se cair < 1.2 → tratar como zona cinza (CENÁRIO D)

---

## CENÁRIO C — Promissor, amostra pequena (PF >= 1.2, n < 30 trades)

**Decisão:** Estender incubação por mais 16 dias.

1. NÃO mexer em código (mesma config v1.0)
2. NÃO reativar serviços auxiliares
3. Congelamento estende até **22/06/2026**
4. Criar nova tag: `git tag -a "incubation-v1.0-extended" -m "extended to 22/06"`
5. Investigação auxiliar (read-only, sem alterar nada):
   - Distribuição de `final_score` nos sinais gerados (incluindo rejeitados)
   - Se > 80% dos scores caem entre 40-55: threshold PODE estar restringindo demais
   - Registrar a hipótese — NÃO testar ainda
6. Em 22/06: reavaliar com amostra maior

---

## CENÁRIO D — Zona cinza (0.9 <= PF < 1.2, qualquer n)

**Decisão:** Investigar antes de qualquer ação.

**Análise obrigatória (rodar no notebook antes de decidir):**

1. PF por bucket de score:
   - Bucket 55-60: PF = ___
   - Bucket 60-70: PF = ___
   - Bucket 70+:   PF = ___

2. Interpretação:
   - Se score 70+ tem PF > 1.5 e 55-60 tem PF < 1.0 → threshold deveria ser 65 → nova incubação v1.1
   - Se PF flat entre buckets → edge não vem do threshold, problema mais profundo
   - Se UM ativo carrega a média positiva → edge é concentrado, investigar isoladamente

3. PF por ativo:
   - PETR4: ___
   - VALE3: ___
   - ITUB4: ___
   - BBDC4: ___

**Possíveis ações (escolher UMA baseada na análise):**

- **Subir threshold para 65** → config mínima, nova incubação v1.1 por 16 dias
- **Cortar ativo(s) perdedor(es)** → reduzir universo, nova incubação v1.1
- **Aceitar edge fraco e pular para Sistema 2** → arquivar v1.0 sem postmortem extenso

**Proibido na zona cinza:** adicionar features, mudar timeframe, mudar RSI/MACD/BB.

---

## CENÁRIO E — Invalidado (PF < 0.9, qualquer n)

**Decisão:** Sistema 1 v1.0 morre. Aprender.

1. Escrever `knowledge-base/postmortem_v1.md`:
   - Hipótese original (copiar de `edge_thesis.md`)
   - O que de fato aconteceu (números reais do notebook)
   - 3 lições aprendidas
   - 3 coisas que faria diferente
   - Análise de regime: o período 22/05-06/07 foi representativo?

2. Arquivar código:
   ```bash
   git tag -a "archived-v1.0" -m "Sistema 1 v1.0 encerrado — PF < 0.9"
   git push origin archived-v1.0
   ```

3. NÃO tentar consertar v1.0 com patches
4. Analisar se o problema foi de edge (hipótese errada) ou de execução (infra, threshold, custo)
5. Pular para Sistema 2 (Historical Similarity Engine com Pythex)
6. Sistema 2 começa do ZERO com nova edge thesis pré-registrada antes de qualquer código

---

## CENÁRIO F — Inviável operacionalmente

**Sinais de invalidação operacional (QUALQUER UM):**
- VM travou 3+ vezes durante a incubação
- Mais de 1h/semana de manutenção ativa necessária
- Mais de 5 gaps de > 30min no heartbeat
- yfinance falhou consistentemente (> 20% dos ciclos com erro)
- `signals.jsonl` ficou dias sem crescer por causa de erros de infra

**Decisão:**

1. Migrar VM para A1.Flex (resolve memory pressure — este passo é seguro)
2. Avaliar substituição do yfinance:
   - Alpha Vantage (free tier: 5 req/min, 500/dia)
   - IEX Cloud (pago, mais estável para B3)
   - API direta do broker (Clear/XP — verificar disponibilidade)
3. Repetir incubação na nova infra com a MESMA config v1.0 (não mudar edge)
4. Se persistir inviável após troca de infra: arquivar v1.0, documentar razão, ir para Sistema 2

---

## Checklist independente do cenário

Antes de qualquer ação pós-incubação, confirmar:

- [ ] Backup `signals_incubation_final_YYYYMMDD.jsonl` salvo em `~/backups/` na Oracle
- [ ] Backup `incubation-v1.0-YYYYMMDD.tar.gz` (logs + db + scheduler.log)
- [ ] Notebook `analise_06_06.ipynb` rodado completamente, outputs salvos
- [ ] Decisão registrada em commit message explícito
- [ ] `edge_thesis.md` atualizado com seção "Resultado da incubação" (PF, WR, n, cenário)
- [ ] Próximo milestone definido no `CLAUDE.md`
- [ ] Se mudou parâmetro qualquer: nova tag `pre-incubation-vN.N` antes de retomar
- [ ] Se for live: MT5 configurado com size mínimo (R$ 100/trade), stop diário ativo

---

## Template: seção "Resultado da incubação" para edge_thesis.md

Copiar e preencher após análise:

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

---

## Fase 2 — Integração Pythex × Trading-AI (Gate: 06/07/2026)

### Pré-requisito
Sistema 1 validado: PF >= 1.2 líquido estável nos 45 dias de incubação.

### O que implementar (app/pythex_client.py)
Criar cliente HTTP para o Pythex Trading Intelligence Layer:
- URL: https://api.pythex.com.br
- Credenciais: PYTHEX_CLIENT_ID e PYTHEX_CLIENT_SECRET no .env

**PythexClient class:**
- Token JWT lazy com renovação automática (expira em 30 dias)
- ingest_pattern(ativo, timeframe, embedding, metadata) → id
- search_similar(ativo, embedding, n_results=10, filters=None) → casos + stats
- get_stats(ativo) → dict

**get_similarity_features(ativo, embedding) → dict**
Retorna 5 features para o XGBoost:
- hist_win_rate
- hist_avg_ret_1h
- hist_avg_ret_1d
- hist_n_casos
- hist_similarity_max

Se Pythex estiver fora do ar: retorna zeros silenciosamente (não quebra o pipeline).

**pattern_to_embedding(candles_df) → list[float]**
Converte janela de 20 candles em vetor normalizado L2:
- retornos pct_change últimos 20 candles
- volume normalizado pelo mean
- atr_ratio últimos 5 candles

> **IMPORTANTE:** O ChromaDB do Pythex em produção usa embeddings de 64 dimensões. O vetor retornado por pattern_to_embedding() deve ter exatamente 64 dimensões.

### Endpoints disponíveis no Pythex
- POST /auth/token
- POST /trading/ingest
- POST /trading/search
- GET  /trading/stats/{ativo}

### Ordem de execução quando o gate abrir
1. Adicionar PYTHEX_CLIENT_ID e PYTHEX_CLIENT_SECRET no .env do servidor
2. Implementar app/pythex_client.py
3. Testar ingest + search com dados reais do histórico
4. Adicionar get_similarity_features() no pipeline de features do XGBoost
5. Rodar backtest comparativo: XGBoost sem vs com features de similaridade
6. Só ativar em produção se Sharpe melhorar
