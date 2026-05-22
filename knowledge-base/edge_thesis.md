# Edge Thesis — Sistema 1 (B3 + IBOV)

**Data de criação:** 21/05/2026  
**Próxima revisão:** 06/06/2026  
**Strategy version:** v1.0  
**Responsável:** Dudu (Hadaward37)  
**Validado por:** Claude (Opus 4.7) + GPT (sócio de construção)

---

## Hipótese central

"RSI(14) combinado com XGBoost em timeframe 15min, aplicado a large caps 
brasileiras (PETR4, VALE3, ITUB4, BBDC4), captura reversões e tendências 
de curto prazo que geram Profit Factor > 1.0 líquido de custos em janelas 
de 5min a 4h de holding period."

Hipótese auxiliar: "O edge, se existir, NÃO é uniforme entre os 4 ativos — 
alguns terão edge, outros não. A análise deve segmentar por ativo."

---

## Configuração congelada para incubação

| Componente | Valor |
|---|---|
| Ativos | PETR4, VALE3, ITUB4, BBDC4 |
| Benchmark | IBOV (logado mas sem trades) |
| Timeframe | 15min |
| Sentimento LLM | DESATIVADO (peso = 0) |
| Modelo ML | XGBoost (LSTM ausente, ensemble degradado — aceito) |
| Threshold confiança inicial | 0.65 |
| Threshold fallback | 0.55 (se <5 trades em 7 dias corridos) |
| Custo simulado por trade | 0.05% (ida) + 0.05% (volta) = 0.10% round-trip |
| Capital paper trading | R$ 10.000 |
| Sizing | Fixo (sem adaptive sizing nesta fase) |
| Período de incubação | 22/05/2026 → 06/06/2026 |
| Janelas de outcome | 5min, 30min, 1h, 4h, 1d |

---

## Critérios de validação (TODOS devem ser atendidos)

A hipótese é considerada **VÁLIDA** em 06/06 se:

- [ ] **Profit Factor (PF) >= 1.2 líquido** — após custos de 0.10% round-trip
- [ ] **PF permanece > 1.0 após remover o melhor trade individual** — proteção contra outlier (sugestão GPT)
- [ ] **Win rate >= 45%** combinado com avg_win > avg_loss
- [ ] **Edge presente em pelo menos 2 dos 4 ativos** (não pode ser concentrado em 1)
- [ ] **Drawdown máximo <= 5%** sobre capital R$ 10.000 (= R$ 500)
- [ ] **Mínimo operacional: 15 trades** — abaixo disso, amostra inutilizável
- [ ] **Mínimo estatístico confortável: 30 trades** — abaixo disso, edge NÃO pode ser declarado robusto, apenas "promissor para continuar pesquisa"

**Métricas auxiliares (não decisivas):**
- Sharpe ratio anualizado (distorce em amostra pequena — informativo, não vinculante)
- Sortino ratio
- Expectancy por trade
- avg_win / avg_loss ratio

---

## Critérios de invalidação (QUALQUER UM dispara pivot)

A hipótese é considerada **INVALIDADA** em 06/06 se:

- [ ] **PF < 0.9 líquido**
- [ ] **Win rate < 35%** combinado com PF < 1.0
- [ ] **Edge concentrado em 1 ativo apenas**
- [ ] **Drawdown máximo > 8%** (= R$ 800)
- [ ] **Menos de 8 trades** — sistema travado, sem o que analisar
- [ ] **Healthcheck disparou >3 vezes** — instabilidade operacional invalida o teste
- [ ] **Critério de sobrevivência operacional violado** (ver seção abaixo)

---

## Zona cinza (0.9 <= PF < 1.2)

Não é validação nem invalidação. **Não adicionar features.** Ações permitidas:

1. **Reduzir escopo:** manter apenas o(s) ativo(s) que mostraram PF > 1.0 individual
2. **Estender incubação:** +30 dias com `strategy_version = v1.1` (apenas mudança de escopo)
3. **Análise forense:** identificar SE houve regime específico (volatilidade alta/baixa, tendência/lateral) onde o sistema funcionou melhor

**Proibido na zona cinza:** adicionar features novas, mudar threshold, mudar timeframe, mudar custos, mudar capital.

---

## Risco de regime de mercado (adicionado por sugestão GPT)

A janela 22/05 → 06/06 pode representar apenas **um subconjunto** dos regimes de mercado possíveis.

**Implicações:**
- Resultados positivos NÃO implicam robustez cross-regime
- Resultados negativos também NÃO invalidam completamente a hipótese se o período tiver sido excepcionalmente anômalo (ex: choque macro, evento de cauda)

**A análise pós-06/06 deve classificar o período por:**
- Volatilidade realizada do IBOV (versus média histórica 30d/90d)
- Regime de tendência vs lateralização (ADX, slope da média móvel)
- Direção do IBOV no período (alta/baixa/lateral)
- Eventos macro relevantes (Copom, payroll EUA, balanços, eventos políticos BR)

Essa classificação será incluída no notebook `analise_06_06.ipynb` como contexto obrigatório antes de qualquer conclusão.

---

## Critério de sobrevivência operacional (adicionado por sugestão GPT)

Mesmo com edge estatístico positivo, o sistema é considerado **operacionalmente inviável** se:

- [ ] Exigir intervenção manual frequente (>1x por semana)
- [ ] Apresentar instabilidade operacional (>1 disparo de healthcheck por semana fora de janela esperada)
- [ ] Consumir mais tempo operacional do que justificável para o edge gerado

**Por quê:** existe sistema matematicamente positivo mas operacionalmente inviável. PF 1.3 que exige 10h/semana de manutenção não escala para um solo founder com outros SaaS em paralelo.

Esse critério é binário: se falhar, sistema é arquivado **mesmo que** os critérios estatísticos sejam atendidos. Pode ser reescrito do zero, mas não pode rodar nessa forma.

---

## Pacto pessoal

> Eu, Dudu, me comprometo formalmente a:
> 
> 1. **Não mexer no código entre 22/05/2026 e 06/06/2026**, exceto se Claude/GPT solicitarem para diagnóstico
> 2. **Respeitar os critérios acima sem racionalizar** os resultados em 06/06
> 3. Se hipótese for **invalidada**, eu vou **pivotar a tese central** — não vou adicionar features para forçar números bons
> 4. Se hipótese cair em **zona cinza**, eu vou seguir o protocolo restrito acima — nada além
> 5. Vou usar os 16 dias para **acumular dataset analisável**, não para "otimizar enquanto roda"
> 6. **Não vou consultar resultados parciais antes de 06/06** com intuito de decidir nada — consulta diária é apenas para verificar saúde operacional (heartbeat, contagem de sinais)
> 7. Se **15 <= trades < 30**, eu NÃO vou declarar "edge encontrado" mesmo com PF bom — vou classificar como "promissor para nova rodada de incubação"
> 
> Assinatura: Dudu  
> Data: 21/05/2026

---

## Premissas registradas (para análise futura)

Coisas que NÃO sei se afetam o resultado, mas estou aceitando como tradeoff:

1. **yfinance pode ter delay/gaps** em comparação a feed institucional — aceito porque é o que tenho
2. **Custo de 0.05% é estimativa conservadora** — corretagem Clear é zero para day trade em ações, mas há emolumentos B3 (~0.0325%) + ISS + slippage estimado
3. **Paper trading não simula impacto de mercado** — irrelevante em PETR4/VALE3/ITUB4/BBDC4 com 100 ações
4. **XGBoost solo (sem LSTM)** — ensemble degradado, aceito porque trocar componente durante incubação invalidaria amostra
5. **Threshold 0.65 pode ser apertado demais** — fallback para 0.55 mitiga, mas se não disparar, amostra pode ficar pequena

---

## Pós 06/06 — Decisão por cenário (pré-registrada)

| Cenário | PF líquido | Trades | Ação |
|---|---|---|---|
| Validado forte | PF >= 1.5 | >= 30 | Adaptive sizing + retreinar HMM + considerar capital real micro (R$ 500-1k) |
| Validado base | 1.2 <= PF < 1.5 | >= 30 | Estender incubação +30d, sem mudanças, validar consistência |
| Promissor (sub-amostra) | PF >= 1.2 | 15 <= n < 30 | **Não declarar edge.** Nova rodada de 30 dias com mesma config |
| Zona cinza | 0.9 <= PF < 1.2 | qualquer | Reduzir escopo aos ativos que funcionaram + análise regime |
| Invalidado | PF < 0.9 | qualquer | **PIVOT.** Pausar Sistema 1. Analisar dataset. Considerar nova hipótese |
| Operacionalmente inviável | qualquer | qualquer | Arquivar sistema atual, considerar reescrita do zero |

Esta tabela é vinculante. Foi feita ANTES dos dados existirem.

---

## Métricas adicionais a calcular no notebook analise_06_06.ipynb

Para análise pós-06/06, o notebook deve gerar:

1. PF líquido geral + PF por ativo
2. PF após remover melhor trade (validação anti-outlier)
3. Win rate geral + por ativo
4. Avg win, avg loss, expectancy por trade
5. Drawdown máximo + duração
6. Equity curve
7. Sharpe e Sortino (auxiliares)
8. Distribuição de trades por janela de outcome (5min, 30min, 1h, 4h, 1d)
9. Classificação de regime do período (vol IBOV, tendência, eventos macro)
10. Distribuição temporal: hora do dia, dia da semana, primeira/segunda metade do mês
11. Healthcheck logs: quantos disparos, quando, por quê

---

## Nota operacional — 22/05/2026

**Decisão de infraestrutura durante incubação:**

A VM Oracle (VM.Standard.E2.1.Micro, 1 OCPU / 1 GB RAM) demonstrou memory 
pressure com 3 serviços Python concorrentes (trading-ai, trading-ai-observator, 
polymarket-bot), causando travamento do sshd em 21/05/2026.

**Ação tomada:**
- `polymarket-bot.service` → STOP + DISABLE
- `polymarket-dashboard.service` → STOP + DISABLE
- Apenas `trading-ai.service` permanece ativo durante a incubação

**Justificativa:**
Garantir recursos exclusivos ao Sistema 1 elimina a variável confusora 
"memory pressure → execution lag → signal loss" da análise pós-06/06.
A incubação mede edge do Sistema 1 — não capacidade da VM de rodar 3 bots.

**Reativação programada:**
Polymarket e dashboard serão reativados após 06/06/2026, junto com 
avaliação de upgrade da VM (candidato: A1.Flex ARM, 24 GB grátis).

**Risco residual aceito:**
Mesmo com apenas 1 serviço, a VM ainda tem só 1 GB de RAM sem swap. 
Se houver travamento durante a incubação, o pacto é: registrar o incidente, 
NÃO mexer no código, fazer reboot via console, e contabilizar o gap no 
notebook de análise 06/06.

---

## Nota operacional — 22/05/2026 (parte 2)

**Hardening de infra durante incubação:**

Ativado 1 GB de swap na VM Oracle para mitigar risco de OOM/travamento 
durante a incubação. Comando: `fallocate /swapfile + mkswap + swapon`, 
persistido em `/etc/fstab`.

**Justificativa:**
Mesmo com observator e polymarket pausados, a VM continua tendo apenas 
1 GB de RAM física. Swap não é solução de performance (é lento), mas é 
rede de segurança contra travamento do sshd em caso de pico de memória 
(GC do Python, payload grande do yfinance, etc).

**Estado pós-swap:**
- RAM física: 957 MB (447 MB livre com trading-ai rodando)
- Swap: 1 GB (zero uso esperado em operação normal)
- Se swap começar a ser usado: sinal de que precisa upgrade para A1.Flex

---

## Débito técnico identificado — 22/05/2026

**Observator NÃO é systemd unit:**

Durante o recovery de 22/05/2026 descobriu-se que `trading-ai-observator` 
estava rodando como processo `nohup` solto, não como systemd service. 
Por isso não reiniciou automaticamente após o reboot da VM (comportamento 
desejado durante a incubação, mas frágil em produção normal).

**Ação programada para pós-06/06/2026 (NÃO FAZER ANTES):**

1. Criar `trading-ai-observator.service` em `/etc/systemd/system/`
2. Definir `Restart=on-failure`, `MemoryLimit=200M` (proteção contra OOM)
3. Habilitar com `systemctl enable`
4. Documentar comando exato de start em `knowledge-base/ops.md`
5. Só reativar APÓS upgrade da VM para A1.Flex ARM (24 GB grátis)

**Por que não fazer agora:**
Mexer em systemd durante a incubação viola o pacto de congelamento. 
O observator está corretamente desativado para a janela de 16 dias.

