# Fractal Lab — Achados de Pesquisa

> Sistema 1 permanece intocado. Todos os resultados são pesquisa pré-incubação.
> Implementação só após 2026-06-06 e nova auditoria walk-forward completa.

---

## Conclusão da Pesquisa — Fractal Lab (2026-05-07)

### Sistema 1: Capturador de Reversão Confirmada por Design

O Sistema 1 **não tem um erro de timing**. O `BB_DEV=2.0` com janela de 20 períodos
é um filtro implícito de exaustão: aguarda o preço atingir 2σ (~2.4 ATR) antes de
sinalizar reversão. Isso é uma decisão de design defensiva, não um delay acidental.

**Por que a espera faz sentido:** Entrando em B2 (0.5–1 ATR), o preço ainda tem
espaço para continuar até B5 (~3 ATR adicionais), derrubando o SL (2.5 ATR) antes
de reverter. O backtest confirmou isso: WR cai de 51.8% → 23.5% com entrada antecipada.

### Adaptive Sizing: Candidato Mais Robusto Identificado

O problema não é o timing — é a **gestão de risco em B5** (75.8% dos trades, PF=1.06).

| Scenario | MaxDD Train | MaxDD Test | DR |
|----------|-------------|------------|-----|
| Baseline 1.0x | 1446 pips | 239 pips | 2.92 |
| **Adaptive B5=0.1x** | **254 pips (−82%)** | **44 pips (−81%)** | 2.82 |
| Adaptive B5=0.0x | 210 pips | 42 pips | 2.75 |

**MaxDD é reduzido consistentemente em −82% (train) e −81% (test).**
Esta é a única métrica que se confirma robustamente out-of-sample.

### Status: Hipótese Operacional Promissora — Não Conclusão Definitiva

| Achado | Evidência | Status |
|--------|-----------|--------|
| B5 tem edge mínimo em bull run Jan-Mai2026 | PF=1.06, n=235 | Confirmado no período |
| B5 pode ter edge em outros regimes | Jun-Aug2025: E=+20 pips | Regime-dependente |
| Adaptive sizing reduz MaxDD | −82% train, −81% test | Robusto entre períodos |
| B5=0.0x superior a B5=0.1x | n_test=7 vs n_test=37 | Inconclusivo (n insuficiente) |

**Recomendação para 06/06:** Testar `B5=0.1x` (não 0.0x) — preserva exposição em
regimes onde B5 é favorável, limita o impacto no MaxDD.

---

## Achado v3 — Timing de Entrada e Stretch Analysis (2026-05-07)

### Problema observado

75.8% dos 310 sinais do Sistema 1 entram com stretch > 2 ATR (Bucket B5),
onde o edge é mínimo (PF=1.06, E=+1.31 pips). O ponto ótimo de performance
é B2 (0.5–1.0 ATR), com PF=5.61 e E=+30.55 pips.

| Bucket         | N    | %     | PF   | E (pips) | Win Rate |
|----------------|------|-------|------|----------|----------|
| B1 [0.0–0.5)  | 14   | 4.5%  | 5.53 | +26.26   | 78.6%    |
| **B2 [0.5–1.0)** | **19** | **6.1%** | **5.61** | **+30.55** | **73.7%** |
| B3 [1.0–1.5)  | 19   | 6.1%  | 3.59 | +24.11   | 68.4%    |
| B4 [1.5–2.0)  | 23   | 7.4%  | 3.02 | +18.91   | 69.6%    |
| **B5 [2.0+]** | **235** | **75.8%** | **1.06** | **+1.31** | **42.5%** |

KS-test B1 vs B5: p=0.0163 (SIGNIFICATIVO — distribuições distintas).

---

### Causa estrutural do timing de entrada

**Componente responsável:** `v_bb_buy` em `core/signals.py` linha 43:

```python
v_bb_buy = (df["close"] <= df["bb_lower"] * (1 + BB_TOLERANCE)).astype(int)
```

**Parâmetros em `config.py`:**
```python
BB_WINDOW = 20      # SMA20 como centro da banda
BB_DEV    = 2.0     # 2 desvios-padrão = ~2.41 ATR de largura média
```

**Mecanismo do delay:**

A condição BB touch dispara quando `close <= SMA20 - 2σ`. Com os dados reais:
- BB half-width médio = **2.41 ATR** (mediana 2.13 ATR)
- Quando `v_bb_buy` dispara, o preço está ≈ 2–3 ATR abaixo do SMA20
- EMA50 ≈ SMA20 no mesmo período → preço está em B5 (>2 ATR da média)

**Contribuição dos outros votantes para o delay:**
- `v_rsi_buy`: RSI < 35 requer ~9+ candles bearish em 14 períodos → confirma após movimento já estabelecido
- `v_macd_up`: Crossover do MACD (12/26 EMA) → leva 26+ bars para confirmar virada
- **Regra 2-de-3**: Pelo menos 2 dos 3 devem disparar simultaneamente → espera pelos mais lentos

**Síntese:** O BB touch (BB_DEV=2.0) é o denominador comum de todas as entradas. Ele requer que o preço percorra 2+ ATR além da média antes de sinalizar — exatamente a definição de B5.

---

### Por que entrada antecipada (B2) falha

**Simulação hipotética** (85 trades com precursor B2 identificado, lookback=12h):

| Método | N | Win Rate | PF | E (pips) | MaxDD |
|--------|---|----------|----|----------|-------|
| Actual (B5) | 85 | 51.8% | 1.56 | +9.10 | 487.5 |
| Early (B2) | 85 | **23.5%** | **0.42** | **−12.41** | **1167.5** |

**Razão matemática:** Preço entra em B2 (stretch ~0.77 ATR) com SL = 2.5 ATR.
O preço segue movendo de B2 → B5 (≈3 ATR adicionais), ultrapassando o SL (2.5 ATR)
antes de reverter. A espera em B5 é **estruturalmente necessária** como filtro de
exaustão — o mercado ainda não terminou o movimento quando está em B2.

> **Insight crítico:** O BB_DEV=2.0 não é um "delay" acidental — é um filtro implícito
> de exaustão de momentum. Ele aguarda confirmação de que o mercado atingiu um extremo
> real, não apenas um pullback temporário.

---

### Solução proposta: Adaptive Sizing (pós-incubação)

Em vez de tentar antecipar a entrada, **reduzir a exposição em B5** onde o edge é mínimo:

| Bucket | Sizing | Justificativa |
|--------|--------|---------------|
| B1 [0.0–0.5) | 1.0x | PF=5.53, alta confiança |
| B2 [0.5–1.0) | 1.0x | PF=5.61, ponto ótimo |
| B3 [1.0–1.5) | 0.7x | PF=3.59, ainda favorável |
| B4 [1.5–2.0) | 0.3x | PF=3.02, edge presente mas declinante |
| B5 [2.0+] | **0.0x** | PF=1.06, E=+1.31 — edge insuficiente |

**Resultado da simulação de sizing (310 trades):**

| Métrica | Atual (1.0x tudo) | Adaptive Sizing |
|---------|-------------------|-----------------|
| Total PnL | +2147.9 pips | +1399.2 pips |
| MaxDD | 487.5 pips | **29.9 pips** |
| Sharpe proxy | 0.205 | **0.496** |
| Trades ativos | 310 | **75** (24.2%) |

- MaxDD reduz **−94%** (de 487.5 para 29.9 pips)
- Sharpe melhora **+142%** (de 0.205 para 0.496)
- Total PnL reduz 34.8% porque bloqueia os 235 trades de B5 (incluindo os com E=+1.31 que somam ruído)

> **Trade-off:** Menos trades, menos PnL bruto, mas risco dramaticamente menor e qualidade
> muito maior por trade ativo.

---

### Resposta às perguntas obrigatórias (v3)

| Pergunta | Resposta |
|----------|----------|
| Edge aumenta com stretch? | **NÃO — DIMINUI** (Spearman ρ=−0.90, p=0.037). Declínio monotônico após B2. |
| Ponto de exaustão ótimo? | **B2 [0.5–1.0 ATR]** — PF=5.61, E=+30.55 (n=19, cautela pelo N pequeno) |
| Reversões extremas (>2 ATR) melhoram PF? | **DEGRADA** — B5 PF=1.06 vs B1 PF=5.53. N=235, resultado robusto. |
| Lucro vem de mean reversion pós-deslocamento? | **NÃO** — Counter-trend (n=22): E=+22.70 vs MR-setup (n=288): E=+5.72. |
| Antecipar entrada preserva Win Rate? | **NÃO** — WR cai de 51.8% para 23.5% (−28.2%), MaxDD aumenta +680 pips. |

---

### Próximos passos (pós-incubação 2026-06-06)

1. **Adaptive Sizing:** Implementar tabela B1–B5 no `core/signals.py` como parâmetro de
   `position_size_factor` sem alterar lógica de sinal ou thresholds.

2. **Validação walk-forward:** Antes de ativar, rodar nova auditoria no período
   mai2024–mai2026 com o sizing aplicado retroativamente.

3. **Monitoramento:** Adicionar `stretch_bucket` ao `data/telemetry.json` para acompanhar
   em tempo real a distribuição de stretch nos sinais gerados.

---

### Arquivos de pesquisa

| Arquivo | Descrição |
|---------|-----------|
| `research/fractal_lab/regime_classifier.py` | Classificador de regimes + stretch features |
| `scripts/run_regime_analysis.py` | Pipeline completo v1/v2/v3 |
| `scripts/simulate_early_entry.py` | Simulação hipotética de entrada antecipada |
| `data/regime_analysis.json` | Relatório completo regime + stretch |
| `data/early_entry_simulation.json` | Resultados da simulação |
