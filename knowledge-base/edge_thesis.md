# Tese de Edge — Sistema 1 (trading-ai)

> **Preenchido em:** 2026-05-21
> **Revisado em 06/06/2026:** [PREENCHER PÓS-INCUBAÇÃO]

## Hipótese Central

Aposto que [FEATURE_OBSERVÁVEL] precede [MOVIMENTO_DE_PREÇO] em [TIMEFRAME] porque [RAZÃO_ECONÔMICA_OU_COMPORTAMENTAL].

**Exemplo do que preencher (NÃO usar este, escrever o seu):**
> "Aposto que RSI < 30 em ativos da B3 com sentimento de notícias positivo nas últimas 24h precede retorno positivo de >0.5% em 1h porque combinação de oversold técnico + catalisador fundamental cria assimetria de risco/retorno favorável a quem compra contra o pânico."

## Como Medir

- **Métrica primária:** Win rate > 55% em janela de [TIMEFRAME]
- **Métrica secundária:** Expectancy > 0 após custos (corretagem + slippage 0.05%)
- **Profit Factor mínimo:** 1.3 líquido
- **Sample size mínimo:** 50 sinais

## Critério de Validação

Só considero a tese válida se simultaneamente:
- [ ] PF rolling 30 dias > 1.3
- [ ] Win rate > 55%
- [ ] Não há concentração em 1-2 ativos (diversificação real)
- [ ] Drawdown máximo < 15% do PnL acumulado

## Critério de Falsificação

Pivota a tese se em 06/06:
- PF < 1.0 → tese morta, repensar do zero
- 1.0 < PF < 1.3 → marginal, investigar quais sinais funcionam (filtrar)
- PF > 1.3 → segue para Adaptive Sizing

## Notas

[ESPAÇO PARA OBSERVAÇÕES DURANTE INCUBAÇÃO]
