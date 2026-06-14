# Checklist de verificação — v1.2 ao vivo (EUR/USD)

**Criado:** 2026-06-14
**Contexto:** a v1.2 entrou no ar em 14/06 (sábado). É a primeira config que
implementa a estratégia validada (LSTM via ONNX + XGBoost + consenso + filtro de
regime). Edge esperado: net PF ~1.33 (backtest 2 anos). Ver `incident_20260613.md`.

> **Quando rodar:** segunda-feira (reabertura do forex) e depois 1x/semana.
> O mercado fecha no fim de semana → até lá fica HOLD com dado estático (normal).

---

## Comando único (read-only, roda na VM)

```powershell
ssh -i "$env:USERPROFILE\.ssh\ssh-key-2026-05-05.key" ubuntu@137.131.228.166 `
  "cd ~/trading-ai && ./venv/bin/python scripts/status_v12.py"
```

(local, contra um db copiado: `.\venv\Scripts\python scripts\status_v12.py`)

---

## O que é "verde" (tudo certo)

1. **Config:** `✅ config v1.2 correta` (v1.2, LSTM on, consenso on, regime on).
2. **Frescura:** `signals_1h` com idade < algumas horas em pregão (até ~65h em
   fim de semana é normal). Se > 72h em dia útil → persistência travou.
3. **Trades:** a partir de segunda, trades começam a aparecer. Esperado:
   WinRate ~50%, PF ~1.47 bruto / ~1.33 net. **Poucos trades é esperado**
   (consenso + filtro de regime são seletivos — ~119 trades em 2 anos no backtest).
4. **Decisão do último candle:** mostra lstm_prob, xgboost_prob, adx, atr_ratio.
   Serve para entender *por que* está HOLD ou abrindo trade.

---

## Bandeiras vermelhas (investigar)

| Sintoma | Provável causa | Ação |
|---|---|---|
| `signals_1h` congelado > 72h em pregão | persistência/serviço parado | `systemctl status trading-ai`; ver `scheduler.log` |
| `lstm_prob` sempre = 50.0 | ONNX não carregou na VM | ver log `LSTM backend: onnx`; conferir `models/lstm_eurusd_1h.onnx` + onnxruntime |
| 0 trades por **muitos** dias de pregão | filtro/consenso muito restritivo OU mercado sem setup | normal se poucos dias; se semanas, revisar com calma (NÃO baixar gates no impulso) |
| WinRate << 45% ou PnL afundando | edge não se sustenta ao vivo | **não mexer no impulso**; juntar amostra e reavaliar a tese |
| Telegram sem alertas há dias | bot/credenciais ou serviço | testar `scheduler.log` e o healthcheck |

---

## Princípios (não violar no impulso)

- **Amostra antes de conclusão.** O edge é real mas pequeno (Sharpe 1.38,
  retorno ~0.9% em 2 anos). Precisa de dezenas de trades antes de julgar.
- **Não baixar gates** (consenso/regime/threshold) para "ter mais trades" —
  foram eles que criaram o edge. Mais trades = a estratégia perdedora (net 0.96).
- **Capital real (mesmo demo) só depois** de ver a v1.2 operando coerente por
  um tempo, e **com o custo real da corretora confirmado** (o edge morre a 0.10%).
- Dados são `STRATEGY_VERSION=v1.2` — não misturar com v1.0/v1.1 na análise.

---

## Se precisar do contexto completo numa conversa nova
Mande para o Claude: `knowledge-base/incident_20260613.md` (diagnóstico),
este checklist, e o estado no topo do `CLAUDE.md`.
