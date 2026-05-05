# TradingView — Pine Script Export

Dois scripts exportados do sistema Python `trading-ai`, com lógica idêntica ao engine de sinais.

---

## Arquivos

| Arquivo | Tipo | Para quem |
|---|---|---|
| `strategy.pine` | Strategy (executa ordens) | Backtesting no TradingView, ordens automáticas via Paper Trading |
| `indicator.pine` | Indicator (só exibe sinais) | Ver sinais e score sem automatizar nada |

---

## Instalação — strategy.pine (Strategy completa)

### Passo 1 — Abrir o Pine Editor

1. Abra o TradingView (tradingview.com)
2. Abra o gráfico do ativo desejado (ex: **EURUSD**, **VALE3**, **PETR4**)
3. Na barra inferior, clique em **"Pine Editor"**

### Passo 2 — Colar o código

1. Selecione todo o texto padrão no editor (`Ctrl+A`)
2. Delete e cole o conteúdo de `strategy.pine`
3. Clique em **"Add to chart"** (ou `Ctrl+Enter`)

### Passo 3 — Configurar o ativo

O script funciona em qualquer ativo e timeframe. Recomendado:

| Ativo | Timeframe | Configuração |
|---|---|---|
| EUR/USD | 1H | Padrão (RSI 35/75, ADX 25) |
| VALE3 / PETR4 | 1H | Padrão (os mesmos parâmetros funcionam) |
| IBOV | 1H | Padrão |

### Passo 4 — Ajustar parâmetros (opcional)

Clique no ícone de engrenagem ⚙️ ao lado do nome da strategy. Parâmetros disponíveis:

**RSI**
- `Período`: 14 (padrão)
- `Compra <`: 35 — RSI abaixo deste valor vota BUY
- `Venda >`: 75 — RSI acima deste valor vota SELL

**MACD**
- Rápida: 12 / Lenta: 26 / Sinal: 9

**Bollinger Bands**
- Período: 20 / Desvio: 2.0
- `Tolerância de toque`: 0.2% — margem além da banda para contar como toque

**ATR & Risco**
- `Stop Loss (× ATR)`: 2.5 — SL fica 2.5× o ATR abaixo/acima do preço de entrada
- `Take Profit (× ATR)`: 4.0 — TP fica 4.0× o ATR

**ADX — Filtro de Tendência**
- `Threshold`: 25 — sinais bloqueados quando ADX < 25 (mercado lateral)

**Estocástico**
- Overbought: 75 / Oversold: 25

**Visual**
- Ativar/desativar BBs, labels de score, linhas SL/TP, dashboard

---

## Instalação — indicator.pine (Só indicadores)

### Passo 1 — Adicionar indicador customizado

1. No gráfico, clique em **"Indicators"** (barra superior)
2. Aba **"Pine Editor"** → cole o conteúdo de `indicator.pine`
3. Clique em **"Add to chart"**

O indicador abre um **painel separado abaixo do gráfico de preço**.

### O que o painel mostra

```
┌─────────────────────────────────────────────────────┐
│  RSI (roxo)   ─────────── zonas verde/vermelho      │
│  Stoch %K (azul) ─────── thresholds pontilhados     │
│  ADX (laranja) ────────── linha de tendência        │
│  Score (barras):                                    │
│    ▲ verde = BUY ativo  (altura = score 0-100)      │
│    ▼ vermelho = SELL ativo                          │
│  Background verde/vermelho quando sinal ativo       │
│  ▲ ▼ shapes nas barras de sinal                    │
└─────────────────────────────────────────────────────┘
```

---

## Lógica de Sinais

### Sistema de votação 2-de-3

O sinal só dispara quando **pelo menos 2 dos 3 indicadores** votam na mesma direção, **E** os filtros ADX e Stoch confirmam.

#### BUY ▲
| Condição | Critério | Papel |
|---|---|---|
| RSI | RSI < 35 | Voto (1 de 3) |
| MACD | Crossover para cima | Voto (1 de 3) |
| Bollinger Bands | Preço toca banda inferior | Voto (1 de 3) |
| ADX | ADX ≥ 25 (tendência) | Filtro obrigatório |
| Estocástico | %K < 25 (sobrevendido) | Filtro obrigatório |

→ **BUY** = (RSI + MACD + BB ≥ 2 votos) **AND** ADX ≥ 25 **AND** Stoch < 25

#### SELL ▼
| Condição | Critério | Papel |
|---|---|---|
| RSI | RSI > 75 | Voto (1 de 3) |
| MACD | Crossover para baixo | Voto (1 de 3) |
| Bollinger Bands | Preço toca banda superior | Voto (1 de 3) |
| ADX | ADX ≥ 25 (tendência) | Filtro obrigatório |
| Estocástico | %K > 75 (sobrecomprado) | Filtro obrigatório |

→ **SELL** = (RSI + MACD + BB ≥ 2 votos) **AND** ADX ≥ 25 **AND** Stoch > 75

### Score 0-5

Cada sinal exibe um score indicando quantas das 5 condições confirmaram:

| Score | Significado |
|---|---|
| 5/5 | Todas as condições alinhadas — sinal muito forte |
| 4/5 | Alta confiança |
| 3/5 | Mínimo necessário (2 votos + filtros) |

### Stop Loss & Take Profit

```
BUY:  SL = Preço − 2.5 × ATR(14)    TP = Preço + 4.0 × ATR(14)
SELL: SL = Preço + 2.5 × ATR(14)    TP = Preço − 4.0 × ATR(14)
```

Risco/Retorno implícito: **1 : 1.6** (TP/SL = 4.0/2.5)

---

## Configurar Alertas

### No TradingView

1. Clique com o botão direito no gráfico → **"Add Alert"**
2. Em **"Condition"**, selecione:
   - `Trading AI` → `Trading AI — BUY` (apenas compras)
   - `Trading AI` → `Trading AI — SELL` (apenas vendas)
   - `Trading AI` → `Trading AI — Qualquer Sinal` (ambos)
3. **"Notify on App"** ou **"Send email"** ou **"Webhook URL"**
4. Clique em **"Create"**

### Mensagem de alerta (formato)
```
BUY | EURUSD | Score: 4/5 | RSI: 28.3
SELL | VALE3 | Score: 3/5 | RSI: 78.1
```

### Webhook para integrar com o Python

Se quiser receber os alertas do TradingView no bot Telegram:
1. Configure um webhook no TradingView apontando para seu servidor
2. O servidor recebe o JSON e encaminha via `core/notifier.py`

---

## Equivalência Python ↔ Pine Script

| Módulo Python | Equivalente Pine Script |
|---|---|
| `core/indicators.py` | Cálculo de RSI, MACD, BB, ATR, ADX, Stoch |
| `core/signals.py` — `_vote_series()` | Sistema de votação 2-de-3 |
| `core/signals.py` — regime filter | Filtro ADX ≥ 25 |
| `core/optimizer.py` — parâmetros v2 | Defaults dos inputs |
| `core/backtest.py` — SL/TP | `strategy.exit()` com stop/limit |

Os resultados do backtesting no TradingView e no Python podem diferir por:
- Dados OHLCV ligeiramente diferentes (yfinance vs feed do TradingView)
- Pine Script usa `close` do bar atual para preço de entrada; Python usa `close` do bar fechado anterior

---

## Resultados históricos (Python — EUR/USD 1H)

| Configuração | Sharpe | Win Rate | Trades |
|---|---|---|---|
| v2 (ADX + Stoch) | 1.299 | 47.2% | 212 |
| v1 (baseline) | 1.494 | 45.2% | 425 |
| Alt #3 (TP 3.0×) | — | 52.7% | 205 |
