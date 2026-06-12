# Asimov Academy — Baixar dados B3 com MT5 e Python
Fonte: https://youtu.be/x0LmTaKMKvY

## Requisitos
- Windows obrigatório
- MetaTrader 5 aberto
- Conta corretora com MT5 (Clear já temos)
- pip install MetaTrader5 pandas

## Código base
```python
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

login, password = open('credentials').read().split()
server = "Clear-Live"

if not mt5.initialize(login=int(login), password=password, server=server):
    print("Falha:", mt5.last_error())
    mt5.shutdown()

# Candles
candles = mt5.copy_rates_range("PETR4", mt5.TIMEFRAME_M1,
    datetime(2024,1,1), datetime(2024,12,31))
df = pd.DataFrame(candles)
df['time'] = pd.to_datetime(df['time'], unit='s')

# Ticks
ticks = mt5.copy_ticks_range("PETR4",
    datetime(2024,1,1), datetime(2024,12,31),
    mt5.COPY_TICKS_ALL)
df_ticks = pd.DataFrame(ticks)
df_ticks['time'] = pd.to_datetime(df_ticks['time'], unit='s')
```

## Dicas críticas
- login DEVE ser int()
- MT5 deve estar aberto antes de rodar
- pd.to_datetime(..., unit='s') obrigatório
- Credenciais em arquivo separado

## Relevância pós 06/07
- Substituir yfinance por MT5 no Sistema 1
- Dados reais B3: PETR4, VALE3, ITUB4, WIN, WDO
- Tick data para Dollar Bars (López de Prado)
