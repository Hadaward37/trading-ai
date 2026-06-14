"""Valida que o pipeline de PRODUÇÃO sob a config v1.2 reproduz o edge validado.

Usa generate_signals_custom (agora com filtro de regime + consenso embutidos) e
o threshold do config. Esperado: ~119 trades, PF bruto ~1.47, net@0.02% ~1.33.
Read-only, local. Se não bater, NÃO deploya.
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from core.collector import fetch_ohlcv
from core.indicators import add_all_indicators
from core.signals import generate_signals_custom
from core.backtest import run_backtest

NOTIONAL = config.PAPER_TRADING_CAPITAL * config.POSITION_SIZE_PCT


def pf(trades, rt_cost=0.0):
    if trades.empty:
        return 0.0
    net = trades["pnl"] - rt_cost * NOTIONAL
    w = net[net > 0].sum(); l = -net[net < 0].sum()
    return float(w / l) if l > 0 else float("inf")


print("Config v1.2 ativa:")
print(f"  LSTM_ENABLED={config.LSTM_ENABLED} | REQUIRE_CONSENSUS={config.XGBOOST_REQUIRE_CONSENSUS}")
print(f"  REGIME_FILTER_ENABLED={config.REGIME_FILTER_ENABLED} "
      f"(ADX<{config.REGIME_FILTER_ADX_MAX}, atr_ratio {config.REGIME_FILTER_ATR_MIN}-{config.REGIME_FILTER_ATR_MAX})")
print(f"  THRESHOLD={config.SIGNAL_CONFIDENCE_THRESHOLD} | comissão={config.PAPER_TRADING_COMMISSION_PCT*2*100:.2f}% rt")
print(f"  VERSION={config.STRATEGY_VERSION}\n")

base = add_all_indicators(fetch_ohlcv(timeframe="1h", symbol="EURUSD=X"))
df = generate_signals_custom(base.copy(), config.RSI_BUY, config.RSI_SELL,
                             check_news=False, check_sentiment=False, ticker="EUR/USD")
# aplica o gate de threshold do config (0 → não bloqueia)
df.loc[df["final_score"] < config.SIGNAL_CONFIDENCE_THRESHOLD, "signal"] = 0

bt = run_backtest(df, sl_atr_mult=config.SL_ATR_MULT, tp_atr_mult=config.TP_ATR_MULT)
print(f"Sinais: {int((df['signal']!=0).sum())} | trades: {bt.total_trades} | "
      f"WinRate: {bt.win_rate:.1f}% | Sharpe: {bt.sharpe_ratio:.2f} | Ret: {bt.total_return:.1f}%")
print(f"PF bruto:           {pf(bt.trades):.3f}")
print(f"PF net @0.02% (fx): {pf(bt.trades, 0.0002):.3f}")
print(f"\nEsperado ~ trades 119 | PF bruto 1.47 | net 1.33")
print("✅ pipeline de produção reproduz o edge validado"
      if bt.total_trades > 80 and pf(bt.trades) > 1.3 else
      "⚠️ divergência — NÃO deploya, investigar")
