"""Compara variantes da estratégia de produção em EUR/USD 1h (read-only, local).

Decompõe o edge: v1.0 (LSTM+XGBoost+consenso) vs v1.1 (XGBoost-only, sem
consenso — o que roda hoje na VM) vs tech-only (sem ML). Responde:
"o LSTM justifica o esforço de portá-lo via ONNX para a VM de 1GB?"

Usa o pipeline REAL (generate_signals_custom + run_backtest) e aplica o gate
de threshold (final_score >= SIGNAL_CONFIDENCE_THRESHOLD) que o run_backtest
sozinho ignora. PF é BRUTO (sem comissão); a coluna 'net~' estima custo.

Uso: .\\venv\\Scripts\\python -X utf8 scripts\\compare_strategy_variants.py
"""
from __future__ import annotations

import sys
import warnings
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

THRESH = config.SIGNAL_CONFIDENCE_THRESHOLD
COMMISSION = config.PAPER_TRADING_COMMISSION_PCT * 2          # round-trip
NOTIONAL = config.PAPER_TRADING_CAPITAL * config.POSITION_SIZE_PCT  # ~1000

VARIANTS = {
    "v1.0 (LSTM+XGB+consenso)": dict(LSTM_ENABLED=True,  XGBOOST_ENABLED=True,  XGBOOST_REQUIRE_CONSENSUS=True),
    "v1.1 (XGBoost-only)":      dict(LSTM_ENABLED=False, XGBOOST_ENABLED=True,  XGBOOST_REQUIRE_CONSENSUS=False),
    "tech-only (sem ML)":       dict(LSTM_ENABLED=False, XGBOOST_ENABLED=False, XGBOOST_REQUIRE_CONSENSUS=False),
}


def profit_factor(trades) -> float:
    if trades.empty:
        return 0.0
    wins = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    losses = -trades.loc[trades["pnl"] < 0, "pnl"].sum()
    return float(wins / losses) if losses > 0 else float("inf")


def net_pf(trades, rt_cost_pct: float = COMMISSION) -> float:
    if trades.empty:
        return 0.0
    cost = rt_cost_pct * NOTIONAL
    net = trades["pnl"] - cost
    wins = net.loc[net > 0].sum()
    losses = -net.loc[net < 0].sum()
    return float(wins / losses) if losses > 0 else float("inf")


def main() -> None:
    print("Fetching EUR/USD 1h (yfinance, ~2y)...")
    base = add_all_indicators(fetch_ohlcv(timeframe="1h", symbol="EURUSD=X"))
    print(f"Barras: {len(base)} ({base.index.min()} -> {base.index.max()})\n")

    print("=== PARTE 1: PF CRU (sem gate de threshold) — o edge da lógica de sinal ===")
    hdr = f"{'Variante':<28} {'Sinais':>7} {'PF bruto':>9} {'PF net~':>8} {'WinRate':>8} {'Sharpe':>7} {'Ret%':>8} {'MaxDD%':>8}"
    print(hdr)
    print("-" * len(hdr))

    fs_by_variant = {}
    for label, flags in VARIANTS.items():
        for k, v in flags.items():
            setattr(config, k, v)
        df = generate_signals_custom(
            base.copy(), config.RSI_BUY, config.RSI_SELL,
            check_news=False, check_sentiment=False, ticker="EUR/USD",
        )
        n_sig = int((df["signal"] != 0).sum())
        fs_by_variant[label] = df.loc[df["signal"] != 0, "final_score"]
        bt = run_backtest(df, sl_atr_mult=config.SL_ATR_MULT, tp_atr_mult=config.TP_ATR_MULT)
        print(f"{label:<28} {n_sig:>7} {profit_factor(bt.trades):>9.3f} {net_pf(bt.trades):>8.3f} "
              f"{bt.win_rate:>7.1f}% {bt.sharpe_ratio:>7.2f} {bt.total_return:>7.1f}% {bt.max_drawdown:>7.1f}%")

    print("\n=== PARTE 2: distribuição do final_score nos sinais (onde está o muro do threshold 55) ===")
    for label, fs in fs_by_variant.items():
        if fs.empty:
            print(f"{label}: sem sinais"); continue
        print(f"\n{label}: n={len(fs)} | max={fs.max():.1f} | média={fs.mean():.1f} | mediana={fs.median():.1f}")
        for t in (30, 40, 45, 50, 55):
            print(f"    final_score >= {t}: {int((fs >= t).sum()):>4} sinais ({(fs >= t).mean()*100:4.1f}%)")

    print("\n=== PARTE 3: + filtro de regime (ADX<35, atr_ratio 0.8-1.5) — com vs sem LSTM ===")
    from core.regime_filter import RegimeFilter
    rf = RegimeFilter(adx_threshold=35, atr_ratio_min=0.8, atr_ratio_max=1.5)
    for label in ("v1.0 (LSTM+XGB+consenso)", "tech-only (sem ML)"):
        for k, v in VARIANTS[label].items():
            setattr(config, k, v)
        df = generate_signals_custom(
            base.copy(), config.RSI_BUY, config.RSI_SELL,
            check_news=False, check_sentiment=False, ticker="EUR/USD",
        )
        df["atr_ratio"] = (df["atr"] / df["atr"].rolling(50).mean()).fillna(0)
        df.loc[~rf.mask(df), "signal"] = 0
        bt = run_backtest(df, sl_atr_mult=config.SL_ATR_MULT, tp_atr_mult=config.TP_ATR_MULT)
        tag = "(roda na VM 1GB)" if "tech" in label else "(precisa LSTM → ONNX/A1)"
        print(f"\n{label} + filtro  {tag}")
        print(f"  trades: {bt.total_trades} | WinRate: {bt.win_rate:.1f}% | Sharpe: {bt.sharpe_ratio:.2f} | Ret: {bt.total_return:.1f}%")
        print(f"  PF bruto={profit_factor(bt.trades):.3f} | net@0.02%fx={net_pf(bt.trades,0.0002):.3f} | net@0.10%B3={net_pf(bt.trades,0.0010):.3f}")

    print("\nNota: PF bruto sem custo. SL/TP de config (2.5/4.0 ATR). atr_ratio = atr/atr_50ma.")


if __name__ == "__main__":
    main()
