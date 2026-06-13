"""Backtest de edge B3 (read-only) — fractal_lab sobre dados cacheados.

Roda a MESMA hipótese mean-reversion do run_example() do hypothesis_runner
nos ativos B3 que temos em cache (ITUB4, BBDC4), para decidir se vale a pena
construir modelos B3. NÃO toca em produção, config, schema ou VM.

Nota metodológica: o fractal_lab usa lógica mean-reversion própria (ATR/EMA),
que é um PROXY do edge B3, não a réplica exata do pipeline de produção
(RSI/MACD/BB + ML). PF/WR/Sharpe/robustez são invariantes de escala.

Uso:
  .\\venv\\Scripts\\python -X utf8 scripts\\backtest_b3.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from research.fractal_lab.research.hypothesis import build_hypothesis
from research.fractal_lab.research.hypothesis_runner import print_report
from research.fractal_lab.core.orchestrator import Orchestrator

warnings.filterwarnings("ignore")

CACHE = Path("data/fractal_cache")

B3_FILES = {
    "ITUB4": CACHE / "ITUB4_1h_2024-2026.csv",
    "BBDC4": CACHE / "BBDC4_1h_2024-2026.csv",
}


def load_b3(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.title() for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def run_asset(asset: str, path: Path) -> None:
    if not path.exists():
        print(f"[SKIP] {asset}: arquivo não encontrado em {path}")
        return
    df = load_b3(path)
    print(f"\n[{asset}] carregado: {len(df)} barras  "
          f"({df.index.min().date()} → {df.index.max().date()})")

    hypothesis = build_hypothesis(
        name=f"MR_ATR1.0_RR2.0_{asset}",
        asset=asset,
        timeframe="1H",
        entry_logic={"type": "mean_reversion", "threshold": 1.0},
        exit_logic={"type": "fixed_rr", "sl_atr_multiplier": 1.0,
                    "tp_atr_multiplier": 2.0, "max_bars": 48},
        features=["atr", "ema_distance", "atr_ratio", "market_bias"],
        test_period=("2024-09-01", "2025-05-31"),
        oos_period=("2025-06-01", "2026-04-30"),
        regime_filter=None,
        description=f"Mean reversion B3 {asset} — proxy de edge, 1:2 RR.",
        pip_multiplier=100.0,   # 1 pip = R$0.01 (ações), não forex 1e4
    )

    orch = Orchestrator(persist=False)
    report = orch.run_hypothesis(df, hypothesis)
    print_report(report)


if __name__ == "__main__":
    print("=" * 60)
    print("BACKTEST EDGE B3 (read-only) — fractal_lab mean-reversion")
    print("=" * 60)
    for asset, path in B3_FILES.items():
        try:
            run_asset(asset, path)
        except Exception as exc:
            print(f"[ERRO] {asset}: {exc!r}")
    print("\n=== FIM ===")
