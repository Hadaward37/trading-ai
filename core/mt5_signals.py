"""MT5 signal export — writes live signals to a file consumed by the MT5 EA.

The MetaTrader5 Python API does not expose chart-object functions (those are
MQL5-only).  The bridge used here is a CSV file that the EA
``TradingAI_Signals.mq5`` polls and uses to draw arrows/lines on the chart.

File format (data/mt5_signals.csv):
  datetime, symbol, timeframe, signal, entry, sl, tp, score, atr
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config

logger = logging.getLogger(__name__)

SIGNALS_CSV  = _ROOT / "data" / "mt5_signals.csv"
SIGNALS_JSON = _ROOT / "data" / "mt5_signals.json"

_CSV_FIELDS = ["datetime", "symbol", "timeframe", "signal", "entry", "sl", "tp", "score", "atr"]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _compute_sl_tp(entry: float, signal: int, atr: float) -> tuple[float, float]:
    sl = entry - signal * config.SL_ATR_MULT * atr
    tp = entry + signal * config.TP_ATR_MULT * atr
    return round(sl, 5), round(tp, 5)


# ── Public API ────────────────────────────────────────────────────────────────

def export_signals(
    df: pd.DataFrame,
    symbol: str = "EURUSD",
    timeframe: str = "H1",
    atr_col: str = "atr",
) -> list[dict]:
    """Compute SL/TP for every non-zero signal bar and write to CSV + JSON.

    Args:
        df:        DataFrame with ``signal``, ``close``, ``score``, and ATR column.
        symbol:    MT5 symbol name.
        timeframe: Label like ``"H1"``.
        atr_col:   Column name for ATR values.

    Returns:
        List of signal dicts written (only rows where signal != 0).
    """
    if "signal" not in df.columns or atr_col not in df.columns:
        logger.warning("export_signals: missing 'signal' or '%s' column", atr_col)
        return []

    SIGNALS_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for ts, row in df.iterrows():
        sig = int(row["signal"])
        if sig == 0:
            continue

        entry = float(row["close"])
        atr   = float(row[atr_col]) if not pd.isna(row[atr_col]) else 0.0
        sl, tp = _compute_sl_tp(entry, sig, atr)

        rows.append({
            "datetime":  str(ts),
            "symbol":    symbol,
            "timeframe": timeframe,
            "signal":    sig,
            "entry":     round(entry, 5),
            "sl":        sl,
            "tp":        tp,
            "score":     round(float(row.get("score", 0)), 1),
            "atr":       round(atr, 6),
        })

    with open(SIGNALS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with open(SIGNALS_JSON, "w") as f:
        json.dump(rows, f, indent=2)

    logger.info("Exported %d signal(s) → %s", len(rows), SIGNALS_CSV)
    return rows


def get_latest_signal(df: pd.DataFrame) -> dict | None:
    """Return the most recent non-zero signal row as a dict, or None."""
    sig_df = df[df["signal"] != 0]
    if sig_df.empty:
        return None
    row = sig_df.iloc[-1]
    sig   = int(row["signal"])
    entry = float(row["close"])
    atr   = float(row["atr"]) if "atr" in row and not pd.isna(row["atr"]) else 0.0
    sl, tp = _compute_sl_tp(entry, sig, atr)
    return {
        "datetime": str(sig_df.index[-1]),
        "signal":   sig,
        "label":    "BUY" if sig == 1 else "SELL",
        "entry":    round(entry, 5),
        "sl":       sl,
        "tp":       tp,
        "score":    round(float(row.get("score", 0)), 1),
        "atr":      round(atr, 6),
        "rsi":      round(float(row.get("rsi", 0)), 1),
        "adx":      round(float(row.get("adx", 0)), 1),
        "stoch_k":  round(float(row.get("stoch_k", 0)), 1),
    }


def print_signal_report(latest: dict | None, symbol: str, timeframe: str) -> None:
    """Print a formatted signal report to stdout."""
    bar = "=" * 58
    print(f"\n{bar}")
    print(f"  trading-ai | {symbol} | {timeframe} | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(bar)

    if latest is None:
        print("  Sinal atual: HOLD (nenhum sinal nos dados recentes)")
        print(bar)
        return

    direction = latest["label"]
    score     = latest["score"]
    bar_label = "[BUY] " if direction == "BUY" else "[SELL]"

    print(f"  Sinal:    {bar_label}  |  Score: {score:.0f}/100")
    print(f"  Barra:    {latest['datetime']}")
    print(f"  Entry:    {latest['entry']:.5f}")
    print(f"  SL:       {latest['sl']:.5f}  (-{config.SL_ATR_MULT}x ATR)")
    print(f"  TP:       {latest['tp']:.5f}  (+{config.TP_ATR_MULT}x ATR)")
    print(f"  ATR:      {latest['atr']:.6f}")
    print(f"  RSI:      {latest['rsi']:.1f}  |  ADX: {latest['adx']:.1f}  |  Stoch %K: {latest['stoch_k']:.1f}")
    print(bar)
    print(f"  Sinais exportados >> data/mt5_signals.csv  (lido pelo EA)")
    print(bar + "\n")
