"""
Fractal Lab - 5m vs 15m comparison: Fold1 vs Holdout.

Uses the 5m data already exported from MT5 (data/fractal_cache/EURUSD_5m_fold1.csv).
Computes coherence and disorder using 5m (micro) vs 15m resampled (macro) within 1h windows.
Compares Fold1 (dez2024-fev2025) against Holdout (mar-mai2026).
Updates data/fractal_report.json with results and answers the key question.

Usage:
    .\\venv\\Scripts\\python scripts\\export_mt5_fractal.py

No MT5 connection required - uses existing CSV.
To re-export from MT5: delete EURUSD_5m_fold1.csv and run with --export flag.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd

SYMBOL      = "EURUSD"
CACHE_DIR   = _ROOT / "data" / "fractal_cache"
CSV_5M_FOLD1 = CACHE_DIR / "EURUSD_5m_fold1.csv"
REPORT_PATH  = _ROOT / "data" / "fractal_report.json"

WINDOW_MINUTES = 60   # 1h windows: 12 micro (5m) candles + 4 macro (15m) candles per window


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_fold1_5m() -> pd.DataFrame:
    """Load Fold1 5m data from the MT5 CSV export."""
    if not CSV_5M_FOLD1.exists():
        print(f"[ERROR] {CSV_5M_FOLD1} not found.")
        print("        Run with --export flag and MT5 open to download it first.")
        sys.exit(1)

    df = pd.read_csv(CSV_5M_FOLD1, index_col="Datetime", parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.dropna(subset=["Open", "Close"]).sort_index()
    print(f"[fold1]   5m loaded: {len(df):,} candles  ({df.index[0].date()} -> {df.index[-1].date()})")
    return df


def load_holdout_5m() -> pd.DataFrame:
    """Load Holdout 5m data from the yfinance parquet cache."""
    from research.fractal_lab.data_loader import get_period_data
    _, df = get_period_data("holdout")
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.dropna(subset=["Open", "Close"]).sort_index()
    print(f"[holdout] 5m loaded: {len(df):,} candles  ({df.index[0].date()} -> {df.index[-1].date()})")
    return df


def resample_to_15m(df_5m: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 5m candles into 15m candles."""
    df15 = df_5m.resample("15min").agg({
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    }).dropna(subset=["Open", "Close"])
    return df15


# ── MT5 export (optional, for re-downloading Fold1) ───────────────────────────

def export_from_mt5() -> None:
    """Download EURUSD 5m for Fold1 period from MT5 and save to CSV."""
    from datetime import datetime as dt

    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("[ERROR] MetaTrader5 not installed: pip install MetaTrader5")
        sys.exit(1)

    import config
    fold1_start = dt(2024, 9, 1,  0, 0, tzinfo=timezone.utc)
    fold1_end   = dt(2025, 2, 28, 23, 59, tzinfo=timezone.utc)

    timeout = getattr(config, "MT5_TIMEOUT_MS", 60_000)
    if not mt5.initialize(timeout=timeout):
        print(f"[ERROR] MT5 initialize failed: {mt5.last_error()}")
        sys.exit(1)

    print(f"[MT5] Fetching {SYMBOL} 5m  {fold1_start.date()} -> {fold1_end.date()} ...")
    rates = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M5, fold1_start, fold1_end)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        print("[ERROR] MT5 returned no data.")
        sys.exit(1)

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"time": "Datetime", "open": "Open", "high": "High",
                             "low": "Low", "close": "Close", "tick_volume": "Volume"})
    df = df.set_index("Datetime")[["Open", "High", "Low", "Close", "Volume"]].sort_index()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_5M_FOLD1)
    print(f"[MT5] Saved {CSV_5M_FOLD1.name} ({len(df):,} rows)")


# ── Metrics ───────────────────────────────────────────────────────────────────

def run_metrics(df_5m: pd.DataFrame, label: str) -> dict:
    from research.fractal_lab.coherence import compute_coherence
    from research.fractal_lab.disorder import compute_disorder

    df_15m = resample_to_15m(df_5m)
    print(f"  15m candles derived: {len(df_15m):,}")

    coherence = compute_coherence(df_5m, df_15m, window_minutes=WINDOW_MINUTES)
    disorder  = compute_disorder(df_5m, df_15m, window_minutes=WINDOW_MINUTES)

    if coherence["available"]:
        print(f"  coherence_score : {coherence['coherence_score']:.1f}  (n_windows={coherence['n_windows']})")
    else:
        print("  coherence_score : N/A")

    if disorder["available"]:
        print(f"  disorder_score  : {disorder['disorder_score']:.1f}  (n_windows={disorder['n_windows']})")
    else:
        print("  disorder_score  : N/A")

    return {
        "coherence": {k: v for k, v in coherence.items() if k != "window_scores"},
        "disorder":  {k: v for k, v in disorder.items()  if k != "window_scores"},
    }


# ── Answer + report ───────────────────────────────────────────────────────────

def build_answer(f1_dis, f1_coh, ho_dis, ho_coh) -> str:
    if f1_dis is None:
        return "INCONCLUSIVO: Fold1 sem dados validos."
    if ho_dis is None:
        return f"INCONCLUSIVO: Holdout sem dados. Fold1 disorder={f1_dis:.1f}"

    delta = f1_dis - ho_dis
    f1_str = f"disorder={f1_dis:.1f}, coherence={f1_coh:.1f}" if f1_coh else f"disorder={f1_dis:.1f}"
    ho_str = f"disorder={ho_dis:.1f}, coherence={ho_coh:.1f}" if ho_coh else f"disorder={ho_dis:.1f}"

    if delta > 5:
        verdict = (
            f"SIM - disorder era MAIOR no Fold1 (delta={delta:+.1f}). "
            f"Mercado mais caotico em dez2024-fev2025: "
            f"sinais de reversao do Sistema 1 falharam."
        )
    elif delta < -5:
        verdict = (
            f"NAO - disorder era MENOR no Fold1 (delta={delta:+.1f}). "
            f"Desordem fractal nao explica as perdas; "
            f"outro fator dominante (ex: tendencia EMA200 em downtrend)."
        )
    else:
        verdict = (
            f"NEUTRO - disorder Fold1 aprox. Holdout (delta={delta:+.1f}, dentro de +-5). "
            f"Sem diferenca estrutural de desordem entre os periodos."
        )

    if f1_coh is not None and ho_coh is not None:
        dcoh = f1_coh - ho_coh
        verdict += f" Coerencia {'maior' if dcoh > 0 else 'menor'} no Fold1 (delta={dcoh:+.1f})."

    return f"{verdict} | Fold1: {f1_str} | Holdout: {ho_str}"


def save_report(fold1: dict, holdout: dict) -> dict:
    report = {}
    if REPORT_PATH.exists():
        with open(REPORT_PATH, encoding="utf-8") as f:
            report = json.load(f)

    f1_dis = fold1["disorder"].get("disorder_score")
    f1_coh = fold1["coherence"].get("coherence_score")
    ho_dis = holdout["disorder"].get("disorder_score")
    ho_coh = holdout["coherence"].get("coherence_score")

    report["updated_at"] = datetime.now(timezone.utc).isoformat()
    report["method"]     = f"5m_vs_15m_resampled  window={WINDOW_MINUTES}min"
    report["fold1_5m"] = {
        "source": str(CSV_5M_FOLD1.name),
        "coherence": fold1["coherence"],
        "disorder":  fold1["disorder"],
    }
    report["holdout_5m"] = {
        "source": "yfinance cache",
        "coherence": holdout["coherence"],
        "disorder":  holdout["disorder"],
    }
    report["comparison"] = {
        "fold1_disorder":  f1_dis,
        "fold1_coherence": f1_coh,
        "holdout_disorder":  ho_dis,
        "holdout_coherence": ho_coh,
        "disorder_delta_fold1_minus_holdout":  round(f1_dis - ho_dis, 2) if f1_dis and ho_dis else None,
        "coherence_delta_fold1_minus_holdout": round(f1_coh - ho_coh, 2) if f1_coh and ho_coh else None,
    }
    report["answer"] = build_answer(f1_dis, f1_coh, ho_dis, ho_coh)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    return report


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true",
                        help="Re-download Fold1 5m from MT5 before running (MT5 must be open)")
    args = parser.parse_args()

    print("=" * 64)
    print("  Fractal Lab - 5m vs 15m  |  Fold1 vs Holdout")
    print(f"  Window: {WINDOW_MINUTES}min  |  Micro: 5m  |  Macro: 15m (resampled)")
    print("=" * 64)

    if args.export:
        export_from_mt5()

    # Load data
    df_fold1   = load_fold1_5m()
    df_holdout = load_holdout_5m()

    # Compute metrics
    print(f"\n[fold1]   Computing metrics...")
    fold1_metrics   = run_metrics(df_fold1, "Fold1")

    print(f"\n[holdout] Computing metrics...")
    holdout_metrics = run_metrics(df_holdout, "Holdout")

    # Save and show results
    report = save_report(fold1_metrics, holdout_metrics)

    print(f"\n{'='*64}")
    print("PERGUNTA: disorder_score era maior antes dos trades ruins do Sistema 1?")
    print(f"RESPOSTA: {report['answer']}")
    print(f"{'='*64}")

    print("\n-- Distribuicao dos scores --")
    for label, m in [("Fold1  ", fold1_metrics), ("Holdout", holdout_metrics)]:
        d = m["disorder"]
        c = m["coherence"]
        if d.get("available"):
            p = d["percentiles"]
            print(f"  {label} disorder   p25={p['p25']:5.1f}  p50={p['p50']:5.1f}  p75={p['p75']:5.1f}  mean={d['disorder_score']:5.1f}")
        if c.get("available"):
            p = c["percentiles"]
            print(f"  {label} coherence  p25={p['p25']:5.1f}  p50={p['p50']:5.1f}  p75={p['p75']:5.1f}  mean={c['coherence_score']:5.1f}")

    print(f"\n[report] Saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
