"""
Export MT5 historical data for Fractal Lab Fold1 analysis.

Downloads EURUSD 1m and 5m for Sep2024–Feb2025 (Fold1) from MetaTrader 5,
saves to CSV, then runs coherence + disorder metrics and compares with the
Holdout baseline already computed in data/fractal_report.json.

PREREQUISITE: MT5 terminal must be open and logged in before running.

Usage:
    .\\venv\\Scripts\\python scripts\\export_mt5_fractal.py
    .\\venv\\Scripts\\python scripts\\export_mt5_fractal.py --skip-export   # reuse existing CSVs
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Windows console may default to cp1252; force UTF-8 so unicode prints work.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd

SYMBOL = "EURUSD"
FOLD1_START = datetime(2024, 9, 1,  0, 0, tzinfo=timezone.utc)
FOLD1_END   = datetime(2025, 2, 28, 23, 59, tzinfo=timezone.utc)

CACHE_DIR   = _ROOT / "data" / "fractal_cache"
CSV_1M      = CACHE_DIR / "EURUSD_1m_fold1.csv"
CSV_5M      = CACHE_DIR / "EURUSD_5m_fold1.csv"
REPORT_PATH = _ROOT / "data" / "fractal_report.json"


# ── MT5 export ────────────────────────────────────────────────────────────────

def _connect_mt5():
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("[ERROR] MetaTrader5 package not installed.")
        print("        Run: pip install MetaTrader5")
        sys.exit(1)

    import config
    timeout = getattr(config, "MT5_TIMEOUT_MS", 60_000)
    if not mt5.initialize(timeout=timeout):
        print(f"[ERROR] MT5 initialize() failed: {mt5.last_error()}")
        print("        Make sure MT5 is open and logged in.")
        sys.exit(1)

    info = mt5.terminal_info()
    print(f"[MT5] Connected — build={info.build}  data_path={info.data_path}")
    return mt5


def _fetch_range(mt5, symbol: str, timeframe, tf_name: str, date_from: datetime, date_to: datetime) -> pd.DataFrame:
    print(f"[MT5] Fetching {symbol} {tf_name}  {date_from.date()} -> {date_to.date()} ...")

    rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)

    if rates is None or len(rates) == 0:
        print(f"[ERROR] No data returned for {symbol} {tf_name}: {mt5.last_error()}")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={
        "time":        "Datetime",
        "open":        "Open",
        "high":        "High",
        "low":         "Low",
        "close":       "Close",
        "tick_volume": "Volume",
    })
    df = df.set_index("Datetime")[["Open", "High", "Low", "Close", "Volume"]]
    df = df.sort_index()

    print(f"[MT5] {tf_name}: {len(df):,} candles  ({df.index[0]} -> {df.index[-1]})")
    return df


def export_fold1(skip_if_exists: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download Fold1 1m/5m from MT5 and save to CSV. Returns (df_1m, df_5m)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if skip_if_exists and CSV_1M.exists() and CSV_5M.exists():
        print("[export] CSVs already exist — loading from cache.")
        df_1m = pd.read_csv(CSV_1M, index_col="Datetime", parse_dates=True)
        df_5m = pd.read_csv(CSV_5M, index_col="Datetime", parse_dates=True)
        df_1m.index = pd.to_datetime(df_1m.index, utc=True)
        df_5m.index = pd.to_datetime(df_5m.index, utc=True)
        print(f"  1m: {len(df_1m):,} candles | 5m: {len(df_5m):,} candles")
        return df_1m, df_5m

    mt5 = _connect_mt5()
    try:
        df_1m = _fetch_range(mt5, SYMBOL, mt5.TIMEFRAME_M1, "1m", FOLD1_START, FOLD1_END)
        df_5m = _fetch_range(mt5, SYMBOL, mt5.TIMEFRAME_M5, "5m", FOLD1_START, FOLD1_END)
    finally:
        mt5.shutdown()
        print("[MT5] Connection closed.")

    if not df_1m.empty:
        df_1m.to_csv(CSV_1M)
        print(f"[export] Saved {CSV_1M.name} ({len(df_1m):,} rows)")
    else:
        print("[WARN] 1m export empty — CSV not saved.")

    if not df_5m.empty:
        df_5m.to_csv(CSV_5M)
        print(f"[export] Saved {CSV_5M.name} ({len(df_5m):,} rows)")
    else:
        print("[WARN] 5m export empty — CSV not saved.")

    return df_1m, df_5m


# ── Fractal metrics ───────────────────────────────────────────────────────────

def _run_metrics(df_1m: pd.DataFrame, df_5m: pd.DataFrame, label: str) -> dict:
    from research.fractal_lab.coherence import compute_coherence
    from research.fractal_lab.disorder import compute_disorder

    print(f"\n[metrics] {label}")
    print(f"  1m candles: {len(df_1m):,}  |  5m candles: {len(df_5m):,}")

    # Ensure column names match what coherence/disorder expect (Title case)
    for col in ("Open", "High", "Low", "Close"):
        if col not in df_1m.columns and col.lower() in df_1m.columns:
            df_1m = df_1m.rename(columns={c: c.title() for c in df_1m.columns})
        if col not in df_5m.columns and col.lower() in df_5m.columns:
            df_5m = df_5m.rename(columns={c: c.title() for c in df_5m.columns})

    coherence = compute_coherence(df_1m, df_5m)
    disorder  = compute_disorder(df_1m, df_5m)

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


def _load_holdout_from_cache() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load holdout 1m/5m from yfinance parquet cache."""
    from research.fractal_lab.data_loader import get_period_data
    return get_period_data("holdout")


# ── Comparison + report ───────────────────────────────────────────────────────

def _answer(fold1_dis, fold1_coh, holdout_dis, holdout_coh) -> str:
    if fold1_dis is None:
        return (
            "INCONCLUSIVO — Fold1 sem dados após exportação MT5. "
            "Verifique se MT5 estava aberto e se o símbolo EURUSD está disponível."
        )

    delta_dis = fold1_dis - holdout_dis if holdout_dis is not None else None
    delta_coh = fold1_coh - holdout_coh if holdout_coh is not None else None

    parts = [
        f"Fold1: disorder={fold1_dis:.1f}, coherence={fold1_coh:.1f}" if fold1_coh else f"Fold1: disorder={fold1_dis:.1f}",
    ]
    if holdout_dis is not None:
        parts.append(
            f"Holdout: disorder={holdout_dis:.1f}, coherence={holdout_coh:.1f}" if holdout_coh else f"Holdout: disorder={holdout_dis:.1f}"
        )

    if delta_dis is None:
        return "INCONCLUSIVO — holdout sem dados para comparação. " + " | ".join(parts)

    if delta_dis > 5:
        verdict = (
            f"SIM — disorder era MAIOR no Fold1 (+{delta_dis:.1f} vs Holdout). "
            f"Mercado mais caotico em set2024-fev2025: sinais de reversao do Sistema 1 falharam."
        )
    elif delta_dis < -5:
        verdict = (
            f"NÃO — disorder era MENOR no Fold1 ({delta_dis:+.1f} vs Holdout). "
            f"Desordem fractal não explica as perdas — outro fator dominante (ex: tendência EMA200)."
        )
    else:
        verdict = (
            f"NEUTRO — disorder Fold1 ≈ Holdout (delta={delta_dis:+.1f}, dentro de ±5). "
            f"Sem diferença estrutural de desordem entre os períodos."
        )

    coh_note = ""
    if delta_coh is not None:
        coh_note = (
            f" Coerência {'maior' if delta_coh > 0 else 'menor'} no Fold1 (delta={delta_coh:+.1f})."
        )

    return f"{verdict}{coh_note} | {' | '.join(parts)}"


def update_report(fold1_metrics: dict, holdout_metrics: dict) -> dict:
    """Load existing report, add Fold1 MT5 results, save updated JSON."""
    if REPORT_PATH.exists():
        with open(REPORT_PATH, encoding="utf-8") as f:
            report = json.load(f)
    else:
        report = {}

    f1_dis = fold1_metrics["disorder"].get("disorder_score")
    f1_coh = fold1_metrics["coherence"].get("coherence_score")
    ho_dis = holdout_metrics["disorder"].get("disorder_score")
    ho_coh = holdout_metrics["coherence"].get("coherence_score")

    report["updated_at"] = datetime.now(timezone.utc).isoformat()
    report["fold1_mt5"] = {
        "source": "MetaTrader5 export",
        "period": f"{FOLD1_START.date()} -> {FOLD1_END.date()}",
        "coherence": fold1_metrics["coherence"],
        "disorder":  fold1_metrics["disorder"],
    }
    report["comparison"] = {
        "fold1_disorder":  f1_dis,
        "fold1_coherence": f1_coh,
        "holdout_disorder":  ho_dis,
        "holdout_coherence": ho_coh,
        "disorder_delta_fold1_minus_holdout":  round(f1_dis - ho_dis, 2) if f1_dis and ho_dis else None,
        "coherence_delta_fold1_minus_holdout": round(f1_coh - ho_coh, 2) if f1_coh and ho_coh else None,
    }
    report["answer"] = _answer(f1_dis, f1_coh, ho_dis, ho_coh)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n[report] Updated {REPORT_PATH}")
    return report


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Export MT5 Fold1 data and run Fractal Lab comparison.")
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip MT5 download if CSVs already exist in data/fractal_cache/",
    )
    args = parser.parse_args()

    print("=" * 64)
    print("  Fractal Lab — Fold1 MT5 Export + Comparison")
    print(f"  Period : {FOLD1_START.date()} -> {FOLD1_END.date()}")
    print(f"  Symbol : {SYMBOL}")
    print("=" * 64)

    # Step 1: Export (or load) Fold1 data
    df_1m_fold1, df_5m_fold1 = export_fold1(skip_if_exists=args.skip_export)

    if df_1m_fold1.empty and df_5m_fold1.empty:
        print("\n[ERROR] Both 1m and 5m exports are empty. Aborting.")
        sys.exit(1)

    # Step 2: Compute metrics for Fold1
    fold1_metrics = _run_metrics(df_1m_fold1, df_5m_fold1, "Fold1 (set2024-fev2025) via MT5")

    # Step 3: Load holdout data (yfinance cache) and compute metrics
    print("\n[metrics] Loading Holdout data …")
    df_1m_ho, df_5m_ho = _load_holdout_from_cache()
    holdout_metrics = _run_metrics(df_1m_ho, df_5m_ho, "Holdout (dez2025-mai2026) via yfinance")

    # Step 4: Update report and print answer
    report = update_report(fold1_metrics, holdout_metrics)

    print(f"\n{'='*64}")
    print("PERGUNTA: disorder_score era maior antes dos trades ruins do Sistema 1?")
    print(f"RESPOSTA: {report['answer']}")
    print(f"{'='*64}")

    # Step 5: Distribution summary
    print("\n── Score distribution ──────────────────────────────────────")
    for period_label, metrics in [("Fold1", fold1_metrics), ("Holdout", holdout_metrics)]:
        dis = metrics["disorder"]
        coh = metrics["coherence"]
        if dis.get("available"):
            p = dis["percentiles"]
            print(f"  {period_label} disorder  p25={p['p25']:.1f}  p50={p['p50']:.1f}  p75={p['p75']:.1f}  mean={dis['disorder_score']:.1f}")
        if coh.get("available"):
            p = coh["percentiles"]
            print(f"  {period_label} coherence p25={p['p25']:.1f}  p50={p['p50']:.1f}  p75={p['p75']:.1f}  mean={coh['coherence_score']:.1f}")
    print()


if __name__ == "__main__":
    main()
