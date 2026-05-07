"""
Fractal Lab v2 - 1m real data via MT5 Clear.

Broker retention window: ~60 days for 1m. Fold1 (set2024-fev2025) is beyond
that limit, so only the Holdout period can be analyzed at 1m resolution.

What this script does:
  1. Exports Holdout 1m + 5m from MT5 Clear  (Mar-May 2026, ~60 days)
  2. Runs v2 metrics: 1m vs 5m, 15min windows  (original plan)
  3. Compares v2 Holdout against v1 Holdout (5m vs 15m) to answer:
     "does more granular data show more disorder?"
  4. Loads Fold1 v1 (5m vs 15m) from report for the full picture
  5. Updates data/fractal_report.json

For --fold1-attempt: tries to export Fold1 1m anyway - documents what MT5 returns.

Usage:
    .\\venv\\Scripts\\python scripts\\export_mt5_fractal.py          # normal run
    .\\venv\\Scripts\\python scripts\\export_mt5_fractal.py --skip-export  # reuse CSVs
    .\\venv\\Scripts\\python scripts\\export_mt5_fractal.py --fold1-attempt
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd

SYMBOL     = "EURUSD"
CACHE_DIR  = _ROOT / "data" / "fractal_cache"
REPORT_PATH = _ROOT / "data" / "fractal_report.json"

# CSVs
CSV_5M_FOLD1      = CACHE_DIR / "EURUSD_5m_fold1.csv"
CSV_1M_HOLDOUT    = CACHE_DIR / "EURUSD_1m_holdout.csv"
CSV_5M_HOLDOUT    = CACHE_DIR / "EURUSD_5m_holdout.csv"
CSV_1M_FOLD1      = CACHE_DIR / "EURUSD_1m_fold1.csv"  # attempted, likely empty

# MT5 limits: 1m data available for ~60 days
MT5_1M_LIMIT_DAYS = 60
MT5_1M_CHUNK_DAYS = 30   # request in chunks to stay well under the ~65k candle limit

FOLD1_START  = datetime(2024, 9,  1,  0,  0, tzinfo=timezone.utc)
FOLD1_END    = datetime(2025, 2, 28, 23, 59, tzinfo=timezone.utc)


# ── MT5 export helpers ────────────────────────────────────────────────────────

def _connect_mt5():
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("[ERROR] MetaTrader5 not installed: pip install MetaTrader5")
        sys.exit(1)
    import config
    timeout = getattr(config, "MT5_TIMEOUT_MS", 60_000)
    if not mt5.initialize(timeout=timeout):
        print(f"[ERROR] MT5 init failed: {mt5.last_error()}")
        print("        Make sure MT5 is open and logged in.")
        sys.exit(1)
    info = mt5.terminal_info()
    print(f"[MT5] Connected  build={info.build}")
    return mt5


def _fetch_chunked(mt5, symbol: str, timeframe, tf_label: str,
                   date_from: datetime, date_to: datetime,
                   chunk_days: int = MT5_1M_CHUNK_DAYS) -> pd.DataFrame:
    """Fetch in daily/weekly chunks to avoid MT5 candle-count limits."""
    parts = []
    cursor = date_from
    while cursor < date_to:
        end = min(cursor + timedelta(days=chunk_days), date_to)
        r = mt5.copy_rates_range(symbol, timeframe, cursor, end)
        if r is not None and len(r) > 1:
            parts.append(pd.DataFrame(r))
        cursor = end

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["time"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"time": "Datetime", "open": "Open", "high": "High",
                             "low": "Low", "close": "Close", "tick_volume": "Volume"})
    df = df.set_index("Datetime")[["Open", "High", "Low", "Close", "Volume"]]
    df = df.sort_index()

    print(f"[MT5] {tf_label}: {len(df):,} candles  "
          f"({df.index[0].date()} -> {df.index[-1].date()})")
    return df


def export_holdout(skip_if_exists: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Export Holdout 1m + 5m from MT5. Returns (df_1m, df_5m)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if skip_if_exists and CSV_1M_HOLDOUT.exists() and CSV_5M_HOLDOUT.exists():
        print("[export] Holdout CSVs found - loading from cache.")
        df1 = _load_csv(CSV_1M_HOLDOUT)
        df5 = _load_csv(CSV_5M_HOLDOUT)
        return df1, df5

    today    = datetime.now(timezone.utc).replace(hour=23, minute=59, second=0)
    date_from = today - timedelta(days=MT5_1M_LIMIT_DAYS)

    mt5 = _connect_mt5()
    try:
        print(f"\n[export] Holdout 1m  ({date_from.date()} -> {today.date()}) ...")
        df_1m = _fetch_chunked(mt5, SYMBOL, mt5.TIMEFRAME_M1, "1m",
                               date_from, today, chunk_days=MT5_1M_CHUNK_DAYS)

        print(f"\n[export] Holdout 5m  ({date_from.date()} -> {today.date()}) ...")
        df_5m = _fetch_chunked(mt5, SYMBOL, mt5.TIMEFRAME_M5, "5m",
                               date_from, today, chunk_days=MT5_1M_CHUNK_DAYS)
    finally:
        mt5.shutdown()
        print("[MT5] Connection closed.")

    if not df_1m.empty:
        df_1m.to_csv(CSV_1M_HOLDOUT)
        print(f"[export] Saved {CSV_1M_HOLDOUT.name}")
    if not df_5m.empty:
        df_5m.to_csv(CSV_5M_HOLDOUT)
        print(f"[export] Saved {CSV_5M_HOLDOUT.name}")

    return df_1m, df_5m


def attempt_fold1_1m() -> pd.DataFrame:
    """Try to export Fold1 1m from MT5. Documents what the broker actually returns."""
    print(f"\n[fold1-attempt] Requesting 1m  {FOLD1_START.date()} -> {FOLD1_END.date()} ...")
    mt5 = _connect_mt5()
    try:
        # single request - expect empty or sentinel
        r = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M1, FOLD1_START, FOLD1_END)
        if r is None or len(r) == 0:
            print(f"[fold1-attempt] Broker returned nothing: {mt5.last_error()}")
            return pd.DataFrame()

        df = pd.DataFrame(r)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        in_range = df[(df["time"] >= FOLD1_START) & (df["time"] <= FOLD1_END)]
        sentinel  = df[~((df["time"] >= FOLD1_START) & (df["time"] <= FOLD1_END))]
        print(f"[fold1-attempt] Rows returned: {len(df):,}")
        print(f"[fold1-attempt]   In Fold1 range: {len(in_range):,}")
        print(f"[fold1-attempt]   Sentinel/OOB:   {len(sentinel):,}")
        if not df.empty:
            print(f"[fold1-attempt]   Actual dates: {df.time.min().date()} -> {df.time.max().date()}")

        if len(in_range) > 100:
            df_out = in_range.rename(columns={"time": "Datetime", "open": "Open", "high": "High",
                                               "low": "Low", "close": "Close", "tick_volume": "Volume"})
            df_out = df_out.set_index("Datetime")[["Open", "High", "Low", "Close", "Volume"]]
            df_out.to_csv(CSV_1M_FOLD1)
            print(f"[fold1-attempt] Saved {len(df_out):,} real candles to {CSV_1M_FOLD1.name}")
            return df_out

        print("[fold1-attempt] RESULT: Fold1 1m unavailable at Clear broker.")
        print("                Retention limit: ~60 days. Fold1 is 15+ months ago.")
        return pd.DataFrame()
    finally:
        mt5.shutdown()


# ── CSV loader ────────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="Datetime", parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "Close"])
    return df.sort_index()


# ── Fractal metrics ───────────────────────────────────────────────────────────

def run_metrics(df_micro: pd.DataFrame, df_macro: pd.DataFrame,
                label: str, window_minutes: int) -> dict:
    from research.fractal_lab.coherence import compute_coherence
    from research.fractal_lab.disorder import compute_disorder

    if df_micro.empty:
        print(f"  [SKIP] {label}: micro data empty")
        return {"coherence": {"available": False}, "disorder": {"available": False}}

    coh = compute_coherence(df_micro, df_macro, window_minutes=window_minutes)
    dis = compute_disorder(df_micro, df_macro, window_minutes=window_minutes)

    coh_str = f"{coh['coherence_score']:.1f} (n={coh['n_windows']})" if coh["available"] else "N/A"
    dis_str = f"{dis['disorder_score']:.1f} (n={dis['n_windows']})" if dis["available"] else "N/A"
    print(f"  coherence : {coh_str}")
    print(f"  disorder  : {dis_str}")

    return {
        "coherence": {k: v for k, v in coh.items() if k != "window_scores"},
        "disorder":  {k: v for k, v in dis.items()  if k != "window_scores"},
    }


# ── Comparison table ──────────────────────────────────────────────────────────

def print_comparison(v1_fold1: dict, v1_holdout: dict,
                     v2_holdout: dict, v2_fold1: dict | None = None) -> None:
    def row(label, m, note=""):
        d = m["disorder"]
        c = m["coherence"]
        dis = f"{d['disorder_score']:5.1f}" if d.get("available") else "  N/A"
        coh = f"{c['coherence_score']:5.1f}" if c.get("available") else "  N/A"
        n_d = d.get("n_windows", 0)
        n_c = c.get("n_windows", 0)
        print(f"  {label:<26} disorder={dis}  coherence={coh}  ({n_d}/{n_c} windows)  {note}")

    print("\n-- Comparison: v1 (5m/15m 60min) vs v2 (1m/5m 15min) --")
    print(f"  {'Label':<26} {'disorder':>12}  {'coherence':>12}  windows")
    print("  " + "-" * 70)
    row("Fold1   v1 [5m/15m  60min]", v1_fold1,   "set-fev 2024-25 (MT5 5m)")
    if v2_fold1:
        row("Fold1   v2 [1m/5m   15min]", v2_fold1, "limited MT5 1m")
    else:
        print(f"  {'Fold1   v2 [1m/5m   15min]':<26} {'UNAVAILABLE':>12}  "
              "  Clear retains ~60d of 1m; Fold1 is 15mo ago")
    print("  " + "-" * 70)
    row("Holdout v1 [5m/15m  60min]", v1_holdout, "mar-mai 2026 (yfinance 5m)")
    row("Holdout v2 [1m/5m   15min]", v2_holdout, "mar-mai 2026 (MT5 1m)")

    # Answer the key question
    d_v1 = v1_holdout["disorder"].get("disorder_score")
    d_v2 = v2_holdout["disorder"].get("disorder_score")
    c_v1 = v1_holdout["coherence"].get("coherence_score")
    c_v2 = v2_holdout["coherence"].get("coherence_score")

    print()
    if d_v1 and d_v2:
        delta_d = d_v2 - d_v1
        delta_c = (c_v2 - c_v1) if c_v1 and c_v2 else None
        direction = "AUMENTOU" if delta_d > 2 else ("DIMINUIU" if delta_d < -2 else "MANTEVE-SE SIMILAR")
        print(f"  Granularidade 1m vs 5m (Holdout): disorder {direction} ({delta_d:+.1f})")
        if delta_c:
            print(f"  Coherence: {'maior' if delta_c > 0 else 'menor'} com 1m ({delta_c:+.1f})")
        print()
        if delta_d > 2:
            print("  INTERPRETACAO: 1m captura mais ruido/microestrutura que o 5m.")
            print("  Se Fold1 1m tivesse sido medido, disorder provavelmente seria ainda")
            print(f"  mais alto que {v1_fold1['disorder'].get('disorder_score', '?'):.1f} (v1 5m Fold1).")
        elif abs(delta_d) <= 2:
            print("  INTERPRETACAO: Granularidade nao altera o sinal significativamente.")
            print("  disorder_score e robusto entre 1m e 5m para este par/periodo.")
        else:
            print("  INTERPRETACAO: 1m mostra MENOS disorder que 5m (incomum).")
            print("  Possivelmente porque 5m suaviza ruido mas 1m reverte para sinal limpo.")


# ── Report update ─────────────────────────────────────────────────────────────

def save_report(v1_fold1: dict, v1_holdout: dict, v2_holdout: dict,
                v2_fold1: dict | None) -> None:
    report = {}
    if REPORT_PATH.exists():
        with open(REPORT_PATH, encoding="utf-8") as f:
            report = json.load(f)

    d_v1 = v1_holdout["disorder"].get("disorder_score")
    d_v2 = v2_holdout["disorder"].get("disorder_score")
    c_v1 = v1_holdout["coherence"].get("coherence_score")
    c_v2 = v2_holdout["coherence"].get("coherence_score")

    report["updated_at"] = datetime.now(timezone.utc).isoformat()

    # Persist v1 baselines so future runs can reload them
    if v1_fold1["disorder"].get("available"):
        report["fold1_5m"] = {"method": "5m_vs_15m window=60min",
                               "coherence": v1_fold1["coherence"],
                               "disorder":  v1_fold1["disorder"]}
    if v1_holdout["disorder"].get("available"):
        report["holdout_5m"] = {"method": "5m_vs_15m window=60min",
                                 "coherence": v1_holdout["coherence"],
                                 "disorder":  v1_holdout["disorder"]}

    report["v2_1m_5m"] = {
        "method": "1m_vs_5m_mt5  window=15min",
        "broker_1m_retention": "~60 days (Clear)",
        "fold1_1m_available": False,
        "fold1_1m_note": "Fold1 (set2024-fev2025) is 15+ months ago; Clear retains only ~60d of 1m data",
        "holdout_v2": {
            "coherence": v2_holdout["coherence"],
            "disorder":  v2_holdout["disorder"],
        },
        "fold1_v2": v2_fold1 if v2_fold1 else None,
    }

    if d_v1 and d_v2:
        delta = d_v2 - d_v1
        direction = "INCREASED" if delta > 2 else ("DECREASED" if delta < -2 else "SIMILAR")
        report["v2_1m_5m"]["granularity_comparison"] = {
            "holdout_v1_disorder": d_v1,
            "holdout_v2_disorder": d_v2,
            "delta": round(delta, 2),
            "direction": direction,
            "answer": (
                f"Com 1m (mais granular) o disorder {direction.lower()} "
                f"(delta={delta:+.1f}) vs 5m. "
                f"Fold1 1m indisponivel na Clear (limite ~60d)."
            ),
        }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[report] Saved to {REPORT_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-export", action="store_true",
                        help="Reuse existing holdout CSVs without connecting to MT5")
    parser.add_argument("--fold1-attempt", action="store_true",
                        help="Also attempt Fold1 1m export (will document broker limit)")
    args = parser.parse_args()

    print("=" * 64)
    print("  Fractal Lab v2 - 1m real data via MT5 Clear")
    print("  v2: 1m vs 5m, 15min windows")
    print("  v1: 5m vs 15m, 60min windows  (reference)")
    print("=" * 64)

    # --- Export Holdout 1m + 5m from MT5 ---
    df_1m_ho, df_5m_ho = export_holdout(skip_if_exists=args.skip_export)

    # --- Optionally attempt Fold1 1m ---
    v2_fold1 = None
    if args.fold1_attempt:
        df_1m_fold1_attempt = attempt_fold1_1m()
        if not df_1m_fold1_attempt.empty:
            df_5m_fold1 = _load_csv(CSV_5M_FOLD1)
            print("\n[v2-fold1] Computing 1m vs 5m metrics for Fold1 ...")
            v2_fold1 = run_metrics(df_1m_fold1_attempt, df_5m_fold1,
                                   "Fold1 1m vs 5m", window_minutes=15)

    # --- Compute v1: 5m vs 15m for both periods (reference baseline) ---
    print("\n[v1-fold1] 5m vs 15m, 60min windows (reference) ...")
    if CSV_5M_FOLD1.exists():
        df_5m_fold1 = _load_csv(CSV_5M_FOLD1)
        df_15m_fold1 = df_5m_fold1.resample("15min").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        ).dropna(subset=["Open", "Close"])
        print(f"  5m: {len(df_5m_fold1):,}  15m: {len(df_15m_fold1):,}")
        v1_fold1 = run_metrics(df_5m_fold1, df_15m_fold1, "Fold1 5m/15m", window_minutes=60)
    else:
        print(f"  [SKIP] {CSV_5M_FOLD1.name} not found")
        v1_fold1 = {"coherence": {"available": False}, "disorder": {"available": False}}

    print("\n[v1-holdout] 5m vs 15m, 60min windows (reference) ...")
    if CSV_5M_HOLDOUT.exists():
        df_5m_ho_v1 = _load_csv(CSV_5M_HOLDOUT)
        df_15m_ho_v1 = df_5m_ho_v1.resample("15min").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        ).dropna(subset=["Open", "Close"])
        print(f"  5m: {len(df_5m_ho_v1):,}  15m: {len(df_15m_ho_v1):,}")
        v1_holdout = run_metrics(df_5m_ho_v1, df_15m_ho_v1, "Holdout 5m/15m", window_minutes=60)
    else:
        v1_holdout = {"coherence": {"available": False}, "disorder": {"available": False}}

    # --- Compute v2: 1m vs 5m for Holdout ---
    print(f"\n[v2-holdout] 1m: {len(df_1m_ho):,} candles  |  5m: {len(df_5m_ho):,} candles")
    print("[v2-holdout] Computing coherence + disorder (1m vs 5m, 15min windows) ...")
    v2_holdout = run_metrics(df_1m_ho, df_5m_ho, "Holdout 1m vs 5m", window_minutes=15)

    # --- Print comparison ---
    print_comparison(v1_fold1, v1_holdout, v2_holdout, v2_fold1)

    # --- Save report ---
    save_report(v1_fold1, v1_holdout, v2_holdout, v2_fold1)


if __name__ == "__main__":
    main()
