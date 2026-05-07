"""
Pairs Trading Research — ITUB4 vs BBDC4  (1H, Jan2024–Mai2026)

FASE 1: Correlação (Pearson, Spearman, rolling 20d)
FASE 2: Spread normalizado (OLS beta + z-score janela 60)
FASE 3: Simulação de sinais (entrada ±2σ, saída 0, stop ±3.5)
FASE 4: Estatísticas (half-life, win rate, PF, MaxDD, cointegração ADF+Johansen)

Sem lookahead. Sem otimização. Sem tocar no Sistema 1.

Usage:
    .\\venv\\Scripts\\python research\\fractal_lab\\pairs_trading.py
    .\\venv\\Scripts\\python research\\fractal_lab\\pairs_trading.py --skip-export
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from scipy import stats

# ── Constants ─────────────────────────────────────────────────────────────────

SYMBOL_A = "ITUB4"
SYMBOL_B = "BBDC4"

# MT5 symbols for Brazilian equities on Clear broker
MT5_SYMBOL_A = "ITUB4"
MT5_SYMBOL_B = "BBDC4"

PERIOD_START = datetime(2024, 1,  1,  0, 0, tzinfo=timezone.utc)
PERIOD_END   = datetime(2026, 5,  7, 23, 59, tzinfo=timezone.utc)

CACHE_DIR   = _ROOT / "data" / "fractal_cache"
REPORT_PATH = _ROOT / "data" / "pairs_trading_report.json"

CSV_A = CACHE_DIR / "ITUB4_1h_2024-2026.csv"
CSV_B = CACHE_DIR / "BBDC4_1h_2024-2026.csv"

# FASE 2 params
OLS_WINDOW        = 60    # rolling OLS window (bars) for beta + alpha estimation
ZSCORE_WINDOW     = 60    # z-score lookback window (bars)

# FASE 3 signal thresholds
ENTRY_ZSCORE      = 2.0
EXIT_ZSCORE       = 0.0
STOP_ZSCORE       = 3.5

# Rolling correlation window (20 trading days × ~7 hourly bars/day ≈ 140 bars)
CORR_ROLL_WINDOW  = 140   # bars (~20 trading days on 1H)

CHUNK_DAYS        = 90    # MT5 fetch chunk size


# ── MT5 data export ───────────────────────────────────────────────────────────

def _connect_mt5():
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("[ERROR] MetaTrader5 not installed: pip install MetaTrader5")
        sys.exit(1)
    if not mt5.initialize():
        print(f"[ERROR] MT5 init failed: {mt5.last_error()}")
        print("        Abra o MT5 e faça login antes de rodar.")
        sys.exit(1)
    info = mt5.terminal_info()
    print(f"[MT5] Conectado  build={info.build}")
    return mt5


def _fetch_1h(mt5, symbol: str, date_from: datetime, date_to: datetime) -> pd.DataFrame:
    """Fetch 1H bars in chunks to avoid MT5 candle-count limits."""
    parts = []
    cursor = date_from
    while cursor < date_to:
        end = min(cursor + timedelta(days=CHUNK_DAYS), date_to)
        r = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, cursor, end)
        if r is not None and len(r) > 1:
            parts.append(pd.DataFrame(r))
        cursor = end

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["time"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={
        "time": "Datetime", "open": "Open", "high": "High",
        "low": "Low", "close": "Close", "tick_volume": "Volume",
    })
    df = df.set_index("Datetime")[["Open", "High", "Low", "Close", "Volume"]].sort_index()
    print(f"  [{symbol}] {len(df):,} barras  {df.index[0].date()} -> {df.index[-1].date()}")
    return df


def export_data(skip_if_exists: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Export ITUB4 + BBDC4 1H data from MT5. Returns (df_a, df_b)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if skip_if_exists and CSV_A.exists() and CSV_B.exists():
        print("[export] CSVs encontrados — carregando do cache.")
        return _load_csv(CSV_A), _load_csv(CSV_B)

    mt5 = _connect_mt5()
    try:
        print(f"\n[export] Buscando {SYMBOL_A} 1H ...")
        df_a = _fetch_1h(mt5, MT5_SYMBOL_A, PERIOD_START, PERIOD_END)
        print(f"\n[export] Buscando {SYMBOL_B} 1H ...")
        df_b = _fetch_1h(mt5, MT5_SYMBOL_B, PERIOD_START, PERIOD_END)
    finally:
        mt5.shutdown()
        print("[MT5] Conexão encerrada.")

    if not df_a.empty:
        df_a.to_csv(CSV_A)
        print(f"[export] Salvo: {CSV_A.name}")
    if not df_b.empty:
        df_b.to_csv(CSV_B)
        print(f"[export] Salvo: {CSV_B.name}")

    return df_a, df_b


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="Datetime", parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"]).sort_index()


# ── Data alignment ─────────────────────────────────────────────────────────────

def align_closes(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """Inner join on Close prices, drop any NaN rows."""
    ca = df_a["Close"].rename(SYMBOL_A)
    cb = df_b["Close"].rename(SYMBOL_B)
    df = pd.concat([ca, cb], axis=1).dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None) if df.index.tz is not None else df.index
    print(f"[align] {len(df):,} barras alinhadas  "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    return df


# ── FASE 1: Correlação ────────────────────────────────────────────────────────

def fase1_correlacao(df: pd.DataFrame) -> dict:
    """Pearson, Spearman, and rolling correlation (CORR_ROLL_WINDOW bars)."""
    a = df[SYMBOL_A]
    b = df[SYMBOL_B]

    pearson_r,  pearson_p  = stats.pearsonr(a, b)
    spearman_r, spearman_p = stats.spearmanr(a, b)

    rolling_corr = a.rolling(CORR_ROLL_WINDOW).corr(b)

    corr_std  = float(rolling_corr.dropna().std())
    corr_min  = float(rolling_corr.dropna().min())
    corr_max  = float(rolling_corr.dropna().max())
    corr_mean = float(rolling_corr.dropna().mean())

    stability = "ESTAVEL" if corr_std < 0.10 else ("MODERADA" if corr_std < 0.20 else "INSTAVEL")

    print(f"\n[FASE 1] Correlação {SYMBOL_A} × {SYMBOL_B}")
    print(f"  Pearson  r={pearson_r:.4f}  p={pearson_p:.4e}")
    print(f"  Spearman r={spearman_r:.4f}  p={spearman_p:.4e}")
    print(f"  Rolling ({CORR_ROLL_WINDOW}b): mean={corr_mean:.4f}  "
          f"std={corr_std:.4f}  min={corr_min:.4f}  max={corr_max:.4f}")
    print(f"  Estabilidade: {stability}")

    return {
        "pearson":  {"r": round(pearson_r, 4),  "p": round(float(pearson_p), 6)},
        "spearman": {"r": round(spearman_r, 4), "p": round(float(spearman_p), 6)},
        "rolling": {
            "window_bars": CORR_ROLL_WINDOW,
            "mean": round(corr_mean, 4),
            "std":  round(corr_std, 4),
            "min":  round(corr_min, 4),
            "max":  round(corr_max, 4),
            "stability": stability,
        },
        "rolling_series": rolling_corr,  # kept in memory, not serialized
    }


# ── FASE 2: Spread normalizado ────────────────────────────────────────────────

def _rolling_ols(a: pd.Series, b: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    """Rolling OLS: a = beta * b + alpha. Returns (beta, alpha) series."""
    betas  = pd.Series(np.nan, index=a.index)
    alphas = pd.Series(np.nan, index=a.index)

    a_arr = a.values
    b_arr = b.values

    for i in range(window - 1, len(a_arr)):
        y = a_arr[i - window + 1 : i + 1]
        x = b_arr[i - window + 1 : i + 1]
        x_mean = x.mean()
        y_mean = y.mean()
        denom  = ((x - x_mean) ** 2).sum()
        if denom < 1e-12:
            continue
        beta  = ((x - x_mean) * (y - y_mean)).sum() / denom
        alpha = y_mean - beta * x_mean
        betas.iloc[i]  = beta
        alphas.iloc[i] = alpha

    return betas, alphas


def fase2_spread(df: pd.DataFrame) -> dict:
    """Rolling OLS spread + z-score."""
    a = df[SYMBOL_A]
    b = df[SYMBOL_B]

    print(f"\n[FASE 2] Spread OLS (janela={OLS_WINDOW}b) + Z-score (janela={ZSCORE_WINDOW}b)")

    beta, alpha = _rolling_ols(a, b, OLS_WINDOW)

    spread  = a - (beta * b + alpha)
    spread_mean = spread.rolling(ZSCORE_WINDOW).mean()
    spread_std  = spread.rolling(ZSCORE_WINDOW).std()
    zscore = (spread - spread_mean) / spread_std.replace(0, np.nan)

    valid = beta.dropna()
    print(f"  Beta médio: {valid.mean():.4f}  std: {valid.std():.4f}  "
          f"min: {valid.min():.4f}  max: {valid.max():.4f}")
    print(f"  Z-score disponível a partir da barra {OLS_WINDOW + ZSCORE_WINDOW - 2}")
    print(f"  |Z| > {ENTRY_ZSCORE}σ: {(zscore.abs() > ENTRY_ZSCORE).sum()} barras "
          f"({(zscore.abs() > ENTRY_ZSCORE).mean() * 100:.1f}% do tempo)")

    df_out = df.copy()
    df_out["beta"]   = beta
    df_out["alpha"]  = alpha
    df_out["spread"] = spread
    df_out["zscore"] = zscore

    return {
        "ols_window":    OLS_WINDOW,
        "zscore_window": ZSCORE_WINDOW,
        "beta_mean":  round(float(valid.mean()), 4),
        "beta_std":   round(float(valid.std()),  4),
        "beta_min":   round(float(valid.min()),  4),
        "beta_max":   round(float(valid.max()),  4),
        "pct_beyond_entry": round(float((zscore.abs() > ENTRY_ZSCORE).mean()), 4),
        "df_spread": df_out,  # enriched DataFrame passed to downstream phases
    }


# ── FASE 3: Simulação de sinais ───────────────────────────────────────────────

def fase3_sinais(df_spread: pd.DataFrame) -> pd.DataFrame:
    """
    Entry: |zscore| > ENTRY_ZSCORE (confirmed on bar close — no lookahead).
    Exit:  zscore crosses 0 (convergence).
    Stop:  |zscore| > STOP_ZSCORE (extreme divergence).

    Direction:
      zscore > +ENTRY → short A / long  B  (signal = -1)
      zscore < -ENTRY → long  A / short B  (signal = +1)

    Returns DataFrame of trades with columns:
      entry_idx, exit_idx, signal, entry_z, exit_z, exit_reason, bars_held, pnl_z
    """
    zs = df_spread["zscore"].values
    n  = len(zs)

    trades = []
    in_trade    = False
    entry_idx   = None
    entry_z     = None
    signal      = None

    for i in range(1, n):
        z = zs[i]
        if np.isnan(z):
            continue

        if not in_trade:
            # Entry: on bar close (i), not peeking ahead
            if z > ENTRY_ZSCORE:
                in_trade  = True
                entry_idx = i
                entry_z   = z
                signal    = -1  # short A / long B
            elif z < -ENTRY_ZSCORE:
                in_trade  = True
                entry_idx = i
                entry_z   = z
                signal    = +1  # long A / short B
        else:
            prev_z = zs[i - 1]
            # Stop: |z| beyond STOP_ZSCORE
            if abs(z) > STOP_ZSCORE:
                trades.append(_make_trade(entry_idx, i, signal, entry_z, z, "STOP", df_spread))
                in_trade = False
                continue

            # Exit: zscore crosses 0
            crossed = (signal == -1 and prev_z > EXIT_ZSCORE and z <= EXIT_ZSCORE) or \
                      (signal == +1 and prev_z < -EXIT_ZSCORE and z >= EXIT_ZSCORE)
            if crossed:
                trades.append(_make_trade(entry_idx, i, signal, entry_z, z, "CONVERGENCE", df_spread))
                in_trade = False

    # Close open trade at end-of-data
    if in_trade:
        trades.append(_make_trade(entry_idx, n - 1, signal, entry_z, zs[n - 1], "EOD", df_spread))

    df_trades = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["entry_time", "exit_time", "signal", "entry_z", "exit_z",
                 "exit_reason", "bars_held", "pnl_z"]
    )

    if not df_trades.empty:
        print(f"\n[FASE 3] {len(df_trades)} trades simulados")
        for reason in ("CONVERGENCE", "STOP", "EOD"):
            n_r = (df_trades["exit_reason"] == reason).sum()
            print(f"  {reason}: {n_r} ({n_r/len(df_trades)*100:.1f}%)")

    return df_trades


def _make_trade(ei: int, xi: int, sig: int, ez: float, xz: float,
                reason: str, df: pd.DataFrame) -> dict:
    idx = df.index
    return {
        "entry_time":  idx[ei],
        "exit_time":   idx[xi],
        "signal":      sig,
        "entry_z":     round(float(ez), 4),
        "exit_z":      round(float(xz), 4) if not np.isnan(xz) else None,
        "exit_reason": reason,
        "bars_held":   xi - ei,
        "pnl_z":       round(float(ez - xz) * sig, 4) if not np.isnan(xz) else 0.0,
    }


# ── FASE 4: Estatísticas ──────────────────────────────────────────────────────

def _half_life(zscore: pd.Series) -> float:
    """
    Ornstein-Uhlenbeck half-life via OLS regression on lag-1 differences.
    ΔZ_t = λ·Z_{t-1} + ε  →  half-life = -ln(2) / λ
    """
    z = zscore.dropna()
    dz   = z.diff().dropna()
    z_lag = z.shift(1).dropna()
    # align
    common = dz.index.intersection(z_lag.index)
    dz, z_lag = dz[common], z_lag[common]

    if len(dz) < 10:
        return np.nan

    slope, _, _, _, _ = stats.linregress(z_lag.values, dz.values)
    if slope >= 0:
        return np.nan  # no mean reversion

    hl = -np.log(2) / slope
    return round(float(hl), 2)


def _adf_test(series: pd.Series) -> dict:
    """Augmented Dickey-Fuller test (H0: unit root → not stationary)."""
    from statsmodels.tsa.stattools import adfuller
    s = series.dropna()
    if len(s) < 20:
        return {"adf_stat": None, "p_value": None, "stationary": False, "note": "insufficient data"}
    result = adfuller(s, autolag="AIC")
    return {
        "adf_stat":  round(float(result[0]), 4),
        "p_value":   round(float(result[1]), 6),
        "stationary": bool(result[1] < 0.05),
        "critical_1pct": round(float(result[4]["1%"]), 4),
        "critical_5pct": round(float(result[4]["5%"]), 4),
    }


def _johansen_test(df: pd.DataFrame) -> dict:
    """Johansen cointegration test (trace statistic, r=0 vs r=1)."""
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    data = df[[SYMBOL_A, SYMBOL_B]].dropna()
    if len(data) < 30:
        return {"cointegrated": False, "note": "insufficient data"}

    res = coint_johansen(data, det_order=0, k_ar_diff=1)
    # trace statistics at r=0 and r<=1
    trace_stat_r0  = round(float(res.lr1[0]), 4)
    trace_stat_r1  = round(float(res.lr1[1]), 4)
    cv_r0_5pct     = round(float(res.cvt[0, 1]), 4)  # 95% critical value at r=0
    cv_r1_5pct     = round(float(res.cvt[1, 1]), 4)  # 95% critical value at r<=1

    # At least 1 cointegrating relation if trace_stat_r0 > cv_r0_5pct
    cointegrated = bool(trace_stat_r0 > cv_r0_5pct)

    return {
        "cointegrated":   cointegrated,
        "trace_r0":       trace_stat_r0,
        "trace_r1":       trace_stat_r1,
        "cv_r0_5pct":     cv_r0_5pct,
        "cv_r1_5pct":     cv_r1_5pct,
        "eigenvalues":    [round(float(e), 6) for e in res.eig],
    }


def _engle_granger_coint(df: pd.DataFrame) -> dict:
    """Engle-Granger cointegration test via statsmodels."""
    from statsmodels.tsa.stattools import coint
    a = df[SYMBOL_A].dropna()
    b = df[SYMBOL_B].dropna()
    idx = a.index.intersection(b.index)
    a, b = a[idx], b[idx]
    if len(a) < 20:
        return {"cointegrated": False, "note": "insufficient data"}
    t_stat, p_value, crit = coint(a, b)
    return {
        "cointegrated": bool(p_value < 0.05),
        "t_stat":       round(float(t_stat), 4),
        "p_value":      round(float(p_value), 6),
        "critical_1pct": round(float(crit[0]), 4),
        "critical_5pct": round(float(crit[1]), 4),
    }


def _regime_performance(df_trades: pd.DataFrame) -> dict:
    """Break down trade performance by exit reason as a proxy for regime quality."""
    if df_trades.empty:
        return {}
    result = {}
    for reason in ("CONVERGENCE", "STOP", "EOD"):
        grp = df_trades[df_trades["exit_reason"] == reason]["pnl_z"]
        if len(grp) == 0:
            continue
        result[reason] = {
            "n":         len(grp),
            "win_rate":  round(float((grp > 0).mean()), 4),
            "mean_pnl":  round(float(grp.mean()), 4),
        }
    return result


def fase4_estatisticas(df_spread: pd.DataFrame,
                       df_trades: pd.DataFrame) -> dict:
    """Compute all FASE 4 statistics."""
    print("\n[FASE 4] Estatísticas")

    # ── Cointegração ──────────────────────────────────────────────────────────
    print("  Teste ADF no spread ...")
    adf_spread = _adf_test(df_spread["spread"])
    print(f"    ADF p={adf_spread.get('p_value')}  stationary={adf_spread.get('stationary')}")

    print("  Teste ADF em cada série (devem ser I(1)) ...")
    adf_a = _adf_test(df_spread[SYMBOL_A])
    adf_b = _adf_test(df_spread[SYMBOL_B])

    print("  Engle-Granger cointegração ...")
    eg = _engle_granger_coint(df_spread)
    print(f"    EG p={eg.get('p_value')}  cointegrado={eg.get('cointegrated')}")

    print("  Johansen cointegração ...")
    joh = _johansen_test(df_spread)
    print(f"    Johansen trace_r0={joh.get('trace_r0')} cv_5pct={joh.get('cv_r0_5pct')} "
          f"cointegrado={joh.get('cointegrated')}")

    # ── Half-life ─────────────────────────────────────────────────────────────
    hl = _half_life(df_spread["zscore"])
    print(f"  Half-life OU: {hl} barras (1H cada)")

    # ── Performance dos trades ────────────────────────────────────────────────
    if df_trades.empty:
        print("  Nenhum trade — sem estatísticas de performance.")
        return {
            "adf_spread": adf_spread, "adf_itub4": adf_a, "adf_bbdc4": adf_b,
            "engle_granger": eg, "johansen": joh, "half_life_bars": hl,
            "n_trades": 0,
        }

    pnl = df_trades["pnl_z"]
    wins   = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    total_win  = float(wins.sum())
    total_loss = abs(float(losses.sum())) or 1e-9

    win_rate = round(float((pnl > 0).mean()), 4)
    pf       = round(total_win / total_loss, 3)

    # Max drawdown on cumulative pnl_z
    cum = pnl.cumsum()
    max_dd = round(float((cum.cummax() - cum).max()), 4)

    avg_bars_to_converge = round(float(
        df_trades[df_trades["exit_reason"] == "CONVERGENCE"]["bars_held"].mean()
    ), 1) if (df_trades["exit_reason"] == "CONVERGENCE").any() else None

    print(f"  N trades: {len(df_trades)}  Win rate: {win_rate:.1%}  "
          f"PF: {pf:.2f}  MaxDD(z): {max_dd:.4f}")
    print(f"  Bars médios até convergência: {avg_bars_to_converge}")

    regime_perf = _regime_performance(df_trades)

    return {
        "adf_spread":       adf_spread,
        "adf_itub4":        adf_a,
        "adf_bbdc4":        adf_b,
        "engle_granger":    eg,
        "johansen":         joh,
        "half_life_bars":   hl,
        "n_trades":         len(df_trades),
        "win_rate":         win_rate,
        "profit_factor":    pf,
        "max_drawdown_z":   max_dd,
        "avg_bars_convergence": avg_bars_to_converge,
        "mean_pnl_z":       round(float(pnl.mean()), 4),
        "regime_breakdown": regime_perf,
    }


# ── Responder perguntas mandatórias ───────────────────────────────────────────

def answer_questions(f1: dict, f4: dict) -> dict:
    answers = {}

    # Q1: Cointegração
    eg_coint  = f4["engle_granger"].get("cointegrated", False)
    joh_coint = f4["johansen"].get("cointegrated", False)
    eg_p      = f4["engle_granger"].get("p_value")
    adf_spread_stat = f4["adf_spread"].get("stationary", False)

    if eg_coint and joh_coint and adf_spread_stat:
        q1 = (f"SIM — cointegrados por 3 critérios: "
              f"Engle-Granger p={eg_p}, "
              f"Johansen trace_r0={f4['johansen'].get('trace_r0')} > cv={f4['johansen'].get('cv_r0_5pct')}, "
              f"ADF spread p={f4['adf_spread'].get('p_value')}.")
    elif eg_coint or joh_coint or adf_spread_stat:
        n_pos = sum([eg_coint, joh_coint, adf_spread_stat])
        q1 = (f"PARCIAL — {n_pos}/3 critérios positivos: "
              f"EG={'SIM' if eg_coint else 'NAO'}, "
              f"Johansen={'SIM' if joh_coint else 'NAO'}, "
              f"ADF spread={'SIM' if adf_spread_stat else 'NAO'}. "
              f"Relação instável ou de curto prazo.")
    else:
        q1 = (f"NAO — nenhum critério de cointegração confirmado: "
              f"EG p={eg_p}, "
              f"Johansen trace_r0={f4['johansen'].get('trace_r0')}, "
              f"ADF spread p={f4['adf_spread'].get('p_value')}.")
    answers["ITUB4 e BBDC4 sao cointegrados?"] = q1

    # Q2: Half-life
    hl = f4.get("half_life_bars")
    if hl and not np.isnan(hl):
        hl_days = round(hl / 7, 1)  # ~7 hourly bars per trading day
        if hl < 1:
            interpretation = "instantanea — sinal fraco (possivel ruido)"
        elif hl < 48:
            interpretation = f"rapida (~{hl_days:.1f}d) — operacional para day/swing trade"
        elif hl < 120:
            interpretation = f"moderada (~{hl_days:.1f}d) — posicoes de ate 2-3 semanas"
        else:
            interpretation = f"lenta (~{hl_days:.1f}d) — risco de capital preso por muito tempo"
        answers["Qual o half-life medio de reversao?"] = (
            f"{hl} barras de 1H (~{hl_days:.1f} dias uteis). Interpretacao: {interpretation}."
        )
    else:
        answers["Qual o half-life medio de reversao?"] = (
            "INCONCLUSIVO — serie sem reversao mensuravel (slope >= 0 no modelo OU)."
        )

    # Q3: Edge estatístico (PF > 1.2)
    pf = f4.get("profit_factor")
    nt = f4.get("n_trades", 0)
    wr = f4.get("win_rate")
    if pf is not None and nt > 0:
        edge = "SIM" if pf > 1.2 else "NAO"
        robustness = "n suficiente (>= 30)" if nt >= 30 else f"n={nt} — interpretar com cautela"
        answers["A estrategia tem edge estatistico (PF > 1.2)?"] = (
            f"{edge} — PF={pf:.2f}  WR={wr:.1%}  n={nt}  ({robustness})."
        )
    else:
        answers["A estrategia tem edge estatistico (PF > 1.2)?"] = (
            "INCONCLUSIVO — sem trades gerados no periodo analisado."
        )

    # Q4: Regimes
    rb = f4.get("regime_breakdown", {})
    conv_wr = rb.get("CONVERGENCE", {}).get("win_rate")
    stop_wr = rb.get("STOP", {}).get("win_rate")
    pearson = f1["pearson"]["r"]
    rolling_stable = f1["rolling"]["stability"]

    if conv_wr is not None:
        q4 = (
            f"Pairs trading funciona melhor quando: "
            f"(1) correlacao estavel e alta (atual Pearson={pearson:.3f}, rolling={rolling_stable}); "
            f"(2) convergencias sao frequentes (WR em CONVERGENCE={conv_wr:.1%}); "
            f"(3) stops sao raros (WR em STOP={stop_wr:.1%} se houver). "
            f"Regimes de tendencia forte (crise, ciclo de juros) tendem a divergir o spread "
            f"sem retornar — sinal de perigo quando rolling_corr cai abaixo de 0.85."
        )
    else:
        q4 = (
            f"Correlacao geral: Pearson={pearson:.3f}, rolling={rolling_stable}. "
            f"Sem trades suficientes para análise de regime."
        )
    answers["Em quais regimes o pairs trading funciona melhor?"] = q4

    return answers


# ── Relatório final ───────────────────────────────────────────────────────────

def print_report(f1: dict, f2: dict, f4: dict, answers: dict) -> None:
    sep = "=" * 66

    print(f"\n{sep}")
    print("  PAIRS TRADING RESEARCH — ITUB4 / BBDC4 (1H Jan2024-Mai2026)")
    print(sep)

    print("\n--- FASE 1: CORRELACAO ---")
    print(f"  Pearson  r={f1['pearson']['r']:.4f}  p={f1['pearson']['p']:.4e}")
    print(f"  Spearman r={f1['spearman']['r']:.4f}  p={f1['spearman']['p']:.4e}")
    rl = f1["rolling"]
    print(f"  Rolling ({rl['window_bars']}b): "
          f"mean={rl['mean']:.4f}  std={rl['std']:.4f}  "
          f"[{rl['min']:.4f}, {rl['max']:.4f}]  → {rl['stability']}")

    print("\n--- FASE 2: SPREAD ---")
    print(f"  Beta OLS medio: {f2['beta_mean']:.4f}  std: {f2['beta_std']:.4f}")
    print(f"  % tempo com |z| > {ENTRY_ZSCORE}: {f2['pct_beyond_entry']:.1%}")

    print("\n--- FASE 4: COINTEGRAÇÃO ---")
    adf = f4["adf_spread"]
    print(f"  ADF spread: stat={adf.get('adf_stat')}  p={adf.get('p_value')}  "
          f"estacionario={adf.get('stationary')}")
    eg = f4["engle_granger"]
    print(f"  Engle-Granger: p={eg.get('p_value')}  cointegrado={eg.get('cointegrated')}")
    joh = f4["johansen"]
    print(f"  Johansen: trace_r0={joh.get('trace_r0')}  cv_5pct={joh.get('cv_r0_5pct')}  "
          f"cointegrado={joh.get('cointegrated')}")
    print(f"  Half-life OU: {f4.get('half_life_bars')} barras 1H")

    print("\n--- FASE 4: PERFORMANCE ---")
    print(f"  N trades: {f4.get('n_trades', 0)}")
    if f4.get("n_trades", 0) > 0:
        print(f"  Win Rate:       {f4.get('win_rate', 0):.1%}")
        print(f"  Profit Factor:  {f4.get('profit_factor', 0):.2f}")
        print(f"  MaxDD (z):      {f4.get('max_drawdown_z', 0):.4f}")
        print(f"  Bars/convergencia: {f4.get('avg_bars_convergence')}")
        print(f"  Mean PnL (z):   {f4.get('mean_pnl_z', 0):.4f}")

    print("\n--- RESPOSTAS ---")
    for q, a in answers.items():
        print(f"\n  Q: {q}")
        print(f"  A: {a}")

    print(f"\n{sep}\n")


def save_report(f1: dict, f2: dict, f4: dict, answers: dict) -> None:
    # Strip non-serializable objects (DataFrames, Series)
    f1_clean = {k: v for k, v in f1.items() if not isinstance(v, (pd.Series, pd.DataFrame))}
    f2_clean = {k: v for k, v in f2.items() if not isinstance(v, (pd.Series, pd.DataFrame))}

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "period":        f"{PERIOD_START.date()} / {PERIOD_END.date()}",
        "symbols":       [SYMBOL_A, SYMBOL_B],
        "timeframe":     "1H",
        "params": {
            "ols_window":       OLS_WINDOW,
            "zscore_window":    ZSCORE_WINDOW,
            "entry_zscore":     ENTRY_ZSCORE,
            "exit_zscore":      EXIT_ZSCORE,
            "stop_zscore":      STOP_ZSCORE,
            "corr_roll_window": CORR_ROLL_WINDOW,
        },
        "fase1_correlacao": f1_clean,
        "fase2_spread":     f2_clean,
        "fase4_stats":      f4,
        "answers":          answers,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
    print(f"[report] Salvo em {REPORT_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Pairs Trading ITUB4/BBDC4")
    parser.add_argument("--skip-export", action="store_true",
                        help="Reutilizar CSVs existentes sem conectar ao MT5")
    args = parser.parse_args()

    print("=" * 66)
    print("  Pairs Trading Research — ITUB4 / BBDC4")
    print(f"  Período: {PERIOD_START.date()} → {PERIOD_END.date()}  |  Timeframe: 1H")
    print("=" * 66)

    # ── Dados ─────────────────────────────────────────────────────────────────
    df_a, df_b = export_data(skip_if_exists=args.skip_export)

    if df_a.empty or df_b.empty:
        print("[ERRO] Dados insuficientes. Verifique MT5 e disponibilidade dos símbolos.")
        sys.exit(1)

    df = align_closes(df_a, df_b)

    if len(df) < OLS_WINDOW + ZSCORE_WINDOW + 10:
        print(f"[ERRO] Apenas {len(df)} barras alinhadas — insuficiente para a análise.")
        sys.exit(1)

    # ── Fases ─────────────────────────────────────────────────────────────────
    f1 = fase1_correlacao(df)

    f2 = fase2_spread(df)
    df_spread = f2["df_spread"]

    df_trades = fase3_sinais(df_spread)

    f4 = fase4_estatisticas(df_spread, df_trades)

    # ── Respostas + relatório ─────────────────────────────────────────────────
    answers = answer_questions(f1, f4)
    print_report(f1, f2, f4, answers)
    save_report(f1, f2, f4, answers)


if __name__ == "__main__":
    main()
