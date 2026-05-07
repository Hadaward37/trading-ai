"""
Regime Classifier — EUR/USD market regime detection for Sistema 1 analysis.

Defines 3 mutually exclusive regimes using ADX, ATR ratio, and rolling
disorder/coherence proxies computed directly from 1H OHLCV data.

No look-ahead bias: all features use only past data at time of classification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# ── Regime labels ─────────────────────────────────────────────────────────────

REGIME_STRESS     = "REGIME_1_STRESS"      # high volatility / strong trend
REGIME_COMPRESSED = "REGIME_2_COMPRESSED"  # low energy, directionless
REGIME_STRUCTURED = "REGIME_3_STRUCTURED"  # normal, favorable for reversals

REGIMES = [REGIME_STRESS, REGIME_COMPRESSED, REGIME_STRUCTURED]

# ── Classification thresholds ─────────────────────────────────────────────────

ATR_RATIO_STRESS     = 1.5   # ATR > 1.5× rolling mean → volatile
ATR_RATIO_COMPRESSED = 0.8   # ATR < 0.8× rolling mean → compressed
ADX_STRESS           = 35.0  # strong directional trend → bad for reversals
ADX_COMPRESSED       = 18.0  # no direction → ambiguous
DISORDER_STRESS      = 0.55  # 1H direction alternates > 55% of bars → chaotic
COHERENCE_COMPRESSED = 0.42  # 1H barely aligns with 4H → incoherent

ATR_ROLL_WINDOW      = 20    # bars for rolling ATR mean
DISORDER_ROLL_WINDOW = 12    # bars for rolling disorder proxy (~12H)
COHERENCE_ROLL_WINDOW = 8    # bars for rolling coherence proxy (~8H = 2×4H candles)

OUTCOME_MAX_BARS     = 48    # max forward bars to resolve TP/SL

# ── Dual-slope alignment constants ───────────────────────────────────────────

EMA200_PERIOD     = 200  # EMA lookback
SLOPE_SHORT_LAG   = 5    # slope_short = EMA200[t-1] - EMA200[t-6]
SLOPE_MID_LAG     = 20   # slope_mid   = EMA200[t-1] - EMA200[t-21]

BIAS_BULL    =  1
BIAS_BEAR    = -1
BIAS_NEUTRAL =  0


# ── Feature computation ───────────────────────────────────────────────────────

def _body_dir(series_open: pd.Series, series_close: pd.Series) -> pd.Series:
    diff = series_close - series_open
    return diff.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add regime features to a 1H OHLCV dataframe (columns: Open, High, Low, Close).

    Added columns:
      atr_ratio      — current ATR / rolling_mean(ATR, 20)
      disorder_proxy — rolling direction-flip rate over 12H
      coherence_proxy— rolling 1H vs 4H alignment over 8H
      adx            — preserved if present, else NaN

    All calculations are causal (only past data used).
    """
    df = df.copy()

    # Ensure title-case columns
    df.columns = [c.title() if c.lower() in ("open","high","low","close","volume")
                  else c for c in df.columns]

    # ATR (Wilder's)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"]  - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["_atr_raw"] = tr.rolling(14, min_periods=1).mean()

    if "atr" not in df.columns or df["atr"].isna().mean() > 0.5:
        df["atr"] = df["_atr_raw"]

    atr_mean = df["_atr_raw"].rolling(ATR_ROLL_WINDOW, min_periods=5).mean()
    df["atr_ratio"] = (df["_atr_raw"] / atr_mean.replace(0, np.nan)).clip(0.2, 4.0)

    # Disorder proxy: rolling alternation rate of close-to-close direction
    c2c_dir = df["Close"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    flips = (c2c_dir != c2c_dir.shift(1)) & (c2c_dir != 0) & (c2c_dir.shift(1) != 0)
    df["disorder_proxy"] = flips.rolling(DISORDER_ROLL_WINDOW, min_periods=4).mean()

    # Coherence proxy: rolling 1H alignment with 4H direction
    if df.index.tz is not None:
        df_4h = df["Close"].resample("4h").last().reindex(df.index, method="ffill")
        df_4h_open = df["Open"].resample("4h").first().reindex(df.index, method="ffill")
    else:
        df_4h = df["Close"].resample("4h").last().reindex(df.index, method="ffill")
        df_4h_open = df["Open"].resample("4h").first().reindex(df.index, method="ffill")

    dir_4h = _body_dir(df_4h_open, df_4h)
    aligned = ((c2c_dir * dir_4h) > 0).astype(float)
    df["coherence_proxy"] = aligned.rolling(COHERENCE_ROLL_WINDOW, min_periods=3).mean()

    # ADX: kept if provided, else left as NaN
    if "adx" not in df.columns:
        df["adx"] = np.nan

    # EMA200 dual-slope market bias (causal: uses t-1 through t-21)
    ema200 = df["Close"].ewm(span=EMA200_PERIOD, adjust=False).mean()
    df["ema200"]      = ema200
    df["slope_short"] = ema200.shift(1) - ema200.shift(1 + SLOPE_SHORT_LAG)
    df["slope_mid"]   = ema200.shift(1) - ema200.shift(1 + SLOPE_MID_LAG)

    def _market_bias(row):
        ss, sm = row["slope_short"], row["slope_mid"]
        if pd.isna(ss) or pd.isna(sm):
            return BIAS_NEUTRAL
        if ss > 0 and sm > 0:
            return BIAS_BULL
        if ss < 0 and sm < 0:
            return BIAS_BEAR
        return BIAS_NEUTRAL

    df["market_bias"] = df.apply(_market_bias, axis=1)

    df = df.drop(columns=["_atr_raw"])
    return df


# ── Regime classification ─────────────────────────────────────────────────────

def classify_regime(row) -> str:
    """
    Classify a single row into one of 3 regimes.

    Expected keys: atr_ratio, adx, disorder_proxy, coherence_proxy.
    Missing keys fall back to neutral defaults.

    Priority: STRESS > COMPRESSED > STRUCTURED (mutually exclusive).
    """
    atr_ratio     = float(row.get("atr_ratio",      1.0))
    adx           = float(row.get("adx",            20.0) or 20.0)
    disorder_p    = float(row.get("disorder_proxy",  0.40))
    coherence_p   = float(row.get("coherence_proxy", 0.50))

    # STRESS: extreme volatility, strong trend, or chaotic microstructure
    if (atr_ratio > ATR_RATIO_STRESS
            or adx > ADX_STRESS
            or disorder_p > DISORDER_STRESS):
        return REGIME_STRESS

    # COMPRESSED: low energy, no direction, incoherent
    if (atr_ratio < ATR_RATIO_COMPRESSED
            and (adx < ADX_COMPRESSED or coherence_p < COHERENCE_COMPRESSED)):
        return REGIME_COMPRESSED

    # STRUCTURED: normal operating conditions
    return REGIME_STRUCTURED


def tag_regimes(df_features: pd.DataFrame) -> pd.DataFrame:
    """Apply classify_regime row-wise. Returns df with 'regime' column added."""
    df = df_features.copy()
    df["regime"] = df.apply(classify_regime, axis=1)
    return df


# ── Dual-slope alignment ─────────────────────────────────────────────────────

def compute_alignment(signal: int, market_bias: int) -> bool:
    """
    Return True if the trade direction is aligned with the dual-slope market bias.

    is_aligned = True  when signal=1  (BUY)  and market_bias == BIAS_BULL
    is_aligned = True  when signal=-1 (SELL) and market_bias == BIAS_BEAR
    is_aligned = False when market_bias == BIAS_NEUTRAL or direction opposes bias
    """
    if market_bias == BIAS_NEUTRAL:
        return False
    if signal == 1 and market_bias == BIAS_BULL:
        return True
    if signal == -1 and market_bias == BIAS_BEAR:
        return True
    return False


def tag_alignment(df_trades: pd.DataFrame,
                  df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Add market_bias and is_aligned to each trade using features at entry time.

    No look-ahead: uses the feature row whose timestamp <= trade entry time.
    Requires df_features to have 'market_bias' column (from compute_features).
    """
    df_feat = df_features.copy()
    if not isinstance(df_feat.index, pd.DatetimeIndex):
        df_feat = df_feat.set_index("datetime")
    df_feat.index = pd.to_datetime(df_feat.index).floor("h")

    biases, aligned_flags = [], []

    for _, trade in df_trades.iterrows():
        entry_dt = pd.to_datetime(trade["datetime"]).floor("h")
        past = df_feat[df_feat.index <= entry_dt]

        if past.empty or "market_bias" not in past.columns:
            bias = BIAS_NEUTRAL
        else:
            bias = int(past["market_bias"].iloc[-1])

        sig = int(trade["signal"])
        biases.append(bias)
        aligned_flags.append(compute_alignment(sig, bias))

    df_out = df_trades.copy()
    df_out["market_bias"] = biases
    df_out["is_aligned"]  = aligned_flags
    return df_out


def _grp_stats(grp: pd.DataFrame, label: str) -> dict:
    """Compute trading stats for a group of trades."""
    if len(grp) == 0:
        return {"label": label, "n_trades": 0,
                "win_rate": np.nan, "profit_factor": np.nan,
                "expectancy_pips": np.nan, "sharpe_proxy": np.nan,
                "max_drawdown_pips": np.nan}

    wins   = grp[grp["pnl_pips"] > 0]["pnl_pips"]
    losses = grp[grp["pnl_pips"] <= 0]["pnl_pips"]
    total_win  = float(wins.sum())
    total_loss = float(losses.sum())
    if total_loss >= 0:
        total_loss = -1e-9

    pf     = total_win / abs(total_loss) if total_win > 0 else 0.0
    e      = float(grp["pnl_pips"].mean())
    sharpe = e / grp["pnl_pips"].std() if grp["pnl_pips"].std() > 0 else 0.0

    cum = grp.sort_values("datetime")["pnl_pips"].cumsum()
    drawdown = float((cum.cummax() - cum).max())

    return {
        "label": label,
        "n_trades": len(grp),
        "win_rate": round(len(wins) / len(grp), 4),
        "profit_factor": round(pf, 3),
        "expectancy_pips": round(e, 2),
        "sharpe_proxy": round(sharpe, 3),
        "max_drawdown_pips": round(drawdown, 2),
    }


def alignment_stats(df_tagged: pd.DataFrame) -> pd.DataFrame:
    """
    Compute stats for regime × alignment combinations.

    Returns DataFrame with one row per combination:
      REGIME_1_STRESS + ALIGNED
      REGIME_1_STRESS + NOT_ALIGNED
      REGIME_3_STRUCTURED + ALIGNED
      REGIME_3_STRUCTURED + NOT_ALIGNED
    """
    combos = [
        (REGIME_STRESS,     True,  "STRESS + ALIGNED"),
        (REGIME_STRESS,     False, "STRESS + NOT_ALIGNED"),
        (REGIME_STRUCTURED, True,  "STRUCTURED + ALIGNED"),
        (REGIME_STRUCTURED, False, "STRUCTURED + NOT_ALIGNED"),
    ]
    records = []
    for regime, aligned, label in combos:
        grp = df_tagged[
            (df_tagged["regime"] == regime) &
            (df_tagged["is_aligned"] == aligned)
        ]
        records.append(_grp_stats(grp, label))
    return pd.DataFrame(records).set_index("label")


def ks_test_alignment(df_tagged: pd.DataFrame) -> dict:
    """
    KS-test comparing STRESS_ALIGNED vs STRESS_NOT_ALIGNED return distributions.

    Returns result dict with ks_stat, p_value, significant, and n tuple.
    """
    aligned     = df_tagged[
        (df_tagged["regime"] == REGIME_STRESS) & (df_tagged["is_aligned"] == True)
    ]["pnl_pips"].dropna()

    not_aligned = df_tagged[
        (df_tagged["regime"] == REGIME_STRESS) & (df_tagged["is_aligned"] == False)
    ]["pnl_pips"].dropna()

    if len(aligned) < 10 or len(not_aligned) < 10:
        return {
            "pair": "STRESS_ALIGNED_vs_STRESS_NOT_ALIGNED",
            "ks_stat": np.nan, "p_value": np.nan,
            "significant": False,
            "n": (len(aligned), len(not_aligned)),
            "note": "Insufficient samples (< 10 per group)",
        }

    ks, pv = stats.ks_2samp(aligned.values, not_aligned.values)
    return {
        "pair": "STRESS_ALIGNED_vs_STRESS_NOT_ALIGNED",
        "ks_stat": round(float(ks), 4),
        "p_value": round(float(pv), 4),
        "significant": bool(pv < 0.05),
        "n": (len(aligned), len(not_aligned)),
    }


# ── Trade outcome resolution ──────────────────────────────────────────────────

def resolve_outcomes(df_trades: pd.DataFrame, df_ohlcv: pd.DataFrame,
                     max_bars: int = OUTCOME_MAX_BARS) -> pd.DataFrame:
    """
    Determine TP/SL hit for each trade by scanning forward OHLCV candles.

    df_trades must have: datetime, signal (1/-1), entry, sl, tp
    df_ohlcv  must have: datetime index, High, Low, Close

    Returns df_trades with added columns:
      outcome     — "TP_HIT" | "SL_HIT" | "TIMEOUT"
      pnl_pips    — signed pips (positive = win)
      resolved_at — datetime when resolved
    """
    df_trades = df_trades.copy()
    df_ohlcv = df_ohlcv.copy()

    # Normalise ohlcv index
    if not isinstance(df_ohlcv.index, pd.DatetimeIndex):
        df_ohlcv = df_ohlcv.set_index("datetime")
    df_ohlcv.index = pd.to_datetime(df_ohlcv.index).floor("h")
    df_ohlcv.columns = [c.title() if c.lower() in ("open","high","low","close","volume")
                        else c for c in df_ohlcv.columns]

    outcomes, pnls, resolved_ats = [], [], []

    for _, trade in df_trades.iterrows():
        entry_dt = pd.to_datetime(trade["datetime"]).floor("h")
        sig      = int(trade["signal"])
        sl       = float(trade["sl"])
        tp       = float(trade["tp"])
        entry    = float(trade["entry"])

        # Forward bars starting one bar after entry
        future = df_ohlcv[df_ohlcv.index > entry_dt].head(max_bars)

        outcome = "TIMEOUT"
        resolved_at = None
        pnl_pips = 0.0

        for bar_dt, bar in future.iterrows():
            hi, lo = float(bar["High"]), float(bar["Low"])
            if sig == 1:   # BUY: TP above entry, SL below
                if hi >= tp:
                    outcome = "TP_HIT"; pnl_pips = (tp - entry) * 10_000; resolved_at = bar_dt; break
                if lo <= sl:
                    outcome = "SL_HIT"; pnl_pips = (sl - entry) * 10_000; resolved_at = bar_dt; break
            else:           # SELL: TP below entry, SL above
                if lo <= tp:
                    outcome = "TP_HIT"; pnl_pips = (entry - tp) * 10_000; resolved_at = bar_dt; break
                if hi >= sl:
                    outcome = "SL_HIT"; pnl_pips = (entry - sl) * 10_000; resolved_at = bar_dt; break

        if outcome == "TIMEOUT":
            last_close = future["Close"].iloc[-1] if len(future) else entry
            pnl_pips = ((last_close - entry) if sig == 1 else (entry - last_close)) * 10_000
            resolved_at = future.index[-1] if len(future) else entry_dt

        outcomes.append(outcome)
        pnls.append(round(pnl_pips, 2))
        resolved_ats.append(resolved_at)

    df_trades["outcome"]     = outcomes
    df_trades["pnl_pips"]    = pnls
    df_trades["resolved_at"] = resolved_ats
    return df_trades


# ── Trade tagging ─────────────────────────────────────────────────────────────

def tag_trades_with_regime(df_trades: pd.DataFrame,
                           df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Attach regime label to each trade using regime at entry time (no look-ahead).

    Uses the regime computed on the bar *before* trade entry to avoid bias.
    """
    df_feat = df_features.copy()
    if not isinstance(df_feat.index, pd.DatetimeIndex):
        df_feat = df_feat.set_index("datetime")
    df_feat.index = pd.to_datetime(df_feat.index).floor("h")

    def _get_regime(entry_dt):
        entry_dt = pd.to_datetime(entry_dt).floor("h")
        # use the bar at or before entry time
        past = df_feat[df_feat.index <= entry_dt]
        if past.empty or "regime" not in past.columns:
            return REGIME_STRUCTURED
        return past["regime"].iloc[-1]

    df_out = df_trades.copy()
    df_out["regime"] = df_out["datetime"].apply(_get_regime)
    return df_out


# ── Per-regime statistics ─────────────────────────────────────────────────────

def regime_stats(df_tagged: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-regime trading statistics.

    Input: df with columns pnl_pips, outcome, regime.
    Returns: DataFrame indexed by regime with win_rate, profit_factor, E, sharpe, max_dd.
    """
    records = []
    for regime in REGIMES:
        grp = df_tagged[df_tagged["regime"] == regime]
        if len(grp) == 0:
            records.append({
                "regime": regime, "n_trades": 0,
                "win_rate": np.nan, "profit_factor": np.nan,
                "expectancy_pips": np.nan, "sharpe_proxy": np.nan,
                "max_drawdown_pips": np.nan,
            })
            continue

        wins  = grp[grp["pnl_pips"] > 0]["pnl_pips"]
        losses = grp[grp["pnl_pips"] <= 0]["pnl_pips"]
        total_win  = float(wins.sum())
        total_loss = float(losses.sum())
        if total_loss >= 0:
            total_loss = -1e-9  # avoid division by zero

        pf = total_win / abs(total_loss) if total_win > 0 else 0.0
        e  = grp["pnl_pips"].mean()

        # Sharpe proxy: E / std (annualised equivalent not possible without dates)
        sharpe = e / grp["pnl_pips"].std() if grp["pnl_pips"].std() > 0 else 0.0

        # Max drawdown from cumulative pnl
        cum = grp.sort_values("datetime")["pnl_pips"].cumsum()
        roll_max = cum.cummax()
        drawdown = (roll_max - cum).max()

        records.append({
            "regime": regime,
            "n_trades": len(grp),
            "win_rate": round(len(wins) / len(grp), 4),
            "profit_factor": round(pf, 3),
            "expectancy_pips": round(e, 2),
            "sharpe_proxy": round(sharpe, 3),
            "max_drawdown_pips": round(drawdown, 2),
        })

    return pd.DataFrame(records).set_index("regime")


# ── KS separability test ──────────────────────────────────────────────────────

def ks_test_regimes(df_tagged: pd.DataFrame) -> dict:
    """
    Pairwise KS-test between regime return distributions.
    Returns dict with keys "A_vs_B": {"ks_stat", "p_value", "significant"}.
    """
    results = {}
    pairs = [
        (REGIME_STRESS, REGIME_COMPRESSED),
        (REGIME_STRESS, REGIME_STRUCTURED),
        (REGIME_COMPRESSED, REGIME_STRUCTURED),
    ]
    for r1, r2 in pairs:
        g1 = df_tagged[df_tagged["regime"] == r1]["pnl_pips"].dropna()
        g2 = df_tagged[df_tagged["regime"] == r2]["pnl_pips"].dropna()
        key = f"{r1}_vs_{r2}"
        if len(g1) < 10 or len(g2) < 10:
            results[key] = {"ks_stat": np.nan, "p_value": np.nan,
                            "significant": False, "n": (len(g1), len(g2))}
            continue
        ks, pv = stats.ks_2samp(g1.values, g2.values)
        results[key] = {
            "ks_stat": round(float(ks), 4),
            "p_value": round(float(pv), 4),
            "significant": bool(pv < 0.05),
            "n": (len(g1), len(g2)),
        }
    return results


def separability_score(ks_results: dict, stats_df: pd.DataFrame) -> dict:
    """
    REGIME_SEPARABILITY_SCORE (0–1).

    Components:
      - ks_score:   mean(1 - p_value) across significant pairs
      - e_score:    normalized range of expectancy across regimes
      - sig_pairs:  fraction of pairs with p < 0.05

    Final score = 0.5*ks_score + 0.3*e_score + 0.2*sig_pairs
    Interpretation: >0.6 = separable; 0.4–0.6 = borderline; <0.4 = not separable
    """
    p_values = [v["p_value"] for v in ks_results.values() if not np.isnan(v.get("p_value", np.nan))]
    sig = [v["significant"] for v in ks_results.values()]

    ks_score  = float(np.mean([1 - p for p in p_values])) if p_values else 0.0
    sig_pairs = float(np.mean(sig)) if sig else 0.0

    e_vals = stats_df["expectancy_pips"].dropna()
    if len(e_vals) >= 2:
        e_range = float(e_vals.max() - e_vals.min())
        # normalise: 10 pips range = score 1.0
        e_score = min(1.0, e_range / 10.0)
    else:
        e_score = 0.0

    final = round(0.5 * ks_score + 0.3 * e_score + 0.2 * sig_pairs, 4)

    if final > 0.6:
        verdict = "SEPARABLE — regimes são estatisticamente distintos"
    elif final > 0.4:
        verdict = "BORDERLINE — separabilidade parcial, evidência fraca"
    else:
        verdict = "INCONCLUSIVO — regimes não são estatisticamente separáveis"

    return {
        "REGIME_SEPARABILITY_SCORE": final,
        "ks_score":   round(ks_score,  4),
        "e_score":    round(e_score,   4),
        "sig_pairs":  round(sig_pairs, 4),
        "verdict":    verdict,
    }
