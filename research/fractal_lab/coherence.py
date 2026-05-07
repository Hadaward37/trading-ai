import numpy as np
import pandas as pd

WINDOW_MINUTES = 15


def _body_direction(series_open: pd.Series, series_close: pd.Series) -> pd.Series:
    """Open-to-close body direction. Works well for 5m+ candles with real OHLC."""
    diff = series_close - series_open
    return diff.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))


def _momentum_direction(series_close: pd.Series) -> pd.Series:
    """Close-to-close momentum. Fallback when Open≈Close (e.g. yfinance 1m EURUSD=X)."""
    diff = series_close.diff()
    return diff.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))


def _auto_direction(df: pd.DataFrame) -> pd.Series:
    """Pick body or momentum direction based on doji ratio."""
    doji_ratio = ((df["Close"] - df["Open"]).abs() == 0).mean()
    if doji_ratio > 0.5:
        return _momentum_direction(df["Close"])
    return _body_direction(df["Open"], df["Close"])


def _ensure_utc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def compute_coherence(
    df_micro: pd.DataFrame,
    df_macro: pd.DataFrame,
    window_minutes: int = WINDOW_MINUTES,
) -> dict:
    """
    Coherence score (0-100): how aligned micro candles are with the macro direction.

    Per window:
      - dominant_macro_dir: body-weighted vote of macro candles in window
      - alignment: % of non-doji micro candles agreeing with dominant direction
      - consistency: 1 - normalized std of micro directions
      - window_score = 0.6 * alignment + 0.4 * consistency

    Final score = mean(window_scores) * 100

    Supports any pair of timeframes (1m/5m, 5m/15m, etc.).
    Direction method is auto-selected per dataframe based on doji ratio.
    """
    if df_micro.empty or df_macro.empty:
        return {"coherence_score": None, "n_windows": 0, "window_scores": [], "available": False}

    df_micro = _ensure_utc(df_micro)
    df_macro = _ensure_utc(df_macro)

    df_micro["_dir"] = _auto_direction(df_micro)
    df_macro["_dir"] = _body_direction(df_macro["Open"], df_macro["Close"])
    df_macro["_body"] = (df_macro["Close"] - df_macro["Open"]).abs()

    min_micro = max(3, window_minutes // 20)

    t_start = max(df_micro.index.min(), df_macro.index.min()).floor(f"{window_minutes}min")
    t_end = min(df_micro.index.max(), df_macro.index.max()).ceil(f"{window_minutes}min")

    windows = pd.date_range(start=t_start, end=t_end, freq=f"{window_minutes}min", tz="UTC")
    window_scores = []

    for w in windows[:-1]:
        w_end = w + pd.Timedelta(minutes=window_minutes)
        micro = df_micro[(df_micro.index >= w) & (df_micro.index < w_end)]
        macro = df_macro[(df_macro.index >= w) & (df_macro.index < w_end)]

        if len(micro) < min_micro or len(macro) < 1:
            continue

        total_body = macro["_body"].sum()
        if total_body == 0:
            continue
        weighted_dir = (macro["_dir"] * macro["_body"]).sum() / total_body
        dominant_dir = 1 if weighted_dir > 0.05 else (-1 if weighted_dir < -0.05 else 0)
        if dominant_dir == 0:
            continue

        non_doji = micro[micro["_dir"] != 0]
        if len(non_doji) == 0:
            continue

        alignment = (non_doji["_dir"] == dominant_dir).mean()
        dir_std = micro["_dir"].std()
        consistency = max(0.0, 1.0 - dir_std)

        window_scores.append(0.6 * alignment + 0.4 * consistency)

    if not window_scores:
        return {"coherence_score": None, "n_windows": 0, "window_scores": [], "available": False}

    arr = np.array(window_scores)
    return {
        "coherence_score": round(float(arr.mean() * 100), 2),
        "n_windows": len(arr),
        "window_scores": [round(float(s), 4) for s in arr],
        "percentiles": {
            "p10": round(float(np.percentile(arr, 10) * 100), 2),
            "p25": round(float(np.percentile(arr, 25) * 100), 2),
            "p50": round(float(np.percentile(arr, 50) * 100), 2),
            "p75": round(float(np.percentile(arr, 75) * 100), 2),
            "p90": round(float(np.percentile(arr, 90) * 100), 2),
        },
        "available": True,
    }
