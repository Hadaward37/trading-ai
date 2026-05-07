import numpy as np
import pandas as pd

WINDOW_MINUTES = 15


def _body_direction(series_open: pd.Series, series_close: pd.Series) -> pd.Series:
    """Open-to-close body direction. Works well for 5m+ candles."""
    diff = series_close - series_open
    return diff.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))


def _momentum_direction(series_close: pd.Series) -> pd.Series:
    """Close-to-close momentum. Used for 1m where Open≈Close in yfinance EURUSD=X."""
    diff = series_close.diff()
    return diff.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))


def _ensure_utc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def compute_coherence(df_1m: pd.DataFrame, df_5m: pd.DataFrame) -> dict:
    """
    Coherence score (0–100): how aligned 1m candles are with the 5m direction.

    Per 15-minute window:
      - dominant_5m_dir: body-weighted vote of 5m candles in window
      - alignment: % of non-doji 1m candles (close-to-close) agreeing with dominant_5m_dir
      - consistency: 1 - normalized std of 1m momentum directions
      - window_score = 0.6 * alignment + 0.4 * consistency

    Final score = mean(window_scores) * 100

    Note: 1m direction uses close-to-close momentum because yfinance EURUSD=X
    1m data returns Open≈Close for most candles.
    """
    if df_1m.empty or df_5m.empty:
        return {"coherence_score": None, "n_windows": 0, "window_scores": [], "available": False}

    df_1m = _ensure_utc(df_1m)
    df_5m = _ensure_utc(df_5m)

    df_1m["_dir"] = _momentum_direction(df_1m["Close"])
    df_5m["_dir"] = _body_direction(df_5m["Open"], df_5m["Close"])
    df_5m["_body"] = (df_5m["Close"] - df_5m["Open"]).abs()

    t_start = max(df_1m.index.min(), df_5m.index.min()).floor(f"{WINDOW_MINUTES}min")
    t_end = min(df_1m.index.max(), df_5m.index.max()).ceil(f"{WINDOW_MINUTES}min")

    windows = pd.date_range(start=t_start, end=t_end, freq=f"{WINDOW_MINUTES}min", tz="UTC")
    window_scores = []

    for w in windows[:-1]:
        w_end = w + pd.Timedelta(minutes=WINDOW_MINUTES)
        m1 = df_1m[(df_1m.index >= w) & (df_1m.index < w_end)]
        m5 = df_5m[(df_5m.index >= w) & (df_5m.index < w_end)]

        if len(m1) < 5 or len(m5) < 1:
            continue

        total_body = m5["_body"].sum()
        if total_body == 0:
            continue
        weighted_dir = (m5["_dir"] * m5["_body"]).sum() / total_body
        dominant_dir = 1 if weighted_dir > 0.05 else (-1 if weighted_dir < -0.05 else 0)
        if dominant_dir == 0:
            continue

        non_doji = m1[m1["_dir"] != 0]
        if len(non_doji) == 0:
            continue

        alignment = (non_doji["_dir"] == dominant_dir).mean()

        # std of {-1, 0, 1} series; max theoretical std ≈ 1 for alternating ±1
        dir_std = m1["_dir"].std()
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
