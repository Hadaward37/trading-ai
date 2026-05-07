import numpy as np
import pandas as pd

WINDOW_MINUTES = 15


def _momentum_direction(series_close: pd.Series) -> pd.Series:
    """Close-to-close momentum direction. Used for 1m (yfinance EURUSD=X Open≈Close)."""
    diff = series_close.diff()
    return diff.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))


def _body_direction(series_open: pd.Series, series_close: pd.Series) -> pd.Series:
    """Open-to-close body direction. Works for 5m+ candles."""
    diff = series_close - series_open
    return diff.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    return pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)


def _local_atr(df: pd.DataFrame, periods: int = 5) -> pd.Series:
    return _true_range(df).rolling(periods, min_periods=1).mean()


def _ensure_utc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def compute_disorder(df_1m: pd.DataFrame, df_5m: pd.DataFrame) -> dict:
    """
    Disorder score (0–100): how chaotic 1m price action is.

    Per 15-minute window:
      - alternation: direction flip rate in 1m candles (0=stable, 1=every candle flips)
      - atr_ratio:   1m ATR / 5m ATR, normalized; high ratio = micro-chaos vs macro
      - inconsistency: % of 1m candles opposing dominant 5m direction

    Final score = mean(0.40*alternation + 0.25*atr_ratio + 0.35*inconsistency) * 100
    """
    if df_1m.empty:
        return {"disorder_score": None, "n_windows": 0, "window_scores": [], "available": False}

    df_1m = _ensure_utc(df_1m)
    df_1m["_dir"] = _momentum_direction(df_1m["Close"])
    df_1m["_atr"] = _local_atr(df_1m)

    has_5m = not df_5m.empty
    if has_5m:
        df_5m = _ensure_utc(df_5m)
        df_5m["_dir"] = _body_direction(df_5m["Open"], df_5m["Close"])
        df_5m["_body"] = (df_5m["Close"] - df_5m["Open"]).abs()
        df_5m["_atr"] = _local_atr(df_5m)

    t_start = df_1m.index.min()
    t_end = df_1m.index.max()
    if has_5m:
        t_start = max(t_start, df_5m.index.min())
        t_end = min(t_end, df_5m.index.max())

    windows = pd.date_range(
        start=t_start.floor(f"{WINDOW_MINUTES}min"),
        end=t_end.ceil(f"{WINDOW_MINUTES}min"),
        freq=f"{WINDOW_MINUTES}min",
        tz="UTC",
    )
    window_scores = []

    for w in windows[:-1]:
        w_end = w + pd.Timedelta(minutes=WINDOW_MINUTES)
        m1 = df_1m[(df_1m.index >= w) & (df_1m.index < w_end)]

        if len(m1) < 5:
            continue

        # Component 1: alternation rate
        dirs = m1["_dir"].values
        flips = sum(
            1 for i in range(1, len(dirs))
            if dirs[i] != dirs[i - 1] and dirs[i] != 0 and dirs[i - 1] != 0
        )
        alternation = flips / (len(dirs) - 1) if len(dirs) > 1 else 0.0

        # Component 2: ATR ratio (1m micro-volatility vs 5m macro)
        atr_ratio_score = 0.5  # neutral default when 5m unavailable
        if has_5m:
            m5 = df_5m[(df_5m.index >= w) & (df_5m.index < w_end)]
            if len(m5) >= 1:
                atr_1m_val = m1["_atr"].mean()
                atr_5m_val = m5["_atr"].mean()
                if atr_5m_val > 0:
                    ratio = atr_1m_val / atr_5m_val
                    # ratio ~1 = neutral; ratio 2+ = high micro-disorder
                    atr_ratio_score = min(1.0, max(0.0, (ratio - 0.5) / 1.5))

        # Component 3: inconsistency with 5m trend
        inconsistency = 0.5  # neutral default
        if has_5m:
            m5 = df_5m[(df_5m.index >= w) & (df_5m.index < w_end)]
            if len(m5) >= 1:
                total_body = m5["_body"].sum()
                if total_body > 0:
                    w_dir = (m5["_dir"] * m5["_body"]).sum() / total_body
                    dom = 1 if w_dir > 0.05 else (-1 if w_dir < -0.05 else 0)
                    if dom != 0:
                        non_doji = m1[m1["_dir"] != 0]
                        if len(non_doji) > 0:
                            inconsistency = (non_doji["_dir"] != dom).mean()

        score = 0.40 * alternation + 0.25 * atr_ratio_score + 0.35 * inconsistency
        window_scores.append(score)

    if not window_scores:
        return {"disorder_score": None, "n_windows": 0, "window_scores": [], "available": False}

    arr = np.array(window_scores)
    return {
        "disorder_score": round(float(arr.mean() * 100), 2),
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
