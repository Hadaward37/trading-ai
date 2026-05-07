import numpy as np
import pandas as pd

WINDOW_MINUTES = 15


def _momentum_direction(series_close: pd.Series) -> pd.Series:
    """Close-to-close momentum direction. Fallback when Open≈Close."""
    diff = series_close.diff()
    return diff.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))


def _body_direction(series_open: pd.Series, series_close: pd.Series) -> pd.Series:
    """Open-to-close body direction. Works for 5m+ candles with real OHLC."""
    diff = series_close - series_open
    return diff.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))


def _auto_direction(df: pd.DataFrame) -> pd.Series:
    """Pick body or momentum direction based on doji ratio."""
    doji_ratio = ((df["Close"] - df["Open"]).abs() == 0).mean()
    if doji_ratio > 0.5:
        return _momentum_direction(df["Close"])
    return _body_direction(df["Open"], df["Close"])


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


def compute_disorder(
    df_micro: pd.DataFrame,
    df_macro: pd.DataFrame = None,
    window_minutes: int = WINDOW_MINUTES,
) -> dict:
    """
    Disorder score (0-100): how chaotic micro price action is.

    Per window:
      - alternation: direction flip rate in micro candles (0=stable, 1=every flip)
      - atr_ratio:   micro ATR / macro ATR, normalized; high = micro-chaos
      - inconsistency: % of micro candles opposing macro direction

    Final score = mean(0.40*alternation + 0.25*atr_ratio + 0.35*inconsistency) * 100

    Supports any pair of timeframes (1m/5m, 5m/15m, etc.).
    """
    if df_macro is None:
        df_macro = pd.DataFrame()

    if df_micro.empty:
        return {"disorder_score": None, "n_windows": 0, "window_scores": [], "available": False}

    df_micro = _ensure_utc(df_micro)
    df_micro["_dir"] = _auto_direction(df_micro)
    df_micro["_atr"] = _local_atr(df_micro)

    has_macro = not df_macro.empty
    if has_macro:
        df_macro = _ensure_utc(df_macro)
        df_macro["_dir"] = _body_direction(df_macro["Open"], df_macro["Close"])
        df_macro["_body"] = (df_macro["Close"] - df_macro["Open"]).abs()
        df_macro["_atr"] = _local_atr(df_macro)

    t_start = df_micro.index.min()
    t_end = df_micro.index.max()
    if has_macro:
        t_start = max(t_start, df_macro.index.min())
        t_end = min(t_end, df_macro.index.max())

    min_micro = max(3, window_minutes // 20)

    windows = pd.date_range(
        start=t_start.floor(f"{window_minutes}min"),
        end=t_end.ceil(f"{window_minutes}min"),
        freq=f"{window_minutes}min",
        tz="UTC",
    )
    window_scores = []

    for w in windows[:-1]:
        w_end = w + pd.Timedelta(minutes=window_minutes)
        micro = df_micro[(df_micro.index >= w) & (df_micro.index < w_end)]

        if len(micro) < min_micro:
            continue

        # Component 1: alternation rate
        dirs = micro["_dir"].values
        flips = sum(
            1 for i in range(1, len(dirs))
            if dirs[i] != dirs[i - 1] and dirs[i] != 0 and dirs[i - 1] != 0
        )
        alternation = flips / (len(dirs) - 1) if len(dirs) > 1 else 0.0

        # Component 2: ATR ratio (micro vs macro volatility)
        atr_ratio_score = 0.5  # neutral when macro unavailable
        if has_macro:
            macro = df_macro[(df_macro.index >= w) & (df_macro.index < w_end)]
            if len(macro) >= 1:
                atr_micro = micro["_atr"].mean()
                atr_macro = macro["_atr"].mean()
                if atr_macro > 0:
                    ratio = atr_micro / atr_macro
                    atr_ratio_score = min(1.0, max(0.0, (ratio - 0.5) / 1.5))

        # Component 3: inconsistency with macro trend
        inconsistency = 0.5  # neutral when macro unavailable
        if has_macro:
            macro = df_macro[(df_macro.index >= w) & (df_macro.index < w_end)]
            if len(macro) >= 1:
                total_body = macro["_body"].sum()
                if total_body > 0:
                    w_dir = (macro["_dir"] * macro["_body"]).sum() / total_body
                    dom = 1 if w_dir > 0.05 else (-1 if w_dir < -0.05 else 0)
                    if dom != 0:
                        non_doji = micro[micro["_dir"] != 0]
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
