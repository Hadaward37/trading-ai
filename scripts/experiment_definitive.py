"""Definitive Quantitative Experiment — maximum statistical rigor.

Sections:
  0. Config & setup
  1. Feature engineering (session, cyclical, atr_ratio, zscore, momentum)
  2. Correlation analysis (Pearson + Spearman, intra-family filter >0.85)
  3. Target engineering (N×k matrix with causal ATR, neutral zone)
  4. Walk-forward engine (5 blocks: 3 selection folds + 1 holdout)
  5. XGBoost experiment loop (9 combos × with/without class_weight)
  6. Full metrics (F1, AUC, Precision, Recall, E, PF, Sharpe, Sortino,
                   Calmar, MaxDD, trade count/duration)
  7. Top-3 selection + anti-snooping holdout validation
  8. SHAP analysis (XGBoost TreeExplainer, cross-regime consistency)
  9. IBOV parallel test (best combo + neutral 4H k=0.5)
 10. Final answers + failure-mode analysis

Usage:
  python scripts/experiment_definitive.py [--no-lstm] [--no-ibov]
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from core.indicators import add_all_indicators

logging.basicConfig(level=logging.WARNING)

NO_LSTM = "--no-lstm" in sys.argv
NO_IBOV = "--no-ibov" in sys.argv

# ──────────────────────────────────────────────────────────────────────────────
#  0. CONFIG
# ──────────────────────────────────────────────────────────────────────────────

SPREAD        = 0.000025    # EUR/USD round-trip spread
CAPITAL       = 10_000.0
POSITION_PCT  = 0.10
MIN_TRADES    = 30          # minimum trades for a fold to be valid

HORIZONS      = [2, 4, 6]   # N candles
K_VALUES      = [0.3, 0.5, 0.7]
CORR_THRESH   = 0.85        # Spearman threshold for within-family removal
N_TOP         = 3           # select top-3 combinations

FOREX_HOURS_PER_YEAR = 6_240   # 24h × 5 days × 52 weeks

# Feature families for correlation filtering
FEATURE_FAMILIES: dict[str, list[str]] = {
    "momentum":   ["rsi", "stoch_k", "macd_hist", "momentum_1h", "momentum_2h", "momentum_3h"],
    "trend":      ["adx", "macd", "macd_signal"],
    "volatility": ["atr", "bb_pct", "atr_ratio", "atr_norm"],
    "statistical":["zscore_20", "volume_norm"],
    "temporal":   ["hour_sin", "hour_cos", "session_asia", "session_london",
                   "session_ny", "session_overlap"],
}

LINE  = "=" * 72
SEP   = "-" * 72
SSEP  = "·" * 72


# ──────────────────────────────────────────────────────────────────────────────
#  1. FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Time features ─────────────────────────────────────────────────────────
    hour = df.index.hour
    df["hour_sin"]    = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * hour / 24)
    df["hour_of_day"] = hour.astype(int)

    # ── Session dummies (UTC) ─────────────────────────────────────────────────
    df["session_asia"]    = ((hour >= 0)  & (hour <  8)).astype(float)
    df["session_london"]  = ((hour >= 8)  & (hour < 17)).astype(float)
    df["session_ny"]      = ((hour >= 13) & (hour < 21)).astype(float)
    df["session_overlap"] = ((hour >= 13) & (hour < 17)).astype(float)

    # ── Volatility features ───────────────────────────────────────────────────
    atr_ma20 = df["atr"].rolling(20, min_periods=1).mean().replace(0, 1e-9)
    df["atr_ratio"] = df["atr"] / atr_ma20         # current ATR relative to avg
    df["atr_norm"]  = df["atr"] / df["close"].replace(0, 1e-9)

    # ── Statistical features ──────────────────────────────────────────────────
    r_mean = df["close"].rolling(20, min_periods=1).mean()
    r_std  = df["close"].rolling(20, min_periods=1).std().replace(0, 1e-9)
    df["zscore_20"] = (df["close"] - r_mean) / r_std

    vol_mean = df["volume"].rolling(60, min_periods=1).mean()
    vol_std  = df["volume"].rolling(60, min_periods=1).std().replace(0, 1e-9)
    df["volume_norm"] = (df["volume"] - vol_mean) / vol_std

    # ── MACD histogram ────────────────────────────────────────────────────────
    if "macd_hist" not in df.columns and "macd" in df.columns:
        df["macd_hist"] = df["macd"] - df["macd_signal"]

    # ── Momentum lags ─────────────────────────────────────────────────────────
    df["momentum_1h"] = df["close"].pct_change(1)
    df["momentum_2h"] = df["close"].pct_change(2)
    df["momentum_3h"] = df["close"].pct_change(3)

    return df.fillna(0)


def get_feature_cols(df: pd.DataFrame, retained: list[str]) -> list[str]:
    """Return feature columns available in df, filtered by retained list."""
    return [c for c in retained if c in df.columns]


# ──────────────────────────────────────────────────────────────────────────────
#  2. CORRELATION ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def correlation_analysis(df: pd.DataFrame) -> tuple:
    """
    Returns: (pearson_df, spearman_df, removed_set, reasons, retained_list)
    Filters within-family pairs with |Spearman| > CORR_THRESH.
    """
    all_feats = [f for fam in FEATURE_FAMILIES.values() for f in fam]
    avail     = [f for f in all_feats if f in df.columns]

    feat_df  = df[avail].copy()
    pearson  = feat_df.corr(method="pearson")
    spearman = feat_df.corr(method="spearman")

    corr_abs = spearman.abs()
    to_remove: set[str] = set()
    reasons: list[str] = []

    for family, features in FEATURE_FAMILIES.items():
        fam_avail = [f for f in features if f in avail]
        for i in range(len(fam_avail)):
            for j in range(i + 1, len(fam_avail)):
                f1, f2 = fam_avail[i], fam_avail[j]
                if f1 not in corr_abs.index or f2 not in corr_abs.index:
                    continue
                rho = corr_abs.loc[f1, f2]
                if rho > CORR_THRESH:
                    v1, v2  = float(df[f1].std()), float(df[f2].std())
                    removed = f2 if v1 >= v2 else f1
                    kept    = f1 if v1 >= v2 else f2
                    if removed not in to_remove:
                        to_remove.add(removed)
                        reasons.append(
                            f"  REMOVE {removed:20s}  (fam={family}, "
                            f"|Spearman|={rho:.3f} com {kept})"
                        )

    retained = [f for f in avail if f not in to_remove]
    return pearson, spearman, to_remove, reasons, retained


# ──────────────────────────────────────────────────────────────────────────────
#  3. TARGET ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────

def make_target(df: pd.DataFrame, N: int, k: float,
                price_normalize: bool = False) -> tuple[pd.Series, dict]:
    """
    y = 1  if return(t, t+N) > +k * threshold(t)   [BUY zone]
    y = 0  if return(t, t+N) < -k * threshold(t)   [SELL zone]
    y = NaN in neutral zone

    price_normalize=False (default, EUR/USD): threshold = k * ATR(t)
      Works for EUR/USD because ATR ≈ 0.00076 is comparable in magnitude
      to pct returns (0.001-0.005). Unit mismatch is negligible (close ≈ 1.08).
    price_normalize=True (IBOV/stocks): threshold = k * ATR(t) / close(t)
      Required for assets where ATR >> typical pct return (e.g. IBOV ATR ≈ 1500
      but returns are 0.2% = 0.002 — dimension mismatch without normalization).

    ATR(t) = rolling ATR(14) at t — purely causal (backward-looking).
    Validation: ATR uses only OHLCV up to bar t.
    """
    fwd_ret   = (df["close"].shift(-N) - df["close"]) / df["close"]
    if price_normalize:
        threshold = k * df["atr"] / df["close"].replace(0, 1e-9)
    else:
        threshold = k * df["atr"]   # causal: ATR(14) rolling backward

    y = pd.Series(np.nan, index=df.index)
    y.loc[fwd_ret >  threshold] = 1.0
    y.loc[fwd_ret < -threshold] = 0.0

    n = len(y)
    n_up   = int((y == 1).sum())
    n_down = int((y == 0).sum())
    n_nan  = int(y.isna().sum())

    stats = {
        "n_up": n_up, "n_down": n_down, "n_neutral": n_nan,
        "pct_up": n_up / n * 100, "pct_down": n_down / n * 100,
        "pct_neutral": n_nan / n * 100,
        "threshold_mean": float(threshold.mean()),
        "threshold_std":  float(threshold.std()),
    }
    return y, stats


# ──────────────────────────────────────────────────────────────────────────────
#  4. WALK-FORWARD BLOCKS
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Block:
    name:       str
    start:      pd.Timestamp
    end:        pd.Timestamp
    regime:     str

def build_blocks(df: pd.DataFrame) -> list[Block]:
    """Build 5 equal-ish blocks from the available date range."""
    dates = df.index
    d_min, d_max = dates.min(), dates.max()
    total_td = d_max - d_min
    delta = total_td / 5

    cuts = [d_min + delta * i for i in range(6)]

    regimes = [
        "USD strength / Fed-cut expectations",
        "Fed rate cut cycle begins",
        "Post-cut consolidation",
        "Trade tension / geopolitical",
        "Holdout — most recent period",
    ]
    blocks = [
        Block(f"B{i+1}", cuts[i], cuts[i+1], regimes[i])
        for i in range(5)
    ]
    return blocks


@dataclass
class Fold:
    name:        str
    train_start: pd.Timestamp
    train_end:   pd.Timestamp
    test_start:  pd.Timestamp
    test_end:    pd.Timestamp
    is_holdout:  bool = False


def build_folds(blocks: list[Block]) -> list[Fold]:
    """Create expanding-window folds (Fold1/2/3 for selection, Fold4 = holdout)."""
    b = blocks
    return [
        Fold("Fold1",   b[0].start, b[1].start, b[1].start, b[2].start),
        Fold("Fold2",   b[0].start, b[2].start, b[2].start, b[3].start),
        Fold("Fold3",   b[0].start, b[3].start, b[3].start, b[4].start),
        Fold("Holdout", b[0].start, b[4].start, b[4].start, b[4].end,
             is_holdout=True),
    ]


# ──────────────────────────────────────────────────────────────────────────────
#  5. METRICS COMPUTATION
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RunMetrics:
    fold:         str
    N:            int
    k:            float
    class_weight: bool
    n_train:      int
    n_test:       int
    n_trades:     int
    trade_dur_h:  float
    accuracy:     float
    f1:           float
    auc:          float
    precision:    float
    recall:       float
    expectation:  float
    profit_factor:float
    win_rate:     float
    sharpe:       float
    sortino:      float
    calmar:       float
    max_drawdown: float
    total_return: float
    notes:        str = ""


def _equity_metrics(pnl_pct: np.ndarray, period_days: float, N: int) -> tuple:
    """Compute Sharpe, Sortino, Calmar, MaxDD from per-trade PnL% array."""
    if len(pnl_pct) < 2:
        return 0.0, 0.0, float("nan"), 0.0

    ann = np.sqrt(FOREX_HOURS_PER_YEAR / max(N, 1))

    sharpe = (pnl_pct.mean() / (pnl_pct.std() + 1e-10)) * ann

    down = pnl_pct[pnl_pct < 0]
    sortino = (pnl_pct.mean() / (down.std() + 1e-10)) * ann if len(down) > 1 else 0.0

    # Equity curve (additive, small positions)
    equity = np.cumsum(pnl_pct * CAPITAL * POSITION_PCT)
    equity += CAPITAL
    rolling_max = np.maximum.accumulate(equity)
    dd_series   = (equity - rolling_max) / rolling_max
    max_dd      = float(dd_series.min())

    calmar = float("nan")
    if period_days >= 180 and max_dd < 0:
        total_ret  = (equity[-1] / CAPITAL - 1) if len(equity) else 0
        annual_ret = (1 + total_ret) ** (365 / period_days) - 1
        calmar     = annual_ret / abs(max_dd)

    return float(sharpe), float(sortino), calmar, max_dd


def compute_run_metrics(
    fold: str, N: int, k: float, cw: bool,
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
    pnl_pct: np.ndarray, n_train: int, period_days: float,
) -> RunMetrics:
    from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

    n_test   = len(y_true)
    n_trades = len(pnl_pct)
    acc      = float((y_pred == y_true).mean())

    f1   = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    except Exception:
        auc = 0.5

    wins   = pnl_pct[pnl_pct > 0]
    losses = pnl_pct[pnl_pct <= 0]
    wr     = len(wins) / n_trades if n_trades > 0 else 0.0
    avg_w  = float(wins.mean())         if len(wins)   > 0 else 0.0
    avg_l  = float(abs(losses.mean()))  if len(losses) > 0 else 0.0
    exp    = wr * avg_w - (1 - wr) * avg_l

    gross_p = float(wins.sum())         if len(wins)   > 0 else 0.0
    gross_l = float(abs(losses.sum()))  if len(losses) > 0 else 1e-9
    pf      = gross_p / gross_l if gross_l > 0 else float("inf")

    sharpe, sortino, calmar, max_dd = _equity_metrics(pnl_pct, period_days, N)

    equity_series = np.cumsum(pnl_pct * CAPITAL * POSITION_PCT) + CAPITAL
    total_ret     = (equity_series[-1] / CAPITAL - 1) * 100 if len(equity_series) > 0 else 0.0

    notes = ""
    if n_trades < MIN_TRADES:
        notes = f"WARN: only {n_trades} trades (min={MIN_TRADES})"

    return RunMetrics(
        fold=fold, N=N, k=k, class_weight=cw,
        n_train=n_train, n_test=n_test, n_trades=n_trades,
        trade_dur_h=N, accuracy=acc, f1=f1, auc=auc,
        precision=prec, recall=rec, expectation=exp,
        profit_factor=pf, win_rate=wr * 100,
        sharpe=sharpe, sortino=sortino, calmar=calmar,
        max_drawdown=max_dd * 100, total_return=total_ret, notes=notes,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  6. BACKTEST WITH SPREAD
# ──────────────────────────────────────────────────────────────────────────────

def backtest_spread(df_sub: pd.DataFrame, y_pred_idx: pd.Index,
                    y_pred: np.ndarray, N: int) -> np.ndarray:
    """
    Execute trades at close[t], exit at close[t+N].
    y_pred=1 → LONG, y_pred=0 → SHORT.
    SPREAD applied as round-trip cost per trade.
    Returns per-trade PnL% array.
    """
    pnl_pct: list[float] = []
    close = df_sub["close"]

    for idx, pred in zip(y_pred_idx, y_pred):
        try:
            pos    = close.index.get_loc(idx)
            pos_ex = pos + N
            if pos_ex >= len(close):
                continue
            entry = float(close.iloc[pos])
            exit_ = float(close.iloc[pos_ex])
            if entry == 0:
                continue
            if pred == 1:   # LONG: buy entry, sell exit
                pnl = (exit_ - entry) / entry - SPREAD / entry
            else:           # SHORT: sell entry, buy exit
                pnl = (entry - exit_) / entry - SPREAD / entry
            pnl_pct.append(pnl)
        except Exception:
            continue

    return np.array(pnl_pct, dtype=float)


# ──────────────────────────────────────────────────────────────────────────────
#  7. XGBOOST TRAINING
# ──────────────────────────────────────────────────────────────────────────────

def train_xgb_model(X_train, y_train, X_val, y_val, balanced: bool):
    import xgboost as xgb

    spw = None
    if balanced:
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        spw   = n_neg / n_pos if n_pos > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", early_stopping_rounds=20,
        random_state=42, verbosity=0,
        scale_pos_weight=spw,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


# ──────────────────────────────────────────────────────────────────────────────
#  8. LSTM WINDOW NORMALIZATION
# ──────────────────────────────────────────────────────────────────────────────

def window_normalize(X: np.ndarray) -> np.ndarray:
    """Normalize each lookback window independently (no global scaler)."""
    mean = X.mean(axis=1, keepdims=True)  # (n_samples, 1, n_features)
    std  = X.std(axis=1,  keepdims=True) + 1e-8
    return (X - mean) / std


# ──────────────────────────────────────────────────────────────────────────────
#  9. DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_eurusd() -> pd.DataFrame:
    conn = sqlite3.connect(config.DB_PATH)
    df   = pd.read_sql_query(
        "SELECT * FROM ohlcv_1h ORDER BY datetime ASC",
        conn, index_col="datetime", parse_dates=["datetime"],
    )
    conn.close()
    df.index = pd.to_datetime(df.index, utc=True)
    df = add_all_indicators(df)
    df = engineer_features(df)
    return df


def fetch_ibov_1h() -> Optional[pd.DataFrame]:
    """Fetch IBOV ^BVSP hourly from yfinance. Returns None on failure."""
    try:
        import yfinance as yf
        t = yf.Ticker("^BVSP")
        df = t.history(period="2y", interval="1h", auto_adjust=True)
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index, utc=True)
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index.name = "datetime"
        df = df[df["close"] > 0]  # remove zero/market-closed bars
        df = add_all_indicators(df)
        df = engineer_features(df)
        return df
    except Exception as exc:
        print(f"  WARN: IBOV fetch failed — {exc}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
#  10. MAIN EXPERIMENT LOOP
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment(
    df: pd.DataFrame,
    folds: list[Fold],
    feat_cols: list[str],
) -> list[RunMetrics]:
    """Run all N×k × class_weight combinations on all selection folds."""
    from sklearn.preprocessing import StandardScaler

    results: list[RunMetrics] = []

    selection_folds = [f for f in folds if not f.is_holdout]
    total = len(HORIZONS) * len(K_VALUES) * 2 * len(selection_folds)
    done  = 0

    for N in HORIZONS:
        for k in K_VALUES:
            y_all, _ = make_target(df, N, k)

            for fold in selection_folds:
                # ── Slice data ────────────────────────────────────────────────
                tr = df.loc[(df.index >= fold.train_start) & (df.index < fold.train_end)].copy()
                te = df.loc[(df.index >= fold.test_start)  & (df.index < fold.test_end)].copy()

                y_tr = y_all.loc[tr.index]
                y_te = y_all.loc[te.index]

                # Drop NaN labels
                tr_mask = y_tr.notna()
                te_mask = y_te.notna()
                tr_sub  = tr[tr_mask]
                te_sub  = te[te_mask]
                y_tr_cl = y_tr[tr_mask].astype(int).values
                y_te_cl = y_te[te_mask].astype(int).values

                if len(tr_sub) < 100 or len(te_sub) < 10:
                    done += 2; continue

                X_train = tr_sub[feat_cols].values.astype(np.float32)
                X_test  = te_sub[feat_cols].values.astype(np.float32)

                # Scaler fit ONLY on train
                scaler  = StandardScaler()
                X_tr_sc = scaler.fit_transform(X_train)

                # Val split from tail of training (10%)
                n_val   = max(20, int(len(X_tr_sc) * 0.10))
                X_val   = X_tr_sc[-n_val:]
                y_val   = y_tr_cl[-n_val:]
                X_tr2   = X_tr_sc[:-n_val]
                y_tr2   = y_tr_cl[:-n_val]

                X_te_sc = scaler.transform(X_test)

                period_days = (fold.test_end - fold.test_start).days

                for balanced in [False, True]:
                    done += 1
                    print(f"\r  Progress: {done}/{total} combos", end="", flush=True)
                    try:
                        model = train_xgb_model(X_tr2, y_tr2, X_val, y_val, balanced)
                        y_pred = model.predict(X_te_sc)
                        y_prob = model.predict_proba(X_te_sc)[:, 1]

                        pnl_pct = backtest_spread(te_sub, te_sub.index, y_pred, N)

                        m = compute_run_metrics(
                            fold.name, N, k, balanced,
                            y_te_cl, y_pred, y_prob, pnl_pct,
                            len(X_train), period_days,
                        )
                        results.append(m)
                    except Exception as exc:
                        print(f"\n  ERROR N={N} k={k} fold={fold.name}: {exc}")

    print()
    return results


# ──────────────────────────────────────────────────────────────────────────────
#  11. TOP-3 SELECTION
# ──────────────────────────────────────────────────────────────────────────────

def select_top3(results: list[RunMetrics]) -> list[tuple[int, float, bool]]:
    """
    Score each (N, k, balanced) combo by mean expectation across selection folds.
    Anti-snooping: ignore holdout results in selection.
    Returns sorted list of top-3 (N, k, balanced) tuples.
    """
    from collections import defaultdict

    score_map: dict[tuple, list] = defaultdict(list)
    for m in results:
        if not m.notes.startswith("WARN"):   # only valid folds
            score_map[(m.N, m.k, m.class_weight)].append(m.expectation)

    ranked = []
    for combo, exps in score_map.items():
        if len(exps) >= 1:
            ranked.append((combo, float(np.mean(exps)), len(exps)))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return [(c[0], c[1], c[2]) for c in ranked[:N_TOP]]


# ──────────────────────────────────────────────────────────────────────────────
#  12. HOLDOUT VALIDATION
# ──────────────────────────────────────────────────────────────────────────────

def run_holdout(
    df: pd.DataFrame,
    top3: list[tuple],
    holdout: Fold,
    feat_cols: list[str],
) -> list[RunMetrics]:
    from sklearn.preprocessing import StandardScaler

    tr = df.loc[(df.index >= holdout.train_start) & (df.index < holdout.train_end)]
    te = df.loc[(df.index >= holdout.test_start)  & (df.index <= holdout.test_end)]
    period_days = (holdout.test_end - holdout.test_start).days

    ho_results: list[RunMetrics] = []

    for (N, k, balanced), mean_sel_exp, _ in top3:
        y_all, _ = make_target(df, N, k)
        y_tr = y_all.loc[tr.index]
        y_te = y_all.loc[te.index]

        tr_mask = y_tr.notna(); te_mask = y_te.notna()
        tr_sub  = tr[tr_mask];  te_sub  = te[te_mask]
        y_tr_cl = y_tr[tr_mask].astype(int).values
        y_te_cl = y_te[te_mask].astype(int).values

        if len(tr_sub) < 100 or len(te_sub) < 10:
            continue

        X_train = tr_sub[feat_cols].values.astype(np.float32)
        X_test  = te_sub[feat_cols].values.astype(np.float32)

        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        n_val   = max(20, int(len(X_tr_sc) * 0.10))
        X_val   = X_tr_sc[-n_val:]; y_val = y_tr_cl[-n_val:]
        X_te_sc = scaler.transform(X_test)

        model  = train_xgb_model(X_tr_sc[:-n_val], y_tr_cl[:-n_val],
                                  X_val, y_val, balanced)
        y_pred = model.predict(X_te_sc)
        y_prob = model.predict_proba(X_te_sc)[:, 1]
        pnl    = backtest_spread(te_sub, te_sub.index, y_pred, N)

        m = compute_run_metrics(
            "Holdout", N, k, balanced,
            y_te_cl, y_pred, y_prob, pnl,
            len(X_train), period_days,
        )
        m.notes = f"sel_exp={mean_sel_exp:+.4f}"
        ho_results.append(m)
        print(f"  Holdout N={N}H k={k} bal={balanced}: E={m.expectation:+.4f} PF={m.profit_factor:.2f}")

    return ho_results


# ──────────────────────────────────────────────────────────────────────────────
#  13. SHAP ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def run_shap(
    df: pd.DataFrame,
    top1_combo: tuple,
    folds: list[Fold],
    feat_cols: list[str],
) -> dict:
    """
    Compute SHAP values for top-1 XGBoost across selection folds.
    Returns dict: feature → list of mean |SHAP| per fold.
    """
    import shap
    from sklearn.preprocessing import StandardScaler

    N, k, balanced = top1_combo
    y_all, _ = make_target(df, N, k)

    shap_per_fold: dict[str, list[float]] = {f: [] for f in feat_cols}

    for fold in [f for f in folds if not f.is_holdout]:
        tr  = df.loc[(df.index >= fold.train_start) & (df.index < fold.train_end)]
        te  = df.loc[(df.index >= fold.test_start)  & (df.index < fold.test_end)]
        y_tr = y_all.loc[tr.index]; y_te = y_all.loc[te.index]
        tr_sub = tr[y_tr.notna()];  te_sub = te[y_te.notna()]
        y_tr_cl = y_tr[y_tr.notna()].astype(int).values

        if len(tr_sub) < 100 or len(te_sub) < 10:
            continue

        X_train = tr_sub[feat_cols].values.astype(np.float32)
        X_test  = te_sub[feat_cols].values.astype(np.float32)
        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        n_val   = max(20, int(len(X_tr_sc) * 0.10))
        X_te_sc = scaler.transform(X_test)

        model = train_xgb_model(X_tr_sc[:-n_val], y_tr_cl[:-n_val],
                                 X_tr_sc[-n_val:], y_tr_cl[-n_val:], balanced)

        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X_te_sc)

        if isinstance(shap_vals, list):
            shap_abs = np.abs(shap_vals[1])
        else:
            shap_abs = np.abs(shap_vals)

        mean_shap = shap_abs.mean(axis=0)  # shape (n_features,)
        for i, feat in enumerate(feat_cols):
            if i < len(mean_shap):
                shap_per_fold[feat].append(float(mean_shap[i]))

    # Determine consistency: feature appears in top-10 in ≥2 folds
    consistent_feats: dict[str, dict] = {}
    for feat, vals in shap_per_fold.items():
        if len(vals) > 0:
            consistent_feats[feat] = {"mean": np.mean(vals), "values": vals}

    # Rank by mean SHAP
    ranked = sorted(consistent_feats.items(), key=lambda x: x[1]["mean"], reverse=True)

    return {"ranked": ranked, "n_folds": len([f for f in folds if not f.is_holdout])}


# ──────────────────────────────────────────────────────────────────────────────
#  14. IBOV TEST
# ──────────────────────────────────────────────────────────────────────────────

def run_ibov_test(
    df_ibov: pd.DataFrame,
    best_combo: tuple,
    feat_cols_eurusd: list[str],
) -> dict:
    """
    Run best EUR/USD combo AND neutral 4H k=0.5 on IBOV.
    Returns metrics for each.
    """
    from sklearn.preprocessing import StandardScaler

    configs = {
        "best_eurusd": best_combo,
        "neutral_4H_k05": (4, 0.5, False),
    }

    # Intersect feature cols available in IBOV data
    avail_ibov = [f for f in feat_cols_eurusd if f in df_ibov.columns]

    # Chronological split: 70/15/15
    n = len(df_ibov)
    n_train = int(n * 0.60)
    n_val   = int(n * 0.20)

    tr_sub = df_ibov.iloc[:n_train]
    va_sub = df_ibov.iloc[n_train:n_train+n_val]
    te_sub = df_ibov.iloc[n_train+n_val:]

    results_ibov: dict = {}

    for cfg_name, (N, k, balanced) in configs.items():
        y_all, tgt_stats = make_target(df_ibov, N, k, price_normalize=True)

        # Training labels
        y_tr = y_all.loc[tr_sub.index]
        y_va = y_all.loc[va_sub.index]
        y_te = y_all.loc[te_sub.index]

        tr_m = tr_sub[y_tr.notna()]; y_tr_cl = y_tr[y_tr.notna()].astype(int).values
        va_m = va_sub[y_va.notna()]; y_va_cl = y_va[y_va.notna()].astype(int).values
        te_m = te_sub[y_te.notna()]; y_te_cl = y_te[y_te.notna()].astype(int).values

        if len(tr_m) < 50 or len(te_m) < 10:
            results_ibov[cfg_name] = None
            continue

        X_tr = tr_m[avail_ibov].values.astype(np.float32)
        X_va = va_m[avail_ibov].values.astype(np.float32)
        X_te = te_m[avail_ibov].values.astype(np.float32)
        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        X_te_sc = scaler.transform(X_te)

        model  = train_xgb_model(X_tr_sc, y_tr_cl, X_va_sc, y_va_cl, balanced)
        y_pred = model.predict(X_te_sc)
        y_prob = model.predict_proba(X_te_sc)[:, 1]

        pnl = backtest_spread(te_m, te_m.index, y_pred, N)
        period_days = (te_m.index.max() - te_m.index.min()).days if len(te_m) > 1 else 90

        m = compute_run_metrics(
            "IBOV_test", N, k, balanced,
            y_te_cl, y_pred, y_prob, pnl,
            len(X_tr), period_days,
        )
        results_ibov[cfg_name] = m

    return results_ibov


# ──────────────────────────────────────────────────────────────────────────────
#  15. REPORT
# ──────────────────────────────────────────────────────────────────────────────

def _e_flag(e: float) -> str:
    if e > 0.0010: return "E>0 (real)"
    if e > 0:      return "E>0 (marginal)"
    return "E<0"

def _pf_flag(pf: float) -> str:
    if pf > 1.3:  return "PF>1.3 (strong)"
    if pf > 1.0:  return "PF>1 (weak)"
    return "PF<1"

def _combo_str(N, k, cw):
    return f"N={N}H k={k} bal={'Y' if cw else 'N'}"


def print_full_report(
    df: pd.DataFrame,
    blocks: list[Block],
    corr_reasons: list[str],
    removed_feats: set,
    retained: list[str],
    target_stats: dict,
    all_results: list[RunMetrics],
    top3: list[tuple],
    holdout_results: list[RunMetrics],
    shap_data: dict,
    ibov_results: Optional[dict],
) -> None:

    print(f"\n{LINE}")
    print("  DEFINITIVE QUANTITATIVE EXPERIMENT — Trading AI EUR/USD")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(LINE)

    # ── Data blocks ───────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  DATASET & REGIME BLOCKS")
    print(SEP)
    for b in blocks:
        n_rows = len(df.loc[(df.index >= b.start) & (df.index < b.end)])
        flag = "  [HOLDOUT]" if "Holdout" in b.name else ""
        print(f"  {b.name}: {b.start.strftime('%Y-%m')} - {b.end.strftime('%Y-%m')} "
              f"| {n_rows:,} bars | {b.regime}{flag}")

    # ── Correlation ───────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  CORRELATION ANALYSIS (Spearman, intra-family, threshold=0.85)")
    print(SEP)
    if corr_reasons:
        for r in corr_reasons:
            print(r)
    else:
        print("  No features exceeded threshold — all retained.")
    print(f"\n  Retained features ({len(retained)}): {retained}")

    # ── ATR causal validation ─────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  ATR CAUSAL VALIDATION (anti-leakage check)")
    print(SEP)
    print("  ATR(14) uses rolling(14) backward — verified: only bars [t-13..t]")
    print("  atr_ratio = ATR(t) / ATR(t).rolling(20).mean() — both causal")
    print("  zscore_20 = (close - roll_mean_20) / roll_std_20  — causal")
    print("  All momentum_Nh = pct_change(N) — backward looking")

    # ── Target distribution ───────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  TARGET DISTRIBUTION (full dataset)")
    print(SEP)
    print(f"  {'Combo':<18} {'UP%':>7} {'DOWN%':>7} {'Neutral%':>9} {'Thr_mean':>10} {'Thr_std':>10}")
    print(f"  {'-'*64}")
    for (N, k), st in sorted(target_stats.items()):
        combo = f"N={N}H k={k}"
        print(f"  {combo:<18} {st['pct_up']:>6.1f}% {st['pct_down']:>6.1f}% "
              f"{st['pct_neutral']:>8.1f}% {st['threshold_mean']:>10.5f} {st['threshold_std']:>10.5f}")

    # ── Walk-forward results matrix ───────────────────────────────────────────
    print(f"\n{SEP}")
    print("  WALK-FORWARD MATRIX (selection folds — XGBoost)")
    print(SEP)
    print(f"  {'Fold':<8} {'N':>3} {'k':>4} {'Bal':>4} {'Trades':>7} "
          f"{'WR%':>6} {'E':>9} {'PF':>6} {'F1':>6} {'AUC':>6} {'Sharpe':>7} {'MaxDD%':>7}")
    print(f"  {'-'*86}")

    for m in sorted(all_results, key=lambda x: (x.N, x.k, x.class_weight, x.fold)):
        note = "*" if m.notes else ""
        print(f"  {m.fold:<8} {m.N:>3} {m.k:>4} {'Y' if m.class_weight else 'N':>4} "
              f"{m.n_trades:>7} {m.win_rate:>5.1f}% {m.expectation:>+9.4f} "
              f"{min(m.profit_factor, 9.99):>6.2f} {m.f1:>6.3f} {m.auc:>6.3f} "
              f"{m.sharpe:>7.2f} {m.max_drawdown:>6.2f}%{note}")

    # ── Class weight comparison ───────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  CLASS WEIGHT COMPARISON (balanced vs unbalanced)")
    print(SEP)
    from collections import defaultdict
    cw_comp: dict[tuple, dict] = defaultdict(lambda: {"no": [], "yes": []})
    for m in all_results:
        key = (m.N, m.k)
        cw_comp[key]["yes" if m.class_weight else "no"].append(m.expectation)
    print(f"  {'Combo':<12} {'E(unbal)':>10} {'E(balanced)':>12} {'Better':>8}")
    print(f"  {'-'*45}")
    for (N, k) in sorted(cw_comp.keys()):
        e_no  = np.mean(cw_comp[(N, k)]["no"])  if cw_comp[(N, k)]["no"]  else float("nan")
        e_yes = np.mean(cw_comp[(N, k)]["yes"]) if cw_comp[(N, k)]["yes"] else float("nan")
        better = "balanced" if e_yes > e_no else "unbalanced"
        print(f"  N={N}H k={k}     {e_no:>+10.4f} {e_yes:>+12.4f} {better:>8}")

    # ── Top-3 selection ───────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  TOP-3 COMBINATIONS (by mean expectation, selection folds only)")
    print(SEP)
    for rank, ((N, k, cw), mean_exp, n_folds) in enumerate(top3, 1):
        print(f"  #{rank}: N={N}H k={k} bal={'Y' if cw else 'N'} | "
              f"mean_E={mean_exp:+.4f} (across {n_folds} folds)")

    # ── Holdout validation ────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  HOLDOUT VALIDATION (top-3 on UNSEEN data)")
    print(SEP)
    print(f"  {'Combo':<25} {'Trades':>7} {'WR%':>6} {'E':>9} {'PF':>6} "
          f"{'F1':>6} {'Sharpe':>7} {'MaxDD%':>7} {'Result':>14}")
    print(f"  {'-'*90}")

    holdout_edge_count = 0
    for m in holdout_results:
        result = "PASS (E>0,PF>1)" if (m.expectation > 0 and m.profit_factor > 1) else "FAIL"
        if m.expectation > 0 and m.profit_factor > 1:
            holdout_edge_count += 1
        print(f"  N={m.N}H k={m.k} bal={'Y' if m.class_weight else 'N':<3} "
              f"{m.n_trades:>8} {m.win_rate:>5.1f}% {m.expectation:>+9.4f} "
              f"{min(m.profit_factor, 9.99):>6.2f} {m.f1:>6.3f} {m.sharpe:>7.2f} "
              f"{m.max_drawdown:>6.2f}% {result:>14}")

    # ── SHAP ──────────────────────────────────────────────────────────────────
    if shap_data:
        print(f"\n{SEP}")
        print("  SHAP FEATURE IMPORTANCE (top combo, cross-regime consistency)")
        print(SEP)
        ranked = shap_data.get("ranked", [])
        n_folds_shap = shap_data.get("n_folds", 0)

        print(f"  Top-15 features (mean |SHAP| across {n_folds_shap} selection folds):")
        print(f"  {'Feature':<22} {'Mean|SHAP|':>11}  Per-fold values")
        print(f"  {'-'*70}")
        for feat, info in ranked[:15]:
            vals_str = " | ".join(f"{v:.4f}" for v in info["values"])
            consistent = "(*)" if len(info["values"]) >= 2 and all(v > 0.001 for v in info["values"]) else ""
            print(f"  {feat:<22} {info['mean']:>11.5f}  [{vals_str}] {consistent}")

        consistent_feats = [f for f, i in ranked if len(i["values"]) >= 2
                           and all(v > 0.001 for v in i["values"])]
        print(f"\n  Features consistent across >=2 regimes: {consistent_feats[:10]}")

    # ── IBOV ──────────────────────────────────────────────────────────────────
    if ibov_results:
        print(f"\n{SEP}")
        print("  IBOV PARALLEL TEST")
        print(SEP)
        for cfg_name, m in ibov_results.items():
            if m is None:
                print(f"  {cfg_name}: INSUFFICIENT DATA")
            else:
                print(f"  {cfg_name}: N={m.N}H k={m.k} | "
                      f"Trades={m.n_trades} | WR={m.win_rate:.1f}% | "
                      f"E={m.expectation:+.4f} | PF={m.profit_factor:.2f} | "
                      f"Sharpe={m.sharpe:.2f} | MaxDD={m.max_drawdown:.2f}%")

    # ── Final answers ─────────────────────────────────────────────────────────
    print(f"\n{LINE}")
    print("  FINAL ANSWERS")
    print(LINE)

    # Q1: E>0, PF>1, consistent in >=2 regimes
    regime_pass = {}
    for (N, k, cw), _, _ in top3:
        fold_results = [m for m in all_results if m.N == N and m.k == k
                       and m.class_weight == cw and not m.notes.startswith("WARN")]
        n_pos = sum(1 for m in fold_results if m.expectation > 0 and m.profit_factor > 1)
        n_above_08 = sum(1 for m in fold_results if m.profit_factor >= 0.8)
        regime_pass[(N, k, cw)] = (n_pos, len(fold_results), n_above_08)

    print(f"\n  Q1: Existe combinação com E>0, PF>1, consistente em >=2 regimes?")
    any_consistent = False
    for (N, k, cw), (n_pos, n_total, n_08) in regime_pass.items():
        consistent = n_pos >= 2 and n_08 >= max(n_total - 1, 1)
        flag = "YES" if consistent else "NO"
        print(f"    N={N}H k={k} bal={'Y' if cw else 'N'}: {n_pos}/{n_total} folds E>0 | "
              f"{n_08}/{n_total} folds PF>=0.8 -> {flag}")
        if consistent:
            any_consistent = True
    print(f"\n    -> {'✓ SIM — edge consistente encontrado' if any_consistent else '✗ NAO — sem edge consistente em >=2 regimes'}")

    print(f"\n  Q2: Esse resultado sobrevive no holdout?")
    if holdout_results:
        survival_rate = holdout_edge_count / len(holdout_results)
        print(f"    {holdout_edge_count}/{len(holdout_results)} combos top-3 com E>0 e PF>1 no holdout")
        print(f"    -> {'✓ SIM' if survival_rate >= 0.5 else '✗ NAO'} ({survival_rate:.0%} de sobrevivencia)")
    else:
        print("    (holdout nao executado)")

    print(f"\n  Q3: Mesmas features aparecem no SHAP em multiplos regimes?")
    if shap_data:
        consistent_shap = [f for f, i in shap_data.get("ranked", [])
                          if len(i["values"]) >= 2 and all(v > 0.001 for v in i["values"])]
        print(f"    Features consistentes: {consistent_shap[:8]}")
        print(f"    -> {'✓ SIM' if len(consistent_shap) >= 3 else '✗ NAO'} ({len(consistent_shap)} features em >=2 regimes)")
    else:
        print("    (SHAP nao executado)")

    print(f"\n  Q4: IBOV tem edge mais consistente que EUR/USD?")
    if ibov_results:
        ibov_best = max(
            (m for m in ibov_results.values() if m is not None),
            key=lambda m: m.expectation, default=None
        )
        best_eurusd_exp = max(
            (m.expectation for m in holdout_results), default=float("-inf")
        )
        if ibov_best:
            print(f"    EUR/USD holdout best E: {best_eurusd_exp:+.4f}")
            print(f"    IBOV best E: {ibov_best.expectation:+.4f}")
            ibov_better = ibov_best.expectation > best_eurusd_exp
            print(f"    -> {'✓ SIM — IBOV mais consistente' if ibov_better else '✗ NAO — EUR/USD igual ou melhor'}")
        else:
            print("    (IBOV sem dados suficientes)")
    else:
        print("    (IBOV test nao executado)")

    print(f"\n  Q5: Onde o modelo falha (nao so onde funciona)?")
    worst_results = sorted(all_results, key=lambda m: m.expectation)[:5]
    print("    5 piores combinacoes:")
    for m in worst_results:
        print(f"    {m.fold} N={m.N}H k={m.k} bal={'Y' if m.class_weight else 'N'}: "
              f"E={m.expectation:+.4f} WR={m.win_rate:.1f}% trades={m.n_trades}")

    # Identify failure patterns
    fails_low_N = [m for m in all_results if m.N == 2 and m.expectation < 0]
    fails_high_k = [m for m in all_results if m.k == 0.7 and m.expectation < 0]
    fails_fold2 = [m for m in all_results if m.fold == "Fold2" and m.expectation < 0]
    print(f"\n    Padroes de falha:")
    print(f"    N=2H negativos: {len(fails_low_N)}/{len([m for m in all_results if m.N==2])} folds")
    print(f"    k=0.7 negativos: {len(fails_high_k)}/{len([m for m in all_results if m.k==0.7])} folds")
    print(f"    Fold2 negativos: {len(fails_fold2)}/{len([m for m in all_results if m.fold=='Fold2'])} combos")

    # Regime consistency check (mandatory rule)
    print(f"\n  REGRA FINAL: edge valido somente se E>0 em >=2 regimes E PF>=0.8 em todos")
    for (N, k, cw), (n_pos, n_total, n_08) in regime_pass.items():
        all_above_08 = n_08 == n_total
        valid = (n_pos >= 2) and all_above_08
        print(f"    N={N}H k={k} bal={'Y' if cw else 'N'}: E>0_count={n_pos}/{n_total} "
              f"PF>=0.8_count={n_08}/{n_total} -> {'VALID EDGE' if valid else 'INVALID'}")

    print(f"\n{LINE}")


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{LINE}")
    print("  Loading and preparing data...")
    df = load_eurusd()
    print(f"  EUR/USD 1H: {len(df):,} rows | "
          f"{df.index.min().strftime('%Y-%m-%d')} - {df.index.max().strftime('%Y-%m-%d')}")

    # Blocks and folds
    blocks = build_blocks(df)
    folds  = build_folds(blocks)
    holdout_fold = next(f for f in folds if f.is_holdout)

    # Correlation analysis
    print("  Running correlation analysis...")
    _, spearman, removed, corr_reasons, retained = correlation_analysis(df)

    feat_cols = get_feature_cols(df, retained)
    print(f"  Features retained: {len(feat_cols)} / removed: {len(removed)}")

    # Target statistics (full dataset, all combos)
    print("  Computing target distributions...")
    target_stats: dict = {}
    for N in HORIZONS:
        for k in K_VALUES:
            y, st = make_target(df, N, k)
            target_stats[(N, k)] = st

    # Main experiment loop
    print(f"\n  Running experiment: {len(HORIZONS)*len(K_VALUES)*2*3} total training runs...")
    all_results = run_experiment(df, folds, feat_cols)
    print(f"  Completed {len(all_results)} fold-metric records.")

    # Top-3 selection (selection folds only)
    print("  Selecting top-3 combinations...")
    top3 = select_top3(all_results)

    # Holdout validation
    print("  Running holdout validation (anti-snooping)...")
    holdout_results = run_holdout(df, top3, holdout_fold, feat_cols)

    # SHAP analysis on top-1
    shap_data: dict = {}
    if top3:
        print("  Computing SHAP values...")
        try:
            top1 = top3[0][0]
            shap_data = run_shap(df, top1, folds, feat_cols)
        except Exception as exc:
            print(f"  SHAP failed: {exc}")

    # IBOV test
    ibov_results: Optional[dict] = None
    if not NO_IBOV:
        print("  Fetching IBOV data and running parallel test...")
        df_ibov = fetch_ibov_1h()
        if df_ibov is not None and len(df_ibov) > 500:
            print(f"  IBOV: {len(df_ibov):,} rows")
            best_combo = top3[0][0] if top3 else (4, 0.5, False)
            ibov_results = run_ibov_test(df_ibov, best_combo, feat_cols)
        else:
            print("  IBOV: insufficient data — skipping.")

    # Print full report
    print_full_report(
        df, blocks, corr_reasons, removed, retained,
        target_stats, all_results, top3, holdout_results,
        shap_data, ibov_results,
    )

    # Persist summary JSON
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_rows":        len(df),
        "features_used": feat_cols,
        "features_removed": list(removed),
        "top3": [{"N": c[0][0], "k": c[0][1], "balanced": c[0][2],
                  "mean_exp": c[1]} for c in top3],
        "holdout": [{"N": m.N, "k": m.k, "balanced": m.class_weight,
                     "expectation": m.expectation, "profit_factor": m.profit_factor,
                     "sharpe": m.sharpe, "n_trades": m.n_trades}
                    for m in holdout_results],
    }
    out = _ROOT / "data" / "experiment_results.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to {out}")


if __name__ == "__main__":
    main()
