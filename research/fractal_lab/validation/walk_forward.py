from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from ..storage.models import Hypothesis, PeriodResult
from ..metrics.performance import compute_metrics, compute_regime_breakdown
from .backtest_engine import BacktestEngine


def _split_temporal(
    df: pd.DataFrame,
    n_folds: int,
    test_ratio: float,
) -> List[Tuple[str, str, str, str]]:
    """
    Divide the df index into (train_start, train_end, test_start, test_end) folds.

    Uses an expanding training window anchored at the start.
    Each fold's test period is fixed at test_ratio of the total period.
    No random splits — order is always preserved.
    """
    total_bars = len(df)
    if total_bars < 20:
        return []

    fold_size = total_bars // n_folds
    test_size = max(1, int(fold_size * test_ratio))
    train_size = fold_size - test_size

    if train_size < 5:
        return []

    folds = []
    for i in range(n_folds):
        fold_start = i * fold_size
        fold_train_end = fold_start + train_size - 1
        fold_test_start = fold_train_end + 1
        fold_test_end = min(fold_start + fold_size - 1, total_bars - 1)

        if fold_test_end <= fold_test_start:
            continue

        train_start = str(df.index[0])
        train_end = str(df.index[fold_train_end])
        test_start = str(df.index[fold_test_start])
        test_end = str(df.index[fold_test_end])

        folds.append((train_start, train_end, test_start, test_end))

    return folds


class WalkForwardValidator:
    """
    Walk-forward validation with expanding training window.

    Splits data into N sequential folds. For each fold:
      1. Trains/validates on all data up to fold boundary
      2. Tests on the fold's OOS slice

    Preserves temporal order — no random splits.
    """

    def __init__(self, n_folds: int = 3, test_ratio: float = 0.4):
        self.n_folds = n_folds
        self.test_ratio = test_ratio
        self._engine = BacktestEngine()

    def run(self, df: pd.DataFrame, hypothesis: Hypothesis) -> List[PeriodResult]:
        folds = _split_temporal(df, self.n_folds, self.test_ratio)
        results: List[PeriodResult] = []

        for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds):
            trades = self._engine.run(
                df=df,
                hypothesis=hypothesis,
                period_start=test_start,
                period_end=test_end,
            )
            metrics = compute_metrics(trades)
            regime_bd = compute_regime_breakdown(trades)

            results.append(PeriodResult(
                period_type=f"wf_fold_{fold_idx + 1}",
                start=test_start,
                end=test_end,
                n_trades=len(trades),
                metrics=metrics,
                trades=trades,
                regime_breakdown=regime_bd,
            ))

        return results
