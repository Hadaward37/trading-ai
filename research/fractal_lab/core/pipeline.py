from __future__ import annotations

from typing import Optional

import pandas as pd

from ..features.base_features import compute_features
from ..regimes.regime_detector import tag_regimes
from ..storage.models import Hypothesis, PeriodResult
from ..metrics.performance import compute_metrics, compute_regime_breakdown
from ..validation.backtest_engine import BacktestEngine
from ..validation.walk_forward import WalkForwardValidator
from ..validation.monte_carlo import run_monte_carlo


class Pipeline:
    """
    Stateless pipeline that connects all research components.

    Steps (in order):
      1. compute_features   — add ATR, EMA, volatility, etc.
      2. tag_regimes        — add regime label per bar
      3. backtest           — generate trades for train / test / OOS periods
      4. walk_forward       — temporal fold validation
      5. monte_carlo        — bootstrap robustness simulation
    """

    def __init__(self, n_wf_folds: int = 3, n_mc_simulations: int = 1000):
        self.n_wf_folds = n_wf_folds
        self.n_mc_simulations = n_mc_simulations
        self._engine = BacktestEngine()
        self._wf = WalkForwardValidator(n_folds=n_wf_folds)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features + regime labels to raw OHLCV DataFrame."""
        df_feat = compute_features(df)
        df_regime = tag_regimes(df_feat)
        return df_regime

    def run_period(
        self,
        df: pd.DataFrame,
        hypothesis: Hypothesis,
        start: str,
        end: str,
        period_type: str,
    ) -> PeriodResult:
        """Run backtest on a single time period and return structured result."""
        trades = self._engine.run(df, hypothesis, period_start=start, period_end=end)
        metrics = compute_metrics(trades)
        regime_bd = compute_regime_breakdown(trades)
        return PeriodResult(
            period_type=period_type,
            start=start,
            end=end,
            n_trades=len(trades),
            metrics=metrics,
            trades=trades,
            regime_breakdown=regime_bd,
        )

    def run_walk_forward(
        self,
        df: pd.DataFrame,
        hypothesis: Hypothesis,
        start: str,
        end: str,
    ):
        """Run walk-forward validation over the test period."""
        ts_start = pd.Timestamp(start)
        ts_end = pd.Timestamp(end)
        if df.index.tz is not None:
            if ts_start.tz is None:
                ts_start = ts_start.tz_localize("UTC")
            if ts_end.tz is None:
                ts_end = ts_end.tz_localize("UTC")
        df_slice = df.loc[(df.index >= ts_start) & (df.index <= ts_end)]
        return self._wf.run(df_slice, hypothesis)

    def run_monte_carlo(self, trades) -> dict:
        return run_monte_carlo(trades, n_simulations=self.n_mc_simulations)
