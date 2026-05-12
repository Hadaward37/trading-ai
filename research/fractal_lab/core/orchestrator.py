from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from ..storage.models import Hypothesis, ResearchReport
from ..storage.database import Database
from ..metrics.robustness import compute_robustness
from .pipeline import Pipeline


class Orchestrator:
    """
    Central coordinator for the Fractal Lab research engine.

    Accepts a Hypothesis and a pre-loaded OHLCV DataFrame,
    runs the full research pipeline, and returns a structured ResearchReport.

    Usage
    -----
    from research.fractal_lab.core.orchestrator import Orchestrator
    from research.fractal_lab.research.hypothesis import build_hypothesis

    hypothesis = build_hypothesis(
        name="MR_ATR_Filter",
        asset="EURUSD",
        timeframe="1H",
        entry_logic={"type": "mean_reversion", "threshold": 1.0},
        exit_logic={"type": "fixed_rr", "sl_atr_multiplier": 1.0, "tp_atr_multiplier": 2.0},
        features=["atr", "ema_distance", "atr_ratio"],
        test_period=("2024-09-01", "2025-08-31"),
        oos_period=("2025-09-01", "2026-04-30"),
    )

    report = Orchestrator().run_hypothesis(df, hypothesis)
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        n_wf_folds: int = 3,
        n_mc_simulations: int = 1000,
        persist: bool = True,
        verbose: bool = True,
    ):
        self._pipeline = Pipeline(n_wf_folds=n_wf_folds, n_mc_simulations=n_mc_simulations)
        self._db_path = db_path
        self._persist = persist
        self._verbose = verbose

    def _log(self, msg: str) -> None:
        if self._verbose:
            print(msg)

    def run_hypothesis(
        self,
        df: pd.DataFrame,
        hypothesis: Hypothesis,
    ) -> ResearchReport:
        """
        Full research pipeline for one hypothesis.

        Parameters
        ----------
        df:
            Raw OHLCV DataFrame with DatetimeIndex (UTC).
            Columns: Open, High, Low, Close (title or lowercase).
        hypothesis:
            Validated Hypothesis object (use build_hypothesis() to create).

        Returns
        -------
        ResearchReport with train, test, OOS results, walk-forward, MC, and robustness.
        """
        self._log(f"[Orchestrator] Starting hypothesis: {hypothesis.name}")

        # ── Step 1: feature + regime computation ──────────────────────────────
        self._log("[Orchestrator] Step 1/5 — computing features and regimes...")
        df_prepared = self._pipeline.prepare(df)

        # ── Step 2: backtest on train, test, OOS ──────────────────────────────
        self._log("[Orchestrator] Step 2/5 — running backtests (train / test / OOS)...")
        test_start, test_end = hypothesis.test_period
        oos_start, oos_end = hypothesis.oos_period

        # Use everything before test period as "train" (OOS-safe split)
        df_idx = df_prepared.index
        train_start = str(df_idx[0])
        ts_test_start = pd.Timestamp(test_start)
        if df_idx.tz is not None and ts_test_start.tz is None:
            ts_test_start = ts_test_start.tz_localize("UTC")
        train_end = str(ts_test_start - pd.Timedelta(hours=1))

        train_result = self._pipeline.run_period(
            df_prepared, hypothesis, train_start, train_end, "train"
        )
        test_result = self._pipeline.run_period(
            df_prepared, hypothesis, test_start, test_end, "test"
        )
        oos_result = self._pipeline.run_period(
            df_prepared, hypothesis, oos_start, oos_end, "oos"
        )

        self._log(
            f"[Orchestrator]   train={train_result.n_trades} trades | "
            f"test={test_result.n_trades} trades | "
            f"oos={oos_result.n_trades} trades"
        )

        # ── Step 3: walk-forward on test period ───────────────────────────────
        self._log("[Orchestrator] Step 3/5 — walk-forward validation...")
        wf_results = self._pipeline.run_walk_forward(
            df_prepared, hypothesis, test_start, test_end
        )

        # ── Step 4: Monte Carlo on test period trades ─────────────────────────
        self._log("[Orchestrator] Step 4/5 — Monte Carlo simulation...")
        mc = self._pipeline.run_monte_carlo(test_result.trades)

        # ── Step 5: robustness scoring ────────────────────────────────────────
        self._log("[Orchestrator] Step 5/5 — robustness scoring...")
        robustness = compute_robustness(wf_results, mc)
        robustness_score = robustness["robustness_score"]

        # ── Verdict ───────────────────────────────────────────────────────────
        verdict = _generate_verdict(test_result, oos_result, robustness)

        # ── Assemble report ───────────────────────────────────────────────────
        report = ResearchReport(
            hypothesis_name=hypothesis.name,
            asset=hypothesis.asset,
            timeframe=hypothesis.timeframe,
            created_at=datetime.utcnow().isoformat(),
            train=train_result,
            test=test_result,
            oos=oos_result,
            walk_forward=wf_results,
            monte_carlo={**mc, **robustness},
            robustness_score=robustness_score,
            verdict=verdict,
        )

        # ── Persist ───────────────────────────────────────────────────────────
        if self._persist:
            try:
                with Database(self._db_path) as db:
                    report_id = db.save_report(report, version=hypothesis.version)
                self._log(f"[Orchestrator] Report saved -> id={report_id}")
            except Exception as exc:
                self._log(f"[Orchestrator] Warning: could not persist report: {exc}")

        self._log(f"[Orchestrator] Done. Verdict: {verdict}")
        return report


# ── Verdict logic ──────────────────────────────────────────────────────────────

def _generate_verdict(test_result, oos_result, robustness: dict) -> str:
    """
    Produce a single-line research verdict.

    Rules (order matters — first match wins):
      PASS       — test PF > 1.3, OOS PF > 1.0, robustness >= 0.6
      OOS_PASS   — test PF > 1.0, OOS PF > 1.0 (partial confirmation)
      DEGRADED   — test OK but OOS PF < 1.0 (overfit signal)
      WEAK_EDGE  — test PF 1.0–1.3, mixed OOS
      FAIL       — test PF < 1.0
      NO_TRADES  — no trades generated
    """
    n_test = test_result.n_trades
    n_oos = oos_result.n_trades

    if n_test == 0:
        return "NO_TRADES"

    pf_test = test_result.metrics.get("profit_factor", 0.0)
    pf_oos = oos_result.metrics.get("profit_factor", 0.0) if n_oos > 0 else float("nan")
    rob = robustness.get("robustness_score", 0.0)

    if pf_test > 1.3 and pf_oos > 1.0 and rob >= 0.6:
        return "PASS"
    if pf_test > 1.0 and pf_oos > 1.0:
        return "OOS_PASS"
    if pf_test > 1.0 and (pf_oos < 1.0 or n_oos == 0):
        return "DEGRADED"
    if 1.0 <= pf_test <= 1.3:
        return "WEAK_EDGE"
    return "FAIL"
