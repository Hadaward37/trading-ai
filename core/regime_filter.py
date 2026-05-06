"""Regime filter — post-processing gate for trading signals.

Logic:
  Trade ONLY when:
    ADX(t) < adx_threshold
    AND
    atr_ratio_min < atr_ratio(t) < atr_ratio_max

This is a PURE post-processing filter — the underlying model is never
retrained. It masks out signals in regimes where the model historically
underperforms (strong-trend / abnormal-volatility environments).

Usage:
  rf   = RegimeFilter(adx_threshold=30, atr_ratio_min=0.8, atr_ratio_max=1.5)
  mask = rf.mask(df)         # pd.Series[bool]
  filtered_preds = preds[mask]
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RegimeFilter:
    """Binary regime gate based on ADX and ATR ratio."""

    adx_threshold:  float = 30.0   # block trades when ADX >= this
    atr_ratio_min:  float = 0.8    # block when atr_ratio <= this
    atr_ratio_max:  float = 1.5    # block when atr_ratio >= this

    # ── Public API ─────────────────────────────────────────────────────────────

    def mask(self, df) -> "pd.Series":
        """Return boolean Series — True where trading IS allowed."""
        import pandas as pd
        return (
            (df["adx"]       <  self.adx_threshold) &
            (df["atr_ratio"] >  self.atr_ratio_min) &
            (df["atr_ratio"] <  self.atr_ratio_max)
        ).astype(bool)

    def retention(self, df) -> float:
        """Fraction of rows where filter is active (0-1)."""
        m = self.mask(df)
        return float(m.mean()) if len(m) > 0 else 0.0

    def describe(self) -> str:
        return (
            f"ADX<{self.adx_threshold} | "
            f"atr_ratio in ({self.atr_ratio_min}, {self.atr_ratio_max})"
        )

    # ── Convenience factory ────────────────────────────────────────────────────

    @classmethod
    def no_filter(cls) -> "RegimeFilter":
        """Degenerate filter that never blocks (all trades pass)."""
        return cls(adx_threshold=999, atr_ratio_min=0.0, atr_ratio_max=999)


# ── Module-level helpers for parameter grid ───────────────────────────────────

ADX_THRESHOLDS   = [20, 25, 30, 35]
ATR_RATIO_RANGES = [(0.7, 1.3), (0.8, 1.5), (0.9, 1.8)]


def parameter_grid() -> list[RegimeFilter]:
    """Return all 12 filter candidates ordered from most permissive to strictest.

    'Most permissive' = highest ADX threshold + widest ATR range.
    This ordering supports the 'choose simplest valid filter' selection strategy.
    """
    candidates = []
    for adx in sorted(ADX_THRESHOLDS, reverse=True):  # most permissive first
        for (atr_min, atr_max) in sorted(
            ATR_RATIO_RANGES,
            key=lambda x: x[1] - x[0],
            reverse=True,  # widest range first
        ):
            candidates.append(RegimeFilter(adx, atr_min, atr_max))
    return candidates
