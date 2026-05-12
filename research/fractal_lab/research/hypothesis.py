from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..storage.models import Hypothesis

VALID_ENTRY_TYPES = [
    "feature_threshold",
    "multi_condition",
    "mean_reversion",
    "trend_follow",
    "crossover",
    "breakout",
    "oscillator_extremes",
    "ema_slope",
    "momentum",
    "momentum_atr",
    "volatility_breakout",
    "range_breakout",
]

VALID_EXIT_TYPES = ["fixed_rr", "fixed_pips"]

VALID_REGIMES = [
    "TRENDING",
    "MEAN_REVERTING",
    "HIGH_VOLATILITY",
    "LOW_VOLATILITY",
    None,
]


def build_hypothesis(
    name: str,
    asset: str,
    timeframe: str,
    entry_logic: Dict[str, Any],
    exit_logic: Dict[str, Any],
    features: List[str],
    test_period: Tuple[str, str],
    oos_period: Tuple[str, str],
    regime_filter: Optional[str] = None,
    description: str = "",
    version: int = 1,
    pip_multiplier: float = 10000.0,
) -> Hypothesis:
    """Build and validate a Hypothesis. Raises ValueError on invalid config."""
    errors: List[str] = []

    if not name.strip():
        errors.append("name cannot be empty")

    if entry_logic.get("type") not in VALID_ENTRY_TYPES:
        errors.append(
            f"entry_logic.type must be one of {VALID_ENTRY_TYPES}, "
            f"got '{entry_logic.get('type')}'"
        )

    if exit_logic.get("type") not in VALID_EXIT_TYPES:
        errors.append(
            f"exit_logic.type must be one of {VALID_EXIT_TYPES}, "
            f"got '{exit_logic.get('type')}'"
        )

    if regime_filter not in VALID_REGIMES:
        errors.append(f"regime_filter must be one of {VALID_REGIMES}")

    if len(test_period) != 2:
        errors.append("test_period must be a (start, end) tuple")

    if len(oos_period) != 2:
        errors.append("oos_period must be a (start, end) tuple")

    if errors:
        raise ValueError(
            "Hypothesis validation failed:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    return Hypothesis(
        name=name,
        asset=asset,
        timeframe=timeframe,
        entry_logic=entry_logic,
        exit_logic=exit_logic,
        features=features,
        test_period=test_period,
        oos_period=oos_period,
        regime_filter=regime_filter,
        description=description,
        version=version,
        pip_multiplier=pip_multiplier,
    )
