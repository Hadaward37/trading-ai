from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Hypothesis:
    name: str
    asset: str
    timeframe: str
    entry_logic: Dict[str, Any]
    exit_logic: Dict[str, Any]
    features: List[str]
    test_period: Tuple[str, str]
    oos_period: Tuple[str, str]
    regime_filter: Optional[str] = None
    description: str = ""
    version: int = 1
    pip_multiplier: float = 10000.0


@dataclass
class Trade:
    entry_time: str
    exit_time: str
    direction: int          # 1 = long, -1 = short
    entry_price: float
    exit_price: float
    pnl_pips: float
    outcome: str            # TP_HIT | SL_HIT | TIMEOUT
    bars_held: int
    tp: float
    sl: float
    regime: Optional[str] = None


@dataclass
class PeriodResult:
    period_type: str        # train | test | oos | wf_fold_N
    start: str
    end: str
    n_trades: int
    metrics: Dict[str, float]
    trades: List[Trade] = field(default_factory=list)
    regime_breakdown: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class ResearchReport:
    hypothesis_name: str
    asset: str
    timeframe: str
    created_at: str
    train: Optional[PeriodResult] = None
    test: Optional[PeriodResult] = None
    oos: Optional[PeriodResult] = None
    walk_forward: List[PeriodResult] = field(default_factory=list)
    monte_carlo: Dict[str, Any] = field(default_factory=dict)
    robustness_score: float = 0.0
    verdict: str = ""
