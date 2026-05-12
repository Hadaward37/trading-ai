"""
Behavioral Map Baselines — constants derived from the full batch analysis.

Source: research/results/behavioral_map/ (run 2026-05-12)
These values represent empirical findings from 80 combinations across 10 assets.

DO NOT EDIT without re-running the behavioral map analysis.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

# ── Regime transition risks ────────────────────────────────────────────────────
# Derived from per_transition_stats.csv (aggregate across 12 best combinations)
# delta_pf = pf_post - pf_pre  (negative = degradation)

TRANSITION_RISKS: Dict[Tuple[str, str], Dict] = {
    # CRITICAL transitions — severe PF collapse expected
    ("LOW_VOLATILITY",  "HIGH_VOLATILITY"):  {
        "delta_pf":  -4.22,
        "severity":  "CRITICAL",
        "message":   "Historicamente a transicao mais destrutiva: PF esperado -4.22. Aguardar estabilizacao.",
    },
    ("MEAN_REVERTING",  "HIGH_VOLATILITY"):  {
        "delta_pf":  -3.54,
        "severity":  "CRITICAL",
        "message":   "Transicao MEAN_REV->HIGH_VOL: colapso de PF esperado (-3.54). Estrategias MR em risco.",
    },
    # WARNING transitions — meaningful degradation
    ("HIGH_VOLATILITY", "LOW_VOLATILITY"):   {
        "delta_pf":  -1.22,
        "severity":  "WARNING",
        "message":   "HIGH_VOL->LOW_VOL: PF tende a cair (-1.22). Estrategias de volatilidade perdem edge.",
    },
    ("HIGH_VOLATILITY", "MEAN_REVERTING"):   {
        "delta_pf":  -0.93,
        "severity":  "WARNING",
        "message":   "HIGH_VOL->MEAN_REV: edge moderadamente reduzido (-0.93).",
    },
    ("LOW_VOLATILITY",  "MEAN_REVERTING"):   {
        "delta_pf":  -1.81,
        "severity":  "WARNING",
        "message":   "LOW_VOL->MEAN_REV: PF esperado -1.81. MR strategies may struggle in regime shift.",
    },
    ("LOW_VOLATILITY",  "TRENDING"):         {
        "delta_pf":  -1.28,
        "severity":  "WARNING",
        "message":   "LOW_VOL->TRENDING: PF esperado -1.28. TF strategies need recalibration lag.",
    },
    ("TRENDING",        "LOW_VOLATILITY"):   {
        "delta_pf":  -0.87,
        "severity":  "WARNING",
        "message":   "TRENDING->LOW_VOL: volatilidade comprimida, sinal de reversao ou pausa.",
    },
    # INFO transitions — minor effect
    ("TRENDING",        "HIGH_VOLATILITY"):  {
        "delta_pf":  -0.50,
        "severity":  "INFO",
        "message":   "TRENDING->HIGH_VOL: leve reducao de PF esperada (-0.50). Monitorar.",
    },
    ("TRENDING",        "MEAN_REVERTING"):   {
        "delta_pf":  +0.10,
        "severity":  "INFO",
        "message":   "TRENDING->MEAN_REV: neutro/leve melhora para estrategias MR.",
    },
    ("HIGH_VOLATILITY", "TRENDING"):         {
        "delta_pf":  +0.20,
        "severity":  "INFO",
        "message":   "HIGH_VOL->TRENDING: recuperacao tipica. TF strategies benefit.",
    },
    ("MEAN_REVERTING",  "TRENDING"):         {
        "delta_pf":  +0.05,
        "severity":  "INFO",
        "message":   "MEAN_REV->TRENDING: neutro.",
    },
    ("MEAN_REVERTING",  "LOW_VOLATILITY"):   {
        "delta_pf":  -0.30,
        "severity":  "INFO",
        "message":   "MEAN_REV->LOW_VOL: leve reducao de edge.",
    },
}

# ── Markov Transition Matrix ───────────────────────────────────────────────────
# P[TO | FROM] — empirical aggregate across 6 assets
# Derived from markov_matrix.csv

MARKOV_MATRIX: Dict[str, Dict[str, float]] = {
    "TRENDING":        {"TRENDING": 0.0,   "MEAN_REVERTING": 0.093, "HIGH_VOLATILITY": 0.696, "LOW_VOLATILITY": 0.211},
    "MEAN_REVERTING":  {"TRENDING": 0.488, "MEAN_REVERTING": 0.0,   "HIGH_VOLATILITY": 0.389, "LOW_VOLATILITY": 0.123},
    "HIGH_VOLATILITY": {"TRENDING": 0.770, "MEAN_REVERTING": 0.102, "HIGH_VOLATILITY": 0.0,   "LOW_VOLATILITY": 0.128},
    "LOW_VOLATILITY":  {"TRENDING": 0.575, "MEAN_REVERTING": 0.066, "HIGH_VOLATILITY": 0.359, "LOW_VOLATILITY": 0.0},
}

# ── Steady-state regime distribution ──────────────────────────────────────────
# Long-run probability of being in each regime (eigendecomposition of Markov matrix)

STEADY_STATE: Dict[str, float] = {
    "TRENDING":        0.406,
    "HIGH_VOLATILITY": 0.367,
    "LOW_VOLATILITY":  0.143,
    "MEAN_REVERTING":  0.084,
}

# ── Regime duration baselines (1H bars) ───────────────────────────────────────

REGIME_DURATION: Dict[str, Dict[str, float]] = {
    "TRENDING":        {"mean": 8.4,  "median": 5.0, "p95": 26.7},
    "MEAN_REVERTING":  {"mean": 5.4,  "median": 4.1, "p95": 14.0},
    "HIGH_VOLATILITY": {"mean": 6.7,  "median": 3.8, "p95": 20.7},
    "LOW_VOLATILITY":  {"mean": 4.0,  "median": 2.9, "p95": 10.4},
}

# ── Cascade thresholds ────────────────────────────────────────────────────────

CASCADE_KILL_SWITCH_N    = 5      # pause after this many consecutive losses
CASCADE_BASELINE_P_LOSS  = 0.625  # historical conditional loss probability (stable ~0.62)
CASCADE_ELEVATED_P_LOSS  = 0.65   # P(loss) above this = elevated risk
CASCADE_MIN_DEPTH        = 2      # min depth to classify as cascade

# ── Per-combination PF baselines ──────────────────────────────────────────────
# Derived from batch results (latest run). Format: (asset, hypothesis) → metrics

HYPOTHESIS_BASELINES: Dict[Tuple[str, str], Dict] = {
    ("BTC",    "MR_RSI_Exhaust"):  {"pf": 1.274, "rob": 0.890, "decay": "STABLE"},
    ("EURUSD", "MR_RSI_Exhaust"):  {"pf": 1.372, "rob": 0.828, "decay": "IMPROVING"},
    ("DXY",    "MR_EMA_Dist"):     {"pf": 1.166, "rob": 0.847, "decay": "STABLE"},
    ("DXY",    "MR_RSI_Exhaust"):  {"pf": 1.199, "rob": 0.844, "decay": "STABLE"},
    ("GOLD",   "TF_Momentum"):     {"pf": 1.219, "rob": 0.870, "decay": "STABLE"},
    ("EURUSD", "MR_BB_Stretch"):   {"pf": 1.241, "rob": 0.715, "decay": "STABLE"},
    ("EURUSD", "MR_EMA_Dist"):     {"pf": 1.241, "rob": 0.762, "decay": "IMPROVING"},
    ("NASDAQ", "VOL_ATR_Expand"):  {"pf": 1.888, "rob": 0.876, "decay": "DECAY"},
    ("SPY",    "MR_BB_Stretch"):   {"pf": 1.260, "rob": 0.474, "decay": "IMPROVING"},
    ("ITUB4",  "MR_RSI_Exhaust"):  {"pf": 1.091, "rob": 0.558, "decay": "DECAY"},
    ("BTC",    "MR_EMA_Dist"):     {"pf": 1.132, "rob": 0.499, "decay": "STABLE"},
    ("PETR4",  "TF_EMA_Slope"):    {"pf": 1.249, "rob": 0.483, "decay": "STABLE"},
    ("GOLD",   "TF_EMA_Slope"):    {"pf": 1.157, "rob": 0.478, "decay": "IMPROVING"},
}

# Fallback baseline when combination is unknown
DEFAULT_BASELINE = {"pf": 1.10, "rob": 0.35, "decay": "UNKNOWN"}

# ── Structural risks by (hypothesis_group, regime) ────────────────────────────
# Based on behavioral market map findings

STRUCTURAL_RISKS: Dict[Tuple[str, str], Dict] = {
    # When a mean-reversion hypothesis runs in HIGH_VOLATILITY
    ("mean_reversion", "HIGH_VOLATILITY"): {
        "severity": "WARNING",
        "message": "MR em HIGH_VOL: PF historicamente volatil (1.0-1.7 dependendo do ativo). Aguardar confirmacao de regime.",
    },
    # Trend-following in MEAN_REVERTING (usually weak)
    ("trend_follow", "MEAN_REVERTING"): {
        "severity": "WARNING",
        "message": "TF em MEAN_REVERTING: edge nulo ou negativo para maioria dos ativos. Excecao: GOLD TF_Momentum (PF 2.6).",
    },
    # Volatility strategies in LOW_VOLATILITY (no signal / signal drought)
    ("volatility", "LOW_VOLATILITY"): {
        "severity": "WARNING",
        "message": "Estrategia de volatilidade em LOW_VOL: frequencia de sinal muito baixa. Robustez nao confiavel.",
    },
    # Critical: any strategy at LOW_VOL -> HIGH_VOL transition
    ("any", "HIGH_VOLATILITY_AFTER_LOW"): {
        "severity": "CRITICAL",
        "message": "Entrando em HIGH_VOL vindo de LOW_VOL: pior transicao historica (delta PF -4.22). Risco estrutural elevado.",
    },
    # Volatility strategies in HIGH_VOLATILITY (too many trades, spread risk)
    ("volatility", "HIGH_VOLATILITY"): {
        "severity": "INFO",
        "message": "VOL em HIGH_VOL: edge presente mas susceptivel a spread real. Monitorar execucao.",
    },
}

# ── Edge health thresholds ────────────────────────────────────────────────────

EDGE_HEALTH_WARNING_THRESHOLD  = 0.75  # rolling PF below 75% of baseline → WARNING
EDGE_HEALTH_CRITICAL_THRESHOLD = 0.50  # rolling PF below 50% of baseline → CRITICAL
ROBUSTNESS_COLLAPSE_THRESHOLD  = 0.20  # robustness below this → CRITICAL

# Hypothesis group mapping
HYPOTHESIS_GROUPS: Dict[str, str] = {
    "MR_BB_Stretch":  "mean_reversion",
    "MR_EMA_Dist":    "mean_reversion",
    "MR_RSI_Exhaust": "mean_reversion",
    "TF_EMA_Slope":   "trend_follow",
    "TF_Momentum":    "trend_follow",
    "TF_Breakout_20": "trend_follow",
    "VOL_ATR_Expand": "volatility",
    "VOL_Range_Bkout": "volatility",
}


def get_baseline(asset: str, hypothesis: str) -> Dict:
    """Return baseline metrics for a combination, with fallback."""
    return HYPOTHESIS_BASELINES.get((asset, hypothesis), DEFAULT_BASELINE)


def get_hypothesis_group(hypothesis: str) -> str:
    return HYPOTHESIS_GROUPS.get(hypothesis, "unknown")


def next_regime_probs(current_regime: str) -> Dict[str, float]:
    """Markov-estimated probabilities for the next regime given current."""
    return MARKOV_MATRIX.get(current_regime, {r: 0.25 for r in MARKOV_MATRIX})
