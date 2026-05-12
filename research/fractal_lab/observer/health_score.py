"""
System Health Score — observational telemetry macro do estado estrutural.

Score 0-100 composto de 5 componentes:
  regime_risk         25% — risco do regime atual + transicao recente
  rolling_pf          25% — rolling PF vs baseline (janelas 30/60/90T)
  robustness          20% — robustness proxy (consistencia entre janelas)
  cascade_state       15% — profundidade da cascata atual
  structural_warnings 15% — severidade e count de alertas estruturais

IMPORTANTE: score e PURAMENTE OBSERVACIONAL.
Nao influencia, nao bloqueia e nao altera o Sistema 1.

O score deve ser interpretavel:
  90-100: EXCELLENT  — sistema dentro dos parametros historicos
  75-89:  GOOD       — pequenas anomalias, edge mantido
  60-74:  MODERATE   — degradacao detectada, monitorar
  40-59:  POOR       — edge em risco, revisar combinacoes
   0-39:  CRITICAL   — multiplas degradacoes, analise necessaria
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .baselines import TRANSITION_RISKS

# ── Score interpretation ───────────────────────────────────────────────────────

HEALTH_BANDS = [
    (90, 100, "EXCELLENT", "Sistema dentro dos parametros historicos"),
    (75,  89, "GOOD",      "Pequenas anomalias, edge mantido"),
    (60,  74, "MODERATE",  "Degradacao detectada — monitorar"),
    (40,  59, "POOR",      "Edge em risco estrutural — revisar"),
    (  0, 39, "CRITICAL",  "Multiplas degradacoes — analise necessaria"),
]


def _band(score: int) -> tuple:
    for lo, hi, label, desc in HEALTH_BANDS:
        if lo <= score <= hi:
            return label, desc
    return "UNKNOWN", ""


# ── Component scorers ──────────────────────────────────────────────────────────

def _score_regime_risk(
    current_regime: str,
    recent_transition: Optional[tuple],
) -> Dict[str, Any]:
    """
    Regime risk component (0 to -25).

    Regime base penalties:
      LOW_VOLATILITY  → -8  (precursor historico de HIGH_VOL crash)
      HIGH_VOLATILITY → -5  (elevated, edge volatil)
      MEAN_REVERTING  → -2  (edge fraco para maioria dos ativos)
      TRENDING        →  0  (regime dominante, melhor edge medio)

    Transition penalty (adicional):
      CRITICAL transition → -12
      WARNING transition  → -6
      INFO transition     → -2
    """
    regime_base = {
        "LOW_VOLATILITY":  -8,
        "HIGH_VOLATILITY": -5,
        "MEAN_REVERTING":  -2,
        "TRENDING":         0,
    }
    base = regime_base.get(current_regime, -3)
    notes = [f"Regime: {current_regime} ({'+' if base == 0 else ''}{base} pts)"]

    trans_penalty = 0
    if recent_transition:
        from_r, to_r = recent_transition
        risk = TRANSITION_RISKS.get((from_r, to_r))
        if risk:
            if risk["severity"] == "CRITICAL":
                trans_penalty = -12
            elif risk["severity"] == "WARNING":
                trans_penalty = -6
            else:
                trans_penalty = -2
            notes.append(f"Transicao {from_r}->{to_r}: {trans_penalty} pts")

    raw = max(-25, base + trans_penalty)
    return {"deduction": raw, "notes": notes}


def _score_rolling_pf(rolling: Dict[str, float], baseline_pf: float) -> Dict[str, Any]:
    """
    Rolling PF component (0 to -25).
    Uses average ratio across available windows.
    """
    if not rolling or baseline_pf <= 0:
        return {"deduction": -8, "notes": ["No rolling PF data"]}

    pf_values = [v for k, v in rolling.items() if k.startswith("pf_")]
    if not pf_values:
        return {"deduction": -8, "notes": ["No rolling PF data"]}

    avg_ratio = sum(v / max(baseline_pf, 1e-9) for v in pf_values) / len(pf_values)
    notes = [f"Avg rolling PF ratio: {avg_ratio:.2%} (baseline={baseline_pf:.3f})"]

    if avg_ratio >= 1.00:
        deduction = 0
        notes.append("Edge acima do baseline")
    elif avg_ratio >= 0.90:
        deduction = -4
        notes.append("Edge levemente abaixo do baseline")
    elif avg_ratio >= 0.75:
        deduction = -10
        notes.append("Edge moderadamente degradado")
    elif avg_ratio >= 0.50:
        deduction = -18
        notes.append("Edge significativamente degradado")
    else:
        deduction = -25
        notes.append("Edge critico (< 50% do baseline)")

    return {"deduction": max(-25, deduction), "notes": notes}


def _score_robustness(robustness_proxy: Optional[float]) -> Dict[str, Any]:
    """
    Robustness proxy component (0 to -20).
    """
    if robustness_proxy is None:
        return {"deduction": -5, "notes": ["Robustness: insufficient data"]}

    notes = [f"Robustness proxy: {robustness_proxy:.3f}"]

    if robustness_proxy >= 0.75:
        deduction = 0
        notes.append("PF consistente entre janelas")
    elif robustness_proxy >= 0.55:
        deduction = -4
        notes.append("Leve inconsistencia entre janelas")
    elif robustness_proxy >= 0.35:
        deduction = -10
        notes.append("Inconsistencia moderada")
    elif robustness_proxy >= 0.20:
        deduction = -15
        notes.append("Robustness fraca")
    else:
        deduction = -20
        notes.append("Robustness collapse detectado")

    return {"deduction": max(-20, deduction), "notes": notes}


def _score_cascade(cascade_streak: int) -> Dict[str, Any]:
    """
    Cascade state component (0 to -15).
    """
    notes = [f"Cascata atual: {cascade_streak} losses consecutivos"]

    if cascade_streak == 0:
        deduction = 0
    elif cascade_streak <= 2:
        deduction = -3
        notes.append("Cascata leve")
    elif cascade_streak <= 4:
        deduction = -8
        notes.append("Cascata moderada")
    elif cascade_streak < 5:
        deduction = -12
        notes.append("Proximo ao threshold de kill-switch (N=5)")
    else:
        deduction = -15
        notes.append("CASCATA CRITICA — threshold de kill-switch atingido")

    return {"deduction": max(-15, deduction), "notes": notes}


def _score_structural(structural_notes: List[str], severity: str) -> Dict[str, Any]:
    """
    Structural warnings component (0 to -15).
    """
    n      = len(structural_notes)
    notes_ = [f"Alertas estruturais: {n} ({severity})"]

    if n == 0:
        deduction = 0
    elif severity == "CRITICAL":
        deduction = -15
        notes_.append("Aviso estrutural CRITICO ativo")
    elif severity == "WARNING" and n >= 2:
        deduction = -10
        notes_.append(f"{n} avisos estruturais ativos")
    elif severity == "WARNING":
        deduction = -5
        notes_.append("1 aviso estrutural ativo")
    else:
        deduction = -2
        notes_.append(f"{n} notas informativas")

    return {"deduction": max(-15, deduction), "notes": notes_}


# ── Main health score ──────────────────────────────────────────────────────────

@dataclass
class HealthScore:
    score: int
    label: str
    description: str
    components: Dict[str, int]
    component_notes: Dict[str, List[str]]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "health_score":      self.score,
            "label":             self.label,
            "description":       self.description,
            "components":        self.components,
            "component_notes":   self.component_notes,
            "timestamp":         self.timestamp,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=True)

    def format_telegram(self, asset: Optional[str] = None) -> str:
        """Format a concise Telegram representation."""
        label_emoji = {
            "EXCELLENT": "💚",
            "GOOD":      "🟢",
            "MODERATE":  "🟡",
            "POOR":      "🔴",
            "CRITICAL":  "🚨",
        }
        emoji  = label_emoji.get(self.label, "📊")
        header = f"Saúde: {self.score}/100 ({self.label})"
        if asset:
            header = f"{asset}: {header}"

        bar_filled = int(self.score / 10)
        bar = "█" * bar_filled + "░" * (10 - bar_filled)

        lines = [
            f"{emoji} *{header}*",
            f"`[{bar}]`",
            f"_{self.description}_",
            "",
            "*Componentes:*",
        ]
        component_names = {
            "regime_risk":          "Regime",
            "rolling_pf":           "Rolling PF",
            "robustness":           "Robustness",
            "cascade_state":        "Cascade",
            "structural_warnings":  "Structural",
        }
        for key, pts in self.components.items():
            name  = component_names.get(key, key)
            sign  = f"{'+' if pts >= 0 else ''}{pts}"
            lines.append(f"  • {name}: `{sign} pts`")

        return "\n".join(lines)


def compute_health_score(
    current_regime: str,
    recent_transition: Optional[tuple],
    rolling: Dict[str, float],
    baseline_pf: float,
    robustness_proxy: Optional[float],
    cascade_streak: int,
    structural_notes: List[str],
    severity: str,
) -> HealthScore:
    """
    Compute System Health Score (0-100) from 5 components.

    All inputs are observational — no decisions, no blocking.
    """
    r1 = _score_regime_risk(current_regime, recent_transition)
    r2 = _score_rolling_pf(rolling, baseline_pf)
    r3 = _score_robustness(robustness_proxy)
    r4 = _score_cascade(cascade_streak)
    r5 = _score_structural(structural_notes, severity)

    total_deduction = r1["deduction"] + r2["deduction"] + r3["deduction"] + r4["deduction"] + r5["deduction"]
    raw_score       = max(0, min(100, 100 + total_deduction))
    label, desc     = _band(raw_score)

    return HealthScore(
        score       = raw_score,
        label       = label,
        description = desc,
        components  = {
            "regime_risk":         r1["deduction"],
            "rolling_pf":          r2["deduction"],
            "robustness":          r3["deduction"],
            "cascade_state":       r4["deduction"],
            "structural_warnings": r5["deduction"],
        },
        component_notes = {
            "regime_risk":         r1["notes"],
            "rolling_pf":          r2["notes"],
            "robustness":          r3["notes"],
            "cascade_state":       r4["notes"],
            "structural_warnings": r5["notes"],
        },
    )


# ── Aggregate health score across multiple combinations ────────────────────────

def compute_aggregate_health(combination_scores: List[HealthScore]) -> HealthScore:
    """
    Aggregate health scores across multiple (asset, hypothesis) combinations.

    Strategy:
      - Report average score as system-level health
      - Component deductions = mean across combinations
    """
    if not combination_scores:
        label, desc = _band(100)
        return HealthScore(
            score=100, label=label, description=desc,
            components={}, component_notes={},
        )

    avg_score = int(round(sum(h.score for h in combination_scores) / len(combination_scores)))
    label, desc = _band(avg_score)

    # Average component deductions
    all_keys = set()
    for h in combination_scores:
        all_keys.update(h.components.keys())

    avg_components = {}
    avg_notes      = {}
    for key in all_keys:
        vals = [h.components[key] for h in combination_scores if key in h.components]
        avg_components[key] = int(round(sum(vals) / len(vals))) if vals else 0
        avg_notes[key] = [
            note
            for h in combination_scores
            for note in h.component_notes.get(key, [])
        ][:3]  # max 3 notes per component

    return HealthScore(
        score=avg_score,
        label=label,
        description=desc,
        components=avg_components,
        component_notes=avg_notes,
    )


def save_health_score(health: HealthScore, path: Optional[Path] = None) -> None:
    """Persist health score to data/observer_health.json."""
    target = path or Path("data/observer_health.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(health.to_json(), encoding="utf-8")
