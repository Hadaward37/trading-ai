"""
Observer monitors — 5 pure-observation modules.

REGRA ABSOLUTA: todos os monitores APENAS observam, registram e emitem eventos.
Nenhum monitor bloqueia sinais, modifica parametros ou toma decisoes.

Causalidade: todos os calculos usam apenas dados disponiveis no momento
da observacao (sem look-ahead sobre trades ou precos futuros).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .events import Event, EventBus, EventType, Severity
from .baselines import (
    TRANSITION_RISKS, MARKOV_MATRIX, STEADY_STATE, REGIME_DURATION,
    CASCADE_KILL_SWITCH_N, CASCADE_BASELINE_P_LOSS, CASCADE_ELEVATED_P_LOSS,
    CASCADE_MIN_DEPTH, STRUCTURAL_RISKS,
    EDGE_HEALTH_WARNING_THRESHOLD, EDGE_HEALTH_CRITICAL_THRESHOLD,
    ROBUSTNESS_COLLAPSE_THRESHOLD,
    get_baseline, get_hypothesis_group, next_regime_probs,
)

# ── PF helpers (causal, inline) ────────────────────────────────────────────────

def _pf(pnls: np.ndarray) -> float:
    wins   = pnls[pnls > 0].sum()
    losses = abs(pnls[pnls <= 0].sum())
    return float(wins / losses) if losses > 1e-9 else (float("inf") if wins > 0 else 0.0)


def _rolling_pf(pnls: np.ndarray, window: int) -> Optional[float]:
    if len(pnls) < window:
        return None
    return _pf(pnls[-window:])


def _sharpe(pnls: np.ndarray, window: int) -> Optional[float]:
    if len(pnls) < window:
        return None
    w   = pnls[-window:]
    std = float(w.std())
    return float(w.mean() / std) if std > 1e-9 else 0.0


# ── 1. Regime Monitor ─────────────────────────────────────────────────────────

class RegimeMonitor:
    """
    Watches the regime sequence in df_prepared.
    Fires regime_change events when transitions occur.
    Compares current regime persistence against historical baseline.

    State: remembers previous regime across calls.
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus   = bus
        self._prev  = None      # previous regime label
        self._entry_bar = 0     # bar index when current regime started
        self._current_bar = 0

    def observe(
        self,
        df_prepared: pd.DataFrame,
        asset: str,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Observe current regime from the prepared DataFrame.
        Returns state dict: {current_regime, bars_in_regime, next_probs, ...}
        """
        if df_prepared.empty or "regime" not in df_prepared.columns:
            return {}

        self._current_bar = len(df_prepared)
        current = str(df_prepared["regime"].iloc[-1])
        ts = timestamp or datetime.now(timezone.utc).isoformat()

        # Count consecutive bars in current regime
        regimes = df_prepared["regime"].values[::-1]  # reverse for counting
        bars_in_regime = 1
        for r in regimes[1:]:
            if r == current:
                bars_in_regime += 1
            else:
                break

        # Transition detected
        if self._prev is not None and self._prev != current:
            self._fire_transition(self._prev, current, asset, bars_in_regime, ts)

        self._prev = current

        # Persistence check: is current regime unusually long?
        baseline = REGIME_DURATION.get(current, {})
        p95 = baseline.get("p95", 999)
        if bars_in_regime > p95 * 1.5:
            self._bus.publish(Event(
                event_type=EventType.STRUCTURAL_WARNING,
                severity=Severity.WARNING,
                asset=asset,
                regime=current,
                message=(
                    f"Regime {current} persiste ha {bars_in_regime} bars "
                    f"(baseline P95 = {p95:.0f}h). Anomalia de persistencia."
                ),
                data={"bars_in_regime": bars_in_regime, "p95_baseline": p95},
            ))

        # Compute next regime probabilities
        next_probs = next_regime_probs(current)

        return {
            "current_regime":   current,
            "bars_in_regime":   bars_in_regime,
            "prev_regime":      self._prev if self._prev != current else None,
            "next_probs":       next_probs,
            "steady_state_pct": STEADY_STATE.get(current, 0.0),
        }

    def _fire_transition(
        self,
        from_r: str,
        to_r: str,
        asset: str,
        bars_new: int,
        ts: str,
    ) -> None:
        risk = TRANSITION_RISKS.get((from_r, to_r))

        if risk:
            severity = Severity[risk["severity"]]
            delta    = risk["delta_pf"]
            message  = (
                f"Transicao {from_r} -> {to_r} | "
                f"Delta PF historico: {delta:+.2f} | "
                f"{risk['message']}"
            )
        else:
            severity = Severity.INFO
            delta    = None
            message  = f"Transicao de regime: {from_r} -> {to_r}"

        self._bus.publish(Event(
            event_type=EventType.REGIME_CHANGE,
            severity=severity,
            asset=asset,
            regime=to_r,
            message=message,
            timestamp=ts,
            data={
                "from_regime":    from_r,
                "to_regime":      to_r,
                "expected_pf_delta": delta,
                "bars_new_regime": bars_new,
                "next_probs":    next_regime_probs(to_r),
            },
        ))


# ── 2. Edge Health Monitor ────────────────────────────────────────────────────

class EdgeHealthMonitor:
    """
    Tracks rolling PF (30, 60, 90 trades) and compares to historical baseline.
    Fires edge_decay_alert when rolling PF drops significantly below baseline.

    Causal: only uses trades already closed (by entry_time order).
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus         = bus
        self._last_alerts: Dict[str, str] = {}   # key → last severity emitted

    def observe(
        self,
        pnls: np.ndarray,
        asset: str,
        hypothesis: str,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute rolling metrics and compare to baseline.
        pnls: chronologically-ordered PnL array (no future data).
        """
        baseline = get_baseline(asset, hypothesis)
        pf_base  = baseline["pf"]
        ts       = timestamp or datetime.now(timezone.utc).isoformat()
        key      = f"{asset}_{hypothesis}"

        state = {
            "n_trades":   len(pnls),
            "pf_baseline": round(pf_base, 3),
            "rolling":     {},
            "edge_health": "HEALTHY",
        }

        worst_severity = None

        for window in [30, 60, 90]:
            rpf = _rolling_pf(pnls, window)
            rsh = _sharpe(pnls, window)
            if rpf is None:
                continue

            state["rolling"][f"pf_{window}"] = round(rpf, 4)
            if rsh is not None:
                state["rolling"][f"sharpe_{window}"] = round(rsh, 4)

            ratio = rpf / max(pf_base, 1e-9)

            if ratio < EDGE_HEALTH_CRITICAL_THRESHOLD:
                sev = Severity.CRITICAL
                state["edge_health"] = "CRITICAL"
            elif ratio < EDGE_HEALTH_WARNING_THRESHOLD:
                sev = Severity.WARNING
                if state["edge_health"] == "HEALTHY":
                    state["edge_health"] = "DEGRADED"
            else:
                sev = None

            # Only fire if severity increased since last alert for this key+window
            last = self._last_alerts.get(f"{key}_{window}")
            if sev and sev.value != last:
                self._last_alerts[f"{key}_{window}"] = sev.value
                self._bus.publish(Event(
                    event_type=EventType.EDGE_DECAY_ALERT,
                    severity=sev,
                    asset=asset,
                    hypothesis=hypothesis,
                    message=(
                        f"Rolling PF ({window}T) = {rpf:.3f} vs baseline {pf_base:.3f} "
                        f"(ratio {ratio:.2%}) -- edge {state['edge_health'].lower()}"
                    ),
                    timestamp=ts,
                    data={
                        "window": window,
                        "rolling_pf": round(rpf, 4),
                        "baseline_pf": pf_base,
                        "ratio": round(ratio, 4),
                        "rolling_sharpe": round(rsh, 4) if rsh is not None else None,
                    },
                ))
                if worst_severity is None or sev == Severity.CRITICAL:
                    worst_severity = sev

        return state


# ── 3. Degradation Detector ───────────────────────────────────────────────────

class DegradationDetector:
    """
    Detects accelerating edge degradation and robustness collapse.

    Compares recent rolling PF (last 30T) vs medium-term (last 90T) to
    detect acceleration. Robustness proxy = PF consistency across windows.
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._prev_pf_30: Dict[str, float] = {}

    def observe(
        self,
        pnls: np.ndarray,
        asset: str,
        hypothesis: str,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        key = f"{asset}_{hypothesis}"
        ts  = timestamp or datetime.now(timezone.utc).isoformat()

        if len(pnls) < 30:
            return {"status": "INSUFFICIENT_DATA"}

        pf_30  = _rolling_pf(pnls, 30)
        pf_60  = _rolling_pf(pnls, 60)
        pf_90  = _rolling_pf(pnls, 90)

        # Robustness proxy: how consistent is PF across windows?
        avail = [p for p in [pf_30, pf_60, pf_90] if p is not None]
        if len(avail) >= 2:
            pf_std = float(np.std(avail))
            pf_mean = float(np.mean(avail))
            robustness_proxy = max(0.0, 1.0 - pf_std / max(abs(pf_mean), 1e-9))
        else:
            robustness_proxy = None

        state = {
            "pf_30":  round(pf_30,  4) if pf_30  else None,
            "pf_60":  round(pf_60,  4) if pf_60  else None,
            "pf_90":  round(pf_90,  4) if pf_90  else None,
            "robustness_proxy": round(robustness_proxy, 4) if robustness_proxy else None,
        }

        # Robustness collapse
        if robustness_proxy is not None and robustness_proxy < ROBUSTNESS_COLLAPSE_THRESHOLD:
            self._bus.publish(Event(
                event_type=EventType.STRUCTURAL_WARNING,
                severity=Severity.CRITICAL,
                asset=asset,
                hypothesis=hypothesis,
                message=(
                    f"Robustness collapse detectado: proxy={robustness_proxy:.3f} "
                    f"< threshold {ROBUSTNESS_COLLAPSE_THRESHOLD}. "
                    "PF muito inconsistente entre janelas — hipotese estruturalmente instavel."
                ),
                timestamp=ts,
                data={
                    "robustness_proxy": round(robustness_proxy, 4),
                    "pf_30": pf_30,
                    "pf_60": pf_60,
                    "pf_90": pf_90,
                },
            ))
            state["degradation_status"] = "COLLAPSE"
            return state

        # Accelerating decay: pf_30 < pf_60 < pf_90 (monotone decline)
        if pf_30 and pf_60 and pf_90:
            if pf_30 < pf_60 and pf_60 < pf_90:
                acceleration = pf_90 - pf_30  # positive number = how much it fell
                if acceleration > 0.15:
                    self._bus.publish(Event(
                        event_type=EventType.EDGE_DECAY_ALERT,
                        severity=Severity.WARNING,
                        asset=asset,
                        hypothesis=hypothesis,
                        message=(
                            f"Decaimento acelerado: PF90={pf_90:.3f} > PF60={pf_60:.3f} > PF30={pf_30:.3f}. "
                            f"Queda acumulada {acceleration:.3f} PF em {90-30} trades."
                        ),
                        timestamp=ts,
                        data={
                            "acceleration": round(acceleration, 4),
                            "pf_30": pf_30, "pf_60": pf_60, "pf_90": pf_90,
                        },
                    ))
                    state["degradation_status"] = "ACCELERATING_DECAY"
                    return state

        # OOS drift: compare recent 30T to global baseline
        baseline = get_baseline(asset, hypothesis)
        if pf_30 and pf_30 < baseline["pf"] * 0.60:
            state["degradation_status"] = "OOS_DRIFT"
        else:
            state["degradation_status"] = "NORMAL"

        return state


# ── 4. Failure Cascade Monitor ────────────────────────────────────────────────

class FailureCascadeMonitor:
    """
    Monitors consecutive loss sequences in real-time trade flow.

    Uses the CAUSAL trade sequence (sorted by entry_time).
    Compares conditional P(loss | N losses) against historical baseline ~0.62.

    Does NOT look at future trades to compute any metric.
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus  = bus
        self._prev_streak: Dict[str, int] = {}    # key → last known streak depth
        self._alerted_depths: Dict[str, int] = {} # key → last depth at which alert was fired

    def observe(
        self,
        wins: np.ndarray,   # 1=win, 0=loss, in chronological order
        asset: str,
        hypothesis: str,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Observe the current streak depth and conditional loss probability.
        wins: array of 1/0 in entry_time order (no future data).
        """
        key = f"{asset}_{hypothesis}"
        ts  = timestamp or datetime.now(timezone.utc).isoformat()

        if len(wins) < 5:
            return {"streak": 0, "status": "INSUFFICIENT_DATA"}

        # Current streak (count consecutive losses from the end)
        streak = 0
        for w in reversed(wins):
            if w == 0:
                streak += 1
            else:
                break

        # Compute conditional P(loss | N consecutive losses) using past data only
        cond_p = self._compute_cond_p(wins, streak) if streak > 0 else None

        state = {
            "current_streak": streak,
            "conditional_p_loss": round(cond_p, 4) if cond_p is not None else None,
            "cascade_risk": "NONE",
        }

        prev_streak = self._prev_streak.get(key, 0)
        last_alert  = self._alerted_depths.get(key, 0)

        if streak >= CASCADE_KILL_SWITCH_N and streak > last_alert:
            severity = Severity.CRITICAL
            self._alerted_depths[key] = streak
            msg = (
                f"CASCATA CRITICA: {streak} losses consecutivos. "
                f"Kill-switch threshold = N={CASCADE_KILL_SWITCH_N}. "
                f"P(proxima perda) estimada: {cond_p:.2%}. "
                "OBSERVACAO: Sistema 1 nao foi bloqueado — apenas alertando."
            )
            state["cascade_risk"] = "CRITICAL"
            self._bus.publish(Event(
                event_type=EventType.CASCADE_ALERT,
                severity=severity,
                asset=asset,
                hypothesis=hypothesis,
                message=msg,
                timestamp=ts,
                data={"streak": streak, "cond_p_loss": cond_p, "threshold": CASCADE_KILL_SWITCH_N},
            ))

        elif streak >= CASCADE_MIN_DEPTH and streak > last_alert:
            # Emit WARNING at defined breakpoints
            if streak % 2 == 0 or streak == CASCADE_MIN_DEPTH:
                self._alerted_depths[key] = streak
                sev = Severity.WARNING if streak >= 3 else Severity.INFO
                state["cascade_risk"] = "WARNING" if sev == Severity.WARNING else "WATCH"
                self._bus.publish(Event(
                    event_type=EventType.CASCADE_ALERT,
                    severity=sev,
                    asset=asset,
                    hypothesis=hypothesis,
                    message=(
                        f"Cascata: {streak} losses consecutivos "
                        f"(threshold={CASCADE_KILL_SWITCH_N}). "
                        f"P(loss|{streak}) = {cond_p:.2%} "
                        f"vs baseline {CASCADE_BASELINE_P_LOSS:.2%}."
                    ),
                    timestamp=ts,
                    data={"streak": streak, "cond_p_loss": cond_p},
                ))

        # Reset alert memory when streak breaks
        if streak == 0 and prev_streak >= CASCADE_KILL_SWITCH_N:
            self._alerted_depths.pop(key, None)
            self._bus.publish(Event(
                event_type=EventType.CASCADE_ALERT,
                severity=Severity.INFO,
                asset=asset,
                hypothesis=hypothesis,
                message=f"Cascata encerrada. Recuperacao apos {prev_streak} losses consecutivos.",
                timestamp=ts,
                data={"prev_streak": prev_streak},
            ))

        self._prev_streak[key] = streak
        return state

    @staticmethod
    def _compute_cond_p(wins: np.ndarray, streak_len: int) -> float:
        """
        Estimate P(next=loss | last streak_len were all losses).
        Causal: uses only past observations. Laplace smoothed.
        """
        n_after = 0
        n_loss  = 0
        for i in range(streak_len, len(wins)):
            window = wins[i - streak_len:i]
            if np.all(window == 0):
                n_after += 1
                if wins[i] == 0:
                    n_loss += 1
        # Laplace smoothing
        return (n_loss + 0.5) / (n_after + 1.0)


# ── 5. Structural Warning Engine ──────────────────────────────────────────────

class StructuralWarningEngine:
    """
    Crosses current regime × hypothesis type × recent transition
    to produce structured risk context per signal.

    Uses behavioral map findings from baselines.py.
    Does NOT modify signals — only produces risk_context dict.
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    def observe(
        self,
        current_regime: str,
        hypothesis: str,
        asset: str,
        recent_transition: Optional[Tuple[str, str]] = None,
        pf_30: Optional[float] = None,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Produce risk_context dict and fire structural warnings if needed.

        Returns a rich risk_context that can be attached to signals.
        """
        ts          = timestamp or datetime.now(timezone.utc).isoformat()
        hyp_group   = get_hypothesis_group(hypothesis)
        baseline    = get_baseline(asset, hypothesis)
        next_probs  = next_regime_probs(current_regime)
        notes       = []
        max_severity = Severity.INFO

        # Check structural risk for (hyp_group, regime)
        for key in [
            (hyp_group, current_regime),
            ("any",     current_regime),
        ]:
            risk = STRUCTURAL_RISKS.get(key)
            if risk:
                sev = Severity[risk["severity"]]
                notes.append(risk["message"])
                if sev.value == "CRITICAL" or (sev.value == "WARNING" and max_severity != Severity.CRITICAL):
                    max_severity = sev

        # Check if current transition is flagged
        if recent_transition:
            from_r, to_r = recent_transition
            t_risk = TRANSITION_RISKS.get((from_r, to_r))
            if t_risk and t_risk["severity"] in ("WARNING", "CRITICAL"):
                sev = Severity[t_risk["severity"]]
                notes.append(f"Transicao recente {from_r}->{to_r}: {t_risk['message']}")
                if sev == Severity.CRITICAL:
                    max_severity = Severity.CRITICAL

            # Special: LOW_VOL → HIGH_VOL is always CRITICAL
            if (from_r, to_r) == ("LOW_VOLATILITY", "HIGH_VOLATILITY"):
                special_risk = STRUCTURAL_RISKS.get(("any", "HIGH_VOLATILITY_AFTER_LOW"), {})
                if special_risk:
                    notes.append(special_risk["message"])
                    max_severity = Severity.CRITICAL

        # Edge health note
        if pf_30 is not None:
            pf_base = baseline["pf"]
            ratio   = pf_30 / max(pf_base, 1e-9)
            if ratio < EDGE_HEALTH_CRITICAL_THRESHOLD:
                notes.append(f"Edge critico: rolling PF30={pf_30:.3f} < 50% do baseline {pf_base:.3f}.")
                max_severity = Severity.CRITICAL
            elif ratio < EDGE_HEALTH_WARNING_THRESHOLD:
                notes.append(f"Edge degradado: rolling PF30={pf_30:.3f} vs baseline {pf_base:.3f}.")
                if max_severity == Severity.INFO:
                    max_severity = Severity.WARNING

        # Decay warning from known state
        if baseline.get("decay") == "DECAY":
            notes.append(f"Aviso de decay: {asset}x{hypothesis} mostrou decaimento -35%+ historicamente.")
            if max_severity == Severity.INFO:
                max_severity = Severity.WARNING

        # Fire event if actionable
        if max_severity != Severity.INFO or notes:
            self._bus.publish(Event(
                event_type=EventType.STRUCTURAL_WARNING,
                severity=max_severity,
                asset=asset,
                hypothesis=hypothesis,
                regime=current_regime,
                message=" | ".join(notes) if notes else f"Structural check: {current_regime} x {hyp_group}",
                timestamp=ts,
                data={
                    "regime":          current_regime,
                    "hyp_group":       hyp_group,
                    "transition":      list(recent_transition) if recent_transition else None,
                    "structural_notes": notes,
                },
            ))

        # Build risk context
        risk_context = {
            "current_regime":        current_regime,
            "recent_transition":     list(recent_transition) if recent_transition else None,
            "expected_pf_delta":     TRANSITION_RISKS.get(recent_transition, {}).get("delta_pf") if recent_transition else None,
            "rolling_pf_30":         round(pf_30, 4) if pf_30 is not None else None,
            "pf_baseline":           baseline["pf"],
            "edge_health":           "CRITICAL" if pf_30 and pf_30 < baseline["pf"] * 0.50
                                     else "DEGRADED" if pf_30 and pf_30 < baseline["pf"] * 0.75
                                     else "HEALTHY",
            "severity":              max_severity.value,
            "structural_notes":      notes,
            "next_regime_probs":     next_probs,
            "steady_state":          STEADY_STATE.get(current_regime, 0.0),
            "decay_status":          baseline.get("decay", "UNKNOWN"),
        }

        return risk_context
