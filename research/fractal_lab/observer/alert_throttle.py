"""
Alert Fatigue Control — deduplication, cooldown, and aggregation.

Previne spam operacional no Telegram sem suprimir alertas genuinos.

Regras:
  - Cooldown por severity: INFO=12H, WARNING=4H, CRITICAL=2H
  - Mesmo (asset, hypothesis, event_type, severity) nao repete dentro do cooldown
  - Bypass se severity aumentou OU metricas degradaram >15%
  - Aggregation: alertas similares do mesmo ciclo agrupados em 1 mensagem

NUNCA suprime alertas de severity CRITICA mais de uma vez (sempre reenvia se
o evento persiste no ciclo seguinte apos o cooldown CRITICAL de 2H).
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .events import Event, EventType, Severity

# ── Cooldown constants ─────────────────────────────────────────────────────────

COOLDOWN_SECONDS: Dict[str, int] = {
    Severity.INFO.value:     12 * 3600,   # 12H
    Severity.WARNING.value:   4 * 3600,   #  4H
    Severity.CRITICAL.value:  2 * 3600,   #  2H
}

_SEVERITY_ORDER: Dict[str, int] = {
    Severity.INFO.value: 0,
    Severity.WARNING.value: 1,
    Severity.CRITICAL.value: 2,
}

# Metric degradation threshold for bypass
METRIC_DEGRADATION_BYPASS = 0.15   # PF dropped >15% vs last alert → bypass cooldown


# ── Alert record ───────────────────────────────────────────────────────────────

@dataclass
class _AlertRecord:
    last_sent_ts: float          # unix timestamp of last sent alert
    last_severity: str           # severity at last send
    last_pf_metric: Optional[float] = None   # rolling PF at last send


# ── Alert Throttle ────────────────────────────────────────────────────────────

class AlertThrottle:
    """
    Tracks per-(asset, hypothesis, event_type, severity) cooldowns.
    State is in-memory (resets on restart — acceptable for continuous operation).
    """

    def __init__(self) -> None:
        self._records: Dict[str, _AlertRecord] = {}

    def _key(self, event: Event) -> str:
        hyp = event.hypothesis or "_"
        return f"{event.asset}|{hyp}|{event.event_type.value}|{event.severity.value}"

    def should_send(
        self,
        event: Event,
        current_pf: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Decide whether to send an event.

        Returns (should_send: bool, reason: str).
        """
        key    = self._key(event)
        now    = time.monotonic()
        record = self._records.get(key)

        # Never seen before — always send
        if record is None:
            self._records[key] = _AlertRecord(
                last_sent_ts=now,
                last_severity=event.severity.value,
                last_pf_metric=current_pf,
            )
            return True, "first_occurrence"

        elapsed  = now - record.last_sent_ts
        cooldown = COOLDOWN_SECONDS[event.severity.value]

        # Bypass #1: severity escalated
        if _SEVERITY_ORDER[event.severity.value] > _SEVERITY_ORDER[record.last_severity]:
            self._records[key] = _AlertRecord(
                last_sent_ts=now,
                last_severity=event.severity.value,
                last_pf_metric=current_pf,
            )
            return True, "severity_escalated"

        # Bypass #2: metrics degraded significantly
        if (current_pf is not None
                and record.last_pf_metric is not None
                and record.last_pf_metric > 0.0):
            degradation = (record.last_pf_metric - current_pf) / record.last_pf_metric
            if degradation > METRIC_DEGRADATION_BYPASS:
                self._records[key] = _AlertRecord(
                    last_sent_ts=now,
                    last_severity=event.severity.value,
                    last_pf_metric=current_pf,
                )
                return True, f"metric_degraded_{degradation:.0%}"

        # Cooldown elapsed
        if elapsed >= cooldown:
            self._records[key] = _AlertRecord(
                last_sent_ts=now,
                last_severity=event.severity.value,
                last_pf_metric=current_pf,
            )
            return True, "cooldown_elapsed"

        # Suppressed
        remaining_min = int((cooldown - elapsed) / 60)
        return False, f"suppressed_{remaining_min}min_remaining"

    def reset(self, asset: str, hypothesis: Optional[str] = None) -> None:
        """Clear cooldowns for a specific combination (e.g., after cascade recovery)."""
        prefix = f"{asset}|{hypothesis or '_'}|"
        self._records = {k: v for k, v in self._records.items() if not k.startswith(prefix)}

    def stats(self) -> Dict:
        """Return current throttle state summary."""
        total   = len(self._records)
        active  = sum(
            1 for r in self._records.values()
            if (time.monotonic() - r.last_sent_ts) < COOLDOWN_SECONDS[r.last_severity]
        )
        return {"total_tracked": total, "in_cooldown": active, "passed": total - active}


# ── Alert Aggregator ──────────────────────────────────────────────────────────

class AlertAggregator:
    """
    Collects events within a cycle and groups them by asset into single messages.

    Groups by: asset (primary) → hypothesis (secondary).
    Within each group, deduplicates by message similarity.
    Max 5 events per message to avoid excessively long Telegram messages.
    """

    _SEVERITY_EMOJI: Dict[str, str] = {
        Severity.INFO.value:     "ℹ️",
        Severity.WARNING.value:  "⚠️",
        Severity.CRITICAL.value: "🚨",
    }

    def __init__(self) -> None:
        self._pending: List[Event] = []

    def add(self, event: Event) -> None:
        """Add an event to the pending batch."""
        # Simple dedup: skip if identical message already pending for same asset
        for existing in self._pending:
            if (existing.asset == event.asset
                    and existing.event_type == event.event_type
                    and existing.message[:60] == event.message[:60]):
                return
        self._pending.append(event)

    def flush(self) -> List[str]:
        """
        Format pending events into grouped Telegram messages.
        Returns list of text strings ready to send. Clears pending.
        """
        if not self._pending:
            return []

        # Sort: CRITICAL first, then WARNING, then INFO
        severity_rank = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}
        self._pending.sort(key=lambda e: (severity_rank.get(e.severity, 3), e.asset))

        # Group by asset
        groups: Dict[str, List[Event]] = defaultdict(list)
        for event in self._pending:
            groups[event.asset].append(event)

        messages: List[str] = []
        for asset, events in groups.items():
            max_sev = min(events, key=lambda e: severity_rank.get(e.severity, 3)).severity
            emoji   = self._SEVERITY_EMOJI.get(max_sev.value, "📡")

            lines = [f"{emoji} *Observer [{max_sev.value}] — {asset}*"]

            seen_msgs = set()
            count = 0
            for e in events:
                short_msg = e.message[:90]
                if short_msg in seen_msgs:
                    continue
                seen_msgs.add(short_msg)

                ev_emoji = self._SEVERITY_EMOJI.get(e.severity.value, "•")
                hyp_tag  = f"[{e.hypothesis}] " if e.hypothesis else ""
                lines.append(f"{ev_emoji} {hyp_tag}{short_msg}")
                count += 1
                if count >= 5:
                    remaining = len(events) - count
                    if remaining > 0:
                        lines.append(f"  _...e mais {remaining} alertas_")
                    break

            messages.append("\n".join(lines))

        self._pending.clear()
        return messages

    def has_pending(self) -> bool:
        return bool(self._pending)

    def pending_count(self) -> int:
        return len(self._pending)

    def max_pending_severity(self) -> Optional[Severity]:
        if not self._pending:
            return None
        severity_rank = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}
        return min(self._pending, key=lambda e: severity_rank.get(e.severity, 3)).severity
