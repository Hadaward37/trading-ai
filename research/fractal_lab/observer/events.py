"""
Typed events for the Observer Agent event bus.

All events are immutable records. The EventBus dispatches them
to registered handlers. Handlers NEVER block or modify system state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class Severity(str, Enum):
    INFO     = "INFO"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"


class EventType(str, Enum):
    REGIME_CHANGE      = "regime_change"
    EDGE_DECAY_ALERT   = "edge_decay_alert"
    CASCADE_ALERT      = "cascade_alert"
    STRUCTURAL_WARNING = "structural_warning"
    HEALTH_SNAPSHOT    = "health_snapshot"


@dataclass(frozen=True)
class Event:
    """Immutable event record. Fired by monitors, consumed by handlers."""

    event_type:  EventType
    severity:    Severity
    asset:       str
    message:     str
    timestamp:   str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    hypothesis:  Optional[str] = None
    regime:      Optional[str] = None
    data:        Dict[str, Any] = field(default_factory=dict)

    # Severity helpers
    @property
    def is_critical(self) -> bool:
        return self.severity == Severity.CRITICAL

    @property
    def is_warning_or_above(self) -> bool:
        return self.severity in (Severity.WARNING, Severity.CRITICAL)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "severity":   self.severity.value,
            "asset":      self.asset,
            "hypothesis": self.hypothesis,
            "regime":     self.regime,
            "message":    self.message,
            "timestamp":  self.timestamp,
            "data":       self.data,
        }

    def __str__(self) -> str:
        hyp = f" [{self.hypothesis}]" if self.hypothesis else ""
        return f"[{self.severity.value}][{self.event_type.value}] {self.asset}{hyp}: {self.message}"


Handler = Callable[[Event], None]


class EventBus:
    """
    Simple synchronous pub-sub event bus.

    Handlers run in registration order. Exceptions in one handler
    are caught and logged so they never suppress subsequent handlers.
    """

    def __init__(self) -> None:
        self._handlers: List[Handler] = []
        self._history:  List[Event]   = []
        self._max_history = 500

    def subscribe(self, handler: Handler) -> None:
        self._handlers.append(handler)

    def publish(self, event: Event) -> None:
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        for handler in self._handlers:
            try:
                handler(event)
            except Exception as exc:
                import sys
                print(f"[EventBus] Handler error: {exc}", file=sys.stderr)

    def recent(self, n: int = 50) -> List[Event]:
        return self._history[-n:]

    def recent_by_severity(self, severity: Severity, n: int = 20) -> List[Event]:
        return [e for e in self._history if e.severity == severity][-n:]

    def clear_history(self) -> None:
        self._history.clear()
