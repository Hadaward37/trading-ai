"""
Fractal Lab Observer Agent.

REGRA ABSOLUTA: apenas observa, registra e alerta.
Nunca bloqueia, decide, otimiza ou recalibra.
"""
from .agent  import ObserverAgent
from .events import Event, EventBus, EventType, Severity

__all__ = ["ObserverAgent", "Event", "EventBus", "EventType", "Severity"]
