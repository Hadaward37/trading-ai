"""
Snapshot Engine + File Logger + Telegram Alerter.

SnapshotEngine   — persists timestamped JSON snapshots to data/observer_snapshots/
FileLogger       — appends events to date-partitioned log files
TelegramAlerter  — optional Telegram push for WARNING/CRITICAL events
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .events import Event, Severity

# ── Paths ──────────────────────────────────────────────────────────────────────

SNAPSHOTS_DIR = Path("data/observer_snapshots")
LOGS_DIR      = Path("data/observer_logs")
STATE_FILE    = Path("data/observer_state.json")
CONTEXT_FILE  = Path("data/observer_context.json")


# ── File Logger ────────────────────────────────────────────────────────────────

class FileLogger:
    """Append events to daily log files with structured JSON-per-line format."""

    def __init__(self, logs_dir: Optional[Path] = None) -> None:
        self._dir = logs_dir or LOGS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

        # Also configure Python logger
        self._py_logger = logging.getLogger("fractal_lab.observer")
        if not self._py_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
            self._py_logger.addHandler(handler)
        self._py_logger.setLevel(logging.INFO)

    def log(self, event: Event) -> None:
        """Write event to today's log file (JSON-per-line) and Python logger."""
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        log_file = self._dir / f"observer_{today}.log"

        line = json.dumps(event.to_dict(), ensure_ascii=True) + "\n"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception as exc:
            self._py_logger.error(f"Failed to write log: {exc}")

        # Mirror to Python logger with severity-appropriate level
        if event.severity == Severity.CRITICAL:
            self._py_logger.critical(str(event))
        elif event.severity == Severity.WARNING:
            self._py_logger.warning(str(event))
        else:
            self._py_logger.info(str(event))


# ── Snapshot Engine ────────────────────────────────────────────────────────────

class SnapshotEngine:
    """
    Persists timestamped JSON snapshots of the full observer state.

    Each snapshot contains: regime state, rolling metrics, edge health,
    risk context, cascade state, degradation state, and all recent events.

    These snapshots form the "context memory" of the observer:
    before / during / after each significant event.
    """

    def __init__(
        self,
        snapshots_dir: Optional[Path] = None,
        state_file: Optional[Path]    = None,
        context_file: Optional[Path]  = None,
    ) -> None:
        self._snap_dir    = snapshots_dir or SNAPSHOTS_DIR
        self._state_file  = state_file    or STATE_FILE
        self._context_file = context_file or CONTEXT_FILE
        self._snap_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        state: Dict[str, Any],
        recent_events: list,
        trigger: str = "periodic",
    ) -> Path:
        """
        Save a full snapshot and update observer_state.json.

        Args:
            state:         Current observer state dict.
            recent_events: List of recent Event objects (to_dict()).
            trigger:       What caused this snapshot (e.g. "CRITICAL_EVENT", "periodic").

        Returns:
            Path to saved snapshot file.
        """
        ts  = datetime.now(timezone.utc)
        key = ts.strftime("%Y%m%d_%H%M%S")

        snapshot = {
            "snapshot_id": key,
            "trigger":     trigger,
            "timestamp":   ts.isoformat(),
            "state":       _json_safe(state),
            "recent_events": [_json_safe(e) for e in recent_events[-30:]],
        }

        # Timestamped snapshot file
        snap_path = self._snap_dir / f"{key}.json"
        snap_path.write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

        # Always update observer_state.json (latest state only)
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        self._state_file.write_text(
            json.dumps({
                "last_updated": ts.isoformat(),
                "trigger":      trigger,
                **_json_safe(state),
            }, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

        return snap_path

    def save_context(self, risk_contexts: Dict[str, Dict]) -> None:
        """
        Write risk_context per (asset, hypothesis) to data/observer_context.json.
        Sistema 1 can read this file to get observer context without coupling.
        """
        payload = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "contexts":     _json_safe(risk_contexts),
        }
        self._context_file.parent.mkdir(parents=True, exist_ok=True)
        self._context_file.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    def list_snapshots(self, n: int = 10) -> list:
        """Return N most recent snapshot file paths."""
        snaps = sorted(self._snap_dir.glob("*.json"), reverse=True)
        return snaps[:n]


# ── Telegram Alerter ──────────────────────────────────────────────────────────

class TelegramAlerter:
    """
    Optional Telegram push for WARNING/CRITICAL events.

    Wraps core.notifier.TelegramNotifier. Gracefully disabled if
    notifier is unavailable or credentials are not set.

    ONLY sends messages — never modifies any system state.
    """

    _SEVERITY_EMOJI = {
        Severity.INFO:     "ℹ️",
        Severity.WARNING:  "⚠️",
        Severity.CRITICAL: "🚨",
    }

    def __init__(self) -> None:
        self._notifier = None
        self._enabled  = False
        self._min_severity = Severity.WARNING

        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        token   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

        if token and chat_id:
            try:
                import sys
                project_root = Path(__file__).resolve().parents[4]
                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))
                from core.notifier import TelegramNotifier
                self._notifier = TelegramNotifier(token=token, chat_id=chat_id)
                self._enabled  = self._notifier.is_configured
            except Exception:
                pass

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self._notifier is not None

    def alert(self, event: Event) -> bool:
        """Send event to Telegram. Returns True on success."""
        if not self.is_enabled:
            return False
        if event.severity == Severity.INFO:
            return False  # don't spam INFO to Telegram

        emoji = self._SEVERITY_EMOJI.get(event.severity, "📡")
        text  = self._format(event, emoji)
        try:
            return self._notifier.send_text(text)
        except Exception:
            return False

    @staticmethod
    def _format(event: Event, emoji: str) -> str:
        hyp  = f"\n*Hipotese:* `{event.hypothesis}`" if event.hypothesis else ""
        reg  = f"\n*Regime:*    `{event.regime}`" if event.regime else ""
        data = event.data

        lines = [
            f"{emoji} *[OBSERVER {event.severity.value}]*",
            f"*Ativo:* `{event.asset}`{hyp}{reg}",
            f"*Evento:* `{event.event_type.value}`",
            "",
            event.message,
        ]

        # Append key data fields
        if "expected_pf_delta" in data and data["expected_pf_delta"] is not None:
            lines.append(f"\n_Delta PF esperado: {data['expected_pf_delta']:+.2f}_")
        if "streak" in data:
            lines.append(f"_Losses consecutivos: {data['streak']}_")
        if "rolling_pf" in data:
            lines.append(f"_Rolling PF: {data['rolling_pf']:.3f}_")

        lines.append(f"\n`{event.timestamp}`")
        return "\n".join(lines)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to JSON-serializable Python types."""
    import math
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if hasattr(obj, "to_dict"):        # DataFrames
        return _json_safe(obj.to_dict("records"))
    if hasattr(obj, "item"):           # numpy scalars
        val = obj.item()
        return None if isinstance(val, float) and not math.isfinite(val) else val
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj
