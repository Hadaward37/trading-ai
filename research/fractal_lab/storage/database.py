from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ..storage.models import ResearchReport

_DEFAULT_DB = Path("data/fractal_lab.db")


def _get_conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS reports (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            hypothesis    TEXT    NOT NULL,
            asset         TEXT    NOT NULL,
            timeframe     TEXT    NOT NULL,
            version       INTEGER DEFAULT 1,
            created_at    TEXT    NOT NULL,
            robustness    REAL    DEFAULT 0.0,
            verdict       TEXT    NOT NULL,
            payload       TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS hypotheses (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT NOT NULL,
            created_at TEXT NOT NULL,
            config     TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_reports_hypothesis ON reports(hypothesis);
        CREATE INDEX IF NOT EXISTS idx_reports_created_at ON reports(created_at);
    """)
    conn.commit()


class Database:
    """Simple SQLite persistence layer for research reports."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or _DEFAULT_DB
        self._conn = _get_conn(self.db_path)
        _init_schema(self._conn)

    def save_report(self, report: ResearchReport, version: int = 1) -> int:
        payload = json.dumps(asdict(report), default=str, ensure_ascii=True)
        cursor = self._conn.execute(
            """INSERT INTO reports
               (hypothesis, asset, timeframe, version, created_at, robustness, verdict, payload)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                report.hypothesis_name,
                report.asset,
                report.timeframe,
                version,
                report.created_at,
                report.robustness_score,
                report.verdict,
                payload,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    def load_report(self, report_id: int) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT payload FROM reports WHERE id = ?", (report_id,)
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["payload"])

    def list_reports(self, hypothesis: Optional[str] = None) -> List[dict]:
        if hypothesis:
            rows = self._conn.execute(
                "SELECT id, hypothesis, asset, timeframe, created_at, robustness, verdict "
                "FROM reports WHERE hypothesis = ? ORDER BY created_at DESC",
                (hypothesis,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, hypothesis, asset, timeframe, created_at, robustness, verdict "
                "FROM reports ORDER BY created_at DESC LIMIT 50"
            ).fetchall()
        return [dict(r) for r in rows]

    def save_hypothesis_config(self, name: str, config: dict) -> int:
        cursor = self._conn.execute(
            "INSERT INTO hypotheses (name, created_at, config) VALUES (?, ?, ?)",
            (name, datetime.utcnow().isoformat(), json.dumps(config, default=str)),
        )
        self._conn.commit()
        return cursor.lastrowid

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
