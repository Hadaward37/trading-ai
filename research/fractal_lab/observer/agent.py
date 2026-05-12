"""
Fractal Lab Observer Agent.

REGRA ABSOLUTA: este agente APENAS observa, registra e alerta.
Nao toma decisoes, nao bloqueia sinais, nao modifica parametros,
nao recalibra modelos, nao interfere com o Sistema 1.

Arquitetura:
  EventBus                 -- pub-sub de eventos tipados
  5 Monitors               -- observam metricas especificas e publicam eventos
  SnapshotEngine           -- persiste estado timestamped em data/observer_snapshots/
  FileLogger + TelegramAlerter -- handlers de eventos

Integracao com Sistema 1 (read-only):
  Le: data/mt5_signals.json, data/trading.db
  Escreve: data/observer_state.json, data/observer_context.json,
           data/observer_snapshots/, data/observer_logs/

Usage:
  # Rodar uma observacao completa:
  agent = ObserverAgent()
  agent.run_once(asset="EURUSD", hypothesis="MR_RSI_Exhaust")

  # Loop continuo (1H entre ciclos):
  agent.run_loop(interval_seconds=3600)

  # Via CLI:
  python -X utf8 -m research.fractal_lab.observer.agent
  python -X utf8 -m research.fractal_lab.observer.agent --asset EURUSD --hypothesis MR_RSI_Exhaust
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Project root setup ─────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Internal imports ───────────────────────────────────────────────────────────

from .events   import Event, EventBus, EventType, Severity
from .monitors import (
    RegimeMonitor,
    EdgeHealthMonitor,
    DegradationDetector,
    FailureCascadeMonitor,
    StructuralWarningEngine,
)
from .snapshots import FileLogger, SnapshotEngine, TelegramAlerter
from .baselines import (
    HYPOTHESIS_BASELINES, HYPOTHESIS_GROUPS,
    get_baseline, get_hypothesis_group, next_regime_probs,
)
from .alert_throttle import AlertThrottle, AlertAggregator
from .health_score   import (
    HealthScore, compute_health_score, compute_aggregate_health, save_health_score
)

# ── Constants ──────────────────────────────────────────────────────────────────

LOOKAHEAD_WINDOW = 0   # enforced: 0 future bars used in any calculation

DEFAULT_ASSETS_HYPOTHESES: List[Tuple[str, str]] = [
    ("EURUSD", "MR_RSI_Exhaust"),
    ("BTC",    "MR_RSI_Exhaust"),
    ("DXY",    "MR_EMA_Dist"),
    ("GOLD",   "TF_Momentum"),
]

# ── Data readers ───────────────────────────────────────────────────────────────

def _load_ohlcv(asset: str) -> pd.DataFrame:
    """Load 1H OHLCV — cache first, then yfinance."""
    from research.fractal_lab.analytics._shared import load_asset_data, ASSETS
    if asset not in ASSETS:
        return pd.DataFrame()
    return load_asset_data(asset)


def _prepare_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features + regimes. Strictly causal (no lookahead)."""
    if df.empty:
        return df
    from research.fractal_lab.features.base_features import compute_features
    from research.fractal_lab.regimes.regime_detector import tag_regimes
    return tag_regimes(compute_features(df))


def _load_sistema1_trades() -> List[Dict]:
    """
    Read recent paper trades from Sistema 1 (read-only).
    Tries: data/trading.db → data/mt5_signals.json → empty list.
    """
    # Attempt trading.db (paper trading)
    db_path = Path("data/trading.db")
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            # Try typical paper-trading table structure
            for table in ("paper_trades", "trades", "signals"):
                try:
                    rows = conn.execute(
                        f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT 500"
                    ).fetchall()
                    if rows:
                        conn.close()
                        return [dict(r) for r in rows]
                except sqlite3.OperationalError:
                    continue
            conn.close()
        except Exception:
            pass

    # Attempt mt5_signals.json
    sig_file = Path("data/mt5_signals.json")
    if sig_file.exists():
        try:
            return json.loads(sig_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    return []


def _extract_pnls_for(asset: str, hypothesis: Optional[str], trades_raw: List[Dict]) -> np.ndarray:
    """
    Extract chronologically-ordered PnL array from Sistema 1 trade records.
    Filters by asset (symbol). Returns empty array if no matching trades.
    """
    if not trades_raw:
        return np.array([])

    matching = []
    asset_variants = {asset, asset.replace("/", ""), asset + "=X", asset + ".SA"}

    for t in trades_raw:
        sym = str(t.get("symbol", t.get("asset", t.get("pair", "")))).upper()
        if sym and any(v.upper() in sym or sym in v.upper() for v in asset_variants):
            pnl = t.get("pnl_pips", t.get("pnl", t.get("profit", None)))
            if pnl is not None:
                try:
                    matching.append(float(pnl))
                except (ValueError, TypeError):
                    pass

    return np.array(matching, dtype=float)


def _extract_wins_for(asset: str, hypothesis: Optional[str], trades_raw: List[Dict]) -> np.ndarray:
    """Extract binary win/loss array (1=win, 0=loss) from Sistema 1 trades."""
    pnls = _extract_pnls_for(asset, hypothesis, trades_raw)
    if len(pnls) == 0:
        return np.array([], dtype=int)
    return (pnls > 0).astype(int)


# ── Observer Agent ─────────────────────────────────────────────────────────────

class ObserverAgent:
    """
    Main Observer Agent.

    Coordinates all 5 monitors via the EventBus.
    Persists state, snapshots, and logs for every observation cycle.
    Sends Telegram alerts for WARNING/CRITICAL events (if configured).

    NEVER blocks Sistema 1 execution. NEVER modifies any parameter.
    """

    def __init__(
        self,
        assets_hypotheses: Optional[List[Tuple[str, str]]] = None,
        snapshot_on_critical: bool = True,
    ) -> None:
        self.targets = assets_hypotheses or list(HYPOTHESIS_BASELINES.keys())

        # Event bus
        self.bus = EventBus()

        # Monitors (maintain internal state across calls)
        self._regime_monitors: Dict[str, RegimeMonitor] = {}
        self._edge_monitors   = EdgeHealthMonitor(self.bus)
        self._degrad_detector = DegradationDetector(self.bus)
        self._cascade_monitor = FailureCascadeMonitor(self.bus)
        self._struct_engine   = StructuralWarningEngine(self.bus)

        # Output handlers
        self._logger    = FileLogger()
        self._snapshots = SnapshotEngine()
        self._telegram  = TelegramAlerter()
        self._snap_on_critical = snapshot_on_critical

        # Alert fatigue control
        self._throttle   = AlertThrottle()
        self._aggregator = AlertAggregator()

        # Health scores (latest per combination + aggregate)
        self._health_scores: Dict[str, HealthScore] = {}
        self._aggregate_health: Optional[HealthScore] = None

        # 6H summary tracking
        self._last_summary_ts: float = 0.0
        self._summary_interval = 6 * 3600   # 6 hours

        # Internal state
        self._state: Dict[str, Any] = {
            "started_at":   datetime.now(timezone.utc).isoformat(),
            "n_cycles":     0,
            "n_events":     0,
            "n_criticals":  0,
            "n_warnings":   0,
            "n_suppressed": 0,
            "last_cycle":   None,
            "health_score": None,
            "assets":       {},
        }
        self._risk_contexts: Dict[str, Dict] = {}

        # Subscribe handlers — throttle wrapper
        self.bus.subscribe(self._on_event)

    # ── Public API ─────────────────────────────────────────────────────────────

    def run_once(
        self,
        asset: Optional[str] = None,
        hypothesis: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run one complete observation cycle.
        If asset/hypothesis are specified, observe only that combination.
        Otherwise, observe all registered targets.
        """
        ts = datetime.now(timezone.utc).isoformat()
        targets = [(asset, hypothesis)] if asset else self.targets

        # Load Sistema 1 trades once (shared across all combinations)
        sistema1_trades = _load_sistema1_trades()

        cycle_state: Dict[str, Any] = {
            "timestamp":          ts,
            "n_targets":          len(targets),
            "n_sistema1_trades":  len(sistema1_trades),
            "combinations":       {},
        }

        for asset_key, hyp_name in targets:
            combo_state = self._observe_combination(
                asset_key, hyp_name, sistema1_trades, ts
            )
            cycle_state["combinations"][f"{asset_key}_{hyp_name}"] = combo_state
            self._risk_contexts[f"{asset_key}_{hyp_name}"] = (
                combo_state.get("risk_context", {})
            )

        # Update agent state
        self._state["n_cycles"]    += 1
        self._state["last_cycle"]   = ts
        self._state["assets"]       = cycle_state["combinations"]
        self._state["n_events"]     = len(self.bus.recent())

        # ── Compute health scores ──────────────────────────────────────────────
        combo_scores = []
        for combo_key, combo_state in cycle_state["combinations"].items():
            ctx      = combo_state.get("risk_context", {})
            edge     = combo_state.get("edge_health", {})
            degrad   = combo_state.get("degradation", {})
            cascade  = combo_state.get("cascade", {})
            regime   = combo_state.get("regime", {})
            baseline = get_baseline(
                combo_state.get("asset", ""),
                combo_state.get("hypothesis") or "",
            )

            hs = compute_health_score(
                current_regime     = ctx.get("current_regime", "TRENDING"),
                recent_transition  = tuple(ctx["recent_transition"]) if ctx.get("recent_transition") else None,
                rolling            = edge.get("rolling", {}),
                baseline_pf        = baseline.get("pf", 1.10),
                robustness_proxy   = degrad.get("robustness_proxy"),
                cascade_streak     = cascade.get("current_streak", 0),
                structural_notes   = ctx.get("structural_notes", []),
                severity           = ctx.get("severity", "INFO"),
            )
            self._health_scores[combo_key] = hs
            combo_scores.append(hs)

        self._aggregate_health = compute_aggregate_health(combo_scores)
        save_health_score(self._aggregate_health)
        self._state["health_score"] = self._aggregate_health.score

        # ── Flush aggregated alerts to Telegram ────────────────────────────────
        if self._aggregator.has_pending():
            for msg in self._aggregator.flush():
                self._telegram._send_raw(msg)

        # ── Periodic snapshot ──────────────────────────────────────────────────
        snap_path = self._snapshots.save(
            state={
                **self._state,
                "health": self._aggregate_health.to_dict() if self._aggregate_health else {},
                "throttle": self._throttle.stats(),
            },
            recent_events=[e.to_dict() for e in self.bus.recent(30)],
            trigger="periodic",
        )

        # ── Publish risk context for Sistema 1 (read-only) ────────────────────
        self._snapshots.save_context(self._risk_contexts)

        # ── 6H summary ────────────────────────────────────────────────────────
        if (time.monotonic() - self._last_summary_ts) >= self._summary_interval:
            self._send_6h_summary()
            self._last_summary_ts = time.monotonic()

        print(
            f"[Observer] Cycle #{self._state['n_cycles']} | "
            f"Health={self._aggregate_health.score if self._aggregate_health else '?'}/100 "
            f"({self._aggregate_health.label if self._aggregate_health else '?'}) | "
            f"Criticals={self._state['n_criticals']} "
            f"Warnings={self._state['n_warnings']} "
            f"Suppressed={self._state['n_suppressed']} | "
            f"Snap={snap_path.name}"
        )

        return cycle_state

    def run_loop(self, interval_seconds: int = 3600) -> None:
        """
        Continuous observation loop.
        Runs run_once() every interval_seconds.
        Runs until interrupted (Ctrl+C).
        """
        print(f"[Observer] Starting loop (interval={interval_seconds}s). Press Ctrl+C to stop.")
        while True:
            try:
                self.run_once()
                print(f"[Observer] Sleeping {interval_seconds}s until next cycle...")
                time.sleep(interval_seconds)
            except KeyboardInterrupt:
                print("\n[Observer] Loop interrupted by user.")
                break
            except Exception as exc:
                print(f"[Observer] Cycle error: {exc}. Continuing after {interval_seconds}s...")
                time.sleep(interval_seconds)

    # ── Internal: observation cycle per combination ────────────────────────────

    def _observe_combination(
        self,
        asset: str,
        hypothesis: Optional[str],
        sistema1_trades: List[Dict],
        ts: str,
    ) -> Dict[str, Any]:
        """
        Full observation cycle for one (asset, hypothesis) combination.

        Steps (all causal):
          1. Load OHLCV + compute features + regimes
          2. Observe current regime and detect transitions
          3. Load Sistema 1 trade history (PnL array)
          4. Edge health monitor (rolling PF vs baseline)
          5. Degradation detector (accelerating decay + robustness proxy)
          6. Cascade monitor (consecutive losses)
          7. Structural warning engine (cross-module risk synthesis)
          8. Snapshot if CRITICAL event fired
        """
        state: Dict[str, Any] = {"asset": asset, "hypothesis": hypothesis}

        # ── Step 1: OHLCV + regimes ────────────────────────────────────────────
        df_raw   = _load_ohlcv(asset)
        df_prep  = _prepare_ohlcv(df_raw) if not df_raw.empty else pd.DataFrame()

        # ── Step 2: Regime monitor ─────────────────────────────────────────────
        key = asset
        if key not in self._regime_monitors:
            self._regime_monitors[key] = RegimeMonitor(self.bus)

        regime_state = self._regime_monitors[key].observe(df_prep, asset, ts)
        current_regime = regime_state.get("current_regime", "UNKNOWN")
        state["regime"] = regime_state

        # Detect the most recent regime transition
        recent_transition: Optional[Tuple[str, str]] = None
        if (regime_state.get("prev_regime") and
                regime_state["prev_regime"] != current_regime):
            recent_transition = (regime_state["prev_regime"], current_regime)

        # ── Step 3: Trade history ──────────────────────────────────────────────
        pnls = _extract_pnls_for(asset, hypothesis, sistema1_trades)
        wins = (pnls > 0).astype(int) if len(pnls) > 0 else np.array([], dtype=int)

        # Also generate synthetic pnls from lab backtests if Sistema 1 has no data
        if len(pnls) < 30 and hypothesis:
            pnls, wins = self._load_lab_trades(asset, hypothesis, df_prep)

        state["n_trades"] = len(pnls)

        # ── Step 4: Edge health monitor ────────────────────────────────────────
        edge_state = self._edge_monitors.observe(pnls, asset, hypothesis or "", ts)
        state["edge_health"] = edge_state

        # ── Step 5: Degradation detector ──────────────────────────────────────
        degrad_state = self._degrad_detector.observe(pnls, asset, hypothesis or "", ts)
        state["degradation"] = degrad_state

        # ── Step 6: Cascade monitor ────────────────────────────────────────────
        cascade_state = self._cascade_monitor.observe(wins, asset, hypothesis or "", ts)
        state["cascade"] = cascade_state

        # ── Step 7: Structural warning engine ─────────────────────────────────
        pf_30 = edge_state.get("rolling", {}).get("pf_30")
        risk_context = self._struct_engine.observe(
            current_regime=current_regime,
            hypothesis=hypothesis or "",
            asset=asset,
            recent_transition=recent_transition,
            pf_30=pf_30,
            timestamp=ts,
        )
        state["risk_context"] = risk_context

        return state

    def _load_lab_trades(
        self,
        asset: str,
        hypothesis: str,
        df_prep: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fallback: re-run backtest via the lab engine to get trade history.
        Causal: uses only data that exists at observation time.
        """
        if df_prep.empty:
            return np.array([]), np.array([], dtype=int)
        try:
            from research.fractal_lab.analytics._shared import (
                run_backtest, TRAIN_PERIOD, OOS_PERIOD,
            )
            trades = run_backtest(df_prep, asset, hypothesis)
            if not trades:
                return np.array([]), np.array([], dtype=int)
            pnls = np.array([t.pnl_pips for t in trades], dtype=float)
            wins = (pnls > 0).astype(int)
            return pnls, wins
        except Exception:
            return np.array([]), np.array([], dtype=int)

    # ── Event handler ──────────────────────────────────────────────────────────

    def _on_event(self, event: Event) -> None:
        """
        Master event handler with Alert Fatigue Control.

        Pipeline:
          1. Log ALL events to file (no suppression)
          2. Throttle check — suppress duplicates within cooldown window
          3. If passes throttle → add to aggregator batch
          4. Immediate snapshot on CRITICAL (bypasses batching)
          5. Update state counters
        """
        # Log all events without filtering
        self._logger.log(event)

        # Throttle check
        current_pf = None
        should_send, reason = self._throttle.should_send(event, current_pf)

        if not should_send:
            self._state["n_suppressed"] = self._state.get("n_suppressed", 0) + 1
            return

        # Count by severity (only non-suppressed)
        if event.severity == Severity.CRITICAL:
            self._state["n_criticals"] += 1
        elif event.severity == Severity.WARNING:
            self._state["n_warnings"]  += 1

        # Add to aggregator batch (sent at end of cycle)
        self._aggregator.add(event)

        # Immediate CRITICAL snapshot + direct Telegram (bypass aggregation)
        if event.is_critical:
            if self._snap_on_critical:
                self._snapshots.save(
                    state=self._state,
                    recent_events=[e.to_dict() for e in self.bus.recent(50)],
                    trigger=f"CRITICAL_{event.event_type.value}",
                )
            # Send CRITICAL immediately, don't wait for batch
            self._telegram.alert(event)

    # ── 6H Summary ────────────────────────────────────────────────────────────

    def get_health_summary(self) -> str:
        """
        Format the 6H observer summary for Telegram.
        Called every 6H from run_once() and from the scheduler integration.
        """
        ts = datetime.now(timezone.utc).strftime("%H:%M UTC")
        agg = self._aggregate_health
        if agg is None:
            return f"📊 *Observer Summary — {ts}*\n_Nenhum dado disponivel ainda._"

        label_emoji = {
            "EXCELLENT": "💚", "GOOD": "🟢",
            "MODERATE": "🟡", "POOR": "🔴", "CRITICAL": "🚨",
        }
        emoji  = label_emoji.get(agg.label, "📊")
        bar_n  = int(agg.score / 10)
        bar    = "█" * bar_n + "░" * (10 - bar_n)

        lines = [
            f"📊 *OBSERVER SUMMARY — {ts}*",
            "",
            f"{emoji} *Saude do Sistema: {agg.score}/100 ({agg.label})*",
            f"`[{bar}]`",
            f"_{agg.description}_",
            "",
        ]

        # Components breakdown
        if agg.components:
            lines.append("*Componentes:*")
            names = {
                "regime_risk": "Regime",
                "rolling_pf": "Rolling PF",
                "robustness": "Robustness",
                "cascade_state": "Cascade",
                "structural_warnings": "Structural",
            }
            for k, pts in agg.components.items():
                sign = f"{'+' if pts >= 0 else ''}{pts}"
                lines.append(f"  • {names.get(k, k)}: `{sign} pts`")
            lines.append("")

        # Top risks (worst combinations)
        worst = sorted(self._health_scores.items(), key=lambda x: x[1].score)[:3]
        if worst:
            lines.append("*Ativos mais degradados:*")
            for key, hs in worst:
                lines.append(f"  🔴 {key}: {hs.score}/100 ({hs.label})")
            lines.append("")

        # Healthiest edge
        best = sorted(self._health_scores.items(), key=lambda x: -x[1].score)[:2]
        if best:
            lines.append("*Edge mais saudavel:*")
            for key, hs in best:
                lines.append(f"  💚 {key}: {hs.score}/100")
            lines.append("")

        # Regime dominance
        regimes: Dict[str, int] = {}
        for ctx in self._risk_contexts.values():
            r = ctx.get("current_regime", "?")
            regimes[r] = regimes.get(r, 0) + 1
        if regimes:
            lines.append("*Regimes dominantes:*")
            for r, cnt in sorted(regimes.items(), key=lambda x: -x[1]):
                lines.append(f"  • {r}: {cnt}/{len(self._risk_contexts)} ativos")
            lines.append("")

        # Alert stats
        lines.append(
            f"*Alertas (session):* {self._state.get('n_criticals', 0)} CRITICAL | "
            f"{self._state.get('n_warnings', 0)} WARNING | "
            f"{self._state.get('n_suppressed', 0)} suprimidos"
        )

        return "\n".join(lines)

    def _send_6h_summary(self) -> None:
        """Send 6H summary to Telegram (fire-and-forget, no errors propagated)."""
        try:
            summary = self.get_health_summary()
            self._telegram._send_raw(summary)  # type: ignore[attr-defined]
        except Exception:
            pass

    # ── Query API ──────────────────────────────────────────────────────────────

    def get_risk_context(self, asset: str, hypothesis: str) -> Dict:
        """Return the latest risk context for a combination (for Sistema 1 to read)."""
        return self._risk_contexts.get(f"{asset}_{hypothesis}", {})

    def get_recent_criticals(self, n: int = 10) -> List[Event]:
        return self.bus.recent_by_severity(Severity.CRITICAL, n)

    def get_state(self) -> Dict:
        return dict(self._state)

    def print_status(self) -> None:
        """Print a human-readable status summary."""
        print(f"\n{'=' * 60}")
        print(f"  OBSERVER STATUS")
        print(f"  Cycles: {self._state['n_cycles']} | "
              f"Events: {self._state['n_events']} | "
              f"Criticals: {self._state['n_criticals']} | "
              f"Warnings: {self._state['n_warnings']}")
        print(f"{'=' * 60}")

        for key, ctx in self._risk_contexts.items():
            sev   = ctx.get("severity", "INFO")
            reg   = ctx.get("current_regime", "?")
            eh    = ctx.get("edge_health", "?")
            pf30  = ctx.get("rolling_pf_30", "?")
            notes = ctx.get("structural_notes", [])
            print(f"  {key:<30} [{sev:<8}] Regime={reg} | Edge={eh} | PF30={pf30}")
            for n in notes[:2]:
                print(f"    {n[:70]}")

        print("=" * 60 + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="runpy")
    parser = argparse.ArgumentParser(description="Fractal Lab Observer Agent")
    parser.add_argument("--asset",      type=str, default=None, help="Single asset to observe")
    parser.add_argument("--hypothesis", type=str, default=None, help="Single hypothesis to observe")
    parser.add_argument("--loop",       action="store_true",    help="Run in continuous loop")
    parser.add_argument("--interval",   type=int, default=3600, help="Loop interval in seconds")
    parser.add_argument("--targets",    nargs="+",              help="Targets: ASSET:HYP ...")
    args = parser.parse_args()

    # Build target list
    targets = None
    if args.targets:
        targets = []
        for t in args.targets:
            if ":" in t:
                a, h = t.split(":", 1)
                targets.append((a.strip(), h.strip()))
    elif args.asset and args.hypothesis:
        targets = [(args.asset, args.hypothesis)]

    agent = ObserverAgent(assets_hypotheses=targets)

    if args.loop:
        agent.run_loop(interval_seconds=args.interval)
    else:
        agent.run_once(
            asset=args.asset if not args.targets else None,
            hypothesis=args.hypothesis if not args.targets else None,
        )
        agent.print_status()


if __name__ == "__main__":
    main()
