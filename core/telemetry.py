"""Live trading telemetry — observational only, never blocks trades.

Records every scheduler cycle to data/telemetry.json:
  - Every signal generated (BUY/SELL/HOLD)
  - Whether regime filter WOULD HAVE blocked it (observation, not action)
  - For blocked signals: resolves hypothetical outcome N cycles later
  - Rolling PF of last 20 closed trades
  - Rolling daily drawdown
  - Performance breakdown by session (Asia/London/NY)
  - Divergence between backtest assumptions and live fills

Telegram alerts fired by telemetry:
  "⚠️ PF Rolling caiu abaixo de 1.0 — monitorar"
  "✅ Trade bloqueado pelo filtro — seria GANHO/PERDA (+X pips)"

RULE: telemetry NEVER raises exceptions that could stop the scheduler.
All public methods are wrapped in try/except internally.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

_ROOT = Path(__file__).parent.parent
_STATE_FILE = _ROOT / "data" / "telemetry.json"

# Regime filter parameters (matched from regime_filter_results.json)
_REGIME_ADX_THRESHOLD  = 35.0
_REGIME_ATR_RATIO_MIN  = 0.8
_REGIME_ATR_RATIO_MAX  = 1.5

# Rolling window for PF computation
_ROLLING_WINDOW = 20
# Candles to wait before resolving a blocked signal's hypothetical outcome
_BLOCKED_RESOLVE_CANDLES = 8   # 8 × 15min = 2H matches the N=2H top combo

_MAX_EVENTS    = 500    # max entries to keep per list
_PF_ALERT_BELOW = 1.0   # alert when rolling PF drops below this

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pips(price_diff: float) -> float:
    return round(price_diff * 10_000, 1)


def _default_state() -> dict:
    return {
        "signals":          [],   # all generated signals
        "blocked_signals":  [],   # signals blocked by regime filter (observation)
        "pending_blocked":  [],   # blocked signals awaiting outcome resolution
        "closed_trades":    [],   # paper trade outcomes
        "rolling_pf":       None,
        "rolling_trades_pnl": [],
        "daily_pnl":        0.0,
        "daily_start_equity": 0.0,
        "daily_date":       None,
        "session_stats":    {
            "Asia":    {"trades": 0, "wins": 0, "pnl": 0.0},
            "London":  {"trades": 0, "wins": 0, "pnl": 0.0},
            "NY":      {"trades": 0, "wins": 0, "pnl": 0.0},
            "Overlap": {"trades": 0, "wins": 0, "pnl": 0.0},
        },
        "slippage":         {"count": 0, "total_pips": 0.0},
        "filter_stats":     {"allowed": 0, "would_block": 0},
        "last_updated":     None,
    }


class Telemetry:
    """Thread-safe live telemetry recorder."""

    def __init__(self) -> None:
        self._state = self._load()

    # ── Public API ─────────────────────────────────────────────────────────────

    def record_signal(
        self,
        symbol:    str,
        name:      str,
        signal:    str,          # "BUY" / "SELL" / "HOLD"
        price:     float,
        adx:       float,
        atr_ratio: float,
        hour_utc:  int,
    ) -> Optional[str]:
        """
        Record a signal event and check regime filter (observation only).

        Returns a Telegram alert string if the filter would have blocked the
        signal, or None otherwise.
        """
        try:
            would_block = (
                adx >= _REGIME_ADX_THRESHOLD or
                atr_ratio <= _REGIME_ATR_RATIO_MIN or
                atr_ratio >= _REGIME_ATR_RATIO_MAX
            )
            session = _session_name(hour_utc)

            event = {
                "ts":         _now_iso(),
                "symbol":     symbol,
                "name":       name,
                "signal":     signal,
                "price":      round(price, 5),
                "adx":        round(adx, 2),
                "atr_ratio":  round(atr_ratio, 3),
                "session":    session,
                "would_block": would_block,
            }

            if signal in ("BUY", "SELL"):
                self._state["signals"] = (
                    self._state["signals"] + [event]
                )[-_MAX_EVENTS:]

                if would_block:
                    self._state["filter_stats"]["would_block"] += 1
                    pending = dict(event)
                    pending["resolve_after_candles"] = _BLOCKED_RESOLVE_CANDLES
                    pending["candles_elapsed"]       = 0
                    self._state["pending_blocked"] = (
                        self._state["pending_blocked"] + [pending]
                    )[-100:]
                else:
                    self._state["filter_stats"]["allowed"] += 1

                self._state["last_updated"] = _now_iso()
                self._save()

        except Exception as exc:
            logger.debug("Telemetry.record_signal error: %s", exc)

        return None

    def record_trade_outcome(
        self,
        symbol:      str,
        direction:   str,    # "BUY" / "SELL"
        entry_price: float,
        exit_price:  float,
        pnl:         float,  # in account currency
        pnl_pct:     float,  # percentage of position
        hour_utc:    int,
        expected_entry: Optional[float] = None,  # for slippage calc
    ) -> Optional[str]:
        """
        Record when a paper trade closes.
        Returns Telegram alert string if rolling PF < 1.0, else None.
        """
        try:
            session = _session_name(hour_utc)
            trade = {
                "ts":          _now_iso(),
                "symbol":      symbol,
                "direction":   direction,
                "entry_price": round(entry_price, 5),
                "exit_price":  round(exit_price, 5),
                "pnl":         round(pnl, 4),
                "pnl_pct":     round(pnl_pct, 4),
                "win":         pnl > 0,
                "session":     session,
            }

            # Slippage vs expected (backtest) entry
            if expected_entry is not None and entry_price != expected_entry:
                slip_pips = _pips(abs(entry_price - expected_entry))
                trade["slippage_pips"] = slip_pips
                sl = self._state["slippage"]
                sl["count"]      += 1
                sl["total_pips"] += slip_pips

            # Update closed trades list
            self._state["closed_trades"] = (
                self._state["closed_trades"] + [trade]
            )[-_MAX_EVENTS:]

            # Update rolling PnL window (last N trades)
            rolling = self._state["rolling_trades_pnl"]
            rolling.append(pnl_pct)
            if len(rolling) > _ROLLING_WINDOW:
                rolling.pop(0)
            self._state["rolling_trades_pnl"] = rolling

            # Compute rolling PF
            alert_msg = self._update_rolling_pf()

            # Update session stats
            ss = self._state["session_stats"].get(session)
            if ss:
                ss["trades"] += 1
                if pnl > 0:
                    ss["wins"] += 1
                ss["pnl"] += pnl

            # Daily PnL
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if self._state.get("daily_date") != today:
                self._state["daily_pnl"]  = 0.0
                self._state["daily_date"] = today
            self._state["daily_pnl"] += pnl

            self._state["last_updated"] = _now_iso()
            self._save()
            return alert_msg

        except Exception as exc:
            logger.debug("Telemetry.record_trade_outcome error: %s", exc)
            return None

    def resolve_blocked_outcomes(
        self,
        df: pd.DataFrame,
    ) -> list[str]:
        """
        Check if any pending blocked signal outcomes can now be resolved.

        For each pending blocked signal:
          - Increment candles_elapsed
          - When candles_elapsed >= resolve_after_candles:
            - Compute whether the hypothetical trade would have been a WIN/LOSS
            - Send Telegram alert
            - Remove from pending list

        Returns list of Telegram alert strings.
        """
        alerts: list[str] = []
        try:
            if not self._state["pending_blocked"]:
                return alerts

            last_close = float(df["close"].iloc[-1])
            still_pending: list[dict] = []
            resolved_list: list[dict] = []

            for p in self._state["pending_blocked"]:
                p["candles_elapsed"] = p.get("candles_elapsed", 0) + 1

                if p["candles_elapsed"] < p["resolve_after_candles"]:
                    still_pending.append(p)
                    continue

                # Resolve outcome
                entry_price = p["price"]
                direction   = p["signal"]   # BUY or SELL
                pips_move   = _pips(last_close - entry_price)

                if direction == "BUY":
                    net_pips = pips_move - _pips(0.000025)  # minus spread
                else:
                    net_pips = -pips_move - _pips(0.000025)

                outcome   = "GANHO" if net_pips > 0 else "PERDA"
                emoji     = "✅" if net_pips > 0 else "❌"
                adx_val   = p.get("adx", 0)
                atr_val   = p.get("atr_ratio", 1)
                session   = p.get("session", "?")

                alert = (
                    f"{emoji} Trade bloqueado pelo filtro — seria {outcome}\n"
                    f"  Sinal: {direction} {p['name']} @ {entry_price:.5f}\n"
                    f"  Resultado hipotetico: {net_pips:+.1f} pips ({p['resolve_after_candles']}H)\n"
                    f"  Motivo bloqueio: ADX={adx_val:.1f}>={_REGIME_ADX_THRESHOLD} "
                    f"ou ATR_ratio={atr_val:.3f} fora de ({_REGIME_ATR_RATIO_MIN},{_REGIME_ATR_RATIO_MAX})\n"
                    f"  Sessao: {session}"
                )
                alerts.append(alert)

                resolved = dict(p)
                resolved["resolved_at"]   = _now_iso()
                resolved["outcome"]       = outcome
                resolved["net_pips"]      = net_pips
                resolved["exit_price"]    = round(last_close, 5)
                resolved_list.append(resolved)

            self._state["pending_blocked"] = still_pending
            self._state["blocked_signals"] = (
                self._state["blocked_signals"] + resolved_list
            )[-_MAX_EVENTS:]

            if alerts:
                self._state["last_updated"] = _now_iso()
                self._save()

        except Exception as exc:
            logger.debug("Telemetry.resolve_blocked_outcomes error: %s", exc)

        return alerts

    def get_summary(self) -> dict:
        """Return a compact summary of current telemetry state."""
        try:
            rolling = self._state.get("rolling_trades_pnl", [])
            wins    = [p for p in rolling if p > 0]
            losses  = [p for p in rolling if p <= 0]

            session_win_rates = {}
            for sess, st in self._state["session_stats"].items():
                n = st["trades"]
                session_win_rates[sess] = (
                    round(st["wins"] / n * 100, 1) if n > 0 else None
                )

            slippage = self._state["slippage"]
            avg_slip = (
                slippage["total_pips"] / slippage["count"]
                if slippage["count"] > 0 else 0.0
            )

            return {
                "rolling_pf":    self._state.get("rolling_pf"),
                "rolling_n":     len(rolling),
                "daily_pnl":     self._state.get("daily_pnl", 0.0),
                "total_signals": len(self._state["signals"]),
                "filter_allowed":   self._state["filter_stats"]["allowed"],
                "filter_would_block": self._state["filter_stats"]["would_block"],
                "pending_blocked_n": len(self._state["pending_blocked"]),
                "session_win_rates": session_win_rates,
                "avg_slippage_pips": round(avg_slip, 2),
                "last_updated":  self._state.get("last_updated"),
            }
        except Exception:
            return {}

    # ── Internal ──────────────────────────────────────────────────────────────

    def _update_rolling_pf(self) -> Optional[str]:
        """Recompute rolling PF from last N trades. Return alert if PF < 1.0."""
        rolling = self._state.get("rolling_trades_pnl", [])
        if len(rolling) < 5:
            self._state["rolling_pf"] = None
            return None

        wins   = [p for p in rolling if p > 0]
        losses = [p for p in rolling if p <= 0]
        gp     = sum(wins)
        gl     = abs(sum(losses)) + 1e-9
        pf     = gp / gl
        self._state["rolling_pf"] = round(pf, 3)

        if pf < _PF_ALERT_BELOW and len(rolling) >= 10:
            win_rate = len(wins) / len(rolling) * 100
            return (
                f"⚠️ PF Rolling caiu abaixo de 1.0 — monitorar\n"
                f"  PF={pf:.3f} | {len(rolling)} trades | WR={win_rate:.1f}%\n"
                f"  Sistema em observacao — nao alterar parametros"
            )
        return None

    def _load(self) -> dict:
        if _STATE_FILE.exists():
            try:
                with open(_STATE_FILE, encoding="utf-8") as f:
                    saved = json.load(f)
                default = _default_state()
                # Merge: keep saved values, add any missing keys from default
                for k, v in default.items():
                    if k not in saved:
                        saved[k] = v
                return saved
            except Exception as exc:
                logger.warning("Telemetry: failed to load state (%s) — starting fresh.", exc)
        return _default_state()

    def _save(self) -> None:
        try:
            _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.debug("Telemetry._save error: %s", exc)


# ── Session helper ─────────────────────────────────────────────────────────────

def _session_name(hour_utc: int) -> str:
    if 13 <= hour_utc < 17:
        return "Overlap"
    if 13 <= hour_utc < 21:
        return "NY"
    if 8  <= hour_utc < 17:
        return "London"
    return "Asia"


# ── Module-level singleton ────────────────────────────────────────────────────

_instance: Optional[Telemetry] = None


def get_telemetry() -> Telemetry:
    global _instance
    if _instance is None:
        _instance = Telemetry()
    return _instance
