from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..storage.models import Hypothesis, Trade


# ── Signal generation ──────────────────────────────────────────────────────────

def _eval_op(val: float, op: str, threshold: float) -> bool:
    ops = {">": val > threshold, "<": val < threshold,
           ">=": val >= threshold, "<=": val <= threshold, "==": val == threshold}
    return ops.get(op, False)


def _generate_signal(row: pd.Series, entry_logic: Dict[str, Any]) -> int:
    """
    Return 1 (long), -1 (short), or 0 (no signal).

    Supported entry types:
      feature_threshold   — single feature vs threshold, fixed direction
      multi_condition     — AND of multiple conditions, fixed direction
      mean_reversion      — buy below EMA, sell above EMA (uses ema_distance)
      trend_follow        — follow market_bias direction
      crossover           — EMA crossover via ema_slope sign change
      breakout            — close > rolling high or < rolling low
      oscillator_extremes — buy when feature < low_threshold, sell when > high_threshold
      ema_slope           — slope direction + price side of EMA20
      momentum            — price move over N bars vs fixed threshold
      momentum_atr        — price move over N bars vs N×ATR (dynamic threshold)
      volatility_breakout — enter in bias direction when ATR expands
      range_breakout      — enter in bias direction when ATR compressed
    """
    logic_type = entry_logic.get("type", "feature_threshold")

    if logic_type == "feature_threshold":
        feature = entry_logic.get("feature", "")
        op = entry_logic.get("operator", ">")
        threshold = float(entry_logic.get("threshold", 0.0))
        direction = entry_logic.get("direction", "long")
        val = row.get(feature)
        if val is None or pd.isna(val):
            return 0
        if _eval_op(float(val), op, threshold):
            return 1 if direction == "long" else -1
        return 0

    if logic_type == "multi_condition":
        conditions = entry_logic.get("conditions", [])
        direction = entry_logic.get("direction", "long")
        for cond in conditions:
            val = row.get(cond.get("feature", ""))
            if val is None or pd.isna(val):
                return 0
            if not _eval_op(float(val), cond.get("operator", ">"), float(cond.get("threshold", 0.0))):
                return 0
        return 1 if direction == "long" else -1

    if logic_type == "mean_reversion":
        dist = row.get("ema_distance", row.get("signed_distance_to_mean", 0.0))
        if dist is None or pd.isna(dist):
            return 0
        threshold = float(entry_logic.get("threshold", 1.0))
        if float(dist) < -threshold:
            return 1   # buy: price far below mean
        if float(dist) > threshold:
            return -1  # sell: price far above mean
        return 0

    if logic_type == "trend_follow":
        bias = row.get("market_bias", 0)
        if bias is None or pd.isna(bias):
            return 0
        return int(bias)

    if logic_type == "crossover":
        # Positive slope_short signals recent EMA crossover up; negative → down
        slope = row.get("ema_slope", row.get("slope_short", 0.0))
        prev_slope = row.get("_prev_ema_slope", slope)
        if slope is None or pd.isna(slope):
            return 0
        threshold = float(entry_logic.get("threshold", 0.0))
        if float(slope) > threshold and (prev_slope is None or float(prev_slope) <= threshold):
            return 1
        if float(slope) < -threshold and (prev_slope is None or float(prev_slope) >= -threshold):
            return -1
        return 0

    if logic_type == "breakout":
        close = row.get("Close", row.get("close", 0.0))
        roll_high = row.get("roll_high", float("nan"))
        roll_low = row.get("roll_low", float("nan"))
        if pd.isna(roll_high) or pd.isna(roll_low):
            return 0
        if float(close) > float(roll_high):
            return 1
        if float(close) < float(roll_low):
            return -1
        return 0

    if logic_type == "oscillator_extremes":
        # RSI-style: buy when feature < low_threshold, sell when > high_threshold
        feature = entry_logic.get("feature", "rsi_proxy")
        low_th = float(entry_logic.get("low_threshold", 30.0))
        high_th = float(entry_logic.get("high_threshold", 70.0))
        val = row.get(feature)
        if val is None or pd.isna(val):
            return 0
        if float(val) < low_th:
            return 1
        if float(val) > high_th:
            return -1
        return 0

    if logic_type == "ema_slope":
        # Long: EMA20 slope positive AND price above EMA20. Short: opposite.
        slope = row.get("ema20_slope", row.get("ema_slope", 0.0))
        dist = row.get("ema20_distance", row.get("ema_distance", 0.0))
        slope_th = float(entry_logic.get("slope_threshold", 0.0))
        if slope is None or pd.isna(slope) or dist is None or pd.isna(dist):
            return 0
        if float(slope) > slope_th and float(dist) > 0:
            return 1
        if float(slope) < -slope_th and float(dist) < 0:
            return -1
        return 0

    if logic_type == "momentum":
        # Directional momentum: price difference over N bars vs fixed threshold
        mom = row.get("momentum", 0.0)
        threshold = float(entry_logic.get("threshold", 0.0))
        if mom is None or pd.isna(mom):
            return 0
        if float(mom) > threshold:
            return 1
        if float(mom) < -threshold:
            return -1
        return 0

    if logic_type == "momentum_atr":
        # Momentum vs dynamic ATR threshold: enter when |momentum| > N×ATR
        mom = row.get("momentum", 0.0)
        atr_val = row.get("atr", 1.0)
        multiplier = float(entry_logic.get("atr_multiplier", 0.5))
        if mom is None or atr_val is None or pd.isna(mom) or pd.isna(atr_val):
            return 0
        threshold = multiplier * float(atr_val)
        if float(mom) > threshold:
            return 1
        if float(mom) < -threshold:
            return -1
        return 0

    if logic_type == "volatility_breakout":
        # Enter in market_bias direction when ATR expands above threshold
        atr_ratio = row.get("atr_ratio", 1.0)
        bias = int(row.get("market_bias", 0) or 0)
        threshold = float(entry_logic.get("threshold", 1.5))
        if atr_ratio is None or pd.isna(atr_ratio):
            return 0
        if float(atr_ratio) > threshold and bias != 0:
            return bias
        return 0

    if logic_type == "range_breakout":
        # Enter in market_bias direction after ATR compression
        atr_ratio = row.get("atr_ratio", 1.0)
        bias = int(row.get("market_bias", 0) or 0)
        threshold = float(entry_logic.get("threshold", 0.7))
        if atr_ratio is None or pd.isna(atr_ratio):
            return 0
        if float(atr_ratio) < threshold and bias != 0:
            return bias
        return 0

    return 0


# ── SL/TP computation ──────────────────────────────────────────────────────────

def _compute_sl_tp(signal: int, entry: float, atr: float, exit_logic: Dict[str, Any]):
    exit_type = exit_logic.get("type", "fixed_rr")

    if exit_type == "fixed_rr":
        sl_mult = float(exit_logic.get("sl_atr_multiplier", 1.0))
        tp_mult = float(exit_logic.get("tp_atr_multiplier", 2.0))
        offset_sl = sl_mult * atr
        offset_tp = tp_mult * atr
    elif exit_type == "fixed_pips":
        pip_mult = float(exit_logic.get("pip_multiplier", 10000.0))
        offset_sl = float(exit_logic.get("sl_pips", 10)) / pip_mult
        offset_tp = float(exit_logic.get("tp_pips", 20)) / pip_mult
    else:
        offset_sl = atr
        offset_tp = 2 * atr

    if signal == 1:  # long
        return entry - offset_sl, entry + offset_tp
    else:            # short
        return entry + offset_sl, entry - offset_tp


# ── Bar exit check ─────────────────────────────────────────────────────────────

def _check_bar_exit(bar: pd.Series, direction: int, sl: float, tp: float):
    """
    Return (outcome, exit_price) or (None, None) if trade stays open.

    If both TP and SL are within the bar's range, pessimistic assumption: SL hit first.
    """
    hi = float(bar.get("High", bar.get("high", 0.0)))
    lo = float(bar.get("Low", bar.get("low", 0.0)))
    close = float(bar.get("Close", bar.get("close", 0.0)))

    if direction == 1:   # long: TP above, SL below
        sl_hit = lo <= sl
        tp_hit = hi >= tp
    else:                # short: TP below, SL above
        sl_hit = hi >= sl
        tp_hit = lo <= tp

    if sl_hit and tp_hit:
        return "SL_HIT", sl   # pessimistic
    if sl_hit:
        return "SL_HIT", sl
    if tp_hit:
        return "TP_HIT", tp
    return None, None


# ── Backtest Engine ────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Candle-based sequential backtest engine.

    Rules:
    - One trade at a time (sequential, no pyramiding)
    - Signal generated on bar t-1, entry at bar t open (no look-ahead)
    - TP/SL checked within each bar using High/Low
    - Regime filter applied at signal bar
    - Timeout closes at last bar's close
    """

    def run(
        self,
        df: pd.DataFrame,
        hypothesis: Hypothesis,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
    ) -> List[Trade]:
        df_work = df.copy()

        if period_start and period_end:
            ts_start = pd.Timestamp(period_start)
            ts_end = pd.Timestamp(period_end)
            if df_work.index.tz is not None:
                if ts_start.tz is None:
                    ts_start = ts_start.tz_localize("UTC")
                if ts_end.tz is None:
                    ts_end = ts_end.tz_localize("UTC")
            df_work = df_work.loc[
                (df_work.index >= ts_start) & (df_work.index <= ts_end)
            ]

        if len(df_work) < 2:
            return []

        # Add prev ema_slope for crossover detection
        if "ema_slope" in df_work.columns:
            df_work["_prev_ema_slope"] = df_work["ema_slope"].shift(1)

        # Add breakout levels if needed
        entry_type = hypothesis.entry_logic.get("type")
        if entry_type == "breakout":
            period = int(hypothesis.entry_logic.get("period", 20))
            df_work["roll_high"] = df_work["Close"].shift(1).rolling(period).max()
            df_work["roll_low"] = df_work["Close"].shift(1).rolling(period).min()

        df_work = df_work.reset_index()
        index_col = df_work.columns[0]  # datetime index column after reset

        max_bars = int(hypothesis.exit_logic.get("max_bars", 48))
        regime_filter = hypothesis.regime_filter
        pip_mult = hypothesis.pip_multiplier

        trades: List[Trade] = []
        active: Optional[Dict] = None

        for i in range(1, len(df_work)):
            prev = df_work.iloc[i - 1]
            curr = df_work.iloc[i]

            if active:
                bars_in_trade = i - active["start_i"]
                outcome, exit_price = _check_bar_exit(
                    curr, active["direction"], active["sl"], active["tp"]
                )

                if outcome or bars_in_trade >= max_bars:
                    if outcome is None:
                        outcome = "TIMEOUT"
                        exit_price = float(curr.get("Close", curr.get("close", active["entry"])))

                    direction = active["direction"]
                    if direction == 1:
                        pnl = (exit_price - active["entry"]) * pip_mult
                    else:
                        pnl = (active["entry"] - exit_price) * pip_mult

                    trades.append(Trade(
                        entry_time=str(active["entry_time"]),
                        exit_time=str(curr[index_col]),
                        direction=direction,
                        entry_price=active["entry"],
                        exit_price=exit_price,
                        pnl_pips=round(pnl, 2),
                        outcome=outcome,
                        bars_held=bars_in_trade,
                        tp=active["tp"],
                        sl=active["sl"],
                        regime=active.get("regime"),
                    ))
                    active = None

            else:
                # Apply regime filter on the signal bar
                if regime_filter:
                    bar_regime = prev.get("regime")
                    if bar_regime != regime_filter:
                        continue

                signal = _generate_signal(prev, hypothesis.entry_logic)
                if signal == 0:
                    continue

                entry_price = float(curr.get("Open", curr.get("open", 0.0)))
                atr = float(prev.get("atr", prev.get("atr14", 0.001)) or 0.001)
                sl, tp = _compute_sl_tp(signal, entry_price, atr, hypothesis.exit_logic)

                active = {
                    "entry_time": curr[index_col],
                    "entry": entry_price,
                    "direction": signal,
                    "sl": sl,
                    "tp": tp,
                    "regime": prev.get("regime"),
                    "start_i": i,
                }

        return trades
