"""MT5 chart annotation — draws SL/TP lines and signal comments on the MT5 chart.

Uses the MetaTrader5 Python API to send graphical objects to the active chart
so the trader can see entry, stop-loss, and take-profit levels at a glance.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config

logger = logging.getLogger(__name__)

# Arrow codes used by MT5
_ARROW_BUY  = 233   # up arrow (↑)
_ARROW_SELL = 234   # down arrow (↓)

# Object name prefix — makes it easy to delete all our objects at once
_OBJ_PREFIX = "trading_ai_"


def _mt5():
    import MetaTrader5 as mt5
    return mt5


def _ts(dt: datetime | pd.Timestamp) -> int:
    """Convert datetime to Unix timestamp (seconds)."""
    if hasattr(dt, "timestamp"):
        return int(dt.timestamp())
    return int(pd.Timestamp(dt).timestamp())


def clear_chart_objects(symbol: str = "EURUSD", chart_id: int = 0) -> None:
    """Remove all trading_ai_ objects from the MT5 chart."""
    mt5 = _mt5()
    objects = mt5.chart_objects_get(chart_id, symbol) or []
    for obj in objects:
        name = getattr(obj, "name", "")
        if name.startswith(_OBJ_PREFIX):
            mt5.object_delete(chart_id, name)
    logger.debug("Cleared chart objects for %s", symbol)


def _delete_if_exists(chart_id: int, name: str) -> None:
    mt5 = _mt5()
    mt5.object_delete(chart_id, name)


def draw_signal(
    symbol: str,
    signal: int,
    entry: float,
    sl: float,
    tp: float,
    bar_time: datetime | pd.Timestamp,
    score: float = 0.0,
    chart_id: int = 0,
    timeframe_label: str = "",
) -> bool:
    """Draw entry arrow, SL line, TP line and comment label on the MT5 chart.

    Args:
        symbol:          MT5 symbol (e.g. ``"EURUSD"``).
        signal:          1=BUY | -1=SELL.
        entry:           Entry price.
        sl:              Stop-loss price.
        tp:              Take-profit price.
        bar_time:        Datetime of the signal bar.
        score:           Confidence score (0-100).
        chart_id:        MT5 chart ID. 0 = currently active chart.
        timeframe_label: E.g. ``"H1"`` — appended to the label text.

    Returns:
        True if all objects were created successfully.
    """
    if signal not in (1, -1):
        return False

    mt5 = _mt5()
    ts  = _ts(bar_time)

    direction = "BUY" if signal == 1 else "SELL"
    color_entry = 0x00AA00 if signal == 1 else 0xCC0000   # green / red (BGR)
    color_sl    = 0x0000CC   # dark red
    color_tp    = 0x006600   # dark green
    arrow_code  = _ARROW_BUY if signal == 1 else _ARROW_SELL

    # Arrow at entry bar
    arrow_name  = f"{_OBJ_PREFIX}{direction}_{ts}"
    _delete_if_exists(chart_id, arrow_name)
    mt5.object_create(chart_id, arrow_name, mt5.OBJ_ARROW, 0, ts, entry)
    mt5.object_set_integer(chart_id, arrow_name, mt5.OBJPROP_ARROWCODE,  arrow_code)
    mt5.object_set_integer(chart_id, arrow_name, mt5.OBJPROP_COLOR,      color_entry)
    mt5.object_set_integer(chart_id, arrow_name, mt5.OBJPROP_WIDTH,      2)

    # SL horizontal line
    sl_name = f"{_OBJ_PREFIX}SL_{ts}"
    _delete_if_exists(chart_id, sl_name)
    mt5.object_create(chart_id, sl_name, mt5.OBJ_HLINE, 0, ts, sl)
    mt5.object_set_integer(chart_id, sl_name, mt5.OBJPROP_COLOR,     color_sl)
    mt5.object_set_integer(chart_id, sl_name, mt5.OBJPROP_STYLE,     mt5.STYLE_DASH)
    mt5.object_set_integer(chart_id, sl_name, mt5.OBJPROP_WIDTH,     1)
    mt5.object_set_string(chart_id,  sl_name, mt5.OBJPROP_TEXT,      f"SL {sl:.5f}")

    # TP horizontal line
    tp_name = f"{_OBJ_PREFIX}TP_{ts}"
    _delete_if_exists(chart_id, tp_name)
    mt5.object_create(chart_id, tp_name, mt5.OBJ_HLINE, 0, ts, tp)
    mt5.object_set_integer(chart_id, tp_name, mt5.OBJPROP_COLOR,     color_tp)
    mt5.object_set_integer(chart_id, tp_name, mt5.OBJPROP_STYLE,     mt5.STYLE_DASH)
    mt5.object_set_integer(chart_id, tp_name, mt5.OBJPROP_WIDTH,     1)
    mt5.object_set_string(chart_id,  tp_name, mt5.OBJPROP_TEXT,      f"TP {tp:.5f}")

    # Text comment label
    label_text = (
        f"trading-ai {direction} {timeframe_label}\n"
        f"Entry: {entry:.5f}  Score: {score:.0f}/100\n"
        f"SL: {sl:.5f}  TP: {tp:.5f}"
    )
    label_name = f"{_OBJ_PREFIX}LBL_{ts}"
    _delete_if_exists(chart_id, label_name)
    mt5.object_create(chart_id, label_name, mt5.OBJ_TEXT, 0, ts, entry)
    mt5.object_set_string(chart_id,  label_name, mt5.OBJPROP_TEXT,      label_text)
    mt5.object_set_integer(chart_id, label_name, mt5.OBJPROP_COLOR,     color_entry)
    mt5.object_set_integer(chart_id, label_name, mt5.OBJPROP_FONTSIZE,  8)

    mt5.chart_redraw(chart_id)
    logger.info("Drew %s signal at %s: entry=%.5f sl=%.5f tp=%.5f", direction, bar_time, entry, sl, tp)
    return True


def annotate_from_dataframe(
    df: pd.DataFrame,
    symbol: str = "EURUSD",
    timeframe_label: str = "H1",
    chart_id: int = 0,
    atr_col: str = "atr",
) -> int:
    """Read signals from a processed DataFrame and draw all non-zero ones on the MT5 chart.

    Expects columns: ``signal``, ``close``, ``score``, ``atr`` (or as per atr_col).
    SL = entry ± SL_ATR_MULT * ATR | TP = entry ± TP_ATR_MULT * ATR

    Returns:
        Number of signal objects drawn.
    """
    if "signal" not in df.columns or atr_col not in df.columns:
        logger.warning("annotate_from_dataframe: missing 'signal' or '%s' column", atr_col)
        return 0

    clear_chart_objects(symbol, chart_id)

    drawn = 0
    for ts, row in df.iterrows():
        sig = int(row["signal"])
        if sig == 0:
            continue

        entry = float(row["close"])
        atr   = float(row[atr_col]) if not pd.isna(row[atr_col]) else 0.0
        sl    = entry - sig * config.SL_ATR_MULT * atr
        tp    = entry + sig * config.TP_ATR_MULT * atr
        score = float(row.get("score", 0))

        ok = draw_signal(
            symbol=symbol,
            signal=sig,
            entry=entry,
            sl=sl,
            tp=tp,
            bar_time=ts,
            score=score,
            chart_id=chart_id,
            timeframe_label=timeframe_label,
        )
        if ok:
            drawn += 1

    logger.info("Annotated %d signal(s) on MT5 chart for %s", drawn, symbol)
    return drawn
