"""Streamlit dashboard — EUR/USD live signals & backtest results."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Allow relative imports regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.backtest import run_backtest
from core.collector import fetch_ohlcv
from core.indicators import add_all_indicators
from core.signals import generate_signals
from db.database import save_ohlcv, save_signals

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading AI — EUR/USD",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Trading AI — EUR/USD")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    timeframe = st.selectbox("Timeframe", ["1h", "15m"], index=0)
    st.markdown("---")
    refresh = st.button("🔄 Refresh market data", use_container_width=True)
    st.caption("Data is cached for 15 min to respect rate limits.")

    st.markdown("---")
    st.subheader("Backtest params")
    initial_capital = st.number_input("Initial capital (USD)", value=10_000, step=1_000)
    position_size   = st.slider("Position size (%)", 1, 50, 10) / 100
    sl_mult         = st.slider("Stop-loss (× ATR)", 0.5, 5.0, 1.5, 0.5)
    tp_mult         = st.slider("Take-profit (× ATR)", 1.0, 10.0, 3.0, 0.5)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=900, show_spinner=False)
def _load(tf: str) -> pd.DataFrame:
    df = fetch_ohlcv(timeframe=tf)
    df = add_all_indicators(df)
    df = generate_signals(df)
    save_ohlcv(df[["open", "high", "low", "close", "volume"]], tf)
    save_signals(df, tf)
    return df


if refresh:
    st.cache_data.clear()

with st.spinner("Fetching latest EUR/USD data …"):
    df = _load(timeframe)

result = run_backtest(
    df,
    initial_capital=float(initial_capital),
    position_size_pct=position_size,
    sl_atr_mult=sl_mult,
    tp_atr_mult=tp_mult,
)

# ── KPI strip ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Return",  f"{result.total_return:+.2f}%")
c2.metric("Sharpe Ratio",  f"{result.sharpe_ratio:.3f}")
c3.metric("Max Drawdown",  f"{result.max_drawdown:.2f}%")
c4.metric("Win Rate",      f"{result.win_rate:.1f}%")
c5.metric("Total Trades",  str(result.total_trades))

st.markdown("---")

# ── Main chart ────────────────────────────────────────────────────────────────
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    row_heights=[0.60, 0.20, 0.20],
    subplot_titles=["EUR/USD + Bollinger Bands", "RSI (14)", "MACD (12 / 26 / 9)"],
)

# Candlestick
fig.add_trace(
    go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="EUR/USD",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ),
    row=1, col=1,
)

# Bollinger Bands — upper / mid / lower with shaded fill
bb_style = [
    ("bb_upper", "dot",   None,       None),
    ("bb_mid",   "solid", None,       None),
    ("bb_lower", "dot",   "tonexty",  "rgba(100,149,237,0.07)"),
]
for col_name, dash, fill, fillcolor in bb_style:
    kwargs: dict = dict(
        x=df.index, y=df[col_name],
        line=dict(color="rgba(100,149,237,0.55)", dash=dash, width=1),
        name=col_name, showlegend=False,
    )
    if fill:
        kwargs["fill"] = fill
        kwargs["fillcolor"] = fillcolor
    fig.add_trace(go.Scatter(**kwargs), row=1, col=1)

# Buy / sell signal markers
buys  = df[df["signal"] ==  1]
sells = df[df["signal"] == -1]
fig.add_trace(go.Scatter(
    x=buys.index, y=buys["low"] * 0.9997, mode="markers",
    marker=dict(symbol="triangle-up", size=11, color="#26a69a"), name="Buy",
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=sells.index, y=sells["high"] * 1.0003, mode="markers",
    marker=dict(symbol="triangle-down", size=11, color="#ef5350"), name="Sell",
), row=1, col=1)

# RSI
fig.add_trace(
    go.Scatter(x=df.index, y=df["rsi"], line=dict(color="#9c27b0", width=1.5), name="RSI"),
    row=2, col=1,
)
for level, color in [(70, "red"), (30, "green")]:
    fig.add_hline(y=level, line_color=color, line_dash="dash", line_width=1, row=2, col=1)

# MACD
hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["macd_hist"]]
fig.add_trace(
    go.Bar(x=df.index, y=df["macd_hist"], marker_color=hist_colors,
           name="Histogram", showlegend=False),
    row=3, col=1,
)
fig.add_trace(
    go.Scatter(x=df.index, y=df["macd"],
               line=dict(color="#2196f3", width=1.5), name="MACD"),
    row=3, col=1,
)
fig.add_trace(
    go.Scatter(x=df.index, y=df["macd_signal"],
               line=dict(color="#ff9800", width=1.5), name="Signal line"),
    row=3, col=1,
)

fig.update_layout(
    height=820,
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", y=1.04, x=0),
    margin=dict(l=0, r=0, t=40, b=0),
)
fig.update_yaxes(range=[0, 100], row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# ── Equity curve ──────────────────────────────────────────────────────────────
st.subheader("Equity Curve")
eq = result.equity_curve
eq_fig = go.Figure(go.Scatter(
    x=eq.index, y=eq.values,
    fill="tozeroy", fillcolor="rgba(38,166,154,0.12)",
    line=dict(color="#26a69a", width=2), name="Equity",
))
eq_fig.update_layout(
    template="plotly_dark", height=220,
    margin=dict(l=0, r=0, t=10, b=0),
    yaxis_title="Capital (USD)",
)
st.plotly_chart(eq_fig, use_container_width=True)

# ── Trade log ─────────────────────────────────────────────────────────────────
if not result.trades.empty:
    st.subheader("Trade Log")
    display = result.trades.copy()
    display["pnl"] = display["pnl"].map(lambda x: f"${x:+.4f}")
    st.dataframe(display, use_container_width=True, hide_index=True)
else:
    st.info("No trades generated for this timeframe / parameter set.")
