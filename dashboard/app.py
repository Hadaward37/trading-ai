"""Streamlit dashboard — multi-asset trading signals, regime & multi-timeframe.

Supported assets: EUR/USD (Forex) | VALE3, PETR4, ITUB4, BBDC4, IBOV (B3)
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from core.backtest import run_backtest
from core.collector import fetch_ohlcv, get_mtf_confluence
from core.indicators import add_all_indicators
from core.news_filter import get_upcoming_events, is_danger_zone
from core.regime import detect_regime, regime_color, regime_emoji
from core.signals import generate_signals_custom
from db.database import save_ohlcv, save_signals

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading AI",
    page_icon="📈",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    # Asset selector
    asset_name = st.selectbox("Ativo", list(config.ASSETS), index=0)
    symbol     = config.ASSETS[asset_name]
    meta       = config.ASSET_META[symbol]

    # Asset info badge
    st.markdown(
        f"<div style='background:#1e1e1e;border-radius:6px;padding:6px 10px;"
        f"font-size:0.85em;color:#aaa;margin-bottom:4px'>"
        f"{meta['flag']} <b style='color:#eee'>{meta['name']}</b> &nbsp;|&nbsp; "
        f"{meta['exchange']} &nbsp;|&nbsp; {meta['currency']}"
        f"</div>",
        unsafe_allow_html=True,
    )

    timeframe = st.selectbox("Timeframe", list(config.TIMEFRAMES), index=1)  # default 1h

    st.markdown("---")
    refresh = st.button("🔄 Refresh market data", use_container_width=True)
    st.caption("Market data cached 15 min.")

    st.markdown("---")
    st.subheader("Strategy params")
    st.caption("Optimised via grid search — Sharpe 1.299")
    rsi_buy     = st.slider("RSI Buy  <",  10, 50, config.RSI_BUY,  step=1)
    rsi_sell    = st.slider("RSI Sell >",  51, 90, config.RSI_SELL, step=1)
    block_range = st.checkbox("Bloquear sinais em Range", value=config.REGIME_BLOCK_RANGE_SIGNALS)

    st.markdown("---")
    st.subheader("Backtest params")
    cap_label       = f"Capital inicial ({meta['currency']})"
    initial_capital = st.number_input(cap_label, value=int(config.INITIAL_CAPITAL), step=1_000)
    position_size   = st.slider("Position size (%)", 1, 50, int(config.POSITION_SIZE_PCT * 100)) / 100
    sl_mult = st.slider("Stop-loss  (x ATR)",  0.5, 5.0, config.SL_ATR_MULT, 0.5)
    tp_mult = st.slider("Take-profit (x ATR)", 1.0, 10.0, config.TP_ATR_MULT, 0.5)


# ── Dynamic page title ────────────────────────────────────────────────────────
st.title(f"📈 Trading AI — {meta['flag']} {meta['name']}")


# ── Cached data loaders ───────────────────────────────────────────────────────
@st.cache_data(ttl=config.CACHE_TTL_SECONDS, show_spinner=False)
def _load_indicators(tf: str, sym: str) -> pd.DataFrame:
    """Fetch OHLCV + all indicators. Cached per (timeframe, symbol)."""
    return add_all_indicators(fetch_ohlcv(timeframe=tf, symbol=sym))


@st.cache_data(ttl=config.CACHE_TTL_SECONDS, show_spinner=False)
def _load_mtf(sym: str) -> dict:
    """Fetch MTF confluence (3 yfinance calls). Cached per symbol."""
    return get_mtf_confluence(symbol=sym)


@st.cache_data(ttl=config.NEWS_CACHE_TTL_SECONDS, show_spinner=False)
def _load_news_events():
    """Fetch next high-impact events from ForexFactory. Cached 1 hour."""
    return get_upcoming_events(n=3)


if refresh:
    st.cache_data.clear()

with st.spinner(f"Fetching {meta['name']} data..."):
    base_df = _load_indicators(timeframe, symbol)

# Regime detection (fast — no network)
regime = detect_regime(base_df)

# Signals with regime + news filter
override_regime = regime if block_range else None
df = generate_signals_custom(
    base_df, rsi_buy=rsi_buy, rsi_sell=rsi_sell,
    regime=override_regime, check_news=True,
)

# Only save to DB for the default asset (EURUSD) to avoid schema complexity
if symbol == config.SYMBOL:
    save_ohlcv(df[["open", "high", "low", "close", "volume"]], timeframe)
    save_signals(df, timeframe)

# Backtest
result = run_backtest(df, initial_capital=float(initial_capital),
                      position_size_pct=position_size, sl_atr_mult=sl_mult, tp_atr_mult=tp_mult)

# ── Regime + MTF row ──────────────────────────────────────────────────────────
col_reg, col_mtf = st.columns([1, 2])

with col_reg:
    rc = regime_color(regime)
    st.markdown(
        f"<div style='background:{rc}22;border:1px solid {rc};border-radius:8px;"
        f"padding:10px 16px;text-align:center'>"
        f"<span style='font-size:1.4em'>{regime_emoji(regime)}</span>"
        f"<br><b style='color:{rc};font-size:1.1em'>{regime}</b>"
        f"<br><small style='color:#aaa'>ADX = {base_df['adx'].iloc[-1]:.1f}</small>"
        f"</div>",
        unsafe_allow_html=True,
    )

with col_mtf:
    with st.spinner("Calculando confluencia MTF..."):
        mtf = _load_mtf(symbol)

    _sig_label = {1: "BUY ▲", -1: "SELL ▼", 0: "HOLD —"}
    _sig_color = {1: "#26a69a", -1: "#ef5350", 0: "#888888"}
    c15, c1h, c4h, cconf = st.columns(4)
    for col, tf_key, label in [(c15, "15m", "15 min"), (c1h, "1h", "1 hora"),
                                (c4h, "4h", "4 horas"), (cconf, "confluence", "Confluencia")]:
        sv = mtf[tf_key]
        col.markdown(
            f"<div style='text-align:center;padding:6px'>"
            f"<small style='color:#aaa'>{label}</small><br>"
            f"<b style='color:{_sig_color[sv]};font-size:1.1em'>{_sig_label[sv]}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.markdown("---")

# ── KPI strip ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Return",  f"{result.total_return:+.2f}%")
k2.metric("Sharpe Ratio",  f"{result.sharpe_ratio:.3f}")
k3.metric("Max Drawdown",  f"{result.max_drawdown:.2f}%")
k4.metric("Win Rate",      f"{result.win_rate:.1f}%")
k5.metric("Total Trades",  str(result.total_trades))

# ── Current signal score ──────────────────────────────────────────────────────
last_row     = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
last_score   = float(last_row.get("score", 0.0))
last_signal  = int(last_row.get("signal", 0))
news_blocked = bool(last_row.get("news_blocked", False))
news_reason  = str(last_row.get("news_reason", ""))

if news_blocked:
    score_color  = "#ff9800"
    signal_label = f"NEWS BLOCK — {news_reason}"
else:
    score_color  = "#26a69a" if last_signal == 1 else ("#ef5350" if last_signal == -1 else "#888")
    signal_label = "BUY" if last_signal == 1 else ("SELL" if last_signal == -1 else "HOLD")

# Price formatted with asset-specific decimal places
d         = meta["decimals"]
cur_price = float(last_row["close"])
st.markdown(
    f"<div style='margin:4px 0 12px;padding:8px 16px;border-left:4px solid {score_color};"
    f"background:#1e1e1e;border-radius:4px'>"
    f"Score atual: <b style='color:{score_color};font-size:1.1em'>{last_score:.0f}/100</b> "
    f"({signal_label}) &nbsp;|&nbsp; "
    f"Preço: <b>{cur_price:.{d}f}</b> {meta['currency']}"
    f"</div>",
    unsafe_allow_html=True,
)

# ── Upcoming economic events ──────────────────────────────────────────────────
st.markdown("---")
danger_now, danger_event = is_danger_zone()

col_ev_title, col_ev_badge = st.columns([3, 1])
with col_ev_title:
    st.subheader("📅 Próximos Eventos de Alto Impacto")
with col_ev_badge:
    if danger_now:
        st.markdown(
            "<div style='background:#ef5350;color:white;border-radius:6px;"
            "padding:6px 12px;text-align:center;margin-top:8px'>"
            "🔴 ZONA DE PERIGO</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='background:#26a69a22;color:#26a69a;border:1px solid #26a69a;"
            "border-radius:6px;padding:6px 12px;text-align:center;margin-top:8px'>"
            "✅ Mercado seguro</div>",
            unsafe_allow_html=True,
        )

with st.spinner("Carregando calendário econômico..."):
    upcoming = _load_news_events()

if upcoming:
    for ev in upcoming:
        ev_time_str  = ev.event_time.strftime("%a %d/%m %H:%Mh UTC")
        minutes_away = int(
            (ev.event_time - datetime.now(timezone.utc)).total_seconds() / 60
        )
        in_window = abs(minutes_away) <= config.NEWS_FILTER_WINDOW_MINUTES
        proximity = (
            f"<span style='color:#ef5350'><b>Em {minutes_away} min</b></span>"
            if in_window
            else f"<span style='color:#aaa'>Em {minutes_away} min</span>"
        )
        border_color = "#ef5350" if in_window else "#555"
        st.markdown(
            f"<div style='padding:6px 12px;border-left:3px solid {border_color};"
            f"background:#1e1e1e;border-radius:4px;margin-bottom:6px'>"
            f"<b style='color:#ef5350'>{ev.currency}</b> &nbsp;"
            f"<b>{ev.title}</b> &nbsp;"
            f"<span style='color:#aaa'>{ev_time_str}</span> &nbsp; {proximity}"
            f"</div>",
            unsafe_allow_html=True,
        )
else:
    st.info("Nenhum evento de alto impacto (USD/EUR/BRL) encontrado nos próximos dias.")

st.markdown("---")

# ── 5-panel main chart ────────────────────────────────────────────────────────
asset_label = meta["name"]
rsi_title   = f"RSI ({config.RSI_PERIOD})  buy<{rsi_buy} / sell>{rsi_sell}"
macd_title  = f"MACD ({config.MACD_FAST}/{config.MACD_SLOW}/{config.MACD_SIGN})"
adx_title   = f"ADX ({config.ADX_WINDOW})  trend>={config.REGIME_ADX_TREND}"
stoch_title = f"Stoch ({config.STOCH_K_WINDOW},{config.STOCH_SMOOTH_K},{config.STOCH_SMOOTH_D})"

fig = make_subplots(
    rows=5, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.02,
    row_heights=[0.42, 0.14, 0.14, 0.14, 0.16],
    subplot_titles=[f"{asset_label} + Bollinger Bands", rsi_title, macd_title, adx_title, stoch_title],
)

# ── Row 1: Candlestick + BB + signals ─────────────────────────────────────────
fig.add_trace(go.Candlestick(
    x=df.index, open=df["open"], high=df["high"],
    low=df["low"], close=df["close"], name=asset_label,
    increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
), row=1, col=1)

for col_name, dash, fill, fc in [
    ("bb_upper", "dot",   None,      None),
    ("bb_mid",   "solid", None,      None),
    ("bb_lower", "dot",   "tonexty", "rgba(100,149,237,0.07)"),
]:
    kw: dict = dict(x=df.index, y=df[col_name],
                    line=dict(color="rgba(100,149,237,0.55)", dash=dash, width=1),
                    name=col_name, showlegend=False)
    if fill:
        kw["fill"] = fill; kw["fillcolor"] = fc
    fig.add_trace(go.Scatter(**kw), row=1, col=1)

buys  = df[df["signal"] ==  1]
sells = df[df["signal"] == -1]

# Signal marker scale: small % offset works for any price magnitude
_marker_offset = 0.0003 if d >= 4 else (0.002 if d >= 2 else 0.005)
fig.add_trace(go.Scatter(x=buys.index,  y=buys["low"]  * (1 - _marker_offset), mode="markers",
    marker=dict(symbol="triangle-up",   size=10, color="#26a69a"), name="Buy"),  row=1, col=1)
fig.add_trace(go.Scatter(x=sells.index, y=sells["high"] * (1 + _marker_offset), mode="markers",
    marker=dict(symbol="triangle-down", size=10, color="#ef5350"), name="Sell"), row=1, col=1)

# ── Row 2: RSI ────────────────────────────────────────────────────────────────
fig.add_trace(go.Scatter(x=df.index, y=df["rsi"],
    line=dict(color="#9c27b0", width=1.5), name="RSI"), row=2, col=1)
fig.add_hline(y=rsi_sell, line_color="red",   line_dash="dash", line_width=1, row=2, col=1)
fig.add_hline(y=rsi_buy,  line_color="green", line_dash="dash", line_width=1, row=2, col=1)

# ── Row 3: MACD ───────────────────────────────────────────────────────────────
hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["macd_hist"]]
fig.add_trace(go.Bar(x=df.index, y=df["macd_hist"], marker_color=hist_colors,
    name="Hist", showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["macd"],
    line=dict(color="#2196f3", width=1.5), name="MACD"), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"],
    line=dict(color="#ff9800", width=1.5), name="Signal"), row=3, col=1)

# ── Row 4: ADX ────────────────────────────────────────────────────────────────
fig.add_trace(go.Scatter(x=df.index, y=df["adx"],
    line=dict(color="#ffd600", width=1.5), name="ADX"), row=4, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["adx_pos"],
    line=dict(color="#26a69a", width=1, dash="dot"), name="+DI"), row=4, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["adx_neg"],
    line=dict(color="#ef5350", width=1, dash="dot"), name="-DI"), row=4, col=1)
fig.add_hline(y=config.REGIME_ADX_TREND, line_color="white",
    line_dash="dash", line_width=1, row=4, col=1)

# ── Row 5: Stochastic ─────────────────────────────────────────────────────────
fig.add_trace(go.Scatter(x=df.index, y=df["stoch_k"],
    line=dict(color="#00bcd4", width=1.5), name="%K"), row=5, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["stoch_d"],
    line=dict(color="#ff9800", width=1.5), name="%D"), row=5, col=1)
fig.add_hline(y=config.STOCH_OVERBOUGHT, line_color="red",   line_dash="dash", line_width=1, row=5, col=1)
fig.add_hline(y=config.STOCH_OVERSOLD,   line_color="green", line_dash="dash", line_width=1, row=5, col=1)

fig.update_layout(
    height=950,
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", y=1.03, x=0),
    margin=dict(l=0, r=0, t=40, b=0),
)
fig.update_yaxes(range=[0, 100], row=2, col=1)
fig.update_yaxes(range=[0, 100], row=5, col=1)

st.plotly_chart(fig, use_container_width=True)

# ── Equity curve ──────────────────────────────────────────────────────────────
st.subheader("Equity Curve")
eq     = result.equity_curve
eq_fig = go.Figure(go.Scatter(
    x=eq.index, y=eq.values, fill="tozeroy",
    fillcolor="rgba(38,166,154,0.12)",
    line=dict(color="#26a69a", width=2), name="Equity",
))
eq_fig.update_layout(template="plotly_dark", height=200,
    margin=dict(l=0, r=0, t=10, b=0),
    yaxis_title=f"Capital ({meta['currency']})")
st.plotly_chart(eq_fig, use_container_width=True)

# ── Trade log ─────────────────────────────────────────────────────────────────
if not result.trades.empty:
    st.subheader("Trade Log")
    display       = result.trades.copy()
    display["pnl"] = display["pnl"].map(lambda x: f"{meta['currency']} {x:+.{d}f}")
    st.dataframe(display, use_container_width=True, hide_index=True)
else:
    st.info("No trades generated for this parameter set.")
