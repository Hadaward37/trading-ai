"""Streamlit dashboard — multi-asset trading signals, regime & multi-timeframe.

Tabs:
  📈 Sinais & Análise  — gráfico, indicadores, backtest, eventos econômicos
  💼 Paper Trading     — carteira virtual, P&L, histórico de operações
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
from core.paper_trading import VirtualPortfolio
from core.regime import detect_regime, regime_color, regime_emoji
from core.signals import generate_signals_custom
from core.trade_journal import daily_summary, export_daily_csv
from core.lstm_trainer import load_metrics as lstm_load_metrics
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

    asset_name = st.selectbox("Ativo", list(config.ASSETS), index=0)
    symbol     = config.ASSETS[asset_name]
    meta       = config.ASSET_META[symbol]

    st.markdown(
        f"<div style='background:#1e1e1e;border-radius:6px;padding:6px 10px;"
        f"font-size:0.85em;color:#aaa;margin-bottom:4px'>"
        f"{meta['flag']} <b style='color:#eee'>{meta['name']}</b> &nbsp;|&nbsp; "
        f"{meta['exchange']} &nbsp;|&nbsp; {meta['currency']}"
        f"</div>",
        unsafe_allow_html=True,
    )

    timeframe = st.selectbox("Timeframe", list(config.TIMEFRAMES), index=1)

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


# ── Dynamic title ─────────────────────────────────────────────────────────────
st.title(f"📈 Trading AI — {meta['flag']} {meta['name']}")


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_data(ttl=config.CACHE_TTL_SECONDS, show_spinner=False)
def _load_indicators(tf: str, sym: str) -> pd.DataFrame:
    return add_all_indicators(fetch_ohlcv(timeframe=tf, symbol=sym))


@st.cache_data(ttl=config.CACHE_TTL_SECONDS, show_spinner=False)
def _load_mtf(sym: str) -> dict:
    return get_mtf_confluence(symbol=sym)


@st.cache_data(ttl=config.NEWS_CACHE_TTL_SECONDS, show_spinner=False)
def _load_news_events():
    return get_upcoming_events(n=3)


if refresh:
    st.cache_data.clear()

with st.spinner(f"Fetching {meta['name']} data..."):
    base_df = _load_indicators(timeframe, symbol)

regime          = detect_regime(base_df)
override_regime = regime if block_range else None
df = generate_signals_custom(
    base_df, rsi_buy=rsi_buy, rsi_sell=rsi_sell,
    regime=override_regime, check_news=True,
)

if symbol == config.SYMBOL:
    save_ohlcv(df[["open", "high", "low", "close", "volume"]], timeframe)
    save_signals(df, timeframe)

result = run_backtest(df, initial_capital=float(initial_capital),
                      position_size_pct=position_size, sl_atr_mult=sl_mult, tp_atr_mult=tp_mult)

d = meta["decimals"]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_signals, tab_lstm, tab_paper = st.tabs(["📈 Sinais & Análise", "🤖 LSTM", "💼 Paper Trading"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Sinais & Análise
# ══════════════════════════════════════════════════════════════════════════════
with tab_signals:

    # ── Regime + MTF ─────────────────────────────────────────────────────────
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

    # ── KPI strip ─────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Return",  f"{result.total_return:+.2f}%")
    k2.metric("Sharpe Ratio",  f"{result.sharpe_ratio:.3f}")
    k3.metric("Max Drawdown",  f"{result.max_drawdown:.2f}%")
    k4.metric("Win Rate",      f"{result.win_rate:.1f}%")
    k5.metric("Total Trades",  str(result.total_trades))

    # ── Current signal score ──────────────────────────────────────────────────
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

    # ── Upcoming events ───────────────────────────────────────────────────────
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
            in_window    = abs(minutes_away) <= config.NEWS_FILTER_WINDOW_MINUTES
            proximity    = (
                f"<span style='color:#ef5350'><b>Em {minutes_away} min</b></span>"
                if in_window else f"<span style='color:#aaa'>Em {minutes_away} min</span>"
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

    # ── 5-panel chart ─────────────────────────────────────────────────────────
    asset_label = meta["name"]
    rsi_title   = f"RSI ({config.RSI_PERIOD})  buy<{rsi_buy} / sell>{rsi_sell}"
    macd_title  = f"MACD ({config.MACD_FAST}/{config.MACD_SLOW}/{config.MACD_SIGN})"
    adx_title   = f"ADX ({config.ADX_WINDOW})  trend>={config.REGIME_ADX_TREND}"
    stoch_title = f"Stoch ({config.STOCH_K_WINDOW},{config.STOCH_SMOOTH_K},{config.STOCH_SMOOTH_D})"

    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        row_heights=[0.42, 0.14, 0.14, 0.14, 0.16],
        subplot_titles=[f"{asset_label} + Bollinger Bands", rsi_title,
                        macd_title, adx_title, stoch_title],
    )

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
    _moff = 0.0003 if d >= 4 else (0.002 if d >= 2 else 0.005)
    fig.add_trace(go.Scatter(x=buys.index,  y=buys["low"]  * (1 - _moff), mode="markers",
        marker=dict(symbol="triangle-up",   size=10, color="#26a69a"), name="Buy"),  row=1, col=1)
    fig.add_trace(go.Scatter(x=sells.index, y=sells["high"] * (1 + _moff), mode="markers",
        marker=dict(symbol="triangle-down", size=10, color="#ef5350"), name="Sell"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["rsi"],
        line=dict(color="#9c27b0", width=1.5), name="RSI"), row=2, col=1)
    fig.add_hline(y=rsi_sell, line_color="red",   line_dash="dash", line_width=1, row=2, col=1)
    fig.add_hline(y=rsi_buy,  line_color="green", line_dash="dash", line_width=1, row=2, col=1)

    hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["macd_hist"]]
    fig.add_trace(go.Bar(x=df.index, y=df["macd_hist"], marker_color=hist_colors,
        name="Hist", showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd"],
        line=dict(color="#2196f3", width=1.5), name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"],
        line=dict(color="#ff9800", width=1.5), name="Signal"), row=3, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["adx"],
        line=dict(color="#ffd600", width=1.5), name="ADX"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["adx_pos"],
        line=dict(color="#26a69a", width=1, dash="dot"), name="+DI"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["adx_neg"],
        line=dict(color="#ef5350", width=1, dash="dot"), name="-DI"), row=4, col=1)
    fig.add_hline(y=config.REGIME_ADX_TREND, line_color="white",
        line_dash="dash", line_width=1, row=4, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["stoch_k"],
        line=dict(color="#00bcd4", width=1.5), name="%K"), row=5, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["stoch_d"],
        line=dict(color="#ff9800", width=1.5), name="%D"), row=5, col=1)
    fig.add_hline(y=config.STOCH_OVERBOUGHT, line_color="red",   line_dash="dash", line_width=1, row=5, col=1)
    fig.add_hline(y=config.STOCH_OVERSOLD,   line_color="green", line_dash="dash", line_width=1, row=5, col=1)

    fig.update_layout(
        height=950, template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.03, x=0),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=5, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # ── Equity curve ──────────────────────────────────────────────────────────
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

    # ── Trade log ─────────────────────────────────────────────────────────────
    if not result.trades.empty:
        st.subheader("Trade Log (Backtest)")
        display = result.trades.copy()
        display["pnl"] = display["pnl"].map(lambda x: f"{meta['currency']} {x:+.{d}f}")
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.info("No trades generated for this parameter set.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LSTM Neural Network
# ══════════════════════════════════════════════════════════════════════════════
with tab_lstm:
    st.subheader("🤖 LSTM — Rede Neural de Direção do Próximo Candle")

    lstm_prob_val = float(df.iloc[-1].get("lstm_prob", 50.0))
    lstm_metrics  = lstm_load_metrics()

    # ── LSTM probability gauge ────────────────────────────────────────────────
    col_gauge, col_meta = st.columns([1, 2])

    with col_gauge:
        direction_label = "ALTA" if lstm_prob_val > 55 else ("BAIXA" if lstm_prob_val < 45 else "NEUTRO")
        gauge_color     = "#26a69a" if lstm_prob_val > 55 else ("#ef5350" if lstm_prob_val < 45 else "#ff9800")

        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=lstm_prob_val,
            delta={"reference": 50, "increasing": {"color": "#26a69a"}, "decreasing": {"color": "#ef5350"}},
            number={"suffix": "%", "font": {"size": 36}},
            title={"text": f"Prob. de Alta<br><b style='color:{gauge_color}'>{direction_label}</b>"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar":  {"color": gauge_color},
                "steps": [
                    {"range": [0,  40], "color": "#2c1a1a"},
                    {"range": [40, 60], "color": "#1e1e1e"},
                    {"range": [60, 100],"color": "#1a2c1a"},
                ],
                "threshold": {"line": {"color": "white", "width": 2}, "value": 50},
            },
        ))
        gauge.update_layout(template="plotly_dark", height=280,
                            margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(gauge, use_container_width=True)

    with col_meta:
        if lstm_metrics:
            acc   = lstm_metrics.get("val_accuracy", 0)
            loss  = lstm_metrics.get("val_loss", 0)
            epoch = lstm_metrics.get("best_epoch", "?")
            total = lstm_metrics.get("epochs_run", "?")
            n_tr  = lstm_metrics.get("n_train", "?")
            n_val = lstm_metrics.get("n_val", "?")
            ts    = lstm_metrics.get("trained_at", "?")[:16].replace("T", " ")

            st.metric("Acuracia (val)", f"{acc:.1%}", delta=f"{acc - 0.5:.1%} vs baseline")
            st.metric("Val Loss",  f"{loss:.4f}")

            st.markdown(
                f"<div style='background:#1e1e1e;border-radius:8px;padding:10px 14px;"
                f"border-left:4px solid #2196f3;font-size:0.88em'>"
                f"<b>Treinado em:</b> {ts} UTC<br>"
                f"<b>Melhor epoch:</b> {epoch} / {total}<br>"
                f"<b>Amostras:</b> {n_tr} treino | {n_val} validacao<br>"
                f"<b>Lookback:</b> {lstm_metrics.get('lookback', 60)} candles<br>"
                f"<b>Features:</b> {', '.join(lstm_metrics.get('features', []))}"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.warning("Modelo LSTM ainda nao treinado. Execute: `python -m core.lstm_trainer`")

    # ── Accuracy history chart ────────────────────────────────────────────────
    if lstm_metrics and lstm_metrics.get("accuracy_history"):
        st.markdown("---")
        st.subheader("Acuracia por Epoch (validacao)")

        acc_hist  = lstm_metrics["accuracy_history"]
        loss_hist = lstm_metrics.get("loss_history", [])
        epochs_x  = list(range(1, len(acc_hist) + 1))

        acc_fig = make_subplots(rows=1, cols=2,
                                subplot_titles=["Val Accuracy", "Val Loss"])
        acc_fig.add_trace(go.Scatter(x=epochs_x, y=acc_hist,
            line=dict(color="#26a69a", width=2), name="Accuracy"), row=1, col=1)
        acc_fig.add_hline(y=0.5, line_color="#ff9800", line_dash="dash",
                          line_width=1, row=1, col=1,
                          annotation_text="baseline 50%")

        if loss_hist:
            acc_fig.add_trace(go.Scatter(x=epochs_x, y=loss_hist,
                line=dict(color="#ef5350", width=2), name="Loss"), row=1, col=2)

        acc_fig.update_layout(template="plotly_dark", height=280,
                              margin=dict(l=0, r=0, t=40, b=0),
                              showlegend=False)
        st.plotly_chart(acc_fig, use_container_width=True)

    # ── Weight breakdown ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Composicao do Score (pesos v3)")

    weights = {
        "RSI":   config.SCORE_WEIGHT_RSI,
        "MACD":  config.SCORE_WEIGHT_MACD,
        "BB":    config.SCORE_WEIGHT_BB,
        "ADX":   config.SCORE_WEIGHT_ADX,
        "Stoch": config.SCORE_WEIGHT_STOCH,
        "LSTM":  getattr(config, "SCORE_WEIGHT_LSTM", 21),
    }
    colors = ["#9c27b0", "#2196f3", "#00bcd4", "#ffd600", "#ff9800", "#26a69a"]

    pie_fig = go.Figure(go.Pie(
        labels=list(weights.keys()),
        values=list(weights.values()),
        marker_colors=colors,
        hole=0.4,
        textinfo="label+percent",
    ))
    pie_fig.update_layout(template="plotly_dark", height=300,
                          margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(pie_fig, use_container_width=True)

    # ── Retrain button ────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🔁 Retreinar modelo LSTM agora", use_container_width=True):
        with st.spinner("Treinando LSTM... (pode levar 1-2 min)"):
            try:
                from core.lstm_trainer import train as lstm_train
                m = lstm_train(force=True)
                st.success(
                    f"Treinamento concluido! Acuracia: {m.get('val_accuracy', 0):.1%} | "
                    f"Loss: {m.get('val_loss', 0):.4f}"
                )
                st.cache_data.clear()
                st.rerun()
            except Exception as exc:
                st.error(f"Erro no treinamento: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Paper Trading
# ══════════════════════════════════════════════════════════════════════════════
with tab_paper:

    portfolio = VirtualPortfolio()
    metrics   = portfolio.get_metrics()

    curr = meta["currency"]

    # ── KPI strip ─────────────────────────────────────────────────────────────
    p1, p2, p3, p4 = st.columns(4)

    pnl_delta = f"{metrics['total_pnl_pct']:+.2f}%"
    p1.metric(
        f"Saldo ({curr})",
        f"{metrics['balance']:,.2f}",
        delta=pnl_delta,
        delta_color="normal",
    )
    p2.metric(
        f"P&L Total ({curr})",
        f"{metrics['total_pnl']:+,.2f}",
        delta=pnl_delta,
        delta_color="normal",
    )
    p3.metric("Win Rate", f"{metrics['win_rate']:.1f}%",
              delta=f"{metrics['wins']}W / {metrics['losses']}L")
    p4.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%",
              delta=f"{metrics['open_positions']} abertas")

    st.markdown("---")

    # ── Daily summary ─────────────────────────────────────────────────────────
    today_summary = daily_summary()
    st.markdown(
        f"<div style='background:#1e1e1e;border-radius:8px;padding:10px 16px;"
        f"border-left:4px solid #2196f3;margin-bottom:12px'>"
        f"📅 <b>Hoje ({today_summary['date']}):</b> &nbsp;"
        f"{today_summary['total_trades']} sinais &nbsp;|&nbsp; "
        f"{today_summary['closed_trades']} fechados &nbsp;|&nbsp; "
        f"P&amp;L: <b style='color:{'#26a69a' if today_summary['total_pnl'] >= 0 else '#ef5350'}'>"
        f"{curr} {today_summary['total_pnl']:+.2f}</b> &nbsp;|&nbsp; "
        f"WinRate: {today_summary['win_rate']:.0f}%"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Open positions ────────────────────────────────────────────────────────
    open_trades = portfolio.get_open_trades()
    st.subheader(f"Posições Abertas ({len(open_trades)})")

    if not open_trades.empty:
        display_open = open_trades[[
            "asset_name", "direction", "entry_price", "stop_loss",
            "take_profit", "position_value", "entry_time", "sentiment",
        ]].copy()
        display_open.columns = [
            "Ativo", "Direção", "Entrada", "Stop Loss",
            "Take Profit", "Valor (R$)", "Hora Entrada", "Sentimento",
        ]
        st.dataframe(display_open, use_container_width=True, hide_index=True)
    else:
        st.info("Nenhuma posição aberta no momento. O scheduler abrirá posições quando houver sinais.")

    st.markdown("---")

    # ── Balance history chart ─────────────────────────────────────────────────
    st.subheader("Evolução do Saldo")
    history = portfolio.get_balance_history()

    if len(history) > 1:
        bal_fig = go.Figure()
        bal_fig.add_trace(go.Scatter(
            x=history["timestamp"], y=history["balance"],
            fill="tozeroy",
            fillcolor="rgba(33,150,243,0.10)",
            line=dict(color="#2196f3", width=2),
            name="Saldo",
        ))
        bal_fig.add_hline(
            y=metrics["initial_capital"],
            line_color="#ff9800", line_dash="dash", line_width=1,
            annotation_text="Capital inicial",
        )
        bal_fig.update_layout(
            template="plotly_dark", height=220,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title=f"Saldo ({curr})",
        )
        st.plotly_chart(bal_fig, use_container_width=True)
    else:
        st.info("Histórico de saldo ainda vazio — aguardando primeiras operações.")

    st.markdown("---")

    # ── Closed trades table ───────────────────────────────────────────────────
    closed_trades = portfolio.get_closed_trades(n=30)
    st.subheader(f"Últimas Operações Fechadas ({min(len(closed_trades), 30)})")

    if not closed_trades.empty:
        display_closed = closed_trades[[
            "asset_name", "direction", "entry_price", "exit_price",
            "exit_reason", "pnl", "pnl_pct", "exit_time", "sentiment",
        ]].copy()

        # Color-coded PnL
        def _fmt_pnl(row):
            sign = "+" if row["pnl"] >= 0 else ""
            return f"{curr} {sign}{row['pnl']:.2f} ({row['pnl_pct']:+.2f}%)"

        display_closed["resultado"] = display_closed.apply(_fmt_pnl, axis=1)
        display_closed = display_closed.drop(columns=["pnl", "pnl_pct"])
        display_closed.columns = [
            "Ativo", "Direção", "Entrada", "Saída",
            "Motivo", "Hora Saída", "Sentimento", "Resultado",
        ]
        st.dataframe(display_closed, use_container_width=True, hide_index=True)
    else:
        st.info("Nenhuma operação fechada ainda.")

    st.markdown("---")

    # ── Export & Reset ────────────────────────────────────────────────────────
    col_exp, col_rst = st.columns([2, 1])

    with col_exp:
        if st.button("📥 Exportar CSV hoje", use_container_width=True):
            csv_path = export_daily_csv()
            st.success(f"CSV exportado: {csv_path}")

    with col_rst:
        if st.button("🔄 Reset carteira", use_container_width=True, type="secondary"):
            if st.session_state.get("confirm_reset"):
                portfolio.reset()
                st.session_state["confirm_reset"] = False
                st.success("Carteira resetada com capital inicial.")
                st.rerun()
            else:
                st.session_state["confirm_reset"] = True
                st.warning("Clique novamente para confirmar o reset.")
