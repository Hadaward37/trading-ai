"""Status da estratégia v1.2 ao vivo — resumo em um comando (read-only).

Roda na VM: ./venv/bin/python scripts/status_v12.py
Mostra: config ativa, frescura do signals_1h, trades/PnL desde v1.2, posições
abertas, portfólio, e a decisão do último candle (por que HOLD ou trade agora).

Não escreve nada. Abre o trading.db em modo read-only.
"""
from __future__ import annotations

import sqlite3
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config

DB = _ROOT / "data" / "trading.db"
V12_SINCE = "2026-06-14"   # data do deploy v1.2 — trades a partir daqui


def _con():
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True)


def section(t): print(f"\n{'='*56}\n{t}\n{'='*56}")


def config_sanity():
    section("1. CONFIG ATIVA")
    print(f"  STRATEGY_VERSION       = {config.STRATEGY_VERSION}")
    print(f"  ASSETS                 = {list(config.ASSETS.values())}")
    print(f"  LSTM_ENABLED           = {config.LSTM_ENABLED}")
    print(f"  XGBOOST_REQUIRE_CONSENSUS = {config.XGBOOST_REQUIRE_CONSENSUS}")
    print(f"  REGIME_FILTER_ENABLED  = {getattr(config,'REGIME_FILTER_ENABLED',False)}")
    print(f"  THRESHOLD              = {config.SIGNAL_CONFIDENCE_THRESHOLD}")
    print(f"  COMISSÃO round-trip    = {config.PAPER_TRADING_COMMISSION_PCT*2*100:.3f}%")
    ok = (config.STRATEGY_VERSION == "v1.2" and config.LSTM_ENABLED
          and config.XGBOOST_REQUIRE_CONSENSUS
          and getattr(config, "REGIME_FILTER_ENABLED", False))
    print(f"  -> {'✅ config v1.2 correta' if ok else '⚠️ config NÃO bate com v1.2'}")


def freshness():
    section("2. FRESCURA DO signals_1h")
    try:
        with _con() as c:
            ts = c.execute("SELECT MAX(datetime) FROM signals_1h").fetchone()[0]
        if not ts:
            print("  ⚠️ tabela vazia"); return
        last = datetime.fromisoformat(str(ts).replace("Z", ""))
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        age_h = (datetime.now(timezone.utc) - last).total_seconds() / 3600
        flag = "✅" if age_h < 72 else "⚠️ POSSÍVEL CONGELAMENTO"
        print(f"  último registro: {ts}  ({age_h:.1f}h atrás)  {flag}")
        print("  (fim de semana de forex: até ~65h é normal)")
    except Exception as e:
        print(f"  erro: {e}")


def trades():
    section(f"3. TRADES v1.2 (entry_time >= {V12_SINCE})")
    try:
        with _con() as c:
            rows = c.execute(
                "SELECT status, direction, pnl, pnl_pct, entry_time, exit_time, final_score "
                "FROM paper_trades WHERE entry_time >= ? ORDER BY entry_time", (V12_SINCE,)
            ).fetchall()
    except Exception as e:
        print(f"  erro (tabela paper_trades?): {e}"); return
    if not rows:
        print("  0 trades desde v1.2 — aguardando reabertura do mercado (forex fecha fim de semana).")
        print("  Na 1ª leva: confira se as direções fazem sentido e o PnL acumula.")
        return
    closed = [r for r in rows if r[2] is not None and str(r[0]).upper() != "OPEN"]
    openp  = [r for r in rows if str(r[0]).upper() == "OPEN" or r[2] is None]
    print(f"  total: {len(rows)} | fechados: {len(closed)} | abertos: {len(openp)}")
    if closed:
        pnls = [r[2] for r in closed]
        wins = [p for p in pnls if p > 0]; losses = [p for p in pnls if p < 0]
        pf = (sum(wins) / -sum(losses)) if losses else float("inf")
        wr = len(wins) / len(closed) * 100
        print(f"  PnL total: {sum(pnls):+.2f} | WinRate: {wr:.1f}% | PF: {pf:.3f}")
        print(f"  (esperado no backtest: WinRate ~50%, PF bruto ~1.47 / net ~1.33)")
    for r in openp:
        print(f"  ABERTO: {r[1]} desde {r[4]} | final_score={r[6]}")


def portfolio():
    section("4. PORTFÓLIO")
    try:
        with _con() as c:
            r = c.execute("SELECT balance, initial_capital, updated_at FROM portfolio_state LIMIT 1").fetchone()
        if r:
            bal, cap, upd = r
            print(f"  saldo: {bal:.2f} | inicial: {cap:.2f} | retorno: {(bal-cap)/cap*100:+.2f}% | atualizado: {upd}")
    except Exception as e:
        print(f"  erro: {e}")


def last_bar_decision():
    section("5. DECISÃO DO ÚLTIMO CANDLE (por que HOLD ou trade)")
    try:
        from core.collector import fetch_ohlcv
        from core.indicators import add_all_indicators
        from core.signals import generate_signals_custom
        df = generate_signals_custom(
            add_all_indicators(fetch_ohlcv(timeframe="1h", symbol="EURUSD=X")),
            config.RSI_BUY, config.RSI_SELL,
            check_news=False, check_sentiment=False, ticker="EUR/USD",
        )
        r = df.iloc[-2]
        sig = {1: "BUY", -1: "SELL", 0: "HOLD"}.get(int(r["signal"]), "?")
        print(f"  candle: {df.index[-2]}")
        print(f"  signal={sig} | score={r.get('score',0):.1f} | final_score={r.get('final_score',0):.1f}")
        print(f"  lstm_prob={r.get('lstm_prob',50):.1f} | xgboost_prob={r.get('xgboost_prob',50):.1f} "
              f"(consenso precisa ambos >50 p/ BUY, <50 p/ SELL)")
        print(f"  adx={r.get('adx',0):.1f} (<35?) | atr_ratio={r.get('atr_ratio',0):.2f} (0.8-1.5?) | "
              f"rsi={r.get('rsi',0):.1f}")
        n_sig = int((df["signal"] != 0).sum())
        print(f"  sinais não-HOLD nas últimas ~12k barras: {n_sig} "
              f"(consenso+regime ativos → poucos é esperado)")
    except Exception as e:
        print(f"  (pipeline indisponível: {e})")


if __name__ == "__main__":
    print(f"STATUS v1.2 | {datetime.now():%Y-%m-%d %H:%M:%S} | db={DB.name}")
    config_sanity()
    freshness()
    trades()
    portfolio()
    last_bar_decision()
    print("\nFim. (read-only — nada foi alterado)")
