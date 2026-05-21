"""
Teste do outcome_filler com sinal artificialmente antigo.
Cria um sinal com timestamp de 10min atrás e roda o filler.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from datetime import datetime, timezone, timedelta
from core.outcome_filler import fill_outcomes

# Backup do arquivo atual
signals_file = Path("logs/signals.jsonl")
backup_content = signals_file.read_text(encoding="utf-8") if signals_file.exists() else ""

# Inserir sinal antigo (10min atrás) para forçar outcome_5min
old_time = datetime.now(timezone.utc) - timedelta(minutes=10)
test_entry = {
    "signal_id": "TEST_OLD_SIGNAL",
    "timestamp_utc": old_time.isoformat(),
    "symbol": "VALE3.SA",
    "signal": "BUY",
    "price_at_signal": 65.00,
    "features": {"rsi": 28.0},
    "regime": "TRENDING",
    "metadata": {"test": True},
    "outcome_5min": None,
    "outcome_30min": None,
    "outcome_1h": None,
    "outcome_4h": None,
    "outcome_1d": None,
}

with open(signals_file, "a", encoding="utf-8") as f:
    f.write(json.dumps(test_entry) + "\n")

# Função de preço fake para teste isolado (não depender de yfinance)
def fake_price(symbol: str) -> float:
    return 66.30  # +2% vs 65.00

fill_outcomes(fake_price)

# Verificar resultado
lines = signals_file.read_text(encoding="utf-8").strip().split("\n")
for line in lines:
    entry = json.loads(line)
    if entry["signal_id"] == "TEST_OLD_SIGNAL":
        assert entry["outcome_5min"] is not None, "outcome_5min não foi preenchido"
        expected = round((66.30 - 65.00) / 65.00 * 100, 4)
        assert abs(entry["outcome_5min"] - expected) < 0.01, f"Cálculo errado: {entry['outcome_5min']} vs {expected}"
        print(f"[OK] outcome_5min calculado: {entry['outcome_5min']}% (esperado ~{expected}%)")
        break
else:
    raise AssertionError("Sinal de teste não encontrado")

# Restaurar arquivo original (remover sinal de teste)
clean_lines = [l for l in lines if json.loads(l)["signal_id"] != "TEST_OLD_SIGNAL"]
signals_file.write_text("\n".join(clean_lines) + "\n" if clean_lines else "", encoding="utf-8")
print("[OK] Arquivo restaurado, sinal de teste removido")
