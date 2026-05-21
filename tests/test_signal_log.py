"""Teste manual de logging de sinal."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.signal_logger import log_signal

signal_id = log_signal(
    symbol="VALE3.SA",
    signal="BUY",
    price_at_signal=65.42,
    features={"rsi": 28.5, "adx": 22.1, "atr": 0.83, "sentiment": 0.6},
    regime="TRENDING",
    metadata={"source": "system_1", "test": True}
)
print(f"[OK] Sinal logado: {signal_id}")

# Validar que arquivo foi criado e contém o sinal
from pathlib import Path
import json

signals_file = Path("logs/signals.jsonl")
assert signals_file.exists(), "logs/signals.jsonl não foi criado"

last_line = signals_file.read_text(encoding="utf-8").strip().split("\n")[-1]
entry = json.loads(last_line)
assert entry["signal_id"] == signal_id, "signal_id não bate"
assert entry["outcome_5min"] is None, "outcome_5min deveria ser null"
assert "features" in entry and entry["features"]["rsi"] == 28.5, "features não persistidas"
print("[OK] Estrutura JSON válida")
print(f"[OK] Conteúdo: {json.dumps(entry, indent=2, ensure_ascii=False)}")
