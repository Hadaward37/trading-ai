"""Verifica que predict() via Keras e via ONNX dão o mesmo resultado.

Local tem TF (usa Keras); a VM usará ONNX. Este teste força os dois caminhos
sobre o mesmo dado real e compara. Read-only.
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core import lstm_model as L
from core.collector import fetch_ohlcv
from core.indicators import add_all_indicators

df = add_all_indicators(fetch_ohlcv(timeframe="1h", symbol="EURUSD=X"))

# 1) Backend automático (local = Keras)
L.invalidate_cache()
p_auto = L.predict(df)
print(f"backend auto = {L._backend!r} | predict = {p_auto}")

# 2) Forçar ONNX (simula a VM sem tensorflow)
L.invalidate_cache()
L._scaler = L.load_scaler()
L._session = L.load_onnx_session()
L._in_name = L._session.get_inputs()[0].name
L._backend = "onnx"
L._loaded = True
p_onnx = L.predict(df)
print(f"backend onnx forçado | predict = {p_onnx}")

diff = abs(p_auto - p_onnx)
print(f"|keras - onnx| = {diff:.4f}")
print("✅ OK" if diff < 0.2 else "⚠️ divergência")
