"""Converte o LSTM Keras (.h5) para ONNX e VERIFICA que a inferência bate.

Objetivo: rodar o LSTM na VM de 1GB via onnxruntime (~50MB), sem tensorflow.
Risco zero: local, não toca em produção. Só escreve models/lstm_eurusd_1h.onnx.

Uso: .\\venv\\Scripts\\python -X utf8 scripts\\convert_lstm_onnx.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core import lstm_model as L

ONNX_PATH = _ROOT / "models" / "lstm_eurusd_1h.onnx"
OPSET = 15


def convert(model) -> None:
    import tensorflow as tf
    import tf2onnx

    spec = (tf.TensorSpec((None, L.LOOKBACK, L.N_FEATURES), tf.float32, name="input"),)
    # Caminho 1: from_keras (mais simples)
    try:
        tf2onnx.convert.from_keras(model, input_signature=spec, opset=OPSET,
                                   output_path=str(ONNX_PATH))
        print("[convert] from_keras OK")
        return
    except Exception as exc:
        print(f"[convert] from_keras falhou ({type(exc).__name__}: {str(exc)[:120]}) — tentando SavedModel")

    # Caminho 2: export SavedModel (Keras 3) + tf2onnx CLI
    with tempfile.TemporaryDirectory() as td:
        sm_dir = Path(td) / "sm"
        try:
            model.export(str(sm_dir))            # Keras 3
        except Exception:
            tf.saved_model.save(model, str(sm_dir))
        cmd = [sys.executable, "-m", "tf2onnx.convert", "--saved-model", str(sm_dir),
               "--output", str(ONNX_PATH), "--opset", str(OPSET)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-1500:]); print(r.stderr[-1500:])
            raise SystemExit("[convert] SavedModel->ONNX falhou")
        print("[convert] SavedModel->ONNX OK")


def real_windows(n: int = 40):
    """Gera n janelas (1,60,8) escaladas a partir de dados reais EUR/USD."""
    from core.collector import fetch_ohlcv
    from core.indicators import add_all_indicators
    df = add_all_indicators(fetch_ohlcv(timeframe="1h", symbol="EURUSD=X"))
    feats = L._build_features(df)
    if feats is None:
        raise SystemExit("Sem features suficientes")
    scaler = L.load_scaler()
    if scaler is not None:
        feats = scaler.transform(feats)
    feats = feats.astype(np.float32)
    idxs = np.linspace(L.LOOKBACK, len(feats), n, endpoint=False, dtype=int)
    return [feats[i - L.LOOKBACK:i].reshape(1, L.LOOKBACK, L.N_FEATURES) for i in idxs]


def main() -> None:
    if not L.MODEL_PATH.exists():
        raise SystemExit(f"Modelo Keras não encontrado: {L.MODEL_PATH}")
    model = L.load_model()
    if model is None:
        raise SystemExit("Falha ao carregar o modelo Keras")
    print(f"[main] Keras carregado. Convertendo -> {ONNX_PATH.name} (opset {OPSET})")
    convert(model)
    print(f"[main] ONNX salvo: {ONNX_PATH} ({ONNX_PATH.stat().st_size/1024:.0f} KB)")

    print("[main] Verificando Keras vs ONNX em janelas reais...")
    import onnxruntime as ort
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    wins = real_windows(40)
    max_diff = 0.0
    for X in wins:
        k = float(model.predict(X, verbose=0)[0][0])
        o = float(sess.run(None, {in_name: X})[0].ravel()[0])
        max_diff = max(max_diff, abs(k - o))
    print(f"[main] janelas={len(wins)} | MAX |keras-onnx| = {max_diff:.2e}")
    if max_diff < 1e-4:
        print("[main] ✅ MATCH — ONNX equivale ao Keras. Seguro usar na VM.")
    else:
        print("[main] ⚠️ Diferença acima de 1e-4 — investigar antes de usar.")


if __name__ == "__main__":
    main()
