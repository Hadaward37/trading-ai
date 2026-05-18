"""
HMM Runner — exemplo de uso do HMMRegimeDetector.

Comando:
    python -X utf8 -m research.fractal_lab.regimes.hmm_runner

Requer:
    pip install hmmlearn joblib
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ── Imports do fractal_lab ─────────────────────────────────────────────────────

from research.fractal_lab.features.base_features import compute_features
from research.fractal_lab.regimes.hmm_detector import (
    HMMConfig,
    HMMRegimeDetector,
    HardCooldown,
)
from research.fractal_lab.regimes.regime_detector import regime_counts  # reutiliza helper


def _load_sample_data() -> pd.DataFrame:
    """Carrega dados do fractal_cache (qualquer CSV disponível)."""
    candidates = sorted(Path("data/fractal_cache").glob("*.csv"))
    if not candidates:
        print("Nenhum CSV encontrado em data/fractal_cache/.")
        print("Execute: .\\venv\\Scripts\\python scripts/ingest_historical.py")
        sys.exit(1)

    path = candidates[0]
    print(f"Carregando: {path}")
    df = pd.read_csv(path, parse_dates=["time"] if "time" in pd.read_csv(path, nrows=1).columns else True)

    # Normaliza nome da coluna de tempo
    for col in ("time", "datetime", "date", "timestamp"):
        if col in df.columns:
            df = df.rename(columns={col: "time"}).set_index("time")
            break

    # Normaliza OHLCV
    rename = {c: c.title() for c in df.columns if c.lower() in ("open", "high", "low", "close", "volume")}
    df = df.rename(columns=rename)

    print(f"  {len(df)} barras | colunas: {list(df.columns[:8])}")
    return df


def run_hmm_experiment(n_states: int = 4, hysteresis_bars: int = 3) -> None:
    print("\n" + "=" * 60)
    print(f"HMM Regime Detection — {n_states} estados, hysteresis={hysteresis_bars}")
    print("=" * 60)

    # 1. Carrega e prepara dados
    df_raw = _load_sample_data()
    df = compute_features(df_raw)
    df = df.dropna()

    print(f"\nBarras após feature engineering: {len(df)}")

    # 2. Split temporal 70/30 (sem data leakage)
    split = int(len(df) * 0.70)
    df_train = df.iloc[:split]
    df_test  = df.iloc[split:]
    print(f"Train: {len(df_train)} | Test: {len(df_test)}")

    # 3. Treina HMM
    config = HMMConfig(
        n_states=n_states,
        hysteresis_bars=hysteresis_bars,
        min_confidence=0.55,
    )
    detector = HMMRegimeDetector(config=config)
    detector.fit(df_train)

    # 4. Testa no holdout
    df_tagged = detector.tag(df_test)
    detector.print_report(df_tagged)

    # 5. Compara com rule-based
    from research.fractal_lab.regimes.regime_detector import tag_regimes
    df_rule = tag_regimes(df_test.copy())
    print("── Rule-based (comparação) ─────────────────────────────")
    rule_dist = regime_counts(df_rule)
    for r, pct in sorted(rule_dist.items(), key=lambda x: -x[1]):
        print(f"  {r:<20} {pct:.1%}")
    print()

    # 6. Demonstra Hard Cooldown
    cooldown = HardCooldown(hours=48)
    cooldown.trigger()
    print(f"Hard Cooldown ativo: {cooldown.is_active()}")
    print(f"Horas restantes: {cooldown.remaining_hours():.1f}h")

    # 7. Salva modelo
    save_path = Path("data/hmm_model.joblib")
    detector.save(save_path)
    print(f"\nModelo salvo em: {save_path}")

    # 8. Demonstra predict_bar (streaming)
    print("\n── predict_bar (últimas 5 barras) ──────────────────────")
    detector2 = HMMRegimeDetector.load(save_path)
    for i in range(-5, 0):
        result = detector2.predict_bar(df_test.iloc[i])
        print(
            f"  [{i:+d}] regime={result['regime']:<20} "
            f"conf={result['confidence']:.2f} "
            f"locked={result['locked']} "
            f"bars_in={result['bars_in_state']}"
        )

    print("\nExperimento concluído.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HMM Regime Detector — runner")
    parser.add_argument("--states",      type=int, default=4, help="Número de estados HMM (4 ou 7)")
    parser.add_argument("--hysteresis",  type=int, default=3, help="Mínimo de barras para aceitar novo regime")
    args = parser.parse_args()

    run_hmm_experiment(n_states=args.states, hysteresis_bars=args.hysteresis)
