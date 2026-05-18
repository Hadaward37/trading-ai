"""
HMM Regime Detector — research/fractal_lab/regimes/hmm_detector.py

Hidden Markov Model (GaussianHMM) para detecção de regimes de mercado.
Drop-in replacement para o rule-based regime_detector.py.

Arquitetura "Autenticação em 2 Etapas":
  Etapa 1 (este módulo): HMM detecta regime estatístico
  Etapa 2 (strategy layer): só entra se regime correto + confirmações técnicas

Características:
  - Aprende distribuições dos dados (sem thresholds fixos)
  - Signal Hysteresis: mínimo N candles antes de aceitar mudança de regime
  - Confidence Score: probabilidade posterior máxima do estado
  - Auto-labeling: mapeia estados HMM → nomes semânticos
  - Compatível com tag_regimes() do detector rule-based

Uso:
  detector = HMMRegimeDetector(n_states=4)
  detector.fit(df_train)
  df_tagged = detector.tag(df_test)   # adiciona colunas: regime, hmm_confidence, hmm_state
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Regime labels (compatíveis com regime_detector.py) ─────────────────────────

TRENDING       = "TRENDING"
MEAN_REVERTING = "MEAN_REVERTING"
HIGH_VOLATILITY = "HIGH_VOLATILITY"
LOW_VOLATILITY  = "LOW_VOLATILITY"
BULL_RUN        = "BULL_RUN"
BEAR_RUN        = "BEAR_RUN"
CHOP            = "CHOP"

# 4-state map (padrão) e 7-state map (expandido)
LABEL_MAP_4 = {0: TRENDING, 1: MEAN_REVERTING, 2: HIGH_VOLATILITY, 3: LOW_VOLATILITY}
LABEL_MAP_7 = {
    0: BULL_RUN, 1: BEAR_RUN, 2: TRENDING,
    3: MEAN_REVERTING, 4: HIGH_VOLATILITY, 5: LOW_VOLATILITY, 6: CHOP,
}

# ── Defaults ───────────────────────────────────────────────────────────────────

DEFAULT_HYSTERESIS_BARS = 3    # mínimo de candles confirmando novo regime antes de mudar
DEFAULT_MIN_CONFIDENCE  = 0.55 # abaixo disso → regime anterior mantido (lock)
N_ITER_DEFAULT          = 200
RANDOM_STATE            = 42

# ── Feature set que alimenta o HMM ────────────────────────────────────────────
# Colunas necessárias no DataFrame (geradas por base_features.py)
REQUIRED_FEATURES = ["log_return", "atr_ratio", "ema_distance", "slope_short"]


# ══════════════════════════════════════════════════════════════════════════════
# Dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HMMConfig:
    n_states: int = 4
    n_iter: int = N_ITER_DEFAULT
    covariance_type: str = "full"          # "full" | "diag" | "tied" | "spherical"
    hysteresis_bars: int = DEFAULT_HYSTERESIS_BARS
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    random_state: int = RANDOM_STATE


@dataclass
class RegimeState:
    """Estado atual do regime com hysteresis."""
    label: str = MEAN_REVERTING
    confidence: float = 0.0
    raw_state: int = 0
    bars_in_state: int = 0
    pending_label: str = MEAN_REVERTING
    pending_count: int = 0


# ══════════════════════════════════════════════════════════════════════════════
# HMM Detector
# ══════════════════════════════════════════════════════════════════════════════

class HMMRegimeDetector:
    """
    Detecta regimes de mercado via GaussianHMM.

    Fluxo:
      1. fit(df_train)    — treina HMM nos dados históricos
      2. tag(df)          — adiciona colunas regime/hmm_confidence ao DataFrame
      3. predict_bar(row) — prediz regime de um único candle (streaming)
    """

    def __init__(self, config: Optional[HMMConfig] = None):
        self.config = config or HMMConfig()
        self.model = None
        self.label_map: dict[int, str] = {}
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std: Optional[np.ndarray] = None
        self._state = RegimeState()
        self._fitted = False

    # ── Feature extraction ─────────────────────────────────────────────────────

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extrai e normaliza features para o HMM."""
        df = df.copy()

        # log_return pode não existir no base_features — calcula aqui
        if "log_return" not in df.columns:
            df["log_return"] = np.log(df["Close"] / df["Close"].shift(1)).fillna(0.0)

        # Garante colunas necessárias com fallback
        for col in REQUIRED_FEATURES:
            if col not in df.columns:
                logger.warning("Feature '%s' ausente — usando zeros.", col)
                df[col] = 0.0

        X = df[REQUIRED_FEATURES].fillna(0.0).values.astype(float)

        # Remove infinitos
        X = np.where(np.isfinite(X), X, 0.0)
        return X

    def _fit_scaler(self, X: np.ndarray) -> np.ndarray:
        self._scaler_mean = X.mean(axis=0)
        self._scaler_std  = X.std(axis=0) + 1e-8
        return (X - self._scaler_mean) / self._scaler_std

    def _apply_scaler(self, X: np.ndarray) -> np.ndarray:
        if self._scaler_mean is None:
            return X
        return (X - self._scaler_mean) / self._scaler_std

    # ── Auto-labeling ──────────────────────────────────────────────────────────

    def _auto_label_states(self) -> None:
        """
        Mapeia estados HMM aprendidos → nomes semânticos com base nas means
        das distribuições Gaussianas.

        Heurística:
          - log_return mean alto positivo → BULL_RUN / TRENDING
          - log_return mean alto negativo → BEAR_RUN
          - atr_ratio mean alto           → HIGH_VOLATILITY
          - atr_ratio mean baixo          → LOW_VOLATILITY
          - demais                        → MEAN_REVERTING / CHOP
        """
        if self.model is None:
            return

        means = self.model.means_          # shape: (n_states, n_features)
        n = self.config.n_states

        # Índice de cada feature em REQUIRED_FEATURES
        idx_ret = REQUIRED_FEATURES.index("log_return")
        idx_atr = REQUIRED_FEATURES.index("atr_ratio")

        ret_means = means[:, idx_ret]
        atr_means = means[:, idx_atr]

        # Normaliza para ranking relativo
        ret_rank = ret_means.argsort()          # do menor ao maior retorno médio
        atr_rank = atr_means.argsort()          # do menor ao maior volatilidade média

        label_map: dict[int, str] = {}

        if n == 4:
            # Maior atr → HIGH_VOL, menor atr → LOW_VOL
            label_map[int(atr_rank[-1])] = HIGH_VOLATILITY
            label_map[int(atr_rank[0])]  = LOW_VOLATILITY
            # Dos restantes: maior retorno → TRENDING, outro → MEAN_REVERTING
            remaining = [s for s in range(n) if s not in label_map]
            ret_remaining = [(s, ret_means[s]) for s in remaining]
            ret_remaining.sort(key=lambda x: abs(x[1]), reverse=True)
            label_map[ret_remaining[0][0]] = TRENDING
            label_map[ret_remaining[1][0]] = MEAN_REVERTING

        elif n == 7:
            label_map[int(atr_rank[-1])] = HIGH_VOLATILITY
            label_map[int(atr_rank[0])]  = LOW_VOLATILITY
            label_map[int(ret_rank[-1])] = BULL_RUN
            label_map[int(ret_rank[0])]  = BEAR_RUN
            remaining = [s for s in range(n) if s not in label_map]
            # Ordena restantes por |slope| médio se disponível
            ret_abs = [(s, abs(ret_means[s])) for s in remaining]
            ret_abs.sort(key=lambda x: x[1], reverse=True)
            labels_rest = [TRENDING, MEAN_REVERTING, CHOP]
            for i, (s, _) in enumerate(ret_abs):
                label_map[s] = labels_rest[min(i, len(labels_rest) - 1)]

        else:
            # Fallback genérico
            for i in range(n):
                label_map[i] = f"REGIME_{i}"

        self.label_map = label_map
        logger.info("Auto-label mapa: %s", label_map)
        logger.info("Means (log_ret): %s", dict(enumerate(ret_means.round(6))))
        logger.info("Means (atr_rat): %s", dict(enumerate(atr_means.round(4))))

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "HMMRegimeDetector":
        """
        Treina o GaussianHMM nos dados históricos.

        Args:
            df: DataFrame com colunas OHLCV + features do base_features.py
        """
        try:
            from hmmlearn import hmm as hmmlearn_hmm
        except ImportError:
            raise ImportError("Execute: pip install hmmlearn")

        X_raw = self._build_features(df)
        X = self._fit_scaler(X_raw)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = hmmlearn_hmm.GaussianHMM(
                n_components=self.config.n_states,
                covariance_type=self.config.covariance_type,
                n_iter=self.config.n_iter,
                random_state=self.config.random_state,
            )
            model.fit(X)

        self.model = model
        self._fitted = True
        self._auto_label_states()

        logger.info(
            "HMM treinado — %d estados, convergiu=%s, log-likelihood=%.4f",
            self.config.n_states,
            model.monitor_.converged,
            model.monitor_.history[-1] if model.monitor_.history else float("nan"),
        )
        return self

    # ── Predict (batch) ────────────────────────────────────────────────────────

    def tag(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona colunas ao DataFrame:
          - hmm_state:      estado bruto do HMM (inteiro)
          - hmm_raw_regime: regime antes do hysteresis
          - hmm_confidence: probabilidade posterior máxima (0..1)
          - regime:         regime final após hysteresis + confidence gate

        Args:
            df: DataFrame com features (pode ser o mesmo de treino)

        Returns:
            Cópia do DataFrame com colunas adicionadas.
        """
        if not self._fitted:
            raise RuntimeError("Chame fit() antes de tag().")

        df = df.copy()
        X_raw = self._build_features(df)
        X = self._apply_scaler(X_raw)

        # Probabilidades posteriores: shape (n_samples, n_states)
        log_proba = self.model.predict_proba(X)
        raw_states = log_proba.argmax(axis=1)
        confidences = log_proba.max(axis=1)

        df["hmm_state"]      = raw_states
        df["hmm_raw_regime"] = [self.label_map.get(s, f"REGIME_{s}") for s in raw_states]
        df["hmm_confidence"] = confidences

        # Aplica hysteresis e confidence gate barra a barra
        regimes = self._apply_hysteresis_vectorized(
            raw_states, confidences, list(df["hmm_raw_regime"])
        )
        df["regime"] = regimes

        return df

    def _apply_hysteresis_vectorized(
        self,
        raw_states: np.ndarray,
        confidences: np.ndarray,
        raw_labels: list[str],
    ) -> list[str]:
        """
        Signal Hysteresis vetorizado.

        Regras:
          1. Confiança < min_confidence → mantém regime atual (lock)
          2. Novo regime diferente do atual → inicia contagem de pendência
          3. Pendência atinge hysteresis_bars candles seguidos → aceita mudança
        """
        h = self.config.hysteresis_bars
        min_conf = self.config.min_confidence
        n = len(raw_states)

        current_label = raw_labels[0] if raw_labels else MEAN_REVERTING
        pending_label = current_label
        pending_count = 0

        result = []
        for i in range(n):
            label = raw_labels[i]
            conf  = confidences[i]

            # Confiança baixa → congela
            if conf < min_conf:
                result.append(current_label)
                pending_count = 0
                pending_label = current_label
                continue

            if label == current_label:
                # Mesmo regime → confirma, reseta pendência
                pending_count = 0
                pending_label = current_label
                result.append(current_label)
            else:
                # Regime diferente → acumula pendência
                if label == pending_label:
                    pending_count += 1
                else:
                    pending_label = label
                    pending_count = 1

                if pending_count >= h:
                    # Hysteresis satisfeito → aceita novo regime
                    current_label = pending_label
                    pending_count = 0

                result.append(current_label)

        return result

    # ── Predict (streaming — um candle por vez) ────────────────────────────────

    def predict_bar(self, row: pd.Series) -> dict:
        """
        Prediz regime para um único candle em modo streaming.

        Mantém estado interno com hysteresis entre chamadas.
        Use para integração com Observer / live feed.

        Returns:
            {
                "regime":      str,   # regime final (com hysteresis)
                "raw_regime":  str,   # regime sem hysteresis
                "confidence":  float,
                "locked":      bool,  # True se confiança baixa trancou o estado
                "bars_in_state": int,
            }
        """
        if not self._fitted:
            raise RuntimeError("Chame fit() antes de predict_bar().")

        # Monta vetor de features
        df_tmp = pd.DataFrame([row])
        X_raw = self._build_features(df_tmp)
        X = self._apply_scaler(X_raw)

        proba = self.model.predict_proba(X)[0]
        raw_state = int(proba.argmax())
        confidence = float(proba.max())
        raw_label = self.label_map.get(raw_state, f"REGIME_{raw_state}")

        s = self._state
        locked = confidence < self.config.min_confidence

        if locked:
            # Confiança baixa → mantém estado atual
            s.pending_count = 0
            s.pending_label = s.label
        elif raw_label == s.label:
            # Confirmação do regime atual
            s.bars_in_state += 1
            s.pending_count = 0
            s.pending_label = s.label
        else:
            # Potencial mudança de regime
            if raw_label == s.pending_label:
                s.pending_count += 1
            else:
                s.pending_label = raw_label
                s.pending_count = 1

            if s.pending_count >= self.config.hysteresis_bars:
                s.label = s.pending_label
                s.bars_in_state = s.pending_count
                s.pending_count = 0

        s.confidence = confidence
        s.raw_state  = raw_state

        return {
            "regime":        s.label,
            "raw_regime":    raw_label,
            "confidence":    confidence,
            "locked":        locked,
            "bars_in_state": s.bars_in_state,
        }

    # ── Diagnóstico ────────────────────────────────────────────────────────────

    def regime_report(self, df_tagged: pd.DataFrame) -> dict:
        """
        Relatório de distribuição de regimes + score de qualidade.

        Retorna dict com:
          - distribution: % por regime
          - hysteresis_stability: % de candles com regime estável (== raw)
          - mean_confidence: confiança média global
          - low_confidence_pct: % de barras com conf < min_confidence
        """
        if "regime" not in df_tagged.columns:
            raise ValueError("DataFrame não taggeado. Chame tag() primeiro.")

        n = len(df_tagged)
        dist = (df_tagged["regime"].value_counts() / n).round(4).to_dict()

        stability = (
            (df_tagged["regime"] == df_tagged["hmm_raw_regime"]).sum() / n
        )
        mean_conf = df_tagged["hmm_confidence"].mean()
        low_conf_pct = (
            df_tagged["hmm_confidence"] < self.config.min_confidence
        ).sum() / n

        return {
            "n_bars":              n,
            "distribution":        dist,
            "hysteresis_stability": round(float(stability), 4),
            "mean_confidence":     round(float(mean_conf), 4),
            "low_confidence_pct":  round(float(low_conf_pct), 4),
            "n_states":            self.config.n_states,
            "hysteresis_bars":     self.config.hysteresis_bars,
            "min_confidence":      self.config.min_confidence,
        }

    def print_report(self, df_tagged: pd.DataFrame) -> None:
        r = self.regime_report(df_tagged)
        print("\n── HMM Regime Report ──────────────────────────────")
        print(f"  Barras analisadas : {r['n_bars']}")
        print(f"  Estados HMM       : {r['n_states']}")
        print(f"  Hysteresis (bars) : {r['hysteresis_bars']}")
        print(f"  Min. confiança    : {r['min_confidence']:.0%}")
        print(f"  Confiança média   : {r['mean_confidence']:.1%}")
        print(f"  Baixa confiança   : {r['low_confidence_pct']:.1%} das barras (travadas)")
        print(f"  Estabilidade      : {r['hysteresis_stability']:.1%} (regime == raw_regime)")
        print("\n  Distribuição de regimes:")
        for regime, pct in sorted(r["distribution"].items(), key=lambda x: -x[1]):
            bar = "█" * int(pct * 40)
            print(f"    {regime:<20} {pct:.1%}  {bar}")
        print("────────────────────────────────────────────────────\n")

    # ── Persistência ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Salva modelo em disco (joblib). Requer: pip install joblib"""
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model":        self.model,
            "label_map":    self.label_map,
            "scaler_mean":  self._scaler_mean,
            "scaler_std":   self._scaler_std,
            "config":       self.config,
        }
        joblib.dump(payload, path)
        logger.info("Modelo salvo em %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "HMMRegimeDetector":
        """Carrega modelo salvo com save()."""
        import joblib
        payload = joblib.load(path)
        det = cls(config=payload["config"])
        det.model        = payload["model"]
        det.label_map    = payload["label_map"]
        det._scaler_mean = payload["scaler_mean"]
        det._scaler_std  = payload["scaler_std"]
        det._fitted      = True
        return det


# ══════════════════════════════════════════════════════════════════════════════
# Hard Cooldown helper (nível estratégia, não regime)
# ══════════════════════════════════════════════════════════════════════════════

class HardCooldown:
    """
    Proíbe reentrada por N horas após saída de trade.

    Uso:
        cooldown = HardCooldown(hours=48)
        cooldown.trigger()                   # após fechar trade
        if cooldown.is_active():             # antes de abrir novo trade
            return  # bloqueado
    """

    def __init__(self, hours: int = 48):
        self.hours = hours
        self._triggered_at: Optional[pd.Timestamp] = None

    def trigger(self, at: Optional[pd.Timestamp] = None) -> None:
        self._triggered_at = at or pd.Timestamp.utcnow()

    def is_active(self, now: Optional[pd.Timestamp] = None) -> bool:
        if self._triggered_at is None:
            return False
        now = now or pd.Timestamp.utcnow()
        elapsed = (now - self._triggered_at).total_seconds() / 3600
        return elapsed < self.hours

    def remaining_hours(self, now: Optional[pd.Timestamp] = None) -> float:
        if not self.is_active(now):
            return 0.0
        now = now or pd.Timestamp.utcnow()
        elapsed = (now - self._triggered_at).total_seconds() / 3600
        return max(0.0, self.hours - elapsed)

    def reset(self) -> None:
        self._triggered_at = None
