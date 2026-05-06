"""FinBERT sentiment analysis for financial headlines.

Model  : ProsusAI/finbert (HuggingFace)
Input  : list[str] — headlines in English or Portuguese
Output : {"label": "positive"|"negative"|"neutral", "score": float 0-1}
Cache  : in-memory dict keyed by SHA-256 hash of joined headline text
"""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

_MODEL_NAME = "ProsusAI/finbert"
_LABEL_MAP  = {"positive": "BULLISH", "negative": "BEARISH", "neutral": "NEUTRAL"}

# ── Module-level singletons (loaded once) ─────────────────────────────────────
_tokenizer = None
_model     = None
_loaded    = False

# ── In-memory results cache ───────────────────────────────────────────────────
_cache: dict[str, dict] = {}


def _ensure_loaded() -> bool:
    global _tokenizer, _model, _loaded
    if _loaded:
        return _model is not None
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        logger.info("Loading FinBERT model from HuggingFace (%s) …", _MODEL_NAME)
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model     = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
        _model.eval()
        logger.info("FinBERT loaded successfully.")
    except Exception as exc:
        logger.warning("FinBERT failed to load: %s", exc)
        _model = None
    _loaded = True
    return _model is not None


def _hash_headlines(headlines: list[str]) -> str:
    text = "\n".join(headlines)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def analyze(headlines: list[str]) -> Optional[dict]:
    """Return aggregated FinBERT sentiment for a list of headlines.

    Returns:
        {"label": "BULLISH"|"BEARISH"|"NEUTRAL", "score": float 0-1}
        or None if model unavailable.

    Results are cached by headline content so repeated calls are cheap.
    """
    if not headlines:
        return None

    cache_key = _hash_headlines(headlines)
    if cache_key in _cache:
        logger.debug("FinBERT cache hit (%s)", cache_key)
        return _cache[cache_key]

    if not _ensure_loaded():
        return None

    try:
        import torch
        import torch.nn.functional as F

        # FinBERT has a 512 token limit — truncate long headlines
        combined_text = " | ".join(headlines)[:2000]

        inputs = _tokenizer(
            combined_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        )

        with torch.no_grad():
            outputs = _model(**inputs)
            probs   = F.softmax(outputs.logits, dim=-1)[0]  # [positive, negative, neutral]

        # FinBERT label order: positive=0, negative=1, neutral=2
        id2label = _model.config.id2label  # {0: 'positive', 1: 'negative', 2: 'neutral'}

        scores   = {id2label[i]: float(probs[i]) for i in range(len(probs))}
        raw_label = max(scores, key=scores.get)  # type: ignore[arg-type]
        confidence = scores[raw_label]

        result = {
            "label":      _LABEL_MAP.get(raw_label, "NEUTRAL"),
            "score":      round(confidence, 4),
            "raw_scores": {_LABEL_MAP.get(k, k): round(v, 4) for k, v in scores.items()},
        }

        _cache[cache_key] = result
        logger.debug(
            "FinBERT result: %s (conf=%.2f) | raw=%s",
            result["label"], result["score"], result["raw_scores"],
        )
        return result

    except Exception as exc:
        logger.warning("FinBERT inference error: %s", exc)
        return None


def clear_cache() -> None:
    _cache.clear()
