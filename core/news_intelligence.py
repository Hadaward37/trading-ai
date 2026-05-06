"""News intelligence pipeline — Fase 2 + FinBERT.

Coleta    : Yahoo Finance RSS (gratuito, sem API key)
Análise   : FinBERT(primary) + Groq(secondary) → GPT-4o → Gemini Flash → keyword fallback

Pipeline:
  _fetch_headlines_rss(ticker)  →  list[str]  — headlines recentes
  _analyze_*                    →  NewsAnalysis — classificação estruturada
  get_market_sentiment(ticker)  →  NewsAnalysis — API pública com cache TTL

Output (NewsAnalysis):
  ticker              : str   — e.g. "EUR/USD", "VALE3"
  sentiment           : str   — "BULLISH" | "BEARISH" | "NEUTRAL"
  confidence          : float — 0.0–1.0
  impact              : str   — "HIGH" | "MEDIUM" | "LOW"
  key_factors         : list[str]
  headlines           : list[str]
  analyzed_at         : datetime (UTC)
  source              : str   — "finbert+groq" | "finbert" | "groq" | "gpt4o" | "gemini" | "keyword" | "no_data"
  error               : str | None
"""

from __future__ import annotations

import json
import logging
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0 Safari/537.36"
    )
}

# Maps human-readable asset names → Yahoo Finance RSS symbols
_NAME_TO_YF: dict[str, str] = {
    meta["name"]: sym for sym, meta in config.ASSET_META.items()
}


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class NewsAnalysis:
    ticker: str
    sentiment: str               # BULLISH | BEARISH | NEUTRAL
    confidence: float            # 0.0–1.0
    impact: str                  # HIGH | MEDIUM | LOW
    key_factors: list[str]
    headlines: list[str]
    analyzed_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    source: str = "no_data"      # finbert+groq | finbert | groq | gpt4o | gemini | keyword | no_data
    error: Optional[str] = None

    def score_adjustment(self, signal: int) -> float:
        """Pontos a adicionar ao score técnico baseado no alinhamento sinal×sentimento.

        BUY  + BULLISH → positivo (boost)
        BUY  + BEARISH → negativo (reduz)
        SELL + BEARISH → positivo (boost)
        SELL + BULLISH → negativo (reduz)
        """
        direction = {"BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0}.get(self.sentiment, 0.0)
        return signal * direction * self.confidence * config.SENTIMENT_SCORE_WEIGHT


_NEUTRAL = NewsAnalysis(
    ticker="", sentiment="NEUTRAL", confidence=0.0, impact="LOW",
    key_factors=[], headlines=[], source="no_data",
)


# ── Cache ─────────────────────────────────────────────────────────────────────

class _SentimentCache:
    def __init__(self) -> None:
        self._store: dict[str, tuple[NewsAnalysis, float]] = {}

    def get(self, ticker: str) -> Optional[NewsAnalysis]:
        entry = self._store.get(ticker)
        if entry is None:
            return None
        analysis, fetched_at = entry
        if (time.monotonic() - fetched_at) > config.NEWS_INTELLIGENCE_TTL_SECONDS:
            return None
        return analysis

    def set(self, ticker: str, analysis: NewsAnalysis) -> None:
        self._store[ticker] = (analysis, time.monotonic())


_cache = _SentimentCache()


# ── Step 1: coleta de headlines via RSS (gratuito) ────────────────────────────

def _fetch_headlines_rss(ticker: str, max_items: int = 8) -> list[str]:
    """Yahoo Finance RSS — retorna headlines recentes para o ticker.

    Usa o símbolo yfinance correspondente ao nome do ativo.
    Não precisa de API key.
    """
    yf_symbol = _NAME_TO_YF.get(ticker, ticker)
    url = f"https://finance.yahoo.com/rss/headline?s={yf_symbol}"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        titles: list[str] = []
        for item in root.findall(".//item"):
            title_el = item.find("title")
            if title_el is not None and title_el.text:
                titles.append(title_el.text.strip())
                if len(titles) >= max_items:
                    break
        logger.info("RSS [%s]: %d headlines coletados", ticker, len(titles))
        return titles
    except Exception as exc:
        logger.warning("RSS fetch falhou para %s: %s", ticker, exc)
        return []


# ── Step 2a: análise via FinBERT (NLP financeiro especializado) ───────────────

_SENTIMENT_SCORE = {"BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0}


def _analyze_with_finbert(headlines: list[str], ticker: str) -> Optional[NewsAnalysis]:
    """Classifica sentimento com ProsusAI/FinBERT. Retorna None em falha."""
    if not getattr(config, "FINBERT_ENABLED", False):
        return None
    try:
        from core.finbert_sentiment import analyze as finbert_analyze
    except ImportError:
        return None

    result = finbert_analyze(headlines)
    if result is None:
        return None

    sentiment  = result["label"]
    confidence = result["score"]
    raw        = result.get("raw_scores", {})

    # Impact derived from confidence level
    if confidence >= 0.80:
        impact = "HIGH"
    elif confidence >= 0.55:
        impact = "MEDIUM"
    else:
        impact = "LOW"

    return NewsAnalysis(
        ticker     = ticker,
        sentiment  = sentiment,
        confidence = confidence,
        impact     = impact,
        key_factors= [f"FinBERT: {k}={v:.0%}" for k, v in raw.items()],
        headlines  = headlines,
        source     = "finbert",
    )


# ── Step 2b: análise via GPT-4o ──────────────────────────────────────────────

_GPT_SYSTEM = (
    "Você é um analista financeiro quantitativo. "
    "Analise as manchetes fornecidas e retorne APENAS um objeto JSON com estes campos:\n"
    '  "sentiment": "BULLISH" | "BEARISH" | "NEUTRAL"\n'
    '  "confidence": número de 0.0 a 1.0\n'
    '  "impact": "HIGH" | "MEDIUM" | "LOW"\n'
    '  "key_factors": array de 2 a 4 strings breves\n'
    "Responda apenas o JSON — sem markdown, sem explicação."
)


def _analyze_with_gpt(headlines: list[str], ticker: str) -> Optional[NewsAnalysis]:
    """Classifica sentimento com GPT-4o. Retorna None em caso de falha."""
    if not config.OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI  # type: ignore[import]
    except ImportError:
        logger.warning("openai não instalado. Execute: pip install openai")
        return None

    block = "\n".join(f"- {h}" for h in headlines)
    try:
        client   = OpenAI(api_key=config.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _GPT_SYSTEM},
                {"role": "user",   "content": f"Ativo: {ticker}\n\nManchetes:\n{block}"},
            ],
            max_tokens=300,
            temperature=0.1,
        )
        data = json.loads(response.choices[0].message.content)
        return NewsAnalysis(
            ticker=ticker,
            sentiment=str(data.get("sentiment", "NEUTRAL")).upper(),
            confidence=float(data.get("confidence", 0.5)),
            impact=str(data.get("impact", "MEDIUM")).upper(),
            key_factors=list(data.get("key_factors", [])),
            headlines=headlines,
            source="gpt4o",
        )
    except Exception as exc:
        logger.warning("GPT-4o falhou para %s: %s", ticker, exc)
        return None


# ── Step 2b: análise via Groq (Llama 3.3 70B) ────────────────────────────────

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def _analyze_with_groq(headlines: list[str], ticker: str) -> Optional[NewsAnalysis]:
    """Classifica sentimento com Llama 3.3 70B via Groq. Retorna None em falha.

    Groq usa a mesma interface do OpenAI SDK — free tier: 14 400 req/dia.
    """
    if not config.GROQ_API_KEY:
        return None
    try:
        from openai import OpenAI  # type: ignore[import]
    except ImportError:
        return None

    block = "\n".join(f"- {h}" for h in headlines)
    try:
        client   = OpenAI(api_key=config.GROQ_API_KEY, base_url=_GROQ_BASE_URL)
        response = client.chat.completions.create(
            model=config.GROQ_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _GPT_SYSTEM},
                {"role": "user",   "content": f"Ativo: {ticker}\n\nManchetes:\n{block}"},
            ],
            max_tokens=300,
            temperature=0.1,
        )
        data = json.loads(response.choices[0].message.content)
        return NewsAnalysis(
            ticker=ticker,
            sentiment=str(data.get("sentiment", "NEUTRAL")).upper(),
            confidence=float(data.get("confidence", 0.5)),
            impact=str(data.get("impact", "MEDIUM")).upper(),
            key_factors=list(data.get("key_factors", [])),
            headlines=headlines,
            source="groq",
        )
    except Exception as exc:
        logger.warning("Groq falhou para %s: %s", ticker, exc)
        return None


# ── Step 2d: análise via Gemini Flash ────────────────────────────────────────

_GEMINI_PROMPT_TPL = (
    "Analise estas manchetes financeiras sobre {ticker} e retorne APENAS "
    "um objeto JSON com os campos: sentiment (BULLISH/BEARISH/NEUTRAL), "
    "confidence (0.0-1.0), impact (HIGH/MEDIUM/LOW), "
    "key_factors (array de 2 a 4 strings).\n\n"
    "Manchetes:\n{block}"
)


def _analyze_with_gemini(headlines: list[str], ticker: str) -> Optional[NewsAnalysis]:
    """Classifica sentimento com Gemini Flash (sem grounding). Retorna None em falha."""
    if not config.GEMINI_API_KEY:
        return None
    try:
        from google import genai          # type: ignore[import]
        from google.genai import types    # type: ignore[import]
    except ImportError:
        logger.warning("google-genai não instalado. Execute: pip install google-genai")
        return None

    block  = "\n".join(f"- {h}" for h in headlines)
    prompt = _GEMINI_PROMPT_TPL.format(ticker=ticker, block=block)
    try:
        client   = genai.Client(api_key=config.GEMINI_API_KEY)
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1),
        )
        text  = (response.text or "").strip()
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start < 0 or end <= start:
            logger.warning("Gemini retornou JSON inválido para %s: %s", ticker, text[:100])
            return None
        data = json.loads(text[start:end])
        return NewsAnalysis(
            ticker=ticker,
            sentiment=str(data.get("sentiment", "NEUTRAL")).upper(),
            confidence=float(data.get("confidence", 0.5)),
            impact=str(data.get("impact", "MEDIUM")).upper(),
            key_factors=list(data.get("key_factors", [])),
            headlines=headlines,
            source="gemini",
        )
    except Exception as exc:
        logger.warning("Gemini falhou para %s: %s", ticker, exc)
        return None


# ── Step 2e: fallback baseado em palavras-chave ───────────────────────────────

_BULLISH_KW = frozenset({
    "alta", "sobe", "subiu", "ganho", "rally", "forte", "recorde",
    "lucro", "superou", "positivo", "compra", "rise", "gain", "rally",
    "strong", "beat", "bullish", "surge", "high", "up", "buy",
})
_BEARISH_KW = frozenset({
    "queda", "cai", "caiu", "perde", "fraco", "risco", "mínimo",
    "prejuízo", "abaixo", "negativo", "venda", "fall", "drop", "decline",
    "weak", "miss", "bearish", "crash", "low", "down", "sell",
})


def _keyword_sentiment(headlines: list[str], ticker: str) -> NewsAnalysis:
    """Sentimento por palavras-chave — fallback quando nenhuma API está disponível."""
    text   = " ".join(headlines).lower()
    words  = set(text.split())
    bull   = len(words & _BULLISH_KW)
    bear   = len(words & _BEARISH_KW)
    total  = bull + bear

    if total == 0:
        return NewsAnalysis(
            ticker=ticker, sentiment="NEUTRAL", confidence=0.2, impact="LOW",
            key_factors=["sem sinais claros nas manchetes"], headlines=headlines,
            source="keyword",
        )

    if bull > bear * 1.3:
        sentiment, confidence = "BULLISH", min(0.55, bull / (total + 1))
    elif bear > bull * 1.3:
        sentiment, confidence = "BEARISH", min(0.55, bear / (total + 1))
    else:
        sentiment, confidence = "NEUTRAL", 0.3

    return NewsAnalysis(
        ticker=ticker, sentiment=sentiment, confidence=confidence,
        impact="LOW", key_factors=["análise por palavras-chave"],
        headlines=headlines, source="keyword",
    )


# ── Combined analysis: FinBERT(70%) + Groq(30%) ───────────────────────────────

_SENTIMENT_DIR = {"BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0}


def _blend_analyses(fb: NewsAnalysis, groq: NewsAnalysis, w_fb: float, w_gr: float) -> NewsAnalysis:
    """Weighted blend of two NewsAnalysis objects into one."""
    # Compute directional score for each
    fb_dir   = _SENTIMENT_DIR.get(fb.sentiment, 0.0)   * fb.confidence
    gr_dir   = _SENTIMENT_DIR.get(groq.sentiment, 0.0) * groq.confidence

    blended  = fb_dir * w_fb + gr_dir * w_gr
    conf     = min(1.0, abs(blended) + 0.05)  # slight floor to avoid 0

    if   blended >  0.05: sentiment = "BULLISH"
    elif blended < -0.05: sentiment = "BEARISH"
    else:                  sentiment = "NEUTRAL"

    # Use higher-impact label
    impact_rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    impact = fb.impact if impact_rank.get(fb.impact, 0) >= impact_rank.get(groq.impact, 0) else groq.impact

    return NewsAnalysis(
        ticker     = fb.ticker,
        sentiment  = sentiment,
        confidence = round(conf, 4),
        impact     = impact,
        key_factors= [f"FinBERT: {fb.sentiment}({fb.confidence:.0%})",
                      f"Groq: {groq.sentiment}({groq.confidence:.0%})"],
        headlines  = fb.headlines,
        source     = "finbert+groq",
    )


def _analyze_combined(headlines: list[str], ticker: str) -> Optional[NewsAnalysis]:
    """Try FinBERT + Groq first; fall through to GPT-4o → Gemini on failure."""
    fb_weight = getattr(config, "FINBERT_WEIGHT", 0.70)

    fb_result   = _analyze_with_finbert(headlines, ticker)
    groq_result = _analyze_with_groq(headlines, ticker)

    if fb_result and groq_result:
        return _blend_analyses(fb_result, groq_result, fb_weight, 1.0 - fb_weight)
    if fb_result:
        return fb_result
    if groq_result:
        return groq_result

    # Fall through to other providers
    return (
        _analyze_with_gpt(headlines, ticker)
        or _analyze_with_gemini(headlines, ticker)
    )


# ── Public API ────────────────────────────────────────────────────────────────

def get_market_sentiment(ticker: str) -> NewsAnalysis:
    """Retorna sentimento de mercado para *ticker*.

    Usa resultado em cache se estiver dentro do TTL.
    Pipeline: RSS (headlines) → GPT-4o → Gemini → keyword fallback.
    Nunca lança exceção — retorna NewsAnalysis NEUTRAL em caso de falha total.
    """
    if not config.NEWS_INTELLIGENCE_ENABLED:
        return _NEUTRAL

    cached = _cache.get(ticker)
    if cached is not None:
        logger.debug("Cache hit sentimento [%s]: %s", ticker, cached.sentiment)
        return cached

    logger.info("Buscando sentimento para %s …", ticker)
    headlines = _fetch_headlines_rss(ticker)

    if not headlines:
        analysis = NewsAnalysis(
            ticker=ticker, sentiment="NEUTRAL", confidence=0.0, impact="LOW",
            key_factors=["nenhuma manchete encontrada"], headlines=[],
            source="no_data",
        )
    else:
        analysis = _analyze_combined(headlines, ticker)
        if analysis is None:
            analysis = _keyword_sentiment(headlines, ticker)

    _cache.set(ticker, analysis)
    logger.info(
        "Sentimento [%s]: %s | conf=%.2f | impact=%s | fonte=%s",
        ticker, analysis.sentiment, analysis.confidence,
        analysis.impact, analysis.source,
    )
    return analysis
