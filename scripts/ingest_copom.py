"""Copom macro context ingestor.

Strategy:
  1. Fetch SELIC rate history from BCB open data API (always works)
  2. Try to download Copom ata PDFs from BCB (if accessible)
  3. Analyze tone: hawkish (rising/stable-high rates) or dovish (falling rates)
  4. Synthesize context document and ingest into Pythex ChromaDB
  5. Save data/copom_tone.json for fast lookup by macro_context.py

Run:
    python scripts/ingest_copom.py
    python scripts/ingest_copom.py --force   # re-ingest even if fresh
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

_ROOT = Path(__file__).resolve().parent.parent
_PYTHEX_ROOT = Path(r"C:\Users\dudut\pythex-ia-engine")
sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TONE_PATH    = _ROOT / "data" / "copom_tone.json"
PDF_DIR      = _ROOT / "data" / "copom_pdfs"
PYTHEX_ORG   = "copom"
SELIC_API    = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.4189/dados?formato=json&dataInicial=01/01/2023"
BCB_HEADERS  = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


# ── SELIC data fetching ───────────────────────────────────────────────────────

def fetch_selic_history(n_months: int = 12) -> list[dict]:
    """Fetch last N months of SELIC target rate from BCB API."""
    try:
        r = requests.get(SELIC_API, headers=BCB_HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
        return data[-n_months:]
    except Exception as exc:
        logger.error("Failed to fetch SELIC from BCB: %s", exc)
        return []


def analyze_selic_tone(history: list[dict]) -> dict:
    """Infer Copom tone from SELIC rate trajectory."""
    if len(history) < 3:
        return {"tone": "neutral", "confidence": "low"}

    rates  = [float(d["valor"].replace(",", ".")) for d in history]
    recent = rates[-3:]
    prev   = rates[-6:-3] if len(rates) >= 6 else rates[:3]

    avg_recent = sum(recent) / len(recent)
    avg_prev   = sum(prev)   / len(prev)
    delta      = avg_recent  - avg_prev
    latest     = rates[-1]

    if delta < -0.3:
        tone = "dovish"
        confidence = "high" if delta < -0.8 else "medium"
    elif delta > 0.3:
        tone = "hawkish"
        confidence = "high" if delta > 0.8 else "medium"
    elif latest >= 13.0:
        tone = "hawkish"       # high absolute rate, even if stable
        confidence = "medium"
    elif latest <= 7.0:
        tone = "dovish"
        confidence = "medium"
    else:
        tone = "neutral"
        confidence = "low"

    return {
        "tone":       tone,
        "confidence": confidence,
        "latest_rate": latest,
        "delta_3m":   round(delta, 2),
        "rates_6m":   [round(r, 2) for r in rates[-6:]],
        "dates_6m":   [d["data"] for d in history[-6:]],
    }


# ── PDF download (best-effort) ────────────────────────────────────────────────

BCB_PDF_PATTERNS = [
    "https://www.bcb.gov.br/content/copom/NotasReuniao/{n}/R{n}-POIT.pdf",
    "https://www.bcb.gov.br/content/copom/NotasReuniao/NotaCP{n}Port.pdf",
]

def _estimate_meeting_range() -> range:
    """Estimate likely Copom meeting numbers for current year."""
    # Copom has met ~8x/year since 1996. By 2026 ~283 meetings.
    year = datetime.now().year
    base = 200 + (year - 2022) * 8   # rough estimate
    return range(max(base - 4, 260), base + 4)

def try_download_pdfs() -> list[Path]:
    """Attempt to download Copom ata PDFs. Returns list of downloaded paths."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for n in _estimate_meeting_range():
        for pat in BCB_PDF_PATTERNS:
            url = pat.format(n=n)
            dest = PDF_DIR / f"copom_{n}.pdf"
            if dest.exists():
                downloaded.append(dest)
                break
            try:
                r = requests.head(url, headers=BCB_HEADERS, timeout=5, allow_redirects=True)
                if r.status_code == 200 and "pdf" in r.headers.get("Content-Type", "").lower():
                    resp = requests.get(url, headers=BCB_HEADERS, timeout=30)
                    dest.write_bytes(resp.content)
                    logger.info("Downloaded ata %d from %s", n, url)
                    downloaded.append(dest)
                    break
            except Exception:
                pass
    logger.info("PDFs downloaded: %d", len(downloaded))
    return downloaded


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract plain text from a Copom ata PDF."""
    try:
        import fitz
        doc  = fitz.open(str(pdf_path))
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text[:20_000]   # cap at 20K chars
    except Exception as exc:
        logger.warning("Failed to extract PDF %s: %s", pdf_path.name, exc)
        return ""


def keyword_tone_from_text(text: str) -> str | None:
    """Extract hawkish/dovish tone from ata text via keyword counting."""
    if not text:
        return None
    text_l = text.lower()

    hawkish = ["elevação", "aumento da selic", "aperto monetário", "cautela adicional",
               "inflação acima", "vigilância", "alerta", "aperto adicional", "manutenção do aperto"]
    dovish  = ["redução da selic", "corte", "afrouxamento", "desinflação", "inflação convergindo",
               "espaço para redução", "flexibilização", "alívio monetário", "queda da selic"]

    h = sum(text_l.count(kw) for kw in hawkish)
    d = sum(text_l.count(kw) for kw in dovish)

    if h > d + 2:
        return "hawkish"
    if d > h + 2:
        return "dovish"
    return None


# ── LLM tone confirmation ─────────────────────────────────────────────────────

def llm_infer_tone(selic_data: dict) -> str:
    """Use GPT-4o-mini to synthesize Copom tone from SELIC trajectory."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return selic_data.get("tone", "neutral")
    try:
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv(_ROOT / ".env")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

        rates  = selic_data.get("rates_6m", [])
        dates  = selic_data.get("dates_6m", [])
        latest = selic_data.get("latest_rate", 0)
        delta  = selic_data.get("delta_3m",    0)

        hist = " | ".join(f"{d}: {r}%" for d, r in zip(dates, rates))
        prompt = (
            f"Banco Central do Brasil — histórico da taxa Selic meta:\n{hist}\n"
            f"Taxa atual: {latest}% aa | Variação 3 meses: {delta:+.2f}pp\n\n"
            "Com base nessa trajetória, o Copom está sendo HAWKISH (aperto monetário, "
            "juros altos/subindo) ou DOVISH (afrouxamento, juros caindo/baixos)?\n"
            "Responda em UMA palavra: hawkish, dovish, ou neutral."
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5, temperature=0,
        )
        import re
        raw  = resp.choices[0].message.content.strip().lower()
        word = re.sub(r"[^a-z]", "", raw.split()[0])  # strip punctuation
        return word if word in ("hawkish", "dovish", "neutral") else selic_data.get("tone", "neutral")
    except Exception as exc:
        logger.warning("LLM tone failed: %s", exc)
        return selic_data.get("tone", "neutral")


# ── ChromaDB ingestion via Pythex ─────────────────────────────────────────────

def build_context_document(selic_data: dict, tone: str, pdf_texts: list[str]) -> str:
    """Build the text document to ingest into Pythex ChromaDB."""
    rates = selic_data.get("rates_6m", [])
    dates = selic_data.get("dates_6m", [])
    hist  = "\n".join(f"  {d}: {r}% aa" for d, r in zip(dates, rates))

    tone_pt = {"hawkish": "Hawkish (aperto monetário)", "dovish": "Dovish (afrouxamento monetário)"}.get(tone, "Neutro")

    doc = (
        f"=== COPOM — Banco Central do Brasil ===\n"
        f"Data: {datetime.now().strftime('%d/%m/%Y')}\n"
        f"Tom atual: {tone_pt}\n"
        f"Taxa Selic atual: {selic_data.get('latest_rate', 'N/A')}% aa\n"
        f"Variação 3 meses: {selic_data.get('delta_3m', 0):+.2f}pp\n\n"
        f"Histórico Selic (últimos 6 meses):\n{hist}\n\n"
        f"Interpretação:\n"
        f"- Tom {tone_pt}\n"
    )
    if tone == "hawkish":
        doc += (
            "- Copom mantendo ou elevando juros para combater inflação\n"
            "- Impacto: BRL tende a se fortalecer | Ações BR tendem a cair\n"
            "- Setores mais afetados: varejo, construção, bancos (crédito mais caro)\n"
        )
    elif tone == "dovish":
        doc += (
            "- Copom cortando juros, sinalizando afrouxamento monetário\n"
            "- Impacto: BRL tende a enfraquecer | Ações BR tendem a subir\n"
            "- Setores beneficiados: varejo, construção, empresas alavancadas\n"
        )

    if pdf_texts:
        doc += f"\n=== Trechos das Atas do Copom ===\n"
        for i, txt in enumerate(pdf_texts[:3], 1):
            doc += f"\n--- Ata {i} ---\n{txt[:2000]}\n"

    return doc


def ingest_into_pythex(document_text: str) -> bool:
    """Save document to Pythex's client folder and rebuild ChromaDB."""
    try:
        pythex_client_dir = _PYTHEX_ROOT / "clientes" / PYTHEX_ORG / "pdfs"
        pythex_client_dir.mkdir(parents=True, exist_ok=True)

        # Write as .txt (Pythex ingest.py supports TextLoader)
        txt_path = pythex_client_dir / "copom_macro_context.txt"
        txt_path.write_text(document_text, encoding="utf-8")
        logger.info("Context document saved to Pythex: %s", txt_path)

        # Import Pythex vectorstore
        import importlib, os as _os
        _orig_dir = _os.getcwd()
        _os.chdir(str(_PYTHEX_ROOT))
        sys.path.insert(0, str(_PYTHEX_ROOT))

        try:
            from dotenv import load_dotenv
            load_dotenv(_PYTHEX_ROOT / ".env")

            from app.vectorstore import get_paths, carregar_vectorstore, criar_vectorstore, reset_vectorstore
            paths = get_paths(PYTHEX_ORG)

            # Reset existing store and recreate
            try:
                existing = carregar_vectorstore(PYTHEX_ORG)
                reset_vectorstore(PYTHEX_ORG, existing)
            except Exception:
                pass

            vs = criar_vectorstore(PYTHEX_ORG)
            logger.info("Pythex ChromaDB updated for org '%s'", PYTHEX_ORG)
            return True
        finally:
            _os.chdir(_orig_dir)

    except Exception as exc:
        logger.warning("Pythex ChromaDB rebuild failed: %s — file written OK, DB created lazily on first query.", exc)
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def run(force: bool = False) -> dict:
    """Full ingest pipeline. Returns final tone dict."""
    # Check cache freshness (6h TTL)
    if TONE_PATH.exists() and not force:
        try:
            cached = json.loads(TONE_PATH.read_text())
            age_h  = (datetime.now(timezone.utc).timestamp() - cached.get("updated_at_ts", 0)) / 3600
            if age_h < 6:
                logger.info("Copom tone cache fresh (%.1fh old). Use --force to refresh.", age_h)
                return cached
        except Exception:
            pass

    logger.info("=== Starting Copom ingestion pipeline ===")

    # 1. SELIC data
    history      = fetch_selic_history(12)
    selic_data   = analyze_selic_tone(history)
    logger.info("SELIC analysis: %s", selic_data)

    # 2. Try PDF download (best effort)
    pdf_paths    = try_download_pdfs()
    pdf_texts    = [t for p in pdf_paths if (t := extract_pdf_text(p))]
    pdf_tone     = None
    for txt in pdf_texts:
        t = keyword_tone_from_text(txt)
        if t:
            pdf_tone = t
            break

    # 3. LLM tone confirmation
    tone = llm_infer_tone(selic_data)
    if pdf_tone and pdf_tone != tone:
        logger.info("PDF tone (%s) overrides LLM tone (%s)", pdf_tone, tone)
        tone = pdf_tone

    # Macro impacts
    impact_brl    = "bearish" if tone == "dovish"  else ("bullish" if tone == "hawkish" else "neutral")
    impact_stocks = "bullish" if tone == "dovish"  else ("bearish" if tone == "hawkish" else "neutral")

    result = {
        "copom_tone":     tone,
        "impact_brl":     impact_brl,
        "impact_stocks":  impact_stocks,
        "selic_rate":     selic_data.get("latest_rate"),
        "selic_delta_3m": selic_data.get("delta_3m"),
        "confidence":     selic_data.get("confidence", "medium"),
        "pdfs_ingested":  len(pdf_paths),
        "updated_at":     datetime.now(timezone.utc).isoformat(),
        "updated_at_ts":  datetime.now(timezone.utc).timestamp(),
    }

    # 4. Build + ingest into Pythex ChromaDB
    doc = build_context_document(selic_data, tone, pdf_texts)
    ok  = ingest_into_pythex(doc)
    result["chromadb_ok"] = ok

    # 5. Save cache
    TONE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TONE_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    logger.info("Tone cache saved: %s", TONE_PATH)

    return result


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")

    force  = "--force" in sys.argv
    result = run(force=force)

    print("\n=== Copom Macro Context ===")
    print(f"  Tom Copom   : {result['copom_tone'].upper()}")
    print(f"  Taxa Selic  : {result['selic_rate']}% aa")
    print(f"  Delta 3m    : {result['selic_delta_3m']:+.2f}pp")
    print(f"  Impacto BRL : {result['impact_brl']}")
    print(f"  Impacto AÇÕES: {result['impact_stocks']}")
    print(f"  Confiança   : {result['confidence']}")
    print(f"  PDFs        : {result['pdfs_ingested']}")
    print(f"  ChromaDB    : {'OK' if result['chromadb_ok'] else 'FALHOU (dados em cache)'}")
    print(f"  Cache       : {TONE_PATH}")
