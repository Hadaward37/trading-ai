"""
Fractal Lab — Comparative Report
Computes coherence + disorder scores for Fold1 vs Holdout and saves data/fractal_report.json
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# allow running from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.fractal_lab.data_loader import (
    get_period_data,
    FOLD1_START, FOLD1_END,
    HOLDOUT_START, HOLDOUT_END,
)
from research.fractal_lab.coherence import compute_coherence
from research.fractal_lab.disorder import compute_disorder

OUTPUT_PATH = Path("data/fractal_report.json")

PERIODS = {
    "fold1": {"start": FOLD1_START, "end": FOLD1_END, "label": "Fold1 (set2024–fev2025)"},
    "holdout": {"start": HOLDOUT_START, "end": HOLDOUT_END, "label": "Holdout (dez2025–mai2026)"},
}


def _summarize(metrics: dict, key: str) -> dict:
    """Strip window_scores from output dict for cleaner JSON."""
    return {k: v for k, v in metrics.items() if k != "window_scores"}


def generate_report(verbose: bool = True) -> dict:
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": "EURUSD=X",
        "note": (
            "yfinance limits: 1m ≤ 7 days back, 5m ≤ 60 days back. "
            "Fold1 (Sep2024–Feb2025) outside these limits — returns empty. "
            "Holdout: 5m covers last ~60d, 1m last ~7d."
        ),
        "periods": PERIODS,
        "results": {},
        "comparison": None,
        "answer": "",
    }

    for period, meta in PERIODS.items():
        if verbose:
            print(f"\n[fractal_lab] {meta['label']}")

        df_1m, df_5m = get_period_data(period)
        n1 = len(df_1m)
        n5 = len(df_5m)

        if verbose:
            print(f"  1m candles: {n1}  |  5m candles: {n5}")

        coherence = compute_coherence(df_1m, df_5m)
        disorder = compute_disorder(df_1m, df_5m)

        if verbose:
            if coherence["available"]:
                print(f"  coherence_score : {coherence['coherence_score']:.1f}  (n_windows={coherence['n_windows']})")
            else:
                print("  coherence_score : N/A (no data)")
            if disorder["available"]:
                print(f"  disorder_score  : {disorder['disorder_score']:.1f}  (n_windows={disorder['n_windows']})")
            else:
                print("  disorder_score  : N/A (no data)")

        # store compact window sample (first 200 points) for distribution analysis
        window_sample = {
            "coherence": coherence.get("window_scores", [])[:200],
            "disorder": disorder.get("window_scores", [])[:200],
        }

        report["results"][period] = {
            "data_points": {"1m": n1, "5m": n5},
            "coherence": _summarize(coherence, "coherence"),
            "disorder": _summarize(disorder, "disorder"),
            "window_sample": window_sample,
        }

    # Comparison
    f1 = report["results"]["fold1"]
    ho = report["results"]["holdout"]
    f1_dis = f1["disorder"].get("disorder_score")
    ho_dis = ho["disorder"].get("disorder_score")
    f1_coh = f1["coherence"].get("coherence_score")
    ho_coh = ho["coherence"].get("coherence_score")

    report["comparison"] = {
        "fold1_disorder": f1_dis,
        "holdout_disorder": ho_dis,
        "fold1_coherence": f1_coh,
        "holdout_coherence": ho_coh,
        "disorder_delta_fold1_minus_holdout": (
            round(f1_dis - ho_dis, 2) if f1_dis is not None and ho_dis is not None else None
        ),
        "coherence_delta_fold1_minus_holdout": (
            round(f1_coh - ho_coh, 2) if f1_coh is not None and ho_coh is not None else None
        ),
    }

    # Answer the key question
    if f1_dis is None and ho_dis is not None:
        ho_coh_str = f"{ho_coh:.1f}" if ho_coh is not None else "N/A"
        answer = (
            f"INCONCLUSIVO — Fold1 sem dados 1m/5m via yfinance (limite: 60d para 5m, 7d para 1m). "
            f"Holdout atual: disorder={ho_dis:.1f}, coherence={ho_coh_str}. "
            f"Para comparação completa, exportar histórico do MT5 e rodar compute_coherence/compute_disorder diretamente."
        )
    elif f1_dis is not None and ho_dis is not None:
        delta = f1_dis - ho_dis
        if delta > 2:
            answer = (
                f"SIM — disorder_score era maior no Fold1 ({f1_dis:.1f}) que no Holdout ({ho_dis:.1f}), "
                f"delta={delta:+.1f}. Consistente com diagnóstico: mercado caótico → sinais de reversão falham."
            )
        elif delta < -2:
            answer = (
                f"NÃO — disorder_score era MENOR no Fold1 ({f1_dis:.1f}) que no Holdout ({ho_dis:.1f}), "
                f"delta={delta:+.1f}. A desordem fractal não explica as perdas do Sistema 1."
            )
        else:
            answer = (
                f"NEUTRO — disorder_score Fold1 ({f1_dis:.1f}) ≈ Holdout ({ho_dis:.1f}), "
                f"delta={delta:+.1f} (dentro de ±2). Sem evidência de diferença estrutural de desordem."
            )
    else:
        answer = "DADOS INSUFICIENTES — nenhum período com dados 1m/5m disponíveis."

    report["answer"] = answer

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    if verbose:
        print(f"\n{'='*64}")
        print("PERGUNTA: disorder_score era maior antes dos trades ruins do Sistema 1?")
        print(f"RESPOSTA: {answer}")
        print(f"{'='*64}")
        print(f"\n[fractal_lab] Report salvo em {OUTPUT_PATH}")

    return report


if __name__ == "__main__":
    generate_report()
