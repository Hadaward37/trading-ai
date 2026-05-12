"""
Run Analytics -- roda regime_analytics + failure_clusters sobre todos os resultados.

Usage:
    python -X utf8 -m research.fractal_lab.research.run_analytics
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from research.fractal_lab.analytics.regime_analytics import RegimeAnalytics, load_all_results
from research.fractal_lab.analytics.failure_clusters import FailureClusterer

if __name__ == "__main__":
    results_root = Path("research/results/batch_runs")
    output_dir   = Path("research/results/regime_analytics")

    print("\n" + "=" * 60)
    print("  FRACTAL LAB ANALYTICS PIPELINE")
    print("=" * 60)

    all_rows = load_all_results(results_root)
    print(f"  Total combinations loaded: {len(all_rows)}")

    print("\n[1/2] Running Regime Analytics...")
    RegimeAnalytics(output_dir=output_dir).run(results_root)

    print("\n[2/2] Running Failure Cluster Analysis...")
    FailureClusterer(output_dir=output_dir).run(all_rows)

    print(f"\n  Output: {output_dir}")
    print("=" * 60)
