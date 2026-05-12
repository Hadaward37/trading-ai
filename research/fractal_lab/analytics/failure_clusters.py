"""
Failure Clusters -- clustering estatistico de padroes de falha.

Objetivo: descobrir padroes RECORRENTES de falha, nao apenas listar losses.
Usa KMeans (primario) e DBSCAN (outliers) sobre features de cada combinacao falhante.

Inputs : batch_summary.json de qualquer run
Outputs: failure_features.csv, cluster_assignments.csv,
         cluster_profiles.csv, failure_clusters_report.txt
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

OUTPUT_DIR = Path("research/results/regime_analytics")

# Threshold: verdicts piores que estes sao considerados "falhas"
FAIL_VERDICTS = {"FAIL", "DEGRADED", "NO_TRADES", "SKIP_NO_DATA", "ERROR"}

# Feature names and their extraction logic (applied to each row dict)
FEATURE_DEFS: Dict[str, str] = {
    "robustness":     "robustness",
    "test_pf":        "test_pf",
    "oos_pf":         "oos_pf",
    "test_sharpe":    "test_sharpe",
    "test_wr":        "test_wr",
    "pct_trending":   "pct_trending",
    "pct_high_vol":   "pct_high_vol",
    "pct_low_vol":    "pct_low_vol",
    "pct_mean_rev":   "pct_mean_rev",
    "pf_degradation": "pf_degradation",
    "mc_prob_profit": "mc_prob_profit",
}


# ── Cluster auto-naming heuristics ─────────────────────────────────────────────

_CLUSTER_NAMES = {
    "overfit":        "DATA_MINING_OVERFIT",
    "trend_trap":     "TREND_CHASING_TRAP",
    "vol_trap":       "VOLATILITY_TRAP",
    "signal_drought": "SIGNAL_DROUGHT",
    "regime_noise":   "REGIME_NOISE",
    "structural":     "STRUCTURAL_FAILURE",
}


def _name_cluster(centroid: Dict[str, float], n: int) -> str:
    """
    Auto-name a cluster based on its centroid feature values.
    Uses a simple priority-based heuristic.
    """
    deg = centroid.get("pf_degradation", 0.0)
    pf_test  = centroid.get("test_pf", 0.0)
    pf_oos   = centroid.get("oos_pf", 0.0)
    rob      = centroid.get("robustness", 0.0)
    trending = centroid.get("pct_trending", 0.0)
    high_vol = centroid.get("pct_high_vol", 0.0)
    mc_prob  = centroid.get("mc_prob_profit", 0.5)

    # Overfit: good test PF but bad OOS
    if pf_test > 1.1 and pf_oos < 0.95 and deg < -0.10:
        return _CLUSTER_NAMES["overfit"]

    # High vol trap: high_vol dominant, poor PF
    if high_vol > 0.45 and pf_test < 0.95:
        return _CLUSTER_NAMES["vol_trap"]

    # Trend trap: trending dominant, but strategy doesn't exploit it
    if trending > 0.60 and pf_test < 0.90:
        return _CLUSTER_NAMES["trend_trap"]

    # Signal drought: very few trades implied (low MC prob, low rob)
    if rob < 0.15 and mc_prob < 0.45:
        return _CLUSTER_NAMES["signal_drought"]

    # Regime noise: mixed regimes, mediocre across the board
    if abs(trending - high_vol) < 0.15 and pf_test < 1.0:
        return _CLUSTER_NAMES["regime_noise"]

    return _CLUSTER_NAMES["structural"]


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_failure_features(all_rows: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate failing rows and extract their feature matrix.

    Returns:
        meta_df  : original metadata (asset, hypothesis, verdict, ...)
        feat_df  : numeric feature matrix (same index as meta_df)
    """
    failing = [
        r for r in all_rows
        if r.get("oos_verdict", "") in FAIL_VERDICTS
        and r.get("test_trades", 0) > 0   # must have actually run
    ]

    if not failing:
        return pd.DataFrame(), pd.DataFrame()

    meta_rows = []
    feat_rows = []

    for row in failing:
        meta_rows.append({
            "asset":       row.get("asset", ""),
            "category":    row.get("category", ""),
            "hypothesis":  row.get("hypothesis", ""),
            "group":       row.get("group", ""),
            "oos_verdict": row.get("oos_verdict", ""),
        })
        feat_rows.append({
            feat: float(row.get(src, 0.0) or 0.0)
            for feat, src in FEATURE_DEFS.items()
        })

    meta_df = pd.DataFrame(meta_rows)
    feat_df = pd.DataFrame(feat_rows)

    # Clip extreme values (e.g. absurd Sharpe from degenerate trades)
    feat_df["test_sharpe"] = feat_df["test_sharpe"].clip(-3.0, 3.0)
    feat_df["test_pf"]     = feat_df["test_pf"].clip(0.0, 3.0)
    feat_df["oos_pf"]      = feat_df["oos_pf"].clip(0.0, 3.0)
    feat_df["pf_degradation"] = feat_df["pf_degradation"].clip(-2.0, 1.0)

    return meta_df, feat_df


def _normalize_features(feat_df: pd.DataFrame) -> np.ndarray:
    """Z-score normalize the feature matrix (manual, no sklearn required)."""
    X = feat_df.values.astype(float)
    means = X.mean(axis=0)
    stds  = X.std(axis=0, ddof=1)
    stds[stds < 1e-9] = 1.0
    return (X - means) / stds


# ── KMeans ────────────────────────────────────────────────────────────────────

def _kmeans_numpy(X: np.ndarray, k: int, n_init: int = 10, max_iter: int = 300, seed: int = 42):
    """Pure numpy KMeans implementation (no sklearn required)."""
    rng = np.random.default_rng(seed)
    best_inertia = float("inf")
    best_labels  = None
    best_centers = None

    for _ in range(n_init):
        # Random initialization (k-means++)
        centers = X[rng.choice(len(X), size=k, replace=False)]

        for _ in range(max_iter):
            # Assign
            dists  = np.linalg.norm(X[:, None] - centers[None], axis=2)
            labels = dists.argmin(axis=1)

            # Update
            new_centers = np.array([
                X[labels == j].mean(axis=0) if (labels == j).any() else centers[j]
                for j in range(k)
            ])

            if np.allclose(centers, new_centers, atol=1e-6):
                break
            centers = new_centers

        inertia = float(sum(
            np.linalg.norm(X[i] - centers[labels[i]]) ** 2 for i in range(len(X))
        ))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels  = labels.copy()
            best_centers = centers.copy()

    return best_labels, best_centers, best_inertia


def _choose_k(X: np.ndarray, k_range: range = range(2, 6)) -> int:
    """Choose k via elbow method (largest drop in inertia)."""
    if len(X) < max(k_range):
        return min(len(X) - 1, 2)

    inertias = {}
    for k in k_range:
        _, _, inertia = _kmeans_numpy(X, k)
        inertias[k] = inertia

    # Elbow: largest relative drop
    best_k = list(k_range)[0]
    best_drop = -1.0
    k_list = list(k_range)
    for i in range(1, len(k_list)):
        prev, curr = inertias[k_list[i - 1]], inertias[k_list[i]]
        drop = (prev - curr) / max(prev, 1e-9)
        if drop > best_drop:
            best_drop = drop
            best_k = k_list[i]

    return best_k


# ── DBSCAN ───────────────────────────────────────────────────────────────────

def _dbscan_numpy(X: np.ndarray, eps: float = 1.5, min_samples: int = 3) -> np.ndarray:
    """Pure numpy DBSCAN for outlier detection. Returns labels (-1 = noise)."""
    n = len(X)
    labels  = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    def neighbors(i):
        dists = np.linalg.norm(X - X[i], axis=1)
        return np.where(dists <= eps)[0]

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        nbrs = neighbors(i)
        if len(nbrs) < min_samples:
            labels[i] = -1  # noise
            continue
        labels[i] = cluster_id
        queue = list(nbrs)
        while queue:
            j = queue.pop()
            if not visited[j]:
                visited[j] = True
                nbrs_j = neighbors(j)
                if len(nbrs_j) >= min_samples:
                    queue.extend(nbrs_j.tolist())
            if labels[j] == -1:
                labels[j] = cluster_id
        cluster_id += 1

    return labels


# ── Cluster profiling ─────────────────────────────────────────────────────────

def profile_clusters(
    meta_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    labels: np.ndarray,
    centers_raw: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Build a profile DataFrame with one row per cluster:
    centroid values, top assets/hypotheses/groups, auto-generated name.
    """
    unique_labels = sorted(set(labels))
    records = []

    for label in unique_labels:
        mask = labels == label
        sub_meta = meta_df[mask]
        sub_feat = feat_df[mask]

        n = int(mask.sum())
        centroid = {col: round(float(sub_feat[col].mean()), 4) for col in feat_df.columns}
        cluster_name = _name_cluster(centroid, n)

        top_assets = sub_meta["asset"].value_counts().head(3).to_dict()
        top_hyps   = sub_meta["hypothesis"].value_counts().head(3).to_dict()
        top_groups = sub_meta["group"].value_counts().head(2).to_dict()
        verdicts   = sub_meta["oos_verdict"].value_counts().to_dict()

        record = {
            "cluster_id":   -1 if label == -1 else label,
            "cluster_name": "OUTLIER" if label == -1 else cluster_name,
            "n_cases":      n,
            "top_assets":   "; ".join(f"{a}({c})" for a, c in top_assets.items()),
            "top_hyps":     "; ".join(f"{h}({c})" for h, c in top_hyps.items()),
            "top_groups":   "; ".join(f"{g}({c})" for g, c in top_groups.items()),
            "verdicts":     "; ".join(f"{v}({c})" for v, c in verdicts.items()),
        }
        record.update({f"centroid_{k}": v for k, v in centroid.items()})
        records.append(record)

    return pd.DataFrame(records)


# ── Report ────────────────────────────────────────────────────────────────────

def _format_failure_report(
    meta_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    labels_km: np.ndarray,
    profiles_km: pd.DataFrame,
    labels_db: np.ndarray,
) -> str:
    n_fail   = len(meta_df)
    n_km_cls = len(set(labels_km))
    n_outliers = int((labels_db == -1).sum())

    lines = [
        "=" * 72,
        "  FAILURE CLUSTER ANALYSIS",
        f"  Generated: {datetime.utcnow().isoformat()}",
        f"  Total failures analyzed: {n_fail}",
        f"  KMeans clusters: {n_km_cls}",
        f"  DBSCAN outliers (anomalous failures): {n_outliers}",
        "=" * 72,
    ]

    for _, row in profiles_km.iterrows():
        if row["cluster_id"] == -1:
            continue
        lines += [
            "",
            f"  CLUSTER {row['cluster_id']} -- {row['cluster_name']}",
            f"  Cases: {row['n_cases']}",
            f"  Assets:      {row['top_assets']}",
            f"  Hypotheses:  {row['top_hyps']}",
            f"  Groups:      {row['top_groups']}",
            f"  Verdicts:    {row['verdicts']}",
            "  Feature centroid:",
            f"    robustness={row.get('centroid_robustness',0):.3f}  "
            f"test_pf={row.get('centroid_test_pf',0):.3f}  "
            f"oos_pf={row.get('centroid_oos_pf',0):.3f}",
            f"    pf_degradation={row.get('centroid_pf_degradation',0):+.3f}  "
            f"test_sharpe={row.get('centroid_test_sharpe',0):.3f}",
            f"    pct_trending={row.get('centroid_pct_trending',0):.1%}  "
            f"pct_high_vol={row.get('centroid_pct_high_vol',0):.1%}  "
            f"pct_mean_rev={row.get('centroid_pct_mean_rev',0):.1%}",
        ]

    # Outlier summary
    outlier_meta = meta_df[labels_db == -1]
    if len(outlier_meta) > 0:
        lines += [
            "",
            f"  DBSCAN OUTLIERS (anomalous, isolated failures) -- {len(outlier_meta)} cases",
        ]
        for _, r in outlier_meta.iterrows():
            lines.append(f"    {r['asset']:8} x {r['hypothesis']:<20} [{r['oos_verdict']}]")

    # Structural insights
    lines += [
        "",
        "=" * 72,
        "  STRUCTURAL INSIGHTS",
        "=" * 72,
    ]

    for _, row in profiles_km[profiles_km["cluster_id"] >= 0].iterrows():
        deg = row.get("centroid_pf_degradation", 0.0)
        rob = row.get("centroid_robustness", 0.0)
        name = row["cluster_name"]

        if name == "DATA_MINING_OVERFIT":
            lines.append(
                f"  [{name}] {row['n_cases']} cases: estrategia parece boa no test "
                f"mas colapsa no OOS (deg={deg:+.1%}). Provavelmente overfitting "
                f"de parametros ao periodo de treino. Nao confiar em PF test isolado."
            )
        elif name == "VOLATILITY_TRAP":
            lines.append(
                f"  [{name}] {row['n_cases']} cases: execucao concentrada em periodos "
                f"de alta volatilidade onde spread/slippage real eliminaria o edge. "
                f"Avaliar com custo de transacao explicito."
            )
        elif name == "TREND_CHASING_TRAP":
            lines.append(
                f"  [{name}] {row['n_cases']} cases: ativo em regime TRENDING mas "
                f"estrategia nao captura a direcao. Possivelmente entrada tardia "
                f"ou RR incorreto para o asset."
            )
        elif name == "SIGNAL_DROUGHT":
            lines.append(
                f"  [{name}] {row['n_cases']} cases: sinal rarissimo (rob={rob:.2f}). "
                f"N de trades insuficiente para qualquer conclusao estatistica valida. "
                f"Parametros muito restritivos ou asset incompativel com a hipotese."
            )
        elif name == "REGIME_NOISE":
            lines.append(
                f"  [{name}] {row['n_cases']} cases: regime misto sem dominancia clara. "
                f"Mercado em transicao ou hipotese regime-agnostica demais. "
                f"Adicionar filtro de regime pode recuperar edge."
            )
        else:
            lines.append(
                f"  [{name}] {row['n_cases']} cases: falha estrutural -- nenhuma feature "
                f"individualmente explica. Hipotese pode ser incompativel com o asset. "
                f"Rejeitar sem revisao."
            )

    lines.append("=" * 72)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

class FailureClusterer:
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, all_rows: List[Dict]) -> Dict[str, Any]:
        meta_df, feat_df = extract_failure_features(all_rows)

        if meta_df.empty:
            print("[FailureClusterer] No failing combinations found.")
            return {}

        print(f"[FailureClusterer] {len(meta_df)} failing combinations for clustering")

        X = _normalize_features(feat_df)

        # KMeans
        k = _choose_k(X)
        print(f"  KMeans: chosen k={k}")
        labels_km, centers_km, _ = _kmeans_numpy(X, k=k)
        profiles_km = profile_clusters(meta_df, feat_df, labels_km)

        # DBSCAN
        labels_db = _dbscan_numpy(X, eps=1.8, min_samples=3)
        n_outliers = int((labels_db == -1).sum())
        print(f"  DBSCAN: {n_outliers} outlier(s) detected")

        # Assignments
        assignments = meta_df.copy()
        assignments["cluster_km"] = labels_km
        assignments["is_dbscan_outlier"] = (labels_db == -1)
        assignments["cluster_name"] = [
            profiles_km.loc[profiles_km["cluster_id"] == lbl, "cluster_name"].iloc[0]
            if lbl in profiles_km["cluster_id"].values else "OUTLIER"
            for lbl in labels_km
        ]
        for col in feat_df.columns:
            assignments[col] = feat_df[col].values

        # Save
        assignments.to_csv(self.output_dir / "cluster_assignments.csv", index=False, encoding="utf-8")
        feat_df.to_csv(self.output_dir / "failure_features.csv", index=False, encoding="utf-8")
        profiles_km.to_csv(self.output_dir / "cluster_profiles.csv", index=False, encoding="utf-8")

        report = _format_failure_report(meta_df, feat_df, labels_km, profiles_km, labels_db)
        (self.output_dir / "failure_clusters_report.txt").write_text(report, encoding="utf-8")
        print(report)

        return {
            "n_failures":  len(meta_df),
            "n_clusters":  k,
            "n_outliers":  n_outliers,
            "assignments": assignments,
            "profiles":    profiles_km,
        }


def run_failure_clusters(all_rows: List[Dict]) -> Dict[str, Any]:
    return FailureClusterer().run(all_rows)


if __name__ == "__main__":
    from .regime_analytics import load_all_results
    rows = load_all_results()
    run_failure_clusters(rows)
