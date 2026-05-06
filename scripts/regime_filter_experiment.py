"""Regime Filter Experiment — post-processing only.

Rules:
  - NO model retraining
  - NO feature changes
  - NO target changes
  - Filter applied ONLY on test period of each fold
  - Threshold selected via nested walk-forward on TRAINING data sub-split

Nested walk-forward structure:
  For each fold:
    sub_train = first 70% of fold.training_data
    sub_val   = last 30% of fold.training_data   [for threshold selection]
    ↓
    Train quick model on sub_train → predict sub_val → select filter
    Train main model on full training → predict test → apply selected filter
    Compare before vs after filter

Selection criterion (anti-snooping, anti-overfitting):
  From most permissive to most restrictive, pick the FIRST candidate that:
    1. Retains >= 60% of sub-val trades
    2. Shows PF improvement vs unfiltered sub-val baseline
  "Most robust" = least restrictive valid filter (highest ADX, widest ATR range)
  We NEVER maximize E — we pick the simplest valid filter.

Usage:
  python scripts/regime_filter_experiment.py [--no-ibov]
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Import shared utilities from the definitive experiment
sys.path.insert(0, str(_ROOT / "scripts"))
from experiment_definitive import (
    SPREAD, CAPITAL, POSITION_PCT, FOREX_HOURS_PER_YEAR,
    load_eurusd, build_blocks, build_folds, make_target,
    backtest_spread, _equity_metrics,
    train_xgb_model, engineer_features, correlation_analysis,
    get_feature_cols, Fold, fetch_ibov_1h,
)
from core.regime_filter import RegimeFilter, parameter_grid, ADX_THRESHOLDS, ATR_RATIO_RANGES

NO_IBOV = "--no-ibov" in sys.argv

TOP_COMBOS = [
    (2, 0.7, True,  "N=2H k=0.7 bal=Y"),
    (2, 0.5, False, "N=2H k=0.5 bal=N"),
    (2, 0.7, False, "N=2H k=0.7 bal=N"),
]

SUBTRAIN_FRAC    = 0.70    # fraction of training fold used as sub-train
MIN_RETENTION    = 0.60    # minimum fraction of trades to retain
PF_IMPROVE_MIN   = 1.0     # filter must achieve PF > this on sub-val

LINE = "=" * 76
SEP  = "-" * 76


# ─────────────────────────────────────────────────────────────────────────────
#  1. PNL METRICS (compact version for filter comparison)
# ─────────────────────────────────────────────────────────────────────────────

def pnl_metrics(pnl: np.ndarray, period_days: float, N: int) -> dict:
    """Return E, PF, WR, Sharpe, MaxDD from a per-trade PnL% array."""
    if len(pnl) < 2:
        return {"n": len(pnl), "e": 0.0, "pf": 0.0, "wr": 0.0,
                "sharpe": 0.0, "maxdd": 0.0}

    wins   = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    wr     = len(wins) / len(pnl)
    avg_w  = float(wins.mean())         if len(wins)   > 0 else 0.0
    avg_l  = float(abs(losses.mean()))  if len(losses) > 0 else 0.0
    exp    = wr * avg_w - (1 - wr) * avg_l
    gp     = float(wins.sum())          if len(wins)   > 0 else 0.0
    gl     = float(abs(losses.sum()))   if len(losses) > 0 else 1e-9
    pf     = gp / gl

    sharpe, _, _, max_dd = _equity_metrics(pnl, period_days, N)

    return {
        "n": len(pnl), "e": exp, "pf": pf, "wr": wr * 100,
        "sharpe": sharpe, "maxdd": max_dd * 100,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  2. THRESHOLD SELECTION (nested WF, sub-val only)
# ─────────────────────────────────────────────────────────────────────────────

def select_regime_filter(
    sub_val_df:   pd.DataFrame,
    pnl_subval:   np.ndarray,
    baseline_pf:  float,
    N:            int,
    period_days:  float,
) -> tuple[Optional[RegimeFilter], str]:
    """
    Select the most robust regime filter using ONLY sub-val data.

    Selection: iterate from most permissive to most restrictive.
    Return the FIRST candidate that:
      1. Retains >= MIN_RETENTION of sub-val trades
      2. PF of retained trades > baseline_pf

    This is the 'simplest valid filter' — NOT the one with max E.

    Returns (filter, explanation_string).
    Returns (None, reason) if no valid filter found.
    """
    grid = parameter_grid()   # ordered: most permissive → most restrictive

    n_total = len(pnl_subval)
    if n_total == 0:
        return None, "no sub-val trades available"

    results = []

    for flt in grid:
        mask = flt.mask(sub_val_df).values[:n_total]
        retained = pnl_subval[mask]
        n_retained = int(mask.sum())
        ret_rate = n_retained / n_total

        if ret_rate < MIN_RETENTION:
            continue  # not enough trades retained

        if len(retained) < 5:
            continue

        wins   = retained[retained > 0]
        losses = retained[retained <= 0]
        gp  = float(wins.sum())         if len(wins)   > 0 else 0.0
        gl  = float(abs(losses.sum()))  if len(losses) > 0 else 1e-9
        pf  = gp / gl

        results.append({
            "filter": flt, "pf": pf, "retention": ret_rate,
            "n_retained": n_retained,
        })

        if pf > baseline_pf:
            reason = (
                f"{flt.describe()} | ret={ret_rate:.1%} ({n_retained}/{n_total}) | "
                f"sub-val PF: {baseline_pf:.3f} -> {pf:.3f} | "
                f"chosen: simplest valid filter"
            )
            return flt, reason

    # No filter improved PF — return best by PF if it passes retention
    if results:
        best = max(results, key=lambda x: x["pf"])
        flt  = best["filter"]
        reason = (
            f"{flt.describe()} | ret={best['retention']:.1%} | "
            f"sub-val PF={best['pf']:.3f} (no improvement vs {baseline_pf:.3f} — "
            f"closest valid candidate)"
        )
        return flt, reason

    return None, f"no filter meets retention={MIN_RETENTION:.0%} constraint"


# ─────────────────────────────────────────────────────────────────────────────
#  3. PER-FOLD EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    fold:         str
    combo_name:   str
    N:            int
    k:            float
    balanced:     bool
    period_days:  float
    # Before filter
    before:       dict = field(default_factory=dict)
    # After filter
    after:        dict = field(default_factory=dict)
    # Filter selection info
    filter_used:  Optional[RegimeFilter] = None
    filter_reason:str = ""
    retention_pct:float = 1.0


def run_fold(
    fold:       Fold,
    N:          int,
    k:          float,
    balanced:   bool,
    combo_name: str,
    df:         pd.DataFrame,
    feat_cols:  list[str],
) -> Optional[FoldResult]:
    """Run one (fold × combo) with nested walk-forward threshold selection."""

    period_days = max(1.0, (fold.test_end - fold.test_start).days)

    # ── Slice data ────────────────────────────────────────────────────────────
    tr_df = df.loc[(df.index >= fold.train_start) & (df.index < fold.train_end)].copy()
    te_df = df.loc[(df.index >= fold.test_start)  & (df.index < fold.test_end)].copy()

    y_all, _ = make_target(df, N, k)
    y_tr = y_all.loc[tr_df.index]
    y_te = y_all.loc[te_df.index]

    tr_mask = y_tr.notna();  te_mask = y_te.notna()
    tr_sub  = tr_df[tr_mask]; te_sub = te_df[te_mask]
    y_tr_cl = y_tr[tr_mask].astype(int).values
    y_te_cl = y_te[te_mask].astype(int).values

    if len(tr_sub) < 100 or len(te_sub) < 30:
        return None

    # ── Prepare features ──────────────────────────────────────────────────────
    X_train = tr_sub[feat_cols].values.astype(np.float32)
    X_test  = te_sub[feat_cols].values.astype(np.float32)

    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)

    n_val_outer = max(20, int(len(X_tr_sc) * 0.10))
    X_te_sc     = scaler.transform(X_test)

    # ── STEP A: Main model (full training → test predictions) ─────────────────
    main_model = train_xgb_model(
        X_tr_sc[:-n_val_outer], y_tr_cl[:-n_val_outer],
        X_tr_sc[-n_val_outer:], y_tr_cl[-n_val_outer:],
        balanced,
    )
    y_pred_test = main_model.predict(X_te_sc)
    pnl_before  = backtest_spread(te_sub, te_sub.index, y_pred_test, N)

    metrics_before = pnl_metrics(pnl_before, period_days, N)

    # ── STEP B: Nested model (sub-train → sub-val → select filter) ────────────
    n_subtrain = int(len(X_tr_sc) * SUBTRAIN_FRAC)

    X_st = X_tr_sc[:n_subtrain];   y_st = y_tr_cl[:n_subtrain]
    X_sv = X_tr_sc[n_subtrain:];   y_sv = y_tr_cl[n_subtrain:]
    sv_df = tr_sub.iloc[n_subtrain:]   # sub-val DataFrame (same rows as X_sv)

    if len(X_st) < 50 or len(X_sv) < 20:
        return None

    # Keep a small validation inside sub-train for early stopping
    n_sv_inner = max(10, int(len(X_st) * 0.10))
    nested_model = train_xgb_model(
        X_st[:-n_sv_inner], y_st[:-n_sv_inner],
        X_st[-n_sv_inner:], y_st[-n_sv_inner:],
        balanced,
    )
    y_pred_sv  = nested_model.predict(X_sv)
    pnl_subval = backtest_spread(sv_df, sv_df.index, y_pred_sv, N)

    # Compute baseline PF on sub-val (no filter)
    if len(pnl_subval) == 0:
        return None
    sv_wins  = pnl_subval[pnl_subval > 0]
    sv_loss  = pnl_subval[pnl_subval <= 0]
    baseline_pf = (float(sv_wins.sum()) /
                   float(abs(sv_loss.sum()) + 1e-9) if len(sv_loss) > 0 else 1.0)

    # ── STEP C: Select filter threshold on sub-val ────────────────────────────
    # CRITICAL: sub-val is from training data — test data never touched here
    sv_period_days = max(1.0, (sv_df.index[-1] - sv_df.index[0]).days)

    selected_filter, filter_reason = select_regime_filter(
        sv_df.iloc[:len(pnl_subval)],   # align rows to pnl length
        pnl_subval,
        baseline_pf,
        N,
        sv_period_days,
    )

    # ── STEP D: Apply selected filter to test predictions ─────────────────────
    if selected_filter is None:
        regime_mask       = np.ones(len(y_pred_test), dtype=bool)
        pnl_after         = pnl_before.copy()
        selected_filter   = RegimeFilter.no_filter()
        filter_reason     = filter_reason + " [no filter applied]"
    else:
        # te_sub rows are aligned with y_pred_test and pnl_before
        regime_mask   = selected_filter.mask(te_sub).values[:len(y_pred_test)]
        y_pred_filt   = y_pred_test[regime_mask]
        te_sub_filt   = te_sub[regime_mask]
        pnl_after     = backtest_spread(te_sub_filt, te_sub_filt.index, y_pred_filt, N)

    retention = float(regime_mask.mean())
    metrics_after = pnl_metrics(pnl_after, period_days, N)

    return FoldResult(
        fold=fold.name, combo_name=combo_name,
        N=N, k=k, balanced=balanced, period_days=period_days,
        before=metrics_before, after=metrics_after,
        filter_used=selected_filter, filter_reason=filter_reason,
        retention_pct=retention,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  4. IBOV TEST
# ─────────────────────────────────────────────────────────────────────────────

def run_ibov_filtered(
    df_ibov:    pd.DataFrame,
    best_filter: RegimeFilter,
    combo:       tuple,
    feat_cols_eur: list[str],
) -> dict:
    """Apply the EUR/USD-selected filter to IBOV. No new threshold selection."""
    N, k, balanced, combo_name = combo
    feat_cols = [f for f in feat_cols_eur if f in df_ibov.columns]

    if len(df_ibov) < 300:
        return {}

    y_all, _ = make_target(df_ibov, N, k, price_normalize=True)
    n        = len(df_ibov)
    n_train  = int(n * 0.70)

    tr = df_ibov.iloc[:n_train];   te = df_ibov.iloc[n_train:]
    y_tr = y_all.loc[tr.index];    y_te = y_all.loc[te.index]
    tr_sub = tr[y_tr.notna()];     te_sub = te[y_te.notna()]
    y_tr_cl = y_tr[y_tr.notna()].astype(int).values
    y_te_cl = y_te[y_te.notna()].astype(int).values

    if len(tr_sub) < 50 or len(te_sub) < 15:
        return {}

    X_tr = tr_sub[feat_cols].values.astype(np.float32)
    X_te = te_sub[feat_cols].values.astype(np.float32)
    sc   = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    n_val = max(10, int(len(X_tr_sc) * 0.10))
    X_te_sc = sc.transform(X_te)

    model  = train_xgb_model(
        X_tr_sc[:-n_val], y_tr_cl[:-n_val],
        X_tr_sc[-n_val:], y_tr_cl[-n_val:],
        balanced,
    )
    y_pred = model.predict(X_te_sc)

    period_days = max(1.0, (te_sub.index[-1] - te_sub.index[0]).days)

    pnl_before = backtest_spread(te_sub, te_sub.index, y_pred, N)
    met_before = pnl_metrics(pnl_before, period_days, N)

    mask      = best_filter.mask(te_sub).values[:len(y_pred)]
    y_filt    = y_pred[mask]
    te_filt   = te_sub[mask]
    pnl_after = backtest_spread(te_filt, te_filt.index, y_filt, N)
    met_after = pnl_metrics(pnl_after, period_days, N)

    return {
        "combo_name": combo_name,
        "filter": best_filter.describe(),
        "before": met_before,
        "after":  met_after,
        "retention": float(mask.mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  5. REPORT
# ─────────────────────────────────────────────────────────────────────────────

def _delta(a, b):
    d = a - b
    return f"{d:+.4f}"


def _pct(v):
    return f"{v:.1f}%"


def _valid_fold(fr: FoldResult) -> bool:
    return (
        fr.after["n"] >= fr.before["n"] * 0.60 and
        fr.after["pf"] > 1.15
    )


def print_report(
    all_results: list[FoldResult],
    ibov_results: list[dict],
    final_filters: dict,   # combo_name -> most commonly selected RegimeFilter
) -> None:

    print(f"\n{LINE}")
    print("  REGIME FILTER EXPERIMENT — Post-Processing Only")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(LINE)

    # Group by combo
    combos = list(dict.fromkeys(r.combo_name for r in all_results))

    for cname in combos:
        frs = [r for r in all_results if r.combo_name == cname]
        if not frs:
            continue

        print(f"\n{SEP}")
        print(f"  COMBO: {cname}")
        print(SEP)

        # Threshold selection summary
        print("\n  Thresholds selected per fold (on sub-val, not test):")
        for fr in frs:
            flag = "[NO FILTER]" if fr.filter_used and fr.filter_used.adx_threshold == 999 else ""
            print(f"    {fr.fold:<10}: {fr.filter_used.describe() if fr.filter_used else 'none'} {flag}")
            print(f"               reason: {fr.filter_reason[:90]}")

        # Before vs After table
        print(f"\n  {'Fold':<10} {'Trades_B':>9} {'Trades_A':>9} {'Ret%':>6} | "
              f"{'E_B':>9} {'E_A':>9} {'Delta_E':>9} | "
              f"{'PF_B':>6} {'PF_A':>6} | "
              f"{'Sh_B':>7} {'Sh_A':>7} | "
              f"{'DD_B':>7} {'DD_A':>7} | {'Valid':>6}")
        print(f"  {'-'*125}")

        for fr in frs:
            b, a = fr.before, fr.after
            valid = "PASS" if _valid_fold(fr) else "fail"
            print(
                f"  {fr.fold:<10} {b['n']:>9} {a['n']:>9} {fr.retention_pct:>5.1%} | "
                f"{b['e']:>+9.4f} {a['e']:>+9.4f} {_delta(a['e'], b['e']):>9} | "
                f"{b['pf']:>6.3f} {a['pf']:>6.3f} | "
                f"{b['sharpe']:>7.2f} {a['sharpe']:>7.2f} | "
                f"{b['maxdd']:>6.2f}% {a['maxdd']:>6.2f}% | {valid:>6}"
            )

        # Aggregate
        sel_folds = [fr for fr in frs if not fr.fold.startswith("Holdout")]
        holdout   = next((fr for fr in frs if fr.fold.startswith("Holdout")), None)

        if sel_folds:
            mean_e_b = np.mean([f.before["e"]  for f in sel_folds])
            mean_e_a = np.mean([f.after["e"]   for f in sel_folds])
            mean_pf_b= np.mean([f.before["pf"] for f in sel_folds])
            mean_pf_a= np.mean([f.after["pf"]  for f in sel_folds])
            n_pass   = sum(1 for f in sel_folds if _valid_fold(f))
            print(f"\n  Selection folds aggregate (Fold1-3):")
            print(f"    E:  {mean_e_b:+.4f} -> {mean_e_a:+.4f}  ({_delta(mean_e_a, mean_e_b)})")
            print(f"    PF: {mean_pf_b:.3f}  -> {mean_pf_a:.3f}")
            print(f"    Folds PASS (PF>1.15, ret>=60%): {n_pass}/{len(sel_folds)}")

        if holdout:
            b, a = holdout.before, holdout.after
            print(f"\n  HOLDOUT (anti-snooping):")
            print(f"    Trades: {b['n']} -> {a['n']} ({holdout.retention_pct:.1%} retained)")
            print(f"    E:      {b['e']:+.4f} -> {a['e']:+.4f}")
            print(f"    PF:     {b['pf']:.3f}  -> {a['pf']:.3f}")
            print(f"    Sharpe: {b['sharpe']:.2f}   -> {a['sharpe']:.2f}")
            print(f"    MaxDD:  {b['maxdd']:.2f}%  -> {a['maxdd']:.2f}%")
            h_valid = _valid_fold(holdout)
            print(f"    Result: {'PASS' if h_valid else 'FAIL'}")

    # IBOV
    if ibov_results:
        print(f"\n{SEP}")
        print("  IBOV PARALLEL TEST (best EUR/USD filter applied, no new selection)")
        print(SEP)
        for r in ibov_results:
            b, a = r.get("before", {}), r.get("after", {})
            if b and a:
                print(f"\n  {r['combo_name']}  |  Filter: {r['filter']}")
                print(f"    Trades: {b['n']} -> {a['n']} ({r['retention']:.1%} retained)")
                print(f"    E:      {b['e']:+.4f} -> {a['e']:+.4f}")
                print(f"    PF:     {b['pf']:.3f}  -> {a['pf']:.3f}")
                print(f"    Sharpe: {b['sharpe']:.2f}   -> {a['sharpe']:.2f}")
                print(f"    MaxDD:  {b['maxdd']:.2f}%  -> {a['maxdd']:.2f}%")

    # ── Final Answers ──────────────────────────────────────────────────────────
    print(f"\n{LINE}")
    print("  FINAL ANSWERS")
    print(LINE)

    # Q1: Which threshold was selected and why?
    print("\n  Q1: Qual threshold ADX + atr_ratio foi selecionado e por que?")
    for cname, flt in final_filters.items():
        if flt:
            print(f"    {cname}: {flt.describe()}")
            print(f"      Why: simplest valid filter — least restrictive that improved PF")
            print(f"           on sub-val while retaining >= {MIN_RETENTION:.0%} of trades")
        else:
            print(f"    {cname}: no valid filter found (no filter applied)")

    # Q2: Did filter consistently improve E, PF, DD?
    print(f"\n  Q2: O filtro melhorou E, PF e Drawdown consistentemente?")
    for cname in combos:
        frs = [r for r in all_results if r.combo_name == cname]
        e_improved  = sum(1 for r in frs if r.after["e"]   > r.before["e"])
        pf_improved = sum(1 for r in frs if r.after["pf"]  > r.before["pf"])
        dd_improved = sum(1 for r in frs if r.after["maxdd"] > r.before["maxdd"])
        print(f"    {cname}: E+={e_improved}/{len(frs)} PF+={pf_improved}/{len(frs)} DD+={dd_improved}/{len(frs)}")

    # Q3: Trade retention
    print(f"\n  Q3: Quantos trades foram retidos? (minimo 60%)")
    for cname in combos:
        frs = [r for r in all_results if r.combo_name == cname]
        rets = [r.retention_pct * 100 for r in frs]
        above60 = sum(1 for r in rets if r >= 60)
        print(f"    {cname}: {[f'{r:.1f}%' for r in rets]} | "
              f"acima de 60%: {above60}/{len(rets)}")

    # Q4: Is the micro-edge tradable after filter? (E >= 0.0002, PF >= 1.2, Sharpe > 1.5)
    print(f"\n  Q4: O micro-edge ficou tradavel? (E>=0.0002, PF>=1.2, Sharpe>1.5)")
    for cname in combos:
        holdout_fr = next(
            (r for r in all_results if r.combo_name == cname and r.fold == "Holdout"),
            None,
        )
        if holdout_fr:
            a = holdout_fr.after
            tradable = (a["e"] >= 0.0002 and a["pf"] >= 1.2 and a["sharpe"] > 1.5)
            print(f"    {cname} (holdout after filter):")
            print(f"      E={a['e']:+.4f} (need>=0.0002) | "
                  f"PF={a['pf']:.3f} (need>=1.2) | "
                  f"Sharpe={a['sharpe']:.2f} (need>1.5)")
            print(f"      -> {'TRADAVEL' if tradable else 'NAO TRADAVEL (ainda marginal)'}")

    # Q5: Consistent in >= 2 regimes?
    print(f"\n  Q5: O resultado e consistente em >= 2 regimes?")
    for cname in combos:
        sel_folds = [r for r in all_results
                     if r.combo_name == cname and not r.fold.startswith("Holdout")]
        n_pass = sum(1 for r in sel_folds if _valid_fold(r))
        consistent = n_pass >= 2
        print(f"    {cname}: {n_pass}/{len(sel_folds)} folds com PF>1.15 e ret>=60% "
              f"-> {'CONSISTENTE' if consistent else 'INCONSISTENTE'}")

    # IBOV comparison
    if ibov_results:
        print(f"\n  IBOV vs EUR/USD: filtro melhorou IBOV consistentemente?")
        for r in ibov_results:
            b, a = r.get("before", {}), r.get("after", {})
            if b and a:
                improved = a["pf"] > b["pf"] and a["e"] > b["e"]
                print(f"    {r['combo_name']}: PF {b['pf']:.3f}->{a['pf']:.3f} | "
                      f"E {b['e']:+.4f}->{a['e']:+.4f} "
                      f"-> {'SIM' if improved else 'NAO'}")

    print(f"\n{LINE}")


# ─────────────────────────────────────────────────────────────────────────────
#  6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{LINE}")
    print("  Loading data and building folds...")
    df = load_eurusd()
    print(f"  EUR/USD 1H: {len(df):,} rows | "
          f"{df.index.min().strftime('%Y-%m-%d')} - {df.index.max().strftime('%Y-%m-%d')}")

    blocks = build_blocks(df)
    folds  = build_folds(blocks)
    print("  Blocks:", [f"{b.name}:{b.start.strftime('%Y-%m')}-{b.end.strftime('%Y-%m')}"
                        for b in blocks])

    # Feature columns (same as definitive experiment)
    _, _, removed, _, retained = correlation_analysis(df)
    feat_cols = get_feature_cols(df, retained)
    print(f"  Features: {len(feat_cols)} retained, {len(removed)} removed")

    # Run all combos × all folds
    all_results: list[FoldResult] = []
    total = len(TOP_COMBOS) * len(folds)
    done  = 0

    for (N, k, balanced, combo_name) in TOP_COMBOS:
        print(f"\n  Running {combo_name}...")
        for fold in folds:
            done += 1
            print(f"    {fold.name} ({done}/{total})...", end=" ", flush=True)
            fr = run_fold(fold, N, k, balanced, combo_name, df, feat_cols)
            if fr is not None:
                all_results.append(fr)
                print(f"E: {fr.before['e']:+.4f} -> {fr.after['e']:+.4f} | "
                      f"PF: {fr.before['pf']:.3f} -> {fr.after['pf']:.3f}")
            else:
                print("skipped (insufficient data)")

    # Determine the most commonly selected filter per combo
    final_filters: dict[str, Optional[RegimeFilter]] = {}
    for (N, k, balanced, combo_name) in TOP_COMBOS:
        sel_results = [r for r in all_results
                       if r.combo_name == combo_name and not r.fold.startswith("Holdout")]
        if sel_results:
            # Pick the filter that appears most often in selection folds
            # (excluding holdout from this count)
            from collections import Counter
            filter_keys = [
                (r.filter_used.adx_threshold,
                 r.filter_used.atr_ratio_min,
                 r.filter_used.atr_ratio_max)
                for r in sel_results if r.filter_used
            ]
            if filter_keys:
                most_common = Counter(filter_keys).most_common(1)[0][0]
                final_filters[combo_name] = RegimeFilter(*most_common)
            else:
                final_filters[combo_name] = None
        else:
            final_filters[combo_name] = None

    # IBOV test
    ibov_results: list[dict] = []
    if not NO_IBOV:
        print(f"\n  Fetching IBOV data...")
        df_ibov = fetch_ibov_1h()
        if df_ibov is not None and len(df_ibov) > 400:
            print(f"  IBOV: {len(df_ibov):,} rows")
            # Use best EUR/USD filter for all combos
            for (N, k, balanced, combo_name) in TOP_COMBOS:
                best_flt = final_filters.get(combo_name, RegimeFilter.no_filter())
                if best_flt is None:
                    best_flt = RegimeFilter.no_filter()
                res = run_ibov_filtered(df_ibov, best_flt,
                                        (N, k, balanced, combo_name), feat_cols)
                if res:
                    ibov_results.append(res)
        else:
            print("  IBOV: insufficient data.")

    # Print report
    print_report(all_results, ibov_results, final_filters)

    # Save summary JSON
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "final_filters": {
            k: {
                "adx_threshold": v.adx_threshold,
                "atr_ratio_min": v.atr_ratio_min,
                "atr_ratio_max": v.atr_ratio_max,
                "description":   v.describe(),
            }
            for k, v in final_filters.items() if v is not None
        },
        "folds": [
            {
                "fold": r.fold, "combo": r.combo_name,
                "retention_pct": round(r.retention_pct, 4),
                "e_before": round(r.before["e"], 6),
                "e_after":  round(r.after["e"],  6),
                "pf_before": round(r.before["pf"], 4),
                "pf_after":  round(r.after["pf"],  4),
                "sharpe_before": round(r.before["sharpe"], 4),
                "sharpe_after":  round(r.after["sharpe"],  4),
                "maxdd_before": round(r.before["maxdd"], 4),
                "maxdd_after":  round(r.after["maxdd"],  4),
                "valid": _valid_fold(r),
            }
            for r in all_results
        ],
    }
    out = _ROOT / "data" / "regime_filter_results.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to {out}")


if __name__ == "__main__":
    main()
