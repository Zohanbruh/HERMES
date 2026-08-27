#!/usr/bin/env python3
"""Ablation study: how much does each control actually contribute?

Two complementary designs are run on the hardened portfolio P2.

*Leave-one-out* removes one control and keeps the rest, which measures the
**marginal** contribution of a control given that everything else is in place.
This is the number a hospital faces when deciding whether to cut a line item.

*Single-control* activates one control alone at its P2 coverage, which measures
its **standalone** contribution.  The two orderings differ, and the gap between
them is the substitution effect between overlapping controls.

Outputs
-------
figures/fig_loo.pdf, fig_only.pdf
paper/tables/tab_loo.tex, tab_only.tex
results/ablation_loo.csv, ablation_only.csv
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from _common import (N_RUNS_MAIN, ROOT, SEEDS, fmt_p, log, save_csv,
                     write_macros, write_table)
from hermes import controls as ctrl
from hermes import statsx as sx
from hermes import viz
from hermes.simulator import SimulationParams, compile_graph, simulate

BASE = "P2_hardened"
LABELS = {c.key: c.label for c in ctrl.CONTROLS}


def _pooled(portfolio, params, cg) -> Dict[str, np.ndarray]:
    losses, cdis, ehr = [], [], []
    for seed in SEEDS:
        r = simulate(portfolio, params, seed=seed, cg=cg)
        losses.append(r.loss)
        cdis.append(r.care_disruption)
        ehr.append(r.ehr_compromise)
    return {"loss": np.concatenate(losses),
            "cdi": np.concatenate(cdis),
            "ehr": np.concatenate(ehr)}


def _paired_stats(variant, reference, params, cg, seed_list=SEEDS):
    diffs, ps, deltas = [], [], []
    for seed in seed_list:
        rv = simulate(variant, params, seed=seed, cg=cg)
        rr = simulate(reference, params, seed=seed, cg=cg)
        d, lo, hi = sx.paired_bootstrap_diff(rv.loss, rr.loss, n_boot=1500,
                                             seed=seed)
        diffs.append((d, lo, hi))
        ps.append(sx.mannwhitney_p(rv.loss, rr.loss))
        deltas.append(sx.cliffs_delta(rv.loss, rr.loss))
    return (float(np.mean([x[0] for x in diffs])),
            float(np.mean([x[1] for x in diffs])),
            float(np.mean([x[2] for x in diffs])),
            float(np.median(ps)),
            float(np.mean(deltas)))


def main() -> None:
    cg = compile_graph()
    params = SimulationParams(n_runs=N_RUNS_MAIN)
    base_portfolio = ctrl.PORTFOLIOS[BASE]
    base = _pooled(base_portfolio, params, cg)
    base_mean = float(base["loss"].mean())
    base_cost = ctrl.annual_cost(ctrl.portfolio_vector(base_portfolio))
    log(f"{BASE}: mean loss {base_mean/1e6:.3f} M, cost {base_cost/1e6:.2f} M/yr")

    # ------------------------------------------------------- leave-one-out
    rows: List[dict] = []
    for key, variant in ctrl.leave_one_out(base_portfolio).items():
        pooled = _pooled(variant, params, cg)
        d, lo, hi, p, delta = _paired_stats(variant, base_portfolio, params, cg)
        cost_saved = base_cost - ctrl.annual_cost(ctrl.portfolio_vector(variant))
        rows.append({
            "key": key, "label": LABELS[key],
            "coverage": base_portfolio[key],
            "mean_loss": float(pooled["loss"].mean()),
            "delta": d, "lo": lo, "hi": hi, "p": p, "cliff": delta,
            "delta_pct": 100.0 * d / base_mean,
            "lo_pct": 100.0 * lo / base_mean,
            "hi_pct": 100.0 * hi / base_mean,
            "ehr_delta": float(pooled["ehr"].mean() - base["ehr"].mean()),
            "cdi_pct": 100.0 * (pooled["cdi"].mean() / base["cdi"].mean() - 1.0),
            "cost_saved": cost_saved,
            "loss_per_cost": d / max(cost_saved, 1.0),
        })
        log(f"  LOO {key:10s} +{rows[-1]['delta_pct']:7.1f}% loss")

    adj = sx.holm_bonferroni([r["p"] for r in rows])
    for r, a in zip(rows, adj):
        r["p_adj"] = a
    loo = pd.DataFrame(rows).sort_values("delta_pct", ascending=False)
    save_csv(loo, "ablation_loo.csv")

    loo_tab = pd.DataFrame([{
        "Control removed": r["label"],
        "Cov.": f"{r['coverage']:.2f}",
        "Mean loss (M)": f"{r['mean_loss'] / 1e6:.3f}",
        r"$\Delta$ loss (M)": f"{r['delta'] / 1e6:.3f}",
        "95\\% CI": f"[{r['lo'] / 1e6:.3f}, {r['hi'] / 1e6:.3f}]",
        r"$\Delta$ loss (\%)": f"{r['delta_pct']:+.1f}",
        r"$\Delta\Pr(\text{EHR})$": f"{r['ehr_delta']:+.4f}",
        r"$\Delta$ CDI (\%)": f"{r['cdi_pct']:+.1f}",
        "Cliff's $\\delta$": f"{r['cliff']:.3f}",
        "$p_{\\text{Holm}}$": fmt_p(r["p_adj"]),
    } for _, r in loo.iterrows()])
    write_table(
        loo_tab, "tab_loo.tex",
        caption=("Leave-one-out ablation on the hardened portfolio P2. Each row "
                 "sets one control to zero coverage and leaves the remainder "
                 "unchanged, so the effect is the marginal contribution of that "
                 "control given the rest of the programme."),
        label="tab:loo", star=True, escape=False, fit=True, tabcolsep=3.5,
        column_format="lrrrrrrrrr",
        note=("CDI is the care-disruption index, the criticality-weighted sum "
              "of outage hours. Positive values mean risk increases when the "
              "control is withdrawn."),
    )
    viz.plot_control_effects(ROOT / "figures" / "fig_loo.pdf", loo)
    log("wrote figures/fig_loo.pdf")

    # ------------------------------------------------------ single control
    none_portfolio = ctrl.PORTFOLIOS["P0_none"]
    none_stats = _pooled(none_portfolio, params, cg)
    none_mean = float(none_stats["loss"].mean())
    rows2: List[dict] = []
    for key, variant in ctrl.only_one(base_portfolio).items():
        pooled = _pooled(variant, params, cg)
        d, lo, hi, p, delta = _paired_stats(variant, none_portfolio, params, cg)
        cost = ctrl.annual_cost(ctrl.portfolio_vector(variant))
        rows2.append({
            "key": key, "label": LABELS[key],
            "coverage": base_portfolio[key],
            "mean_loss": float(pooled["loss"].mean()),
            "delta": d, "lo": lo, "hi": hi, "p": p, "cliff": delta,
            "delta_pct": 100.0 * d / none_mean,
            "lo_pct": 100.0 * lo / none_mean,
            "hi_pct": 100.0 * hi / none_mean,
            "cost": cost,
            "avoided_per_cost": -d * 1.5 / max(cost, 1.0),
        })
        log(f"  ONLY {key:10s} {rows2[-1]['delta_pct']:7.1f}% loss vs P0")

    adj2 = sx.holm_bonferroni([r["p"] for r in rows2])
    for r, a in zip(rows2, adj2):
        r["p_adj"] = a
    only = pd.DataFrame(rows2).sort_values("delta_pct")
    save_csv(only, "ablation_only.csv")

    only_tab = pd.DataFrame([{
        "Sole control": r["label"],
        "Cov.": f"{r['coverage']:.2f}",
        "Cost (M/yr)": f"{r['cost'] / 1e6:.2f}",
        "Mean loss (M)": f"{r['mean_loss'] / 1e6:.2f}",
        r"$\Delta$ vs P0 (\%)": f"{r['delta_pct']:+.1f}",
        "95\\% CI (\\%)": f"[{r['lo_pct']:+.1f}, {r['hi_pct']:+.1f}]",
        "ALE avoided / cost": f"{r['avoided_per_cost']:.1f}",
        "Cliff's $\\delta$": f"{r['cliff']:.3f}",
        "$p_{\\text{Holm}}$": fmt_p(r["p_adj"]),
    } for _, r in only.iterrows()])
    write_table(
        only_tab, "tab_only.tex",
        caption=("Single-control ablation. Each row activates one control alone, "
                 "at its P2 coverage, against the uncontrolled estate P0. The "
                 "ordering differs from Table~\\ref{tab:loo}, which is the "
                 "signature of substitution between overlapping controls."),
        label="tab:only", star=True, escape=False, fit=True, tabcolsep=3.5,
        column_format="lrrrrrrrr",
        note=("``ALE avoided / cost'' divides annualised loss avoided by annual "
              "programme cost, so values above one indicate a control that pays "
              "for itself in expectation."),
    )
    viz.plot_ablation_bars(
        ROOT / "figures" / "fig_only.pdf", only,
        value="delta_pct", err=None, label_col="label",
        xlabel="Change in expected loss vs uncontrolled (\\%)",
    )
    log("wrote figures/fig_only.pdf")

    # ------------------------------------------------------------- macros
    top_loo = loo.iloc[0]
    second_loo = loo.iloc[1]
    top_only = only.iloc[0]
    macros = {
        "LOOTop": top_loo["label"],
        "LOOTopPct": f"{top_loo['delta_pct']:.1f}",
        "LOOSecond": second_loo["label"],
        "LOOSecondPct": f"{second_loo['delta_pct']:.1f}",
        "LOOLeast": loo.iloc[-1]["label"],
        "LOOLeastPct": f"{loo.iloc[-1]['delta_pct']:.1f}",
        "OnlyTop": top_only["label"],
        "OnlyTopPct": f"{abs(top_only['delta_pct']):.1f}",
        "OnlyAwarenessRatio": f"{float(only.set_index('key').loc['awareness', 'avoided_per_cost']):.1f}",
        "OnlySegRatio": f"{float(only.set_index('key').loc['seg', 'avoided_per_cost']):.1f}",
        "AblationConfigs": len(rows) + len(rows2),
        "OnlyBackupPct": f"{abs(float(only.set_index('key').loc['backup', 'delta_pct'])):.1f}",
        "OnlyMfaPct": f"{abs(float(only.set_index('key').loc['mfa', 'delta_pct'])):.1f}",
        "LOOMfaPct": f"{float(loo.set_index('key').loc['mfa', 'delta_pct']):.1f}",
        "LOOSegPct": f"{float(loo.set_index('key').loc['seg', 'delta_pct']):.1f}",
        "LOOSiemPct": f"{float(loo.set_index('key').loc['siem', 'delta_pct']):.1f}",
    }
    write_macros(macros)


if __name__ == "__main__":
    main()
