#!/usr/bin/env python3
"""Main experiment: risk under each named control portfolio.

Outputs
-------
figures/fig_convergence.pdf, fig_heatmap.pdf, fig_lec.pdf, fig_dwell.pdf
paper/tables/tab_main.tex, tab_zone_rates.tex
results/main_portfolios.csv, main_zone_rates.csv, main_runs.npz
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from _common import (N_RUNS_MAIN, RESULTS, ROOT, SEEDS, fmt_p, log, save_csv,
                     save_json, write_macros, write_table)
from hermes import controls as ctrl
from hermes import economics as eco
from hermes import statsx as sx
from hermes import viz
from hermes.simulator import (SimulationParams, compile_graph, simulate,
                              zone_compromise_rates)
from hermes.topology import ZONES

SHORT_NAME = {
    "P0_none": "P0 uncontrolled",
    "P1_baseline": "P1 typical",
    "P2_hardened": "P2 hardened",
    "P3_zero_trust": "P3 zero trust",
    "P4_upper_bound": "P4 upper bound",
}


def main() -> None:
    cg = compile_graph()
    params = SimulationParams(n_runs=N_RUNS_MAIN)

    per_seed: Dict[str, List[dict]] = {}
    pooled_loss: Dict[str, np.ndarray] = {}
    pooled_cdi: Dict[str, np.ndarray] = {}
    pooled_dwell: Dict[str, np.ndarray] = {}
    zone_rates: Dict[str, Dict[str, float]] = {}
    ref_loss_by_seed: Dict[int, np.ndarray] = {}

    for name, portfolio in ctrl.PORTFOLIOS.items():
        rows, losses, cdis, dwells = [], [], [], []
        rate_acc: Dict[str, List[float]] = {}
        for seed in SEEDS:
            res = simulate(portfolio, params, seed=seed, cg=cg)
            rows.append(res.summary())
            losses.append(res.loss)
            cdis.append(res.care_disruption)
            dwells.append(res.dwell_hours[res.foothold])
            for z, v in zone_compromise_rates(res).items():
                rate_acc.setdefault(z, []).append(v)
            if name == "P1_baseline":
                ref_loss_by_seed[seed] = res.loss
        per_seed[name] = rows
        pooled_loss[name] = np.concatenate(losses)
        pooled_cdi[name] = np.concatenate(cdis)
        pooled_dwell[name] = np.concatenate(dwells)
        zone_rates[name] = {z: float(np.mean(v)) for z, v in rate_acc.items()}
        log(f"{name:16s} mean loss {pooled_loss[name].mean()/1e6:8.3f} M  "
            f"({len(SEEDS)} seeds x {params.n_runs:,})")

    # ---------------------------------------------------------------- table
    table_rows = []
    macros: Dict[str, str] = {}
    ale_by_name: Dict[str, float] = {}
    for name, portfolio in ctrl.PORTFOLIOS.items():
        seed_means = np.array([r["mean_loss"] for r in per_seed[name]])
        loss = pooled_loss[name]
        point, lo, hi = sx.bootstrap_ci(loss, n_boot=3000)
        cost = ctrl.annual_cost(ctrl.portfolio_vector(portfolio))
        ale = eco.annualised_loss(loss, params.annual_attempt_rate)
        ale_by_name[name] = ale
        s = {k: float(np.mean([r[k] for r in per_seed[name]]))
             for k in per_seed[name][0]}
        table_rows.append({
            "Portfolio": SHORT_NAME[name],
            r"Cost (M/yr)": f"{cost / 1e6:.2f}",
            r"$\Pr(\text{foothold})$": f"{s['p_foothold']:.3f}",
            "Blast radius": f"{s['mean_blast_radius']:.2f}",
            r"$\Pr(\text{EHR})$": f"{s['p_ehr_compromise']:.3f}",
            r"$\Pr(\text{clin. outage})$": f"{s['p_clinical_outage']:.3f}",
            "Dwell med. (h)": f"{s['median_dwell_h']:.1f}",
            "Records (k)": f"{s['mean_records'] / 1e3:.1f}",
            "Mean loss (M)": f"{point / 1e6:.2f}",
            "95\\% CI": f"[{lo / 1e6:.2f}, {hi / 1e6:.2f}]",
            "P95 loss (M)": f"{s['p95_loss'] / 1e6:.2f}",
            "ALE (M/yr)": f"{ale / 1e6:.2f}",
        })
        # LaTeX control sequences may not contain digits, so P0..P4 are spelled out.
        tag = {"P0": "Pzero", "P1": "Pone", "P2": "Ptwo",
               "P3": "Pthree", "P4": "Pfour"}[name.split("_")[0]]
        macros[f"{tag}Loss"] = f"{point / 1e6:.2f}"
        macros[f"{tag}ALE"] = f"{ale / 1e6:.2f}"
        macros[f"{tag}Cost"] = f"{cost / 1e6:.2f}"
        macros[f"{tag}Blast"] = f"{s['mean_blast_radius']:.2f}"
        macros[f"{tag}PEHR"] = f"{s['p_ehr_compromise']:.3f}"
        macros[f"{tag}SeedSD"] = f"{seed_means.std(ddof=1) / 1e6:.3f}"

    main_df = pd.DataFrame(table_rows)
    save_csv(main_df, "main_portfolios.csv")
    write_table(
        main_df, "tab_main.tex",
        caption=("Risk under each control portfolio. Every row pools "
                 f"{len(SEEDS)} independent seeds of {params.n_runs:,} "
                 "campaign replications. Confidence intervals are percentile "
                 "bootstrap intervals on the mean campaign loss; ALE is the "
                 "annualised loss expectancy at "
                 f"$\\lambda={params.annual_attempt_rate}$ campaigns per year."),
        label="tab:main", star=True, escape=False, fit=True, tabcolsep=3.5,
        column_format="lrrrrrrrrrrr",
        note=("Monetary values are in model currency units, calibrated to "
              "published breach and outage costs for a large multi-site "
              "provider. All figures are simulation output, not measurements."),
    )

    # ------------------------------------------------- paired comparisons
    comp_rows = []
    for name in ctrl.PORTFOLIOS:
        if name == "P1_baseline":
            continue
        diffs, ps, deltas = [], [], []
        for seed in SEEDS:
            res = simulate(ctrl.PORTFOLIOS[name], params, seed=seed, cg=cg)
            d, lo, hi = sx.paired_bootstrap_diff(res.loss, ref_loss_by_seed[seed],
                                                 n_boot=2000, seed=seed)
            diffs.append((d, lo, hi))
            ps.append(sx.mannwhitney_p(res.loss, ref_loss_by_seed[seed]))
            deltas.append(sx.cliffs_delta(res.loss, ref_loss_by_seed[seed]))
        d = float(np.mean([x[0] for x in diffs]))
        lo = float(np.mean([x[1] for x in diffs]))
        hi = float(np.mean([x[2] for x in diffs]))
        delta = float(np.mean(deltas))
        comp_rows.append({
            "name": name, "diff": d, "lo": lo, "hi": hi,
            "p": float(np.median(ps)), "delta": delta,
        })
    adj = sx.holm_bonferroni([r["p"] for r in comp_rows])
    comp_df = pd.DataFrame([{
        "Comparison": f"{SHORT_NAME[r['name']]} vs P1 typical",
        r"$\Delta$ mean loss (M)": f"{r['diff'] / 1e6:.2f}",
        "95\\% CI": f"[{r['lo'] / 1e6:.2f}, {r['hi'] / 1e6:.2f}]",
        "Change": ("$-$" if r["diff"] < 0 else "$+$") +
                  f"{abs(100 * r['diff'] / pooled_loss['P1_baseline'].mean()):.1f}\\%",
        "Cliff's $\\delta$": f"{r['delta']:.3f} ({sx.interpret_delta(r['delta'])})",
        "$p_{\\text{Holm}}$": fmt_p(a),
    } for r, a in zip(comp_rows, adj)])
    save_csv(comp_df, "main_comparisons.csv")
    write_table(
        comp_df, "tab_compare.tex",
        caption=("Paired comparison of each portfolio against the typical "
                 "posture P1 under common random numbers. Pairing removes the "
                 "shared campaign randomness, which shrinks the interval by "
                 "roughly an order of magnitude relative to an unpaired test."),
        label="tab:compare", escape=False, column_format="lrrrrr",
        star=True, fit=True, tabcolsep=4,
        note=("$p$-values are two-sided Mann--Whitney, adjusted across the "
              "family by the Holm step-down procedure. Effect sizes are "
              "reported because with $10^{5}$ replications any non-zero "
              "difference is statistically significant."),
    )

    # ---------------------------------------------------------- convergence
    traces = {SHORT_NAME[n]: sx.convergence_trace(pooled_loss[n])
              for n in ("P0_none", "P1_baseline", "P2_hardened", "P3_zero_trust")}
    viz.plot_convergence(ROOT / "figures" / "fig_convergence.pdf", traces)
    log("wrote figures/fig_convergence.pdf")
    need = sx.required_replications(pooled_loss["P1_baseline"], 0.02)
    macros["ReqRuns"] = f"{need:,}"

    # -------------------------------------------------------------- heatmap
    order = [z for z in ZONES if z != "internet"]
    label = {
        "email": "E-mail", "remote_access": "Remote access",
        "vendor_gateway": "Vendor gateway", "corp_workstations": "Corp. endpoints",
        "identity": "Identity", "clinical_workstations": "Clinical endpoints",
        "ehr_core": "EHR core", "pacs": "PACS", "lab_lis": "Lab / LIS",
        "pharmacy": "Pharmacy", "iomt": "IoMT", "billing": "Billing",
        "backup": "Backup / DR",
    }
    hm = pd.DataFrame(
        [[zone_rates[n][z] for z in order] for n in ctrl.PORTFOLIOS],
        index=[SHORT_NAME[n] for n in ctrl.PORTFOLIOS],
        columns=[label[z] for z in order],
    )
    viz.plot_zone_heatmap(ROOT / "figures" / "fig_heatmap.pdf", hm)
    log("wrote figures/fig_heatmap.pdf")
    zr = hm.round(4).reset_index().rename(columns={"index": "Portfolio"})
    save_csv(zr, "main_zone_rates.csv")
    # Transposed for the paper: 13 zones as rows reads far better in two columns
    # than 13 zones as columns.
    zt = hm.T.round(4).reset_index()
    zt.columns = ["Zone"] + [c for c in hm.index]
    write_table(
        zt, "tab_zone_rates.tex",
        caption=("Marginal probability that each zone is compromised, by "
                 "control portfolio. Numeric form of Fig.~\\ref{fig:heatmap}."),
        label="tab:zonerates", escape=True, fit=True, tabcolsep=4,
        column_format="l" + "r" * len(hm.index),
        note=("Perimeter zones remain the most exposed under every posture; the "
              "identity service and the clinical zones downstream of it fall "
              "fastest as coverage rises."),
    )

    # ------------------------------------------------------------------ LEC
    curves, var_rows = {}, []
    for n in ("P0_none", "P1_baseline", "P2_hardened", "P3_zero_trust"):
        agg = eco.aggregate_annual_loss(pooled_loss[n], params.annual_attempt_rate)
        curves[SHORT_NAME[n]] = eco.loss_exceedance_curve(agg)
        var_rows.append({
            "Portfolio": SHORT_NAME[n],
            "Mean (M)": f"{agg.mean() / 1e6:.2f}",
            "VaR$_{95}$ (M)": f"{eco.value_at_risk(agg, 0.95) / 1e6:.2f}",
            "VaR$_{99}$ (M)": f"{eco.value_at_risk(agg, 0.99) / 1e6:.2f}",
            "TVaR$_{95}$ (M)": f"{eco.tail_value_at_risk(agg, 0.95) / 1e6:.2f}",
            "P(loss $>$ 10M)": f"{(agg > 1e7).mean():.3f}",
        })
    viz.plot_lec(ROOT / "figures" / "fig_lec.pdf", curves)
    log("wrote figures/fig_lec.pdf")
    var_df = pd.DataFrame(var_rows)
    save_csv(var_df, "main_var.csv")
    write_table(
        var_df, "tab_var.tex",
        caption=("Aggregate annual loss distribution, obtained by compounding "
                 "a Poisson campaign count with the simulated campaign loss."),
        label="tab:var", escape=False, column_format="lrrrrr",
        star=True, fit=True, tabcolsep=4,
    )

    # ---------------------------------------------------------------- dwell
    viz.plot_dwell_distributions(
        ROOT / "figures" / "fig_dwell.pdf",
        {SHORT_NAME[n]: pooled_dwell[n]
         for n in ("P0_none", "P1_baseline", "P2_hardened", "P3_zero_trust")},
    )
    log("wrote figures/fig_dwell.pdf")

    # -------------------------------------------------------------- persist
    np.savez_compressed(
        RESULTS / "main_runs.npz",
        **{f"loss_{n}": pooled_loss[n] for n in pooled_loss},
        **{f"cdi_{n}": pooled_cdi[n] for n in pooled_cdi},
    )
    log("wrote results/main_runs.npz")
    save_json({"ale": ale_by_name,
               "seeds": list(SEEDS),
               "n_runs_per_seed": params.n_runs}, "main_ale.json")

    # extra prose numbers
    p1_seed = np.array([r["mean_loss"] for r in per_seed["P1_baseline"]])
    macros["PoneSeedSDPct"] = f"{100 * p1_seed.std(ddof=1) / p1_seed.mean():.2f}\\%"
    agg_p1 = eco.aggregate_annual_loss(pooled_loss["P1_baseline"],
                                       params.annual_attempt_rate)
    macros["PoneVaR"] = f"{eco.value_at_risk(agg_p1, 0.95) / agg_p1.mean():.1f}"
    macros["PoneTailProb"] = f"{(agg_p1 > 1e7).mean():.3f}"
    macros["PzeroDwell"] = f"{np.median(pooled_dwell['P0_none']):.0f}"
    macros["PthreeDwell"] = f"{np.median(pooled_dwell['P3_zero_trust']):.0f}"
    macros["CDIBase"] = f"{pooled_cdi['P1_baseline'].mean():.0f}"
    macros["CDIHard"] = f"{pooled_cdi['P2_hardened'].mean():.0f}"
    macros["CDIDrop"] = (
        f"{100 * (1 - pooled_cdi['P2_hardened'].mean() / pooled_cdi['P1_baseline'].mean()):.1f}")
    macros["LossDropHard"] = (
        f"{100 * (1 - pooled_loss['P2_hardened'].mean() / pooled_loss['P1_baseline'].mean()):.1f}")
    macros["LossDropZT"] = (
        f"{100 * (1 - pooled_loss['P3_zero_trust'].mean() / pooled_loss['P1_baseline'].mean()):.1f}")
    macros["ResidualZT"] = (
        f"{100 * pooled_loss['P3_zero_trust'].mean() / pooled_loss['P4_upper_bound'].mean():.0f}")
    macros["TotalReps"] = f"{len(ctrl.PORTFOLIOS) * len(SEEDS) * params.n_runs:,}"
    write_macros(macros)


if __name__ == "__main__":
    main()
