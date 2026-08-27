#!/usr/bin/env python3
"""Budget-constrained control investment.

Produces the achievable risk/spend frontier and the return-on-security-
investment table that a hospital board would use to compare postures.

Outputs
-------
figures/fig_frontier.pdf
paper/tables/tab_invest.tex, tab_frontier.tex
results/frontier.csv, investment.csv
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from _common import N_RUNS_SWEEP, ROOT, SEEDS, log, save_csv, write_macros, write_table
from hermes import controls as ctrl
from hermes import economics as eco
from hermes.simulator import SimulationParams, compile_graph, simulate

SWEEP_SEED = 777
SHORT = {"P0_none": "P0", "P1_baseline": "P1", "P2_hardened": "P2",
         "P3_zero_trust": "P3", "P4_upper_bound": "P4"}
NAMES = {"P0_none": "P0 uncontrolled", "P1_baseline": "P1 typical",
         "P2_hardened": "P2 hardened", "P3_zero_trust": "P3 zero trust",
         "P4_upper_bound": "P4 upper bound"}


def main() -> None:
    cg = compile_graph()
    params = SimulationParams(n_runs=SimulationParams().n_runs)
    sweep = SimulationParams(n_runs=N_RUNS_SWEEP)
    rate = params.annual_attempt_rate

    # ---- annualised loss for the named portfolios (multi-seed) -----------
    ales: Dict[str, float] = {}
    for name, portfolio in ctrl.PORTFOLIOS.items():
        means = [float(simulate(portfolio, params, seed=s, cg=cg).loss.mean())
                 for s in SEEDS]
        ales[name] = float(np.mean(means)) * rate
    rows = eco.investment_table(ales, ctrl.PORTFOLIOS, reference="P1_baseline")

    inv = pd.DataFrame([{
        "Portfolio": NAMES[r.name],
        "Cost (M/yr)": f"{r.annual_cost / 1e6:.2f}",
        r"$\Delta$ cost (M/yr)": f"{r.incremental_cost / 1e6:+.2f}",
        "ALE (M/yr)": f"{r.ale / 1e6:.2f}",
        "ALE avoided (M/yr)": f"{r.ale_avoided / 1e6:+.2f}",
        "Net benefit (M/yr)": f"{r.net_benefit / 1e6:+.2f}",
        "ROSI": ("--" if not np.isfinite(r.rosi) else f"{r.rosi:.1f}"),
    } for r in rows])
    save_csv(inv, "investment.csv")
    write_table(
        inv, "tab_invest.tex",
        caption=("Return on security investment relative to the typical "
                 "posture P1. ROSI is undefined where the incremental spend is "
                 "non-positive."),
        label="tab:invest", escape=False, star=True, fit=True, tabcolsep=4,
        column_format="lrrrrrr",
        note=("ROSI is proportional to the assumed campaign frequency "
              f"$\\lambda={rate}$ per year and inherits all of its uncertainty; "
              "the ordering of portfolios, however, does not depend on "
              "$\\lambda$."),
    )

    # ---- greedy frontier -------------------------------------------------
    cache: Dict[tuple, float] = {}

    def evaluate(portfolio) -> float:
        key = tuple(round(portfolio[k], 4) for k in ctrl.CONTROL_KEYS)
        if key not in cache:
            cache[key] = float(
                simulate(portfolio, sweep, seed=SWEEP_SEED, cg=cg).loss.mean()
            )
        return cache[key]

    budgets = list(np.linspace(0.05e6, 1.95e6, 26))
    log(f"greedy frontier over {len(budgets)} budget levels ...")
    records = eco.greedy_frontier(evaluate, budgets, step=0.05)
    log(f"  {len(records)} frontier points, {len(cache)} evaluations")

    fr = pd.DataFrame([{
        "budget": r["budget"], "cost": r["cost"],
        "loss": r["loss"], "ale": r["loss"] * rate,
        **{f"cov_{k}": r["portfolio"][k] for k in ctrl.CONTROL_KEYS},
    } for r in records])
    save_csv(fr, "frontier.csv")

    pts = pd.DataFrame([{
        "short": SHORT[n],
        "cost": ctrl.annual_cost(ctrl.portfolio_vector(p)),
        "ale": ales[n],
    } for n, p in ctrl.PORTFOLIOS.items()])
    from hermes import viz
    viz.plot_frontier(ROOT / "figures" / "fig_frontier.pdf", fr, pts)
    log("wrote figures/fig_frontier.pdf")

    show = fr.iloc[:: max(1, len(fr) // 9)].copy()
    ftab = pd.DataFrame([{
        "Budget (M/yr)": f"{r['cost'] / 1e6:.2f}",
        "ALE (M/yr)": f"{r['ale'] / 1e6:.3f}",
        **{c.label.split("(")[0].strip()[:11]: f"{r['cov_' + c.key]:.2f}"
           for c in ctrl.CONTROLS},
    } for _, r in show.iterrows()])
    write_table(
        ftab, "tab_frontier.tex",
        caption=("Coverage allocation along the greedy risk/spend frontier. "
                 "Each row is the portfolio the greedy allocator holds when the "
                 "stated budget is first exhausted."),
        label="tab:frontier", star=True, escape=True, fit=True, tabcolsep=3.5,
        column_format="rr" + "r" * len(ctrl.CONTROLS),
        note=("Greedy allocation is not guaranteed optimal for a non-additive "
              "objective, so the curve is reported as achievable rather than "
              "optimal."),
    )

    # ---- efficiency of the named postures against the frontier -----------
    gaps = []
    for n, p in ctrl.PORTFOLIOS.items():
        cost = ctrl.annual_cost(ctrl.portfolio_vector(p))
        near = fr[fr["cost"] <= cost + 1e-6]
        if len(near) == 0:
            continue
        best = float(near["ale"].min())
        gaps.append({"name": n, "cost": cost, "ale": ales[n], "best": best,
                     "gap_pct": 100.0 * (ales[n] - best) / max(ales[n], 1e-9)})
    gap_df = pd.DataFrame(gaps)
    save_csv(gap_df, "frontier_gap.csv")

    p2_gap = gap_df.set_index("name").loc["P2_hardened", "gap_pct"] \
        if "P2_hardened" in gap_df["name"].values else float("nan")
    knee = fr.iloc[(fr["ale"].diff() / fr["cost"].diff()).abs().fillna(0).argmax()] \
        if len(fr) > 2 else fr.iloc[0]
    write_macros({
        "ROSIHard": f"{[r for r in rows if r.name == 'P2_hardened'][0].rosi:.1f}",
        "ROSIZT": f"{[r for r in rows if r.name == 'P3_zero_trust'][0].rosi:.1f}",
        "NetBenefitHard": f"{[r for r in rows if r.name=='P2_hardened'][0].net_benefit/1e6:.2f}",
        "FrontierEvals": f"{len(cache):,}",
        "FrontierGapHard": f"{p2_gap:.0f}",
        "FrontierKneeCost": f"{float(knee['cost']) / 1e6:.2f}",
    })


if __name__ == "__main__":
    main()
