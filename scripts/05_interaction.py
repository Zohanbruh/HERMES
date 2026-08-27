#!/usr/bin/env python3
"""Pairwise control interaction surfaces.

The Sobol analysis reports that interaction accounts for a large share of the
variance but does not say which pairs interact.  This script sweeps two control
pairs on a grid and reports both the response surface and a formal test of
super- or sub-additivity at the grid centre.

Outputs
-------
figures/fig_interaction_seg_patch.pdf, fig_interaction_mfa_aware.pdf
paper/tables/tab_interaction.tex
results/interaction_*.csv
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from _common import N_RUNS_SWEEP, ROOT, log, save_csv, write_macros, write_table
from hermes import controls as ctrl
from hermes import viz
from hermes.simulator import SimulationParams, compile_graph, simulate

GRID = np.linspace(0.0, 1.0, 11)
SWEEP_SEED = 4242
PAIRS: List[Tuple[str, str, str]] = [
    ("seg", "patch", "seg_patch"),
    ("mfa", "awareness", "mfa_aware"),
]
LABELS = {c.key: c.label for c in ctrl.CONTROLS}


def main() -> None:
    cg = compile_graph()
    params = SimulationParams(n_runs=N_RUNS_SWEEP)
    base = ctrl.PORTFOLIOS["P1_baseline"]

    def loss(portfolio) -> float:
        return float(simulate(portfolio, params, seed=SWEEP_SEED, cg=cg).loss.mean())

    rows: List[dict] = []
    for a_key, b_key, tag in PAIRS:
        surface = np.zeros((GRID.size, GRID.size))
        records = []
        for i, b_val in enumerate(GRID):
            for j, a_val in enumerate(GRID):
                p = dict(base)
                p[a_key], p[b_key] = float(a_val), float(b_val)
                surface[i, j] = loss(p) / 1e6
                records.append({a_key: a_val, b_key: b_val,
                                "mean_loss_M": surface[i, j]})
        save_csv(pd.DataFrame(records), f"interaction_{tag}.csv")
        viz.plot_interaction(
            ROOT / "figures" / f"fig_interaction_{tag}.pdf",
            surface, GRID, GRID,
            xlabel=f"{LABELS[a_key]} coverage",
            ylabel=f"{LABELS[b_key]} coverage",
            cbar_label="Expected loss (M)",
        )
        log(f"wrote figures/fig_interaction_{tag}.pdf")

        # Formal additivity check at the centre of the grid.
        p00 = dict(base); p00[a_key] = 0.0; p00[b_key] = 0.0
        p10 = dict(base); p10[a_key] = 0.8; p10[b_key] = 0.0
        p01 = dict(base); p01[a_key] = 0.0; p01[b_key] = 0.8
        p11 = dict(base); p11[a_key] = 0.8; p11[b_key] = 0.8
        l00, l10, l01, l11 = loss(p00), loss(p10), loss(p01), loss(p11)
        additive = l00 - (l00 - l10) - (l00 - l01)
        interaction = l11 - additive
        rows.append({
            "Pair": f"{LABELS[a_key]} $\\times$ {LABELS[b_key]}",
            "Neither (M)": f"{l00 / 1e6:.3f}",
            "A only (M)": f"{l10 / 1e6:.3f}",
            "B only (M)": f"{l01 / 1e6:.3f}",
            "Both (M)": f"{l11 / 1e6:.3f}",
            "Additive pred. (M)": f"{additive / 1e6:.3f}",
            "Interaction (M)": f"{interaction / 1e6:+.3f}",
            "Regime": "sub-additive" if interaction > 0 else "super-additive",
        })
        log(f"  {tag}: interaction {interaction/1e6:+.3f} M")

    tab = pd.DataFrame(rows)
    save_csv(tab, "interaction_summary.csv")
    write_table(
        tab, "tab_interaction.tex",
        caption=("Additivity check for two control pairs at 0.80 coverage "
                 "against the typical posture P1. The additive prediction "
                 "assumes the two individual reductions simply sum; the "
                 "residual is the interaction."),
        label="tab:interaction", star=True, escape=False, fit=True, tabcolsep=3.5,
        column_format="lrrrrrrl",
        note=("A positive residual means the pair delivers *less* than the sum "
              "of its parts, because the two controls block overlapping paths. "
              "Additive control scoring, which several maturity frameworks use "
              "implicitly, therefore overstates the benefit of stacking "
              "similar controls."),
    )
    write_macros({
        "InterSegPatch": rows[0]["Interaction (M)"].replace("+", ""),
        "InterMfaAware": rows[1]["Interaction (M)"].replace("+", ""),
        "InterRegime": rows[0]["Regime"],
        "InterGridPoints": int(2 * GRID.size * GRID.size),
    })


if __name__ == "__main__":
    main()
