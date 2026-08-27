#!/usr/bin/env python3
"""Global sensitivity analysis of the risk model.

Two questions are separated.  *Which inputs drive the variance of expected loss?*
is answered by Sobol indices over the seven control coverages and three model
uncertainties.  *How far can each input move the answer on its own?* is answered
by a one-at-a-time swing analysis around the typical posture P1.

The gap between first-order and total-order indices is the interaction share,
which is the quantity that justifies modelling defence in depth multiplicatively
rather than adding up control effects.

Outputs
-------
figures/fig_sobol.pdf, fig_tornado.pdf
paper/tables/tab_sobol.tex, tab_tornado.tex
results/sobol.csv, tornado.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import ROOT, log, save_csv, write_macros, write_table
from hermes import controls as ctrl
from hermes import sensitivity as sens
from hermes import viz
from hermes.simulator import SimulationParams, compile_graph, simulate

N_BASE = 512           # power of two, as the Sobol sequence requires
N_RUNS_EVAL = 1_024    # replications inside one model evaluation
EVAL_SEED = 909

PARAMS = [
    sens.Parameter("mfa", "MFA coverage", 0.0, 1.0),
    sens.Parameter("seg", "Segmentation coverage", 0.0, 1.0),
    sens.Parameter("patch", "Patch coverage", 0.0, 1.0),
    sens.Parameter("edr", "EDR coverage", 0.0, 1.0),
    sens.Parameter("awareness", "Awareness coverage", 0.0, 1.0),
    sens.Parameter("backup", "Immutable backup coverage", 0.0, 1.0),
    sens.Parameter("siem", "SIEM coverage", 0.0, 1.0),
    sens.Parameter("containment", "Containment latency $m_L$ (h)", 12.0, 96.0),
    sens.Parameter("impact_delay", "Dwell before impact $\\delta$ (h)", 4.0, 36.0),
    sens.Parameter("detect_scale", "Monitoring efficacy scale", 0.5, 2.0),
]


def main() -> None:
    cg = compile_graph()

    def model(vec: np.ndarray) -> float:
        portfolio = {k: float(vec[i]) for i, k in enumerate(ctrl.CONTROL_KEYS)}
        p = SimulationParams(
            n_runs=N_RUNS_EVAL,
            containment_median_h=float(vec[7]),
            impact_delay_h=float(vec[8]),
            detect_scale=float(vec[9]),
        )
        return float(simulate(portfolio, p, seed=EVAL_SEED, cg=cg).loss.mean())

    log(f"Sobol: {N_BASE} base samples, {N_BASE * (len(PARAMS) + 2):,} evaluations")
    result = sens.sobol_analysis(model, PARAMS, n_base=N_BASE, seed=2024,
                                 n_boot=300, progress=True)
    rows = result.as_rows()
    sob = pd.DataFrame(rows)
    save_csv(sob, "sobol.csv")

    tab = pd.DataFrame([{
        "Input": r["label"],
        "$S_1$": f"{r['S1']:.3f}",
        "95\\% CI": f"[{r['S1_lo']:.3f}, {r['S1_hi']:.3f}]",
        "$S_T$": f"{r['ST']:.3f}",
        "95\\% CI ": f"[{r['ST_lo']:.3f}, {r['ST_hi']:.3f}]",
        "$S_T - S_1$": f"{r['ST'] - r['S1']:.3f}",
    } for r in sorted(rows, key=lambda x: -float(x["ST"]))])
    write_table(
        tab, "tab_sobol.tex",
        caption=("Variance-based sensitivity of expected campaign loss. $S_1$ is "
                 "the first-order index, $S_T$ the total-order index; their "
                 "difference is the share of variance a factor explains only "
                 "through interaction with other factors."),
        label="tab:sobol", escape=False, star=True, fit=True, tabcolsep=4,
        column_format="lrrrrr",
        note=(f"Saltelli design with $N={N_BASE}$ base samples, "
              f"{result.n_evaluations:,} model evaluations of "
              f"{N_RUNS_EVAL:,} replications each. Intervals are bootstrap "
              "intervals over the base rows. Jansen estimators can return "
              "slightly negative $S_1$ for inactive factors; such values are "
              "reported unmodified."),
    )
    viz.plot_sobol(ROOT / "figures" / "fig_sobol.pdf", rows)
    log("wrote figures/fig_sobol.pdf")

    # ------------------------------------------------------------ tornado
    p1 = ctrl.PORTFOLIOS["P1_baseline"]
    baseline_vec = np.array(
        [p1[k] for k in ctrl.CONTROL_KEYS] + [34.0, 12.0, 1.0], dtype=float
    )
    tor = sens.oat_tornado(model, PARAMS, baseline_vec)
    tor_df = pd.DataFrame(tor)
    save_csv(tor_df, "tornado.csv")
    base_y = float(tor[0]["baseline_output"])
    tor_tab = pd.DataFrame([{
        "Input": r["label"],
        "Low (M)": f"{r['low_output'] / 1e6:.2f}",
        "High (M)": f"{r['high_output'] / 1e6:.2f}",
        "Swing (M)": f"{r['swing'] / 1e6:.2f}",
        "Swing / baseline": f"{r['swing'] / base_y:.2f}",
    } for r in tor])
    write_table(
        tor_tab, "tab_tornado.tex",
        caption=("One-at-a-time swing of expected campaign loss around the "
                 "typical posture P1, ordered by swing magnitude."),
        label="tab:tornado", escape=False, fit=True, tabcolsep=4,
        column_format="lrrrr",
        note=("A local analysis is shown for interpretability only; the "
              "variance decomposition in Table~\\ref{tab:sobol} is the evidence, "
              "because the model is not additive."),
    )
    tor_df["value"] = tor_df["swing"] / 1e6
    viz.plot_ablation_bars(
        ROOT / "figures" / "fig_tornado.pdf", tor_df,
        value="value", err=None, label_col="label",
        xlabel="Swing in expected loss (M)",
    )
    log("wrote figures/fig_tornado.pdf")

    inter = float(np.sum(np.maximum(result.st - result.s1, 0.0)) /
                  max(np.sum(result.st), 1e-9))
    top = sob.sort_values("ST", ascending=False).iloc[0]
    write_macros({
        "SobolEvals": f"{result.n_evaluations:,}",
        "SobolBase": N_BASE,
        "SobolRunsPerEval": f"{N_RUNS_EVAL:,}",
        "SobolTop": str(top["label"]),
        "SobolTopST": f"{float(top['ST']):.3f}",
        "SobolInteraction": f"{100 * inter:.0f}",
        "SobolSumFirstOrder": f"{float(np.sum(np.maximum(result.s1, 0.0))):.2f}",
    })


if __name__ == "__main__":
    main()
