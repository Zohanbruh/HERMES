#!/usr/bin/env python3
"""Face-validity check against publicly reported healthcare incidents.

The model is synthetic, so it cannot be validated against a held-out data set.
What it can be asked to do is reproduce, without further tuning, the order of
magnitude of five quantities that were reported publicly for 2024-era incidents
and for the peer-reviewed incident literature.  Where the model falls outside a
reported range, that is stated rather than hidden.

Outputs
-------
figures/fig_calibration.pdf
paper/tables/tab_calibration.tex
results/calibration.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import N_RUNS_MAIN, ROOT, SEEDS, log, save_csv, write_macros, write_table
from hermes import controls as ctrl
from hermes import viz
from hermes.simulator import SimulationParams, compile_graph, simulate
from hermes.topology import CLINICAL_ZONES, ZONES

#: (label, low, high, source shown in the table)
TARGETS = {
    "care_disruption": (
        "Share of landed attacks that disrupt care delivery",
        0.70, 0.80,
        "Neprash \\emph{et al.}, \\emph{JAMA Health Forum} 2022"),
    "clinical_outage": (
        "Clinical-system outage in a major incident (h)",
        336.0, 1008.0,
        "Ascension 2024; Synnovis 2024; Change Healthcare 2024"),
    "records": (
        "PHI records exposed at one large provider (M)",
        0.5, 6.0,
        "HHS OCR breach portal, 2024 provider incidents"),
    "dwell": (
        "Dwell time from intrusion to detection (h)",
        24.0, 240.0,
        "Public incident timelines and vendor dwell-time reporting"),
    "cost": (
        "Direct cost of a major provider incident (M)",
        5.0, 120.0,
        "Reported recovery costs, large US and UK providers"),
}


def main() -> None:
    cg = compile_graph()
    params = SimulationParams(n_runs=N_RUNS_MAIN)
    clin_idx = [i for i, n in enumerate(cg.nodes) if n in CLINICAL_ZONES]
    # A campaign counts as "landed" only when the adversary achieved impact
    # inside the estate.  Conditioning instead on any impact at all would put
    # phished mailboxes in the denominator, which is not what the epidemiological
    # literature counts as a ransomware attack on a hospital.
    internal_idx = [i for i, n in enumerate(cg.nodes)
                    if str(ZONES[n]["tier"]) in ("endpoint", "core", "clinical",
                                                 "business")]

    care, clin_out, recs, dwell, cost = [], [], [], [], []
    for seed in SEEDS:
        r = simulate(ctrl.PORTFOLIOS["P1_baseline"], params, seed=seed, cg=cg)
        landed = (r.outage_hours[:, internal_idx] > 0).any(axis=1)
        clinical = (r.outage_hours[:, clin_idx] > 0).any(axis=1)
        care.append(float(clinical[landed].mean()) if landed.any() else np.nan)
        oh = r.outage_hours[:, clin_idx].max(axis=1)
        clin_out.append(oh[clinical])
        recs.append(r.records_exposed[landed] / 1e6)
        d = r.dwell_hours[r.foothold & np.isfinite(r.detect_time)]
        dwell.append(d)
        cost.append(r.loss[landed] / 1e6)

    care_rate = float(np.mean(care))
    care_sd = float(np.std(care, ddof=1))
    care_lo = care_rate - 1.96 * care_sd / np.sqrt(len(care))
    care_hi = care_rate + 1.96 * care_sd / np.sqrt(len(care))
    clin_out = np.concatenate(clin_out)
    recs = np.concatenate(recs)
    dwell = np.concatenate(dwell)
    cost = np.concatenate(cost)

    sim = {
        "care_disruption": np.array([care_rate]),
        "clinical_outage": clin_out,
        "records": recs,
        "dwell": dwell,
        "cost": cost,
    }

    rows = []
    for key, (label, lo, hi, source) in TARGETS.items():
        arr = sim[key]
        if key == "care_disruption":
            med, q05, q95 = care_rate, care_lo, care_hi
            shown = f"{care_rate:.2f} [{care_lo:.2f}, {care_hi:.2f}]"
        else:
            med = float(np.median(arr))
            q05, q95 = (float(np.quantile(arr, 0.50)),
                        float(np.quantile(arr, 0.95)))
            shown = f"{med:.1f} [{q05:.1f}, {q95:.1f}]"
        overlap = not (q95 < lo or q05 > hi)
        rows.append({
            "Observable": label,
            "Reported range": f"{lo:g}--{hi:g}",
            "Source": source,
            "HERMES (median [P50, P95])": shown,
            "Consistent": r"\checkmark" if overlap else r"$\times$",
        })
        log(f"  {key:18s} sim={shown:>24s}  target=[{lo}, {hi}]  ok={overlap}")

    cal = pd.DataFrame(rows)
    save_csv(cal, "calibration.csv")
    write_table(
        cal, "tab_calibration.tex",
        caption=("Face-validity check of the typical posture P1 against publicly "
                 "reported quantities. No parameter was tuned to these targets "
                 "after the model was fixed; the one shortfall is discussed in "
                 "Section~\\ref{sec:threats} rather than corrected."),
        label="tab:calibration", star=True, escape=False, fit=True, tabcolsep=4,
        column_format="lllll",
        note=("Consistency means the reported range overlaps the simulated "
              "median-to-95th-percentile band. This is a face-validity check, "
              "not a statistical goodness-of-fit test: the reported figures come "
              "from different providers, jurisdictions and reporting regimes and "
              "cannot form a like-for-like sample."),
    )

    viz.plot_calibration(ROOT / "figures" / "fig_calibration.pdf", [
        {"samples": clin_out, "reported": (336.0, 1008.0),
         "xlabel": "Clinical outage (h)", "title": "Outage duration"},
        {"samples": recs[recs > 0], "reported": (0.5, 6.0),
         "xlabel": "Records exposed (M)", "title": "Breach size"},
        {"samples": dwell, "reported": (24.0, 240.0),
         "xlabel": "Dwell to detection (h)", "title": "Dwell time", "logx": True},
        {"samples": cost[cost > 0], "reported": (5.0, 120.0),
         "xlabel": "Incident cost (M)", "title": "Direct cost", "logx": True},
    ])
    log("wrote figures/fig_calibration.pdf")

    n_ok = int((cal["Consistent"] == r"\checkmark").sum())
    write_macros({
        "CalCareRate": f"{care_rate:.2f}",
        "CalCareLo": f"{care_lo:.2f}",
        "CalCareHi": f"{care_hi:.2f}",
        "CalOutageMed": f"{np.median(clin_out):.0f}",
        "CalDwellMed": f"{np.median(dwell):.0f}",
        "CalRecordsMed": f"{np.median(recs):.2f}",
        "CalCostMed": f"{np.median(cost):.1f}",
        "CalPassed": n_ok,
        "CalTotal": len(cal),
    })


if __name__ == "__main__":
    main()
