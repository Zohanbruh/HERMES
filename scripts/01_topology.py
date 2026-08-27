#!/usr/bin/env python3
"""Emit the reference-architecture figure and the descriptive tables.

Outputs
-------
figures/fig_architecture.pdf
paper/tables/tab_zones.tex
paper/tables/tab_edges.tex
paper/tables/tab_params.tex
results/topology_zones.csv, results/topology_edges.csv
"""

from __future__ import annotations

import pandas as pd

from _common import ROOT, log, save_csv, write_macros, write_table
from hermes import viz
from hermes.controls import CONTROLS, ETA
from hermes.simulator import SimulationParams
from hermes.topology import EDGES, ZONES, build_graph, edge_kind_counts, validate_graph

LAYOUT = {
    "internet": (0.0, 1.0),
    "email": (1.6, 2.1),
    "remote_access": (1.6, 1.0),
    "vendor_gateway": (1.6, -0.1),
    "corp_workstations": (3.2, 2.1),
    "clinical_workstations": (3.2, 1.0),
    "identity": (4.8, 1.7),
    "ehr_core": (6.4, 1.0),
    "pacs": (4.8, 0.3),
    "lab_lis": (3.2, -0.1),
    "pharmacy": (6.4, 2.1),
    "iomt": (4.8, -1.0),
    "billing": (3.2, -1.2),
    "backup": (6.4, -0.2),
}

SHORT = {
    "internet": "Internet",
    "email": "E-mail",
    "remote_access": "Remote\naccess",
    "vendor_gateway": "Vendor\ngateway",
    "corp_workstations": "Corp.\nendpoints",
    "clinical_workstations": "Clinical\nendpoints",
    "identity": "Identity\n(AD/IAM)",
    "ehr_core": "EHR core",
    "pacs": "PACS",
    "lab_lis": "Lab / LIS",
    "pharmacy": "Pharmacy",
    "iomt": "IoMT",
    "billing": "Billing",
    "backup": "Backup / DR",
}


def main() -> None:
    hg = build_graph()
    validate_graph(hg)
    log(f"topology: {hg.n_nodes} zones, {hg.graph.number_of_edges()} edges")
    log(f"edge kinds: {edge_kind_counts(hg)}")

    # ---- figure -----------------------------------------------------------
    edges = [(e.src, e.dst, e.kind) for e in EDGES]
    tiers = {z: str(a["tier"]) for z, a in ZONES.items()}
    viz.plot_architecture(ROOT / "figures" / "fig_architecture.pdf",
                          edges, LAYOUT, SHORT, tiers)
    log("wrote figures/fig_architecture.pdf")

    # ---- zone table -------------------------------------------------------
    zone_rows = []
    for zone, a in ZONES.items():
        if zone == "internet":
            continue
        zone_rows.append({
            "Zone": str(a["label"]),
            "Tier": str(a["tier"]).title(),
            "Crit.": f"{float(a['criticality']):.2f}",
            "PHI records": f"{int(a['phi_records']):,}",
            "Outage cost/h": f"{float(a['downtime_cost']):,.0f}",
            "Restore (h)": f"{float(a['restore_hours']):.0f}",
            "MTD (h)": f"{float(a['mtd_hours']):.0f}",
            r"$d_i$ (/h)": f"{float(a['detect_base']):.4f}",
        })
    zones_df = pd.DataFrame(zone_rows)
    save_csv(zones_df, "topology_zones.csv")
    write_table(
        zones_df, "tab_zones.tex",
        caption=("Reference hospital asset inventory. Criticality is the "
                 "patient-safety weight used by the care-disruption index; "
                 "MTD is the maximum tolerable downtime; $d_i$ is the per-hour "
                 "detection hazard with no dedicated monitoring."),
        label="tab:zones", star=True, escape=False, fit=True, tabcolsep=4,
        column_format="llrrrrrr",
        note=("The estate holds 2.29 million protected-health-information "
              "records in total. All values are model parameters of a "
              "synthetic reference architecture, not measurements of any real "
              "provider."),
    )

    # ---- edge table -------------------------------------------------------
    edge_rows = []
    for e in EDGES:
        edge_rows.append({
            "From": SHORT[e.src].replace("\n", " "),
            "To": SHORT[e.dst].replace("\n", " "),
            "ATT\\&CK": e.technique,
            "Kind": e.kind.replace("_", "-"),
            r"$p^0_{ij}$": f"{e.p_base:.2f}",
            r"$\tau_{ij}$ (h)": f"{e.tau:.0f}",
        })
    edges_df = pd.DataFrame(edge_rows)
    save_csv(edges_df, "topology_edges.csv")
    write_table(
        edges_df, "tab_edges.tex",
        caption=("Adversary steps in the reference attack graph. $p^0_{ij}$ is "
                 "the probability that a competent adversary can traverse the "
                 "step when no mitigating control is present; $\\tau_{ij}$ is "
                 "the mean time to traverse it."),
        label="tab:edges", star=True, escape=False, fit=True, tabcolsep=4,
        column_format="llllrr",
        note="Techniques are MITRE ATT\\&CK Enterprise identifiers.",
    )

    # ---- efficacy matrix --------------------------------------------------
    eta_rows = []
    for c in CONTROLS:
        row = {"Control": c.label}
        for kind, val in ETA[c.key].items():
            row[kind.replace("_", "-")] = f"{val:.2f}"
        row["Cost (M/yr)"] = f"{c.annual_cost_full / 1e6:.2f}"
        eta_rows.append(row)
    eta_df = pd.DataFrame(eta_rows)
    save_csv(eta_df, "control_efficacy.csv")
    write_table(
        eta_df, "tab_eta.tex",
        caption=("Prevention efficacy $\\eta_{c,k}$ of each control against "
                 "each class of adversary step, at full coverage, with the "
                 "annual programme cost at full coverage."),
        label="tab:eta", star=True, escape=False, fit=True, tabcolsep=4,
        column_format="l" + "r" * (eta_df.shape[1] - 1),
        note=("No entry exceeds $0.80$: the model never allows a single control "
              "to eliminate a step outright, which keeps residual risk "
              "non-zero for every portfolio."),
    )

    # ---- parameter table --------------------------------------------------
    p = SimulationParams()
    param_rows = [
        ("Campaign horizon", "$T$", f"{p.horizon_hours} h",
         "30 days; longer horizons change results by $<1\\%$"),
        ("Replications per seed", "$N$", f"{p.n_runs:,}",
         "chosen from the convergence study, Fig.~\\ref{fig:convergence}"),
        ("Containment latency (median)", "$m_L$", f"{p.containment_median_h:.0f} h",
         "detection to full network containment"),
        ("Containment latency (shape)", "$\\sigma_L$", f"{p.containment_sigma:.2f}",
         "log-normal"),
        ("Dwell before impact", "$\\delta$", f"{p.impact_delay_h:.0f} h",
         "hours a zone must be held before it suffers an outage"),
        ("Restoration noise", "$\\sigma_R$", f"{p.restore_sigma:.2f}", "log-normal"),
        ("Backup-loss penalty", "$\\kappa$", f"{p.backup_loss_penalty:.2f}",
         "restoration multiplier when recovery infrastructure is destroyed"),
        ("Breach cost coefficient", "$a$", f"{p.breach_cost_coeff:.0f}",
         "cost $= a\\,R^{b}$ in records $R$"),
        ("Breach cost exponent", "$b$", f"{p.breach_cost_exponent:.2f}",
         "sub-linear, reflecting volume effects in reported breach costs"),
        ("Ransom (median)", "$m_\\Omega$", f"{p.ransom_median / 1e6:.2f} M",
         "log-normal, $\\sigma = " + f"{p.ransom_sigma:.2f}$"),
        ("P(pay $\\mid$ backups lost)", "$\\pi^-$", f"{p.pay_prob_no_backup:.2f}", ""),
        ("P(pay $\\mid$ backups intact)", "$\\pi^+$", f"{p.pay_prob_with_backup:.2f}", ""),
        ("P(regulatory action)", "$\\pi_F$", f"{p.fine_prob:.2f}",
         "capped at " + f"{p.fine_cap / 1e6:.0f} M"),
        ("Exfiltrated share (mean)", "$\\mu_X$", f"{p.exfil_fraction_mean:.2f}",
         "of the records held by a compromised zone"),
        ("Campaign frequency", "$\\lambda$", f"{p.annual_attempt_rate:.1f}/yr",
         "serious targeted campaigns against one large provider"),
    ]
    params_df = pd.DataFrame(param_rows,
                             columns=["Parameter", "Symbol", "Value", "Note"])
    save_csv(params_df, "simulation_parameters.csv")
    write_table(
        params_df, "tab_params.tex",
        caption="Simulation parameters and their calibrated values.",
        label="tab:params", star=True, escape=False, tabcolsep=4,
        column_format="llll",
        note=("Calibration sources and the reasoning behind each value are "
              "given in Appendix~\\ref{app:calibration}."),
    )

    total_phi = sum(int(a["phi_records"]) for a in ZONES.values())
    write_macros({
        "NumZones": hg.n_nodes - 1,
        "NumEdges": hg.graph.number_of_edges(),
        "TotalPHI": f"{total_phi / 1e6:.2f}",
        "HorizonHours": p.horizon_hours,
        "MainRuns": f"{p.n_runs:,}",
        "CampaignRate": f"{p.annual_attempt_rate:.1f}",
    })


if __name__ == "__main__":
    main()
