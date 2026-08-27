#!/usr/bin/env python3
"""Detection benchmark on synthetic hospital SOC telemetry.

Closes the loop on the simulator's detection hazard: the propagation model
assumes monitoring raises the per-hour probability of noticing an intrusion, and
this experiment asks what that probability actually looks like for standard
detectors operating on the signals a hospital SOC would collect.

Three evaluations are reported.

1. **Group-aware cross-validation** on telemetry from the typical posture P1.
   Folds are split by campaign, never by hour, so no detector sees another hour
   of the same intrusion during training.
2. **Feature-group ablation**, which identifies where the discriminative signal
   lives and how much temporal context contributes.
3. **Posture shift**, training on P1 telemetry and testing on the hardened
   posture P2, where intrusions are shorter and quieter.

Outputs
-------
figures/fig_detection_curves.pdf, fig_det_ablation.pdf
paper/tables/tab_detection.tex, tab_det_ablation.tex, tab_shift.tex
results/detection_cv.csv, detection_ablation.csv, detection_shift.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import GroupKFold

from _common import ROOT, log, save_csv, write_macros, write_table
from hermes import controls as ctrl
from hermes import viz
from hermes.detection import (FEATURES, FEATURE_GROUPS, TelemetryConfig,
                              ablate_feature_groups, build_models,
                              domain_shift_eval, generate_telemetry,
                              run_benchmark)
from hermes.simulator import SimulationParams, compile_graph, simulate

N_CAMPAIGNS = 110
HOURS = 360
SEED = 31


def main() -> None:
    cg = compile_graph()
    sim_params = SimulationParams(n_runs=3_000)

    log("simulating campaigns for telemetry ...")
    res_p1 = simulate(ctrl.PORTFOLIOS["P1_baseline"], sim_params, seed=SEED, cg=cg)
    res_p2 = simulate(ctrl.PORTFOLIOS["P2_hardened"], sim_params, seed=SEED + 1, cg=cg)

    cfg = TelemetryConfig(n_campaigns=N_CAMPAIGNS, hours_per_campaign=HOURS,
                          seed=4242)
    data_p1 = generate_telemetry(res_p1, cfg)
    data_p2 = generate_telemetry(res_p2, TelemetryConfig(
        n_campaigns=N_CAMPAIGNS, hours_per_campaign=HOURS, seed=9191))
    base_rate = float(data_p1["label"].mean())
    log(f"P1 telemetry {data_p1.shape}, positive rate {base_rate:.4f}")
    log(f"P2 telemetry {data_p2.shape}, positive rate {data_p2['label'].mean():.4f}")

    # ------------------------------------------------- cross-validated CV
    log("cross-validated benchmark ...")
    cv = run_benchmark(data_p1, n_splits=4, seed=SEED)
    save_csv(cv, "detection_cv.csv")

    agg = cv.groupby("model").agg(
        roc=("roc_auc", "mean"), roc_sd=("roc_auc", "std"),
        pr=("pr_auc", "mean"), pr_sd=("pr_auc", "std"),
        f1=("best_f1", "mean"), f1_sd=("best_f1", "std"),
        rec=("recall_at_fpr", "mean"),
        prec50=("precision_at_recall50", "mean"),
        brier=("brier", "mean"),
        alerts=("alerts_per_day", "mean"),
    ).reset_index().sort_values("pr", ascending=False)

    det_tab = pd.DataFrame([{
        "Detector": r["model"],
        "ROC-AUC": f"{r['roc']:.3f} $\\pm$ {r['roc_sd']:.3f}",
        "PR-AUC": f"{r['pr']:.3f} $\\pm$ {r['pr_sd']:.3f}",
        "Best $F_1$": f"{r['f1']:.3f} $\\pm$ {r['f1_sd']:.3f}",
        "Rec.@1\\%FPR": f"{r['rec']:.3f}",
        "Prec.@50\\%Rec.": f"{r['prec50']:.3f}",
        "Brier": f"{r['brier']:.3f}",
        "Alerts/day": f"{r['alerts']:.1f}",
    } for _, r in agg.iterrows()])
    write_table(
        det_tab, "tab_detection.tex",
        caption=("Intrusion-detection benchmark on synthetic hospital SOC "
                 "telemetry from the typical posture P1, four-fold "
                 "cross-validation grouped by campaign. Mean $\\pm$ standard "
                 "deviation across folds."),
        label="tab:detection", star=True, escape=False, fit=True, tabcolsep=3.5,
        column_format="lrrrrrrr",
        note=(f"Base rate of intrusion-active hours is {base_rate:.3f}, so "
              "PR-AUC rather than accuracy is the meaningful summary. Alert "
              "volume is measured at each detector's own threshold for 50\\,\\% "
              "recall. The generator injects benign confounders and boundary "
              "label noise; near-perfect scores here would indicate a leaky "
              "generator rather than a good detector."),
    )
    log(agg[["model", "roc", "pr", "rec"]].round(3).to_string(index=False))

    # ---------------------------------------------------------- ROC and PR
    x = data_p1[list(FEATURES)].to_numpy()
    y = data_p1["label"].to_numpy().astype(int)
    groups = data_p1["campaign"].to_numpy()
    tr, te = next(GroupKFold(n_splits=4).split(x, y, groups))
    roc_curves, pr_curves = {}, {}
    for name, model in build_models(seed=SEED).items():
        model.fit(x[tr], y[tr])
        s = model.predict_proba(x[te])[:, 1]
        fpr, tpr, _ = roc_curve(y[te], s)
        prec, rec, _ = precision_recall_curve(y[te], s)
        roc_curves[name] = (fpr, tpr)
        pr_curves[name] = (rec, prec)
    viz.plot_detection_curves(ROOT / "figures" / "fig_detection_curves.pdf",
                              roc_curves, pr_curves, base_rate)
    log("wrote figures/fig_detection_curves.pdf")

    # -------------------------------------------------------- ablation
    log("feature-group ablation ...")
    abl = ablate_feature_groups(data_p1, seed=SEED)
    save_csv(abl, "detection_ablation.csv")
    full = float(abl[abl["setting"] == "all"]["pr_auc"].iloc[0])
    abl["delta_pr"] = abl["pr_auc"] - full
    abl["label"] = abl["setting"].str.replace("only:", "only ", regex=False) \
                                 .str.replace("drop:", "without ", regex=False)
    abl_tab = pd.DataFrame([{
        "Feature set": r["label"],
        "$|F|$": int(r["n_features"]),
        "PR-AUC": f"{r['pr_auc']:.3f} $\\pm$ {r['pr_auc_sd']:.3f}",
        r"$\Delta$ PR-AUC": f"{r['delta_pr']:+.3f}",
        "ROC-AUC": f"{r['roc_auc']:.3f}",
        "Rec.@1\\%FPR": f"{r['recall_at_fpr']:.3f}",
    } for _, r in abl.iterrows()])
    write_table(
        abl_tab, "tab_det_ablation.tex",
        caption=("Feature-group ablation for the gradient-boosted detector. "
                 "``only'' keeps a single group; ``without'' removes it."),
        label="tab:detablation", escape=False, fit=True, tabcolsep=4,
        column_format="lrrrrr",
        note=("The temporal group holds rolling-window derivations of the raw "
              "signals and is strictly backward-looking, so its contribution is "
              "not label leakage."),
    )
    plot_df = abl[abl["setting"] != "all"].copy()
    viz.plot_ablation_bars(ROOT / "figures" / "fig_det_ablation.pdf", plot_df,
                           value="pr_auc", err="pr_auc_sd", label_col="label",
                           xlabel="PR-AUC")
    log("wrote figures/fig_det_ablation.pdf")

    # ----------------------------------------------------------- shift
    log("posture-shift evaluation ...")
    shift = domain_shift_eval(data_p1, data_p2, seed=SEED)
    save_csv(shift, "detection_shift.csv")
    within = agg.set_index("model")
    shift_tab = pd.DataFrame([{
        "Detector": r["model"],
        "PR-AUC (within P1)": f"{float(within.loc[r['model'], 'pr']):.3f}",
        "PR-AUC (P1 $\\to$ P2)": f"{r['pr_auc']:.3f}",
        "Relative drop": f"{100 * (1 - r['pr_auc'] / float(within.loc[r['model'], 'pr'])):.1f}\\%",
        "ROC-AUC (P1 $\\to$ P2)": f"{r['roc_auc']:.3f}",
        "Rec.@1\\%FPR": f"{r['recall_at_fpr']:.3f}",
    } for _, r in shift.iterrows()])
    write_table(
        shift_tab, "tab_shift.tex",
        caption=("Posture shift. Detectors are trained on telemetry from the "
                 "typical posture P1 and evaluated on the hardened posture P2, "
                 "where intrusions are shorter, smaller and quieter."),
        label="tab:shift", escape=False, star=True, fit=True, tabcolsep=4,
        column_format="lrrrrr",
        note=("Cross-validation within a single posture cannot expose this gap, "
              "which matters because a hospital that hardens its estate changes "
              "the distribution its own detectors were tuned on."),
    )

    best = agg.iloc[0]
    zb = agg[agg["model"] == "Z-score baseline"].iloc[0]
    temporal_drop = float(
        abl[abl["setting"] == "drop:temporal"]["pr_auc"].iloc[0]) - full
    shift_best = shift.sort_values("pr_auc", ascending=False).iloc[0]
    write_macros({
        "DetBase": f"{base_rate:.3f}",
        "DetHours": f"{len(data_p1):,}",
        "DetCampaigns": N_CAMPAIGNS,
        "DetBest": str(best["model"]),
        "DetBestROC": f"{best['roc']:.3f}",
        "DetBestPR": f"{best['pr']:.3f}",
        "DetBestRec": f"{best['rec']:.3f}",
        "DetBaselinePR": f"{zb['pr']:.3f}",
        "DetBaselineROC": f"{zb['roc']:.3f}",
        "DetTemporalDrop": f"{abs(temporal_drop):.3f}",
        "DetShiftPR": f"{shift_best['pr_auc']:.3f}",
        "DetShiftDrop": f"{100 * (1 - float(shift_best['pr_auc']) / float(within.loc[shift_best['model'], 'pr'])):.0f}",
    })


if __name__ == "__main__":
    main()
