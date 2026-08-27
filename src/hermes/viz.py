"""Figure generation.

All figures are written as vector PDF at a fixed physical width so that they
drop into a two-column IEEE layout without rescaling, which keeps font sizes
consistent between the figure and the body text.  A single colour-blind-safe
palette is used throughout, and every figure that encodes an estimate also
encodes its uncertainty.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #

#: Okabe-Ito, safe for the three common forms of colour vision deficiency.
PALETTE: Tuple[str, ...] = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#555555",
)

COL_SINGLE = 3.4   # inches, one IEEE column
COL_DOUBLE = 7.1   # inches, full text width


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.0,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.4,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.2,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })


def save(fig: plt.Figure, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Architecture diagram
# --------------------------------------------------------------------------- #

def plot_architecture(path: Path | str, edges: Sequence[Tuple[str, str, str]],
                      layout: Mapping[str, Tuple[float, float]],
                      labels: Mapping[str, str],
                      tiers: Mapping[str, str]) -> Path:
    """Draw the reference hospital attack graph."""
    apply_style()
    tier_colour = {
        "external": "#BBBBBB", "perimeter": "#56B4E9", "endpoint": "#E69F00",
        "core": "#D55E00", "clinical": "#009E73", "business": "#CC79A7",
    }
    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 3.5))
    for src, dst, kind in edges:
        if src not in layout or dst not in layout:
            continue
        x1, y1 = layout[src]
        x2, y2 = layout[dst]
        style = "arc3,rad=0.14" if kind != "lateral" else "arc3,rad=-0.20"
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), connectionstyle=style,
            arrowstyle="-|>", mutation_scale=7,
            linewidth=0.55, color="#777777", alpha=0.65,
            shrinkA=17, shrinkB=17, zorder=1,
        ))
    for node, (x, y) in layout.items():
        colour = tier_colour.get(tiers.get(node, "core"), "#888888")
        ax.add_patch(FancyBboxPatch(
            (x - 0.46, y - 0.20), 0.92, 0.40,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=0.7, edgecolor="#333333", facecolor=colour,
            alpha=0.85, zorder=2,
        ))
        ax.text(x, y, labels.get(node, node), ha="center", va="center",
                fontsize=5.9, zorder=3, wrap=True)
    xs = [p[0] for p in layout.values()]
    ys = [p[1] for p in layout.values()]
    ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
    ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
    ax.set_axis_off()
    handles = [plt.Line2D([0], [0], marker="s", linestyle="none", markersize=5,
                          markerfacecolor=c, markeredgecolor="#333333", label=t.title())
               for t, c in tier_colour.items()]
    ax.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
              bbox_to_anchor=(0.5, -0.06))
    return save(fig, path)


# --------------------------------------------------------------------------- #
# Result figures
# --------------------------------------------------------------------------- #

def plot_zone_heatmap(path: Path | str, matrix: pd.DataFrame,
                      cbar_label: str = "P(zone compromised)") -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 2.5))
    data = matrix.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="YlOrRd", vmin=0.0, vmax=max(data.max(), 1e-6),
                   aspect="auto")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=38, ha="right")
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5.4,
                    color="white" if val > 0.55 * data.max() else "#222222")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.012)
    cb.set_label(cbar_label, fontsize=6.8)
    cb.ax.tick_params(labelsize=6)
    return save(fig, path)


def plot_convergence(path: Path | str, traces: Mapping[str, Dict[str, np.ndarray]],
                     ylabel: str = "Running mean loss (M)") -> Path:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(COL_DOUBLE, 2.2))
    for k, (name, tr) in enumerate(traces.items()):
        c = PALETTE[k % len(PALETTE)]
        axes[0].plot(tr["n"], tr["mean"] / 1e6, color=c, label=name)
        axes[0].fill_between(tr["n"], (tr["mean"] - 1.96 * tr["se"]) / 1e6,
                             (tr["mean"] + 1.96 * tr["se"]) / 1e6,
                             color=c, alpha=0.15, linewidth=0)
        rel = 1.96 * tr["se"] / np.maximum(np.abs(tr["mean"]), 1e-9)
        axes[1].loglog(tr["n"], rel, color=c, label=name)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Replications")
    axes[0].set_ylabel(ylabel)
    axes[0].legend(frameon=False)
    axes[1].axhline(0.05, color="#999999", linestyle=":", linewidth=0.9)
    axes[1].text(axes[1].get_xlim()[0] * 1.2, 0.053, "5 % target", fontsize=5.8,
                 color="#666666")
    axes[1].set_xlabel("Replications")
    axes[1].set_ylabel("Relative 95 % CI half-width")
    return save(fig, path)


def plot_lec(path: Path | str, curves: Mapping[str, Tuple[np.ndarray, np.ndarray]],
             ) -> Path:
    """Loss exceedance curves on log-log axes."""
    apply_style()
    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.3))
    for k, (name, (grid, prob)) in enumerate(curves.items()):
        keep = prob > 0
        ax.loglog(grid[keep] / 1e6, prob[keep], color=PALETTE[k % len(PALETTE)],
                  label=name)
    ax.set_xlabel("Annual loss threshold (M)")
    ax.set_ylabel("P(annual loss > threshold)")
    ax.legend(frameon=False, loc="lower left")
    return save(fig, path)


def plot_control_effects(path: Path | str, df: pd.DataFrame,
                         value: str = "delta_pct", err_lo: str = "lo_pct",
                         err_hi: str = "hi_pct",
                         xlabel: str = "Increase in expected loss when removed (%)"
                         ) -> Path:
    apply_style()
    d = df.sort_values(value)
    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.35))
    y = np.arange(len(d))
    lo = np.abs(d[value] - d[err_lo])
    hi = np.abs(d[err_hi] - d[value])
    ax.barh(y, d[value], color=PALETTE[0], alpha=0.85, height=0.62,
            xerr=[lo, hi], error_kw=dict(elinewidth=0.7, capsize=1.8,
                                         ecolor="#333333"))
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"])
    ax.set_xlabel(xlabel)
    ax.axvline(0, color="#333333", linewidth=0.6)
    return save(fig, path)


def plot_sobol(path: Path | str, rows: Sequence[Mapping[str, object]]) -> Path:
    apply_style()
    d = pd.DataFrame(list(rows)).sort_values("ST")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.6))
    ax.barh(y + 0.19, d["ST"], height=0.36, color=PALETTE[1], alpha=0.9,
            label="Total order $S_T$",
            xerr=[d["ST"] - d["ST_lo"], d["ST_hi"] - d["ST"]],
            error_kw=dict(elinewidth=0.6, capsize=1.5, ecolor="#333333"))
    ax.barh(y - 0.19, d["S1"], height=0.36, color=PALETTE[0], alpha=0.9,
            label="First order $S_1$",
            xerr=[np.maximum(d["S1"] - d["S1_lo"], 0), np.maximum(d["S1_hi"] - d["S1"], 0)],
            error_kw=dict(elinewidth=0.6, capsize=1.5, ecolor="#333333"))
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"])
    ax.set_xlabel("Sobol index")
    ax.legend(frameon=False, loc="lower right")
    return save(fig, path)


def plot_frontier(path: Path | str, frontier: pd.DataFrame,
                  points: Optional[pd.DataFrame] = None) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.3))
    ax.plot(frontier["cost"] / 1e6, frontier["ale"] / 1e6, color=PALETTE[0],
            marker="o", markersize=2.4, label="Greedy frontier")
    if points is not None and len(points):
        ax.scatter(points["cost"] / 1e6, points["ale"] / 1e6, s=22,
                   color=PALETTE[1], zorder=4, label="Named portfolios")
        for _, r in points.iterrows():
            ax.annotate(r["short"], (r["cost"] / 1e6, r["ale"] / 1e6),
                        textcoords="offset points", xytext=(4, 4), fontsize=5.8)
    ax.set_xlabel("Annual control spend (M)")
    ax.set_ylabel("Annualised loss expectancy (M)")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    return save(fig, path)


def plot_interaction(path: Path | str, grid: np.ndarray,
                     x_vals: Sequence[float], y_vals: Sequence[float],
                     xlabel: str, ylabel: str, cbar_label: str) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.4))
    im = ax.imshow(grid, origin="lower", cmap="viridis", aspect="auto",
                   extent=(min(x_vals), max(x_vals), min(y_vals), max(y_vals)))
    cs = ax.contour(np.linspace(min(x_vals), max(x_vals), grid.shape[1]),
                    np.linspace(min(y_vals), max(y_vals), grid.shape[0]),
                    grid, colors="white", linewidths=0.5, levels=6)
    ax.clabel(cs, inline=True, fontsize=5.2, fmt="%.1f")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(cbar_label, fontsize=6.6)
    cb.ax.tick_params(labelsize=6)
    return save(fig, path)


def plot_detection_curves(path: Path | str,
                          roc: Mapping[str, Tuple[np.ndarray, np.ndarray]],
                          pr: Mapping[str, Tuple[np.ndarray, np.ndarray]],
                          base_rate: float) -> Path:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(COL_DOUBLE, 2.3))
    for k, (name, (fpr, tpr)) in enumerate(roc.items()):
        axes[0].plot(fpr, tpr, color=PALETTE[k % len(PALETTE)], label=name)
    axes[0].plot([0, 1], [0, 1], color="#999999", linestyle=":", linewidth=0.8)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].legend(frameon=False, loc="lower right")

    for k, (name, (rec, prec)) in enumerate(pr.items()):
        axes[1].plot(rec, prec, color=PALETTE[k % len(PALETTE)], label=name)
    axes[1].axhline(base_rate, color="#999999", linestyle=":", linewidth=0.8)
    axes[1].text(0.03, base_rate * 1.08, "chance", fontsize=5.8, color="#666666")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_ylim(0, 1.02)
    return save(fig, path)


def plot_ablation_bars(path: Path | str, df: pd.DataFrame, value: str,
                       err: Optional[str], label_col: str, xlabel: str) -> Path:
    apply_style()
    d = df.sort_values(value)
    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.5))
    y = np.arange(len(d))
    kw = {}
    if err and err in d:
        kw["xerr"] = d[err]
        kw["error_kw"] = dict(elinewidth=0.7, capsize=1.8, ecolor="#333333")
    ax.barh(y, d[value], color=PALETTE[2], alpha=0.88, height=0.62, **kw)
    ax.set_yticks(y)
    ax.set_yticklabels(d[label_col])
    ax.set_xlabel(xlabel)
    return save(fig, path)


def plot_calibration(path: Path | str, panels: Sequence[Dict[str, object]]) -> Path:
    """Simulated distribution against publicly reported incident values."""
    apply_style()
    fig, axes = plt.subplots(1, len(panels), figsize=(COL_DOUBLE, 2.1))
    if len(panels) == 1:
        axes = [axes]
    for ax, panel in zip(axes, panels):
        data = np.asarray(panel["samples"], dtype=float)
        ax.hist(data, bins=int(panel.get("bins", 34)), color=PALETTE[0],
                alpha=0.55, density=True, edgecolor="none")
        lo, hi = panel["reported"]
        ax.axvspan(lo, hi, color=PALETTE[1], alpha=0.22, linewidth=0)
        ax.axvline(lo, color=PALETTE[1], linewidth=0.9)
        ax.axvline(hi, color=PALETTE[1], linewidth=0.9)
        ax.set_xlabel(str(panel["xlabel"]))
        ax.set_title(str(panel["title"]), fontsize=7.0)
        if panel.get("logx"):
            ax.set_xscale("log")
        ax.set_yticks([])
    axes[0].set_ylabel("Density")
    handles = [
        plt.Line2D([0], [0], color=PALETTE[0], alpha=0.6, linewidth=4,
                   label="HERMES simulation"),
        plt.Line2D([0], [0], color=PALETTE[1], linewidth=4, alpha=0.4,
                   label="Publicly reported range"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.10))
    return save(fig, path)


def plot_dwell_distributions(path: Path | str,
                             dwell: Mapping[str, np.ndarray]) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.2))
    for k, (name, arr) in enumerate(dwell.items()):
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if arr.size == 0:
            continue
        xs = np.sort(arr)
        ys = np.arange(1, xs.size + 1) / xs.size
        ax.step(xs, ys, where="post", color=PALETTE[k % len(PALETTE)], label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Dwell time to detection (h)")
    ax.set_ylabel("Empirical CDF")
    ax.legend(frameon=False, loc="lower right")
    return save(fig, path)
