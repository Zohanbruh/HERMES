"""Shared plumbing for the experiment scripts.

Keeping paths, seeds, and the LaTeX writer in one place means the manuscript and
the repository can never drift apart: every number in the paper is emitted by a
script in this directory into ``paper/tables`` or ``paper/figures``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

#: Seeds used for every repeated experiment.  Five independent seeds let us
#: report between-seed variability in addition to within-seed Monte Carlo error.
SEEDS: Sequence[int] = (11, 23, 37, 53, 71)

#: Replications per seed for headline results.
N_RUNS_MAIN = 20_000
#: Replications per seed for sweeps, where many configurations are evaluated.
N_RUNS_SWEEP = 4_000

for d in (RESULTS, FIGURES):
    d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def save_csv(df: pd.DataFrame, name: str) -> Path:
    path = RESULTS / name
    df.to_csv(path, index=False)
    log(f"wrote {path.relative_to(ROOT)}  ({len(df)} rows)")
    return path


def save_json(obj: Mapping[str, object], name: str) -> Path:
    path = RESULTS / name

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    path.write_text(json.dumps(obj, indent=2, default=default))
    log(f"wrote {path.relative_to(ROOT)}")
    return path


# --------------------------------------------------------------------------- #
# LaTeX
# --------------------------------------------------------------------------- #

_LATEX_ESCAPES = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}",
}


def tex_escape(s: object) -> str:
    text = str(s)
    for old, new in _LATEX_ESCAPES.items():
        text = text.replace(old, new)
    return text


def write_table(
    df: pd.DataFrame,
    name: str,
    caption: str,
    label: str,
    column_format: Optional[str] = None,
    star: bool = False,
    note: Optional[str] = None,
    escape: bool = True,
    fontsize: str = r"\footnotesize",
    out_dir: Optional[Path] = None,
    fit: bool = False,
    tabcolsep: Optional[float] = None,
) -> Path:
    """Emit a self-contained ``table`` float that the manuscript ``\\input``s."""
    out_dir = out_dir or (ROOT / "paper" / "tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    env = "table*" if star else "table"
    fmt = column_format or ("l" + "r" * (df.shape[1] - 1))

    body = df.to_latex(
        index=False,
        escape=escape,
        column_format=fmt,
        na_rep="--",
    )
    # Strip the tabular wrapper's default rules and use booktabs consistently.
    lines: List[str] = []
    lines.append(rf"\begin{{{env}}}[!t]")
    lines.append(r"\centering")
    lines.append(fontsize)
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    if tabcolsep is not None:
        lines.append(rf"\setlength{{\tabcolsep}}{{{tabcolsep}pt}}")
    if fit:
        # Wide result tables are scaled to the text block rather than allowed to
        # overflow the column; the scale factors involved are mild.
        width = r"\textwidth" if star else r"\columnwidth"
        lines.append(rf"\resizebox{{{width}}}{{!}}{{%")
        lines.append(body.strip())
        lines.append("}")
    else:
        lines.append(body.strip())
    if note:
        width = r"\linewidth" if not star else r"\textwidth"
        lines.append(rf"\vspace{{2pt}}")
        lines.append(rf"\begin{{minipage}}{{{width}}}\scriptsize {note}\end{{minipage}}")
    lines.append(rf"\end{{{env}}}")

    path = out_dir / name
    path.write_text("\n".join(lines) + "\n")
    log(f"wrote {path.relative_to(ROOT)}")
    return path


def write_macros(values: Mapping[str, object], name: str = "numbers.tex") -> Path:
    """Emit ``\\newcommand`` definitions so prose numbers cannot drift.

    Every quantity quoted in the running text of the manuscript is defined here
    by a script, so editing a parameter and re-running the pipeline updates the
    prose as well as the tables.
    """
    out_dir = ROOT / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    existing: Dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith(r"\newcommand{\\"):
                continue
            if line.startswith(r"\newcommand{"):
                key = line.split("{")[1].split("}")[0].lstrip("\\")
                val = line.split("}{", 1)[1].rsplit("}", 1)[0]
                existing[key] = val
    for k, v in values.items():
        existing[str(k)] = str(v)
    lines = [r"% Auto-generated by scripts/*.py -- do not edit by hand."]
    for k in sorted(existing):
        lines.append(rf"\newcommand{{\{k}}}{{{existing[k]}}}")
    path.write_text("\n".join(lines) + "\n")
    log(f"wrote {path.relative_to(ROOT)}  ({len(existing)} macros)")
    return path


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #

def fmt_money(x: float, unit: float = 1e6, digits: int = 2) -> str:
    return f"{x / unit:.{digits}f}"


def fmt_pct(x: float, digits: int = 1) -> str:
    return f"{100.0 * x:.{digits}f}"


def fmt_ci(point: float, lo: float, hi: float, scale: float = 1.0,
           digits: int = 2) -> str:
    return (f"{point / scale:.{digits}f} "
            f"[{lo / scale:.{digits}f}, {hi / scale:.{digits}f}]")


def fmt_p(p: float) -> str:
    if p < 1e-4:
        return r"$<10^{-4}$"
    return f"{p:.4f}"
