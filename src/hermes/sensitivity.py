"""Global and local sensitivity analysis.

Local (one-at-a-time) sensitivity is cheap and easy to read but is only valid
when the model is close to additive.  A defence-in-depth model is explicitly
multiplicative, so interactions matter and a variance-based global method is
required for the headline claim.  Both are provided: the Sobol indices are the
evidence, the tornado plot is the exposition.

The Sobol estimators are the Saltelli cross-sampling scheme with the Jansen
formulations of the first-order and total-order indices, which are the
lower-variance choices for the sample sizes used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import qmc


@dataclass(frozen=True)
class Parameter:
    """A model input varied during sensitivity analysis."""

    key: str
    label: str
    low: float
    high: float

    def scale(self, u: np.ndarray) -> np.ndarray:
        return self.low + u * (self.high - self.low)


@dataclass
class SobolResult:
    names: List[str]
    labels: List[str]
    s1: np.ndarray
    s1_ci: np.ndarray      # (D, 2)
    st: np.ndarray
    st_ci: np.ndarray      # (D, 2)
    n_base: int
    n_evaluations: int
    mean_output: float
    var_output: float

    def as_rows(self) -> List[Dict[str, object]]:
        return [
            {
                "parameter": self.names[i],
                "label": self.labels[i],
                "S1": float(self.s1[i]),
                "S1_lo": float(self.s1_ci[i, 0]),
                "S1_hi": float(self.s1_ci[i, 1]),
                "ST": float(self.st[i]),
                "ST_lo": float(self.st_ci[i, 0]),
                "ST_hi": float(self.st_ci[i, 1]),
            }
            for i in range(len(self.names))
        ]


def saltelli_sample(
    params: Sequence[Parameter], n_base: int, seed: int = 2024
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the ``A``, ``B`` and stacked ``AB`` design matrices.

    ``n_base`` should be a power of two so that the Sobol sequence retains its
    balance properties; a warning-free power-of-two check is enforced.
    """
    d = len(params)
    if n_base & (n_base - 1) != 0:
        raise ValueError("n_base must be a power of two for a balanced Sobol design")
    engine = qmc.Sobol(d=2 * d, scramble=True, seed=seed)
    raw = engine.random(n_base)
    a_u, b_u = raw[:, :d], raw[:, d:]

    def scale(u: np.ndarray) -> np.ndarray:
        return np.column_stack([p.scale(u[:, i]) for i, p in enumerate(params)])

    a, b = scale(a_u), scale(b_u)
    ab = np.empty((d, n_base, d), dtype=float)
    for i in range(d):
        mat = a.copy()
        mat[:, i] = b[:, i]
        ab[i] = mat
    return a, b, ab


def _jansen_indices(
    ya: np.ndarray, yb: np.ndarray, yab: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """First-order and total-order indices, Jansen estimators."""
    var = np.var(np.concatenate([ya, yb]), ddof=1)
    if var <= 0.0:
        d = yab.shape[0]
        return np.zeros(d), np.zeros(d)
    s1 = (var - 0.5 * np.mean((yb[None, :] - yab) ** 2, axis=1)) / var
    st = 0.5 * np.mean((ya[None, :] - yab) ** 2, axis=1) / var
    return s1, st


def sobol_analysis(
    model: Callable[[np.ndarray], float],
    params: Sequence[Parameter],
    n_base: int = 512,
    seed: int = 2024,
    n_boot: int = 400,
    progress: bool = False,
) -> SobolResult:
    """Variance-based sensitivity of ``model`` to ``params``.

    ``model`` receives one parameter vector and returns a scalar.  Bootstrap
    intervals are obtained by resampling the base rows, which correctly
    propagates the shared randomness across the ``A``, ``B`` and ``AB`` designs.
    """
    d = len(params)
    a, b, ab = saltelli_sample(params, n_base, seed=seed)

    def run(matrix: np.ndarray) -> np.ndarray:
        return np.array([model(row) for row in matrix], dtype=float)

    if progress:
        print(f"  Sobol: evaluating {n_base * (d + 2)} model runs ...", flush=True)
    ya, yb = run(a), run(b)
    yab = np.array([run(ab[i]) for i in range(d)], dtype=float)   # (D, n_base)

    s1, st = _jansen_indices(ya, yb, yab)

    rng = np.random.default_rng(seed + 1)
    s1_boot = np.empty((n_boot, d))
    st_boot = np.empty((n_boot, d))
    for k in range(n_boot):
        idx = rng.integers(0, n_base, n_base)
        s1_boot[k], st_boot[k] = _jansen_indices(ya[idx], yb[idx], yab[:, idx])

    s1_ci = np.quantile(s1_boot, [0.025, 0.975], axis=0).T
    st_ci = np.quantile(st_boot, [0.025, 0.975], axis=0).T

    combined = np.concatenate([ya, yb])
    return SobolResult(
        names=[p.key for p in params],
        labels=[p.label for p in params],
        s1=s1, s1_ci=s1_ci, st=st, st_ci=st_ci,
        n_base=n_base,
        n_evaluations=int(n_base * (d + 2)),
        mean_output=float(combined.mean()),
        var_output=float(combined.var(ddof=1)),
    )


def oat_tornado(
    model: Callable[[np.ndarray], float],
    params: Sequence[Parameter],
    baseline: np.ndarray,
) -> List[Dict[str, float]]:
    """One-at-a-time swing of the output when each input moves to its bounds."""
    base = model(np.asarray(baseline, dtype=float))
    rows: List[Dict[str, float]] = []
    for i, p in enumerate(params):
        lo_vec = np.array(baseline, dtype=float)
        hi_vec = np.array(baseline, dtype=float)
        lo_vec[i], hi_vec[i] = p.low, p.high
        lo_y, hi_y = model(lo_vec), model(hi_vec)
        rows.append(
            {
                "parameter": p.key,
                "label": p.label,
                "low_output": float(lo_y),
                "high_output": float(hi_y),
                "baseline_output": float(base),
                "swing": float(abs(hi_y - lo_y)),
            }
        )
    rows.sort(key=lambda r: r["swing"], reverse=True)
    return rows
