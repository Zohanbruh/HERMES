"""Statistical helpers used to report simulation results honestly.

Simulation output is cheap, so a naive experiment can manufacture arbitrarily
small p-values simply by increasing the replication count.  Every comparison in
this work is therefore reported with (a) a confidence interval on the estimate,
(b) a distribution-free effect size, and (c) a family-wise error correction over
the whole comparison family.  Point estimates are never reported alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats


#: Maximum number of resampled cells held in memory at once.  Bootstrapping a
#: 100k-element sample with a dense index matrix would otherwise exhaust RAM.
_BOOT_CELLS: int = 8_000_000


# --------------------------------------------------------------------------- #
# Uncertainty on a mean
# --------------------------------------------------------------------------- #

def mc_standard_error(x: np.ndarray) -> float:
    """Monte Carlo standard error of the sample mean."""
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(x.size))


def bootstrap_ci(
    x: np.ndarray,
    statistic=np.mean,
    n_boot: int = 5_000,
    alpha: float = 0.05,
    seed: int = 12345,
) -> Tuple[float, float, float]:
    """Percentile bootstrap interval for ``statistic``.

    Returns ``(point, lower, upper)``.  The percentile interval is used rather
    than a normal interval because simulated loss is heavy-tailed and strongly
    right-skewed, so the sampling distribution of the mean is not symmetric at
    the replication counts used here.
    """
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot, dtype=float)
    # Resampling is done in chunks: a dense (n_boot, n) index matrix would need
    # tens of gigabytes at the replication counts used in the main experiment.
    chunk = max(1, int(_BOOT_CELLS // max(x.size, 1)))
    done = 0
    while done < n_boot:
        k = min(chunk, n_boot - done)
        idx = rng.integers(0, x.size, size=(k, x.size))
        reps[done:done + k] = statistic(x[idx], axis=1)
        done += k
    lo, hi = np.quantile(reps, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(statistic(x)), float(lo), float(hi)


def paired_bootstrap_diff(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 5_000,
    alpha: float = 0.05,
    seed: int = 12345,
) -> Tuple[float, float, float]:
    """Bootstrap interval for ``mean(a) - mean(b)`` under common random numbers.

    ``a`` and ``b`` must be aligned replication-by-replication, which holds when
    both were produced with the same seed.  Pairing removes the shared
    randomness and typically shrinks the interval by an order of magnitude
    relative to an unpaired comparison.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired comparison requires equal-length samples")
    d = a - b
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot, dtype=float)
    chunk = max(1, int(_BOOT_CELLS // max(d.size, 1)))
    done = 0
    while done < n_boot:
        k = min(chunk, n_boot - done)
        idx = rng.integers(0, d.size, size=(k, d.size))
        reps[done:done + k] = d[idx].mean(axis=1)
        done += k
    lo, hi = np.quantile(reps, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(d.mean()), float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Effect sizes
# --------------------------------------------------------------------------- #

def cliffs_delta(a: np.ndarray, b: np.ndarray, max_n: int = 4_000,
                 seed: int = 7) -> float:
    """Cliff's delta, a rank-based effect size in ``[-1, 1]``.

    Interpretation thresholds in common use: ``|d| < 0.147`` negligible,
    ``< 0.33`` small, ``< 0.474`` medium, otherwise large.  For tractability on
    large samples the statistic is computed on a random subsample when either
    input exceeds ``max_n``; the estimator remains unbiased.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size > max_n:
        a = rng.choice(a, max_n, replace=False)
    if b.size > max_n:
        b = rng.choice(b, max_n, replace=False)
    order = np.argsort(b)
    b_sorted = b[order]
    greater = np.searchsorted(b_sorted, a, side="left")
    geq = np.searchsorted(b_sorted, a, side="right")
    less = b_sorted.size - geq
    return float((greater.sum() - less.sum()) / (a.size * b.size))


def interpret_delta(d: float) -> str:
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    if ad < 0.330:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def relative_reduction(a: np.ndarray, b: np.ndarray) -> float:
    """Fractional reduction of the mean of ``a`` relative to ``b``."""
    mb = float(np.mean(b))
    if mb == 0.0:
        return float("nan")
    return float(1.0 - np.mean(a) / mb)


# --------------------------------------------------------------------------- #
# Hypothesis tests and multiplicity
# --------------------------------------------------------------------------- #

@dataclass
class Comparison:
    name: str
    estimate: float
    ci_low: float
    ci_high: float
    p_value: float
    p_adjusted: float
    delta: float
    delta_label: str


def mannwhitney_p(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Mann-Whitney U p-value; distribution-free."""
    if np.allclose(a, b):
        return 1.0
    try:
        return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    """Holm step-down adjusted p-values, controlling the family-wise error rate."""
    p = np.asarray(list(p_values), dtype=float)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adjusted[idx] = min(1.0, running)
    return [float(v) for v in adjusted]


def compare_family(
    samples: Dict[str, np.ndarray],
    reference: np.ndarray,
    paired: bool = True,
    seed: int = 12345,
) -> List[Comparison]:
    """Compare every entry of ``samples`` against ``reference`` with Holm control."""
    names = list(samples.keys())
    raw_p, rows = [], []
    for name in names:
        arr = np.asarray(samples[name], dtype=float)
        if paired and arr.shape == reference.shape:
            est, lo, hi = paired_bootstrap_diff(arr, reference, seed=seed)
        else:
            e1, l1, h1 = bootstrap_ci(arr, seed=seed)
            e0, l0, h0 = bootstrap_ci(reference, seed=seed + 1)
            est, lo, hi = e1 - e0, l1 - h0, h1 - l0
        p = mannwhitney_p(arr, reference)
        d = cliffs_delta(arr, reference)
        raw_p.append(p)
        rows.append((name, est, lo, hi, p, d))

    adj = holm_bonferroni(raw_p)
    return [
        Comparison(n, e, lo, hi, p, a, d, interpret_delta(d))
        for (n, e, lo, hi, p, d), a in zip(rows, adj)
    ]


# --------------------------------------------------------------------------- #
# Convergence diagnostics
# --------------------------------------------------------------------------- #

def convergence_trace(x: np.ndarray, points: int = 60) -> Dict[str, np.ndarray]:
    """Running mean and MC standard error as the replication count grows."""
    x = np.asarray(x, dtype=float)
    n = x.size
    grid = np.unique(np.geomspace(50, n, points).astype(int))
    means = np.array([x[:k].mean() for k in grid])
    ses = np.array([x[:k].std(ddof=1) / np.sqrt(k) for k in grid])
    return {"n": grid, "mean": means, "se": ses}


def required_replications(x: np.ndarray, rel_precision: float = 0.02,
                          z: float = 1.96) -> int:
    """Replications needed so the half-width of a 95 % CI is ``rel_precision``.

    Uses the standard normal-approximation sample-size expression
    :math:`n = (z\\,\\sigma / (\\epsilon\\,\\mu))^2`.
    """
    x = np.asarray(x, dtype=float)
    mu, sd = float(x.mean()), float(x.std(ddof=1))
    if mu == 0.0:
        return int(x.size)
    return int(np.ceil((z * sd / (rel_precision * mu)) ** 2))
