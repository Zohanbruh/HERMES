# Reproducibility checklist

## Environment used for the reported numbers

| Item | Value |
|---|---|
| OS | Ubuntu 24.04 (x86-64) |
| Python | 3.12 |
| NumPy | 2.4.4 |
| SciPy | 1.17.1 |
| pandas | 3.0.2 |
| scikit-learn | 1.8.0 |
| matplotlib | 3.10.8 |
| networkx | 3.6.1 |
| Cores used | 1 |
| Wall-clock for `make all` | ~15 minutes |

## Determinism

Every stochastic component is driven by an explicitly seeded
`numpy.random.default_rng`. No global random state is used anywhere in `src/`.

* Simulation seeds: `SEEDS = (11, 23, 37, 53, 71)` (`scripts/_common.py`).
* Sobol design seed: 2024; model-evaluation seed: 909.
* Interaction sweep seed: 4242. Frontier seed: 777. Detection seed: 31.
* Bootstrap seeds are passed explicitly at every call site.

Re-running `make all` on the same versions reproduces `results/` exactly.

## Experiment scale

| Experiment | Configurations | Replications each | Total |
|---|---|---|---|
| Main portfolios | 5 | 5 x 20,000 | 500,000 |
| Paired comparisons | 4 | 5 x 20,000 | 400,000 |
| Leave-one-out ablation | 7 | 5 x 20,000 | 700,000 |
| Single-control ablation | 7 | 5 x 20,000 | 700,000 |
| Sobol | 6,144 | 1,024 | 6,291,456 |
| Interaction sweeps | 242 + 8 | 4,000 | 1,000,000 |
| Greedy frontier | ~700 | 4,000 | 2,800,000 |

## Statistical protocol

1. All interval estimates are percentile bootstrap intervals.
2. Portfolio comparisons are **paired** under common random numbers.
3. Every family of comparisons is corrected by the Holm step-down procedure.
4. Every comparison reports Cliff's delta alongside the p-value, because at
   these replication counts any non-zero difference is significant.
5. The replication count was chosen from a convergence study, not assumed.

## Validating the estimators themselves

`tests/test_sensitivity_detection.py::test_sobol_recovers_ishigami_ranking`
checks the Sobol implementation against the Ishigami function, whose first-order
indices are known analytically (0.314, 0.442, 0.0). The implementation recovers
`S1[x2] = 0.442` to within 0.06 and correctly assigns `x3` zero first-order but
non-zero total-order variance.

## Detection benchmark integrity

* Cross-validation folds are grouped by campaign, never by hour.
* Temporal features use strictly backward-looking windows; a unit test verifies
  that truncating the future leaves them unchanged.
* A unit test asserts that no single raw feature separates the classes perfectly.
* The reported PR-AUC is well under 1.0. Near-perfect scores on a self-generated
  corpus would indicate a leaky generator, not a good detector.
