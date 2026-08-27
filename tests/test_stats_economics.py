"""Statistical helpers and the economic layer."""
import numpy as np
import pytest

from hermes import controls as ctrl
from hermes import economics as eco
from hermes import statsx as sx


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    x = rng.lognormal(mean=1.0, sigma=0.8, size=5_000)
    point, lo, hi = sx.bootstrap_ci(x, n_boot=800)
    assert lo <= point <= hi
    assert np.isclose(point, x.mean())


def test_bootstrap_ci_narrows_with_sample_size():
    rng = np.random.default_rng(1)
    small = rng.normal(size=500)
    large = rng.normal(size=20_000)
    _, l1, h1 = sx.bootstrap_ci(small, n_boot=600)
    _, l2, h2 = sx.bootstrap_ci(large, n_boot=600)
    assert (h2 - l2) < (h1 - l1)


def test_paired_bootstrap_requires_alignment():
    with pytest.raises(ValueError):
        sx.paired_bootstrap_diff(np.zeros(10), np.zeros(11))


def test_paired_interval_is_tighter_than_unpaired():
    rng = np.random.default_rng(2)
    common = rng.normal(size=4_000)
    a = common + rng.normal(scale=0.05, size=4_000)
    b = common + rng.normal(scale=0.05, size=4_000) + 0.2
    _, plo, phi = sx.paired_bootstrap_diff(a, b, n_boot=800)
    _, alo, ahi = sx.bootstrap_ci(a, n_boot=800)
    _, blo, bhi = sx.bootstrap_ci(b, n_boot=800)
    assert (phi - plo) < ((ahi - alo) + (bhi - blo))


def test_cliffs_delta_sign_and_bounds():
    a = np.arange(1_000, dtype=float)
    b = np.arange(1_000, dtype=float) + 5_000
    assert sx.cliffs_delta(a, b) == pytest.approx(-1.0, abs=1e-9)
    assert sx.cliffs_delta(b, a) == pytest.approx(1.0, abs=1e-9)
    assert abs(sx.cliffs_delta(a, a.copy())) < 0.05


def test_interpret_delta_thresholds():
    assert sx.interpret_delta(0.05) == "negligible"
    assert sx.interpret_delta(0.20) == "small"
    assert sx.interpret_delta(0.40) == "medium"
    assert sx.interpret_delta(0.80) == "large"


def test_holm_is_monotone_and_conservative():
    p = [0.001, 0.02, 0.04, 0.5]
    adj = sx.holm_bonferroni(p)
    assert all(a >= b for a, b in zip(adj, p))
    assert adj == sorted(adj)
    assert all(0.0 <= a <= 1.0 for a in adj)


def test_holm_single_hypothesis_is_identity():
    assert sx.holm_bonferroni([0.03])[0] == pytest.approx(0.03)


def test_required_replications_scales_with_precision():
    rng = np.random.default_rng(3)
    x = rng.lognormal(0.0, 1.0, 5_000)
    assert sx.required_replications(x, 0.01) > sx.required_replications(x, 0.05)


def test_convergence_trace_is_increasing_in_n():
    rng = np.random.default_rng(4)
    tr = sx.convergence_trace(rng.normal(size=3_000))
    assert np.all(np.diff(tr["n"]) > 0)
    assert tr["se"][-1] < tr["se"][0]


def test_annualised_loss_scales_linearly():
    x = np.array([1.0, 2.0, 3.0])
    assert eco.annualised_loss(x, 2.0) == pytest.approx(4.0)


def test_aggregate_annual_loss_mean_matches_ale():
    rng = np.random.default_rng(5)
    loss = rng.lognormal(12.0, 1.0, 5_000)
    agg = eco.aggregate_annual_loss(loss, 1.5, n_years=40_000, seed=7)
    assert agg.mean() == pytest.approx(loss.mean() * 1.5, rel=0.05)


def test_exceedance_curve_is_non_increasing():
    rng = np.random.default_rng(6)
    agg = rng.lognormal(12.0, 1.0, 20_000)
    _, prob = eco.loss_exceedance_curve(agg)
    assert np.all(np.diff(prob) <= 1e-12)


def test_tvar_dominates_var():
    rng = np.random.default_rng(8)
    agg = rng.lognormal(12.0, 1.2, 20_000)
    assert eco.tail_value_at_risk(agg, 0.95) >= eco.value_at_risk(agg, 0.95)


def test_rosi_undefined_without_incremental_spend():
    assert np.isnan(eco.rosi(10.0, 5.0, 1.0, 1.0))


def test_rosi_sign():
    assert eco.rosi(10.0, 2.0, 1.0) > 0
    assert eco.rosi(10.0, 9.9, 5.0) < 0


def test_investment_table_reference_row_is_zero():
    ales = {"P0_none": 90e6, "P1_baseline": 11e6, "P2_hardened": 1.2e6,
            "P3_zero_trust": 0.4e6, "P4_upper_bound": 0.25e6}
    rows = eco.investment_table(ales, ctrl.PORTFOLIOS, reference="P1_baseline")
    ref = [r for r in rows if r.name == "P1_baseline"][0]
    assert ref.ale_avoided == pytest.approx(0.0)
    assert ref.incremental_cost == pytest.approx(0.0)


def test_greedy_frontier_is_monotone_in_cost():
    def evaluate(p):
        return 100.0 - 10.0 * sum(p.values())
    rec = eco.greedy_frontier(evaluate, [1e5, 3e5, 6e5], step=0.10)
    costs = [r["cost"] for r in rec]
    assert costs == sorted(costs)
    assert all(0.0 <= v <= 1.0 for r in rec for v in r["portfolio"].values())
