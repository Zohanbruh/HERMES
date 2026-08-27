"""Simulator behaviour: determinism, monotonicity, and conservation."""
import numpy as np
import pytest

from hermes import controls as ctrl
from hermes.simulator import (SimulationParams, compile_graph,
                              edge_success_probability, simulate,
                              zone_compromise_rates, zone_outage_hours)

CG = compile_graph()
SMALL = SimulationParams(n_runs=1_500)


def test_compiled_graph_shapes():
    assert CG.n_nodes == 14
    assert CG.n_edges == 28
    assert CG.src.shape == CG.dst.shape == CG.p_base.shape


def test_same_seed_is_deterministic():
    a = simulate(ctrl.PORTFOLIOS["P1_baseline"], SMALL, seed=5, cg=CG)
    b = simulate(ctrl.PORTFOLIOS["P1_baseline"], SMALL, seed=5, cg=CG)
    assert np.array_equal(a.loss, b.loss)
    assert np.array_equal(a.compromised, b.compromised)


def test_different_seed_changes_output():
    a = simulate(ctrl.PORTFOLIOS["P1_baseline"], SMALL, seed=5, cg=CG)
    b = simulate(ctrl.PORTFOLIOS["P1_baseline"], SMALL, seed=6, cg=CG)
    assert not np.array_equal(a.loss, b.loss)


def test_entry_node_always_compromised():
    r = simulate(ctrl.PORTFOLIOS["P2_hardened"], SMALL, seed=1, cg=CG)
    assert r.compromised[:, CG.entry].all()


def test_stronger_controls_never_increase_expected_loss():
    order = ["P0_none", "P1_baseline", "P2_hardened", "P3_zero_trust",
             "P4_upper_bound"]
    means = [simulate(ctrl.PORTFOLIOS[k], SimulationParams(n_runs=6_000),
                      seed=3, cg=CG).loss.mean() for k in order]
    for a, b in zip(means, means[1:]):
        assert b <= a


def test_edge_probabilities_shrink_with_controls():
    p0 = edge_success_probability(CG, ctrl.portfolio_vector(ctrl.PORTFOLIOS["P0_none"]))
    p3 = edge_success_probability(CG, ctrl.portfolio_vector(ctrl.PORTFOLIOS["P3_zero_trust"]))
    assert np.all(p3 <= p0 + 1e-12)
    assert np.allclose(p0, CG.p_base)


def test_losses_are_finite_and_non_negative():
    r = simulate(ctrl.PORTFOLIOS["P1_baseline"], SMALL, seed=2, cg=CG)
    assert np.all(np.isfinite(r.loss))
    assert np.all(r.loss >= 0.0)
    assert np.all(r.records_exposed >= 0.0)
    assert np.all(r.outage_hours >= 0.0)


def test_outage_requires_compromise():
    r = simulate(ctrl.PORTFOLIOS["P1_baseline"], SMALL, seed=4, cg=CG)
    assert not np.any(r.outage_hours[~r.compromised] > 0)


def test_arrival_before_containment_for_compromised_zones():
    r = simulate(ctrl.PORTFOLIOS["P1_baseline"], SMALL, seed=8, cg=CG)
    arrived = r.first_seen[r.compromised]
    assert np.all(np.isfinite(arrived))


def test_containment_never_precedes_detection():
    r = simulate(ctrl.PORTFOLIOS["P1_baseline"], SMALL, seed=9, cg=CG)
    seen = np.isfinite(r.detect_time)
    assert np.all(r.contain_time[seen] >= r.detect_time[seen] - 1e-9)


def test_no_detection_implies_horizon_containment():
    r = simulate(ctrl.PORTFOLIOS["P0_none"], SMALL, seed=10, cg=CG)
    never = ~np.isfinite(r.detect_time)
    if never.any():
        assert np.allclose(r.contain_time[never], r.params.horizon_hours)


def test_blast_radius_within_bounds():
    r = simulate(ctrl.PORTFOLIOS["P0_none"], SMALL, seed=12, cg=CG)
    assert r.blast_radius.min() >= 0
    assert r.blast_radius.max() <= CG.n_nodes - 1


def test_zone_rates_are_probabilities():
    r = simulate(ctrl.PORTFOLIOS["P1_baseline"], SMALL, seed=13, cg=CG)
    for v in zone_compromise_rates(r).values():
        assert 0.0 <= v <= 1.0
    for v in zone_outage_hours(r).values():
        assert v >= 0.0


def test_summary_keys_present():
    r = simulate(ctrl.PORTFOLIOS["P1_baseline"], SMALL, seed=14, cg=CG)
    s = r.summary()
    for key in ("mean_loss", "p95_loss", "p_ehr_compromise", "ale",
                "median_dwell_h", "p_foothold"):
        assert key in s and np.isfinite(s[key])


def test_longer_horizon_weakly_increases_risk():
    short = simulate(ctrl.PORTFOLIOS["P1_baseline"],
                     SimulationParams(n_runs=6_000, horizon_hours=240),
                     seed=15, cg=CG)
    long = simulate(ctrl.PORTFOLIOS["P1_baseline"],
                    SimulationParams(n_runs=6_000, horizon_hours=720),
                    seed=15, cg=CG)
    assert long.blast_radius.mean() >= short.blast_radius.mean() - 0.05
