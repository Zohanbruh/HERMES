"""Control-model algebra: monotonicity, bounds, and cost behaviour."""
import numpy as np
import pytest

from hermes import controls as ctrl


def test_efficacy_tables_are_well_formed():
    ctrl.validate_efficacy()


def test_portfolio_vector_rejects_out_of_range():
    p = dict(ctrl.PORTFOLIOS["P1_baseline"])
    p["mfa"] = 1.4
    with pytest.raises(ValueError):
        ctrl.portfolio_vector(p)


def test_portfolio_vector_rejects_missing_control():
    p = dict(ctrl.PORTFOLIOS["P1_baseline"])
    del p["siem"]
    with pytest.raises(KeyError):
        ctrl.portfolio_vector(p)


def test_prevention_multiplier_bounds():
    eta = ctrl.eta_matrix()
    for name, p in ctrl.PORTFOLIOS.items():
        m = ctrl.prevention_multiplier(ctrl.portfolio_vector(p), eta)
        assert np.all(m > 0.0), name
        assert np.all(m <= 1.0 + 1e-12), name


def test_prevention_is_monotone_in_coverage():
    eta = ctrl.eta_matrix()
    weak = ctrl.prevention_multiplier(ctrl.portfolio_vector(ctrl.PORTFOLIOS["P1_baseline"]), eta)
    strong = ctrl.prevention_multiplier(ctrl.portfolio_vector(ctrl.PORTFOLIOS["P3_zero_trust"]), eta)
    assert np.all(strong <= weak + 1e-12)


def test_zero_portfolio_is_identity():
    eta = ctrl.eta_matrix()
    m = ctrl.prevention_multiplier(ctrl.portfolio_vector(ctrl.PORTFOLIOS["P0_none"]), eta)
    assert np.allclose(m, 1.0)


def test_detection_gain_at_least_one():
    for p in ctrl.PORTFOLIOS.values():
        assert ctrl.detection_multiplier(ctrl.portfolio_vector(p)) >= 1.0 - 1e-12


def test_containment_and_recovery_multipliers_shrink():
    x0 = ctrl.portfolio_vector(ctrl.PORTFOLIOS["P0_none"])
    x3 = ctrl.portfolio_vector(ctrl.PORTFOLIOS["P3_zero_trust"])
    assert ctrl.containment_multiplier(x3) < ctrl.containment_multiplier(x0)
    assert ctrl.recovery_multiplier(x3) < ctrl.recovery_multiplier(x0)


def test_cost_is_monotone_and_zero_at_zero():
    assert ctrl.annual_cost(ctrl.portfolio_vector(ctrl.PORTFOLIOS["P0_none"])) == 0.0
    costs = [ctrl.annual_cost(ctrl.portfolio_vector(ctrl.PORTFOLIOS[k]))
             for k in ("P0_none", "P1_baseline", "P2_hardened",
                       "P3_zero_trust", "P4_upper_bound")]
    assert costs == sorted(costs)


def test_leave_one_out_zeroes_exactly_one_control():
    base = ctrl.PORTFOLIOS["P2_hardened"]
    for key, variant in ctrl.leave_one_out(base).items():
        assert variant[key] == 0.0
        others = [k for k in ctrl.CONTROL_KEYS if k != key]
        assert all(variant[k] == base[k] for k in others)


def test_only_one_activates_exactly_one_control():
    base = ctrl.PORTFOLIOS["P2_hardened"]
    for key, variant in ctrl.only_one(base).items():
        assert variant[key] == base[key]
        assert all(variant[k] == 0.0 for k in ctrl.CONTROL_KEYS if k != key)
