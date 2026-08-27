"""Sensitivity estimators and the telemetry generator."""
import numpy as np
import pytest

from hermes import controls as ctrl
from hermes import sensitivity as sens
from hermes.detection import (BASE_FEATURES, FEATURE_GROUPS, FEATURES,
                              TelemetryConfig, generate_telemetry)
from hermes.simulator import SimulationParams, compile_graph, simulate

PARAMS = [sens.Parameter(f"x{i}", f"x{i}", 0.0, 1.0) for i in range(3)]


def test_saltelli_requires_power_of_two():
    with pytest.raises(ValueError):
        sens.saltelli_sample(PARAMS, 100)


def test_saltelli_shapes_and_bounds():
    a, b, ab = sens.saltelli_sample(PARAMS, 64)
    assert a.shape == b.shape == (64, 3)
    assert ab.shape == (3, 64, 3)
    assert a.min() >= 0.0 and a.max() <= 1.0


def test_ab_matrix_swaps_exactly_one_column():
    a, b, ab = sens.saltelli_sample(PARAMS, 32)
    for i in range(3):
        assert np.allclose(ab[i][:, i], b[:, i])
        others = [j for j in range(3) if j != i]
        assert np.allclose(ab[i][:, others], a[:, others])


def test_sobol_recovers_ishigami_ranking():
    """Ishigami is the standard analytic benchmark for Sobol estimators."""
    params = [sens.Parameter(f"x{i}", f"x{i}", -np.pi, np.pi) for i in range(3)]

    def ishigami(v):
        return np.sin(v[0]) + 7.0 * np.sin(v[1]) ** 2 + 0.1 * v[2] ** 4 * np.sin(v[0])

    res = sens.sobol_analysis(ishigami, params, n_base=2048, n_boot=50)
    # Analytic values: S1 = (0.314, 0.442, 0.0); ST(x3) > 0 through interaction.
    assert res.s1[1] > res.s1[0] > res.s1[2]
    assert res.s1[1] == pytest.approx(0.442, abs=0.06)
    assert res.st[2] > 0.15
    assert res.st[2] > res.s1[2]


def test_sobol_total_at_least_first_order_for_active_inputs():
    params = [sens.Parameter(f"x{i}", f"x{i}", 0.0, 1.0) for i in range(3)]
    res = sens.sobol_analysis(lambda v: v[0] + 2 * v[1] * v[2],
                              params, n_base=1024, n_boot=40)
    assert np.all(res.st >= res.s1 - 0.05)


def test_oat_tornado_orders_by_swing():
    params = [sens.Parameter("a", "a", 0.0, 1.0), sens.Parameter("b", "b", 0.0, 1.0)]
    rows = sens.oat_tornado(lambda v: 5.0 * v[0] + v[1], params, np.array([0.5, 0.5]))
    assert rows[0]["parameter"] == "a"
    assert rows[0]["swing"] > rows[1]["swing"]


def test_telemetry_schema_and_labels():
    cg = compile_graph()
    r = simulate(ctrl.PORTFOLIOS["P1_baseline"], SimulationParams(n_runs=400),
                 seed=1, cg=cg)
    df = generate_telemetry(r, TelemetryConfig(n_campaigns=25,
                                               hours_per_campaign=120))
    assert set(FEATURES).issubset(df.columns)
    assert df["label"].isin([0, 1]).all()
    assert df.groupby("campaign").size().nunique() == 1
    assert len(df) == 25 * 120


def test_telemetry_is_not_trivially_separable():
    """A single raw feature must not perfectly separate the classes."""
    cg = compile_graph()
    r = simulate(ctrl.PORTFOLIOS["P1_baseline"], SimulationParams(n_runs=600),
                 seed=2, cg=cg)
    df = generate_telemetry(r, TelemetryConfig(n_campaigns=40,
                                               hours_per_campaign=200))
    pos, neg = df[df.label == 1], df[df.label == 0]
    for f in BASE_FEATURES:
        assert pos[f].min() < neg[f].max(), f"{f} separates the classes perfectly"


def test_telemetry_has_both_classes_and_realistic_imbalance():
    cg = compile_graph()
    r = simulate(ctrl.PORTFOLIOS["P1_baseline"], SimulationParams(n_runs=600),
                 seed=3, cg=cg)
    df = generate_telemetry(r, TelemetryConfig(n_campaigns=40,
                                               hours_per_campaign=240))
    rate = df["label"].mean()
    assert 0.005 < rate < 0.45


def test_temporal_features_are_backward_looking():
    """Shuffling the future must not change a backward-looking window."""
    cg = compile_graph()
    r = simulate(ctrl.PORTFOLIOS["P1_baseline"], SimulationParams(n_runs=300),
                 seed=4, cg=cg)
    df = generate_telemetry(r, TelemetryConfig(n_campaigns=6,
                                               hours_per_campaign=100))
    from hermes.detection import add_temporal_features
    one = df[df.campaign == 0].copy()
    truncated = add_temporal_features(one.iloc[:50][list(BASE_FEATURES) +
                                                    ["hour", "campaign", "label"]])
    assert np.allclose(truncated["flows_roll6"].to_numpy(),
                       one["flows_roll6"].to_numpy()[:50])


def test_feature_groups_partition_the_feature_set():
    flat = [f for feats in FEATURE_GROUPS.values() for f in feats]
    assert sorted(flat) == sorted(FEATURES)
    assert len(flat) == len(set(flat))
