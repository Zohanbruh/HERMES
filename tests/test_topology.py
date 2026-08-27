"""Topology invariants that must hold for the model to be meaningful."""
import networkx as nx
import pytest

from hermes.topology import (CLINICAL_ZONES, EDGE_KINDS, EDGES, ENTRY_NODE,
                             PHI_ZONES, ZONES, build_graph, edge_kind_counts,
                             shortest_paths_to, validate_graph)


def test_graph_validates():
    validate_graph(build_graph())


def test_entry_has_no_inbound_edges():
    g = build_graph().graph
    assert g.in_degree(ENTRY_NODE) == 0


def test_every_zone_reachable_from_entry():
    g = build_graph().graph
    reachable = nx.descendants(g, ENTRY_NODE) | {ENTRY_NODE}
    assert set(g.nodes) == reachable


def test_edge_probabilities_in_range():
    for e in EDGES:
        assert 0.0 < e.p_base <= 1.0
        assert e.tau > 0.0
        assert e.kind in EDGE_KINDS


def test_edge_kind_counts_sum_to_edge_count():
    hg = build_graph()
    assert sum(edge_kind_counts(hg).values()) == hg.graph.number_of_edges()


def test_declared_zone_groups_exist():
    for z in CLINICAL_ZONES + PHI_ZONES:
        assert z in ZONES


def test_phi_zones_have_records():
    for z in PHI_ZONES:
        assert float(ZONES[z]["phi_records"]) > 0


def test_ehr_is_reachable_by_multiple_paths():
    paths = shortest_paths_to(build_graph(), "ehr_core")
    assert len(paths) > 3, "a single-path target would make the model degenerate"


def test_malformed_topology_is_rejected():
    hg = build_graph()
    hg.graph.add_edge("ehr_core", ENTRY_NODE, p_base=0.5, tau=1.0,
                      technique="TX", kind="lateral", note="")
    with pytest.raises(ValueError):
        validate_graph(hg)
