"""Hospital reference architecture expressed as a probabilistic attack graph.

The topology in this module is a *reference model*: it is not a map of any real
hospital.  It is assembled from publicly documented healthcare network
segmentation guidance and from the publicly reported kill chains of the 2024
Change Healthcare, Ascension and Synnovis incidents (see ``docs/CALIBRATION.md``).

Every zone carries the attributes needed by the Monte Carlo engine:

``criticality``      patient-safety weight in [0, 1] used by the care-disruption index
``phi_records``      protected-health-information records held by the zone
``downtime_cost``    direct operational loss per hour of outage (currency units)
``restore_hours``    mean hours to restore the zone from clean backup
``mtd_hours``        maximum tolerable downtime, used for the resilience metric
``detect_base``      per-hour probability that an active intrusion in the zone is
                     noticed with *no* dedicated monitoring in place

Every edge carries:

``p_base``           probability that a competent adversary eventually succeeds at
                     the step, absent any mitigating control
``tau``              mean hours spent attempting the step
``technique``        MITRE ATT&CK technique identifier for traceability
``kind``             coarse label used by the control-mitigation matrix
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import networkx as nx

# --------------------------------------------------------------------------- #
# Zone definitions
# --------------------------------------------------------------------------- #

ENTRY_NODE = "internet"

#: Zone identifier -> attribute dictionary.
ZONES: Dict[str, Dict[str, float | str]] = {
    "internet": dict(
        label="Internet (adversary foothold)",
        tier="external",
        criticality=0.0,
        phi_records=0,
        downtime_cost=0.0,
        restore_hours=0.0,
        mtd_hours=0.0,
        detect_base=0.0,
    ),
    "email": dict(
        label="Corporate e-mail / user mailbox",
        tier="perimeter",
        criticality=0.05,
        phi_records=15_000,
        downtime_cost=1_200.0,
        restore_hours=12.0,
        mtd_hours=48.0,
        detect_base=0.004,
    ),
    "remote_access": dict(
        label="Remote access portal (VPN / Citrix)",
        tier="perimeter",
        criticality=0.10,
        phi_records=0,
        downtime_cost=2_500.0,
        restore_hours=16.0,
        mtd_hours=24.0,
        detect_base=0.005,
    ),
    "vendor_gateway": dict(
        label="Third-party / supplier gateway",
        tier="perimeter",
        criticality=0.15,
        phi_records=0,
        downtime_cost=3_000.0,
        restore_hours=24.0,
        mtd_hours=24.0,
        detect_base=0.003,
    ),
    "corp_workstations": dict(
        label="Administrative workstations",
        tier="endpoint",
        criticality=0.10,
        phi_records=40_000,
        downtime_cost=4_000.0,
        restore_hours=48.0,
        mtd_hours=72.0,
        detect_base=0.006,
    ),
    "identity": dict(
        label="Identity services (AD / IAM)",
        tier="core",
        criticality=0.55,
        phi_records=0,
        downtime_cost=18_000.0,
        restore_hours=96.0,
        mtd_hours=8.0,
        detect_base=0.008,
    ),
    "clinical_workstations": dict(
        label="Clinical workstations / ward PCs",
        tier="endpoint",
        criticality=0.65,
        phi_records=60_000,
        downtime_cost=15_000.0,
        restore_hours=72.0,
        mtd_hours=12.0,
        detect_base=0.005,
    ),
    "ehr_core": dict(
        label="EHR application and database",
        tier="core",
        criticality=1.00,
        phi_records=850_000,
        downtime_cost=55_000.0,
        restore_hours=240.0,
        mtd_hours=4.0,
        detect_base=0.010,
    ),
    "pacs": dict(
        label="PACS / diagnostic imaging",
        tier="clinical",
        criticality=0.80,
        phi_records=310_000,
        downtime_cost=22_000.0,
        restore_hours=132.0,
        mtd_hours=8.0,
        detect_base=0.004,
    ),
    "lab_lis": dict(
        label="Laboratory / pathology LIS",
        tier="clinical",
        criticality=0.85,
        phi_records=280_000,
        downtime_cost=26_000.0,
        restore_hours=168.0,
        mtd_hours=6.0,
        detect_base=0.004,
    ),
    "pharmacy": dict(
        label="Pharmacy and e-prescribing",
        tier="clinical",
        criticality=0.75,
        phi_records=190_000,
        downtime_cost=17_000.0,
        restore_hours=96.0,
        mtd_hours=8.0,
        detect_base=0.004,
    ),
    "iomt": dict(
        label="Connected medical devices (IoMT)",
        tier="clinical",
        criticality=0.70,
        phi_records=25_000,
        downtime_cost=12_000.0,
        restore_hours=96.0,
        mtd_hours=12.0,
        detect_base=0.002,
    ),
    "billing": dict(
        label="Billing / claims clearing-house",
        tier="business",
        criticality=0.25,
        phi_records=520_000,
        downtime_cost=30_000.0,
        restore_hours=216.0,
        mtd_hours=48.0,
        detect_base=0.005,
    ),
    "backup": dict(
        label="Backup and disaster-recovery estate",
        tier="core",
        criticality=0.45,
        phi_records=0,
        downtime_cost=9_000.0,
        restore_hours=168.0,
        mtd_hours=72.0,
        detect_base=0.006,
    ),
}

#: Zones that are part of the clinical delivery path.  Used for the
#: care-disruption index and for the "clinical outage" outcome variable.
CLINICAL_ZONES: Tuple[str, ...] = (
    "ehr_core",
    "pacs",
    "lab_lis",
    "pharmacy",
    "iomt",
    "clinical_workstations",
)

#: Zones whose compromise is treated as a reportable PHI breach.
PHI_ZONES: Tuple[str, ...] = tuple(
    z for z, a in ZONES.items() if float(a["phi_records"]) > 0
)


# --------------------------------------------------------------------------- #
# Edge definitions
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Edge:
    """A single adversary step between two zones."""

    src: str
    dst: str
    p_base: float
    tau: float
    technique: str
    kind: str
    note: str = ""

    def as_tuple(self) -> Tuple[str, str, Dict[str, object]]:
        return (
            self.src,
            self.dst,
            dict(
                p_base=self.p_base,
                tau=self.tau,
                technique=self.technique,
                kind=self.kind,
                note=self.note,
            ),
        )


#: Coarse edge categories.  The control-mitigation matrix is defined over these.
EDGE_KINDS: Tuple[str, ...] = (
    "phishing",          # user-mediated initial access
    "ext_exploit",       # exploitation of an internet-facing service
    "cred_abuse",        # use of valid / stolen credentials
    "supply_chain",      # trusted third-party connection
    "lateral",           # east-west movement inside the estate
    "priv_esc",          # privilege escalation on a host
    "backup_tamper",     # deliberate destruction of recovery capability
)

EDGES: Tuple[Edge, ...] = (
    # ---- initial access ---------------------------------------------------
    Edge("internet", "email", 0.78, 24.0, "T1566", "phishing",
         "Credential-harvesting or malicious-attachment phishing campaign"),
    Edge("internet", "remote_access", 0.62, 36.0, "T1133", "cred_abuse",
         "Valid accounts on an externally exposed remote-access portal"),
    Edge("internet", "remote_access", 0.48, 48.0, "T1190", "ext_exploit",
         "Exploitation of an unpatched edge appliance"),
    Edge("internet", "vendor_gateway", 0.44, 60.0, "T1199", "supply_chain",
         "Trusted-relationship abuse via a service provider"),
    # ---- perimeter -> endpoint -------------------------------------------
    Edge("email", "corp_workstations", 0.86, 8.0, "T1204", "phishing",
         "User execution of the delivered payload"),
    Edge("remote_access", "corp_workstations", 0.82, 10.0, "T1021", "cred_abuse", ""),
    Edge("remote_access", "clinical_workstations", 0.68, 14.0, "T1021", "cred_abuse", ""),
    Edge("vendor_gateway", "lab_lis", 0.74, 16.0, "T1199", "supply_chain",
         "Supplier link into pathology, as in the 2024 Synnovis incident"),
    Edge("vendor_gateway", "billing", 0.66, 18.0, "T1199", "supply_chain",
         "Supplier link into claims processing"),
    # ---- endpoint -> identity --------------------------------------------
    Edge("corp_workstations", "identity", 0.80, 20.0, "T1003", "priv_esc",
         "Credential dumping followed by domain escalation"),
    Edge("clinical_workstations", "identity", 0.70, 24.0, "T1003", "priv_esc", ""),
    Edge("corp_workstations", "clinical_workstations", 0.84, 12.0, "T1021", "lateral", ""),
    # ---- identity -> everything ------------------------------------------
    Edge("identity", "ehr_core", 0.90, 14.0, "T1078", "cred_abuse",
         "Domain-admin ticket reused against the EHR estate"),
    Edge("identity", "pacs", 0.86, 16.0, "T1078", "cred_abuse", ""),
    Edge("identity", "lab_lis", 0.85, 16.0, "T1078", "cred_abuse", ""),
    Edge("identity", "pharmacy", 0.83, 18.0, "T1078", "cred_abuse", ""),
    Edge("identity", "billing", 0.80, 20.0, "T1078", "cred_abuse", ""),
    Edge("identity", "backup", 0.76, 26.0, "T1490", "backup_tamper",
         "Deletion of shadow copies and backup catalogues"),
    Edge("identity", "iomt", 0.58, 30.0, "T1078", "cred_abuse", ""),
    # ---- east-west without domain compromise ------------------------------
    Edge("clinical_workstations", "ehr_core", 0.60, 26.0, "T1210", "lateral", ""),
    Edge("clinical_workstations", "pacs", 0.66, 22.0, "T1210", "lateral", ""),
    Edge("clinical_workstations", "iomt", 0.72, 20.0, "T1210", "lateral",
         "Flat ward VLAN shared with biomedical devices"),
    Edge("lab_lis", "ehr_core", 0.62, 24.0, "T1210", "lateral", ""),
    Edge("pacs", "ehr_core", 0.56, 28.0, "T1210", "lateral", ""),
    Edge("pharmacy", "ehr_core", 0.52, 28.0, "T1210", "lateral", ""),
    Edge("billing", "ehr_core", 0.46, 32.0, "T1210", "lateral", ""),
    Edge("corp_workstations", "backup", 0.50, 36.0, "T1490", "backup_tamper", ""),
    Edge("iomt", "clinical_workstations", 0.48, 30.0, "T1210", "lateral", ""),
)


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #

@dataclass
class HospitalGraph:
    """Immutable container bundling the graph with convenient index maps."""

    graph: nx.MultiDiGraph
    nodes: List[str] = field(default_factory=list)
    index: Dict[str, int] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def out_edges(self, node: str) -> Iterable[Tuple[str, str, Dict[str, object]]]:
        return self.graph.out_edges(node, data=True)

    def attr(self, node: str, key: str):
        return self.graph.nodes[node][key]


def build_graph() -> HospitalGraph:
    """Return the reference hospital attack graph.

    A :class:`networkx.MultiDiGraph` is used because two zones can be connected
    by more than one distinct adversary technique (for example, the remote
    access portal can be reached either by credential abuse or by exploiting an
    unpatched appliance).  Keeping these as separate parallel edges preserves
    the mapping to individual mitigations.
    """
    g = nx.MultiDiGraph()
    for zone, attrs in ZONES.items():
        g.add_node(zone, **attrs)
    for edge in EDGES:
        src, dst, data = edge.as_tuple()
        g.add_edge(src, dst, **data)

    nodes = list(ZONES.keys())
    return HospitalGraph(graph=g, nodes=nodes, index={n: i for i, n in enumerate(nodes)})


def validate_graph(hg: HospitalGraph) -> None:
    """Sanity checks that must hold for the model to be meaningful.

    Raises
    ------
    ValueError
        If the topology is malformed.
    """
    g = hg.graph
    if ENTRY_NODE not in g:
        raise ValueError("entry node missing from topology")
    if g.in_degree(ENTRY_NODE) != 0:
        raise ValueError("entry node must have no inbound edges")

    for _, _, data in g.edges(data=True):
        if not 0.0 < float(data["p_base"]) <= 1.0:
            raise ValueError(f"p_base out of range: {data}")
        if float(data["tau"]) <= 0.0:
            raise ValueError(f"tau must be positive: {data}")
        if data["kind"] not in EDGE_KINDS:
            raise ValueError(f"unknown edge kind: {data['kind']}")

    reachable = nx.descendants(g, ENTRY_NODE) | {ENTRY_NODE}
    unreachable = set(g.nodes) - reachable
    if unreachable:
        raise ValueError(f"unreachable zones: {sorted(unreachable)}")

    for zone in CLINICAL_ZONES + PHI_ZONES:
        if zone not in g:
            raise ValueError(f"declared zone {zone} absent from topology")


def edge_kind_counts(hg: HospitalGraph) -> Dict[str, int]:
    """Number of edges of each kind, used in the reproducibility appendix."""
    counts: Dict[str, int] = {k: 0 for k in EDGE_KINDS}
    for _, _, data in hg.graph.edges(data=True):
        counts[str(data["kind"])] += 1
    return counts


def shortest_paths_to(hg: HospitalGraph, target: str) -> List[List[str]]:
    """All simple paths from the entry node to ``target``.

    Used only for reporting: the simulator does not enumerate paths.
    """
    simple = nx.DiGraph()
    for u, v, data in hg.graph.edges(data=True):
        w = -1.0 * float(data["p_base"])
        if not simple.has_edge(u, v) or w < simple[u][v]["w"]:
            simple.add_edge(u, v, w=w)
    return [list(p) for p in nx.all_simple_paths(simple, ENTRY_NODE, target, cutoff=5)]
