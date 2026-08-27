"""Monte Carlo engine for adversary propagation, detection, and impact.

The engine advances a discrete-time (one hour) stochastic process over the
reference hospital attack graph.  All ``n_runs`` replications are advanced
simultaneously with vectorised NumPy operations, which keeps a 20 000-replication
experiment inside a few seconds on a laptop.

Process
-------
1. **Propagation.**  While the intrusion is active, every edge whose source zone
   is compromised and whose destination zone is not fires with per-hour hazard
   :math:`h_{ij} = \\min(1,\\, p_{ij}(x) / \\tau_{ij})`.
2. **Detection.**  Every compromised zone emits signal.  The per-hour probability
   that the intrusion is *not* detected is
   :math:`\\prod_i (1 - g(x)\\, d_i)` over compromised zones :math:`i`, where
   :math:`g(x)` is the monitoring gain of the control portfolio.
3. **Containment.**  Detection is followed by a log-normal containment latency.
   Propagation halts when containment completes.
4. **Impact.**  A zone suffers a clinical or business outage only if the
   adversary held it for at least ``impact_delay`` hours before containment.
   This is what makes early detection valuable and is consistent with published
   incident timelines in which several compromised systems were isolated before
   encryption.
5. **Loss.**  Outage hours, exposed records, ransom, and regulatory exposure are
   converted to a monetary loss and to a care-disruption index.

Everything the engine consumes is a documented parameter in
:class:`SimulationParams`; nothing is hard-coded inside the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from . import controls as ctrl
from .topology import (
    CLINICAL_ZONES,
    EDGE_KINDS,
    ENTRY_NODE,
    HospitalGraph,
    build_graph,
    validate_graph,
)

# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SimulationParams:
    """All tunable quantities of the model.

    Defaults are the calibrated values documented in ``docs/CALIBRATION.md``.
    """

    horizon_hours: int = 720            # 30 days
    n_runs: int = 20_000

    # -- detection and response -------------------------------------------
    detect_scale: float = 1.0           # global multiplier on per-zone hazards
    containment_median_h: float = 34.0  # median hours from detection to contained
    containment_sigma: float = 0.65     # log-normal shape

    # -- impact ------------------------------------------------------------
    impact_delay_h: float = 12.0        # dwell needed in a zone before outage
    restore_sigma: float = 0.45         # log-normal noise on restoration
    backup_loss_penalty: float = 2.20   # restoration multiplier if backups hit

    # -- economics ---------------------------------------------------------
    breach_cost_coeff: float = 373.0    # cost = coeff * records ** exponent
    breach_cost_exponent: float = 0.75
    ransom_median: float = 1_450_000.0
    ransom_sigma: float = 0.90
    pay_prob_no_backup: float = 0.62    # P(pay | backups destroyed)
    pay_prob_with_backup: float = 0.11
    fine_prob: float = 0.28
    fine_per_record: float = 24.0
    fine_cap: float = 18_000_000.0

    # -- exposure ----------------------------------------------------------
    exfil_fraction_mean: float = 0.55   # share of a zone's records actually taken
    exfil_fraction_sigma: float = 0.25

    # -- frequency ---------------------------------------------------------
    annual_attempt_rate: float = 1.5    # serious targeted campaigns per year

    def with_(self, **kw) -> "SimulationParams":
        """Return a copy with selected fields replaced."""
        return replace(self, **kw)


# --------------------------------------------------------------------------- #
# Compiled topology arrays
# --------------------------------------------------------------------------- #

@dataclass
class CompiledGraph:
    """Flat NumPy views of the attack graph, built once and reused."""

    nodes: Tuple[str, ...]
    node_index: Dict[str, int]
    src: np.ndarray            # (E,) int
    dst: np.ndarray            # (E,) int
    p_base: np.ndarray         # (E,) float
    tau: np.ndarray            # (E,) float
    kind_index: np.ndarray     # (E,) int into EDGE_KINDS
    criticality: np.ndarray    # (V,) float
    phi_records: np.ndarray    # (V,) float
    downtime_cost: np.ndarray  # (V,) float
    restore_hours: np.ndarray  # (V,) float
    mtd_hours: np.ndarray      # (V,) float
    detect_base: np.ndarray    # (V,) float
    clinical_mask: np.ndarray  # (V,) bool
    entry: int
    backup_idx: int
    ehr_idx: int

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return int(self.src.size)


def compile_graph(hg: Optional[HospitalGraph] = None) -> CompiledGraph:
    """Flatten a :class:`HospitalGraph` into arrays for the vectorised kernel."""
    if hg is None:
        hg = build_graph()
    validate_graph(hg)

    nodes = tuple(hg.nodes)
    node_index = {n: i for i, n in enumerate(nodes)}
    kind_index_map = {k: i for i, k in enumerate(EDGE_KINDS)}

    src, dst, p_base, tau, kinds = [], [], [], [], []
    for u, v, data in hg.graph.edges(data=True):
        src.append(node_index[u])
        dst.append(node_index[v])
        p_base.append(float(data["p_base"]))
        tau.append(float(data["tau"]))
        kinds.append(kind_index_map[str(data["kind"])])

    def node_array(key: str) -> np.ndarray:
        return np.array([float(hg.graph.nodes[n][key]) for n in nodes], dtype=float)

    return CompiledGraph(
        nodes=nodes,
        node_index=node_index,
        src=np.asarray(src, dtype=np.int32),
        dst=np.asarray(dst, dtype=np.int32),
        p_base=np.asarray(p_base, dtype=float),
        tau=np.asarray(tau, dtype=float),
        kind_index=np.asarray(kinds, dtype=np.int32),
        criticality=node_array("criticality"),
        phi_records=node_array("phi_records"),
        downtime_cost=node_array("downtime_cost"),
        restore_hours=node_array("restore_hours"),
        mtd_hours=node_array("mtd_hours"),
        detect_base=node_array("detect_base"),
        clinical_mask=np.array([n in CLINICAL_ZONES for n in nodes], dtype=bool),
        entry=node_index[ENTRY_NODE],
        backup_idx=node_index["backup"],
        ehr_idx=node_index["ehr_core"],
    )


# --------------------------------------------------------------------------- #
# Results container
# --------------------------------------------------------------------------- #

@dataclass
class SimulationResult:
    """Per-replication outcome arrays plus the configuration that produced them."""

    portfolio: Dict[str, float]
    params: SimulationParams
    seed: int
    compromised: np.ndarray        # (n_runs, V) bool
    first_seen: np.ndarray         # (n_runs, V) float hours, inf if never
    detect_time: np.ndarray        # (n_runs,) float hours, inf if never detected
    contain_time: np.ndarray       # (n_runs,) float hours
    outage_hours: np.ndarray       # (n_runs, V) float
    records_exposed: np.ndarray    # (n_runs,) float
    loss: np.ndarray               # (n_runs,) float
    care_disruption: np.ndarray    # (n_runs,) float
    ehr_outage_hours: np.ndarray   # (n_runs,) float
    nodes: Tuple[str, ...] = field(default_factory=tuple)

    # ---- derived summaries ------------------------------------------------
    @property
    def n_runs(self) -> int:
        return int(self.loss.size)

    @property
    def blast_radius(self) -> np.ndarray:
        """Number of zones compromised, excluding the external entry node."""
        return self.compromised.sum(axis=1).astype(float) - 1.0

    @property
    def ehr_compromise(self) -> np.ndarray:
        idx = self.nodes.index("ehr_core")
        return self.compromised[:, idx].astype(float)

    @property
    def clinical_outage(self) -> np.ndarray:
        """Indicator that at least one clinical zone suffered an outage."""
        mask = np.array([n in CLINICAL_ZONES for n in self.nodes])
        return (self.outage_hours[:, mask] > 0).any(axis=1).astype(float)

    @property
    def foothold(self) -> np.ndarray:
        """Indicator that initial access succeeded at all."""
        return (self.blast_radius >= 1.0)

    @property
    def dwell_hours(self) -> np.ndarray:
        """Hours from initial access to detection; horizon if never detected.

        Only defined for replications in which the adversary gained a foothold;
        runs without a foothold are excluded by :meth:`summary` because a
        never-started intrusion has no meaningful dwell time and would
        otherwise inflate the statistic for well-defended portfolios.
        """
        return np.where(np.isfinite(self.detect_time), self.detect_time,
                        float(self.params.horizon_hours))

    def summary(self) -> Dict[str, float]:
        fh = self.foothold
        dwell = self.dwell_hours[fh] if fh.any() else np.array([np.nan])
        ehr_idx = self.nodes.index("ehr_core")
        ehr_hit = self.compromised[:, ehr_idx]
        ehr_cond = (self.ehr_outage_hours[ehr_hit] if ehr_hit.any()
                    else np.array([0.0]))
        return {
            "mean_loss": float(self.loss.mean()),
            "median_loss": float(np.median(self.loss)),
            "p95_loss": float(np.quantile(self.loss, 0.95)),
            "p99_loss": float(np.quantile(self.loss, 0.99)),
            "mean_blast_radius": float(self.blast_radius.mean()),
            "p_foothold": float(fh.mean()),
            "p_ehr_compromise": float(self.ehr_compromise.mean()),
            "p_clinical_outage": float(self.clinical_outage.mean()),
            "mean_dwell_h": float(np.mean(dwell)),
            "median_dwell_h": float(np.median(dwell)),
            "mean_ehr_outage_h": float(self.ehr_outage_hours.mean()),
            "cond_ehr_outage_h": float(np.mean(ehr_cond)),
            "mean_records": float(self.records_exposed.mean()),
            "mean_cdi": float(self.care_disruption.mean()),
            "ale": float(self.loss.mean() * self.params.annual_attempt_rate),
        }


# --------------------------------------------------------------------------- #
# Kernel
# --------------------------------------------------------------------------- #

def edge_success_probability(
    cg: CompiledGraph, x: np.ndarray, eta: Optional[np.ndarray] = None
) -> np.ndarray:
    """Per-edge traversal probability under portfolio ``x``.

    This is the quantity :math:`p_{ij}(x)` of the control model: the probability
    that a competent adversary can *ever* traverse the step, given unlimited
    attempts within the campaign.  It is drawn once per replication, which is
    the standard attack-graph semantics: a control either removes an adversary
    path or it does not, rather than merely slowing an inevitable traversal.
    """
    eta = ctrl.eta_matrix() if eta is None else eta
    kind_mult = ctrl.prevention_multiplier(x, eta)      # (n_kinds,)
    return np.clip(cg.p_base * kind_mult[cg.kind_index], 0.0, 1.0)


def _incoming_edges(cg: CompiledGraph) -> Dict[int, np.ndarray]:
    """Map each node index to the indices of its inbound edges."""
    out: Dict[int, np.ndarray] = {}
    for v in range(cg.n_nodes):
        idx = np.flatnonzero(cg.dst == v)
        if idx.size:
            out[v] = idx
    return out


def _arrival_times(
    cg: CompiledGraph, weights: np.ndarray, incoming: Mapping[int, np.ndarray]
) -> np.ndarray:
    """Earliest time each zone is reached, by Bellman-Ford relaxation.

    ``weights`` has shape ``(n_runs, n_edges)`` and holds ``inf`` for edges the
    adversary cannot traverse in that replication.  Because the graph has only
    14 zones, at most 13 relaxation sweeps are needed; the loop exits as soon as
    a sweep changes nothing.
    """
    n = weights.shape[0]
    t = np.full((n, cg.n_nodes), np.inf, dtype=float)
    t[:, cg.entry] = 0.0
    for _ in range(cg.n_nodes - 1):
        changed = False
        for v, eidx in incoming.items():
            cand = (t[:, cg.src[eidx]] + weights[:, eidx]).min(axis=1)
            better = cand < t[:, v]
            if better.any():
                t[better, v] = cand[better]
                changed = True
        if not changed:
            break
    return t


def _detection_time(
    arrival: np.ndarray,
    log_miss: np.ndarray,
    horizon: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Detection time under a piecewise-exponential detection hazard.

    Every compromised zone ``v`` contributes a constant per-hour log-survival
    rate ``log_miss[v] = log(1 - h_v)``.  The aggregate hazard therefore steps
    up at each arrival, so the survival function is piecewise exponential and
    can be inverted exactly instead of being approximated by hourly Bernoulli
    draws.  Removing that discretisation eliminates a bias that would otherwise
    grow with the detection gain of the control portfolio.
    """
    n, v = arrival.shape
    order = np.argsort(arrival, axis=1, kind="stable")
    t_sorted = np.take_along_axis(arrival, order, axis=1)             # (n, V)
    lm_sorted = log_miss[order]                                       # (n, V)
    lm_sorted = np.where(np.isfinite(t_sorted), lm_sorted, 0.0)

    # Rate active on the segment that starts at the k-th arrival.
    rate = np.cumsum(lm_sorted, axis=1)                               # (n, V)

    # Segment end points: next arrival, or the horizon for the final segment.
    t_start = np.where(np.isfinite(t_sorted), t_sorted, horizon)
    t_start = np.minimum(t_start, horizon)
    t_end = np.concatenate(
        [t_start[:, 1:], np.full((n, 1), horizon, dtype=float)], axis=1
    )
    t_end = np.maximum(t_end, t_start)

    seg_len = t_end - t_start
    seg_drop = rate * seg_len                                         # <= 0
    log_s_start = np.concatenate(
        [np.zeros((n, 1)), np.cumsum(seg_drop, axis=1)[:, :-1]], axis=1
    )

    target = np.log(rng.random(n))                                    # log U
    hit = log_s_start + seg_drop <= target[:, None]
    detect = np.full(n, np.inf, dtype=float)
    any_hit = hit.any(axis=1)
    if any_hit.any():
        k = np.argmax(hit, axis=1)
        rows = np.flatnonzero(any_hit)
        kk = k[rows]
        r = rate[rows, kk]
        safe = np.where(r < 0.0, r, -1.0)
        detect[rows] = t_start[rows, kk] + (
            target[rows] - log_s_start[rows, kk]
        ) / safe
        detect[rows] = np.minimum(detect[rows], horizon)
    return detect


def simulate(
    portfolio: Mapping[str, float],
    params: Optional[SimulationParams] = None,
    seed: int = 0,
    cg: Optional[CompiledGraph] = None,
) -> SimulationResult:
    """Run the Monte Carlo experiment for one control portfolio.

    Parameters
    ----------
    portfolio
        Mapping ``{control_key: coverage}`` covering every control.
    params
        Model parameters; defaults to the calibrated :class:`SimulationParams`.
    seed
        Seed for the PCG64 bit generator.  Reusing the same seed across
        portfolios implements common random numbers, which sharply reduces the
        variance of paired differences.
    cg
        Pre-compiled graph; supplying one avoids repeated compilation in sweeps.
    """
    params = params or SimulationParams()
    cg = cg or compile_graph()
    x = ctrl.portfolio_vector(portfolio)
    eta = ctrl.eta_matrix()

    rng = np.random.default_rng(seed)
    n, v, e, t_max = params.n_runs, cg.n_nodes, cg.n_edges, float(params.horizon_hours)

    p_edge = edge_success_probability(cg, x, eta)             # (E,)
    det_gain = ctrl.detection_multiplier(x) * params.detect_scale
    contain_mult = ctrl.containment_multiplier(x)
    recover_mult = ctrl.recovery_multiplier(x)

    # ---- 1. edge percolation and traversal delays -------------------------
    passable = rng.random((n, e)) < p_edge[None, :]
    delay = rng.exponential(scale=1.0, size=(n, e)) * cg.tau[None, :]
    weights = np.where(passable, delay, np.inf)

    # ---- 2. earliest arrival time per zone --------------------------------
    incoming = _incoming_edges(cg)
    first_seen = _arrival_times(cg, weights, incoming)
    first_seen[first_seen > t_max] = np.inf

    # ---- 3. detection and containment -------------------------------------
    zone_hazard = np.clip(cg.detect_base * det_gain, 0.0, 0.95)   # (V,)
    log_miss = np.log1p(-zone_hazard)                             # (V,) <= 0
    detect_time = _detection_time(first_seen, log_miss, t_max, rng)

    latency = (
        params.containment_median_h
        * contain_mult
        * np.exp(params.containment_sigma * rng.standard_normal(n))
    )
    contain_time = np.where(
        np.isfinite(detect_time), np.minimum(detect_time + latency, t_max), t_max
    )

    compromised = first_seen <= contain_time[:, None]
    compromised[:, cg.entry] = True

    # ------------------------------------------------------------------ #
    # Impact
    # ------------------------------------------------------------------ #
    dwell_in_zone = contain_time[:, None] - first_seen        # (n, V)
    dwell_in_zone = np.where(np.isfinite(dwell_in_zone), dwell_in_zone, 0.0)
    suffered = compromised & (dwell_in_zone >= params.impact_delay_h)
    suffered[:, cg.entry] = False

    backup_lost = suffered[:, cg.backup_idx]
    restore_scale = np.where(backup_lost, params.backup_loss_penalty, 1.0)

    noise = np.exp(params.restore_sigma * rng.standard_normal((n, v)))
    outage = (
        suffered
        * cg.restore_hours[None, :]
        * recover_mult
        * restore_scale[:, None]
        * noise
    )

    # ---- exposure ---------------------------------------------------------
    exfil_frac = np.clip(
        params.exfil_fraction_mean
        + params.exfil_fraction_sigma * rng.standard_normal((n, v)),
        0.0,
        1.0,
    )
    records = (compromised * cg.phi_records[None, :] * exfil_frac).sum(axis=1)

    # ---- monetary loss ----------------------------------------------------
    downtime_cost = (outage * cg.downtime_cost[None, :]).sum(axis=1)
    breach_cost = params.breach_cost_coeff * np.power(
        np.maximum(records, 0.0), params.breach_cost_exponent
    )

    pay_prob = np.where(backup_lost, params.pay_prob_no_backup,
                        params.pay_prob_with_backup)
    encrypted = suffered[:, cg.clinical_mask].any(axis=1)
    pays = (rng.random(n) < pay_prob) & encrypted
    ransom = pays * params.ransom_median * np.exp(
        params.ransom_sigma * rng.standard_normal(n)
    )

    fined = (rng.random(n) < params.fine_prob) & (records > 0)
    fine = np.minimum(records * params.fine_per_record, params.fine_cap) * fined

    loss = downtime_cost + breach_cost + ransom + fine
    cdi = (outage * cg.criticality[None, :]).sum(axis=1)

    return SimulationResult(
        portfolio=dict(portfolio),
        params=params,
        seed=seed,
        compromised=compromised,
        first_seen=first_seen,
        detect_time=detect_time,
        contain_time=contain_time,
        outage_hours=outage,
        records_exposed=records,
        loss=loss,
        care_disruption=cdi,
        ehr_outage_hours=outage[:, cg.ehr_idx],
        nodes=cg.nodes,
    )


def zone_compromise_rates(result: SimulationResult) -> Dict[str, float]:
    """Marginal probability that each zone is compromised."""
    return {
        node: float(result.compromised[:, i].mean())
        for i, node in enumerate(result.nodes)
        if node != ENTRY_NODE
    }


def zone_outage_hours(result: SimulationResult) -> Dict[str, float]:
    """Mean outage hours per zone."""
    return {
        node: float(result.outage_hours[:, i].mean())
        for i, node in enumerate(result.nodes)
        if node != ENTRY_NODE
    }
