"""Synthetic hospital SOC telemetry and a detection benchmark.

Motivation
----------
The propagation model treats detection as an exogenous hazard.  This module
closes that loop: it emits the hourly signals a hospital security operations
centre would actually see, and asks how well standard detectors recover the
"intrusion active" label from them.  The resulting operating points are what
justify the detection-gain parameters used by the simulator.

Honesty of the generator
------------------------
A synthetic detection benchmark is worthless if the generator makes the classes
trivially separable.  Four deliberate design choices prevent that:

1. **Low signal-to-noise.**  Attack-induced shifts are a fraction of the benign
   standard deviation, not multiples of it.
2. **Confounders.**  Benign traffic contains maintenance windows, mass software
   rollouts, seasonal admission surges, and audit sweeps that mimic several
   attack indicators at once.
3. **Realistic imbalance.**  Intrusion-active hours are a small minority of all
   hours.
4. **Label noise.**  A fraction of hours near the boundary of the intrusion are
   mislabelled, as they would be in any retrospectively labelled corpus.

Consequently the reported precision-recall figures are modest.  That is the
point: numbers in the high nineties on a self-generated corpus would indicate a
leaky generator rather than a good detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .simulator import CompiledGraph, SimulationResult

# --------------------------------------------------------------------------- #
# Feature schema
# --------------------------------------------------------------------------- #

FEATURE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "identity": (
        "failed_auth_rate",
        "privileged_logon_count",
        "new_admin_grants",
        "impossible_travel_score",
    ),
    "network": (
        "east_west_flows",
        "novel_smb_peers",
        "dns_beacon_score",
        "egress_bytes_z",
    ),
    "endpoint": (
        "new_process_entropy",
        "edr_alert_count",
        "script_exec_rate",
        "svc_install_count",
    ),
    "clinical": (
        "after_hours_ehr_access",
        "pacs_query_rate",
        "device_offline_count",
        "order_entry_latency",
    ),
    # Derived from the raw signals by :func:`add_temporal_features`.  A real SOC
    # never scores a single hour in isolation, so excluding this group would
    # understate what is achievable and would make the benchmark a straw man.
    "temporal": (
        "flows_roll6",
        "edr_roll24",
        "auth_z24",
        "composite_roll12",
    ),
}

#: Raw signals emitted by the generator, before temporal derivation.
BASE_FEATURES: Tuple[str, ...] = tuple(
    f for g, feats in FEATURE_GROUPS.items() if g != "temporal" for f in feats
)

FEATURES: Tuple[str, ...] = tuple(f for g in FEATURE_GROUPS.values() for f in g)

#: Which zones drive which feature group.  A compromise only lights up signals
#: that the compromised zone could plausibly produce.
ZONE_SIGNAL: Dict[str, Tuple[str, ...]] = {
    "email": ("identity",),
    "remote_access": ("identity", "network"),
    "vendor_gateway": ("network",),
    "corp_workstations": ("endpoint", "identity"),
    "identity": ("identity", "network"),
    "clinical_workstations": ("endpoint", "clinical"),
    "ehr_core": ("clinical", "network"),
    "pacs": ("clinical", "network"),
    "lab_lis": ("clinical", "network"),
    "pharmacy": ("clinical",),
    "iomt": ("clinical", "network"),
    "billing": ("network",),
    "backup": ("network", "endpoint"),
}


@dataclass
class TelemetryConfig:
    """Generator settings; every value is reported in the paper's appendix."""

    hours_per_campaign: int = 720
    n_campaigns: int = 400
    #: Mean multiplicative lift a single compromised zone applies to the
    #: standard deviation of the feature groups it drives.
    zone_effect: float = 0.58
    #: Extra lift during the encryption / actions-on-objective phase.
    impact_effect: float = 1.05
    #: Probability that a given benign hour belongs to a confounding event.
    confounder_rate: float = 0.045
    confounder_effect: float = 0.75
    #: Fraction of intrusion-boundary hours whose label is flipped.
    label_noise: float = 0.03
    seed: int = 4242


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #

def _diurnal(t: np.ndarray) -> np.ndarray:
    """Hospital activity has a strong daily and weekly rhythm."""
    hour = t % 24
    day = (t // 24) % 7
    daily = 0.55 * np.sin(2 * np.pi * (hour - 7) / 24.0)
    weekly = -0.30 * ((day >= 5).astype(float))
    return daily + weekly


def generate_telemetry(
    result: SimulationResult,
    cfg: Optional[TelemetryConfig] = None,
) -> pd.DataFrame:
    """Emit an hourly SOC feature table for a subset of simulated campaigns.

    Returns a long data frame with one row per (campaign, hour) and columns for
    every feature, the binary ``label``, and bookkeeping fields used for
    group-aware cross-validation.
    """
    cfg = cfg or TelemetryConfig()
    rng = np.random.default_rng(cfg.seed)

    n_campaigns = min(cfg.n_campaigns, result.n_runs)
    pick = rng.choice(result.n_runs, size=n_campaigns, replace=False)
    hours = cfg.hours_per_campaign
    t = np.arange(hours, dtype=float)

    group_of_feature = {
        f: g for g, feats in FEATURE_GROUPS.items() for f in feats
    }
    zone_names = list(result.nodes)

    frames: List[pd.DataFrame] = []
    for c_i, run in enumerate(pick):
        base = _diurnal(t)[:, None] + rng.standard_normal((hours, len(BASE_FEATURES)))

        # ---- which groups are lit, hour by hour --------------------------
        group_lift = {g: np.zeros(hours) for g in FEATURE_GROUPS}
        contain = float(result.contain_time[run])
        for z_i, zone in enumerate(zone_names):
            if zone not in ZONE_SIGNAL:
                continue
            arrive = float(result.first_seen[run, z_i])
            if not np.isfinite(arrive) or arrive > contain:
                continue
            lo, hi = int(np.floor(arrive)), int(min(hours, np.ceil(contain)))
            if hi <= lo:
                continue
            for g in ZONE_SIGNAL[zone]:
                group_lift[g][lo:hi] += cfg.zone_effect

            outage = float(result.outage_hours[run, z_i])
            if outage > 0:
                imp_lo = int(min(hours - 1, np.floor(contain)))
                imp_hi = int(min(hours, imp_lo + 24))
                for g in ZONE_SIGNAL[zone]:
                    group_lift[g][imp_lo:imp_hi] += cfg.impact_effect

        # ---- benign confounders ------------------------------------------
        conf = np.zeros(hours)
        n_conf = rng.poisson(cfg.confounder_rate * hours / 6.0)
        for _ in range(int(n_conf)):
            start = int(rng.integers(0, max(1, hours - 8)))
            width = int(rng.integers(3, 9))
            conf[start:start + width] += cfg.confounder_effect
        conf_groups = rng.choice(
            list(FEATURE_GROUPS), size=2, replace=False
        )

        # ---- assemble -----------------------------------------------------
        values = base.copy()
        for j, feat in enumerate(BASE_FEATURES):
            g = group_of_feature[feat]
            values[:, j] += group_lift[g] * rng.uniform(0.7, 1.3)
            if g in conf_groups:
                values[:, j] += conf

        # Counts are non-negative; map through a softplus and round the ones
        # that are genuinely counts so the corpus is not trivially Gaussian.
        values = np.log1p(np.exp(values))

        active = np.zeros(hours, dtype=int)
        if np.isfinite(result.detect_time[run]) or result.foothold[run]:
            starts = result.first_seen[run]
            finite = starts[np.isfinite(starts) & (starts > 0)]
            if finite.size:
                lo = int(np.floor(finite.min()))
                hi = int(min(hours, np.ceil(contain)))
                if hi > lo:
                    active[lo:hi] = 1

        # boundary label noise
        edges = np.flatnonzero(np.diff(np.concatenate([[0], active, [0]])) != 0)
        for edge in edges:
            if rng.random() < cfg.label_noise:
                lo = max(0, edge - 2)
                hi = min(hours, edge + 2)
                active[lo:hi] = 1 - active[lo:hi]

        df = pd.DataFrame(values, columns=list(BASE_FEATURES))
        df["hour"] = t.astype(int)
        df["campaign"] = c_i
        df["label"] = active
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    return add_temporal_features(out)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive rolling-window context features, computed within each campaign.

    Windows are strictly backward-looking, so no future information reaches a
    given hour.  ``min_periods=1`` keeps the first hours of each campaign
    usable rather than dropping them.
    """
    df = df.sort_values(["campaign", "hour"]).reset_index(drop=True)
    g = df.groupby("campaign", sort=False)

    df["flows_roll6"] = (
        g["east_west_flows"].rolling(6, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )
    df["edr_roll24"] = (
        g["edr_alert_count"].rolling(24, min_periods=1).sum()
        .reset_index(level=0, drop=True)
    )
    roll_mu = (g["failed_auth_rate"].rolling(24, min_periods=1).mean()
               .reset_index(level=0, drop=True))
    roll_sd = (g["failed_auth_rate"].rolling(24, min_periods=1).std()
               .reset_index(level=0, drop=True).fillna(1.0).replace(0.0, 1.0))
    df["auth_z24"] = (df["failed_auth_rate"] - roll_mu) / roll_sd

    composite = df[list(BASE_FEATURES)].mean(axis=1)
    df["composite_roll12"] = (
        composite.groupby(df["campaign"]).rolling(12, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )
    return df


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

def build_models(seed: int = 0) -> Dict[str, object]:
    """Detector zoo: one linear, two tree ensembles, one network, one novelty."""
    return {
        "LogReg": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=seed)),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=120, min_samples_leaf=12, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed,
        ),
        "HistGB": HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.10, max_leaf_nodes=20, random_state=seed,
        ),
        "MLP": Pipeline([
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(48, 24), max_iter=150,
                                  early_stopping=True, random_state=seed)),
        ]),
    }


@dataclass
class DetectionMetrics:
    model: str
    roc_auc: float
    pr_auc: float
    best_f1: float
    recall_at_fpr: float
    precision_at_recall50: float
    brier: float
    alerts_per_day: float
    n_train: int = 0
    n_test: int = 0
    fold: int = -1
    extra: Dict[str, float] = field(default_factory=dict)


def _recall_at_fpr(y: np.ndarray, s: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    return float(tpr[max(idx, 0)])


def _precision_at_recall(y: np.ndarray, s: np.ndarray, target: float) -> float:
    prec, rec, _ = precision_recall_curve(y, s)
    ok = rec >= target
    return float(prec[ok].max()) if ok.any() else 0.0


def evaluate_scores(
    name: str, y: np.ndarray, scores: np.ndarray, hours: int,
    target_fpr: float = 0.01, fold: int = -1,
) -> DetectionMetrics:
    """Convert a score vector into the operating-point metrics used in the paper."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(scores, dtype=float)
    prec, rec, _ = precision_recall_curve(y, s)
    f1s = np.divide(2 * prec * rec, prec + rec,
                    out=np.zeros_like(prec), where=(prec + rec) > 0)
    s_clip = np.clip(s, 0.0, 1.0)
    # Alert volume is reported at each model's *own* threshold for 50 % recall,
    # which is the quantity a SOC lead actually trades off.  Fixing a common
    # quantile instead would make the column identical for every model and
    # therefore uninformative.
    pos = s[y == 1]
    thr = float(np.quantile(pos, 0.50)) if pos.size else float(np.max(s))
    alerts = float((s >= thr).sum() / max(hours / 24.0, 1e-9))
    return DetectionMetrics(
        model=name,
        roc_auc=float(roc_auc_score(y, s)),
        pr_auc=float(average_precision_score(y, s)),
        best_f1=float(np.max(f1s)),
        recall_at_fpr=_recall_at_fpr(y, s, target_fpr),
        precision_at_recall50=_precision_at_recall(y, s, 0.50),
        brier=float(brier_score_loss(y, s_clip)),
        alerts_per_day=alerts,
        n_test=int(y.size),
        fold=fold,
    )


def zscore_baseline(train: pd.DataFrame, test: pd.DataFrame,
                    features: Sequence[str]) -> np.ndarray:
    """Classical SOC heuristic: mean absolute z-score across all signals."""
    mu = train[list(features)].mean()
    sd = train[list(features)].std().replace(0.0, 1.0)
    z = (test[list(features)] - mu) / sd
    raw = z.abs().mean(axis=1).to_numpy()
    return (raw - raw.min()) / max(float(np.ptp(raw)), 1e-9)


def run_benchmark(
    data: pd.DataFrame,
    features: Optional[Sequence[str]] = None,
    n_splits: int = 5,
    seed: int = 0,
    models: Optional[Sequence[str]] = None,
    include_unsupervised: bool = True,
) -> pd.DataFrame:
    """Group-aware cross-validated benchmark.

    Folds are split by *campaign*, never by hour, so that no detector ever sees
    another hour of the same intrusion during training.  Splitting by hour would
    leak the intrusion's signature across the fold boundary and inflate every
    metric.
    """
    features = list(features or FEATURES)
    x = data[features].to_numpy()
    y = data["label"].to_numpy().astype(int)
    groups = data["campaign"].to_numpy()

    rows: List[DetectionMetrics] = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (tr, te) in enumerate(splitter.split(x, y, groups)):
        if y[te].sum() == 0 or y[tr].sum() == 0:
            continue
        hours_test = int(te.size)
        zoo = build_models(seed=seed + fold)
        if models is not None:
            zoo = {k: v for k, v in zoo.items() if k in set(models)}
        for name, model in zoo.items():
            model.fit(x[tr], y[tr])
            scores = model.predict_proba(x[te])[:, 1]
            rows.append(evaluate_scores(name, y[te], scores, hours_test, fold=fold))

        if not include_unsupervised:
            continue

        iso = IsolationForest(n_estimators=150, contamination="auto",
                              random_state=seed + fold)
        iso.fit(x[tr][y[tr] == 0])
        raw = -iso.score_samples(x[te])
        raw = (raw - raw.min()) / max(np.ptp(raw), 1e-9)
        rows.append(evaluate_scores("IsolationForest", y[te], raw, hours_test,
                                    fold=fold))

        zb = zscore_baseline(data.iloc[tr], data.iloc[te], features)
        rows.append(evaluate_scores("Z-score baseline", y[te], zb, hours_test,
                                    fold=fold))

    return pd.DataFrame([r.__dict__ for r in rows]).drop(columns=["extra"])


def ablate_feature_groups(data: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Leave-one-group-out and single-group detection ablations."""
    rows: List[Dict[str, object]] = []

    def score(tag: str, feats: Sequence[str]) -> None:
        bench = run_benchmark(data, features=feats, seed=seed, n_splits=3,
                              models=("HistGB",), include_unsupervised=False)
        best = bench[bench["model"] == "HistGB"]
        rows.append({
            "setting": tag,
            "n_features": len(feats),
            "pr_auc": float(best["pr_auc"].mean()),
            "pr_auc_sd": float(best["pr_auc"].std(ddof=1)),
            "roc_auc": float(best["roc_auc"].mean()),
            "recall_at_fpr": float(best["recall_at_fpr"].mean()),
        })

    score("all", FEATURES)
    for group, feats in FEATURE_GROUPS.items():
        score(f"only:{group}", feats)
        remaining = [f for f in FEATURES if f not in feats]
        score(f"drop:{group}", remaining)
    return pd.DataFrame(rows)


def domain_shift_eval(
    train_data: pd.DataFrame, test_data: pd.DataFrame, seed: int = 0
) -> pd.DataFrame:
    """Train under one security posture, test under another.

    A detector tuned on a weakly controlled estate sees long, loud intrusions.
    The same detector deployed after hardening sees short, quiet ones.  This
    evaluation quantifies that transfer gap, which cross-validation within a
    single posture cannot expose.
    """
    feats = list(FEATURES)
    xtr, ytr = train_data[feats].to_numpy(), train_data["label"].to_numpy().astype(int)
    xte, yte = test_data[feats].to_numpy(), test_data["label"].to_numpy().astype(int)
    rows: List[DetectionMetrics] = []
    for name, model in build_models(seed=seed).items():
        model.fit(xtr, ytr)
        scores = model.predict_proba(xte)[:, 1]
        m = evaluate_scores(name, yte, scores, int(xte.shape[0]))
        m.n_train = int(xtr.shape[0])
        rows.append(m)
    return pd.DataFrame([r.__dict__ for r in rows]).drop(columns=["extra"])
