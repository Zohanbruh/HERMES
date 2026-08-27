"""Security-control model.

A control portfolio is a vector :math:`x \\in [0,1]^{|C|}` where each component is
the *coverage* of one control -- the fraction of the relevant estate on which the
control is correctly deployed and enforced.  Coverage, rather than a binary
on/off flag, is used because every publicly documented healthcare incident that
this work calibrates against involved a control that was present but incomplete
(for example, multi-factor authentication that was deployed but not enforced on
one internet-facing portal).

Two effects are modelled.

*Prevention.*  A control reduces the per-attempt success probability of the
adversary steps it covers.  Effects compose multiplicatively, which is the
standard defence-in-depth assumption and, importantly, never drives a
probability below zero or above one:

.. math::
   p_{ij}(x) = p_{ij}^{0}\\prod_{c \\in C}\\bigl(1 - \\eta_{c,ij}\\, x_c\\bigr)

*Detection and recovery.*  Monitoring controls raise the per-hour hazard that an
active intrusion is observed; backup controls shorten restoration.  These are
handled by :func:`detection_multiplier` and :func:`recovery_multiplier` and are
consumed by the simulator rather than by the propagation kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from .topology import EDGE_KINDS

# --------------------------------------------------------------------------- #
# Control catalogue
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Control:
    key: str
    label: str
    #: Annual cost at full (100 %) coverage, in the model currency.
    annual_cost_full: float
    #: Reference for the efficacy prior; see docs/CALIBRATION.md.
    rationale: str


CONTROLS: Tuple[Control, ...] = (
    Control("mfa", "Phishing-resistant MFA", 210_000,
            "Blocks reuse of harvested credentials on remote and privileged paths"),
    Control("seg", "Network segmentation", 480_000,
            "Removes flat-network east-west paths between clinical VLANs"),
    Control("patch", "Vulnerability and patch management", 320_000,
            "Shrinks the window in which a known CVE is exploitable"),
    Control("edr", "Endpoint detection and response", 260_000,
            "Interrupts payload execution and raises endpoint telemetry"),
    Control("awareness", "Security culture and awareness programme", 95_000,
            "Lowers phishing click-through and speeds user reporting"),
    Control("backup", "Immutable / offline backup", 175_000,
            "Preserves a clean restore point when the estate is encrypted"),
    Control("siem", "Centralised monitoring and response (SIEM/SOC)", 410_000,
            "Raises detection hazard and shortens containment latency"),
)

CONTROL_KEYS: Tuple[str, ...] = tuple(c.key for c in CONTROLS)
N_CONTROLS: int = len(CONTROL_KEYS)

# --------------------------------------------------------------------------- #
# Prevention efficacy matrix  eta[control, edge_kind]
# --------------------------------------------------------------------------- #

#: Fractional reduction in per-attempt success probability at full coverage.
#: Values are deliberately conservative -- no control exceeds 0.80 -- because
#: the empirical literature consistently reports residual risk even for
#: well-implemented controls.
ETA: Dict[str, Dict[str, float]] = {
    "mfa": {
        "phishing": 0.35, "ext_exploit": 0.00, "cred_abuse": 0.78,
        "supply_chain": 0.30, "lateral": 0.15, "priv_esc": 0.25,
        "backup_tamper": 0.20,
    },
    "seg": {
        "phishing": 0.00, "ext_exploit": 0.10, "cred_abuse": 0.20,
        "supply_chain": 0.45, "lateral": 0.72, "priv_esc": 0.10,
        "backup_tamper": 0.55,
    },
    "patch": {
        "phishing": 0.05, "ext_exploit": 0.70, "cred_abuse": 0.05,
        "supply_chain": 0.15, "lateral": 0.35, "priv_esc": 0.45,
        "backup_tamper": 0.10,
    },
    "edr": {
        "phishing": 0.40, "ext_exploit": 0.25, "cred_abuse": 0.15,
        "supply_chain": 0.15, "lateral": 0.38, "priv_esc": 0.50,
        "backup_tamper": 0.30,
    },
    "awareness": {
        "phishing": 0.55, "ext_exploit": 0.00, "cred_abuse": 0.18,
        "supply_chain": 0.05, "lateral": 0.03, "priv_esc": 0.03,
        "backup_tamper": 0.00,
    },
    "backup": {
        "phishing": 0.00, "ext_exploit": 0.00, "cred_abuse": 0.00,
        "supply_chain": 0.00, "lateral": 0.00, "priv_esc": 0.00,
        "backup_tamper": 0.65,
    },
    "siem": {
        "phishing": 0.05, "ext_exploit": 0.10, "cred_abuse": 0.12,
        "supply_chain": 0.10, "lateral": 0.20, "priv_esc": 0.18,
        "backup_tamper": 0.25,
    },
}

#: Multiplier on the per-hour detection hazard at full coverage.
DETECTION_GAIN: Dict[str, float] = {
    "mfa": 1.10, "seg": 1.25, "patch": 1.05, "edr": 3.20,
    "awareness": 1.60, "backup": 1.00, "siem": 4.50,
}

#: Fractional reduction in containment latency at full coverage.
CONTAINMENT_GAIN: Dict[str, float] = {
    "mfa": 0.05, "seg": 0.30, "patch": 0.05, "edr": 0.25,
    "awareness": 0.15, "backup": 0.05, "siem": 0.45,
}

#: Fractional reduction in restoration time at full coverage.
RECOVERY_GAIN: Dict[str, float] = {
    "mfa": 0.00, "seg": 0.10, "patch": 0.05, "edr": 0.05,
    "awareness": 0.05, "backup": 0.55, "siem": 0.10,
}


# --------------------------------------------------------------------------- #
# Named portfolios used in the experiments
# --------------------------------------------------------------------------- #

def _p(**kw: float) -> Dict[str, float]:
    base = {k: 0.0 for k in CONTROL_KEYS}
    base.update(kw)
    unknown = set(kw) - set(CONTROL_KEYS)
    if unknown:
        raise KeyError(f"unknown control(s): {sorted(unknown)}")
    return base


#: ``P0`` reflects an estate with essentially no dedicated security programme.
#: ``P1`` is the modal posture reported by healthcare providers in public
#: maturity surveys.  ``P2`` adds the controls most often named in post-incident
#: reporting.  ``P3`` is a zero-trust-aligned target state.  ``P4`` is an
#: unattainable upper bound used to bracket the achievable risk reduction.
PORTFOLIOS: Dict[str, Dict[str, float]] = {
    "P0_none": _p(),
    "P1_baseline": _p(mfa=0.35, seg=0.20, patch=0.45, edr=0.40,
                      awareness=0.30, backup=0.50, siem=0.25),
    "P2_hardened": _p(mfa=0.75, seg=0.55, patch=0.70, edr=0.75,
                      awareness=0.65, backup=0.85, siem=0.70),
    "P3_zero_trust": _p(mfa=0.95, seg=0.90, patch=0.85, edr=0.90,
                        awareness=0.80, backup=0.95, siem=0.90),
    "P4_upper_bound": _p(**{k: 1.0 for k in CONTROL_KEYS}),
}


# --------------------------------------------------------------------------- #
# Vector helpers
# --------------------------------------------------------------------------- #

def portfolio_vector(portfolio: Mapping[str, float]) -> np.ndarray:
    """Convert a ``{control: coverage}`` mapping to an ordered array."""
    missing = set(CONTROL_KEYS) - set(portfolio)
    if missing:
        raise KeyError(f"portfolio missing control(s): {sorted(missing)}")
    vec = np.array([float(portfolio[k]) for k in CONTROL_KEYS], dtype=float)
    if np.any(vec < 0.0) or np.any(vec > 1.0):
        raise ValueError("control coverage must lie in [0, 1]")
    return vec


def eta_matrix() -> np.ndarray:
    """Return ``eta`` as an array of shape ``(n_controls, n_edge_kinds)``."""
    return np.array(
        [[ETA[c][k] for k in EDGE_KINDS] for c in CONTROL_KEYS], dtype=float
    )


def prevention_multiplier(x: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Per-edge-kind multiplier ``prod_c (1 - eta[c,k] * x[c])``.

    Parameters
    ----------
    x
        Coverage vector of shape ``(n_controls,)``.
    eta
        Efficacy matrix of shape ``(n_controls, n_edge_kinds)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_edge_kinds,)``, every entry in ``(0, 1]``.
    """
    return np.prod(1.0 - eta * x[:, None], axis=0)


def detection_multiplier(x: np.ndarray) -> float:
    """Multiplicative gain applied to the per-hour detection hazard."""
    gain = 1.0
    for i, key in enumerate(CONTROL_KEYS):
        gain *= 1.0 + (DETECTION_GAIN[key] - 1.0) * float(x[i])
    return float(gain)


def containment_multiplier(x: np.ndarray) -> float:
    """Multiplier on containment latency; smaller is better."""
    mult = 1.0
    for i, key in enumerate(CONTROL_KEYS):
        mult *= 1.0 - CONTAINMENT_GAIN[key] * float(x[i])
    return float(mult)


def recovery_multiplier(x: np.ndarray) -> float:
    """Multiplier on restoration time; smaller is better."""
    mult = 1.0
    for i, key in enumerate(CONTROL_KEYS):
        mult *= 1.0 - RECOVERY_GAIN[key] * float(x[i])
    return float(mult)


def annual_cost(x: np.ndarray) -> float:
    """Annual programme cost of a portfolio.

    Cost is superlinear in coverage (exponent 1.35) because the final
    percentage points of coverage in a hospital estate -- legacy modalities,
    unmanaged biomedical devices, clinical workflow exceptions -- are
    disproportionately expensive.
    """
    total = 0.0
    for i, control in enumerate(CONTROLS):
        total += control.annual_cost_full * float(x[i]) ** 1.35
    return float(total)


def leave_one_out(portfolio: Mapping[str, float]) -> Dict[str, Dict[str, float]]:
    """Ablation helper: portfolio with each control zeroed in turn."""
    out: Dict[str, Dict[str, float]] = {}
    for key in CONTROL_KEYS:
        variant = dict(portfolio)
        variant[key] = 0.0
        out[key] = variant
    return out


def only_one(portfolio: Mapping[str, float]) -> Dict[str, Dict[str, float]]:
    """Ablation helper: only one control active at its portfolio coverage."""
    out: Dict[str, Dict[str, float]] = {}
    for key in CONTROL_KEYS:
        variant = {k: 0.0 for k in CONTROL_KEYS}
        variant[key] = float(portfolio[key])
        out[key] = variant
    return out


def validate_efficacy() -> None:
    """Guard against typos in the hand-authored efficacy tables."""
    for control in CONTROL_KEYS:
        if set(ETA[control]) != set(EDGE_KINDS):
            raise ValueError(f"eta row {control} does not cover all edge kinds")
        for kind, value in ETA[control].items():
            if not 0.0 <= value <= 0.80:
                raise ValueError(
                    f"eta[{control}][{kind}]={value} outside the admissible [0, 0.80]"
                )
    for table, name in ((DETECTION_GAIN, "DETECTION_GAIN"),
                        (CONTAINMENT_GAIN, "CONTAINMENT_GAIN"),
                        (RECOVERY_GAIN, "RECOVERY_GAIN")):
        if set(table) != set(CONTROL_KEYS):
            raise ValueError(f"{name} does not cover all controls")


def describe(portfolio: Mapping[str, float]) -> str:
    """Compact human-readable rendering used in logs and tables."""
    parts: Sequence[str] = [f"{k}={portfolio[k]:.2f}" for k in CONTROL_KEYS]
    return " ".join(parts)
