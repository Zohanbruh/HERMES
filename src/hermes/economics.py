"""Loss aggregation and investment analysis.

The simulator produces a per-campaign loss distribution.  This module turns that
into the quantities a hospital board actually decides on: annualised loss
expectancy, a loss exceedance curve, return on security investment, and a
budget-constrained portfolio.

The formulation follows the Factor Analysis of Information Risk convention of
separating loss *frequency* from loss *magnitude*, but takes the magnitude
distribution from the attack-graph simulation rather than from an elicited
distribution, so that magnitude responds to the control portfolio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import controls as ctrl


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def annualised_loss(loss: np.ndarray, rate: float) -> float:
    """Annualised loss expectancy: campaign frequency times mean campaign loss."""
    return float(np.mean(loss) * rate)


def aggregate_annual_loss(
    loss: np.ndarray, rate: float, n_years: int = 50_000, seed: int = 99
) -> np.ndarray:
    """Simulate the *annual* aggregate loss by compounding a Poisson count.

    A single campaign's loss distribution understates the tail a board cares
    about, because a bad year can contain more than one campaign.  Drawing
    :math:`N \\sim \\mathrm{Poisson}(\\lambda)` campaigns per year and summing
    :math:`N` independent campaign losses gives the aggregate distribution used
    for the loss exceedance curve.
    """
    rng = np.random.default_rng(seed)
    counts = rng.poisson(rate, size=n_years)
    loss = np.asarray(loss, dtype=float)
    totals = np.zeros(n_years, dtype=float)
    hot = counts > 0
    if hot.any():
        max_k = int(counts.max())
        draws = rng.choice(loss, size=(n_years, max_k), replace=True)
        mask = np.arange(max_k)[None, :] < counts[:, None]
        totals = (draws * mask).sum(axis=1)
    return totals


def loss_exceedance_curve(
    annual_losses: np.ndarray, grid: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(threshold, P(annual loss > threshold))``."""
    annual_losses = np.asarray(annual_losses, dtype=float)
    if grid is None:
        top = max(float(np.quantile(annual_losses, 0.9995)), 1.0)
        grid = np.geomspace(1e4, top, 200)
    prob = np.array([(annual_losses > g).mean() for g in grid])
    return grid, prob


def value_at_risk(annual_losses: np.ndarray, level: float = 0.95) -> float:
    return float(np.quantile(annual_losses, level))


def tail_value_at_risk(annual_losses: np.ndarray, level: float = 0.95) -> float:
    """Mean loss conditional on exceeding the value at risk."""
    var = value_at_risk(annual_losses, level)
    tail = annual_losses[annual_losses >= var]
    return float(tail.mean()) if tail.size else var


# --------------------------------------------------------------------------- #
# Investment analysis
# --------------------------------------------------------------------------- #

@dataclass
class InvestmentResult:
    name: str
    annual_cost: float
    ale: float
    ale_avoided: float
    net_benefit: float
    rosi: float
    incremental_cost: float
    incremental_benefit: float


def rosi(
    ale_reference: float, ale_portfolio: float, cost_portfolio: float,
    cost_reference: float = 0.0
) -> float:
    """Return on security investment relative to a reference posture.

    .. math::
       \\mathrm{ROSI} =
       \\frac{(\\mathrm{ALE}_{\\text{ref}} - \\mathrm{ALE}_{p}) - \\Delta C}{\\Delta C}

    Returns ``nan`` when the incremental spend is zero, because the ratio is
    undefined there and reporting a large finite number would be misleading.
    """
    delta_cost = cost_portfolio - cost_reference
    if delta_cost <= 0.0:
        return float("nan")
    avoided = ale_reference - ale_portfolio
    return float((avoided - delta_cost) / delta_cost)


def investment_table(
    ales: Mapping[str, float],
    portfolios: Mapping[str, Mapping[str, float]],
    reference: str,
) -> List[InvestmentResult]:
    """Assemble a cost/benefit row per portfolio against a common reference."""
    ref_ale = float(ales[reference])
    ref_cost = ctrl.annual_cost(ctrl.portfolio_vector(portfolios[reference]))
    rows: List[InvestmentResult] = []
    for name, portfolio in portfolios.items():
        cost = ctrl.annual_cost(ctrl.portfolio_vector(portfolio))
        ale = float(ales[name])
        avoided = ref_ale - ale
        rows.append(
            InvestmentResult(
                name=name,
                annual_cost=cost,
                ale=ale,
                ale_avoided=avoided,
                net_benefit=avoided - (cost - ref_cost),
                rosi=rosi(ref_ale, ale, cost, ref_cost),
                incremental_cost=cost - ref_cost,
                incremental_benefit=avoided,
            )
        )
    return rows


# --------------------------------------------------------------------------- #
# Budget-constrained portfolio search
# --------------------------------------------------------------------------- #

def greedy_frontier(
    evaluate: Callable[[Dict[str, float]], float],
    budgets: Sequence[float],
    step: float = 0.05,
    start: Optional[Mapping[str, float]] = None,
) -> List[Dict[str, object]]:
    """Greedy marginal-benefit allocation of a security budget.

    At each iteration the control whose next ``step`` of coverage buys the
    largest reduction in expected loss per unit of additional spend is advanced.
    Greedy allocation is not guaranteed optimal for a non-submodular objective,
    so the resulting curve is reported as an achievable frontier rather than as
    the optimum; a full enumeration over the same grid is provided in
    ``scripts/06_frontier.py`` for the coarse grid, and agrees closely.

    Parameters
    ----------
    evaluate
        Maps a portfolio to expected campaign loss.  Should be deterministic for
        a fixed portfolio, which is achieved by fixing the simulator seed.
    budgets
        Increasing budget levels at which the current portfolio is recorded.
    step
        Coverage increment considered at each greedy iteration.
    """
    keys = list(ctrl.CONTROL_KEYS)
    x = dict(start) if start else {k: 0.0 for k in keys}
    x = {k: float(x.get(k, 0.0)) for k in keys}

    records: List[Dict[str, object]] = []
    budgets = sorted(budgets)
    b_i = 0
    current_cost = ctrl.annual_cost(ctrl.portfolio_vector(x))
    current_loss = evaluate(x)

    while b_i < len(budgets):
        best_key, best_gain, best_cost, best_loss = None, -np.inf, None, None
        for k in keys:
            if x[k] >= 1.0 - 1e-9:
                continue
            trial = dict(x)
            trial[k] = min(1.0, trial[k] + step)
            cost = ctrl.annual_cost(ctrl.portfolio_vector(trial))
            if cost > budgets[-1]:
                continue
            loss = evaluate(trial)
            d_cost = cost - current_cost
            if d_cost <= 0:
                continue
            gain = (current_loss - loss) / d_cost
            if gain > best_gain:
                best_key, best_gain = k, gain
                best_cost, best_loss = cost, loss
        if best_key is None:
            break
        x[best_key] = min(1.0, x[best_key] + step)
        current_cost, current_loss = float(best_cost), float(best_loss)
        while b_i < len(budgets) and current_cost >= budgets[b_i]:
            records.append(
                {"budget": budgets[b_i], "cost": current_cost,
                 "loss": current_loss, "portfolio": dict(x)}
            )
            b_i += 1
    return records
