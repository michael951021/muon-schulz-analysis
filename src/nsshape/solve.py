"""Constrained fit of the quintic family: gamma -> coefficients + certificate.

Stochastic and slow by design; runs offline via scripts/generate_coefficients.py.
Training code never imports this module -- it reads the committed table instead.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import differential_evolution

from nsshape.polynomial import (
    Coeffs,
    band_extent,
    first_critical_point,
    invariance_violation,
    positivity_violation,
    reachable_bound,
)

#: Scores at or above this are infeasible. Feasible scores are log10(sigma_min),
#: which is always negative.
PENALTY = 100.0

#: Search box for (a, b, c).
BOUNDS = [(0.5, 9.0), (-20.0, 0.0), (0.0, 15.0)]


@dataclass(frozen=True)
class FitResult:
    gamma: float
    n_steps: int
    coeffs: Coeffs
    sigma_min: float
    scale: float
    achieved_half_width: float
    critical_point: float
    invariance_bound: float
    invariance_margin: float
    positivity_margin: float
    feasible: bool
    seed: int

    def to_dict(self) -> dict:
        d = asdict(self)
        d["coeffs"] = list(self.coeffs)
        return d


def objective(params, gamma: float, n_steps: int = 5) -> float:
    """Minimized quantity: log10(sigma_min) when feasible, >= PENALTY otherwise.

    Constraint order matters -- invariance is checked before the band, because
    without it the solver escapes to a saturating degenerate optimum that
    satisfies any band constraint while destroying the matrix.
    """
    coeffs = (float(params[0]), float(params[1]), float(params[2]))
    if coeffs[0] <= 0:
        return PENALTY

    B = reachable_bound(coeffs)
    if not np.isfinite(B):
        return PENALTY + 1.0

    pos = positivity_violation(coeffs, B)
    if pos > 0:
        return PENALTY + pos

    extent = band_extent(coeffs, gamma, n_steps)
    if extent.sigma_min >= 1.0:
        return PENALTY / 2.0
    return float(np.log10(extent.sigma_min))


def _certify(coeffs: Coeffs, gamma: float, n_steps: int, seed: int) -> FitResult:
    """Build a FitResult by re-measuring every property from scratch."""
    B = reachable_bound(coeffs)
    extent = band_extent(coeffs, gamma, n_steps)
    bounded = bool(np.isfinite(B))
    inv_margin = -invariance_violation(coeffs, B) if bounded else float("-inf")
    pos_margin = -positivity_violation(coeffs, B) if bounded else float("-inf")
    feasible = bool(
        bounded
        and extent.sigma_min < 1.0
        and pos_margin >= 0
        and np.isfinite(extent.scale)
    )
    return FitResult(
        gamma=gamma,
        n_steps=n_steps,
        coeffs=coeffs,
        sigma_min=extent.sigma_min,
        scale=extent.scale,
        achieved_half_width=extent.achieved_half_width,
        critical_point=first_critical_point(coeffs),
        invariance_bound=B,
        invariance_margin=inv_margin,
        positivity_margin=pos_margin,
        feasible=feasible,
        seed=seed,
    )


def fit_gamma(
    gamma: float,
    n_steps: int = 5,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    maxiter: int = 400,
    popsize: int = 40,
) -> FitResult:
    """Fit coefficients for one gamma, restarting across seeds.

    Multi-seed restart is required, not optional: a single differential-evolution
    run converged to an infeasible local optimum (a=5.78, b=-11.99) at
    gamma=0.50 while succeeding at 0.40 and 0.60 from identical settings.
    """
    best: FitResult | None = None
    for seed in seeds:
        result = differential_evolution(
            objective,
            BOUNDS,
            args=(gamma, n_steps),
            seed=seed,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-12,
            polish=True,
        )
        candidate = _certify(
            (float(result.x[0]), float(result.x[1]), float(result.x[2])),
            gamma,
            n_steps,
            seed,
        )
        if not candidate.feasible:
            continue
        if best is None or candidate.sigma_min < best.sigma_min:
            best = candidate

    if best is not None:
        return best

    nan = float("nan")
    return FitResult(
        gamma=gamma,
        n_steps=n_steps,
        coeffs=(nan, nan, nan),
        sigma_min=1.0,
        scale=nan,
        achieved_half_width=nan,
        critical_point=nan,
        invariance_bound=nan,
        invariance_margin=nan,
        positivity_margin=nan,
        feasible=False,
        seed=seeds[0],
    )
