"""Odd quintic Newton-Schulz polynomials: evaluation, composition, analysis.

Pure numpy, no torch. This module is the reference implementation that the
optimizer backends and the committed coefficient table are checked against.

The polynomial is  P(x) = a*x + b*x^3 + c*x^5,  applied N times to the singular
values of a Frobenius-normalized matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Coeffs = tuple[float, float, float]

#: Keller Jordan's hand-tuned quintic, the standard Muon baseline.
KELLER: Coeffs = (3.4445, -4.7750, 2.0315)

#: Iterates beyond this magnitude are treated as saturated rather than infinite,
#: so that infeasible candidates score badly instead of raising.
DIVERGENCE_BOUND = 1e6

#: Log-spaced grid of input singular values used for all band measurements.
DEFAULT_SIGMA_GRID = np.logspace(-7, 0, 3000)


def evaluate(coeffs: Coeffs, x) -> np.ndarray:
    """P(x) = a*x + b*x^3 + c*x^5."""
    a, b, c = coeffs
    x = np.asarray(x, dtype=np.float64)
    x2 = x * x
    return a * x + b * x2 * x + c * x2 * x2 * x


def derivative(coeffs: Coeffs, x) -> np.ndarray:
    """P'(x) = a + 3b*x^2 + 5c*x^4."""
    a, b, c = coeffs
    x = np.asarray(x, dtype=np.float64)
    x2 = x * x
    return a + 3.0 * b * x2 + 5.0 * c * x2 * x2


def compose(coeffs: Coeffs, x, n_steps: int = 5) -> np.ndarray:
    """Apply P n_steps times.

    Divergent candidates saturate at +/-DIVERGENCE_BOUND rather than producing
    inf/nan, so the solver can score them without special-casing.
    """
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(n_steps):
            x = evaluate(coeffs, x)
            x = np.nan_to_num(
                x,
                nan=DIVERGENCE_BOUND,
                posinf=DIVERGENCE_BOUND,
                neginf=-DIVERGENCE_BOUND,
            )
            x = np.clip(x, -DIVERGENCE_BOUND, DIVERGENCE_BOUND)
    return x


def first_critical_point(coeffs: Coeffs) -> float:
    """Smallest positive root of P', or inf if P' has no positive root.

    P'(x) = a + 3b x^2 + 5c x^4 is a quadratic in u = x^2.

    Reported as a diagnostic only. It is NOT a constraint: Keller's quintic has
    x* = 0.5545 and the whole useful family is non-monotone on [0, 1], which is
    precisely how large singular values get squashed back down.
    """
    a, b, c = coeffs
    if c == 0.0:
        if b == 0.0:
            return np.inf
        u = -a / (3.0 * b)
        return float(np.sqrt(u)) if u > 0 else np.inf
    disc = 9.0 * b * b - 20.0 * a * c
    if disc < 0:
        return np.inf
    root = np.sqrt(disc)
    candidates = ((-3.0 * b - root) / (10.0 * c), (-3.0 * b + root) / (10.0 * c))
    positive = [u for u in candidates if u > 0]
    return float(np.sqrt(min(positive))) if positive else np.inf


def reachable_bound(
    coeffs: Coeffs,
    n: int = 4001,
    max_iter: int = 200,
    escape: float = 1e4,
) -> float:
    """Smallest B >= 1 with [0, B] forward-invariant and reachable from [0, 1].

    Found as the fixed point of  B <- max(1, sup_[0,B] P),  started at B = 1.
    Returns inf if the iteration escapes, which is exactly the divergent case.

    This is the correct notion of boundedness, and it is independent of gamma.
    Naive alternatives fail: `sup_[0,1+gamma] P <= 1+gamma` rejects Keller, whose
    reachable set is bounded at 1.2024 even though P(1.3) = 1.53 > 1.3. Keller is
    a working optimizer, so any constraint that excludes it is the wrong one.
    """
    B = 1.0
    for _ in range(max_iter):
        xs = np.linspace(0.0, B, n)
        nxt = max(1.0, float(evaluate(coeffs, xs).max()))
        if not np.isfinite(nxt) or nxt > escape:
            return np.inf
        if abs(nxt - B) < 1e-12:
            return B
        B = nxt
    return B


def invariance_violation(coeffs: Coeffs, B: float, n: int = 2001) -> float:
    """max_[0,B] P - B. Non-positive means [0, B] is forward-invariant."""
    xs = np.linspace(0.0, B, n)
    return float(evaluate(coeffs, xs).max() - B)


def positivity_violation(coeffs: Coeffs, B: float, n: int = 2001) -> float:
    """-min_(0,B] P. Non-positive means P > 0 on (0, B].

    This is the binding shape constraint, standing in for monotonicity.
    """
    xs = np.linspace(B / n, B, n)
    return float(-evaluate(coeffs, xs).min())


@dataclass(frozen=True)
class BandExtent:
    """Result of measuring how much input range a composed map pulls into band.

    sigma_min: smallest input singular value still inside the band. 1.0 means
        the map has no usable range (failure sentinel).
    scale: midpoint of the achieved image, used as the output divisor so that
        band center is 1 and gamma is decoupled from the learning rate.
    achieved_half_width: relative half-width actually attained, <= gamma.
    """

    sigma_min: float
    scale: float
    achieved_half_width: float


def band_extent(
    coeffs: Coeffs,
    gamma: float,
    n_steps: int = 5,
    grid: np.ndarray | None = None,
) -> BandExtent:
    """Largest input range [sigma_min, 1] whose image has relhw <= gamma.

    Walks leftward from x = 1 along the grid, growing the image interval, and
    stops at the first point that would push relative half-width past gamma.
    """
    if grid is None:
        grid = DEFAULT_SIGMA_GRID
    image = compose(coeffs, grid, n_steps)
    if not np.all(np.isfinite(image)) or image[-1] <= 0:
        return BandExtent(1.0, float("nan"), float("nan"))

    i = len(grid) - 1
    lo = hi = float(image[-1])
    best = BandExtent(float(grid[-1]), (lo + hi) / 2.0, 0.0)
    while i > 0:
        value = float(image[i - 1])
        if value <= 0:
            break
        new_lo, new_hi = min(lo, value), max(hi, value)
        if (new_hi - new_lo) / (new_hi + new_lo) > gamma:
            break
        lo, hi = new_lo, new_hi
        i -= 1
        best = BandExtent(float(grid[i]), (lo + hi) / 2.0, (hi - lo) / (hi + lo))
    return best
