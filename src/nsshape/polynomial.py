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
        nxt = max(1.0, _interval_max(coeffs, B))
        if not np.isfinite(nxt) or nxt > escape:
            return np.inf
        if abs(nxt - B) < 1e-12:
            return max(B, nxt)
        B = nxt
    return np.inf


def _interval_max(coeffs: Coeffs, B: float) -> float:
    """Analytic maximum using every real derivative root and both endpoints."""
    a, b, c = coeffs
    roots = np.roots([5*c, 3*b, a]) if b != 0 or c != 0 else []
    xs = [0., B]
    xs.extend(float(np.sqrt(r.real)) for r in roots
              if abs(r.imag) < 1e-12 and 0 < r.real < B*B)
    return float(evaluate(coeffs, xs).max())


def invariance_violation(coeffs: Coeffs, B: float, n: int = 2001) -> float:
    """max_[0,B] P - B. Non-positive means [0, B] is forward-invariant."""
    return _interval_max(coeffs, B) - B


def positivity_violation(coeffs: Coeffs, B: float, n: int = 2001) -> float:
    """Negative minimum of P(x)/x on [0,B]; strict negativity proves positivity.

    Unlike sampling P near zero, this margin has a well-defined positive lower
    bound. The unused n argument is retained for compatibility.
    """
    a, b, c = coeffs
    us = [0., B*B]
    if c != 0 and 0 < -b/(2*c) < B*B:
        us.append(-b/(2*c))
    return -float(min(a+b*u+c*u*u for u in us))


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
    """Largest grid-aligned range satisfying an analytic image-interval check.

    Continuity means the image of an interval is an interval. Propagating its
    endpoint/critical-point extrema through each composition checks the entire
    continuous domain, without sampling away narrow oscillations. Threshold
    resolution remains limited by the input grid; arithmetic is float64.
    """
    if grid is None:
        grid = DEFAULT_SIGMA_GRID
    grid = np.asarray(grid)
    if not 0 <= gamma < 1 or n_steps < 1 or len(grid) < 2 or not np.all(np.diff(grid) > 0) or grid[0] <= 0 or grid[-1] != 1:
        raise ValueError("invalid gamma, steps, or sigma grid")
    endpoint = compose(coeffs, 1., n_steps)
    if endpoint <= 0 or endpoint >= DIVERGENCE_BOUND:
        return BandExtent(1.0, float("nan"), float("nan"))
    left, right = 0, len(grid)-1
    while left < right:
        mid = (left+right)//2
        lo, hi = composed_bounds(coeffs, float(grid[mid]), 1., n_steps)
        feasible = np.isfinite(lo+hi) and lo > 0 and (hi-lo)/(hi+lo) <= gamma
        if feasible:
            right = mid
        else:
            left = mid+1
    lo, hi = composed_bounds(coeffs, float(grid[left]), 1., n_steps)
    return BandExtent(float(grid[left]), (lo+hi)/2, (hi-lo)/(hi+lo))


def composed_bounds(coeffs: Coeffs, lower: float, upper: float, n_steps: int = 5) -> tuple[float, float]:
    """Continuous scalar image bounds from analytic extrema, in float64."""
    a, b, c = coeffs
    roots = np.roots([5*c, 3*b, a]) if b != 0 or c != 0 else []
    critical = [float(np.sqrt(r.real)) for r in roots if abs(r.imag) < 1e-12 and r.real > 0]
    critical += [-x for x in critical.copy()]
    def value(x):
        return x*(a+x*x*(b+c*x*x))
    for _ in range(n_steps):
        points = [lower, upper]+[x for x in critical if lower < x < upper]
        values = [value(x) for x in points]
        lower, upper = min(values), max(values)
        if not np.isfinite(lower+upper) or max(abs(lower), abs(upper)) > DIVERGENCE_BOUND:
            return float('-inf'), float('inf')
    return lower, upper
