"""Odd quintic Newton-Schulz polynomials: evaluation, composition, analysis.

Pure numpy, no torch. This module is the reference implementation that the
optimizer backends and the committed coefficient table are checked against.

The polynomial is  P(x) = a*x + b*x^3 + c*x^5,  applied N times to the singular
values of a Frobenius-normalized matrix.
"""

from __future__ import annotations

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
