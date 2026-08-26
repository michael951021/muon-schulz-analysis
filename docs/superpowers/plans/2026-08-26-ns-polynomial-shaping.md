# Newton–Schulz Polynomial Shaping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the verified quintic coefficient family and the instrumented Muon optimizer that stages 3–5 of the ablation will consume.

**Architecture:** A pure-numpy polynomial/analysis core (`polynomial.py`) is the single source of mathematical truth. A stochastic solver (`solve.py`) uses it offline to produce a committed JSON coefficient table; training code only ever reads that table. Torch appears solely in `optim/` and `diagnostics/`, behind interfaces that take a coefficient-set *name* and a *hook*, so neither knows what model it is training.

**Tech Stack:** Python 3.12, numpy, scipy (solver only), torch (optimizer + logger only), pytest, uv.

## Global Constraints

- Step count is pinned at **N = 5** for every family member — the ablation is compute-matched. N is a parameter with default 5, never swept in this plan.
- The band constraint is **relative** half-width `relhw(S) = (max S − min S)/(max S + min S)`, not absolute deviation from 1. Scale-invariance is what decouples γ from η.
- Forward-invariance bound is the **relaxed** `B = max(1+γ, sup_{[0,1]} P)`. Never `B = 1+γ`.
- The binding shape constraint is **positivity on (0, B]**, NOT monotonicity. Keller's quintic has x\* = 0.5545 and is non-monotone on [0,1]; a monotonicity constraint excludes the entire useful family.
- σ grid is `np.logspace(-7, 0, 3000)` everywhere.
- Keller baseline coefficients, exactly: `a=3.4445, b=-4.7750, c=2.0315`.
- `polynomial.py` and `solve.py` must not import torch. `optim/` and `diagnostics/` must not import scipy.
- Property tests in `tests/test_coefficient_table.py` must not import `nsshape.solve`.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | uv/hatchling package `nsshape`, deps, pytest config |
| `src/nsshape/polynomial.py` | Evaluate, compose, analyze quintics. Pure numpy. Reference truth. |
| `src/nsshape/solve.py` | Constrained fit γ → coefficients + certificate. scipy. |
| `src/nsshape/coefficients.py` | Load/validate the committed table; resolve entries by name. |
| `src/nsshape/optim/orthogonalize.py` | `ns_quintic` (bf16) and `exact_svd_polar` (fp32) backends. |
| `src/nsshape/optim/muon.py` | `ParameterizedMuon` optimizer. |
| `src/nsshape/diagnostics/schema.py` | `SpectrumRecord` dataclass + JSONL serialization. |
| `src/nsshape/diagnostics/svd_logger.py` | Sampling hook, fp32 SVD before/after, JSONL writer. |
| `data/coefficients/v1.json` | Generated, committed artifact. |
| `scripts/generate_coefficients.py` | Runs the solver once, writes the table. |
| `scripts/plot_composed_maps.py` | Diagnostic plots of C(x) per entry. |

---

### Task 1: Package scaffold and polynomial primitives

**Files:**
- Create: `pyproject.toml`, `src/nsshape/__init__.py`, `src/nsshape/polynomial.py`
- Test: `tests/test_polynomial.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Coeffs = tuple[float, float, float]`; `evaluate(coeffs, x) -> np.ndarray`; `derivative(coeffs, x) -> np.ndarray`; `compose(coeffs, x, n_steps=5) -> np.ndarray`; `DEFAULT_SIGMA_GRID: np.ndarray`; `KELLER: Coeffs`; `DIVERGENCE_BOUND: float`.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "nsshape"
version = "0.1.0"
description = "Newton-Schulz polynomial shaping ablation for Muon"
requires-python = ">=3.10,<3.14"
dependencies = ["numpy>=1.24"]

[project.optional-dependencies]
solve = ["scipy>=1.11"]
torch = ["torch>=2.2"]
plot = ["matplotlib>=3.8"]
dev = ["pytest>=8.0", "scipy>=1.11", "torch>=2.2", "matplotlib>=3.8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/nsshape"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create the venv and install**

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

- [ ] **Step 3: Write the failing test**

`tests/test_polynomial.py`:

```python
import numpy as np
import pytest
from nsshape.polynomial import KELLER, compose, derivative, evaluate


def test_evaluate_matches_explicit_quintic():
    coeffs = (2.0, -1.5, 0.5)
    x = np.array([0.0, 0.5, 1.0])
    expected = 2.0 * x - 1.5 * x**3 + 0.5 * x**5
    np.testing.assert_allclose(evaluate(coeffs, x), expected)


def test_evaluate_is_odd():
    coeffs = (3.0, -4.0, 2.0)
    x = np.linspace(0.1, 1.5, 20)
    np.testing.assert_allclose(evaluate(coeffs, -x), -evaluate(coeffs, x))


def test_evaluate_accepts_scalar():
    assert evaluate((2.0, 0.0, 0.0), 3.0) == pytest.approx(6.0)


def test_derivative_matches_finite_difference():
    coeffs = KELLER
    x = np.linspace(0.05, 1.2, 50)
    h = 1e-6
    fd = (evaluate(coeffs, x + h) - evaluate(coeffs, x - h)) / (2 * h)
    np.testing.assert_allclose(derivative(coeffs, x), fd, rtol=1e-5)


def test_compose_zero_steps_is_identity():
    x = np.linspace(0.0, 1.0, 10)
    np.testing.assert_allclose(compose(KELLER, x, n_steps=0), x)


def test_compose_one_step_equals_evaluate():
    x = np.linspace(0.0, 1.0, 10)
    np.testing.assert_allclose(compose(KELLER, x, n_steps=1), evaluate(KELLER, x))


def test_compose_five_steps_is_manual_nesting():
    x = np.linspace(0.01, 1.0, 10)
    manual = x
    for _ in range(5):
        manual = evaluate(KELLER, manual)
    np.testing.assert_allclose(compose(KELLER, x, n_steps=5), manual)


def test_compose_fixes_zero():
    assert compose(KELLER, np.array([0.0]))[0] == pytest.approx(0.0)


def test_compose_saturates_instead_of_overflowing():
    """Divergent coefficients must return finite saturated values, not inf/nan."""
    out = compose((50.0, 50.0, 50.0), np.array([0.9]), n_steps=5)
    assert np.all(np.isfinite(out))


def test_keller_c_of_one_is_known_value():
    assert compose(KELLER, np.array([1.0]))[0] == pytest.approx(0.6964, abs=1e-3)
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_polynomial.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'nsshape.polynomial'`

- [ ] **Step 5: Write `src/nsshape/__init__.py`**

```python
"""Newton-Schulz polynomial shaping for Muon."""

__version__ = "0.1.0"
```

- [ ] **Step 6: Write `src/nsshape/polynomial.py` (primitives only)**

```python
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
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_polynomial.py -q`
Expected: 10 passed

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/nsshape/__init__.py src/nsshape/polynomial.py tests/test_polynomial.py
git commit -m "feat: package scaffold and quintic polynomial primitives"
```

---

### Task 2: Polynomial analysis — critical point, invariance, positivity, band extent

**Files:**
- Modify: `src/nsshape/polynomial.py` (append)
- Test: `tests/test_polynomial_analysis.py`

**Interfaces:**
- Consumes: `evaluate`, `compose`, `KELLER`, `DEFAULT_SIGMA_GRID` from Task 1.
- Produces: `first_critical_point(coeffs) -> float`; `invariance_bound(coeffs, gamma, n=2001) -> float`; `invariance_violation(coeffs, B, n=2001) -> float`; `positivity_violation(coeffs, B, n=2001) -> float`; `BandExtent` dataclass with fields `sigma_min: float`, `scale: float`, `achieved_half_width: float`; `band_extent(coeffs, gamma, n_steps=5, grid=None) -> BandExtent`.

Semantics, all verified numerically during design:
- `first_critical_point` returns the smallest positive root of P′, or `inf` if none.
- `invariance_violation(coeffs, B) <= 0` means [0, B] is forward-invariant.
- `positivity_violation(coeffs, B) <= 0` means P > 0 on (0, B].
- `band_extent` walks leftward from x=1 along the grid, growing the image interval, and stops when relative half-width would exceed γ. `sigma_min == 1.0` signals failure (no usable range).

- [ ] **Step 1: Write the failing test**

`tests/test_polynomial_analysis.py`:

```python
import numpy as np
import pytest
from nsshape.polynomial import (
    KELLER,
    band_extent,
    derivative,
    evaluate,
    first_critical_point,
    invariance_bound,
    invariance_violation,
    positivity_violation,
)


def test_keller_critical_point_is_known_value():
    """Keller's quintic is monotone only up to x* = 0.5545 -- this is the
    fact that rules out any monotonicity-on-[0,1] constraint."""
    assert first_critical_point(KELLER) == pytest.approx(0.5545, abs=1e-3)


def test_critical_point_is_a_root_of_the_derivative():
    xstar = first_critical_point(KELLER)
    assert derivative(KELLER, xstar) == pytest.approx(0.0, abs=1e-9)


def test_derivative_positive_below_critical_point():
    xstar = first_critical_point(KELLER)
    xs = np.linspace(1e-6, xstar * 0.999, 200)
    assert np.all(derivative(KELLER, xs) > 0)


def test_monotone_polynomial_has_no_critical_point():
    assert first_critical_point((2.0, 0.0, 0.0)) == np.inf


def test_pure_cubic_critical_point():
    # P = 3x - x^3 -> P' = 3 - 3x^2 -> x* = 1
    assert first_critical_point((3.0, -1.0, 0.0)) == pytest.approx(1.0)


def test_invariance_bound_is_relaxed_for_keller():
    """B must be max(1+gamma, sup_[0,1] P), not 1+gamma."""
    assert invariance_bound(KELLER, 0.30) == pytest.approx(1.30, abs=1e-3)


def test_keller_violates_strict_invariance():
    """Documented design finding: +0.23 at B=1.3."""
    assert invariance_violation(KELLER, 1.3) == pytest.approx(0.23, abs=0.01)


def test_invariance_violation_negative_for_contraction():
    # P = 0.5x maps [0,1] into [0,0.5]
    assert invariance_violation((0.5, 0.0, 0.0), 1.0) < 0


def test_keller_satisfies_positivity():
    """Keller passes positivity with a small margin (min P = +0.0022)."""
    B = invariance_bound(KELLER, 0.30)
    assert positivity_violation(KELLER, B) < 0


def test_positivity_violation_detects_negative_lobe():
    # P = x - 4x^3 goes negative well before x=1
    assert positivity_violation((1.0, -4.0, 0.0), 1.0) > 0


def test_keller_band_extent_matches_design_measurement():
    """Keller achieves sigma_min = 1.42e-3 at gamma = 0.30."""
    r = band_extent(KELLER, 0.30)
    assert r.sigma_min == pytest.approx(1.42e-3, rel=0.05)
    assert r.achieved_half_width <= 0.30


def test_band_extent_image_actually_lies_in_the_band():
    """Independent re-derivation: recompute the image and check relhw."""
    from nsshape.polynomial import compose

    gamma = 0.30
    r = band_extent(KELLER, gamma)
    xs = np.logspace(np.log10(r.sigma_min), 0.0, 500)
    img = compose(KELLER, xs)
    relhw = (img.max() - img.min()) / (img.max() + img.min())
    assert relhw <= gamma + 1e-3


def test_band_extent_scale_is_image_midpoint():
    from nsshape.polynomial import compose

    r = band_extent(KELLER, 0.30)
    xs = np.logspace(np.log10(r.sigma_min), 0.0, 500)
    img = compose(KELLER, xs)
    assert r.scale == pytest.approx((img.min() + img.max()) / 2, rel=1e-2)


def test_band_extent_wider_gamma_reaches_smaller_sigma():
    assert band_extent(KELLER, 0.40).sigma_min < band_extent(KELLER, 0.30).sigma_min


def test_band_extent_signals_failure_with_sigma_min_one():
    """A map sending 1 to a nonpositive value has no usable range."""
    r = band_extent((-1.0, 0.0, 0.0), 0.3)
    assert r.sigma_min == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_polynomial_analysis.py -q`
Expected: FAIL — `ImportError: cannot import name 'band_extent'`

- [ ] **Step 3: Append the analysis functions to `src/nsshape/polynomial.py`**

```python
from dataclasses import dataclass


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


def invariance_bound(coeffs: Coeffs, gamma: float, n: int = 2001) -> float:
    """The relaxed forward-invariance bound B = max(1+gamma, sup_[0,1] P).

    The strict bound 1+gamma spuriously excludes Keller-like members, whose band
    is centered near 0.7 rather than 1.
    """
    xs = np.linspace(0.0, 1.0, n)
    return float(max(1.0 + gamma, evaluate(coeffs, xs).max()))


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/ -q`
Expected: 25 passed

- [ ] **Step 5: Commit**

```bash
git add src/nsshape/polynomial.py tests/test_polynomial_analysis.py
git commit -m "feat: polynomial analysis - critical point, invariance, positivity, band extent"
```

---

### Task 3: The constrained solver

**Files:**
- Create: `src/nsshape/solve.py`
- Test: `tests/test_solve.py`

**Interfaces:**
- Consumes: everything from Task 2.
- Produces: `FitResult` dataclass with fields `gamma: float`, `n_steps: int`, `coeffs: Coeffs`, `sigma_min: float`, `scale: float`, `achieved_half_width: float`, `critical_point: float`, `invariance_bound: float`, `invariance_margin: float`, `positivity_margin: float`, `feasible: bool`, `seed: int`; `objective(params, gamma, n_steps=5) -> float`; `fit_gamma(gamma, n_steps=5, seeds=(0,1,2,3,4), maxiter=400, popsize=40) -> FitResult`.

Margins are `-violation`, so **positive means satisfied**.

- [ ] **Step 1: Write the failing test**

`tests/test_solve.py`:

```python
import numpy as np
import pytest
from nsshape.polynomial import KELLER, band_extent
from nsshape.solve import fit_gamma, objective

pytestmark = pytest.mark.slow


def test_objective_rejects_nonpositive_leading_coefficient():
    assert objective((-1.0, 0.0, 0.0), 0.3) >= 100.0


def test_objective_rejects_saturating_degenerate_solution():
    """a~8, c~12 saturates everything -- the degenerate optimum that motivated
    the invariance constraint. It must score as infeasible."""
    assert objective((8.0, 0.0, 12.0), 0.3) >= 100.0


def test_objective_accepts_keller():
    """Keller must be feasible, i.e. score below the penalty floor."""
    assert objective(KELLER, 0.30) < 0.0


def test_objective_returns_log10_sigma_min_for_feasible():
    expected = np.log10(band_extent(KELLER, 0.30).sigma_min)
    assert objective(KELLER, 0.30) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("gamma", [0.15, 0.30, 0.50])
def test_fit_is_feasible_and_satisfies_its_own_constraints(gamma):
    r = fit_gamma(gamma, seeds=(0, 1, 2))
    assert r.feasible
    assert r.sigma_min < 1.0
    assert r.achieved_half_width <= gamma + 1e-6
    assert r.invariance_margin >= 0
    assert r.positivity_margin >= 0


def test_fit_beats_keller_at_gamma_030():
    """The fitted optimum must reach further than Keller at the same band."""
    r = fit_gamma(0.30, seeds=(0, 1, 2))
    assert r.sigma_min < band_extent(KELLER, 0.30).sigma_min


def test_keller_is_within_25_percent_of_frontier():
    """Spec acceptance test: Keller is a NEAR-OPTIMAL interior point."""
    r = fit_gamma(0.30, seeds=(0, 1, 2))
    keller_sigma_min = band_extent(KELLER, 0.30).sigma_min
    assert keller_sigma_min <= r.sigma_min * 1.25


def test_wider_gamma_reaches_smaller_sigma_min():
    """The frontier must be monotone -- a wider band buys dynamic range."""
    tight = fit_gamma(0.20, seeds=(0, 1, 2))
    wide = fit_gamma(0.40, seeds=(0, 1, 2))
    assert wide.sigma_min < tight.sigma_min


def test_gamma_005_is_infeasible_at_five_steps():
    """Documented result: exact polar is unreachable by 5 quintic steps."""
    assert not fit_gamma(0.05, seeds=(0, 1, 2)).feasible


def test_multi_seed_recovers_gamma_050():
    """Single-seed DE fell into an infeasible local optimum at gamma=0.50."""
    assert fit_gamma(0.50, seeds=(0, 1, 2, 3, 4)).feasible
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_solve.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'nsshape.solve'`

- [ ] **Step 3: Write `src/nsshape/solve.py`**

```python
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
    invariance_bound,
    invariance_violation,
    positivity_violation,
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

    B = invariance_bound(coeffs, gamma)
    inv = invariance_violation(coeffs, B)
    if inv > 0:
        return PENALTY + inv

    pos = positivity_violation(coeffs, B)
    if pos > 0:
        return PENALTY + pos

    extent = band_extent(coeffs, gamma, n_steps)
    if extent.sigma_min >= 1.0:
        return PENALTY / 2.0
    return float(np.log10(extent.sigma_min))


def _certify(coeffs: Coeffs, gamma: float, n_steps: int, seed: int) -> FitResult:
    """Build a FitResult by re-measuring every property from scratch."""
    B = invariance_bound(coeffs, gamma)
    extent = band_extent(coeffs, gamma, n_steps)
    inv_margin = -invariance_violation(coeffs, B)
    pos_margin = -positivity_violation(coeffs, B)
    feasible = bool(
        extent.sigma_min < 1.0
        and inv_margin >= 0
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
```

- [ ] **Step 4: Register the `slow` marker in `pyproject.toml`**

Add under `[tool.pytest.ini_options]`:

```toml
markers = ["slow: solver fits, minutes not seconds"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_solve.py -q`
Expected: 10 passed (several minutes — these run differential evolution)

- [ ] **Step 6: Commit**

```bash
git add src/nsshape/solve.py tests/test_solve.py pyproject.toml
git commit -m "feat: constrained multi-seed solver for the quintic family"
```

---

### Task 4: Generate and commit the coefficient table

**Files:**
- Create: `src/nsshape/coefficients.py`, `scripts/generate_coefficients.py`, `data/coefficients/v1.json`
- Test: `tests/test_coefficients_loader.py`

**Interfaces:**
- Consumes: `FitResult` from Task 3, `KELLER` from Task 1.
- Produces: `CoefficientSet` dataclass with fields `name: str`, `kind: str` (`"fitted" | "baseline" | "exact_svd"`), `gamma: float | None`, `coeffs: Coeffs | None`, `n_steps: int`, `scale: float`, `sigma_min: float`, `achieved_half_width: float`, `critical_point: float`, `invariance_bound: float`, `invariance_margin: float`, `positivity_margin: float`; `load_table(path=None) -> dict[str, CoefficientSet]`; `get(name, path=None) -> CoefficientSet`; `TABLE_PATH: Path`; `EXACT_SVD_NAME = "exact_svd"`.

The γ grid: `[0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]` — eight fitted entries, plus `keller` and `exact_svd`, giving ten named entries.

The `exact_svd` entry has `coeffs=None`, `gamma=0.0`, `scale=1.0`, `sigma_min=0.0`; it is a sentinel the optimizer dispatches on, not a polynomial.

- [ ] **Step 1: Write the failing test**

`tests/test_coefficients_loader.py`:

```python
import pytest
from nsshape.coefficients import EXACT_SVD_NAME, CoefficientSet, get, load_table


def test_table_loads():
    assert len(load_table()) >= 10


def test_table_contains_keller_and_exact_svd():
    table = load_table()
    assert "keller" in table
    assert EXACT_SVD_NAME in table


def test_keller_entry_has_exact_published_coefficients():
    k = get("keller")
    assert k.coeffs == (3.4445, -4.7750, 2.0315)
    assert k.kind == "baseline"


def test_exact_svd_sentinel_has_no_coefficients():
    e = get(EXACT_SVD_NAME)
    assert e.coeffs is None
    assert e.kind == "exact_svd"
    assert e.gamma == 0.0
    assert e.scale == 1.0


def test_every_entry_has_five_steps():
    assert all(cs.n_steps == 5 for cs in load_table().values())


def test_fitted_entries_span_the_documented_gamma_range():
    gammas = sorted(cs.gamma for cs in load_table().values() if cs.kind == "fitted")
    assert gammas[0] == pytest.approx(0.12)
    assert gammas[-1] == pytest.approx(0.60)
    assert len(gammas) == 8


def test_get_rejects_unknown_name():
    with pytest.raises(KeyError, match="unknown coefficient set"):
        get("does_not_exist")


def test_entries_are_frozen():
    with pytest.raises(Exception):
        get("keller").name = "mutated"


def test_returns_coefficient_set_instances():
    assert all(isinstance(cs, CoefficientSet) for cs in load_table().values())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_coefficients_loader.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'nsshape.coefficients'`

- [ ] **Step 3: Write `src/nsshape/coefficients.py`**

```python
"""Load and validate the committed coefficient table.

The table is a generated artifact produced once, offline, by
scripts/generate_coefficients.py. Training code reads it and never runs the
solver, so every run is traceable to exact coefficients by name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from nsshape.polynomial import Coeffs

TABLE_PATH = Path(__file__).resolve().parents[2] / "data" / "coefficients" / "v1.json"

#: Sentinel name for the exact-SVD polar arm (not a polynomial).
EXACT_SVD_NAME = "exact_svd"


@dataclass(frozen=True)
class CoefficientSet:
    name: str
    kind: str
    gamma: float | None
    coeffs: Coeffs | None
    n_steps: int
    scale: float
    sigma_min: float
    achieved_half_width: float
    critical_point: float
    invariance_bound: float
    invariance_margin: float
    positivity_margin: float

    @property
    def is_exact_svd(self) -> bool:
        return self.kind == "exact_svd"


def _parse(entry: dict) -> CoefficientSet:
    coeffs = entry.get("coeffs")
    return CoefficientSet(
        name=entry["name"],
        kind=entry["kind"],
        gamma=entry.get("gamma"),
        coeffs=tuple(coeffs) if coeffs is not None else None,
        n_steps=entry["n_steps"],
        scale=entry["scale"],
        sigma_min=entry["sigma_min"],
        achieved_half_width=entry["achieved_half_width"],
        critical_point=entry["critical_point"],
        invariance_bound=entry["invariance_bound"],
        invariance_margin=entry["invariance_margin"],
        positivity_margin=entry["positivity_margin"],
    )


def load_table(path: Path | None = None) -> dict[str, CoefficientSet]:
    """Read the committed table, keyed by entry name."""
    path = path or TABLE_PATH
    payload = json.loads(path.read_text())
    return {e["name"]: _parse(e) for e in payload["entries"]}


def get(name: str, path: Path | None = None) -> CoefficientSet:
    """Resolve one entry by name."""
    table = load_table(path)
    if name not in table:
        raise KeyError(f"unknown coefficient set {name!r}; have {sorted(table)}")
    return table[name]
```

- [ ] **Step 4: Write `scripts/generate_coefficients.py`**

```python
"""Generate the committed coefficient table. Run once; commit the output.

Usage:  uv run python scripts/generate_coefficients.py
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from nsshape.coefficients import EXACT_SVD_NAME, TABLE_PATH
from nsshape.polynomial import (
    KELLER,
    band_extent,
    first_critical_point,
    invariance_bound,
    invariance_violation,
    positivity_violation,
)
from nsshape.solve import fit_gamma

GAMMA_GRID = [0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
SEEDS = (0, 1, 2, 3, 4)
KELLER_GAMMA = 0.30


def _entry_from_fit(name, kind, fit):
    return {
        "name": name,
        "kind": kind,
        "gamma": fit.gamma,
        "coeffs": list(fit.coeffs),
        "n_steps": fit.n_steps,
        "scale": fit.scale,
        "sigma_min": fit.sigma_min,
        "achieved_half_width": fit.achieved_half_width,
        "critical_point": fit.critical_point,
        "invariance_bound": fit.invariance_bound,
        "invariance_margin": fit.invariance_margin,
        "positivity_margin": fit.positivity_margin,
        "seed": fit.seed,
    }


def main() -> None:
    entries = []

    for gamma in GAMMA_GRID:
        fit = fit_gamma(gamma, seeds=SEEDS)
        if not fit.feasible:
            raise SystemExit(f"gamma={gamma} infeasible; adjust GAMMA_GRID")
        entries.append(_entry_from_fit(f"gamma_{gamma:.2f}", "fitted", fit))
        print(f"gamma={gamma:.2f}  coeffs={fit.coeffs}  sigma_min={fit.sigma_min:.3e}")

    # Keller baseline: measured, not fitted.
    B = invariance_bound(KELLER, KELLER_GAMMA)
    extent = band_extent(KELLER, KELLER_GAMMA)
    entries.append(
        {
            "name": "keller",
            "kind": "baseline",
            "gamma": KELLER_GAMMA,
            "coeffs": list(KELLER),
            "n_steps": 5,
            "scale": extent.scale,
            "sigma_min": extent.sigma_min,
            "achieved_half_width": extent.achieved_half_width,
            "critical_point": first_critical_point(KELLER),
            "invariance_bound": B,
            "invariance_margin": -invariance_violation(KELLER, B),
            "positivity_margin": -positivity_violation(KELLER, B),
            "seed": None,
        }
    )

    entries.append(
        {
            "name": EXACT_SVD_NAME,
            "kind": "exact_svd",
            "gamma": 0.0,
            "coeffs": None,
            "n_steps": 5,
            "scale": 1.0,
            "sigma_min": 0.0,
            "achieved_half_width": 0.0,
            "critical_point": float("inf"),
            "invariance_bound": 1.0,
            "invariance_margin": float("inf"),
            "positivity_margin": float("inf"),
            "seed": None,
        }
    )

    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(
        json.dumps(
            {
                "version": 1,
                "generated": date.today().isoformat(),
                "n_steps": 5,
                "sigma_grid": "logspace(-7, 0, 3000)",
                "seeds": list(SEEDS),
                "entries": entries,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {len(entries)} entries to {TABLE_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Generate the table**

Run: `uv run python scripts/generate_coefficients.py`
Expected: prints one line per γ, then `wrote 10 entries`. Takes several minutes.

Note: `float("inf")` serializes to `Infinity`, which `json.loads` accepts.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_coefficients_loader.py -q`
Expected: 9 passed

- [ ] **Step 7: Commit**

```bash
git add src/nsshape/coefficients.py scripts/generate_coefficients.py \
        data/coefficients/v1.json tests/test_coefficients_loader.py
git commit -m "feat: generate and commit the coefficient table"
```

---

### Task 5: Independent property verification of the committed table

**Files:**
- Create: `tests/test_coefficient_table.py`

**Interfaces:**
- Consumes: `load_table` from Task 4, analysis functions from Task 2.
- Produces: nothing (tests only).

This task is the scientific claim. It must **not** import `nsshape.solve` — the
certificates stored by the solver are re-derived here from the raw coefficients.

- [ ] **Step 1: Write the test**

`tests/test_coefficient_table.py`:

```python
"""Independent re-verification of the committed coefficient table.

Deliberately does NOT import nsshape.solve. Every property is recomputed from
the raw coefficients, so a solver bug cannot certify itself.
"""

import numpy as np
import pytest
from nsshape.coefficients import load_table
from nsshape.polynomial import (
    KELLER,
    band_extent,
    compose,
    evaluate,
    first_critical_point,
    invariance_violation,
    positivity_violation,
)

TABLE = load_table()
POLYNOMIALS = [(n, cs) for n, cs in TABLE.items() if cs.coeffs is not None]
FITTED = [(n, cs) for n, cs in POLYNOMIALS if cs.kind == "fitted"]


@pytest.mark.parametrize("name,cs", POLYNOMIALS)
def test_entry_is_forward_invariant(name, cs):
    assert invariance_violation(cs.coeffs, cs.invariance_bound) <= 1e-9


@pytest.mark.parametrize("name,cs", POLYNOMIALS)
def test_entry_is_positive_on_its_bound(name, cs):
    assert positivity_violation(cs.coeffs, cs.invariance_bound) <= 1e-9


@pytest.mark.parametrize("name,cs", POLYNOMIALS)
def test_entry_maps_its_range_into_its_claimed_band(name, cs):
    """Recompute the image directly rather than trusting the stored extent."""
    xs = np.logspace(np.log10(cs.sigma_min), 0.0, 2000)
    image = compose(cs.coeffs, xs, cs.n_steps)
    assert np.all(image > 0)
    relhw = (image.max() - image.min()) / (image.max() + image.min())
    assert relhw <= cs.gamma + 1e-3


@pytest.mark.parametrize("name,cs", POLYNOMIALS)
def test_stored_scale_is_the_image_midpoint(name, cs):
    xs = np.logspace(np.log10(cs.sigma_min), 0.0, 2000)
    image = compose(cs.coeffs, xs, cs.n_steps)
    assert cs.scale == pytest.approx((image.min() + image.max()) / 2, rel=1e-2)


@pytest.mark.parametrize("name,cs", POLYNOMIALS)
def test_normalized_output_is_centered_on_one(name, cs):
    """After dividing by scale, the band is centered at 1 -- this is what
    decouples gamma from the learning rate."""
    xs = np.logspace(np.log10(cs.sigma_min), 0.0, 2000)
    image = compose(cs.coeffs, xs, cs.n_steps) / cs.scale
    assert (image.min() + image.max()) / 2 == pytest.approx(1.0, rel=1e-2)


@pytest.mark.parametrize("name,cs", POLYNOMIALS)
def test_stored_critical_point_is_correct(name, cs):
    assert first_critical_point(cs.coeffs) == pytest.approx(cs.critical_point, rel=1e-6)


@pytest.mark.parametrize("name,cs", POLYNOMIALS)
def test_stored_margins_are_nonnegative(name, cs):
    assert cs.invariance_margin >= -1e-9
    assert cs.positivity_margin >= -1e-9


@pytest.mark.parametrize("name,cs", POLYNOMIALS)
def test_leading_coefficient_positive(name, cs):
    assert cs.coeffs[0] > 0


def test_frontier_is_monotone_in_gamma():
    """A wider band must buy strictly more dynamic range."""
    ordered = sorted(FITTED, key=lambda kv: kv[1].gamma)
    sigmas = [cs.sigma_min for _, cs in ordered]
    assert all(b < a for a, b in zip(sigmas, sigmas[1:])), sigmas


def test_keller_lies_inside_the_fitted_frontier():
    """Spec acceptance test: standard Muon is a near-optimal interior point."""
    keller = TABLE["keller"]
    fitted_at_030 = TABLE["gamma_0.30"]
    assert keller.sigma_min >= fitted_at_030.sigma_min
    assert keller.sigma_min <= fitted_at_030.sigma_min * 1.25


def test_keller_entry_matches_direct_measurement():
    extent = band_extent(KELLER, 0.30)
    keller = TABLE["keller"]
    assert keller.sigma_min == pytest.approx(extent.sigma_min, rel=1e-9)
    assert keller.scale == pytest.approx(extent.scale, rel=1e-9)


def test_no_entry_is_monotone_across_the_unit_interval():
    """Documents why the constraint is positivity, not monotonicity: the whole
    useful family turns over well before x=1."""
    for name, cs in POLYNOMIALS:
        assert cs.critical_point < 1.0, name


def test_table_covers_the_documented_gamma_span():
    gammas = sorted(cs.gamma for _, cs in FITTED)
    assert min(gammas) <= 0.12
    assert max(gammas) >= 0.60
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/test_coefficient_table.py -q`
Expected: all pass. If `test_frontier_is_monotone_in_gamma` fails, a γ hit a bad
local optimum — rerun `generate_coefficients.py` with more seeds.

- [ ] **Step 3: Verify the isolation constraint holds**

Run: `! grep -q "nsshape.solve" tests/test_coefficient_table.py && echo "isolated OK"`
Expected: `isolated OK`

- [ ] **Step 4: Commit**

```bash
git add tests/test_coefficient_table.py
git commit -m "test: independent property verification of the coefficient table"
```

---

### Task 6: Orthogonalization backends

**Files:**
- Create: `src/nsshape/optim/__init__.py`, `src/nsshape/optim/orthogonalize.py`
- Test: `tests/test_orthogonalize.py`

**Interfaces:**
- Consumes: `CoefficientSet` from Task 4.
- Produces: `ns_quintic(G, coeffs, n_steps=5, scale=1.0, eps=1e-7) -> Tensor`; `exact_svd_polar(G) -> Tensor`; `orthogonalize(G, cs: CoefficientSet) -> Tensor` (dispatches on `cs.is_exact_svd`).

`ns_quintic` follows Muon: Frobenius-normalize, operate on the short side by
transposing tall matrices, run in bf16, then divide by `scale` and cast back.

- [ ] **Step 1: Write the failing test**

`tests/test_orthogonalize.py`:

```python
import pytest

torch = pytest.importorskip("torch")

from nsshape.coefficients import get
from nsshape.optim.orthogonalize import exact_svd_polar, ns_quintic, orthogonalize


def _singular_values(M):
    return torch.linalg.svdvals(M.float())


def test_exact_svd_polar_is_orthogonal():
    torch.manual_seed(0)
    G = torch.randn(32, 32)
    s = _singular_values(exact_svd_polar(G))
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)


def test_exact_svd_polar_preserves_shape_for_wide_and_tall():
    for shape in [(8, 32), (32, 8)]:
        assert exact_svd_polar(torch.randn(*shape)).shape == shape


def test_exact_svd_polar_matches_u_vh():
    torch.manual_seed(1)
    G = torch.randn(16, 24)
    U, _, Vh = torch.linalg.svd(G, full_matrices=False)
    assert torch.allclose(exact_svd_polar(G), U @ Vh, atol=1e-5)


def test_ns_quintic_preserves_shape():
    for shape in [(8, 32), (32, 8), (16, 16)]:
        out = ns_quintic(torch.randn(*shape), get("keller").coeffs)
        assert out.shape == shape


def test_ns_quintic_drives_singular_values_into_band():
    """A well-conditioned matrix should come out with relhw <= gamma."""
    torch.manual_seed(2)
    cs = get("keller")
    G = torch.randn(64, 64)
    s = _singular_values(ns_quintic(G, cs.coeffs, scale=cs.scale))
    relhw = (s.max() - s.min()) / (s.max() + s.min())
    assert relhw <= cs.gamma + 0.05


def test_ns_quintic_normalized_band_centers_near_one():
    torch.manual_seed(3)
    cs = get("keller")
    s = _singular_values(ns_quintic(torch.randn(64, 64), cs.coeffs, scale=cs.scale))
    assert (s.min() + s.max()).item() / 2 == pytest.approx(1.0, abs=0.15)


def test_ns_quintic_roughly_agrees_with_exact_polar():
    """On a well-conditioned matrix the two backends should align in direction."""
    torch.manual_seed(4)
    cs = get("keller")
    G = torch.randn(48, 48)
    approx = ns_quintic(G, cs.coeffs, scale=cs.scale).float()
    exact = exact_svd_polar(G).float()
    cosine = (approx * exact).sum() / (approx.norm() * exact.norm())
    assert cosine > 0.95


def test_ns_quintic_is_scale_invariant_in_input():
    """Frobenius normalization means input scaling must not change the output."""
    torch.manual_seed(5)
    cs = get("keller")
    G = torch.randn(24, 24)
    a = ns_quintic(G, cs.coeffs, scale=cs.scale)
    b = ns_quintic(G * 100.0, cs.coeffs, scale=cs.scale)
    assert torch.allclose(a.float(), b.float(), atol=1e-2)


def test_ns_quintic_handles_rank_deficient_input():
    G = torch.zeros(16, 16)
    G[0, 0] = 1.0
    assert torch.all(torch.isfinite(ns_quintic(G, get("keller").coeffs)))


def test_orthogonalize_dispatches_to_svd_for_sentinel():
    torch.manual_seed(6)
    G = torch.randn(16, 16)
    assert torch.allclose(orthogonalize(G, get("exact_svd")), exact_svd_polar(G), atol=1e-5)


def test_orthogonalize_dispatches_to_ns_for_polynomial():
    torch.manual_seed(7)
    cs = get("keller")
    G = torch.randn(16, 16)
    assert torch.allclose(
        orthogonalize(G, cs).float(),
        ns_quintic(G, cs.coeffs, cs.n_steps, cs.scale).float(),
    )


def test_orthogonalize_rejects_non_matrix():
    with pytest.raises(ValueError, match="2D"):
        orthogonalize(torch.randn(4, 4, 4), get("keller"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_orthogonalize.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'nsshape.optim'`

- [ ] **Step 3: Write `src/nsshape/optim/__init__.py`**

```python
"""Optimizer and orthogonalization backends."""
```

- [ ] **Step 4: Write `src/nsshape/optim/orthogonalize.py`**

```python
"""Two interchangeable orthogonalization backends behind one signature.

ns_quintic is the practical Newton-Schulz step in bf16; exact_svd_polar is the
gamma=0 anchor in fp32. Because they share a signature, the SVD arm needs no
special-casing anywhere in the training loop.
"""

from __future__ import annotations

import torch

from nsshape.coefficients import CoefficientSet
from nsshape.polynomial import Coeffs


def ns_quintic(
    G: torch.Tensor,
    coeffs: Coeffs,
    n_steps: int = 5,
    scale: float = 1.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Newton-Schulz quintic iteration, Muon-style.

    Operates on the short side by transposing tall inputs, which keeps the
    Gram matrix as small as possible. Runs in bf16 -- the low precision is part
    of what the ablation is testing, so it is deliberate, not incidental.

    The output is divided by `scale` so the singular-value band is centered on
    1, which decouples the band shape from the learning rate.
    """
    if G.ndim != 2:
        raise ValueError(f"expected a 2D matrix, got shape {tuple(G.shape)}")
    a, b, c = coeffs
    X = G.bfloat16()
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(n_steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return (X / scale).to(G.dtype)


def exact_svd_polar(G: torch.Tensor) -> torch.Tensor:
    """Exact polar factor U @ Vh in fp32 -- the gamma = 0 anchor."""
    if G.ndim != 2:
        raise ValueError(f"expected a 2D matrix, got shape {tuple(G.shape)}")
    U, _, Vh = torch.linalg.svd(G.float(), full_matrices=False)
    return (U @ Vh).to(G.dtype)


def orthogonalize(G: torch.Tensor, cs: CoefficientSet) -> torch.Tensor:
    """Dispatch to the backend named by a coefficient set."""
    if G.ndim != 2:
        raise ValueError(f"expected a 2D matrix, got shape {tuple(G.shape)}")
    if cs.is_exact_svd:
        return exact_svd_polar(G)
    return ns_quintic(G, cs.coeffs, cs.n_steps, cs.scale)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_orthogonalize.py -q`
Expected: 12 passed

- [ ] **Step 6: Commit**

```bash
git add src/nsshape/optim/ tests/test_orthogonalize.py
git commit -m "feat: Newton-Schulz and exact-SVD orthogonalization backends"
```

---

### Task 7: Parameterized Muon optimizer

**Files:**
- Create: `src/nsshape/optim/muon.py`
- Test: `tests/test_muon.py`

**Interfaces:**
- Consumes: `orthogonalize` from Task 6, `get` from Task 4.
- Produces: `ParameterizedMuon(params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.0, coeff_set="keller", hook=None)`, a `torch.optim.Optimizer` with `.step(closure=None)` and attribute `.coefficient_set: CoefficientSet`.

The hook, if given, is called as `hook(step, param_index, M_before, M_after)`
where `M_before` is the momentum buffer and `M_after` the orthogonalized update.
Task 8 supplies the concrete hook.

- [ ] **Step 1: Write the failing test**

`tests/test_muon.py`:

```python
import pytest

torch = pytest.importorskip("torch")

from nsshape.optim.muon import ParameterizedMuon


def _param(shape, seed=0):
    torch.manual_seed(seed)
    p = torch.randn(*shape, requires_grad=True)
    p.grad = torch.randn(*shape)
    return p


def test_step_changes_parameters():
    p = _param((16, 16))
    before = p.detach().clone()
    ParameterizedMuon([p], lr=0.1).step()
    assert not torch.allclose(p.detach(), before)


def test_step_is_a_noop_without_gradients():
    p = _param((16, 16))
    p.grad = None
    before = p.detach().clone()
    ParameterizedMuon([p], lr=0.1).step()
    assert torch.allclose(p.detach(), before)


def test_zero_lr_leaves_parameters_unchanged():
    p = _param((16, 16))
    before = p.detach().clone()
    ParameterizedMuon([p], lr=0.0).step()
    assert torch.allclose(p.detach(), before)


def test_momentum_buffer_is_created():
    p = _param((16, 16))
    opt = ParameterizedMuon([p], lr=0.1)
    opt.step()
    assert "momentum_buffer" in opt.state[p]


def test_resolves_coefficient_set_by_name():
    opt = ParameterizedMuon([_param((8, 8))], coeff_set="keller")
    assert opt.coefficient_set.coeffs == (3.4445, -4.7750, 2.0315)


def test_rejects_unknown_coefficient_set():
    with pytest.raises(KeyError, match="unknown coefficient set"):
        ParameterizedMuon([_param((8, 8))], coeff_set="nope")


def test_exact_svd_arm_runs_without_special_casing():
    p = _param((16, 16))
    before = p.detach().clone()
    ParameterizedMuon([p], lr=0.1, coeff_set="exact_svd").step()
    assert not torch.allclose(p.detach(), before)


def test_rejects_non_matrix_parameters():
    p = torch.randn(16, requires_grad=True)
    p.grad = torch.randn(16)
    with pytest.raises(ValueError, match="2D"):
        ParameterizedMuon([p], lr=0.1).step()


def test_hook_is_called_with_before_and_after():
    calls = []
    p = _param((16, 16))
    opt = ParameterizedMuon(
        [p], lr=0.1, hook=lambda s, i, b, a: calls.append((s, i, b.shape, a.shape))
    )
    opt.step()
    opt.step()
    assert [c[0] for c in calls] == [0, 1]
    assert calls[0][2] == (16, 16) and calls[0][3] == (16, 16)


def test_step_counter_advances():
    opt = ParameterizedMuon([_param((8, 8))], lr=0.1)
    opt.step()
    opt.step()
    assert opt.step_count == 2


def test_weight_decay_shrinks_a_zero_gradient_parameter():
    p = _param((16, 16))
    p.grad = torch.zeros(16, 16)
    before = p.detach().clone()
    ParameterizedMuon([p], lr=0.1, weight_decay=0.1).step()
    assert p.detach().abs().sum() < before.abs().sum()


def test_training_reduces_a_simple_quadratic_loss():
    """End-to-end sanity: the optimizer must actually optimize."""
    torch.manual_seed(0)
    W = torch.randn(16, 16, requires_grad=True)
    target = torch.randn(16, 16)
    opt = ParameterizedMuon([W], lr=0.05)
    first = last = None
    for i in range(60):
        opt.zero_grad()
        loss = ((W - target) ** 2).mean()
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()
    assert last < first
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_muon.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'nsshape.optim.muon'`

- [ ] **Step 3: Write `src/nsshape/optim/muon.py`**

```python
"""Muon, parameterized by a named coefficient set.

The optimizer knows only a coefficient-set name and an optional hook. It has no
idea what model it is training, which is what lets the same code drive both
cifar10-airbench and modded-nanogpt.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from nsshape.coefficients import get
from nsshape.optim.orthogonalize import orthogonalize


class ParameterizedMuon(torch.optim.Optimizer):
    """Muon with a swappable orthogonalization backend.

    Args:
        params: 2D parameters only. Embeddings, biases and norms belong in a
            separate AdamW group, as in standard Muon.
        lr: learning rate.
        momentum: heavy-ball coefficient on the gradient buffer.
        nesterov: use the Nesterov-style lookahead buffer.
        weight_decay: decoupled weight decay.
        coeff_set: entry name in the committed coefficient table.
        hook: optional callable(step, param_index, M_before, M_after).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        coeff_set: str = "keller",
        hook: Callable[[int, int, torch.Tensor, torch.Tensor], None] | None = None,
    ):
        self.coefficient_set = get(coeff_set)
        self.hook = hook
        self.step_count = 0
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        index = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError(
                        f"ParameterizedMuon requires 2D parameters, got {p.ndim}D; "
                        "put other parameters in a separate AdamW group"
                    )

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(p.grad)
                M = p.grad.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf

                update = orthogonalize(M, self.coefficient_set)

                if self.hook is not None:
                    self.hook(self.step_count, index, M, update)

                if group["weight_decay"] != 0.0:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])

                # Standard Muon shape correction: scale by sqrt of the aspect
                # ratio so wide and tall matrices take comparable-sized steps.
                shape_scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                p.add_(update, alpha=-group["lr"] * shape_scale)
                index += 1

        self.step_count += 1
        return loss
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_muon.py -q`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add src/nsshape/optim/muon.py tests/test_muon.py
git commit -m "feat: parameterized Muon optimizer with swappable backend"
```

---

### Task 8: SVD spectral logger

**Files:**
- Create: `src/nsshape/diagnostics/__init__.py`, `src/nsshape/diagnostics/schema.py`, `src/nsshape/diagnostics/svd_logger.py`
- Test: `tests/test_svd_logger.py`

**Interfaces:**
- Consumes: nothing from earlier tasks except being hook-compatible with Task 7.
- Produces: `SpectrumRecord` frozen dataclass with fields `step: int`, `param_index: int`, `phase: str` (`"before" | "after"`), `shape: list[int]`, `quantiles: dict[str, float]`, `band_ratio: float`, `histogram: list[int]`, `bin_edges: list[float]`, `n_singular_values: int`; `.to_json() -> str`; `SpectrumRecord.from_json(str) -> SpectrumRecord`; `spectrum_record(step, param_index, phase, M, n_bins=32) -> SpectrumRecord`; `SVDLogger(path, every_k=50, max_params=4, n_bins=32, seed=0)` with `.__call__(step, param_index, M_before, M_after)`, `.should_sample(step) -> bool`, `.close()`, and context-manager support.

`band_ratio` is `max σ / min σ` over nonzero singular values — the direct
empirical counterpart of the band the polynomial is supposed to produce.

- [ ] **Step 1: Write the failing test**

`tests/test_svd_logger.py`:

```python
import json

import pytest

torch = pytest.importorskip("torch")

from nsshape.diagnostics.schema import SpectrumRecord, spectrum_record
from nsshape.diagnostics.svd_logger import SVDLogger
from nsshape.optim.muon import ParameterizedMuon


def test_spectrum_record_round_trips():
    r = spectrum_record(3, 1, "before", torch.randn(8, 8))
    assert SpectrumRecord.from_json(r.to_json()) == r


def test_spectrum_record_json_is_one_line():
    assert "\n" not in spectrum_record(0, 0, "after", torch.randn(8, 8)).to_json()


def test_quantiles_are_ordered():
    q = spectrum_record(0, 0, "before", torch.randn(16, 16)).quantiles
    values = [q["p0"], q["p25"], q["p50"], q["p75"], q["p100"]]
    assert values == sorted(values)


def test_band_ratio_is_one_for_an_orthogonal_matrix():
    Q, _ = torch.linalg.qr(torch.randn(16, 16))
    assert spectrum_record(0, 0, "after", Q).band_ratio == pytest.approx(1.0, abs=1e-4)


def test_band_ratio_exceeds_one_for_random_matrix():
    torch.manual_seed(0)
    assert spectrum_record(0, 0, "before", torch.randn(32, 32)).band_ratio > 2.0


def test_histogram_counts_all_singular_values():
    r = spectrum_record(0, 0, "before", torch.randn(16, 24))
    assert sum(r.histogram) == r.n_singular_values == 16


def test_record_uses_fp32_even_for_bf16_input():
    """The whole point of the logger is an explicit fp32 SVD."""
    r = spectrum_record(0, 0, "after", torch.randn(16, 16).bfloat16())
    assert all(isinstance(v, float) for v in r.quantiles.values())


def test_should_sample_respects_every_k(tmp_path):
    log = SVDLogger(tmp_path / "s.jsonl", every_k=10)
    assert log.should_sample(0) and log.should_sample(10)
    assert not log.should_sample(5)
    log.close()


def test_logger_writes_before_and_after_records(tmp_path):
    path = tmp_path / "s.jsonl"
    with SVDLogger(path, every_k=1) as log:
        log(0, 0, torch.randn(8, 8), torch.randn(8, 8))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["phase"] for r in records] == ["before", "after"]


def test_logger_skips_unsampled_steps(tmp_path):
    path = tmp_path / "s.jsonl"
    with SVDLogger(path, every_k=10) as log:
        for step in range(5):
            log(step, 0, torch.randn(8, 8), torch.randn(8, 8))
    assert len(path.read_text().splitlines()) == 2


def test_logger_limits_sampled_parameters(tmp_path):
    path = tmp_path / "s.jsonl"
    with SVDLogger(path, every_k=1, max_params=2) as log:
        for index in range(10):
            log(0, index, torch.randn(8, 8), torch.randn(8, 8))
    assert len(path.read_text().splitlines()) == 4  # 2 params x 2 phases


def test_logger_parameter_choice_is_seed_reproducible(tmp_path):
    def run(name):
        path = tmp_path / name
        with SVDLogger(path, every_k=1, max_params=2, seed=7) as log:
            for index in range(10):
                log(0, index, torch.randn(8, 8), torch.randn(8, 8))
        return [json.loads(l)["param_index"] for l in path.read_text().splitlines()]

    assert run("a.jsonl") == run("b.jsonl")


def test_logger_integrates_with_optimizer_as_a_hook(tmp_path):
    """The logger must plug into ParameterizedMuon with no adapter."""
    path = tmp_path / "s.jsonl"
    torch.manual_seed(0)
    p = torch.randn(16, 16, requires_grad=True)
    p.grad = torch.randn(16, 16)
    with SVDLogger(path, every_k=1) as log:
        ParameterizedMuon([p], lr=0.05, coeff_set="keller", hook=log).step()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 2
    before = next(r for r in records if r["phase"] == "before")
    after = next(r for r in records if r["phase"] == "after")
    assert after["band_ratio"] < before["band_ratio"]
```

The last assertion is the diagnostic's whole purpose: orthogonalization must
measurably compress the spectrum.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_svd_logger.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'nsshape.diagnostics'`

- [ ] **Step 3: Write `src/nsshape/diagnostics/__init__.py`**

```python
"""Spectral diagnostics for the orthogonalization step."""
```

- [ ] **Step 4: Write `src/nsshape/diagnostics/schema.py`**

```python
"""Record schema for logged singular-value spectra."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import torch

QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class SpectrumRecord:
    step: int
    param_index: int
    phase: str
    shape: list[int]
    quantiles: dict[str, float]
    band_ratio: float
    histogram: list[int]
    bin_edges: list[float]
    n_singular_values: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> SpectrumRecord:
        return cls(**json.loads(line))


def spectrum_record(
    step: int,
    param_index: int,
    phase: str,
    M: torch.Tensor,
    n_bins: int = 32,
) -> SpectrumRecord:
    """Run an explicit fp32 SVD and summarize the spectrum.

    fp32 is deliberate: the training step may run in bf16, but the diagnostic
    must be trustworthy enough to prove to a reviewer what the polynomial did.
    """
    s = torch.linalg.svdvals(M.detach().float())
    nonzero = s[s > 0]
    band_ratio = (
        float(nonzero.max() / nonzero.min()) if nonzero.numel() > 0 else float("inf")
    )
    quantile_values = torch.quantile(s, torch.tensor(QUANTILES))
    counts, edges = torch.histogram(s, bins=n_bins)
    return SpectrumRecord(
        step=step,
        param_index=param_index,
        phase=phase,
        shape=list(M.shape),
        quantiles={
            f"p{int(q * 100)}": float(v) for q, v in zip(QUANTILES, quantile_values)
        },
        band_ratio=band_ratio,
        histogram=[int(c) for c in counts],
        bin_edges=[float(e) for e in edges],
        n_singular_values=int(s.numel()),
    )
```

- [ ] **Step 5: Write `src/nsshape/diagnostics/svd_logger.py`**

```python
"""Sampling hook that logs singular-value spectra before and after the step.

Attaches to ParameterizedMuon as a plain callable, so the same logger works for
cifar10-airbench and modded-nanogpt without an adapter.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch

from nsshape.diagnostics.schema import spectrum_record


class SVDLogger:
    """Log fp32 spectra of a sampled subset of momentum matrices every k steps.

    Args:
        path: JSONL output file.
        every_k: sample on steps where step % every_k == 0.
        max_params: at most this many distinct parameters per sampled step.
        n_bins: histogram resolution.
        seed: controls which parameters get sampled, for reproducibility.
    """

    def __init__(
        self,
        path: str | Path,
        every_k: int = 50,
        max_params: int = 4,
        n_bins: int = 32,
        seed: int = 0,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.every_k = every_k
        self.max_params = max_params
        self.n_bins = n_bins
        self._rng = random.Random(seed)
        self._handle = self.path.open("w")
        self._sampled_this_step = 0
        self._current_step: int | None = None

    def should_sample(self, step: int) -> bool:
        return step % self.every_k == 0

    def __call__(
        self,
        step: int,
        param_index: int,
        M_before: torch.Tensor,
        M_after: torch.Tensor,
    ) -> None:
        if not self.should_sample(step):
            return
        if step != self._current_step:
            self._current_step = step
            self._sampled_this_step = 0
        if self._sampled_this_step >= self.max_params:
            return
        self._sampled_this_step += 1

        for phase, M in (("before", M_before), ("after", M_after)):
            record = spectrum_record(step, param_index, phase, M, self.n_bins)
            self._handle.write(record.to_json() + "\n")
        self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> SVDLogger:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_svd_logger.py -q`
Expected: 13 passed

- [ ] **Step 7: Run the whole suite**

Run: `uv run pytest -q -m "not slow"`
Expected: all pass

- [ ] **Step 8: Commit**

```bash
git add src/nsshape/diagnostics/ tests/test_svd_logger.py
git commit -m "feat: fp32 SVD spectral logger"
```

---

### Task 9: Diagnostic plots and README

**Files:**
- Create: `scripts/plot_composed_maps.py`, `README.md`

**Interfaces:**
- Consumes: `load_table` from Task 4, `compose` from Task 1.
- Produces: nothing importable.

- [ ] **Step 1: Write `scripts/plot_composed_maps.py`**

```python
"""Plot the composed map C(x) for every entry in the coefficient table.

Usage:  uv run python scripts/plot_composed_maps.py [--out figures/]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nsshape.coefficients import load_table
from nsshape.polynomial import compose

XS = np.logspace(-4, 0, 1200)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    table = load_table()
    entries = sorted(
        ((n, cs) for n, cs in table.items() if cs.coeffs is not None),
        key=lambda kv: kv[1].gamma,
    )

    fig, (ax_map, ax_frontier) = plt.subplots(1, 2, figsize=(13, 5))

    for name, cs in entries:
        style = "--" if cs.kind == "baseline" else "-"
        width = 2.5 if cs.kind == "baseline" else 1.3
        ax_map.semilogx(
            XS,
            compose(cs.coeffs, XS, cs.n_steps) / cs.scale,
            style,
            lw=width,
            label=f"{name} (γ={cs.gamma:.2f})",
        )
    ax_map.axhline(1.0, color="k", lw=0.6, alpha=0.4)
    ax_map.set_xlabel("input singular value σ")
    ax_map.set_ylabel("C(σ) / scale")
    ax_map.set_title("Composed 5-step map, normalized to band center 1")
    ax_map.legend(fontsize=7, ncol=2)

    fitted = [(n, cs) for n, cs in entries if cs.kind == "fitted"]
    ax_frontier.loglog(
        [cs.gamma for _, cs in fitted],
        [cs.sigma_min for _, cs in fitted],
        "o-",
        label="fitted frontier",
    )
    for name, cs in entries:
        if cs.kind == "baseline":
            ax_frontier.loglog(cs.gamma, cs.sigma_min, "r*", ms=16, label=name)
    ax_frontier.set_xlabel("band half-width γ")
    ax_frontier.set_ylabel("σ_min reached")
    ax_frontier.set_title("Dynamic range vs band width")
    ax_frontier.legend(fontsize=8)

    fig.tight_layout()
    out = args.out / "composed_maps.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate the figure**

Run: `uv run python scripts/plot_composed_maps.py`
Expected: `wrote figures/composed_maps.png`. Inspect it: the Keller star should
sit just above the fitted frontier line at γ = 0.30.

- [ ] **Step 3: Write `README.md`**

Cover: what the project tests (H1), the γ family formulation, how to install
(`uv venv --python 3.12 && uv pip install -e ".[dev]"`), how to regenerate the
table, how to run tests (`-m "not slow"` for the fast suite), and the layout
table from this plan's File Structure section. State the three load-bearing
constraints — relative band, relaxed invariance bound, positivity not
monotonicity — since each is a trap that looks wrong until you hit it.

- [ ] **Step 4: Add `figures/` to `.gitignore`**

```bash
echo "figures/" >> .gitignore
```

- [ ] **Step 5: Commit**

```bash
git add scripts/plot_composed_maps.py README.md .gitignore
git commit -m "feat: diagnostic plots and README"
```

---

## Deferred to the stage 3 spec

Sweep orchestration, the Colab notebook, and the cifar10-airbench fork. All three
depend on the coefficient table existing, which is this plan's output.
