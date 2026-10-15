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
    # The reachable bound is a fixed point found to ~1e-12, so the invariance
    # residual sits at machine epsilon rather than exactly zero.
    assert r.invariance_margin >= -1e-9
    assert r.positivity_margin >= 0


def test_fit_beats_keller_at_gamma_030():
    """The fitted optimum must reach further than Keller at the same band."""
    r = fit_gamma(0.30, seeds=(0, 1, 2))
    assert r.sigma_min < band_extent(KELLER, 0.30).sigma_min


def test_keller_lies_inside_the_frontier_with_measurable_headroom():
    """Spec acceptance test. Keller is an interior point of the family, but the
    fitted optimum at its own band reaches ~57% further in dynamic range. That
    headroom is the thing H1 predicts is exploitable."""
    r = fit_gamma(0.30, seeds=(0, 1, 2))
    keller_sigma_min = band_extent(KELLER, 0.30).sigma_min
    ratio = keller_sigma_min / r.sigma_min
    assert 1.2 < ratio < 2.5, ratio


def test_wider_gamma_reaches_smaller_sigma_min():
    """The frontier must be monotone -- a wider band buys dynamic range."""
    tight = fit_gamma(0.20, seeds=(0, 1, 2))
    wide = fit_gamma(0.40, seeds=(0, 1, 2))
    assert wide.sigma_min < tight.sigma_min


def test_tight_bands_are_feasible_but_lose_dynamic_range():
    """Under the RELATIVE band, tight gammas are reachable in 5 steps -- they
    just cover far less input range. (An earlier absolute-band formulation made
    gamma=0.05 look infeasible; that conclusion did not survive the change.)"""
    tight = fit_gamma(0.05, seeds=(0, 1, 2))
    assert tight.feasible
    assert tight.sigma_min > fit_gamma(0.30, seeds=(0, 1, 2)).sigma_min


def test_fitted_polynomials_have_bounded_iteration():
    r = fit_gamma(0.30, seeds=(0, 1, 2))
    assert np.isfinite(r.invariance_bound)


def test_multi_seed_recovers_gamma_050():
    """Single-seed DE fell into an infeasible local optimum at gamma=0.50."""
    assert fit_gamma(0.50, seeds=(0, 1, 2, 3, 4)).feasible
