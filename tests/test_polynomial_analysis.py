import numpy as np
import pytest
from nsshape.polynomial import (
    KELLER,
    band_extent,
    compose,
    derivative,
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
    gamma = 0.30
    r = band_extent(KELLER, gamma)
    xs = np.logspace(np.log10(r.sigma_min), 0.0, 500)
    img = compose(KELLER, xs)
    relhw = (img.max() - img.min()) / (img.max() + img.min())
    assert relhw <= gamma + 1e-3


def test_band_extent_scale_is_image_midpoint():
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
