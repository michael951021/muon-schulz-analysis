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
    first_critical_point,
    invariance_violation,
    positivity_violation,
    reachable_bound,
)

TABLE = load_table()
POLYNOMIALS = [(n, cs) for n, cs in TABLE.items() if cs.coeffs is not None]
FITTED = [(n, cs) for n, cs in POLYNOMIALS if cs.kind == "fitted"]


@pytest.mark.parametrize("name,cs", POLYNOMIALS)
def test_entry_iteration_is_bounded(name, cs):
    B = reachable_bound(cs.coeffs)
    assert np.isfinite(B)
    assert B == pytest.approx(cs.invariance_bound, rel=1e-6)


@pytest.mark.parametrize("name,cs", POLYNOMIALS)
def test_entry_is_forward_invariant_at_its_bound(name, cs):
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
    """Spec acceptance test: standard Muon is an interior point, with headroom."""
    keller = TABLE["keller"]
    fitted_at_030 = TABLE["gamma_0.30"]
    assert keller.sigma_min > fitted_at_030.sigma_min
    assert keller.sigma_min < fitted_at_030.sigma_min * 2.5


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
    assert min(gammas) <= 0.01
    assert max(gammas) >= 0.60


def test_tight_bands_cover_less_input_range():
    """The core tradeoff: accuracy costs dynamic range."""
    assert TABLE["gamma_0.01"].sigma_min > TABLE["gamma_0.60"].sigma_min * 10
