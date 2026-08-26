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
