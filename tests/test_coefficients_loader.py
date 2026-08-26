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
    assert gammas[0] == pytest.approx(0.01)
    assert gammas[-1] == pytest.approx(0.60)
    assert len(gammas) == 9


def test_get_rejects_unknown_name():
    with pytest.raises(KeyError, match="unknown coefficient set"):
        get("does_not_exist")


def test_entries_are_frozen():
    with pytest.raises(Exception):
        get("keller").name = "mutated"


def test_returns_coefficient_set_instances():
    assert all(isinstance(cs, CoefficientSet) for cs in load_table().values())
