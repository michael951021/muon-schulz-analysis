import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nsshape.coefficients import get
from nsshape.optim.orthogonalize import exact_svd_polar, ns_quintic, orthogonalize

#: bf16 carries an 8-bit mantissa, so ~0.4% relative error compounds over the
#: five steps. Every tolerance here is set by that, not by the polynomial.
BF16_ATOL = 0.05


def _singular_values(M):
    return torch.linalg.svdvals(M.float())


def _controlled_matrix(n, sigma_lo, sigma_hi, seed=1):
    """Matrix whose Frobenius-normalized singular values span [sigma_lo, sigma_hi].

    Needed because a plain Gaussian is not a valid probe: Frobenius
    normalization caps its largest singular value near 1/sqrt(n) and its
    smallest falls below the certified sigma_min, so it lands outside the
    domain the coefficients were fitted for.
    """
    g = torch.Generator().manual_seed(seed)
    U, _ = torch.linalg.qr(torch.randn(n, n, generator=g))
    V, _ = torch.linalg.qr(torch.randn(n, n, generator=g))
    s = torch.logspace(np.log10(sigma_lo), np.log10(sigma_hi), n)
    s = s / s.norm()
    return U @ torch.diag(s) @ V.T, s


def test_exact_svd_polar_is_orthogonal():
    torch.manual_seed(0)
    s = _singular_values(exact_svd_polar(torch.randn(32, 32)))
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


@pytest.mark.parametrize(
    "name", ["gamma_0.05", "gamma_0.20", "keller", "gamma_0.60"]
)
def test_ns_quintic_drives_singular_values_into_the_certified_band(name):
    """Inputs inside the certified range must come out with relhw <= gamma."""
    cs = get(name)
    M, s_in = _controlled_matrix(64, cs.sigma_min * 3, 1.0)
    assert s_in.min() >= cs.sigma_min, "probe must stay inside the certified range"
    s = _singular_values(ns_quintic(M, cs.coeffs, scale=cs.scale))
    relhw = ((s.max() - s.min()) / (s.max() + s.min())).item()
    assert relhw <= cs.gamma + BF16_ATOL


@pytest.mark.parametrize("name", ["gamma_0.05", "gamma_0.20", "keller"])
def test_normalized_output_lands_within_one_plus_minus_gamma(name):
    """The scale divisor puts the band on [1-gamma, 1+gamma]."""
    cs = get(name)
    M, _ = _controlled_matrix(64, cs.sigma_min * 3, 1.0)
    s = _singular_values(ns_quintic(M, cs.coeffs, scale=cs.scale))
    assert s.min().item() >= 1.0 - cs.gamma - BF16_ATOL
    assert s.max().item() <= 1.0 + cs.gamma + BF16_ATOL


def test_singular_values_below_sigma_min_are_not_pulled_into_the_band():
    """Documents the domain limit. Frobenius normalization means a plain
    Gaussian's smallest singular values fall below the certified sigma_min, and
    those are left behind -- which is exactly what sigma_min certifies."""
    torch.manual_seed(3)
    cs = get("keller")
    G = torch.randn(64, 64)
    s_in = _singular_values(G) / G.float().norm()
    assert s_in.min() < cs.sigma_min, "this probe is meant to be out of range"
    s = _singular_values(ns_quintic(G, cs.coeffs, scale=cs.scale))
    assert s.min().item() < 1.0 - cs.gamma


def test_frobenius_normalization_caps_the_largest_singular_value():
    """sigma_max/||G||_F ~ 1/sqrt(n) for a well-conditioned n x n matrix, so
    real inputs occupy a compressed sub-range of the certified domain."""
    n = 64
    Q, _ = torch.linalg.qr(torch.randn(n, n))
    s = _singular_values(Q) / Q.float().norm()
    assert s.max().item() == pytest.approx(1.0 / np.sqrt(n), rel=1e-3)


def test_ns_quintic_roughly_agrees_with_exact_polar():
    """On a well-conditioned matrix the two backends should align in direction."""
    torch.manual_seed(4)
    cs = get("keller")
    G = torch.randn(48, 48)
    approx = ns_quintic(G, cs.coeffs, scale=cs.scale).float()
    exact = exact_svd_polar(G).float()
    cosine = (approx * exact).sum() / (approx.norm() * exact.norm())
    assert cosine > 0.95


def test_tight_gamma_agrees_with_exact_polar_more_closely_than_wide_gamma():
    """The whole premise: smaller gamma really is closer to the polar factor."""
    torch.manual_seed(8)
    G = torch.randn(48, 48)
    exact = exact_svd_polar(G).float()

    def cosine(name):
        cs = get(name)
        approx = ns_quintic(G, cs.coeffs, scale=cs.scale).float()
        return ((approx * exact).sum() / (approx.norm() * exact.norm())).item()

    assert cosine("gamma_0.01") > cosine("gamma_0.60")


def test_ns_quintic_is_scale_invariant_in_input():
    """Frobenius normalization means input scaling must not change the output."""
    torch.manual_seed(5)
    cs = get("keller")
    G = torch.randn(24, 24)
    a = ns_quintic(G, cs.coeffs, scale=cs.scale)
    b = ns_quintic(G * 100.0, cs.coeffs, scale=cs.scale)
    assert torch.allclose(a.float(), b.float(), atol=BF16_ATOL)


def test_ns_quintic_handles_rank_deficient_input():
    G = torch.zeros(16, 16)
    G[0, 0] = 1.0
    assert torch.all(torch.isfinite(ns_quintic(G, get("keller").coeffs)))


def test_orthogonalize_dispatches_to_svd_for_sentinel():
    torch.manual_seed(6)
    G = torch.randn(16, 16)
    assert torch.allclose(
        orthogonalize(G, get("exact_svd")), exact_svd_polar(G), atol=1e-5
    )


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
