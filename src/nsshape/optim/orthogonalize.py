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

    Operates on the short side by transposing tall inputs, which keeps the Gram
    matrix as small as possible. Runs in bf16 -- the low precision is part of
    what the ablation is testing, so it is deliberate, not incidental.

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
