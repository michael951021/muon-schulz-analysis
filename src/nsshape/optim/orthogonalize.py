"""Two interchangeable orthogonalization backends behind one signature.

ns_quintic is the practical Newton-Schulz step in bf16; exact_svd_polar is the
gamma=0 anchor in fp32. Because they share a signature, the SVD arm needs no
special-casing anywhere in the training loop.
"""

from __future__ import annotations

import torch

from nsshape.coefficients import CoefficientSet
from nsshape.polynomial import Coeffs


def _require_2d(G: torch.Tensor) -> None:
    if G.ndim != 2:
        raise ValueError(f"expected a 2D matrix, got shape {tuple(G.shape)}")


def ns_quintic(
    G: torch.Tensor,
    coeffs: Coeffs,
    n_steps: int = 5,
    scale: float = 1.0,
    eps: float = 1e-7,
    precision: str = "bf16",
) -> torch.Tensor:
    """Newton-Schulz quintic iteration, Muon-style.

    Operates on the short side by transposing tall inputs, which keeps the Gram
    matrix as small as possible. Runs in bf16 -- the low precision is part of
    what the ablation is testing, so it is deliberate, not incidental.

    The output is divided by `scale` so the singular-value band is centered on
    1, which decouples the band shape from the learning rate.
    """
    _require_2d(G)
    if precision not in {"bf16", "fp32"}:
        raise ValueError("precision must be bf16 or fp32")
    if n_steps < 1 or scale <= 0:
        raise ValueError("n_steps and scale must be positive")
    a, b, c = coeffs
    X = G.to(torch.bfloat16 if precision == "bf16" else torch.float32)
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
    """Canonical partial polar factor; discard numerical null-space directions.

    The rank tolerance is the conventional fp32 matrix-rank tolerance. Zero
    input returns zero, matching polynomial backends rather than an arbitrary
    orthogonal completion of the null space.
    """
    _require_2d(G)
    U, s, Vh = torch.linalg.svd(G.float(), full_matrices=False)
    tolerance = max(G.shape) * torch.finfo(torch.float32).eps * s.max()
    return ((U * (s > tolerance).to(U.dtype)) @ Vh).to(G.dtype)


def orthogonalize(G: torch.Tensor, cs: CoefficientSet, precision: str = "bf16") -> torch.Tensor:
    """Dispatch to the backend named by a coefficient set.

    Shape validation lives in the backends, so both arms reject the same inputs.
    """
    if cs.is_exact_svd:
        return exact_svd_polar(G)
    return ns_quintic(G, cs.coeffs, cs.n_steps, cs.scale, precision=precision)
