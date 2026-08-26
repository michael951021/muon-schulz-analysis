"""Record schema for logged singular-value spectra."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import torch

QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class SpectrumRecord:
    """One fp32 SVD observation of a matrix, before or after orthogonalization.

    band_ratio is max(sigma)/min(sigma) over nonzero singular values -- the
    direct empirical counterpart of the band the polynomial should produce.
    """

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
