"""Load and validate the committed coefficient table.

The table is a generated artifact produced once, offline, by
scripts/generate_coefficients.py. Training code reads it and never runs the
solver, so every run is traceable to exact coefficients by name.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from nsshape.polynomial import Coeffs

TABLE_PATH = Path(__file__).resolve().parents[2] / "data" / "coefficients" / "v1.json"
if not TABLE_PATH.exists():
    TABLE_PATH = Path(__file__).resolve().parent / "data" / "coefficients" / "v1.json"

#: Sentinel name for the exact-SVD polar arm (not a polynomial).
EXACT_SVD_NAME = "exact_svd"


@dataclass(frozen=True)
class CoefficientSet:
    """One named entry: a polynomial plus its verified certificate.

    `scale` is the divisor that centers the output band on 1, which is what
    keeps the band shape independent of the learning rate.
    """

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
        critical_point=entry["critical_point"] if entry["critical_point"] is not None else math.inf,
        invariance_bound=entry["invariance_bound"],
        invariance_margin=entry["invariance_margin"] if entry["invariance_margin"] is not None else math.inf,
        positivity_margin=entry["positivity_margin"] if entry["positivity_margin"] is not None else math.inf,
    )


def load_table(path: Path | None = None) -> dict[str, CoefficientSet]:
    """Read the committed table, keyed by entry name."""
    path = path or TABLE_PATH
    payload = json.loads(path.read_text())
    if payload.get("version") != 1:
        raise ValueError("unsupported coefficient table version")
    table = {}
    for entry in payload["entries"]:
        cs = _parse(entry)
        if cs.name in table:
            raise ValueError(f"duplicate coefficient set {cs.name}")
        if cs.kind not in {"fitted", "baseline", "exact_svd"}:
            raise ValueError(f"invalid kind for {cs.name}")
        if not math.isfinite(cs.scale) or cs.scale <= 0 or cs.n_steps < 1:
            raise ValueError(f"invalid scale or steps for {cs.name}")
        if not cs.is_exact_svd:
            if not all(math.isfinite(value) for value in (cs.sigma_min, cs.invariance_bound,
                        cs.invariance_margin, cs.positivity_margin, cs.critical_point)):
                raise ValueError(f"nonfinite certificate for {cs.name}")
            if cs.coeffs is None or len(cs.coeffs) != 3 or not all(map(math.isfinite, cs.coeffs)):
                raise ValueError(f"invalid coefficients for {cs.name}")
            if cs.gamma is None or not 0 <= cs.achieved_half_width <= cs.gamma < 1:
                raise ValueError(f"invalid band for {cs.name}")
            if not 0 < cs.sigma_min < 1 or not math.isfinite(cs.invariance_bound):
                raise ValueError(f"invalid certificate for {cs.name}")
            if cs.invariance_margin < -1e-9 or cs.positivity_margin <= 0:
                raise ValueError(f"infeasible certificate for {cs.name}")
        elif cs.coeffs is not None:
            raise ValueError("exact_svd must not have coefficients")
        table[cs.name] = cs
    return table


def get(name: str, path: Path | None = None) -> CoefficientSet:
    """Resolve one entry by name."""
    table = load_table(path)
    if name not in table:
        raise KeyError(f"unknown coefficient set {name!r}; have {sorted(table)}")
    return table[name]
