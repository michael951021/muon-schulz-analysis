"""Load and validate the committed coefficient table.

The table is a generated artifact produced once, offline, by
scripts/generate_coefficients.py. Training code reads it and never runs the
solver, so every run is traceable to exact coefficients by name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from nsshape.polynomial import Coeffs

TABLE_PATH = Path(__file__).resolve().parents[2] / "data" / "coefficients" / "v1.json"

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
        critical_point=entry["critical_point"],
        invariance_bound=entry["invariance_bound"],
        invariance_margin=entry["invariance_margin"],
        positivity_margin=entry["positivity_margin"],
    )


def load_table(path: Path | None = None) -> dict[str, CoefficientSet]:
    """Read the committed table, keyed by entry name."""
    path = path or TABLE_PATH
    payload = json.loads(path.read_text())
    return {e["name"]: _parse(e) for e in payload["entries"]}


def get(name: str, path: Path | None = None) -> CoefficientSet:
    """Resolve one entry by name."""
    table = load_table(path)
    if name not in table:
        raise KeyError(f"unknown coefficient set {name!r}; have {sorted(table)}")
    return table[name]
