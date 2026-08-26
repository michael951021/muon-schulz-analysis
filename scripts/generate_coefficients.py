"""Generate the committed coefficient table. Run once; commit the output.

Usage:  uv run python scripts/generate_coefficients.py
"""

from __future__ import annotations

import json
from datetime import date

from nsshape.coefficients import EXACT_SVD_NAME, TABLE_PATH
from nsshape.polynomial import (
    KELLER,
    band_extent,
    first_critical_point,
    invariance_violation,
    positivity_violation,
    reachable_bound,
)
from nsshape.solve import fit_gamma

#: Spans hyper-accurate polar (0.01) -> standard Muon (~0.30) -> distorted (0.60).
GAMMA_GRID = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60]
SEEDS = (0, 1, 2, 3, 4)

#: Keller's own achieved relative half-width, measured not assumed.
KELLER_GAMMA = 0.30


def _entry_from_fit(name: str, kind: str, fit) -> dict:
    return {
        "name": name,
        "kind": kind,
        "gamma": fit.gamma,
        "coeffs": list(fit.coeffs),
        "n_steps": fit.n_steps,
        "scale": fit.scale,
        "sigma_min": fit.sigma_min,
        "achieved_half_width": fit.achieved_half_width,
        "critical_point": fit.critical_point,
        "invariance_bound": fit.invariance_bound,
        "invariance_margin": fit.invariance_margin,
        "positivity_margin": fit.positivity_margin,
        "seed": fit.seed,
    }


def main() -> None:
    entries = []

    for gamma in GAMMA_GRID:
        fit = fit_gamma(gamma, seeds=SEEDS)
        if not fit.feasible:
            raise SystemExit(f"gamma={gamma} infeasible; adjust GAMMA_GRID")
        entries.append(_entry_from_fit(f"gamma_{gamma:.2f}", "fitted", fit))
        print(
            f"gamma={gamma:.2f}  coeffs=("
            f"{fit.coeffs[0]:.4f}, {fit.coeffs[1]:.4f}, {fit.coeffs[2]:.4f})"
            f"  sigma_min={fit.sigma_min:.3e}"
        )

    # Keller baseline: measured, not fitted.
    B = reachable_bound(KELLER)
    extent = band_extent(KELLER, KELLER_GAMMA)
    entries.append(
        {
            "name": "keller",
            "kind": "baseline",
            "gamma": KELLER_GAMMA,
            "coeffs": list(KELLER),
            "n_steps": 5,
            "scale": extent.scale,
            "sigma_min": extent.sigma_min,
            "achieved_half_width": extent.achieved_half_width,
            "critical_point": first_critical_point(KELLER),
            "invariance_bound": B,
            "invariance_margin": -invariance_violation(KELLER, B),
            "positivity_margin": -positivity_violation(KELLER, B),
            "seed": None,
        }
    )
    print(f"keller     coeffs={KELLER}  sigma_min={extent.sigma_min:.3e}")

    entries.append(
        {
            "name": EXACT_SVD_NAME,
            "kind": "exact_svd",
            "gamma": 0.0,
            "coeffs": None,
            "n_steps": 5,
            "scale": 1.0,
            "sigma_min": 0.0,
            "achieved_half_width": 0.0,
            "critical_point": float("inf"),
            "invariance_bound": 1.0,
            "invariance_margin": float("inf"),
            "positivity_margin": float("inf"),
            "seed": None,
        }
    )

    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(
        json.dumps(
            {
                "version": 1,
                "generated": date.today().isoformat(),
                "n_steps": 5,
                "sigma_grid": "logspace(-7, 0, 3000)",
                "seeds": list(SEEDS),
                "entries": entries,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {len(entries)} entries to {TABLE_PATH}")


if __name__ == "__main__":
    main()
