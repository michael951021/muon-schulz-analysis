"""Plot the composed map C(x) for the coefficient table.

Usage:  uv run python scripts/plot_composed_maps.py [--out figures/]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nsshape.coefficients import load_table
from nsshape.polynomial import compose

#: The composed map equioscillates rapidly inside the band, so it needs far more
#: samples than a smooth curve would. At low density the oscillation aliases into
#: visual noise instead of rendering as the solid band it actually is.
XS = np.logspace(-5, 0, 400_000)

#: Small multiples: tightest band, mid, standard Muon, widest band.
FEATURED = ["gamma_0.01", "gamma_0.10", "keller", "gamma_0.60"]


def _plot_one(ax, name, cs) -> None:
    image = compose(cs.coeffs, XS, cs.n_steps) / cs.scale
    ax.fill_between(
        [XS[0], XS[-1]],
        1 - cs.gamma,
        1 + cs.gamma,
        color="tab:green",
        alpha=0.15,
        label=f"target band ±{cs.gamma:.2f}",
    )
    ax.semilogx(XS, image, lw=0.4, color="tab:blue")
    ax.axvline(
        cs.sigma_min,
        color="tab:red",
        ls="--",
        lw=1.2,
        label=f"σ_min = {cs.sigma_min:.1e}",
    )
    ax.axhline(1.0, color="k", lw=0.6, alpha=0.4)
    ax.set_xlim(XS[0], XS[-1])
    ax.set_ylim(0, 1.8)
    ax.set_xlabel("input singular value σ")
    ax.set_ylabel("C(σ) / scale")
    ax.set_title(f"{name}  (γ = {cs.gamma:.2f})", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")


def _plot_frontier(ax, entries) -> None:
    fitted = [(n, cs) for n, cs in entries if cs.kind == "fitted"]
    ax.loglog(
        [cs.gamma for _, cs in fitted],
        [cs.sigma_min for _, cs in fitted],
        "o-",
        label="fitted frontier",
    )
    for name, cs in entries:
        if cs.kind == "baseline":
            ax.loglog(cs.gamma, cs.sigma_min, "r*", ms=18, label=name, zorder=5)
    ax.set_xlabel("band half-width γ")
    ax.set_ylabel("σ_min reached")
    ax.set_title("Dynamic range vs band width", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    table = load_table()
    entries = sorted(
        ((n, cs) for n, cs in table.items() if cs.coeffs is not None),
        key=lambda kv: kv[1].gamma,
    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, name in zip(axes[:, :2].ravel(), FEATURED):
        _plot_one(ax, name, table[name])

    gs = axes[0, 2].get_gridspec()
    axes[0, 2].remove()
    axes[1, 2].remove()
    _plot_frontier(fig.add_subplot(gs[:, 2]), entries)

    fig.tight_layout()
    out = args.out / "composed_maps.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
