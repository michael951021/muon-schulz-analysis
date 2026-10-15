# nsshape — Newton–Schulz polynomial shaping for Muon

Does the Muon optimizer work well *because* its Newton–Schulz iteration
approximates the polar factor, or because of the particular **shape** it imposes
on the singular-value spectrum?

Muon's quintic was tuned for speed, not accuracy. It does not converge to the
polar factor at all — it drives singular values into a wide band. **H1:** that
band shape is the operative quantity, and shapes other than "as close to exact
polar as possible" may train better. An unsuccessful comparison would bound
this tested family and training budget; it would not establish universal
optimality of geometric alignment.

This repository contains the coefficient family, instrumented optimizer, and a
budget-calibrated CIFAR/transformer study launcher. The full GPU study has not
been run in this checkout. See [the two-3090 protocol](docs/dual3090-study.md) for
an executable 6–10-hour plan and [local smoke results](reports/local_smoke/results.md)
for CPU training checks. Smoke results do not answer H1.

## The γ family

One-parameter family of odd quintics `P(x) = ax + bx³ + cx⁵`, composed **N = 5**
times so every member is exactly compute-matched. For each γ:

```
minimize    σ_min
over        a, b, c
subject to  relhw( C([σ_min, 1]) ) ≤ γ            (band)
            reachable_bound(P) < ∞                 (bounded iteration)
            P(x) > 0  on (0, B]                    (positivity)

where       relhw(S) = (max S − min S) / (max S + min S)
            C = P∘P∘P∘P∘P
```

γ is the band half-width; σ_min is the smallest input singular value the map
still pulls into the band. A wider band buys the ability to lift near-zero
singular values within a fixed step budget — that is the real tradeoff Muon
makes, and the axis this ablation sweeps.

## Three constraints that are load-bearing

Each of these looks like an unnecessary complication until you hit the failure it
prevents. All three were established empirically; the naive version fails.

**The band is relative, not absolute.** A polynomial emitting every singular
value at 0.7 is the *same optimizer* as one emitting 1.0 with a 1.43× smaller
learning rate — the scale is absorbed by η. Using absolute deviation from 1
confounds γ with η and turns the 2D sweep into a diagonal. Relative half-width is
scale-invariant by construction. Each entry stores the band center as `scale`,
applied as a divisor at use time.

**Boundedness is a reachable-set fixed point.** The constraint is that iterating
`B ← max(1, sup_[0,B] P)` from `B = 1` converges. Tempting alternatives reject
Muon itself: `sup_[0,1+γ] P ≤ 1+γ` gives B = 1.3 for Keller while P(1.3) = 1.53,
so it self-rejects — yet Keller's actual reachable set is bounded at 1.2024 and
it is a working optimizer. Any constraint that excludes Keller is the wrong
constraint. The fixed point is also γ-independent, which is correct: boundedness
has nothing to do with band width.

**Positivity, not monotonicity.** Keller's quintic is monotone only up to
x\* = 0.5545; the committed fitted members have x\* ≈ 0.393–0.644. The
non-monotonicity is essential — it is how the polynomial squashes large singular
values back down. Requiring monotonicity on [0,1] excludes the entire useful
family. `P′ > 0 on [0, x*]` is vacuous, being the definition of x\*. What
actually binds is positivity on (0, B]; x\* is recorded as a diagnostic.

## What the table shows

Nine fitted entries span γ ∈ [0.01, 0.60] — hyper-accurate polar through standard
Muon to deliberately distorted — plus the `keller` baseline and an `exact_svd`
sentinel. The frontier is strictly monotone: tighter bands cover less input
range.

Keller's quintic sits **inside** the frontier at its own achieved band γ ≈ 0.30,
reaching σ_min = 1.42e−3 against a fitted optimum of 9.04e−4. So standard Muon is
a genuine interior member of the family with roughly 57% headroom in dynamic
range — which is exactly the slack H1 predicts might be exploitable.

Note that `exact_svd` is not a γ → 0 limit of the polynomials but a separate
backend with a different measured cost. Its canonical partial polar factor
preserves zero and excludes numerical null-space directions; full-rank spectra
have zero band width.

## A caveat when reading spectra

Muon Frobenius-normalizes before iterating, so for an n×n matrix σ_max lands near
1/√n and the smallest singular values fall *below* the certified σ_min. Real
inputs therefore occupy a compressed sub-range of the certified domain [σ_min, 1].
Band *width* still holds on any subinterval — an image subset can only be
narrower — but the band *center* equals 1 only over the full range. Singular
values below σ_min are not pulled into the band at all, which is precisely what
σ_min certifies. Use controlled spectra, not plain Gaussians, when probing this.

## Install

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

## Use

```python
from nsshape.optim.muon import ParameterizedMuon
from nsshape.diagnostics.svd_logger import SVDLogger

logger = SVDLogger("runs/spectra.jsonl", every_k=50, max_params=4)
opt = ParameterizedMuon(matrix_params, lr=0.02, coeff_set="gamma_0.20", hook=logger)
```

`coeff_set` is any entry name in the table: `gamma_0.01` … `gamma_0.60`,
`keller`, or `exact_svd`. Pass only 2D parameters; embeddings, biases and norms
belong in a separate AdamW group, as in standard Muon.

## Tests

```bash
uv run pytest -m "not slow"   # fast suite, ~2s
uv run pytest                 # includes solver fits, ~2min
```

The table tests avoid the solver. `tests/test_research_integrity.py` additionally
checks continuous image bounds through independently computed polynomial extrema.
Production certificates use analytic interval extrema in float64; input threshold
resolution remains grid-limited. These numerical checks do not prove a globally
optimal frontier or predict training performance.

## Training study on two 3090s

After the installation and dataset-preparation steps in
[the protocol](docs/dual3090-study.md), run:

```bash
bash scripts/launch_dual3090.sh 8 runs/dual3090
```

The launcher calibrates the actual GPUs, runs two independent jobs concurrently,
selects configurations on validation data, confirms on fresh seeds, and generates
`reports/dual3090/results.md`. The approximately 124M GPT confirmation uses a
matched, calibrated token budget and is shorter than the original 2–5B-token
proposal. Installation, data preparation, preflight, and tests precede the timed
training budget.

`ParameterizedMuon(..., precision="fp32")` isolates the intended polynomial
shape; default `precision="bf16"` retains the practical Muon implementation.
The study tests both, because scalar band certificates do not certify bf16
matrix arithmetic. Checkpoints include exact coefficients, precision, and step.
For uniform diagnostic sampling, pass `population_size` and `coefficient_set` to
`SVDLogger`; without a population size it logs the first parameters explicitly.

## Regenerating artifacts

```bash
uv run python scripts/generate_coefficients.py   # rewrites data/coefficients/v1.json
uv run python scripts/plot_composed_maps.py      # writes figures/composed_maps.png
```

The solver is stochastic and restarts across five seeds, keeping the best
feasible result. A single run is not reliable — one seed converged to an
infeasible local optimum at γ = 0.50 while succeeding at 0.40 and 0.60 from
identical settings. If `test_frontier_is_monotone_in_gamma` fails after
regenerating, a γ hit a bad optimum; rerun with more seeds.

## Layout

| Path | Responsibility |
|---|---|
| `src/nsshape/polynomial.py` | Evaluate, compose, analyze quintics. Pure numpy, reference truth. |
| `src/nsshape/solve.py` | Constrained fit γ → coefficients + certificate. scipy. |
| `src/nsshape/coefficients.py` | Load/validate the committed table; resolve entries by name. |
| `src/nsshape/optim/orthogonalize.py` | `ns_quintic` (bf16) and `exact_svd_polar` (fp32) backends. |
| `src/nsshape/optim/muon.py` | `ParameterizedMuon` optimizer. |
| `src/nsshape/diagnostics/` | fp32 SVD spectral logger and record schema. |
| `data/coefficients/v1.json` | Generated, committed artifact. |

`polynomial.py` and `solve.py` never import torch; `optim/` and `diagnostics/`
never import scipy. The optimizer takes a coefficient-set *name* and the logger
takes a *hook*, so neither knows what model it is training — which is what lets
the same code drive both cifar10-airbench and modded-nanogpt.

## Design docs

- `docs/superpowers/specs/2026-08-26-ns-polynomial-ablation-design.md`
- `docs/superpowers/plans/2026-08-26-ns-polynomial-shaping.md`
