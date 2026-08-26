# Newton–Schulz Polynomial Shaping: Design (Stages 1–2)

**Date:** 2026-08-26
**Status:** Approved
**Scope:** Stages 1 and 2 of the ablation study — the polynomial family and the
instrumented optimizer. Stages 3–5 (CIFAR grid, nanogpt confirmation, paper) get
their own specs.

## Research question

Muon orthogonalizes momentum matrices with a fixed quintic Newton–Schulz
iteration whose coefficients were tuned for speed, not for accuracy of the polar
factor. The resulting map does not converge to the polar factor at all: it drives
singular values into a wide band.

**H1:** the band *shape* is the operative quantity, and shapes other than "as
close to exact polar as possible" may train better. If H1 fails, the fallback
result is a rigorous defense of geometric alignment — that among practical NS
steps, aiming for the exact polar factor is optimal.

Either outcome is publishable; the design must not privilege one.

## Stage 1: the polynomial family

### Formulation

One-parameter family of odd quintics, with the step count pinned at N = 5 so that
every family member is exactly compute-matched:

```
P(x)  = a·x + b·x³ + c·x⁵
C(x)  = P∘P∘P∘P∘P            (N = 5)
```

For each γ, solve:

```
minimize    σ_min
over        a, b, c
subject to  relhw( C([σ_min, 1]) ) ≤ γ                   (band)
            max_{x∈[0,B]} P(x) ≤ B                       (forward invariance)
            where B = max(1+γ, sup_{[0,1]} P)
            P(x) > 0   ∀ x ∈ (0, B]                      (positivity)

where       relhw(S) = (max S − min S) / (max S + min S)
```

γ is the band half-width; σ_min is the smallest input singular value the map can
still pull into the band. The objective maximizes input dynamic range subject to
the band, which is the real tradeoff Muon makes: a wider band buys the ability to
lift near-zero singular values within a fixed step budget.

The band is expressed as the **relative** half-width of the image. This is
scale-invariant by construction, which is what decouples γ from η (see below) —
no explicit rescaling is needed during the fit. The band center is recorded per
entry and applied as a divisor when the coefficients are used.

Solved by differential evolution plus local polish over a log-spaced σ grid
(1e-7 to 1, ~3000 points), **restarted across multiple seeds keeping the best
feasible result**. A single run is not reliable: at γ = 0.50 one seed converged
to an infeasible local optimum (a = 5.78, b = −11.99) while γ = 0.40 and γ = 0.60
succeeded from the same settings.

### Three constraints that are load-bearing

These were established empirically during design; naive formulations fail.

**Output normalization (decouples γ from η).** The composed map's overall scale
is absorbed entirely by the learning rate: a polynomial emitting all singular
values at 0.7 is the same optimizer as one emitting 1.0 with a 1.43× smaller η.
Measured `C(1)` across candidate members ranges over [0.78, 1.45], and Keller's
own quintic has `C(1) = 0.696`. Left free, this makes the γ × η sweep a diagonal
rather than a grid. **Every coefficient set is therefore rescaled to band-center
1 before use**, and the ablation varies band shape only.

**Forward invariance (makes the problem well-posed).** Without
`max_{[0,B]} P ≤ B`, the solver escapes to a ≈ 8, c ≈ 12, saturating all singular
values to a common large value where the band ratio is trivially 1 — a degenerate
optimum that satisfies any band constraint while destroying the matrix. The
invariance constraint also guarantees the iteration cannot diverge.

**γ → 0 is infeasible at N = 5.** No coefficients achieve γ = 0.05: five quintic
steps cannot map a wide input range into a tight band. This is a result, not an
obstacle. Exact polar is *outside* the reachable family, so the SVD baseline is
not merely a slower family member — it is qualitatively different. The sweep
covers γ ∈ [~0.12, 0.6]; the exact-SVD arm anchors the γ = 0 end.

### Validation: standard Muon is a near-optimal interior point

Measured, at N = 5:

| source | a | b | c | achieved γ | σ_min |
|---|---|---|---|---|---|
| Keller (standard Muon) | 3.4445 | −4.7750 | 2.0315 | 0.299 | 1.42e−3 |
| fit at γ = 0.30 | 3.635 | −4.789 | 1.910 | 0.30 | 1.17e−3 |

Keller's quintic lands just **inside** the fitted frontier: at its own achieved
band of γ ≈ 0.30 it reaches σ_min = 1.42e−3 against a fitted optimum of 1.17e−3,
so it captures dynamic range within 21% of the best available at that band. The
baseline is therefore a genuine interior member of the family rather than an
outside comparison, which is what makes the γ axis a legitimate ablation. It also
independently corroborates that Keller's hand-tuned coefficients are close to
optimal for the band they target.

The acceptance test for the solver is reproducing this: **the fitted table must
place Keller inside the frontier at γ ≈ 0.30, within ~25% on σ_min.** Coefficient
proximity is not the test — nearby coefficient triples exist at several γ, and the
frontier position is the meaningful claim.

**Implementation note — positivity, not monotonicity.** Keller's quintic is
monotone only up to its first critical point x\* = 0.5545, and every fitted member
is similar (x\* ≈ 0.56–0.63). The non-monotonicity is essential: it is how the
polynomial squashes large singular values back down. Any constraint requiring
monotonicity across [0, 1] excludes standard Muon and the entire useful family.
`P′ > 0 on [0, x*]` is vacuous — it is the definition of x\*. The constraint that
actually binds is **positivity on (0, B]**, which Keller satisfies with margin
(min P = +0.0022). x\* is recorded per entry as a reported diagnostic, not a
constraint.

Keller also violates *strict* invariance on [0, 1.3] by +0.23, because its true
band is centered near 0.7 rather than 1; hence the relaxed bound
`B = max(1+γ, sup_{[0,1]} P)` above.

### Deliverable

6–10 coefficient sets spanning hyper-accurate → standard-Muon → deliberately
distorted, committed as a generated artifact with a certificate per entry
(achieved σ_min, achieved band, constraint margins).

## Stage 2: instrumented optimizer

### Modules

Five units in dependency order. All are CPU-testable; the entire mathematical
core is verifiable without a GPU.

**`nsshape.polynomial`** — pure numpy, no torch. Evaluates P, composes N-fold,
locates critical points, checks invariance and positivity, measures achieved
band and σ_min. The reference everything else is checked against.

**`nsshape.solve`** — the constrained fit. Takes (γ, N), returns coefficients
plus certificate. Depends on `polynomial`. Kept separate so the solver may be
slow and stochastic while everything downstream is deterministic.

**`nsshape.coefficients`** — loads and validates `data/coefficients/v1.json`, a
committed generated artifact. The solver runs once, offline, seeded; training
never invokes it. The table includes the fitted γ set, the Keller baseline, and
an `exact_svd` sentinel as named entries. Sweeps reference entries **by name**,
so every run record is traceable to exact coefficients.

**`nsshape.optim`** — `orthogonalize.py` exposes two interchangeable backends
behind one signature: `ns_quintic(coeffs, N)` in bf16 (the real thing) and
`exact_svd_polar()` in fp32 (the γ = 0 anchor). `muon.py` is the parameterized
optimizer, taking a coefficient-set name. Swappable backends mean the SVD arm
needs no special-casing in the training loop.

**`nsshape.diagnostics`** — the SVD logger. Samples a subset of layer momentum
matrices every k steps and runs explicit fp32 SVD **before and after** the
orthogonalization step, writing one JSONL record per observation (step, layer,
singular-value quantiles, band ratio, histogram). Isolated behind a hook
interface so it attaches to airbench and nanogpt identically — required for
stage 4, which must show transformer momenta behave like airbench ones.

### Key seam

The optimizer takes a *coefficient-set name*; the logger takes a *hook*. Neither
knows what model it is training. This is what makes stage 3 (airbench) and stage
4 (nanogpt) cheap.

### Layout

```
pyproject.toml                        # uv, package = nsshape
src/nsshape/
  polynomial.py
  solve.py
  coefficients.py
  optim/{__init__,orthogonalize,muon}.py
  diagnostics/{__init__,svd_logger,schema}.py
data/coefficients/v1.json             # generated, committed
scripts/{generate_coefficients,plot_composed_maps}.py
tests/
```

### Testing

The property tests *are* the scientific claim, so they deliberately share no code
with the solver. Given the committed table, independently re-verify for every
entry: positive on (0, B], forward-invariant on [0, B], and mapping [σ_min, 1]
into its claimed relative band. Plus `ns_quintic` vs `exact_svd_polar` agreement on
well-conditioned matrices, and a logger round-trip on a tiny model.

## Compute plan

Colab-first. The repo is `pip install -e .`-able from a notebook cell, with
results written to Drive or GCS. Stage 3's CIFAR grid (~10γ × 6η × 4 seeds at
~2.6 s/run) fits comfortably in a single Colab session on an A100.

Stage 4 does not: 124M parameters at 2–5B tokens is roughly 15–25 A100-hours and
Colab VMs are ephemeral. That stage runs on a persistent GCP Compute Engine VM
(`a2-highgpu-1g`) or a rented H100 for a day, and gets its launcher when stage 3
has selected the three configurations.

## Out of scope (deferred to stage 3)

Sweep orchestration, the Colab notebook, and the cifar10-airbench fork. All three
need the coefficient table to exist first. No Triton — the NS step is a handful of
matmuls and torch suffices.

## Success criteria

1. A committed coefficient table of 6–10 sets whose properties are independently
   re-verified by tests that do not share code with the solver.
2. The table places Keller inside the fitted frontier at its achieved γ ≈ 0.30
   (σ_min within ~25% of the fitted optimum), confirming standard Muon is a
   near-optimal interior family member.
3. A parameterized Muon that runs any table entry, plus exact SVD, through one
   interface.
4. An SVD logger producing before/after spectra, demonstrated on a CPU model.
