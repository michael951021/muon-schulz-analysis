# Code and results audit

Audit date: 2026-10-13. Scope: this local checkout, its committed artifacts,
tests, and fresh CPU numerical probes. No external literature review.

**Historical audit:** The findings below describe the implementation before
remediation. See [the results in the README](../README.md#results) and
[the current two-3090 protocol](../docs/dual3090-study.md) for subsequent changes.
The original audit metrics are preserved rather than overwritten.

## Verdict

The project is coherent as a prototype for stages 1–2: a coefficient family,
parameterized optimizer, and spectral diagnostics. It is **not a completed
experimental answer to the research question**, and is not yet ready for a
trustworthy GPU training sweep. The central question is whether Muon benefits
from deliberately shaping the singular-value spectrum rather than closely
approximating the polar factor. No CIFAR or transformer training runs, accuracy
tables, learning-rate sweeps, or final study report were found in this checkout.
There are no training launchers or datasets here, and this environment reports
no CUDA or MPS device. Running the available numerical experiments cannot answer
the training question. This report records the results that can actually be
reproduced; it does not substitute a toy optimization test for that study.

## Existing results and fresh verification

`data/coefficients/v1.json`, generated 2026-08-26, contains nine fitted quintics,
the Keller baseline, and an exact-SVD sentinel. `figures/composed_maps.png` shows
the composed scalar maps and fitted tradeoff; figures are ignored by Git.
The README explains these preliminary results. Design and implementation-plan
documents describe future work, rather than completed training experiments.

Fresh results are in `reports/audit_metrics.json`, reproducible with:

```sh
.venv/bin/python scripts/audit_results.py
.venv/bin/python -m pytest -m 'not slow' -q
.venv/bin/python -m pytest -q
```

The audit evaluates each five-step scalar composition at one million log-spaced
points over its claimed domain, independently of the production evaluation
functions. Positivity is checked through the minimum of the quadratic factor
`a + b*u + c*u²`, where `u=x²`; single-step maxima are checked at derivative
roots and endpoints. The metrics include versions and the coefficient-table
SHA-256. Dense sampling supports the band claim numerically; it is not a formal
continuous-domain certificate or a proof of global optimization.

| Entry | Claimed γ | Smallest certified input σ | Dense measured half-width |
|---|---:|---:|---:|
| gamma_0.01 | 0.01 | 0.0177589 | 0.0099753 |
| gamma_0.02 | 0.02 | 0.0130028 | 0.0199448 |
| gamma_0.05 | 0.05 | 0.00657061 | 0.0499141 |
| gamma_0.10 | 0.10 | 0.00344757 | 0.0998058 |
| gamma_0.15 | 0.15 | 0.00221879 | 0.1499648 |
| gamma_0.20 | 0.20 | 0.00156458 | 0.1999528 |
| gamma_0.30 | 0.30 | 0.000904313 | 0.2999640 |
| gamma_0.45 | 0.45 | 0.000461907 | 0.4495686 |
| gamma_0.60 | 0.60 | 0.000246300 | 0.5999056 |
| Keller | 0.30 | 0.00142031 | 0.2994431 |

All ten scalar maps meet their band targets on the dense probe and are strictly
positive on the stored bounded interval. The fitted input threshold falls
strictly as γ rises. The widest family member reaches inputs about 72 times
smaller than the tightest. At γ=0.30 the fitted threshold is 36.3% lower than
Keller's, corresponding to 57.1% greater reciprocal dynamic range. This is
coefficient-level headroom, **not a 57% training improvement**. The stochastic,
finite-box, grid-based solver establishes a best-found family, not a proven
globally optimal frontier.

The fast suite passed: **181 passed, 13 deselected**, in 4.74 seconds. The full
suite, including fresh stochastic solver fits, passed: **194 passed** in
127.45 seconds. Passing these tests does not cover the additional findings below.

## Material findings

1. **The nominal γ axis does not reliably survive bf16.** On a reproducible
   64×64 CPU probe, seed 7, with normalized input singular values within the
   claimed domain, gamma_0.45 produces a measured half-width of **0.6950** rather
   than ≤0.45. gamma_0.01 yields 0.0160 rather than ≤0.01. These are observations
   of this CPU implementation, not GPU measurements. Existing backend tests
   omit gamma_0.45 and the tightest entry from their band assertions and allow
   an absolute 0.05 tolerance. A training sweep needs measured realized spectra
   for every arm and device; the scalar certificates do not certify bf16 matrix
   arithmetic.

2. **Exact SVD can move parameters with zero momentum.**
   `exact_svd_polar(zeros(8,8))` has Frobenius norm 2.8284; every polynomial
   backend returns zero. SVD supplies arbitrary null-space directions at rank
   deficiency, so `U @ Vh` is not the same zero-preserving operation as the
   polynomial. At minimum define and test zero and rank-deficient behavior
   before comparing training arms. The existing zero-gradient test uses Keller.

3. **The logger is not GPU-ready as written.** In
   `diagnostics/schema.py`, the quantile tensor is created on CPU, and
   `torch.histogram` is called on the original spectrum device. This device path
   is not covered by CPU tests; explicitly staging detached fp32 spectra to CPU
   would support the intended CUDA training use. No CUDA execution was possible
   in this environment. Logged inputs are raw momenta, not the actual normalized
   bf16 inputs. The schema also lacks coefficient identity, normalization norm,
   and counts below σ_min, which would make domain comparisons traceable.

4. **Sampling and parameter identity are misleading.** `SVDLogger` allocates a
   seeded RNG but never uses it: it always takes the first `max_params` calls.
   Its reproducibility test therefore passes trivially. `muon.py` advances the
   parameter index only for parameters with gradients, so identities can shift
   when a parameter has no gradient. Use stable parameter identities and either
   implement sampling or document the first-parameters policy.

5. **The coefficient certificates are approximate.** Analytic extrema exceed
   several stored invariant bounds by small amounts, up to about **7.1e−8**.
   The production bound and invariance tests use sampled grids, and `_certify`
   does not include its invariance margin in the feasible predicate.
   `reachable_bound` returns a finite last iterate even if its iteration limit
   expires without convergence. These do not overturn the observed scalar
   tradeoff, but prevent calling the present certificates rigorous. Use analytic
   extrema and an explicit convergence/feasibility check for stronger claims.
   The independent table tests avoid the solver but reuse the production
   polynomial-analysis functions; they are not fully independent certificates.

6. **Packaging and persistence need attention.** The loader expects a table in
   the repository's top-level `data/` directory, while the wheel configuration
   packages only `src/nsshape`. An editable checkout works; the configured wheel
   does not include the required artifact. The loader does not validate table
   version, duplicate names, or certificate fields despite its description.
   Standard optimizer state dictionaries omit `coefficient_set` and
   `step_count`, so resuming can silently change the arm or reset logging steps.
   The exact-SVD entry contains nonstandard JSON `Infinity` values.

7. **Documentation overstates the conclusion and has inconsistent criteria.**
   The design's acceptance paragraph allows a Keller/fitted threshold ratio of
   1.2–2.5, while its final success criteria say “within ~25%”; the actual ratio
   is 1.571. Its reported critical-point range of roughly 0.56–0.63 also does
   not describe the committed fitted range, about 0.393–0.644. Failure to observe
   a training improvement would bound this tested family and experimental
   budget; it would not alone prove geometric alignment is generally optimal
   or guarantee publishability. Exact SVD has higher finite computational cost,
   rather than literally “unbounded cost.”

## What this says about the goal

The preliminary mathematical result is useful: five equal-step-cost quintics
can trade spectral accuracy for reach into small singular values, and Keller
is a feasible comparison with measurable headroom at approximately the same
band width. It establishes a plausible ablation, not its outcome.

To answer H1, first resolve precision and baseline semantics, make diagnostics
usable on the target device, then implement the deferred CIFAR γ×learning-rate
grid with repeated seeds, matched data and training budgets, held-out selection,
actual spectra, and runtime measurements. Compare exact SVD explicitly by both
training outcome and cost. Transformer confirmation remains a separate missing
stage. No claim about better accuracy, lower validation loss, or generalization
is supported by the current artifacts.

This audit adds a reproducible measurement script and reports. It leaves the
existing solver, optimizer, and coefficient table unchanged so findings remain
reproducible against the original implementation.
