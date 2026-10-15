# Remediation and experiment status

The code audit has been followed through to implementation fixes, numerical
verification, a real CPU training smoke study, and an executable two-RTX-3090
protocol. The full GPU experiment is **prepared, not executed**. This checkout
does not have CUDA, and no remote machine access was supplied. The requested
6–10-hour target is implemented as a configurable elapsed training ceiling;
default 8 hours on both GPUs, with data preparation performed beforehand.

## Fixes completed

- Partial SVD removes numerical null-space directions and returns zero on zero
  momentum. Regression tests cover zero-gradient optimizer behavior.
- Polynomial NS supports fp32 and bf16 explicitly. All family members are checked
  against scalar reference behavior in fp32; bf16 deviations remain measured
  properties of the practical implementation, rather than hidden behind a
  nominal band certificate.
- Continuous scalar image bounds propagate exact polynomial extrema in float64.
  Forward invariance uses derivative roots and endpoints; unconverged bound
  searches fail closed. Positivity checks the minimum of P(x)/x, including
  double-zero failures. Independent tests use a separate polynomial representation
  to verify each whole composed interval. These are numerical certificates, not
  outward-rounded interval proofs or global optimality proofs.
- Stored certificates were recomputed without changing the fitted coefficients.
  JSON contains null for undefined exact-SVD certificate diagnostics rather than
  nonstandard Infinity. The loader rejects invalid versions, duplicate entries,
  infeasible margins, malformed coefficients, and invalid scales.
- The logger supports uniform seeded sampling over a known population, stable
  parameter indices, CPU histogram/quantile summaries after device SVD, coefficient
  and precision identity, normalized input range, and counts below σ_min.
  A zero spectrum uses a null undefined band ratio in strict JSON.
- Optimizer checkpoints preserve exact coefficients, precision, and step count.
- The wheel bundles its coefficient table and loads all 11 entries outside the
  source checkout. The dependency lockfile includes experiment requirements.

## Runs completed locally

The [local smoke report](local_smoke/results.md) contains 25 real tinyshakespeare
training runs: ten LR-tuning runs and fifteen confirmations over three fresh
seeds, five arm/precision combinations, and 600 confirmation steps. All arms
train successfully. Their loss differences are small and the paired intervals
do not resolve an advantage for the selected gamma_0.10 probe over Keller or
exact SVD. This is a pipeline check, not a test of the planned CIFAR/FineWeb H1.

`verification/current_metrics.json` records fresh scalar and bf16 probes against
the recertified table. The original `audit_metrics.json` remains the pre-fix
snapshot. Backend changes do not eliminate inherent bf16 rounding error; the
target study therefore separates fp32 geometry from bf16 sensitivity.

The full suite passed: **239 passed, 1 skipped**, in 91.04 seconds. Test output is
recorded in `verification/tests.txt`. CUDA-specific checks
are skipped locally and required on the target machine. A prior local smoke
attempt encountered a worker exit; a direct rerun succeeded, and the final
release run completes the full protocol. Incomplete attempts are not included
in the reported results.

## Run still required

Follow [the two-3090 protocol](../docs/dual3090-study.md) to install dependencies,
prepare checksum-pinned CIFAR-10 and document-disjoint FineWeb splits, then run:

```bash
bash scripts/launch_dual3090.sh 8 runs/dual3090
```

It first verifies both GPUs and full-size memory requirements, then calibrates
every arm on both devices, runs the validation-selected CIFAR grid and fresh-seed
checks, selects language LRs, and runs matched-work confirmations. Calibration
uses the slower device profile, accounts for selection barriers, and orders jobs
by estimated cost. It reserves confirmation compute before starting the CIFAR
grid. Reports compare the selected shape against both Keller and exact SVD with
descriptive paired intervals, plus runtime and spectra.

The approximately 124M language model receives a measured token budget rather
than the original 2–5B-token proposal. No GPU performance, CIFAR accuracy, or
FineWeb validation/test result is claimed before execution. The final GPU
results will be written to `reports/dual3090/results.md`, with raw records and
metrics beside it.
