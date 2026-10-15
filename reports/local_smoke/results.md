# Local training smoke results

Scope: CPU tinyshakespeare, two-layer 64-wide character transformer. **This is a pipeline check, not the planned CIFAR/FineWeb confirmation.**

Learning rates were selected on validation loss in tuning runs. Reported test metrics come only from fresh confirmation seeds, at the final checkpoint. Lower loss is better. Paired differences use the same seed against Keller at the same precision.

## Language

| Arm | Precision | LR | Steps | Test loss mean ± SD | Accuracy | Seconds/run | Loss difference vs Keller (95% CI) |
|---|---|---:|---:|---:|---:|---:|---|
| exact_svd | fp32 | 0.02 | 600 | 2.1946 ± 0.0329 | 36.52% | 4.1 | +0.0006 [-0.0392, +0.0404] |
| gamma_0.10 | bf16 | 0.02 | 600 | 2.2025 ± 0.0266 | 36.17% | 9.2 | +0.0074 [-0.0164, +0.0311] |
| gamma_0.10 | fp32 | 0.02 | 600 | 2.2058 ± 0.0205 | 35.98% | 2.9 | +0.0118 [-0.0085, +0.0321] |
| keller | bf16 | 0.02 | 600 | 2.1951 ± 0.0251 | 36.48% | 9.0 | +0.0000 [+0.0000, +0.0000] |
| keller | fp32 | 0.02 | 600 | 2.1940 ± 0.0285 | 36.30% | 3.0 | +0.0000 [+0.0000, +0.0000] |

Confirmation seeds: [10, 11, 12].

| Arm | Precision | Paired test-loss difference vs exact SVD (95% CI) |
|---|---|---|
| exact_svd | fp32 | +0.0000 [+0.0000, +0.0000] |
| gamma_0.10 | bf16 | +0.0079 [-0.0117, +0.0275] |
| gamma_0.10 | fp32 | +0.0112 [-0.0313, +0.0537] |
| keller | bf16 | +0.0005 [-0.0416, +0.0427] |
| keller | fp32 | -0.0006 [-0.0404, +0.0392] |

## Interpretation

Intervals are Student-t intervals over paired seed differences, not uncertainty over individual examples. Three confirmation seeds give low statistical power. The intervals are descriptive and are not corrected for multiple comparisons. Selection among fitted arms occurred before these fresh seeds, but interpretation still applies only to the selected family and this training regime.

Exact SVD is a partial polar factor with numerical null-space directions removed; its cost is measured, rather than described as compute-matched to the quintics. All polynomial arms use five NS iterations. Runtime includes periodic validation and diagnostics; diagnostic time is available separately in metrics.json.

No conclusion about H1, CIFAR accuracy, or FineWeb language modeling follows from these smoke results. The two-GPU protocol remains to be run on the target machine.

## Provenance

Input: `runs/local_smoke_release/smoke.json`.

The raw result file records exact coefficient values and their hash, source fingerprint, dataset hashes and splits, versions, measured memory, curves, and completed work. Per-job spectra and checkpoints remain in the run directory. This report should be read alongside those artifacts; aggregate loss does not certify the realized spectral band.
