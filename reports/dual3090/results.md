# Dual RTX 3090 ablation results

Scope: budget-limited CIFAR-10 CNN sweep and approximately 124M-parameter GPT on FineWeb.

Learning rates were selected on validation loss in tuning runs. Reported test metrics come only from fresh confirmation seeds, at the final checkpoint. Lower loss is better. Paired differences use the same seed against Keller at the same precision.

## Cifar

| Arm | Precision | LR | Steps | Test loss mean ± SD | Accuracy | Seconds/run | Loss difference vs Keller (95% CI) |
|---|---|---:|---:|---:|---:|---:|---|
| exact_svd | fp32 | 0.08 | 2160 | 0.5892 ± 0.0030 | 79.81% | 28.4 | -0.0005 [-0.0145, +0.0135] |
| gamma_0.20 | bf16 | 0.08 | 2160 | 0.5844 ± 0.0058 | 79.43% | 14.3 | -0.0047 [-0.0081, -0.0013] |
| gamma_0.20 | fp32 | 0.08 | 2160 | 0.5848 ± 0.0052 | 79.76% | 14.3 | -0.0049 [-0.0119, +0.0020] |
| keller | bf16 | 0.08 | 2160 | 0.5891 ± 0.0071 | 79.30% | 14.4 | +0.0000 [+0.0000, +0.0000] |
| keller | fp32 | 0.08 | 2160 | 0.5897 ± 0.0066 | 79.20% | 14.3 | +0.0000 [+0.0000, +0.0000] |

Confirmation seeds: [10, 11, 12].

| Arm | Precision | Paired test-loss difference vs exact SVD (95% CI) |
|---|---|---|
| exact_svd | fp32 | +0.0000 [+0.0000, +0.0000] |
| gamma_0.20 | bf16 | -0.0048 [-0.0118, +0.0021] |
| gamma_0.20 | fp32 | -0.0044 [-0.0124, +0.0035] |
| keller | bf16 | -0.0001 [-0.0102, +0.0100] |
| keller | fp32 | +0.0005 [-0.0135, +0.0145] |

## Language

| Arm | Precision | LR | Steps | Test loss mean ± SD | Accuracy | Seconds/run | Loss difference vs Keller (95% CI) |
|---|---|---:|---:|---:|---:|---:|---|
| exact_svd | fp32 | 0.005 | 3082 | 5.4738 ± 0.0354 | 20.45% | 8108.5 | +0.0005 [-0.0047, +0.0057] |
| gamma_0.20 | bf16 | 0.005 | 3082 | 5.4731 ± 0.0344 | 20.47% | 345.5 | -0.0021 [-0.0065, +0.0023] |
| gamma_0.20 | fp32 | 0.005 | 3082 | 5.4729 ± 0.0334 | 20.41% | 657.2 | -0.0004 [-0.0041, +0.0033] |
| keller | bf16 | 0.005 | 3082 | 5.4752 ± 0.0335 | 20.39% | 345.3 | +0.0000 [+0.0000, +0.0000] |
| keller | fp32 | 0.005 | 3082 | 5.4733 ± 0.0348 | 20.42% | 657.3 | +0.0000 [+0.0000, +0.0000] |

Confirmation seeds: [10, 11, 12].

| Arm | Precision | Paired test-loss difference vs exact SVD (95% CI) |
|---|---|---|
| exact_svd | fp32 | +0.0000 [+0.0000, +0.0000] |
| gamma_0.20 | bf16 | -0.0007 [-0.0088, +0.0075] |
| gamma_0.20 | fp32 | -0.0009 [-0.0068, +0.0049] |
| keller | bf16 | +0.0014 [-0.0046, +0.0074] |
| keller | fp32 | -0.0005 [-0.0057, +0.0047] |

## Diverged tuning runs

These tuning runs hit a non-finite loss or gradient. Their learning rate was ineligible for selection for that arm and precision; nothing was dropped silently.

| Task | Arm | Precision | LR | Seed | Status | Step reached |
|---|---|---|---:|---:|---|---:|
| cifar | gamma_0.45 | bf16 | 0.005 | 0 | nonfinite_grad | 1912/2160 |
| cifar | gamma_0.45 | bf16 | 0.02 | 1 | nonfinite_grad | 706/2160 |
| cifar | gamma_0.60 | bf16 | 0.08 | 1 | nonfinite_loss | 1162/2160 |

## Interpretation

Intervals are Student-t intervals over paired seed differences, not uncertainty over individual examples. Three confirmation seeds give low statistical power. The intervals are descriptive and are not corrected for multiple comparisons. Selection among fitted arms occurred before these fresh seeds, but interpretation still applies only to the selected family and this training regime.

Exact SVD is a partial polar factor with numerical null-space directions removed; its cost is measured, rather than described as compute-matched to the quintics. All polynomial arms use five NS iterations. Runtime includes periodic validation and diagnostics; diagnostic time is available separately in metrics.json.

The paired interval does not resolve a language-loss difference between the selected fp32 fit and Keller. This is not proof of universal optimality. Compare the exact-SVD anchor and realized bf16 spectra before attributing differences to shape alone.

The approximately 124M model uses context 256 and the calibrated token budget shown in study.json. It is a shortened confirmation, not a 2–5B-token reproduction of modded-nanogpt or the airbench speedrun.

The fitted fp32 arm does not show a wholly negative paired interval against exact SVD. This experiment therefore does not resolve H1 in favor of deliberate distortion over the exact-polar anchor.

## Provenance

Input: `runs/dual3090-r3/study.json`.

The raw result file records exact coefficient values and their hash, source fingerprint, dataset hashes and splits, versions, measured memory, curves, and completed work. Per-job spectra and checkpoints remain in the run directory. This report should be read alongside those artifacts; aggregate loss does not certify the realized spectral band.
