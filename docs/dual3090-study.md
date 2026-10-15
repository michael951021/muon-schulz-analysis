# An executable study for two RTX 3090s

Target: **6–10 elapsed hours on a machine with two 24GB RTX 3090s**. Default
ceiling: 8 hours after dataset preparation, approximately 16 GPU-hours if both
GPUs stay busy. This is a ceiling, not a promise of measured throughput. Local
verification used CPU; target GPU calibration is mandatory.

The project asks whether deliberately shaped singular-value bands improve
training relative to standard Muon or exact polar updates. The original plan's
2–5B-token transformer study is beyond the evidence established here and is
replaced by a shorter, explicitly budget-limited confirmation. We do not claim
that a two-3090 pilot reproduces airbench or modded-nanogpt performance.

## Protocol

Two GPUs run two **independent experiments** at once, with one process on each
GPU. No distributed training or NVLink is needed. This gives more independent
seeds for the compute budget and avoids combining unlike arms inside one run.

| Stage | Work | Default allocation |
|---|---|---:|
| Calibration | Test GPU backends, memory, and throughput including exact SVD | Covered by reserve |
| CIFAR selection and confirmation | 21 arm/precision combinations × 3 LRs × 2 tuning seeds; then 5 arms × 3 fresh seeds | 2.8 h ceiling |
| Language LR selection | 5 arms × 3 LRs × 1 tuning seed | 1.2 h ceiling |
| Language confirmation | 5 arms × 3 fresh seeds on approximately 124M-parameter GPT | Remaining time, nominally 3.2 h |
| Reserve | Startup, calibration, checkpoint writes, throughput variability, report | 0.8 h |

The 21 CIFAR combinations are all 11 entries in fp32 and the 10 polynomial
entries in bf16. Exact SVD always computes in fp32. Each polynomial performs
five iterations. The LR grid is `{0.005, 0.02, 0.08}`; tuning seeds are `{0,1}`;
confirmation seeds are `{10,11,12}`. All parameters, seeds, allocations, and
selection rules are frozen in `configs/dual_3090.json` before GPU results exist.

CIFAR uses a small three-convolution CNN with GroupNorm, GELU, horizontal-flip
augmentation, and explicit flattened convolution matrices. Muon operates on
the four matrix parameters; scalar parameters use AdamW. The official training
set is split into 45,000 training and 5,000 validation images using fixed seed
1729; the official 10,000-image test set is reserved for confirmation. Every arm
gets identical initialized weights, shuffled examples, augmentations, and
training steps for a given seed. The final checkpoint is the endpoint.

Select each arm's LR by mean final validation cross-entropy across its two
tuning seeds. Select the fitted fp32 family member by its tuned mean validation
loss. Neither test accuracy nor the transformer result selects this member.
Confirm the selected member, Keller, and exact SVD in fp32, plus the same selected
member and Keller in bf16. The bf16 comparison tests implementation sensitivity;
the fp32 comparison more closely isolates the intended polynomial geometry.

Language uses a conventional causal GPT: 12 layers, width 768, 12 heads,
context 256, tied GPT-2 input/output embeddings, **123,767,040 parameters**.
Hidden matrices use Muon; embeddings, positional embeddings, and normalization
parameters use AdamW. CUDA forward/backward uses bf16 autocast, and NS precision
is separately controlled. TF32 is disabled. Global gradient clipping, cosine LR
decay, warmup, and AdamW settings are identical across arms. This architecture
has no speedrun-specific modifications.

FineWeb is GPT-2-tokenized with exact dataset revision recorded. SHA-256 of each
document assigns it to train, validation, or test; repeated identical documents
are omitted. Default preparation writes 100M training tokens and 1M tokens in
each evaluation split. Validation and test evaluate the same first 65,536
non-overlapping target tokens for every arm and seed. Training samples chunks
with a separately seeded RNG. Tune language LRs independently using seed 0,
then confirm on seeds 10–12; the tuning horizon can be shorter than confirmation,
which limits the reliability of LR extrapolation and is recorded explicitly.

## How the compute ceiling works

Each worker measures synchronized per-step times and full worker overhead.
Calibration uses eight real steps per arm on each GPU and takes the slower
profile. The scheduler estimates two-worker makespan with selection barriers,
includes periodic validation costs, reserves checkpoint time, applies
a 1.25 safety multiplier, and solves for a **common step count per stage**.
Slow exact-SVD runs therefore reduce the work allotted to all comparable arms;
they do not get fewer tokens to make them look cheaper.

CIFAR permits 400–2,160 steps per run (about 2.3–12.3 training epochs at batch
256). Language tuning requires at least 100 steps, confirmation at least 400,
and each is capped at 20,000 steps. With batch 8 × context 256, confirmation
therefore spans approximately **0.82M–40.96M tokens per run**, as determined by
the measured machine. It is an early-training comparison, not mature 2–5B-token
pretraining. Record the chosen budget before interpreting any outcome.

If minimum budgets do not fit, the scheduler stops and asks for a deliberate
protocol revision through its error message; it does not silently shrink the
study or drop seeds/arms. Unused capped CIFAR and tuning time goes to language
confirmation. A global deadline terminates outstanding workers and leaves the
study incomplete. `budget_progress.json` accumulates elapsed time across normal
exits and resumes; abrupt OS-level kills cannot reliably account for time since
the last process exit. A manually interrupted job restarts from initialization,
while complete jobs are reused. No partial job appears in the final report.

Completed jobs are keyed by immutable specs and source fingerprints. Changing
code, coefficients, configuration, or hours requires a new output directory.
The frozen `stage_plan.json` keeps step counts stable on resume. Existing valid
checkpoints support inspection and downstream use; within-job training resume
is not implemented.

## Run on the GPU machine

Use Linux, Python 3.11 or 3.12, a compatible NVIDIA driver, two visible 3090s,
at least 16GB RAM, and **40GB free disk** for dependencies, data/cache, and
confirmation checkpoints. These are planning allowances, not local measured
GPU requirements. Preflight performs a full-size forward/backward and exact-SVD
update on each GPU and reports measured peak allocation before scheduling.

From this checkout on the target machine:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install -e '.[experiment]' pytest

.venv/bin/python scripts/prepare_data.py --source cifar --out data/datasets
.venv/bin/python scripts/prepare_data.py --source fineweb --out data/datasets/fineweb

# Starts the compute clock after data preparation. Run inside tmux if using SSH.
bash scripts/launch_dual3090.sh 8 runs/dual3090
```

The CUDA 12.4/PyTorch 2.5.1 install is a known supported option, not a requirement
to downgrade an already working supported GPU environment. The local tests used
a different PyTorch version, so target-machine preflight and tests remain
required. For a 6- or 10-hour ceiling, replace `8`. Only the scheduler is inside
that timed budget; installation, dataset preparation, preflight, and tests are
outside it. Preparation can take significant network time and is intentionally
separate from GPU training.

Resume with the same command/output directory. If a hardware or worker failure
occurs, inspect `failure.json` and the corresponding `jobs/*/worker.log` before
resuming. A provenance change requires a fresh directory. Do not combine jobs
from differently sized budgets into one matched-work report.

## Output and interpretation

`runs/dual3090/study.json` records all calibration, tuning, and confirmation
results, chosen steps/tokens, selected LRs, source/table hashes, dataset manifests,
versions, runtime, memory, and curves. Each job also saves before/after spectra;
every successful confirmation job saves model and optimizer checkpoints.

The launcher creates:

```text
reports/dual3090/results.md
reports/dual3090/metrics.json
reports/dual3090/raw_results.json
```

The primary endpoint is held-out test cross-entropy at matched work. Report
CIFAR accuracy as a secondary endpoint and compare the fitted arm against both
Keller and exact SVD using paired seed differences. The report prints descriptive
95% Student-t intervals; three seeds have limited power and the intervals are
not corrected for multiple comparisons. A negative interval against exact SVD
is evidence consistent with H1 **in this regime**, not a universal proof.
An interval crossing zero is unresolved evidence, not a defense of universal
polar optimality. Runtime comparisons must state that exact SVD has a different
cost and that logged diagnostic overhead is included separately.

Always inspect realized spectra, normalized input ranges, and counts below
σ_min, especially for bf16. Nominal γ is not a guarantee of bf16 matrix output
geometry. The local smoke report has real training numbers but makes no claim
about CIFAR/FineWeb or target-device results.

To reproduce the local pipeline check without a GPU:

```bash
mkdir -p data/datasets
curl -L --fail https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/datasets/tinyshakespeare.txt
.venv/bin/python scripts/prepare_data.py --source shakespeare --text data/datasets/tinyshakespeare.txt --out data/datasets/shakespeare
.venv/bin/python scripts/run_local_smoke.py --out runs/local_smoke
.venv/bin/python scripts/report_study.py --input runs/local_smoke/smoke.json --out reports/local_smoke
```

Preparation records the source SHA-256; the reported run used
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
Choose a fresh token-output directory if its manifest already exists to preserve
the prior dataset provenance.

Dataset and installation references: [official CIFAR-10 source](https://www.cs.toronto.edu/~kriz/cifar.html),
[FineWeb dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb),
[official PyTorch previous-version installation commands](https://pytorch.org/get-started/previous-versions/).
