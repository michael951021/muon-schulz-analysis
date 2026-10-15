# AI handoff: queue the Muon spectral-shaping study on the-box

## Request and current status

The user wants this project run on **two RTX 3090s for approximately 6–10 hours**,
using this host's existing GPU queue. The prepared default is an **8-hour
training ceiling on both GPUs**, with installation and dataset preparation
outside that clock. Queue the study through the host's normal GPU workflow;
do not start competing processes on occupied GPUs.

The project asks whether shaping momentum singular-value spectra can improve
Muon training over Keller's standard quintic or the exact-polar reference.
Code audit fixes, coefficient certificates, tests, experiment launchers, and
report generation are implemented. **The full GPU study has not run.**
Local evidence: 239 tests passed, one CUDA test skipped, and 25 real CPU
tinyshakespeare training runs completed. The smoke results do not resolve the
research hypothesis. GPU throughput, full-size memory, and FineWeb execution
must be verified on this machine.

This archive includes the actual working tree, including uncommitted fixes.
It deliberately excludes `.git`, virtual environments, credentials, downloaded
datasets, and prior run checkpoints. It is a source bundle, not a GPU container
or an offline dependency installation. Internet access is needed for setup.

## Read first

1. `docs/dual3090-study.md`: complete protocol, budget, setup, and interpretation.
2. `configs/dual_3090.json`: frozen selection rules, seeds, and default budget.
3. `reports/completion.md`: completed fixes and outstanding execution.
4. `reports/local_smoke/results.md`: historical CPU evidence, not GPU results.
5. `scripts/launch_dual3090.sh` and `scripts/run_study.py`: launch and scheduling.

`reports/audit.md` describes the original pre-fix implementation; its opening
note points to the remediation report. Historical report paths refer to the
original Mac workspace and are not paths to use on this host.

## Integrate with the existing queue

Inspect this host's instructions and existing GPU job mechanism first. No
scheduler name or submission command is assumed here. Adapt only the job wrapper
and resource binding to that mechanism, within its normal approval rules.

- Reserve **two 24GB RTX 3090s on the same machine**, together, for the entire
  study. The launcher runs independent experiments concurrently, one per GPU;
  it is not distributed training and needs neither DDP nor NVLink.
- Request **9 hours of GPU-job wall time**: up to 8 hours for the study plus room
  for preflight, fast tests, and report writing. This remains within the user's
  6–10-hour machine budget. Stage installation and datasets before reserving
  GPUs. If the queue includes preparation in its wall clock, account for that
  explicitly rather than granting extra training time.
- Plan for at least 4 CPU cores, 16GB host RAM, and 40GB free disk. More host RAM
  is welcome. Full-size preflight measures actual GPU memory before scheduling.
- Use a persistent project directory and output directory that survive job
  completion or queue interruption. Capture stdout/stderr to a persistent log.
- Ensure the queue exposes both allocated GPUs to the launcher. The child
  scheduler **sets `CUDA_VISIBLE_DEVICES` to each entry of `config.devices`**.
  Default entries are `0` and `1`. If the host queue allocates different physical
  IDs or exports UUID identifiers, set these two entries to the allocation's
  actual identifiers before the first run; do not accidentally escape the
  scheduler's allocation by forcing unrelated physical cards. Validate this
  binding using the host's own conventions.
- Do not use a single-GPU allocation or launch two copies of the whole study.
  One launcher owns both cards. Do not run preflight on cards you have not
  reserved.
- Keep the 8-hour budget by default. Do not silently change model, seeds,
  learning-rate grid, minimum work, or exact-SVD arm to make the job fit.
  If minimum work cannot fit the measured machine, preserve its failure and
  report the required protocol revision.

## Prepare before the GPU allocation

Run from the unpacked project root. Use Python 3.11 or 3.12. A known supported
installation option is shown below; reuse a compatible existing environment
when appropriate rather than downgrading it. Native dependencies and a compatible
NVIDIA driver are not included in the archive.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install -e '.[experiment]' pytest

.venv/bin/python scripts/prepare_data.py --source cifar --out data/datasets
.venv/bin/python scripts/prepare_data.py --source fineweb --out data/datasets/fineweb
```

Preparation verifies the official CIFAR archive and creates document-disjoint,
revision-recorded FineWeb train/validation/test token files with checksums.
If a FineWeb manifest already exists, validate and reuse it; the preparation
script intentionally refuses to overwrite it. See the protocol for details.

`uv.lock` is included, but the explicit installation commands above are the
documented CUDA setup. The machine's installed Torch/CUDA versions are recorded
and target preflight/tests must pass. Do not regenerate the coefficient family:
the solver is offline and the committed table is already recertified.

## Payload to execute inside the reserved GPU job

Use an absolute project path in the native job wrapper, then:

```bash
cd "$HOME/muon_schulz_extension"
mkdir -p runs
set -o pipefail
bash scripts/launch_dual3090.sh 8 runs/dual3090 2>&1 | tee runs/dual3090-launch.log
```

Change the `cd` path if the host stages jobs elsewhere. With a compatible
existing Python environment, set `NSSHAPE_PYTHON` to its absolute Python binary.
The launcher runs preflight on both cards, fast tests, calibrated experiments,
and report generation. Its exit status must be preserved by the queue wrapper.
No separate `torchrun`, process-per-GPU launcher, or external repetition loop is
needed; `run_study.py` handles both worker slots itself.

The stages are: all 21 arm/precision combinations on CIFAR × three LRs × two
tuning seeds; five selected/reference combinations × three new confirmation
seeds; language LR tuning; five combinations × three new language confirmations.
The GPT has 123,767,040 parameters, context 256. Training steps/tokens are matched
within each stage and calibrated to the actual hardware. This is a shorter
early-training study, not the original 2–5B-token proposal.

## Monitor, resume, and report

- `runs/dual3090/manifest.json`: immutable configuration and provenance.
- `runs/dual3090/stage_plan.json`: frozen calibrated step counts.
- `runs/dual3090/budget_progress.json`: cumulative recorded elapsed budget.
- `runs/dual3090/jobs/*/worker.log`: individual worker logs.
- `runs/dual3090/jobs/*/result.json`: actual completed work and metrics.
- `runs/dual3090/jobs/*/spectra.jsonl`: momentum spectra and domain diagnostics.
- `runs/dual3090/jobs/*/checkpoint.pt`: successful confirmation checkpoints.
- `runs/dual3090/study.json`: authoritative completed-study artifact.
- `reports/dual3090/{results.md,metrics.json,raw_results.json}`: final reports.

On failure, inspect `failure.json` and worker logs before retrying. Resume with
the same command and persistent output directory: complete jobs are reused;
interrupted jobs restart from initialization. Reacquire both GPUs first.
Changing source, configuration, environment, allocation IDs, or hours requires
a fresh output directory; do not mix unmatched runs to fabricate completion.
The budget ledger updates on normal exits; an abrupt OS kill can lose accounting
since the prior process exit, so also check the host queue's actual usage before
requeueing. Requeue only within the user's remaining compute allowance.

If training completes but report generation fails, generate the report again:

```bash
.venv/bin/python scripts/report_study.py --input runs/dual3090/study.json --out reports/dual3090
```

Summarize the primary held-out loss comparison against both Keller and exact
SVD, paired-seed intervals, selected LRs, actual steps/tokens, runtime, memory,
and realized fp32/bf16 spectra. Inspect counts below the certified input threshold
before attributing broad spectra to a broken polynomial. Distinguish completed,
failed, and unresolved results. Three seeds and a short budget cannot establish
universal optimizer optimality.

Deliver the final report directory, launch log, study JSON, and relevant spectra
to the user. Keep checkpoints available on this host without automatically
transferring gigabytes. If queuing rather than executing immediately, report the
native queue job ID, requested resources, persistent paths, and expected start
information supplied by that queue.

## Bundle integrity

`BUNDLE_MANIFEST.json` lists SHA-256 hashes for every included file except itself.
Verify them after extraction before changing any allocation settings:

```bash
python3 - <<'PY'
import hashlib, json
from pathlib import Path
manifest = json.loads(Path('BUNDLE_MANIFEST.json').read_text())
for name, expected in manifest['files'].items():
    actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f'Checksum mismatch: {name}')
print(f"Verified {len(manifest['files'])} bundled files")
PY
```
