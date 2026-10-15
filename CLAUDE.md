# muon-schulz: Muon spectral-shaping study on the-box's two 3090s

Imported from `muon-schulz-dual3090.tar.gz` (manifest verified). Read `HANDOFF.md` and
`docs/dual3090-study.md`: the experimental protocol is frozen. Do not change the model, seeds,
LR grid, minimum work, exact-SVD arm, `configs/dual_3090.json` or the 8-hour budget to make a run fit.

The queue runs it as two jobs (TODO.md), no agent:
- `muonprep` (CPU job): `scripts/the_box_prep.sh` builds `.venv` (Python 3.12, torch 2.5.1+cu124),
  downloads CIFAR, builds FineWeb token splits, runs the fast CPU tests.
- `muonstudy` (GPU job, both 3090s): `scripts/the_box_gpu_job.sh` wraps `scripts/launch_dual3090.sh 8 runs/dual3090`.
  The sandbox exposes physical GPUs 0 and 1, matching `config.devices` ["0","1"].

Persistent outputs (shared, survive retries): `runs/dual3090/` (study.json, jobs/*/worker.log, spectra,
checkpoints), `runs/dual3090-launch.log`. Reports: `reports/dual3090/`. A retry reuses completed jobs in
`runs/dual3090` and the budget ledger there, so it never grants extra training hours.
