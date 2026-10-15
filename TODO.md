# Tasks

- [x] **Muon prep: venv, CIFAR, FineWeb, fast tests** — staged before the GPU allocation, per HANDOFF.md (id: muonprep) (after: -) (run: `bash scripts/the_box_prep.sh`) (mem: 16) (cpus: 8) (timeout: 6h)
- [ ] **Muon dual-3090 study (8 h training ceiling)** — both 3090s, preflight + fast tests + calibrated study + report; results in runs/dual3090-r3 and reports/dual3090 (r1 runs/dual3090, r2 runs/dual3090-r2: crashed on bf16 tuning divergence, kept as record) (id: muonstudy) (after: muonprep) (run: `bash scripts/the_box_gpu_job.sh`) (gpu) (expect: runs/dual3090-r3/study.json reports/dual3090/results.md) (mem: 32) (cpus: 8) (timeout: 9h)
