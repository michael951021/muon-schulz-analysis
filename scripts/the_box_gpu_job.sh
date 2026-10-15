#!/usr/bin/env bash
# the-box queue job `muonstudy` (GPU job): lib/jobs.py starts this once both 3090s are free (model server
# stopped, agents parked) with /dev/nvidia* passed into the sandbox. Only the job wrapper is adapted here;
# the payload is HANDOFF.md's, unchanged, and its exit status is the job's.
set -euo pipefail
cd "$(dirname "$0")/.."
export NSSHAPE_PYTHON="$PWD/.venv/bin/python"
nvidia-smi -L
n=$("$NSSHAPE_PYTHON" -c 'import torch; print(torch.cuda.device_count())')
[ "$n" = 2 ] || { echo "need both 3090s visible, torch sees $n" >&2; exit 1; }
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

# Heartbeat for the queue's stall watchdog: relay the newest worker-log line, only when it changed. A hung
# study prints nothing, so the watchdog still catches it.
mkdir -p runs
( last=""; while sleep 300; do
    f=$(ls -t runs/dual3090/jobs/*/worker.log 2>/dev/null | head -1) || true
    [ -n "$f" ] || continue
    line="$(basename "$(dirname "$f")"): $(tail -n 1 "$f")"
    [ "$line" != "$last" ] && { echo "[progress] $line"; last="$line"; }
  done ) &
hb=$!
trap 'kill $hb 2>/dev/null || true' EXIT

bash scripts/launch_dual3090.sh 8 runs/dual3090 2>&1 | tee -a runs/dual3090-launch.log
