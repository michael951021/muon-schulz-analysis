#!/usr/bin/env bash
# the-box queue job `muonprep` (CPU, in the loop sandbox, network on): everything HANDOFF.md says to stage
# before the GPUs are reserved. Idempotent: a retry skips what is already in place.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

if ! "$PY" -c 'import torch, scipy, datasets, tiktoken, matplotlib, pytest, nsshape' 2>/dev/null; then
  echo "### venv (Python 3.12, torch 2.5.1+cu124, project [experiment])"
  python3.12 -m venv .venv
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install --progress-bar off 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cu124
  "$PY" -m pip install --progress-bar off -e '.[experiment]' pytest
fi
"$PY" -c 'import sys, torch; print("python", sys.version.split()[0], "torch", torch.__version__, "cuda build", torch.version.cuda)'

echo "### CIFAR"   # a slow mirror prints nothing for many minutes: report growth so the stall watchdog knows
( while sleep 240; do ls -s data/datasets/*.download 2>/dev/null | awk '{print "[download] " $2 " " int($1/1024) " MB"}'; done ) &
"$PY" scripts/prepare_data.py --source cifar --out data/datasets
kill $! 2>/dev/null || true

echo "### FineWeb"
fw=data/datasets/fineweb
if [ -f "$fw/manifest.json" ]; then   # reuse only if every split still matches its recorded sha256
  "$PY" - "$fw" <<'PY'
import hashlib, json, sys
from pathlib import Path
out = Path(sys.argv[1]); m = json.loads((out / "manifest.json").read_text())
for split, f in m["files"].items():
    if hashlib.sha256((out / f"{split}.bin").read_bytes()).hexdigest() != f["sha256"]:
        raise SystemExit(f"{out}/{split}.bin does not match its manifest: move {out} aside and rerun")
print(f"FineWeb manifest valid (revision {m['revision']}): reused")
PY
else
  rm -f "$fw"/*.bin   # a killed earlier try left partial splits and no manifest
  # datasets' streaming thread can abort the interpreter at exit (PyGILState_Release, exit 134) after the
  # manifest is written: the manifest, written last, is what says the splits are complete.
  "$PY" scripts/prepare_data.py --source fineweb --out "$fw" || test -f "$fw/manifest.json"
fi

echo "### fast tests (CPU; the CUDA test skips here and runs in the GPU job's preflight)"
"$PY" -m pytest -m 'not slow' -q
