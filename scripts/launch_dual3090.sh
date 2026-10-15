#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${NSSHAPE_PYTHON:-.venv/bin/python}"
HOURS="${1:-8}"
OUT="${2:-runs/dual3090}"
"$PYTHON_BIN" scripts/preflight.py
"$PYTHON_BIN" -m pytest -m 'not slow' -q
"$PYTHON_BIN" scripts/run_study.py --config configs/dual_3090.json --hours "$HOURS" --out "$OUT"
"$PYTHON_BIN" scripts/report_study.py --input "$OUT/study.json" --out reports/dual3090
