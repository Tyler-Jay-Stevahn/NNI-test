#!/usr/bin/env bash
# run.sh — generate proposals, verify them, then smoke-test compile every model.
# Usage: bash pipeline/run.sh
set -u
cd "$(dirname "$0")/.."   # repo root

PY=.venv/bin/python
if [ ! -x "$PY" ]; then
  echo "venv missing — create it first:" >&2
  echo "  uv venv .venv && uv pip install --python .venv/bin/python numpy torch --index-url https://download.pytorch.org/whl/cpu" >&2
  exit 1
fi

echo "== gen (models only) =="
"$PY" gen_proposals.py
echo "== verify (math/physics/thermo + live citations) =="
"$PY" verify_proposals.py
echo "== smoke-test compile (tiny train/test/predict) =="
"$PY" pipeline/smoke_test.py --all
