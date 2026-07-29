#!/usr/bin/env bash
# One-shot local reproduction. Run from the repository root: bash scripts/run_all.sh
set -euo pipefail

echo "[1/3] Downloading data (labels + contracts) ..."
python -m src.data_download

echo "[2/3] Building labels and printing distribution ..."
python -m src.labeling

echo "[3/3] Training and evaluating all five models ..."
python -m src.pipeline

echo
echo "Done. Results are in artifacts/metrics.json"
