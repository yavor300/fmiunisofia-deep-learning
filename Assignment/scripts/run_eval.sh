#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"
CHECKPOINT="${2:-}"
python -m src.cityseg.training.evaluate --config "${CONFIG}" --checkpoint "${CHECKPOINT}"
