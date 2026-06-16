#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"
python -m src.cityseg.training.train --config "${CONFIG}"
