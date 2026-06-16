#!/usr/bin/env bash
set -euo pipefail

python -m src.cityseg.eda.analyze_dataset --config configs/default.yaml
