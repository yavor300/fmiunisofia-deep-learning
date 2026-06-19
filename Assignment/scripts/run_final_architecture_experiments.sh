#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

make baseline RUN_NAME=000_baseline_majority

experiments=(
  "020_final_tiny_unet_ce_cosine"
  "021_final_unet_resnet34_ce_cosine"
  "022_final_unetplusplus_resnet34_ce_cosine"
  "023_final_fpn_resnet34_ce_cosine"
  "024_final_pspnet_resnet34_ce_cosine"
  "025_final_deeplabv3plus_resnet50_ce_cosine"
)

for experiment in "${experiments[@]}"; do
  config="configs/experiments/${experiment}.yaml"
  checkpoint="outputs/train/${experiment}/checkpoints/best.pt"

  make model-shape CONFIG="${config}"
  make train CONFIG="${config}" RUN_NAME="${experiment}"
  make eval \
    CONFIG="${config}" \
    CHECKPOINT="${checkpoint}" \
    RUN_NAME="${experiment}_eval" \
    EVAL_SPLIT=val \
    MAX_EXAMPLES=5
done

make report
