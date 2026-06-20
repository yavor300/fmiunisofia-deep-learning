#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

#make baseline RUN_NAME=000_baseline_majority

# architecture_experiments=(
#   "020_final_tiny_unet_ce_cosine"
#   "021_final_unet_resnet34_ce_cosine"
#   "022_final_unetplusplus_resnet34_ce_cosine"
#   "023_final_fpn_resnet34_ce_cosine"
#   "024_final_pspnet_resnet34_ce_cosine"
#   "025_final_deeplabv3plus_resnet50_ce_cosine"
# )

scheduler_experiments=(
  "026_final_unet_resnet34_ce_step"
  "027_final_unet_resnet34_ce_plateau"
)

loss_experiments=(
  "028_final_unet_resnet34_dice_cosine"
  "029_final_unet_resnet34_focal_cosine"
  "030_final_unet_resnet34_ce_dice_cosine"
  "031_final_unet_resnet34_focal_dice_cosine"
)

preprocessing_experiments=(
  "032_final_unet_resnet34_resize_only_ce_cosine"
  "033_final_unet_resnet34_strong_aug_ce_cosine"
)

experiments=(
#  "${architecture_experiments[@]}"
  "${scheduler_experiments[@]}"
  "${loss_experiments[@]}"
  "${preprocessing_experiments[@]}"
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
