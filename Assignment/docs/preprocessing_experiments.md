# Preprocessing Experiments

Phase 10 adds Albumentations pipelines for comparing input preprocessing choices.

| Config | Strategy | Normalization |
| --- | --- | --- |
| `configs/experiments/011_tiny_unet_resize_only.yaml` | Resize only | None |
| `configs/experiments/012_tiny_unet_basic_aug.yaml` | Flip, brightness/contrast, random crop | None |
| `configs/experiments/013_tiny_unet_strong_aug_imagenet.yaml` | Flip, brightness/contrast, blur, scale/crop | ImageNet |

Each experiment uses the same Tiny U-Net and cross-entropy loss so differences can be attributed primarily to preprocessing.

Training writes `history.csv`, checkpoints, plots, and resolved config files under `outputs/train/<run-name>/`. Add the resulting validation metrics to `reports/experiment_results.csv` for the final model report comparison.
