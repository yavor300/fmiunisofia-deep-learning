# Experiment Configs

This directory contains reproducible YAML configs for the project experiments. Each config overrides `configs/default.yaml`; the training and evaluation commands save the resolved config next to the run outputs.

## Naming Convention

Config names start with a numeric ID so the experiment history stays readable:

- `000`: mandatory majority-class baseline.
- `001`, `008`-`013`: phase and smoke experiments.
- `020`-`033`: first final comparison set.
- `034`-`039`: model-capacity and stronger-encoder checks.
- `040`-`048`: priority expansion experiments centered on CE+Dice and stronger models.
- `049`: optional SegFormer experiment.

Keep the historical configs. They are useful for smoke tests, phase evidence, and ablation comparisons, even if only a subset is used in the final report.

## Baseline

```bash
make baseline RUN_NAME=000_baseline_majority
```

This should be the first row in `reports/experiment_results.csv` and `reports/model_report.xlsx`.

## Smoke And Phase Checks

Use these when validating that the pipeline still works without running the full final suite:

| Config | Purpose |
|---|---|
| `001_tiny_unet_ce_step.yaml` | Small Tiny U-Net training run with cross-entropy and StepLR. |
| `008_deeplabv3plus_resnet50_focal_cosine.yaml` | Early DeepLabV3+ focal-loss check. |
| `009_tiny_unet_dice_step.yaml` | Dice-loss check. |
| `010_tiny_unet_focal_step.yaml` | Focal-loss check. |
| `011_tiny_unet_resize_only.yaml` | Resize-only preprocessing check. |
| `012_tiny_unet_basic_aug.yaml` | Basic augmentation check. |
| `013_tiny_unet_strong_aug_imagenet.yaml` | Strong augmentation plus ImageNet normalization check. |

Example:

```bash
make train CONFIG=configs/experiments/001_tiny_unet_ce_step.yaml RUN_NAME=001_tiny_unet_ce_step
```

## First Final Comparison Set

These configs compare architectures, schedulers, losses, and preprocessing while keeping the setup controlled.

| Group | Configs | Purpose |
|---|---|---|
| Architecture | `020`-`025` | Tiny U-Net, U-Net, U-Net++, FPN, PSPNet, DeepLabV3+. |
| Scheduler | `026`-`027` | Step decay and ReduceLROnPlateau against the cosine setup. |
| Loss | `028`-`031` | Dice, focal, CE+Dice, focal+Dice. |
| Preprocessing | `032`-`033` | Resize-only and stronger augmentation against the basic setup. |

Recommended anchor config:

```bash
make train CONFIG=configs/experiments/030_final_unet_resnet34_ce_dice_cosine.yaml RUN_NAME=030_final_unet_resnet34_ce_dice_cosine
```

## Capacity And Encoder Checks

These configs test whether model capacity or encoder strength changes the result:

| Config | Purpose |
|---|---|
| `034_final_tiny_unet_16_ce_cosine.yaml` | Tiny U-Net with fewer base channels. |
| `035_final_tiny_unet_32_ce_cosine.yaml` | Tiny U-Net medium capacity. |
| `036_final_tiny_unet_64_ce_cosine.yaml` | Tiny U-Net larger capacity. |
| `037_final_unet_resnet18_ce_cosine.yaml` | U-Net with lighter ResNet encoder. |
| `038_final_unet_resnet50_ce_cosine.yaml` | U-Net with stronger ResNet encoder. |
| `039_final_deeplabv3plus_resnet101_ce_cosine.yaml` | DeepLabV3+ with ResNet101 encoder. |

## Priority Expansion Experiments

These are the strongest final-report candidates and ablations:

| Config | Purpose |
|---|---|
| `040_unet_resnet34_ce_dice_50ep.yaml` | Longer training for the U-Net ResNet34 CE+Dice direction. |
| `041_unet_resnet34_ce_dice_strong_aug.yaml` | Strong augmentation with U-Net ResNet34 CE+Dice. |
| `042_fpn_resnet34_ce_dice.yaml` | FPN comparison using CE+Dice. |
| `043_deeplabv3plus_resnet101_ce_dice.yaml` | DeepLabV3+ ResNet101 with CE+Dice. |
| `044_unetplusplus_resnet34_ce_dice.yaml` | U-Net++ ResNet34 with CE+Dice. |
| `045_unet_efficientnet_b3_ce_dice.yaml` | U-Net with EfficientNet-B3 encoder. |
| `046_fpn_efficientnet_b3_ce_dice.yaml` | FPN with EfficientNet-B3 encoder. |
| `047_deeplabv3plus_efficientnet_b3_ce_dice.yaml` | DeepLabV3+ with EfficientNet-B3 encoder. |
| `048_unet_resnet34_ce_lovasz.yaml` | U-Net ResNet34 with CE+Lovasz as an IoU-oriented loss. |

## Optional SegFormer

```bash
make model-shape CONFIG=configs/experiments/049_segformer_mit_b1_ce_dice_optional.yaml
make train CONFIG=configs/experiments/049_segformer_mit_b1_ce_dice_optional.yaml RUN_NAME=049_segformer_mit_b1_ce_dice_optional
```

This run depends on `transformers` and Hugging Face model availability. If downloads are rate-limited, set `HF_TOKEN` in `.env`.

## Train And Evaluate Pattern

For every experiment you want in the report, train and then evaluate the best checkpoint:

```bash
make train CONFIG=configs/experiments/<experiment_id>.yaml RUN_NAME=<experiment_id>
make eval CONFIG=configs/experiments/<experiment_id>.yaml CHECKPOINT=outputs/train/<experiment_id>/checkpoints/best.pt RUN_NAME=<experiment_id>_eval EVAL_SPLIT=val MAX_EXAMPLES=5
```

Both training and evaluation append rows to `reports/experiment_results.csv`. The model report is regenerated from that CSV:

```bash
make report
```

## Batch Runner

The batch runner contains grouped arrays for the final experiment suite:

```bash
bash scripts/run_final_architecture_experiments.sh
```

Open the script before running and uncomment the groups you want in the `experiments=(...)` array. This avoids accidentally launching every expensive experiment at once.

## Report Metric

Mean IoU is the primary ranking metric. Mean Dice and pixel accuracy are included as supporting metrics, with percentage changes versus the baseline.
