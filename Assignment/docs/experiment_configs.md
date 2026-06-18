# Experiment Config Index

These configs cover the baseline, loss comparisons, scheduler comparisons, architecture comparisons, and preprocessing comparisons.

## Baseline

```bash
make baseline RUN_NAME=000_baseline_majority
```

## Loss and Scheduler Checks

```bash
make train CONFIG=configs/experiments/001_tiny_unet_ce_step.yaml RUN_NAME=001_tiny_unet_ce_step
make train CONFIG=configs/experiments/002_tiny_unet_ce_cosine.yaml RUN_NAME=002_tiny_unet_ce_cosine
make train CONFIG=configs/experiments/009_tiny_unet_dice_step.yaml RUN_NAME=009_tiny_unet_dice_step
make train CONFIG=configs/experiments/010_tiny_unet_focal_step.yaml RUN_NAME=010_tiny_unet_focal_step
```

## Architecture Comparisons

```bash
make train CONFIG=configs/experiments/003_unet_resnet34_ce_cosine.yaml RUN_NAME=003_unet_resnet34_ce_cosine
make train CONFIG=configs/experiments/004_unetplusplus_resnet34_ce_cosine.yaml RUN_NAME=004_unetplusplus_resnet34_ce_cosine
make train CONFIG=configs/experiments/005_fpn_resnet34_ce_cosine.yaml RUN_NAME=005_fpn_resnet34_ce_cosine
make train CONFIG=configs/experiments/006_pspnet_resnet34_ce_cosine.yaml RUN_NAME=006_pspnet_resnet34_ce_cosine
make train CONFIG=configs/experiments/007_deeplabv3plus_resnet50_ce_cosine.yaml RUN_NAME=007_deeplabv3plus_resnet50_ce_cosine
make train CONFIG=configs/experiments/008_deeplabv3plus_resnet50_focal_cosine.yaml RUN_NAME=008_deeplabv3plus_resnet50_focal_cosine
```

## Preprocessing Comparisons

```bash
make train CONFIG=configs/experiments/011_tiny_unet_resize_only.yaml RUN_NAME=011_tiny_unet_resize_only
make train CONFIG=configs/experiments/012_tiny_unet_basic_aug.yaml RUN_NAME=012_tiny_unet_basic_aug
make train CONFIG=configs/experiments/013_tiny_unet_strong_aug_imagenet.yaml RUN_NAME=013_tiny_unet_strong_aug_imagenet
```

## Evaluation Pattern

After a training run finishes, evaluate the best checkpoint:

```bash
make eval CONFIG=configs/experiments/003_unet_resnet34_ce_cosine.yaml CHECKPOINT=outputs/train/003_unet_resnet34_ce_cosine/checkpoints/best.pt RUN_NAME=003_unet_resnet34_ce_cosine_eval EVAL_SPLIT=val MAX_EXAMPLES=5
```

All SMP configs use `encoder_weights: null` so they do not require downloading ImageNet weights. Change this to `imagenet` only when the environment can download or already has cached encoder weights.
