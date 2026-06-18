# Cityscapes Semantic Segmentation

This project builds an end-to-end semantic segmentation pipeline for urban street scenes using the Cityscapes dataset. The goal is to train and compare segmentation models, evaluate them with reproducible experiments, and provide a Streamlit app for visual predictions.

## Dataset

Cityscapes must be downloaded manually from the official Cityscapes website because access is gated by its license. After downloading, place the files under:

```text
data/raw/cityscapes/
├── leftImg8bit/
│   ├── train/
│   ├── val/
│   └── test/
└── gtFine/
    ├── train/
    ├── val/
    └── test/
```

The raw and processed dataset folders are ignored by git.

## Installation

Create or activate a virtual environment, then install dependencies:

```bash
make install
```

If you are using the existing environment in this folder:

```bash
source .venv/bin/activate
make install
```

## Training

Run the mandatory CPU-only majority-class baseline:

```bash
make baseline
```

This writes `reports/experiment_results.csv`, `reports/model_report.xlsx`, and a resolved config under `outputs/baseline/<run-name>/`.

For a quick smoke run on a few masks:

```bash
make baseline RUN_NAME=baseline_smoke MAX_TRAIN_MASKS=5 MAX_EVAL_MASKS=5
```

Validate that a configured model returns logits with the expected shape:

```bash
make model-shape CONFIG=configs/experiments/001_tiny_unet_ce_step.yaml
```

Run training with the default config:

```bash
make train
```

Or pass another config:

```bash
make train CONFIG=configs/experiments/001_tiny_unet_ce_step.yaml
```

Use a stable output folder name when you want reproducible artifacts:

```bash
make train CONFIG=configs/experiments/001_tiny_unet_ce_step.yaml RUN_NAME=demo_train
```

Each run writes `resolved_config.yaml` under `outputs/train/<run-name>/`.

Resume training from the last checkpoint:

```bash
make train CONFIG=configs/experiments/001_tiny_unet_ce_step.yaml RUN_NAME=demo_train RESUME=outputs/train/demo_train/checkpoints/last.pt
```

Compare learning-rate schedulers with matched Tiny U-Net runs:

```bash
make train CONFIG=configs/experiments/001_tiny_unet_ce_step.yaml RUN_NAME=tiny_unet_step
make train CONFIG=configs/experiments/002_tiny_unet_ce_cosine.yaml RUN_NAME=tiny_unet_cosine
```

Each run writes `history.csv` and `learning_rate.png`.

Compare preprocessing strategies with matched Tiny U-Net runs:

```bash
make train CONFIG=configs/experiments/011_tiny_unet_resize_only.yaml RUN_NAME=pre_resize_only
make train CONFIG=configs/experiments/012_tiny_unet_basic_aug.yaml RUN_NAME=pre_basic_aug
make train CONFIG=configs/experiments/013_tiny_unet_strong_aug_imagenet.yaml RUN_NAME=pre_strong_aug
```

## Evaluation

Evaluate a checkpoint with:

```bash
make eval CONFIG=configs/default.yaml CHECKPOINT=checkpoints/model.pt
```

Evaluation also supports named output folders:

```bash
make eval CONFIG=configs/experiments/001_tiny_unet_ce_step.yaml CHECKPOINT=checkpoints/model.pt RUN_NAME=demo_eval
```

Each evaluation run writes `resolved_config.yaml` under `outputs/eval/<run-name>/`.

## Exploratory Data Analysis

Once the dataset is available, generate dataset analysis outputs:

```bash
make eda
```

## Model Report

Build the experiment report from `reports/experiment_results.csv`:

```bash
make report
```

## Streamlit App

Launch the demo app:

```bash
make app
```

## Tests

Tests use BDD-style `unittest` test cases and are designed to run without the Cityscapes dataset by using synthetic data:

```bash
make test
```
