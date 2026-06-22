# Cityscapes Semantic Segmentation

End-to-end semantic segmentation project for urban street scenes using the Cityscapes dataset. The project includes dataset loading, exploratory data analysis, reproducible YAML experiments, baseline and neural models, training, evaluation, experiment logging, an Excel model report, and a Streamlit inference app.

## What Is Implemented

- Cityscapes label conversion to 19 train IDs plus `255` ignore labels.
- Synthetic-data tests that run without downloading Cityscapes.
- Dataset EDA with class imbalance charts, image-size statistics, overlays, rare-class examples, and anomaly checks.
- Majority-class baseline for the first report row.
- Tiny U-Net, SMP architectures, and optional SegFormer support.
- Losses: cross-entropy, Dice, focal, CE+Dice, focal+Dice, and CE+Lovasz.
- Metrics: mean IoU, mean Dice, pixel accuracy, per-class IoU, and confusion matrix.
- Training with mixed precision, checkpointing, resume support, schedulers, tqdm progress, plots, and history CSV.
- Evaluation with qualitative examples, overlays, error maps, per-class metrics, and report logging.
- `reports/model_report.xlsx` generation with highlighted best model and training curves.
- Streamlit app for checkpoint inference without requiring the full dataset.

## Dataset

Cityscapes must be downloaded manually from the official Cityscapes website because access is gated by its license. Place the extracted files here:

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

Use `train` for fitting, `val` for model comparison and report metrics, and reserve `test` for final inference-style runs. Cityscapes test masks may not include public ground-truth labels, so validation is the practical evaluation split for this coursework pipeline.

## Installation

Create or activate a virtual environment, then install dependencies:

```bash
source .venv/bin/activate
make install
```

If Hugging Face downloads are needed for optional models, create `.env` with `HF_TOKEN=<your-token>`. The project loads it before model creation.

## Quick Health Checks

Run the test suite:

```bash
make test
```

Validate that a model config can be instantiated:

```bash
make model-shape CONFIG=configs/experiments/001_tiny_unet_ce_step.yaml
```

## EDA

Generate dataset analysis, figures, and written comments:

```bash
make eda
```

Outputs include `docs/dataset_analysis.md` and figures in `reports/figures/`.

## Baseline

Run the mandatory CPU-friendly majority-class baseline:

```bash
make baseline RUN_NAME=000_baseline_majority
```

For a quick smoke run:

```bash
make baseline RUN_NAME=baseline_smoke MAX_TRAIN_MASKS=5 MAX_EVAL_MASKS=5
```

## Training

Train any experiment config:

```bash
make train CONFIG=configs/experiments/030_final_unet_resnet34_ce_dice_cosine.yaml RUN_NAME=030_final_unet_resnet34_ce_dice_cosine
```

Resume from a checkpoint:

```bash
make train CONFIG=configs/experiments/030_final_unet_resnet34_ce_dice_cosine.yaml RUN_NAME=030_final_unet_resnet34_ce_dice_cosine RESUME=outputs/train/030_final_unet_resnet34_ce_dice_cosine/checkpoints/last.pt
```

Training outputs go to `outputs/train/<run-name>/` and include:

- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `resolved_config.yaml`
- `history.csv`
- loss, metric, learning-rate, and preview figures

## Evaluation

Evaluate the best checkpoint from a training run:

```bash
make eval CONFIG=configs/experiments/030_final_unet_resnet34_ce_dice_cosine.yaml CHECKPOINT=outputs/train/030_final_unet_resnet34_ce_dice_cosine/checkpoints/best.pt RUN_NAME=030_final_unet_resnet34_ce_dice_cosine_eval EVAL_SPLIT=val MAX_EXAMPLES=5
```

Evaluation outputs go to `outputs/eval/<run-name>/` and `reports/figures/`. They include global metrics, per-class IoU, a confusion matrix, qualitative examples, overlays, error maps, and an error-analysis summary.

## Recommended End-to-End Flow

```bash
make test
make eda
make baseline RUN_NAME=000_baseline_majority
make train CONFIG=configs/experiments/030_final_unet_resnet34_ce_dice_cosine.yaml RUN_NAME=030_final_unet_resnet34_ce_dice_cosine
make eval CONFIG=configs/experiments/030_final_unet_resnet34_ce_dice_cosine.yaml CHECKPOINT=outputs/train/030_final_unet_resnet34_ce_dice_cosine/checkpoints/best.pt RUN_NAME=030_final_unet_resnet34_ce_dice_cosine_eval EVAL_SPLIT=val MAX_EXAMPLES=5
make report
```

For the larger final comparison, use:

```bash
bash scripts/run_final_architecture_experiments.sh
```

Before running it, open the script and uncomment the experiment groups you want. The optional SegFormer run is kept separate because it depends on `transformers` and Hugging Face model availability.

## Model Report

Build the Excel report from `reports/experiment_results.csv`:

```bash
make report
```

The report is written to `reports/model_report.xlsx`. It keeps hyperparameters first, then the three main metrics: mean IoU, mean Dice, and pixel accuracy. Mean IoU is the primary metric used to identify the best model.

## Streamlit App

Launch the inference UI:

```bash
make app
```

The app accepts a checkpoint path and an uploaded image. By default it reads the model architecture and encoder from the checkpoint config, which avoids mismatches such as loading FPN weights into Tiny U-Net. You can manually override the architecture and encoder from the sidebar when needed.

## Experiment Configs

Experiment configs live in `configs/experiments/`. See `configs/experiments/README.md` for the complete config map and recommended usage.

## Tests

Tests use BDD-style `unittest` test cases and synthetic data:

```bash
make test
```

The test naming convention follows `test_when_<condition>_then_<expectation>`.
