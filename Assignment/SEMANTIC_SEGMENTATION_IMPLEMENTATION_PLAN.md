# Semantic Segmentation of Urban Street Scenes — Implementation Plan for Codex Agent

## 1. Project Goal

Build a complete deep learning course project for **semantic segmentation of urban street scene images**.

The final system must:

1. Load and explore a semantic segmentation image dataset.
2. Train multiple segmentation models.
3. Compare multiple loss functions:
   - pixel-wise cross-entropy loss;
   - Dice loss;
   - focal loss;
   - optional combined losses, such as cross-entropy + Dice.
4. Demonstrate learning-rate scheduling:
   - step decay;
   - cosine annealing.
5. Experiment with different input preprocessing and augmentation strategies.
6. Produce a structured **model report file** with all experiments.
7. Provide a **Streamlit web application** where a user uploads an image and receives a segmentation mask overlay.
8. Include tests following behavior-driven naming conventions.

The recommended project title is:

> **Semantic Segmentation of Urban Street Scenes using U-Net and Deep Learning**

---

## 2. Core Dataset Decision

Use **Cityscapes** as the target dataset when available.

However, because Cityscapes requires registration and can be large, implement the data layer so that it can also work with a smaller compatible dataset or a prepared subset.

### Supported dataset layout

The implementation should support this structure:

```text
data/
  raw/
    cityscapes/
      leftImg8bit/
        train/
        val/
        test/
      gtFine/
        train/
        val/
        test/
  processed/
    train/
      images/
      masks/
    val/
      images/
      masks/
    test/
      images/
      masks/
```

For the MVP, prioritize the `processed/` format because it is simple and easier to use in experiments.

Each image must have a corresponding segmentation mask.

Example naming convention:

```text
data/processed/train/images/aachen_000000_000019_leftImg8bit.png
data/processed/train/masks/aachen_000000_000019_gtFine_labelIds.png
```

---

## 3. Target Classes

Start with a reduced class set to keep the project realistic.

### MVP class mapping

Implement a configurable class mapping, for example:

| Original concept | Reduced class |
|---|---|
| road, sidewalk | flat |
| building, wall, fence | construction |
| pole, traffic light, traffic sign | object |
| vegetation, terrain | nature |
| sky | sky |
| person, rider | human |
| car, truck, bus, train, motorcycle, bicycle | vehicle |
| unlabeled / ignored | ignore |

The implementation must allow two modes:

1. **Reduced mode** — recommended for faster experiments.
2. **Full mode** — optional, for all Cityscapes semantic classes.

Use an `ignore_index`, for example `255`, for pixels that should not contribute to the loss.

---

## 4. Required Repository Structure

Create the project with the following structure:

```text
semantic-segmentation-project/
  README.md
  requirements.txt
  pyproject.toml                         # optional but recommended
  .gitignore
  .env.example

  configs/
    default.yaml
    experiments/
      exp_001_baseline_unet.yaml
      exp_002_unet_dice_loss.yaml
      exp_003_unet_focal_loss.yaml
      exp_004_unet_augmented.yaml
      exp_005_unet_step_decay.yaml
      exp_006_unet_cosine_annealing.yaml
      exp_007_resnet_unet_transfer.yaml

  data/
    README.md
    raw/
    processed/

  notebooks/
    01_dataset_exploration.ipynb         # optional
    02_visualize_predictions.ipynb       # optional

  reports/
    figures/
    predictions/
    model_report.xlsx
    model_report_template.md
    final_project_summary.md

  src/
    __init__.py
    app.py                               # Streamlit app

    data/
      __init__.py
      dataset.py
      transforms.py
      class_mapping.py
      split_dataset.py
      validate_dataset.py

    models/
      __init__.py
      unet.py
      unet_resnet_encoder.py
      simple_autoencoder_segmentation.py
      model_factory.py

    training/
      __init__.py
      train.py
      evaluate.py
      losses.py
      metrics.py
      schedulers.py
      experiment_runner.py
      checkpointing.py
      reproducibility.py

    visualization/
      __init__.py
      masks.py
      plots.py
      predictions.py

    reporting/
      __init__.py
      experiment_logger.py
      model_report.py

    utils/
      __init__.py
      paths.py
      config.py
      device.py

  tests/
    test_dataset.py
    test_losses.py
    test_metrics.py
    test_model_output_shape.py
    test_class_mapping.py
    test_streamlit_smoke.py
```

---

## 5. Implementation Principles

Follow these principles throughout the project:

1. Prefer scripts and configuration files over manually edited notebooks.
2. Each experiment must be reproducible.
3. Set random seeds globally.
4. Keep train, validation, and test strictly separated.
5. Apply data augmentation only to the training set.
6. Track both loss and metrics for training and validation.
7. Save model checkpoints and experiment metadata.
8. Every trained model must produce one row in the model report.
9. The Streamlit app must use the best saved model, not retrain a model.
10. Keep the MVP small enough to run on a normal laptop or Google Colab.

---

## 6. Stage 1 — Environment Setup

### 6.1 Create virtual environment

Implement documentation for:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 6.2 Create `requirements.txt`

Include at least:

```text
torch
torchvision
torchmetrics
numpy
pandas
matplotlib
pillow
opencv-python
scikit-learn
pyyaml
tqdm
streamlit
openpyxl
pytest
```

Optional:

```text
segmentation-models-pytorch
albumentations
```

### 6.3 Add reproducibility utility

Create:

```text
src/training/reproducibility.py
```

Required function:

```python
def set_seed(seed: int) -> None:
    ...
```

It must set seeds for:

- Python random;
- NumPy;
- PyTorch CPU;
- PyTorch CUDA when available.

---

## 7. Stage 2 — Dataset Loading and Validation

### 7.1 Implement custom PyTorch dataset

Create:

```text
src/data/dataset.py
```

Implement:

```python
class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_transform=None,
        mask_transform=None,
        joint_transform=None,
        class_mapper=None,
    ):
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        ...
```

Requirements:

1. Load images with PIL.
2. Convert RGB images to tensors.
3. Load masks as integer class IDs.
4. Do not normalize masks as images.
5. Ensure returned image shape is:

```text
[C, H, W]
```

6. Ensure returned mask shape is:

```text
[H, W]
```

7. Ensure image dtype is:

```text
torch.float32
```

8. Ensure mask dtype is:

```text
torch.long
```

### 7.2 Image and mask transformation rules

For images:

- Use tensor conversion that scales image values to `[0.0, 1.0]`.
- Resize images to a configurable size, initially `128x256` or `256x512`.
- Optionally normalize using ImageNet mean/std if using a pretrained encoder.

For masks:

- Resize masks using nearest-neighbor interpolation only.
- Never use bilinear interpolation for class masks.
- Never apply color jitter to masks.

### 7.3 Implement dataset validation script

Create:

```text
src/data/validate_dataset.py
```

It must check:

1. Every image has a corresponding mask.
2. Every mask has valid class IDs.
3. Images and masks have compatible spatial dimensions.
4. No corrupted images.
5. Class frequency distribution.

Save validation output to:

```text
reports/dataset_validation_summary.md
```

---

## 8. Stage 3 — Exploratory Data Analysis

Create an EDA script:

```text
src/data/explore_dataset.py
```

The script must generate:

1. Number of images per split.
2. Image size distribution.
3. Mask class distribution.
4. Pixel frequency per class.
5. Example image-mask pairs.
6. Overlay visualizations.
7. Detection of potential anomalies:
   - empty masks;
   - masks with only ignore pixels;
   - very small images;
   - unexpected class IDs.

Save figures to:

```text
reports/figures/
```

Recommended figures:

```text
class_distribution_train.png
class_distribution_val.png
image_size_distribution.png
sample_image_mask_pairs.png
sample_overlays.png
```

---

## 9. Stage 4 — Baseline Model

The first row in the model report must be a baseline.

For semantic segmentation, implement two baselines:

### 9.1 Majority pixel baseline

Predict the most frequent class from the training masks for every pixel.

Example:

```text
Every pixel -> road/flat
```

### 9.2 Optional random baseline

Predict classes according to the training class distribution.

### 9.3 Baseline metrics

Evaluate the baseline on the validation and test sets using:

1. mean Intersection over Union, `mIoU`;
2. pixel accuracy;
3. Dice score.

Do not use more than three main metrics in the model report.

---

## 10. Stage 5 — Metrics

Create:

```text
src/training/metrics.py
```

Implement the following:

```python
def pixel_accuracy(preds: torch.Tensor, targets: torch.Tensor, ignore_index: int = 255) -> float:
    ...

def mean_iou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: int = 255) -> float:
    ...

def dice_score(preds: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: int = 255) -> float:
    ...
```

Expected model output shape:

```text
[B, num_classes, H, W]
```

Expected target shape:

```text
[B, H, W]
```

Prediction step:

```python
predicted_mask = logits.argmax(dim=1)
```

Metric requirements:

1. Ignore pixels equal to `ignore_index`.
2. Return macro-averaged metrics across classes.
3. Avoid division-by-zero errors.
4. Include unit tests.

---

## 11. Stage 6 — Loss Functions

Create:

```text
src/training/losses.py
```

Implement:

### 11.1 Pixel-wise cross-entropy loss

Use:

```python
torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
```

Important:

- The model must return raw logits.
- Do not apply softmax before `CrossEntropyLoss`.

### 11.2 Dice loss

Implement multiclass Dice loss.

Requirements:

1. Convert targets to one-hot representation.
2. Apply softmax to logits inside the loss.
3. Ignore `ignore_index`.
4. Use smoothing epsilon to avoid division by zero.

### 11.3 Focal loss

Implement multiclass focal loss.

Suggested parameters:

```python
gamma = 2.0
alpha = None
```

### 11.4 Combined loss

Optional but recommended:

```python
combined_loss = cross_entropy_loss + dice_loss
```

### 11.5 Tests

Add tests verifying:

1. Loss returns scalar tensor.
2. Loss decreases when prediction is closer to target.
3. Ignore index is handled correctly.
4. Loss works with batch size greater than 1.

---

## 12. Stage 7 — Model Architectures

### 12.1 Simple segmentation autoencoder

Create:

```text
src/models/simple_autoencoder_segmentation.py
```

Implement a small encoder-decoder CNN.

Purpose:

- verify training pipeline;
- establish simple deep learning baseline;
- run quickly.

Architecture:

```text
Input image
  -> Conv + ReLU + MaxPool
  -> Conv + ReLU + MaxPool
  -> bottleneck
  -> ConvTranspose / Upsample
  -> ConvTranspose / Upsample
  -> 1x1 Conv
  -> logits [B, num_classes, H, W]
```

### 12.2 U-Net

Create:

```text
src/models/unet.py
```

Implement U-Net with:

1. encoder blocks;
2. bottleneck;
3. decoder blocks;
4. skip connections;
5. final `1x1` convolution.

Requirements:

1. Input: `[B, 3, H, W]`.
2. Output: `[B, num_classes, H, W]`.
3. Use `BatchNorm2d` optionally.
4. Use `ReLU` or `LeakyReLU`.
5. Use bilinear upsampling or transposed convolutions as configurable option.

### 12.3 Optional transfer learning model

Create:

```text
src/models/unet_resnet_encoder.py
```

Use a pretrained ResNet encoder if feasible.

This is optional for MVP but recommended for a stronger final project.

### 12.4 Model factory

Create:

```text
src/models/model_factory.py
```

Implement:

```python
def create_model(model_name: str, num_classes: int, **kwargs) -> torch.nn.Module:
    ...
```

Supported names:

```text
simple_autoencoder
unet
unet_resnet
```

---

## 13. Stage 8 — Training Pipeline

Create:

```text
src/training/train.py
```

Implement a reusable training loop.

### 13.1 Required inputs

The training script must read a YAML config:

```bash
python -m src.training.train --config configs/experiments/exp_001_baseline_unet.yaml
```

### 13.2 Config fields

Example:

```yaml
experiment:
  id: exp_001
  name: baseline_unet_cross_entropy
  seed: 42

data:
  train_images: data/processed/train/images
  train_masks: data/processed/train/masks
  val_images: data/processed/val/images
  val_masks: data/processed/val/masks
  test_images: data/processed/test/images
  test_masks: data/processed/test/masks
  image_size: [128, 256]
  num_classes: 7
  ignore_index: 255
  class_mapping: reduced

model:
  name: unet
  base_channels: 32
  bilinear: true
  batch_norm: true
  dropout: 0.0

training:
  epochs: 20
  batch_size: 4
  optimizer: adamw
  learning_rate: 0.001
  weight_decay: 0.0001
  loss: cross_entropy
  scheduler: none
  gradient_clip_norm: 1.0

augmentation:
  enabled: false
  horizontal_flip: false
  random_crop: false
  color_jitter: false

outputs:
  checkpoint_dir: reports/checkpoints/exp_001
  prediction_dir: reports/predictions/exp_001
  metrics_file: reports/experiments/exp_001_metrics.json
```

### 13.3 Training loop requirements

For each epoch:

1. Set model to train mode.
2. Loop over training batches.
3. Move tensors to device.
4. Forward pass.
5. Calculate loss.
6. Backpropagate.
7. Apply gradient clipping if configured.
8. Optimizer step.
9. Calculate training metrics.
10. Run validation loop with `model.eval()` and `torch.no_grad()`.
11. Save epoch-level metrics.
12. Save best checkpoint based on validation mIoU.

Track:

```text
train_loss
val_loss
train_mIoU
val_mIoU
train_pixel_accuracy
val_pixel_accuracy
train_dice
val_dice
learning_rate
epoch_duration
```

---

## 14. Stage 9 — Learning Rate Scheduling

Create:

```text
src/training/schedulers.py
```

Support:

### 14.1 No scheduler

```yaml
scheduler: none
```

### 14.2 Step decay

```yaml
scheduler:
  name: step_decay
  step_size: 10
  gamma: 0.1
```

Use:

```python
torch.optim.lr_scheduler.StepLR
```

### 14.3 Cosine annealing

```yaml
scheduler:
  name: cosine_annealing
  t_max: 20
  eta_min: 0.00001
```

Use:

```python
torch.optim.lr_scheduler.CosineAnnealingLR
```

Log the learning rate after every epoch.

---

## 15. Stage 10 — Input Processing and Augmentation Experiments

Implement image augmentations carefully because semantic segmentation requires identical spatial transforms for image and mask.

### 15.1 Safe augmentations

Use only augmentations that preserve pixel labels:

1. horizontal flip;
2. resize;
3. random crop;
4. small rotation if the mask is transformed in the same way;
5. brightness/contrast adjustment for image only;
6. normalization.

### 15.2 Avoid unsafe augmentations

Avoid:

1. vertical flip for street scenes;
2. strong color shifts that change visual meaning;
3. perspective distortions unless carefully justified;
4. any mask interpolation except nearest neighbor.

### 15.3 Required preprocessing experiments

Create experiments for:

1. image size `128x256`;
2. image size `256x512`;
3. no augmentation;
4. horizontal flip augmentation;
5. horizontal flip + color jitter;
6. normalization vs no normalization.

---

## 16. Stage 11 — Experiment Plan

Implement at least the following experiments.

| ID | Model | Loss | Scheduler | Augmentation | Input size | Purpose |
|---|---|---|---|---|---|---|
| baseline_001 | majority pixel | none | none | none | original | non-neural baseline |
| exp_001 | simple autoencoder | cross-entropy | none | none | 128x256 | verify pipeline |
| exp_002 | U-Net | cross-entropy | none | none | 128x256 | first strong model |
| exp_003 | U-Net | Dice loss | none | none | 128x256 | compare loss functions |
| exp_004 | U-Net | focal loss | none | none | 128x256 | handle class imbalance |
| exp_005 | U-Net | CE + Dice | none | none | 128x256 | combined loss |
| exp_006 | U-Net | CE + Dice | step decay | none | 128x256 | scheduler comparison |
| exp_007 | U-Net | CE + Dice | cosine annealing | none | 128x256 | scheduler comparison |
| exp_008 | U-Net | CE + Dice | cosine annealing | horizontal flip | 128x256 | augmentation |
| exp_009 | U-Net | CE + Dice | cosine annealing | flip + color jitter | 128x256 | stronger augmentation |
| exp_010 | U-Net | CE + Dice | cosine annealing | best augmentation | 256x512 | higher resolution |
| exp_011 | ResNet U-Net | CE + Dice | cosine annealing | best augmentation | 256x512 | transfer learning |

The MVP is complete after `exp_008`.

Experiments `exp_009` to `exp_011` are recommended for a stronger final result.

---

## 17. Stage 12 — Model Report File

Create:

```text
reports/model_report.xlsx
```

Also create a Markdown template:

```text
reports/model_report_template.md
```

The Excel report is mandatory.

### 17.1 Model report philosophy

The report must tell the story of the modeling process.

Each row is one hypothesis / experiment.

The first row must be the baseline model.

The rows must stay in chronological experiment order.

Do not sort the rows by performance after the experiments are complete.

### 17.2 Main sheet: `Experiments`

Columns:

| Column | Description |
|---|---|
| Experiment ID | Example: `exp_004` |
| Date | Date of run |
| Model | Majority baseline, Simple Autoencoder, U-Net, ResNet U-Net |
| Input Size | Example: `128x256` |
| Num Classes | Reduced or full |
| Loss | CE, Dice, Focal, CE+Dice |
| Optimizer | AdamW, SGD |
| Learning Rate | Initial LR |
| Weight Decay | Regularization |
| Scheduler | None, StepLR, CosineAnnealingLR |
| Batch Size | Batch size |
| Epochs | Number of epochs |
| Augmentation | None, flip, flip+jitter |
| Train Loss | Final train loss |
| Val Loss | Best or final validation loss |
| Test mIoU | Main metric |
| Test mIoU Change vs Baseline | Percentage change |
| Test Pixel Accuracy | Secondary metric |
| Test Dice Score | Secondary metric |
| Checkpoint Path | Best model file |
| Comments | Interpretation and error analysis |

Use no more than three main metrics:

1. Test mIoU;
2. Test pixel accuracy;
3. Test Dice score.

### 17.3 Summary area

At the top or bottom of the report, explicitly write:

```text
Best model: <experiment_id>
Reason: <short explanation based on validation/test mIoU, loss stability, and visual predictions>
```

### 17.4 Diagrams

Add plots in the Excel file or save them in `reports/figures/` and link/reference them.

Required plots:

1. train vs validation mIoU;
2. train vs validation loss;
3. learning rate schedule;
4. class distribution;
5. comparison of loss functions;
6. comparison of schedulers.

### 17.5 Best model examples

Create a separate sheet:

```text
Best Model Examples
```

Include 4-5 examples:

1. original image;
2. ground-truth mask;
3. predicted mask;
4. overlay;
5. short comment.

Add both correct-looking and incorrect-looking predictions.

### 17.6 Styling requirements

The report should not look like a raw pandas export.

Implement styling with `openpyxl`:

1. bold header row;
2. freeze top row;
3. color important cells;
4. highlight best metric;
5. add comments;
6. set column widths;
7. use readable number formatting;
8. include clear title and best-model statement.

---

## 18. Stage 13 — Prediction Visualization

Create:

```text
src/visualization/masks.py
src/visualization/predictions.py
```

Implement:

```python
def decode_segmentation_mask(mask: np.ndarray, color_map: dict[int, tuple[int, int, int]]) -> np.ndarray:
    ...

def overlay_mask_on_image(image: Image.Image, color_mask: np.ndarray, alpha: float = 0.5) -> Image.Image:
    ...

def save_prediction_grid(image, ground_truth, prediction, output_path):
    ...
```

Requirements:

1. Use consistent colors for classes.
2. Include a legend in saved visualizations.
3. Save visual comparison figures after each experiment.
4. Support Streamlit visualization.

---

## 19. Stage 14 — Evaluation Script

Create:

```text
src/training/evaluate.py
```

Command:

```bash
python -m src.training.evaluate \
  --config configs/experiments/exp_008_unet_augmented.yaml \
  --checkpoint reports/checkpoints/exp_008/best_model.pt \
  --split test
```

The evaluation script must:

1. Load the model.
2. Load the requested split.
3. Calculate test metrics.
4. Save predictions for selected samples.
5. Append final test results to the model report.
6. Save a JSON metrics file.

---

## 20. Stage 15 — Streamlit Web Application

Create:

```text
src/app.py
```

Run with:

```bash
streamlit run src/app.py
```

### 20.1 App functionality

The app must:

1. Load the best trained model checkpoint.
2. Allow image upload.
3. Preprocess the uploaded image.
4. Run model inference.
5. Display:
   - original image;
   - predicted segmentation mask;
   - overlay image;
   - class legend.
6. Allow selecting transparency value for overlay.
7. Show short explanation of predicted classes.
8. Optionally allow choosing among several saved model checkpoints.

### 20.2 App layout

Suggested layout:

```text
Title: Semantic Segmentation of Urban Street Scenes

Sidebar:
  - Model checkpoint
  - Image size
  - Overlay alpha
  - Class mapping mode

Main:
  - Upload image
  - Original image
  - Predicted mask
  - Overlay
  - Class legend
```

### 20.3 App constraints

The app must not train a model.

The app must fail gracefully if no checkpoint exists.

---

## 21. Stage 16 — Tests

Use `pytest`.

Follow behavior-driven test naming:

```python
def test_when_mask_contains_ignore_index_then_metric_ignores_pixels():
    ...
```

### 21.1 Required tests

#### Dataset tests

```text
tests/test_dataset.py
```

Test:

1. dataset length;
2. image and mask shapes;
3. image dtype;
4. mask dtype;
5. invalid image/mask pairing raises error.

#### Loss tests

```text
tests/test_losses.py
```

Test:

1. cross-entropy returns scalar;
2. Dice loss returns scalar;
3. focal loss returns scalar;
4. ignore index is handled.

#### Metric tests

```text
tests/test_metrics.py
```

Test:

1. perfect prediction gives high score;
2. wrong prediction gives lower score;
3. ignore index is ignored;
4. class with zero pixels does not crash.

#### Model tests

```text
tests/test_model_output_shape.py
```

Test:

```text
input:  [2, 3, 128, 256]
output: [2, num_classes, 128, 256]
```

#### Class mapping tests

```text
tests/test_class_mapping.py
```

Test:

1. valid class IDs are mapped correctly;
2. unknown class ID becomes ignore index;
3. output mask preserves spatial shape.

---

## 22. Stage 17 — README

Create a full `README.md`.

It must include:

1. project goal;
2. dataset description;
3. setup instructions;
4. dataset preparation instructions;
5. training commands;
6. evaluation commands;
7. Streamlit app command;
8. model report explanation;
9. experiment summary table;
10. final result summary;
11. limitations;
12. future improvements.

Suggested command section:

```bash
# Install dependencies
pip install -r requirements.txt

# Validate dataset
python -m src.data.validate_dataset --config configs/default.yaml

# Run dataset exploration
python -m src.data.explore_dataset --config configs/default.yaml

# Train one experiment
python -m src.training.train --config configs/experiments/exp_001_baseline_unet.yaml

# Evaluate best checkpoint
python -m src.training.evaluate --config configs/experiments/exp_001_baseline_unet.yaml --split test

# Run Streamlit app
streamlit run src/app.py
```

---

## 23. Stage 18 — Final Presentation Support

Create:

```text
reports/final_project_summary.md
```

It must include:

1. problem definition;
2. why semantic segmentation matters;
3. dataset overview;
4. preprocessing;
5. model architectures;
6. loss functions;
7. learning-rate schedules;
8. experiments;
9. best model;
10. error analysis;
11. Streamlit demo screenshots;
12. conclusion.

---

## 24. Suggested Implementation Order for Codex

Follow this exact order.

### Phase 1 — Project skeleton

1. Create repository structure.
2. Add `requirements.txt`.
3. Add config loading.
4. Add seed utility.
5. Add README draft.

### Phase 2 — Data layer

1. Implement class mapping.
2. Implement segmentation dataset.
3. Implement transforms.
4. Implement dataset validation.
5. Implement EDA script.
6. Add dataset tests.

### Phase 3 — Metrics and losses

1. Implement pixel accuracy.
2. Implement mIoU.
3. Implement Dice score.
4. Implement cross-entropy wrapper.
5. Implement Dice loss.
6. Implement focal loss.
7. Add tests for metrics and losses.

### Phase 4 — Models

1. Implement simple autoencoder segmentation model.
2. Implement U-Net.
3. Implement model factory.
4. Add output-shape tests.

### Phase 5 — Training

1. Implement dataloaders.
2. Implement optimizer creation.
3. Implement scheduler creation.
4. Implement training loop.
5. Implement validation loop.
6. Implement checkpoint saving.
7. Save metrics JSON after each run.

### Phase 6 — Experiments

1. Implement majority-class baseline.
2. Run simple autoencoder experiment.
3. Run U-Net with cross-entropy.
4. Run U-Net with Dice loss.
5. Run U-Net with focal loss.
6. Run U-Net with CE + Dice.
7. Run StepLR experiment.
8. Run CosineAnnealingLR experiment.
9. Run augmentation experiment.
10. Run higher-resolution experiment if resources allow.

### Phase 7 — Reporting

1. Implement experiment logger.
2. Implement model report Excel writer.
3. Add plots.
4. Add best-model sheet.
5. Add prediction examples.

### Phase 8 — Streamlit

1. Implement checkpoint loading.
2. Implement image upload.
3. Implement preprocessing.
4. Implement prediction.
5. Implement mask decoding.
6. Implement overlay.
7. Add class legend.

### Phase 9 — Polish

1. Finalize README.
2. Finalize model report.
3. Finalize final summary.
4. Run all tests.
5. Verify Streamlit app manually.

---

## 25. Definition of Done

The project is complete when all of the following are true:

1. The repository has a clean, understandable structure.
2. Dataset validation works.
3. At least one image-mask pair can be visualized.
4. At least one neural segmentation model trains end-to-end.
5. At least six experiments are present in the model report.
6. The model report includes:
   - baseline row;
   - hyperparameters;
   - no more than three main metrics;
   - percentage change vs baseline;
   - comments;
   - best model statement;
   - train vs validation loss plot;
   - train vs validation mIoU plot;
   - prediction examples.
7. The best checkpoint is saved.
8. The evaluation script calculates test metrics.
9. Streamlit app loads a checkpoint and predicts a segmentation mask.
10. Tests pass with `pytest`.
11. README explains how to reproduce the project.
12. Final summary is ready for presentation.

---

## 26. Minimum Viable Project Scope

If time is limited, complete only this scope:

1. Processed dataset format only.
2. Reduced class mapping only.
3. Image size `128x256`.
4. Majority-class baseline.
5. Simple autoencoder model.
6. U-Net model.
7. Cross-entropy, Dice, and focal loss.
8. StepLR and CosineAnnealingLR experiments.
9. Basic model report with at least six rows.
10. Basic Streamlit upload and prediction app.

---

## 27. Recommended Main Metric

Use **mean Intersection over Union** as the main metric.

Reason:

- It is standard for semantic segmentation.
- It evaluates overlap between predicted and true masks.
- It is more informative than pixel accuracy when classes are imbalanced.

Secondary metrics:

1. pixel accuracy;
2. Dice score.

---

## 28. Important Technical Notes

1. The model output must be logits, not probabilities.
2. Use `CrossEntropyLoss` directly on logits.
3. Use `argmax(dim=1)` for predicted masks.
4. Do not resize masks with bilinear interpolation.
5. Keep masks as integer class IDs.
6. Use `ignore_index` for unlabeled pixels.
7. Data augmentation must transform images and masks consistently.
8. Do not apply augmentation to validation or test sets.
9. Always save the best model based on validation mIoU.
10. Always log experiment configuration and random seed.

---

## 29. Suggested Commit Plan

Use small commits:

```text
init: create semantic segmentation project structure
feat(data): add segmentation dataset and class mapping
feat(data): add dataset validation and exploration scripts
feat(metrics): add pixel accuracy, miou and dice score
feat(losses): add cross entropy, dice and focal loss
feat(models): add simple autoencoder segmentation model
feat(models): add unet model
feat(train): add configurable training pipeline
feat(train): add learning rate schedulers
feat(report): add model report generation
feat(app): add streamlit segmentation demo
test: add dataset, loss, metric and model tests
docs: add full readme and final project summary
```

---

## 30. Codex Agent Starting Prompt

Use the following prompt to start implementation:

```text
You are implementing a course project for semantic segmentation of urban street scene images.

Follow the file `SEMANTIC_SEGMENTATION_IMPLEMENTATION_PLAN.md` exactly.

Start by creating the repository structure, requirements.txt, config files, README draft, and the first implementation files for:
- class mapping;
- segmentation dataset;
- transforms;
- metrics;
- losses;
- simple autoencoder model;
- U-Net model;
- training pipeline;
- model report generation;
- Streamlit app.

Prioritize a working MVP over advanced features. Use PyTorch, torchvision, torchmetrics, pandas, matplotlib, openpyxl, pytest, and Streamlit.

All experiments must be logged into `reports/model_report.xlsx`.
The first row of the report must be a majority-class baseline.
The final app must load the best saved checkpoint and display original image, predicted mask, overlay, and legend.
```
