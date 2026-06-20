# Cityscapes Semantic Segmentation Experiment Expansion Plan for Codex

## 1. Goal

Extend the existing Cityscapes semantic segmentation project with a controlled set of additional experiments. The goal is not to randomly train many models, but to build a clear modelling story for the final model report:

1. Start from the current best-performing direction.
2. Compare architectures under similar conditions.
3. Compare encoders under the same architecture.
4. Compare loss functions relevant to semantic segmentation.
5. Compare learning-rate schedules required by the assignment.
6. Compare preprocessing and augmentation variants.
7. Keep all experiments reproducible through YAML configuration files.

The experiments should be implemented so they can be launched by a Codex/CLI agent without manual code edits.

---

## 2. Assumptions About the Current Project

The repository already contains a semantic segmentation pipeline for Cityscapes using PyTorch and, most likely, `segmentation_models_pytorch` or a similar model factory.

The expected dataset structure is:

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

The model should train on Cityscapes `train`, validate on Cityscapes `val`, and use Cityscapes `test` only for qualitative inference unless a custom split is implemented.

Use Cityscapes `trainIds` labels with:

```yaml
num_classes: 19
ignore_index: 255
```

The project should continue logging experiment results into the existing experiment tracking table or CSV file.

---

## 3. Important Rules for Codex

Before adding new functionality, inspect the repository and reuse existing modules wherever possible.

Do not rewrite the whole training pipeline. Extend the current structure through small changes:

- add new YAML config files;
- extend the model factory only if an architecture is missing;
- extend the loss factory only if a loss is missing;
- extend the scheduler factory only if a scheduler is missing;
- keep existing experiment IDs and outputs unchanged;
- keep all new experiments reproducible;
- make sure every experiment writes metrics, plots, checkpoints, predictions, and a row in the experiment results file.

All new experiments should follow the same base configuration style:

```yaml
training:
  epochs: 30
  batch_size: 4
  num_workers: 8
  mixed_precision: true
  progress_bar: true

scheduler:
  name: cosine_annealing
  t_max: 30
  eta_min: 0.000001

model:
  architecture: deeplabv3plus
  encoder_name: resnet101
  encoder_weights: imagenet
  in_channels: 3
  num_classes: 19

loss:
  name: cross_entropy

preprocessing:
  resize_height: 512
  resize_width: 1024
  crop_height: 512
  crop_width: 512
  augmentations: basic_aug
  eval_augmentations: resize_only
  normalize: imagenet
```

---

## 4. Recommended Experiment Roadmap

### 4.1 Priority Level 1 — Most Important Experiments

These experiments should be implemented first. They are designed around a strong baseline direction: U-Net / ResNet34 / ImageNet weights / CrossEntropy + Dice / Cosine Annealing.

| Experiment ID | Architecture | Encoder | Loss | Scheduler | Purpose |
|---|---|---|---|---|---|
| `040_unet_resnet34_ce_dice_50ep` | U-Net | ResNet34 | CrossEntropy + Dice | Cosine Annealing | Check whether the current best model improves with longer training. |
| `041_unet_resnet34_ce_dice_strong_aug` | U-Net | ResNet34 | CrossEntropy + Dice | Cosine Annealing | Check whether stronger augmentation improves generalization. |
| `042_fpn_resnet34_ce_dice` | FPN | ResNet34 | CrossEntropy + Dice | Cosine Annealing | Compare FPN against U-Net using the same encoder and loss. |
| `043_deeplabv3plus_resnet101_ce_dice` | DeepLabV3+ | ResNet101 | CrossEntropy + Dice | Cosine Annealing | Test a stronger DeepLabV3+ configuration. |
| `044_unetplusplus_resnet34_ce_dice` | U-Net++ | ResNet34 | CrossEntropy + Dice | Cosine Annealing | Compare U-Net++ against standard U-Net. |
| `045_unet_efficientnet_b3_ce_dice` | U-Net | EfficientNet-B3 | CrossEntropy + Dice | Cosine Annealing | Test a stronger encoder under the best architecture. |
| `046_fpn_efficientnet_b3_ce_dice` | FPN | EfficientNet-B3 | CrossEntropy + Dice | Cosine Annealing | Test FPN with a stronger encoder. |
| `047_deeplabv3plus_efficientnet_b3_ce_dice` | DeepLabV3+ | EfficientNet-B3 | CrossEntropy + Dice | Cosine Annealing | Combine multi-scale context with a stronger encoder. |
| `048_unet_resnet34_ce_lovasz` | U-Net | ResNet34 | CrossEntropy + Lovasz | Cosine Annealing | Test a loss function closer to IoU optimization. |
| `049_segformer_mit_b1_ce_dice_optional` | SegFormer | MiT-B1 | CrossEntropy + Dice | Cosine Annealing | Optional modern transformer-based comparison. Do not block the project if unsupported. |

---

## 5. YAML Config Files to Add

Create one YAML file per experiment under the existing config directory. If the repository already has a convention, follow it. Otherwise use:

```text
configs/experiments/
```

### 5.1 `040_unet_resnet34_ce_dice_50ep.yaml`

```yaml
experiment:
  id: 040_unet_resnet34_ce_dice_50ep
  description: Longer training run for the current best U-Net ResNet34 CE+Dice direction.

training:
  epochs: 50
  batch_size: 4
  num_workers: 8
  mixed_precision: true
  progress_bar: true

scheduler:
  name: cosine_annealing
  t_max: 50
  eta_min: 0.000001

model:
  architecture: unet
  encoder_name: resnet34
  encoder_weights: imagenet
  in_channels: 3
  num_classes: 19

loss:
  name: cross_entropy_dice
  ignore_index: 255

preprocessing:
  resize_height: 512
  resize_width: 1024
  crop_height: 512
  crop_width: 512
  augmentations: basic_aug
  eval_augmentations: resize_only
  normalize: imagenet
```

### 5.2 `041_unet_resnet34_ce_dice_strong_aug.yaml`

```yaml
experiment:
  id: 041_unet_resnet34_ce_dice_strong_aug
  description: U-Net ResNet34 CE+Dice with stronger augmentation.

training:
  epochs: 30
  batch_size: 4
  num_workers: 8
  mixed_precision: true
  progress_bar: true

scheduler:
  name: cosine_annealing
  t_max: 30
  eta_min: 0.000001

model:
  architecture: unet
  encoder_name: resnet34
  encoder_weights: imagenet
  in_channels: 3
  num_classes: 19

loss:
  name: cross_entropy_dice
  ignore_index: 255

preprocessing:
  resize_height: 512
  resize_width: 1024
  crop_height: 512
  crop_width: 512
  augmentations: strong_aug
  eval_augmentations: resize_only
  normalize: imagenet
```

### 5.3 `042_fpn_resnet34_ce_dice.yaml`

```yaml
experiment:
  id: 042_fpn_resnet34_ce_dice
  description: FPN ResNet34 with CrossEntropy + Dice loss.

training:
  epochs: 30
  batch_size: 4
  num_workers: 8
  mixed_precision: true
  progress_bar: true

scheduler:
  name: cosine_annealing
  t_max: 30
  eta_min: 0.000001

model:
  architecture: fpn
  encoder_name: resnet34
  encoder_weights: imagenet
  in_channels: 3
  num_classes: 19

loss:
  name: cross_entropy_dice
  ignore_index: 255

preprocessing:
  resize_height: 512
  resize_width: 1024
  crop_height: 512
  crop_width: 512
  augmentations: basic_aug
  eval_augmentations: resize_only
  normalize: imagenet
```

### 5.4 `043_deeplabv3plus_resnet101_ce_dice.yaml`

```yaml
experiment:
  id: 043_deeplabv3plus_resnet101_ce_dice
  description: Strong DeepLabV3+ configuration with ResNet101 and CE+Dice.

training:
  epochs: 30
  batch_size: 4
  num_workers: 8
  mixed_precision: true
  progress_bar: true

scheduler:
  name: cosine_annealing
  t_max: 30
  eta_min: 0.000001

model:
  architecture: deeplabv3plus
  encoder_name: resnet101
  encoder_weights: imagenet
  in_channels: 3
  num_classes: 19

loss:
  name: cross_entropy_dice
  ignore_index: 255

preprocessing:
  resize_height: 512
  resize_width: 1024
  crop_height: 512
  crop_width: 512
  augmentations: basic_aug
  eval_augmentations: resize_only
  normalize: imagenet
```

### 5.5 `044_unetplusplus_resnet34_ce_dice.yaml`

```yaml
experiment:
  id: 044_unetplusplus_resnet34_ce_dice
  description: U-Net++ ResNet34 with CrossEntropy + Dice loss.

training:
  epochs: 30
  batch_size: 4
  num_workers: 8
  mixed_precision: true
  progress_bar: true

scheduler:
  name: cosine_annealing
  t_max: 30
  eta_min: 0.000001

model:
  architecture: unetplusplus
  encoder_name: resnet34
  encoder_weights: imagenet
  in_channels: 3
  num_classes: 19

loss:
  name: cross_entropy_dice
  ignore_index: 255

preprocessing:
  resize_height: 512
  resize_width: 1024
  crop_height: 512
  crop_width: 512
  augmentations: basic_aug
  eval_augmentations: resize_only
  normalize: imagenet
```

### 5.6 `045_unet_efficientnet_b3_ce_dice.yaml`

```yaml
experiment:
  id: 045_unet_efficientnet_b3_ce_dice
  description: U-Net with EfficientNet-B3 encoder and CE+Dice loss.

training:
  epochs: 30
  batch_size: 4
  num_workers: 8
  mixed_precision: true
  progress_bar: true

scheduler:
  name: cosine_annealing
  t_max: 30
  eta_min: 0.000001

model:
  architecture: unet
  encoder_name: efficientnet-b3
  encoder_weights: imagenet
  in_channels: 3
  num_classes: 19

loss:
  name: cross_entropy_dice
  ignore_index: 255

preprocessing:
  resize_height: 512
  resize_width: 1024
  crop_height: 512
  crop_width: 512
  augmentations: basic_aug
  eval_augmentations: resize_only
  normalize: imagenet
```

### 5.7 `046_fpn_efficientnet_b3_ce_dice.yaml`

```yaml
experiment:
  id: 046_fpn_efficientnet_b3_ce_dice
  description: FPN with EfficientNet-B3 encoder and CE+Dice loss.

training:
  epochs: 30
  batch_size: 4
  num_workers: 8
  mixed_precision: true
  progress_bar: true

scheduler:
  name: cosine_annealing
  t_max: 30
  eta_min: 0.000001

model:
  architecture: fpn
  encoder_name: efficientnet-b3
  encoder_weights: imagenet
  in_channels: 3
  num_classes: 19

loss:
  name: cross_entropy_dice
  ignore_index: 255

preprocessing:
  resize_height: 512
  resize_width: 1024
  crop_height: 512
  crop_width: 512
  augmentations: basic_aug
  eval_augmentations: resize_only
  normalize: imagenet
```

### 5.8 `047_deeplabv3plus_efficientnet_b3_ce_dice.yaml`

```yaml
experiment:
  id: 047_deeplabv3plus_efficientnet_b3_ce_dice
  description: DeepLabV3+ with EfficientNet-B3 encoder and CE+Dice loss.

training:
  epochs: 30
  batch_size: 4
  num_workers: 8
  mixed_precision: true
  progress_bar: true

scheduler:
  name: cosine_annealing
  t_max: 30
  eta_min: 0.000001

model:
  architecture: deeplabv3plus
  encoder_name: efficientnet-b3
  encoder_weights: imagenet
  in_channels: 3
  num_classes: 19

loss:
  name: cross_entropy_dice
  ignore_index: 255

preprocessing:
  resize_height: 512
  resize_width: 1024
  crop_height: 512
  crop_width: 512
  augmentations: basic_aug
  eval_augmentations: resize_only
  normalize: imagenet
```

### 5.9 `048_unet_resnet34_ce_lovasz.yaml`

```yaml
experiment:
  id: 048_unet_resnet34_ce_lovasz
  description: U-Net ResNet34 with CrossEntropy + Lovasz loss.

training:
  epochs: 30
  batch_size: 4
  num_workers: 8
  mixed_precision: true
  progress_bar: true

scheduler:
  name: cosine_annealing
  t_max: 30
  eta_min: 0.000001

model:
  architecture: unet
  encoder_name: resnet34
  encoder_weights: imagenet
  in_channels: 3
  num_classes: 19

loss:
  name: cross_entropy_lovasz
  ignore_index: 255

preprocessing:
  resize_height: 512
  resize_width: 1024
  crop_height: 512
  crop_width: 512
  augmentations: basic_aug
  eval_augmentations: resize_only
  normalize: imagenet
```

### 5.10 `049_segformer_mit_b1_ce_dice_optional.yaml`

This experiment is optional. SegFormer may not be supported by the current `segmentation_models_pytorch` setup. Codex should first inspect the repository and dependencies. If SegFormer is not already supported, implement it only if it can be added cleanly without breaking the core pipeline.

```yaml
experiment:
  id: 049_segformer_mit_b1_ce_dice_optional
  description: Optional transformer-based SegFormer experiment with MiT-B1 backbone.

training:
  epochs: 30
  batch_size: 2
  num_workers: 8
  mixed_precision: true
  progress_bar: true

scheduler:
  name: cosine_annealing
  t_max: 30
  eta_min: 0.000001

model:
  architecture: segformer
  encoder_name: mit_b1
  encoder_weights: imagenet
  in_channels: 3
  num_classes: 19

loss:
  name: cross_entropy_dice
  ignore_index: 255

preprocessing:
  resize_height: 512
  resize_width: 1024
  crop_height: 512
  crop_width: 512
  augmentations: basic_aug
  eval_augmentations: resize_only
  normalize: imagenet
```

---

## 6. Additional Architecture Experiments

After the priority experiments are implemented, add optional experiments for lightweight or attention-based architectures if they are supported by the current model factory.

### 6.1 MAnet

```yaml
model:
  architecture: manet
  encoder_name: resnet34
  encoder_weights: imagenet
  in_channels: 3
  num_classes: 19
```

Suggested experiment ID:

```text
050_manet_resnet34_ce_dice
```

Purpose: test an attention/multi-scale encoder-decoder architecture.

### 6.2 PAN

```yaml
model:
  architecture: pan
  encoder_name: resnet34
  encoder_weights: imagenet
  in_channels: 3
  num_classes: 19
```

Suggested experiment ID:

```text
051_pan_resnet34_ce_dice
```

Purpose: test a pyramid-attention style model after FPN.

### 6.3 LinkNet

```yaml
model:
  architecture: linknet
  encoder_name: resnet34
  encoder_weights: imagenet
  in_channels: 3
  num_classes: 19
```

Suggested experiment ID:

```text
052_linknet_resnet34_ce_dice
```

Purpose: evaluate a lighter model and compare speed/VRAM against segmentation quality.

---

## 7. Loss Function Work

Codex should inspect the existing loss factory. If some losses are already implemented, reuse them. If not, add them in a minimal, tested way.

Required or useful losses:

| Loss name in YAML | Purpose |
|---|---|
| `cross_entropy` | Main baseline loss for multi-class segmentation. |
| `dice` | Region-overlap loss, useful for segmentation. |
| `cross_entropy_dice` | Strong practical default. Combines pixel classification and region overlap. |
| `focal` | Helps focus training on hard pixels/classes. |
| `focal_dice` | Useful if focal alone is unstable or insufficient. |
| `cross_entropy_lovasz` | Optional IoU-oriented experiment. |
| `weighted_cross_entropy` | Optional class-imbalance experiment. |
| `tversky` | Optional imbalance-oriented segmentation loss. |
| `focal_tversky` | Optional extension of Tversky for hard examples. |

### 7.1 Expected loss factory behavior

The loss factory should support code like:

```python
criterion = build_loss(config.loss)
```

It should correctly handle:

```yaml
loss:
  name: cross_entropy_dice
  ignore_index: 255
```

If a loss internally combines two losses, expose weights only if needed. A simple default is acceptable:

```yaml
loss:
  name: cross_entropy_dice
  ce_weight: 1.0
  dice_weight: 1.0
  ignore_index: 255
```

---

## 8. Scheduler Work

The assignment requires demonstrating learning-rate changes over epochs, specifically cosine annealing and step decay. The project should already support those or should be extended to support them.

Recommended schedulers:

| Scheduler name in YAML | Priority | Purpose |
|---|---:|---|
| `cosine_annealing` | Required | Main scheduler for most final experiments. |
| `step_decay` | Required | Required by assignment for comparison. |
| `one_cycle` | Optional | Useful practical comparison. |
| `polynomial_decay` | Optional | Common in semantic segmentation pipelines. |

### 8.1 Example OneCycle config

```yaml
scheduler:
  name: one_cycle
  max_lr: 0.001
  pct_start: 0.1
```

### 8.2 Example Polynomial Decay config

```yaml
scheduler:
  name: polynomial_decay
  power: 0.9
  eta_min: 0.000001
```

---

## 9. Preprocessing and Augmentation Experiments

Add or verify these preprocessing presets:

| Preset | Purpose |
|---|---|
| `resize_only` | Deterministic evaluation preprocessing. |
| `basic_aug` | Basic training augmentation. |
| `strong_aug` | Stronger generalization test. |
| `basic_aug_no_hflip` | Control experiment without horizontal flip. |

### 9.1 Wide crop experiment

Cityscapes images are wide street-scene images. A square crop can lose horizontal context. Add at least one experiment with a wider crop:

```yaml
preprocessing:
  resize_height: 512
  resize_width: 1024
  crop_height: 512
  crop_width: 768
  augmentations: basic_aug
  eval_augmentations: resize_only
  normalize: imagenet
```

Suggested experiment ID:

```text
053_unet_resnet34_ce_dice_wide_crop
```

### 9.2 Larger crop experiment

Only run this if GPU memory allows it:

```yaml
preprocessing:
  resize_height: 768
  resize_width: 1536
  crop_height: 768
  crop_width: 768
  augmentations: basic_aug
  eval_augmentations: resize_only
  normalize: imagenet
```

Suggested experiment ID:

```text
054_unet_resnet34_ce_dice_large_crop
```

If CUDA out-of-memory occurs, reduce:

```yaml
training:
  batch_size: 2
```

---

## 10. Metrics to Track

Do not add too many metrics to the final model report. Use at most three primary metrics:

1. `mIoU` — primary metric.
2. `Pixel Accuracy` — easy to interpret, but can be misleading with class imbalance.
3. `Mean Dice` or `Macro F1` — useful segmentation overlap metric.

For each experiment, save:

```text
reports/metrics/<experiment_id>.json
reports/figures/<experiment_id>/train_val_loss.png
reports/figures/<experiment_id>/train_val_miou.png
outputs/predictions/<experiment_id>/
checkpoints/<experiment_id>/best.ckpt
```

The experiment results table should include:

```text
Experiment ID
Architecture
Encoder
Encoder Weights
Loss
Scheduler
Epochs
Batch Size
Input Resize
Crop Size
Augmentation
mIoU
Pixel Accuracy
Mean Dice / Macro F1
Comments
```

Keep the table readable. Do not make it too wide.

---

## 11. Model Report Requirements

Codex should ensure the experiment pipeline can export or update a model report file.

The report must tell a story, not just dump metrics.

Required report behavior:

1. One row per experiment.
2. Rows stay in chronological order.
3. Include a baseline row.
4. Include no more than three core metrics.
5. Include a `Comments` column.
6. Highlight or clearly mark the best model.
7. Include train-vs-validation loss plots.
8. Include train-vs-validation main metric plots.
9. Add 4-5 qualitative examples for the best model.
10. Include correct and incorrect predictions if possible.

---

## 12. Suggested Commands

Follow the existing project command style. If no convention exists, support commands like:

```bash
python scripts/train.py --config configs/experiments/040_unet_resnet34_ce_dice_50ep.yaml
python scripts/evaluate.py --config configs/experiments/040_unet_resnet34_ce_dice_50ep.yaml --checkpoint checkpoints/040_unet_resnet34_ce_dice_50ep/best.ckpt
python scripts/predict.py --config configs/experiments/040_unet_resnet34_ce_dice_50ep.yaml --checkpoint checkpoints/040_unet_resnet34_ce_dice_50ep/best.ckpt
```

Add a batch runner script:

```bash
bash scripts/run_experiment_batch.sh configs/experiments/040_unet_resnet34_ce_dice_50ep.yaml configs/experiments/042_fpn_resnet34_ce_dice.yaml
```

If the repo already has a batch script, extend it instead of creating a duplicate.

---

## 13. Implementation Checklist for Codex

### Step 1 — Inspect the repository

- Locate the config parser.
- Locate the model factory.
- Locate the loss factory.
- Locate the scheduler factory.
- Locate the dataset class.
- Locate the training entrypoint.
- Locate the metrics code.
- Locate the experiment result logging code.

### Step 2 — Add config files

Create:

```text
configs/experiments/040_unet_resnet34_ce_dice_50ep.yaml
configs/experiments/041_unet_resnet34_ce_dice_strong_aug.yaml
configs/experiments/042_fpn_resnet34_ce_dice.yaml
configs/experiments/043_deeplabv3plus_resnet101_ce_dice.yaml
configs/experiments/044_unetplusplus_resnet34_ce_dice.yaml
configs/experiments/045_unet_efficientnet_b3_ce_dice.yaml
configs/experiments/046_fpn_efficientnet_b3_ce_dice.yaml
configs/experiments/047_deeplabv3plus_efficientnet_b3_ce_dice.yaml
configs/experiments/048_unet_resnet34_ce_lovasz.yaml
configs/experiments/049_segformer_mit_b1_ce_dice_optional.yaml
```

### Step 3 — Extend the model factory if needed

Required architectures:

```text
unet
unetplusplus
fpn
deeplabv3plus
```

Optional architectures:

```text
manet
pan
linknet
segformer
```

If `segformer` is not supported by the current dependency stack, leave it disabled and document it as optional.

### Step 4 — Extend the loss factory if needed

Required:

```text
cross_entropy
cross_entropy_dice
focal
```

Recommended:

```text
cross_entropy_lovasz
weighted_cross_entropy
focal_dice
tversky
focal_tversky
```

### Step 5 — Extend scheduler support if needed

Required:

```text
cosine_annealing
step_decay
```

Optional:

```text
one_cycle
polynomial_decay
```

### Step 6 — Add augmentation presets if needed

Required:

```text
resize_only
basic_aug
strong_aug
basic_aug_no_hflip
```

### Step 7 — Add tests

Add or update tests for:

- config loading;
- model creation for each supported architecture;
- loss creation for each supported loss;
- scheduler creation;
- Cityscapes dataset path resolution;
- one tiny training step on a small fake batch;
- metrics computation with `ignore_index=255`.

### Step 8 — Update documentation

Update the project README with:

- how to download Cityscapes;
- expected dataset structure;
- how to generate `labelTrainIds` masks;
- how to run one experiment;
- how to run a batch of experiments;
- how to open the Streamlit app;
- where the model report is generated.

---

## 14. Acceptance Criteria

The task is complete when:

1. All priority YAML config files exist.
2. The training script can load each config without errors.
3. The model factory supports all non-optional architectures in the configs.
4. The loss factory supports all non-optional losses in the configs.
5. The scheduler factory supports cosine annealing and step decay.
6. The dataset loader works with `data/raw/cityscapes`.
7. Training one experiment produces:
   - checkpoint;
   - metrics file;
   - loss curve;
   - mIoU curve;
   - prediction examples;
   - experiment result row.
8. Tests pass.
9. README is updated.
10. Optional SegFormer support is either implemented cleanly or documented as skipped.

---

## 15. Recommended Execution Order

Run experiments in this order:

```text
040_unet_resnet34_ce_dice_50ep
041_unet_resnet34_ce_dice_strong_aug
042_fpn_resnet34_ce_dice
043_deeplabv3plus_resnet101_ce_dice
044_unetplusplus_resnet34_ce_dice
045_unet_efficientnet_b3_ce_dice
046_fpn_efficientnet_b3_ce_dice
047_deeplabv3plus_efficientnet_b3_ce_dice
048_unet_resnet34_ce_lovasz
049_segformer_mit_b1_ce_dice_optional
```

If GPU memory is limited, run these first:

```text
040_unet_resnet34_ce_dice_50ep
042_fpn_resnet34_ce_dice
044_unetplusplus_resnet34_ce_dice
048_unet_resnet34_ce_lovasz
```

If time is limited, the minimum useful subset is:

```text
040_unet_resnet34_ce_dice_50ep
042_fpn_resnet34_ce_dice
043_deeplabv3plus_resnet101_ce_dice
048_unet_resnet34_ce_lovasz
```

---

## 16. Final Note for Codex

Do not optimize only for the highest score. Optimize for a clear, defensible model report. Each experiment should answer a specific question:

- Did longer training help?
- Did stronger augmentation help?
- Did another architecture help?
- Did a stronger encoder help?
- Did an IoU-oriented loss help?
- Did a different crop size help?
- Did the scheduler change improve convergence?

Every experiment must have a short comment explaining what changed and whether it improved the model.
