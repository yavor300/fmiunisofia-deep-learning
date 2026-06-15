# Semantic Segmentation on Cityscapes — Implementation Plan for Codex Agent

> This document is intended to be given directly to a Codex agent. It defines the full implementation roadmap for a course project on **semantic image segmentation** using the **Cityscapes** dataset. The final system must include dataset exploration, multiple model experiments, loss-function comparisons, learning-rate scheduling experiments, input preprocessing experiments, a Streamlit UI, tests, and a model report.

---

## 1. Project Goal

Build an end-to-end semantic segmentation project for urban street-scene images. The user should be able to upload or select an image of a road/street scene and obtain a pixel-level semantic segmentation mask on a car driving on a street.

The project must satisfy these requirements:

1. Study and document scientific papers and semantic segmentation techniques.
2. Compare at least these loss functions:
   - pixel-wise cross-entropy loss;
   - Dice loss;
   - focal loss.
3. Explore the Cityscapes data:
   - number of observations;
   - number and type of features;
   - class distribution and statistical relationships;
   - anomalies and invalid samples;
   - visual analysis with comments.
4. Run experiments with an autoencoder-type neural network.
5. Demonstrate learning-rate scheduling:
   - cosine annealing;
   - step decay.
6. Run experiments with different input preprocessing strategies.
7. Build a Streamlit UI.
8. Add tests.
9. Produce a model report file and a presentation-ready project summary.

---

## 2. Existing Reference Material

Use the existing repository/documentation as a starting point, but refactor it into a more structured and reproducible project.

Existing example implementation:

- `semantic-segmentation-cityscapes.md` describes a PyTorch + Segmentation Models PyTorch pipeline for Cityscapes.
- It already references U-Net and DeepLabV3+.
- It includes a training workflow based on `run.sh` and `launch.py`.
- It mentions model visualization through prediction overlays and time-lapse videos.

Lecture notes to incorporate:

- Semantic segmentation assigns one class/category to every pixel.
- The model input is an image and the target output is a segmentation mask.
- Masks are 2D matrices where each value is a class ID.
- FCN-style models use fully convolutional prediction and upsampling.
- Skip connections combine coarse semantic information with finer spatial information.
- U-Net uses an encoder-decoder structure with concatenation skip connections.
- Transposed convolution, bilinear interpolation, and other upsampling techniques should be explained in the report.

Model report notes to incorporate:

- Each row in the model report is one trained/evaluated hypothesis.
- The first row must be the baseline model.
- Keep at most three test metrics.
- Include percentage change compared with the baseline.
- Keep a rightmost `Comments` column.
- Explicitly state the best model and why.
- Include train-vs-validation metric and loss diagrams.
- Do not sort experiments after completion; preserve chronological experiment order.

---

## 3. Recommended Scientific Papers and Techniques

Create `docs/literature_review.md` and summarize the following papers. Keep each summary short, practical, and connected to implementation choices.

### 3.1 Core papers

1. **Cityscapes Dataset**  
   M. Cordts et al., *The Cityscapes Dataset for Semantic Urban Scene Understanding*, CVPR 2016.  
   URL: https://arxiv.org/abs/1604.01685  
   Use this to describe the dataset, train/validation/test split, urban classes, and benchmark metrics.

2. **Fully Convolutional Networks (FCN)**  
   J. Long, E. Shelhamer, T. Darrell, *Fully Convolutional Networks for Semantic Segmentation*, CVPR 2015.  
   URL: https://arxiv.org/abs/1411.4038  
   Use this as the historical baseline for dense pixel prediction and skip connections.

3. **U-Net**  
   O. Ronneberger, P. Fischer, T. Brox, *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015.  
   URL: https://arxiv.org/abs/1505.04597  
   Use this for the autoencoder/encoder-decoder architecture requirement.

4. **Learning Deconvolution Network**  
   H. Noh, S. Hong, B. Han, *Learning Deconvolution Network for Semantic Segmentation*, ICCV 2015.  
   URL: https://arxiv.org/abs/1505.04366  
   Use this for unpooling/deconvolution decoder discussion.

5. **DeepLabV3+**  
   L.-C. Chen et al., *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation*, ECCV 2018.  
   URL: https://arxiv.org/abs/1802.02611  
   Use this for atrous convolution, ASPP, and boundary refinement.

### 3.2 Additional architectures worth trying

6. **PSPNet**  
   H. Zhao et al., *Pyramid Scene Parsing Network*, CVPR 2017.  
   URL: https://arxiv.org/abs/1612.01105  
   Try this if the library supports it. It is useful for global context aggregation.

7. **ENet**  
   A. Paszke et al., *ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation*, 2016.  
   URL: https://arxiv.org/abs/1606.02147  
   Include as a real-time lightweight segmentation architecture candidate.

8. **BiSeNet**  
   C. Yu et al., *BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation*, ECCV 2018.  
   URL: https://arxiv.org/abs/1808.00897  
   Consider only if time allows; useful for comparing accuracy/speed trade-offs.

9. **SegFormer**  
   E. Xie et al., *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*, NeurIPS 2021.  
   URL: https://arxiv.org/abs/2105.15203  
   Optional advanced experiment using a transformer encoder. Use Hugging Face or an existing PyTorch implementation if feasible.

### 3.3 Loss-function references

10. **Focal Loss**  
    T.-Y. Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017.  
    URL: https://arxiv.org/abs/1708.02002  
    Use the focal loss idea to handle class imbalance.

11. **Dice / V-Net**  
    F. Milletari, N. Navab, S.-A. Ahmadi, *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation*, 2016.  
    URL: https://arxiv.org/abs/1606.04797  
    Use this as a source for Dice-style overlap optimization.

12. **Generalized Dice Loss**  
    C. Sudre et al., *Generalised Dice Overlap as a Deep Learning Loss Function for Highly Unbalanced Segmentations*, 2017.  
    URL: https://arxiv.org/abs/1707.03237  
    Use this for discussion of class imbalance and region-based losses.

---

## 4. Proposed Architectures for Experiments

Implement the project so that model choice is controlled by a YAML config. Start with stable models from `segmentation_models_pytorch` where possible.

### Required / high priority

1. **Greedy statistical baseline**
   - Predict the most frequent class for every pixel.
   - This is required for the first row of the model report.
   - It is not expected to perform well, but it provides a meaningful baseline.

2. **Small custom U-Net / autoencoder-style network**
   - Implement a simple encoder-decoder model manually in PyTorch.
   - This directly satisfies the autoencoder-type neural network requirement.
   - Use it before trying larger pretrained backbones.

3. **U-Net with pretrained encoder**
   - Example: `Unet + resnet34` or `Unet + efficientnet-b0`.
   - Use this as the first strong model.

4. **DeepLabV3+**
   - Example: `DeepLabV3Plus + resnet50` or `DeepLabV3Plus + efficientnet-b4`.
   - Use this as the primary advanced convolutional architecture.

### Medium priority

5. **FPN**
   - Good for multi-scale features.
   - Usually available in SMP.

6. **PSPNet**
   - Good for scene-level context.
   - Try if supported by the chosen library/version.

7. **U-Net++**
   - Useful to compare deeper skip-connection structures.

8. **LinkNet / PAN / MAnet**
   - Try only if training budget allows.

### Optional advanced

9. **SegFormer-B0 or SegFormer-B1**
   - Use only after the PyTorch/SMP pipeline is stable.
   - Keep this as a bonus experiment because it may require a different library path.

---

## 5. Target Repository Structure

Refactor or create the project with this structure:

```text
semantic-segmentation-cityscapes/
├── app/
│   └── streamlit_app.py
├── configs/
│   ├── default.yaml
│   ├── experiments/
│   │   ├── 000_baseline_majority.yaml
│   │   ├── 001_tiny_unet_ce_step.yaml
│   │   ├── 002_tiny_unet_dice_step.yaml
│   │   ├── 003_unet_resnet34_ce_cosine.yaml
│   │   ├── 004_unet_resnet34_dice_cosine.yaml
│   │   ├── 005_unet_resnet34_focal_cosine.yaml
│   │   ├── 006_deeplabv3plus_resnet50_ce_step.yaml
│   │   ├── 007_deeplabv3plus_resnet50_dice_cosine.yaml
│   │   ├── 008_deeplabv3plus_efficientnetb4_focal_cosine.yaml
│   │   ├── 009_fpn_resnet34_ce_cosine.yaml
│   │   └── 010_segformer_b0_ce_cosine.yaml
├── data/
│   ├── raw/                  # not committed
│   ├── processed/            # not committed
│   └── README.md
├── docs/
│   ├── literature_review.md
│   ├── dataset_analysis.md
│   ├── architecture_notes.md
│   └── project_summary.md
├── notebooks/
│   └── optional_exploration.ipynb
├── reports/
│   ├── figures/
│   ├── model_report.xlsx
│   └── experiment_results.csv
├── scripts/
│   ├── run_train.sh
│   ├── run_eval.sh
│   ├── run_eda.sh
│   └── download_cityscapes_placeholder.md
├── src/
│   └── cityseg/
│       ├── __init__.py
│       ├── constants.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── cityscapes_dataset.py
│       │   ├── label_mapping.py
│       │   └── transforms.py
│       ├── eda/
│       │   ├── __init__.py
│       │   └── analyze_dataset.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── factory.py
│       │   ├── tiny_unet.py
│       │   └── majority_baseline.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── train.py
│       │   ├── evaluate.py
│       │   ├── losses.py
│       │   ├── metrics.py
│       │   ├── schedulers.py
│       │   ├── checkpointing.py
│       │   └── experiment_logger.py
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── predict.py
│       │   └── visualization.py
│       └── reporting/
│           ├── __init__.py
│           ├── build_model_report.py
│           └── plots.py
├── tests/
│   ├── test_label_mapping.py
│   ├── test_dataset.py
│   ├── test_losses.py
│   ├── test_metrics.py
│   ├── test_model_factory.py
│   ├── test_training_step.py
│   └── test_streamlit_smoke.py
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── README.md
└── Makefile
```

---

## 6. Implementation Phases and Sequential Tasks

### Phase 0 — Project setup

**Goal:** Create a reproducible Python project that can run locally and on GPU machines such as RunPod.

Tasks:

1. Create `pyproject.toml` or `requirements.txt` with these dependencies:
   - `torch`
   - `torchvision`
   - `segmentation-models-pytorch`
   - `timm`
   - `albumentations`
   - `opencv-python`
   - `numpy`
   - `pandas`
   - `Pillow`
   - `matplotlib`
   - `scikit-learn`
   - `torchmetrics`
   - `PyYAML`
   - `tqdm`
   - `openpyxl`
   - `streamlit`
   - `pytest`
   - `ruff` or `flake8`
2. Add `.gitignore` entries for:
   - raw dataset files;
   - checkpoints;
   - generated reports;
   - virtual environments;
   - cache folders.
3. Add `Makefile` commands:

```makefile
install:
	pip install -r requirements.txt

eda:
	python -m src.cityseg.eda.analyze_dataset --config configs/default.yaml

train:
	python -m src.cityseg.training.train --config $(CONFIG)

eval:
	python -m src.cityseg.training.evaluate --config $(CONFIG) --checkpoint $(CHECKPOINT)

report:
	python -m src.cityseg.reporting.build_model_report --results reports/experiment_results.csv

app:
	streamlit run app/streamlit_app.py

test:
	pytest -q
```

4. Add a clear `README.md` with:
   - project goal;
   - dataset instructions;
   - installation;
   - training commands;
   - evaluation commands;
   - Streamlit command;
   - testing command.

Acceptance criteria:

- `make install` works.
- `make test` runs even before the dataset is downloaded, using synthetic test data.
- The project structure exists.

---

### Phase 1 — Dataset integration

**Goal:** Implement robust Cityscapes loading and label conversion.

Tasks:

1. In `data/README.md`, explain that Cityscapes must be downloaded manually from the official website because of licensing.
2. Expected raw structure:

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

3. Implement `src/cityseg/constants.py`:
   - Cityscapes class names;
   - RGB color palette;
   - mapping from original labels to `trainId`;
   - `IGNORE_INDEX = 255`.
4. Implement `src/cityseg/data/label_mapping.py`:
   - `convert_label_ids_to_train_ids(mask)`;
   - `decode_train_ids_to_colors(mask)`;
   - `get_class_names()`;
   - `get_palette()`.
5. Implement `src/cityseg/data/cityscapes_dataset.py`:
   - accepts `root`, `split`, `image_size`, `crop_size`, `transforms`;
   - returns `image_tensor` with shape `[3, H, W]`;
   - returns `mask_tensor` with shape `[H, W]` and dtype `torch.long`;
   - never normalizes masks with `ToTensor`; masks must remain class IDs;
   - uses `IGNORE_INDEX` for ignored classes.
6. Implement train/validation dataloaders with configurable:
   - batch size;
   - number of workers;
   - image size/crop size;
   - pin memory;
   - shuffle.

Acceptance criteria:

- Dataset can load at least one image/mask pair.
- The image tensor and mask tensor have expected shapes and dtypes.
- Unique values in masks are valid train IDs or `255`.
- Unit tests pass with a tiny synthetic Cityscapes-like folder.

---

### Phase 2 — Exploratory Data Analysis

**Goal:** Produce dataset analysis required by the assignment.

Create `src/cityseg/eda/analyze_dataset.py`.

The script must produce:

1. Number of images per split.
2. Image size distribution.
3. Number of classes.
4. Pixel count per class.
5. Percentage of pixels per class.
6. Class imbalance visualization.
7. Example image/mask overlays.
8. Rare class examples.
9. Anomaly checks:
   - missing images;
   - missing masks;
   - unreadable files;
   - invalid mask values;
   - mismatched image/mask dimensions.
10. `docs/dataset_analysis.md` with written interpretation.
11. Figures in `reports/figures/`:
   - `class_distribution.png`;
   - `image_size_distribution.png`;
   - `sample_overlays.png`;
   - `rare_classes_examples.png` if possible.

Acceptance criteria:

- `make eda` generates figures and `docs/dataset_analysis.md`.
- The EDA report contains comments, not only charts.
- The report explicitly discusses class imbalance and its connection to focal/Dice loss.

---

### Phase 3 — Configuration system

**Goal:** Make all experiments reproducible through YAML configs.

Create `configs/default.yaml`:

```yaml
seed: 42
project_name: cityscapes-semantic-segmentation

paths:
  data_root: data/raw/cityscapes
  output_dir: outputs
  reports_dir: reports

training:
  epochs: 30
  batch_size: 4
  num_workers: 4
  mixed_precision: true
  device: cuda
  gradient_clip_norm: 1.0

optimizer:
  name: adamw
  lr: 0.0003
  weight_decay: 0.0001

scheduler:
  name: cosine_annealing
  step_size: 10
  gamma: 0.1
  t_max: 30
  eta_min: 0.000001

model:
  architecture: unet
  encoder_name: resnet34
  encoder_weights: imagenet
  in_channels: 3
  num_classes: 19

loss:
  name: cross_entropy
  ignore_index: 255
  class_weights: null

preprocessing:
  resize_height: 512
  resize_width: 1024
  crop_height: 512
  crop_width: 512
  normalize: imagenet
  augmentations: basic

metrics:
  main_metric: mean_iou
  include:
    - mean_iou
    - mean_dice
    - pixel_accuracy
```

Tasks:

1. Implement config loading.
2. Allow experiment configs to override defaults.
3. Save the exact config used for each run in the experiment output folder.
4. Add `seed_everything(seed)`.

Acceptance criteria:

- Any config can be passed via `--config`.
- Every output folder includes the resolved config.

---

### Phase 4 — Baseline model

**Goal:** Create the mandatory first model-report row.

Implement `src/cityseg/models/majority_baseline.py`.

Tasks:

1. Use training masks to compute the most frequent non-ignored class.
2. Predict that class for every pixel.
3. Evaluate on validation/test split.
4. Store results in `reports/experiment_results.csv`.

Metrics:

- Mean IoU.
- Mean Dice.
- Pixel accuracy.

Acceptance criteria:

- Baseline runs without GPU.
- It becomes experiment `000_baseline_majority`.
- It is the first row in `model_report.xlsx`.

---

### Phase 5 — Loss functions

**Goal:** Implement and compare pixel-wise cross-entropy, Dice, and focal loss.

Create `src/cityseg/training/losses.py`.

Implement:

1. `CrossEntropyLoss(ignore_index=255)`
2. `DiceLoss(mode="multiclass", ignore_index=255)`
3. `FocalLoss(gamma=2.0, alpha=None, ignore_index=255)`
4. Optional combined losses:
   - `CrossEntropyDiceLoss`
   - `FocalDiceLoss`

Important implementation notes:

- Model output shape: `[B, C, H, W]`.
- Target shape: `[B, H, W]`.
- Use `ignore_index=255` consistently.
- Dice loss should operate on probabilities after `softmax`.
- Exclude ignored pixels from the Dice computation.

Acceptance criteria:

- Unit tests verify all losses return finite scalar tensors.
- Losses support backward pass.
- Losses handle ignored pixels.

---

### Phase 6 — Metrics

**Goal:** Implement stable segmentation metrics.

Create `src/cityseg/training/metrics.py`.

Implement:

1. Mean Intersection over Union (`mean_iou`).
2. Mean Dice coefficient (`mean_dice`).
3. Pixel accuracy.
4. Per-class IoU for error analysis.
5. Confusion matrix for segmentation.

Important:

- Main metric: `mean_iou`.
- Ignore `255` pixels.
- Save per-class IoU separately, but keep only three main metrics in the model report.

Acceptance criteria:

- Unit tests cover perfect prediction, completely wrong prediction, and ignored labels.
- Metrics work on CPU and GPU tensors.

---

### Phase 7 — Models

**Goal:** Implement a model factory that can instantiate all experiment architectures.

Create `src/cityseg/models/factory.py`.

Required models:

1. `majority_baseline`
2. `tiny_unet`
3. `unet` through SMP
4. `deeplabv3plus` through SMP
5. `fpn` through SMP
6. `pspnet` through SMP if supported
7. `unetplusplus` through SMP if supported
8. Optional `segformer` through a separate implementation path

Tiny U-Net requirements:

- Encoder block 1: Conv-BN-ReLU x2 + MaxPool.
- Encoder block 2: Conv-BN-ReLU x2 + MaxPool.
- Bottleneck: Conv-BN-ReLU x2.
- Decoder: transposed convolution or bilinear upsampling + convolution.
- Skip connections through concatenation.
- Final `1x1` convolution to `num_classes`.

Acceptance criteria:

- Every model returns logits of shape `[B, 19, H, W]` for an input of shape `[B, 3, H, W]`.
- Model factory unit tests pass.

---

### Phase 8 — Training loop

**Goal:** Implement a reusable training pipeline.

Create `src/cityseg/training/train.py`.

Tasks:

1. Load config.
2. Set seed.
3. Build train/validation datasets.
4. Build model.
5. Build loss.
6. Build optimizer.
7. Build scheduler.
8. Train for configured epochs.
9. Support mixed precision with `torch.cuda.amp`.
10. Log per epoch:
    - train loss;
    - validation loss;
    - validation mean IoU;
    - validation mean Dice;
    - validation pixel accuracy;
    - learning rate.
11. Save:
    - best checkpoint by validation mean IoU;
    - last checkpoint;
    - resolved config;
    - training history CSV;
    - train/validation loss plot;
    - train/validation metric plot.
12. Add early stopping as optional config.

Output folder example:

```text
outputs/003_unet_resnet34_ce_cosine_2026-06-12_15-30-00/
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── config.yaml
├── history.csv
├── train_val_loss.png
├── train_val_mean_iou.png
└── predictions_preview.png
```

Acceptance criteria:

- Training can run for one epoch on a tiny synthetic dataset.
- Training can resume from a checkpoint if `--resume` is passed.
- Best checkpoint is selected by validation mean IoU.

---

### Phase 9 — Learning-rate schedulers

**Goal:** Demonstrate cosine annealing and step decay.

Create `src/cityseg/training/schedulers.py`.

Implement:

1. `StepLR`
2. `CosineAnnealingLR`
3. Optional `ReduceLROnPlateau`

Add experiments:

- Same model/loss with `step_decay`.
- Same model/loss with `cosine_annealing`.

Acceptance criteria:

- Training history CSV includes learning rate per epoch.
- Report includes LR curves or a short scheduler comparison.

---

### Phase 10 — Input preprocessing and augmentation experiments

**Goal:** Compare different input preprocessing strategies.

Implement `src/cityseg/data/transforms.py` using Albumentations.

Preprocessing variants:

1. `resize_only`
   - Resize image/mask to fixed resolution.
2. `random_crop`
   - Resize then random crop.
3. `basic_aug`
   - Horizontal flip.
   - Random brightness/contrast.
   - Random crop.
4. `strong_aug`
   - Horizontal flip.
   - Color jitter or brightness/contrast.
   - Gaussian blur.
   - Random scale/crop.
5. `imagenet_normalization`
   - Use ImageNet mean/std for pretrained encoders.
6. `cityscapes_normalization`
   - Compute mean/std from training images and use them.

Important:

- Apply geometric transforms to image and mask together.
- Apply color transforms only to images.
- Use nearest-neighbor interpolation for masks.

Acceptance criteria:

- At least three preprocessing strategies are evaluated.
- Results are included in the model report.

---

### Phase 11 — Evaluation and error analysis

**Goal:** Evaluate checkpoints and produce qualitative and quantitative results.

Create `src/cityseg/training/evaluate.py`.

Tasks:

1. Load checkpoint.
2. Run evaluation on validation or test split.
3. Save global metrics.
4. Save per-class IoU.
5. Save confusion matrix.
6. Save prediction examples:
   - original image;
   - ground-truth mask;
   - predicted mask;
   - overlay;
   - error map.
7. Identify common errors:
   - confusion between sidewalk/road;
   - small objects like poles, traffic signs, riders;
   - boundary errors;
   - rare classes;
   - distant objects.

Acceptance criteria:

- Evaluation can be run independently of training.
- Qualitative examples are saved to `reports/figures/`.

---

### Phase 12 — Experiment tracking

**Goal:** Keep experiments organized and reproducible without requiring a complex external service.

Create `src/cityseg/training/experiment_logger.py`.

Tasks:

1. Append one row per experiment to `reports/experiment_results.csv`.
2. Include columns:
   - `experiment_id`
   - `date`
   - `architecture`
   - `encoder`
   - `pretrained_weights`
   - `loss`
   - `optimizer`
   - `learning_rate`
   - `scheduler`
   - `epochs`
   - `batch_size`
   - `image_size`
   - `crop_size`
   - `augmentation`
   - `normalization`
   - `mean_iou`
   - `mean_iou_change_vs_baseline_pct`
   - `mean_dice`
   - `mean_dice_change_vs_baseline_pct`
   - `pixel_accuracy`
   - `pixel_accuracy_change_vs_baseline_pct`
   - `checkpoint_path`
   - `comments`
3. Preserve chronological order.
4. Do not sort automatically.

Acceptance criteria:

- Every finished training or evaluation run appends one row.
- Baseline row is always first.

---

### Phase 13 — Model report Excel file

**Goal:** Generate `reports/model_report.xlsx` that follows the lecture notes.

Create `src/cityseg/reporting/build_model_report.py` using `openpyxl`.

Report requirements:

1. First row: baseline model.
2. Each row: one hypothesis/model experiment.
3. Hyperparameters first, metrics second.
4. Max three main metrics:
   - mean IoU;
   - mean Dice;
   - pixel accuracy.
5. Each metric should include percentage change vs baseline.
6. Rightmost column must be `Comments`.
7. Add a visible note above or below the table:
   - `Best model: <experiment_id>, because <reason>.`
8. Highlight best metric values.
9. Do not make the table too wide.
10. Add charts:
    - train vs validation loss;
    - train vs validation mean IoU.
11. Optional sheet:
    - correct and incorrect prediction examples.

Suggested sheets:

1. `Summary`
2. `Experiments`
3. `Training Curves`
4. `Per-Class IoU`
5. `Examples`

Acceptance criteria:

- `make report` creates a readable Excel report.
- Best model is obvious within 1–2 seconds of opening the file.
- The table is styled, not a raw pandas export.

---

### Phase 14 — Streamlit UI

**Goal:** Create a user-facing web app for inference.

Create `app/streamlit_app.py`.

UI features:

1. Title and short project description.
2. Sidebar settings:
   - checkpoint path;
   - model architecture;
   - encoder;
   - image size;
   - opacity of overlay;
   - option to show class legend.
3. Upload image with `st.file_uploader`.
4. Preprocess uploaded image.
5. Run model inference.
6. Display:
   - original image;
   - predicted segmentation mask;
   - overlay;
   - optional top classes by pixel percentage.
7. Add button to download predicted mask as PNG.
8. Add button to download overlay as PNG.

Implementation notes:

- Cache model loading with `st.cache_resource`.
- Use CPU fallback if CUDA is unavailable.
- Use the same preprocessing as validation/inference.
- The UI should not require the full dataset, only a trained checkpoint.

Acceptance criteria:

- `make app` starts Streamlit.
- Uploading an image produces a segmentation output.
- App works on CPU for small images.

---

### Phase 15 — Tests

**Goal:** Add tests to make the project reliable and suitable for course submission.

Create tests for:

1. Label mapping:
   - valid IDs map correctly;
   - ignored labels become `255`.
2. Dataset:
   - synthetic image/mask pair loads correctly;
   - image/mask shapes are correct;
   - mask dtype is `torch.long`.
3. Losses:
   - CE/Dice/Focal return finite scalar values;
   - losses support backward pass;
   - ignore index works.
4. Metrics:
   - perfect prediction gives best score;
   - ignored pixels do not affect score.
5. Model factory:
   - each configured model returns `[B, C, H, W]`.
6. Training step:
   - one mini-batch forward/backward/update works.
7. Inference:
   - prediction returns a color mask.
8. Streamlit smoke test:
   - app module imports without running inference.

Acceptance criteria:

- `pytest -q` passes.
- Tests do not require the full Cityscapes dataset.
- Tests do not require a GPU.

---

### Phase 16 — Presentation and final documentation

**Goal:** Produce final project materials.

Create:

1. `docs/project_summary.md`
2. `docs/literature_review.md`
3. `docs/dataset_analysis.md`
4. `docs/architecture_notes.md`
5. `reports/model_report.xlsx`
6. `presentation_outline.md`

The presentation outline should contain:

1. Problem statement.
2. Dataset: Cityscapes.
3. What semantic segmentation outputs.
4. Dataset exploration findings.
5. Architectures tried.
6. Loss functions compared.
7. Learning-rate schedulers compared.
8. Preprocessing/augmentation experiments.
9. Best model and why.
10. Error analysis.
11. Streamlit demo screenshots.
12. Conclusion and future work.

Acceptance criteria:

- Documentation can be read independently of the code.
- The final report explains not only what worked, but also what did not work.

---

## 7. Recommended Experiment Plan

Run experiments in this chronological order. Keep this order in the model report.

| ID | Purpose | Architecture | Encoder | Loss | Scheduler | Preprocessing | Expected Learning |
|---:|---|---|---|---|---|---|---|
| 000 | Baseline | Majority class | N/A | N/A | N/A | N/A | Establish baseline |
| 001 | Autoencoder requirement | Tiny U-Net | Custom | Cross-entropy | Step decay | Resize only | Verify pipeline |
| 002 | Loss comparison | Tiny U-Net | Custom | Dice | Step decay | Resize only | Compare Dice vs CE |
| 003 | First strong model | U-Net | ResNet34 | Cross-entropy | Cosine | ImageNet norm + crop | Strong baseline |
| 004 | Loss comparison | U-Net | ResNet34 | Dice | Cosine | ImageNet norm + crop | Region overlap effect |
| 005 | Loss comparison | U-Net | ResNet34 | Focal | Cosine | ImageNet norm + crop | Class imbalance effect |
| 006 | Scheduler comparison | U-Net | ResNet34 | Cross-entropy | Step decay | ImageNet norm + crop | Compare with cosine |
| 007 | Advanced CNN | DeepLabV3+ | ResNet50 | Cross-entropy | Cosine | ImageNet norm + crop | Atrous convolution effect |
| 008 | Loss + advanced CNN | DeepLabV3+ | ResNet50 | Focal | Cosine | ImageNet norm + crop | Small/rare classes |
| 009 | Strong backbone | DeepLabV3+ | EfficientNet-B4 | CE + Dice | Cosine | Basic augmentation | Accuracy improvement |
| 010 | Preprocessing | U-Net | ResNet34 | CE + Dice | Cosine | Strong augmentation | Augmentation effect |
| 011 | Multi-scale model | FPN | ResNet34 | Cross-entropy | Cosine | Basic augmentation | Feature pyramid effect |
| 012 | Context model | PSPNet | ResNet34 | Cross-entropy | Cosine | Basic augmentation | Global context effect |
| 013 | Skip variant | U-Net++ | ResNet34 | CE + Dice | Cosine | Basic augmentation | Dense skip connections |
| 014 | Optional transformer | SegFormer-B0 | MiT-B0 | Cross-entropy | Cosine | Resize/crop | Transformer comparison |

Minimum acceptable number of experiments: 8, including the baseline.  
Target number of experiments: 12–15.  
Optional extended number: 20+ if GPU time allows.

---

## 8. RunPod / GPU Execution Notes

1. Start with small crops such as `256x512` or `512x512` to verify training.
2. Increase resolution only after the pipeline is stable.
3. Use mixed precision to reduce VRAM usage.
4. If CUDA runs out of memory:
   - reduce batch size;
   - reduce crop size;
   - use a smaller encoder;
   - disable heavy augmentations;
   - use gradient accumulation.
5. Save checkpoints frequently because cloud GPU sessions can terminate.
6. Keep `reports/experiment_results.csv` and best checkpoints backed up.

Suggested initial training command:

```bash
python -m src.cityseg.training.train \
  --config configs/experiments/001_tiny_unet_ce_step.yaml
```

Suggested strong-model command:

```bash
python -m src.cityseg.training.train \
  --config configs/experiments/007_deeplabv3plus_resnet50_ce_cosine.yaml
```

Evaluation command:

```bash
python -m src.cityseg.training.evaluate \
  --config configs/experiments/007_deeplabv3plus_resnet50_ce_cosine.yaml \
  --checkpoint outputs/007_deeplabv3plus_resnet50_ce_cosine/checkpoints/best.pt
```

---

## 9. Coding Instructions for Codex

When implementing, follow these rules:

1. Prefer scripts and modules over notebooks.
2. Do not hardcode paths except in default config files.
3. Do not commit raw Cityscapes data or trained checkpoints.
4. Keep all experiment parameters in YAML configs.
5. Ensure all scripts can run from the repository root.
6. Make functions small and testable.
7. Add type hints where practical.
8. Add docstrings to public functions.
9. Keep GPU-specific code optional; tests must run on CPU.
10. Use deterministic seeds where possible.
11. Save all outputs under `outputs/` or `reports/`.
12. Never normalize masks as images.
13. Always use nearest-neighbor interpolation for masks.
14. Always respect `ignore_index=255` in losses and metrics.
15. Make the Streamlit app depend only on a checkpoint and config, not on the training dataset.

---

## 10. Definition of Done

The project is complete when all of the following are true:

1. Cityscapes dataset loading works.
2. EDA report and visualizations are generated.
3. At least one baseline and at least seven neural experiments are completed.
4. Cross-entropy, Dice, and focal loss are implemented and compared.
5. Step decay and cosine annealing are implemented and compared.
6. At least three preprocessing/augmentation strategies are tested.
7. U-Net or tiny U-Net satisfies the autoencoder-type model requirement.
8. DeepLabV3+ is trained and evaluated.
9. Additional architectures are attempted if time allows.
10. Model report Excel file is generated and styled.
11. Best model is explicitly stated and justified.
12. Train/validation loss and metric plots are included.
13. Error analysis examples are included.
14. Streamlit UI works for inference.
15. Tests pass with `pytest -q`.
16. README explains how to install, train, evaluate, test, and run the app.
17. Final project summary and presentation outline are created.

---

## 11. Suggested First Codex Prompt

Use this prompt to start implementation:

```text
You are working in a Python/PyTorch repository for a Cityscapes semantic segmentation course project. Implement the project structure described in semantic-segmentation-cityscapes-implementation-plan.md.

Start with Phase 0 to Phase 3 only:
1. Create the repository structure.
2. Add requirements.txt, Makefile, .gitignore, and README.md.
3. Implement config loading with YAML overrides.
4. Implement Cityscapes constants, label mapping, and a Dataset class.
5. Add synthetic-data unit tests for label mapping and dataset loading.
6. Do not implement training yet.
7. Ensure pytest passes on CPU without requiring the Cityscapes dataset.

Follow the coding instructions in the implementation plan exactly.
```

After that, continue with:

```text
Continue with Phase 4 to Phase 8:
1. Implement the majority-class baseline.
2. Implement losses, metrics, model factory, tiny U-Net, and the training loop.
3. Add tests for losses, metrics, model output shapes, and one training step.
4. Add configs for experiments 000–007.
5. Ensure pytest passes on CPU.
```

Then:

```text
Continue with Phase 9 to Phase 16:
1. Add schedulers, augmentations, evaluation, experiment logging, model report generation, Streamlit app, and final documentation templates.
2. Add smoke tests for inference/reporting/app import.
3. Ensure the complete workflow is documented in README.md.
```
