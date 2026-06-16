# Dataset Analysis

## Overview

- Dataset root: `data/raw/cityscapes`
- Total images found: 5000
- Total masks found: 5000
- Number of train classes: 19
- Pixel/figure sample limit per split: 50

## Images Per Split

- `train`: 2975 images; 50 sampled for pixel EDA
- `val`: 500 images; 50 sampled for pixel EDA
- `test`: 1525 images; 50 sampled for pixel EDA

## Image Size Distribution

| Width | Height | Images |
| ---: | ---: | ---: |
| 2048 | 1024 | 150 |

## Class Distribution

| Class | Pixels | Percentage |
| --- | ---: | ---: |
| road | 70268061 | 38.0954% |
| sidewalk | 11384115 | 6.1718% |
| building | 43798798 | 23.7452% |
| wall | 602854 | 0.3268% |
| fence | 1259517 | 0.6828% |
| pole | 3194979 | 1.7321% |
| traffic light | 400655 | 0.2172% |
| traffic sign | 1106086 | 0.5997% |
| vegetation | 28825446 | 15.6275% |
| terrain | 1504880 | 0.8159% |
| sky | 5431163 | 2.9445% |
| person | 2327441 | 1.2618% |
| rider | 262899 | 0.1425% |
| car | 11670936 | 6.3273% |
| truck | 708579 | 0.3842% |
| bus | 580826 | 0.3149% |
| train | 101684 | 0.0551% |
| motorcycle | 371341 | 0.2013% |
| bicycle | 652639 | 0.3538% |

## Class Imbalance

The dominant class is `road` with 38.10% of labeled pixels. The rarest observed classes are: train, rider, motorcycle, traffic light, bus. This imbalance matters for training: plain pixel-wise cross-entropy can over-reward predictions of frequent classes, while focal loss can emphasize hard or under-represented pixels and Dice loss can optimize region overlap so small classes are not judged only by raw pixel volume.

## Anomaly Checks

- `missing_images`: 0
- `missing_masks`: 0
- `unreadable_files`: 0
- `invalid_mask_values`: 0
- `mismatched_dimensions`: 0

No structural anomalies were found in the scanned files.

## Figures

- `reports/figures/class_distribution.png`
- `reports/figures/image_size_distribution.png`
- `reports/figures/sample_overlays.png`
- `reports/figures/rare_classes_examples.png`
