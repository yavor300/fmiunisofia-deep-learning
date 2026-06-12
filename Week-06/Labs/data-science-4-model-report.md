# Data Science Task 04 - Model Report

## Best Model
M1 (UNet_v1) is the best model because it achieves higher test Dice than the baseline while keeping stable IoU and pixel accuracy.

## Dataset Context
- Dataset: `DATA/segmentation_cats_dogs`
- Split (train/val/test): 5173/1108/1109

## Best Validation Metrics (M1)
- Loss: 0.3326
- Pixel Accuracy: 0.8538
- IoU: 0.6081
- Dice: 0.7548

## Experiments Table
| Model ID | Hypothesis | Epochs | Batch Size | Learning Rate | Image Size | Test Pixel Accuracy | Δ vs Baseline (Acc) | Test IoU | Δ vs Baseline (IoU) | Test Dice | Δ vs Baseline (Dice) | Context/Explanation | Comments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0 | Baseline (all-background predictor) | - | - | - | 128x128 | 0.7016 | 0.00% | 0.0000 | 0.00% | 0.0000 | 0.00% | Greedy statistical baseline used as first row in the report. | Serves only as a reference point for percentage change. |
| M1 | UNet_v1 (binary semantic segmentation) | 5 | 8 | 0.001 | 128x128 | 0.8518 | +21.41% | 0.6030 | N/A | 0.7506 | N/A | Binary semantic segmentation with U-Net trained on foreground-vs-non-foreground masks. | Selected as best model due to superior Dice and consistent IoU/accuracy compared to M0. |

## Diagrams
1. Train vs Validation metric (Dice): tracks segmentation quality over epochs.
2. Train vs Validation loss: tracks optimization and generalization gap over epochs.
