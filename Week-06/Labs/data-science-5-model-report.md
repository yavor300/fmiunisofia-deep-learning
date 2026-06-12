# Data Science Task 05 - Panoptic Segmentation Report

## Best Model
M2 (U-Net semantic mask + Mask R-CNN instance overlays) is the best model because it keeps the semantic branch performance from M1 and adds instance-level panoptic fusion with measurable foreground recall.

## Dataset Exploration
- Dataset: `DATA/relabelled_coco`
- Number of semantic classes used (including background): 21
- Classes used: background, sky-other-merged, person, tree-merged, wall-other-merged, grass-merged, sea, building-other-merged, playingfield, pavement-merged, road, sand, snow, dirt-merged, window-other...

## Train / Validation / Test
- Split sizes: 280/60/60
- Epochs: 3
- Batch size: 6
- Learning rate: 0.001
- Image size: 128x128
- Max samples used: 400
- Top semantic categories used: 20
- Mask R-CNN instance threshold: 0.5

## Experiments Table
| Model ID | Hypothesis | Epochs | Batch Size | Learning Rate | Image Size | Test Pixel Accuracy | Delta vs Baseline (Acc) | Test Mean IoU | Delta vs Baseline (mIoU) | Test Foreground Recall | Delta vs Baseline (Recall) | Context/Explanation | Comments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0 | Baseline (predict background class only) | - | - | - | 128x128 | 0.5093 | 0.00% | 0.0268 | 0.00% | 0.0000 | 0.00% | Greedy statistical baseline used as the first row, following the model-report guideline. | Reference point for all percentage-change columns. |
| M1 | U-Net semantic segmentation | 3 | 6 | 0.001 | 128x128 | 0.5475 | +7.50% | 0.0520 | +93.92% | 0.0000 | 0.00% | Semantic branch predicts one class per pixel using relabelled COCO masks. | Improves semantic segmentation metrics over the baseline. |
| M2 | U-Net + Mask R-CNN panoptic fusion | 3 | 6 | 0.001 | 128x128 | 0.5475 | +7.50% | 0.0520 | +93.92% | 0.4787 | N/A | Combines the semantic U-Net mask with Mask R-CNN instance masks as described in the segmentation notes. | Best model because it adds instance-level behavior while preserving the semantic branch metrics. |

## Semantic Branch (U-Net)
### Validation
- Loss: 1.9863
- Pixel Accuracy: 0.5255
- Mean IoU: 0.0462

### Test
- Loss: 1.7562
- Pixel Accuracy: 0.5475
- Mean IoU: 0.0520

## Panoptic Fusion (Semantic + Mask R-CNN)
- Panoptic coverage: 0.4749
- Foreground recall: 0.4787
- Average fused instance masks per image: 6.32

## Diagrams
1. Train vs Validation metric (Mean IoU): shows whether semantic segmentation quality improves similarly on train and validation data.
2. Train vs Validation loss: shows optimization progress and whether train/validation losses diverge.
