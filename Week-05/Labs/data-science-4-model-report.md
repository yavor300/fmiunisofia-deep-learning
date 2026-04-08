# Model Report - Task 04

Best model: **yolo** because it has the highest test F1 score (0.6350).

## Context
- YOLO loaded from existing checkpoint and evaluated.

## Main Experiment Table
Rows are kept in experiment order. First row is the baseline model.

| Hypothesis | Architecture | Epochs | Batch Size | Learning Rate | Optimizer | Test Precision (vs baseline) | Test Recall (vs baseline) | Test F1 (vs baseline) | Comments |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| baseline_no_detection | Predict no objects | n/a | n/a | n/a | n/a | 0.0000 (n/a) | 0.0000 (n/a) | 0.0000 (n/a) | Greedy statistical baseline for detection. |
| rcnn_like | FasterRCNN-MobileNetV3 | loaded checkpoint | 4 | 0.0005 | AdamW | 0.5812 (n/a) | 0.6491 (n/a) | 0.6133 (n/a) | Mean IoU=0.8115. Reload stable. |
| faster_rcnn | FasterRCNN-ResNet50 | loaded checkpoint | 4 | 0.0005 | AdamW | 0.1818 (n/a) | 0.3314 (n/a) | 0.2348 (n/a) | Mean IoU=0.6554. Reload delta F1=0.001058. |
| yolo | YOLOv8n | loaded checkpoint | 4 | internal (ultralytics default) | internal (ultralytics default) | 0.8727 (n/a) | 0.4990 (n/a) | 0.6350 (n/a) | Mean IoU=0.6354. Reload stable. |

## Diagrams
- Train vs validation loss: `data-science-4-train-vs-val-loss.png`
- Train vs validation main metric (F1): `data-science-4-train-vs-val-metric.png`

## Notes
- Metrics include value and percentage change vs baseline.
- Table is intentionally not sorted.
- Best row should be highlighted/bolded when moved to Excel.