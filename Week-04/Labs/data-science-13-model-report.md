# Model Report - Task 13

Best model: **DigitCNN_v1**. Why: DigitCNN_v1 is the best model because it improves both test accuracy and test macro F1 over the baseline.

## Experiment Table
Each row is a hypothesis. Baseline is first. Metrics are on the test set.

| Hypothesis | Architecture | Epochs | Batch Size | Learning Rate | Optimizer | Test Accuracy (vs baseline) | Test Macro F1 (vs baseline) | Comments |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| Baseline_majority_class | Predict most frequent training class (1) | n/a | n/a | n/a | n/a | 0.1000 (+0.00%) | 0.0182 (+0.00%) | Greedy statistical baseline; no learning. |
| DigitCNN_v1 | Conv2d(1->16) + ReLU + MaxPool2d(2) + Conv2d(16->32) + ReLU + MaxPool2d(2) + Linear(128->10) | 20 | 64 | 0.001 | AdamW | 0.9472 (+847.22%) | 0.9460 (+5103.08%) | Learns spatial patterns and outperforms baseline on both metrics. |

## Data And Setup
- Dataset: `sklearn.datasets.load_digits`
- Input: 8x8 grayscale images, 10 classes.
- Splits: train/validation/test from stratified splits.
- Main metric: Accuracy. Secondary metric: Macro F1.

## Diagrams
- Train vs validation loss: `data-science-13-training-curves.png` (left panel).
- Train vs validation metric (accuracy): `data-science-13-training-curves.png` (right panel).

## Notes
- Rows are kept in experiment order (baseline first), not sorted.
- The best model row should be highlighted when moving this table to Excel.