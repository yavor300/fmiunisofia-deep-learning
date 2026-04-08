# Model Report - Task 02

Best model: **vgg11** because it has the highest weighted F1 on the test set (0.8482) while maintaining strong accuracy (0.8506).

## Dataset Setup
- Classes: glioma, meningioma, notumor, pituitary
- Train images: 5712
- Validation images: 655
- Test images: 656

## Transfer Learning
- custom_cnn: fallback to random initialization
- inception_v3: pretrained backbone used
- vgg11: pretrained backbone used
- resnet18: fallback to random initialization

## Main Experiment Table
Rows are kept in experiment order. First row is the baseline model.

| Hypothesis | Architecture | Epochs | Batch Size | Learning Rate | Optimizer | Test Accuracy (vs baseline) | Test F1 Weighted (vs baseline) | Test Recall Weighted (vs baseline) | Comments |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| baseline_majority_class | Predict most common train class (notumor) | n/a | n/a | n/a | n/a | 0.3095 (+0.00%) | 0.1463 (+0.00%) | 0.3095 (+0.00%) | Greedy statistical baseline; no learning. |
| custom_cnn | CustomTumorCNN (random init) | 3 | 24 | 0.001 | AdamW | 0.6372 (+105.91%) | 0.6048 (+313.52%) | 0.6372 (+105.91%) | Improves baseline. Reload stable. |
| inception_v3 | InceptionV3 (pretrained) | 3 | 24 | 0.001 | AdamW | 0.8491 (+174.38%) | 0.8446 (+477.49%) | 0.8491 (+174.38%) | Improves baseline. Reload delta F1=0.694116. |
| vgg11 | VGG11 (pretrained) | 3 | 24 | 0.001 | AdamW | 0.8506 (+174.88%) | 0.8482 (+479.90%) | 0.8506 (+174.88%) | Improves baseline. Reload stable. |
| resnet18 | ResNet18 (random init) | 3 | 24 | 0.001 | AdamW | 0.7119 (+130.05%) | 0.7085 (+384.44%) | 0.7119 (+130.05%) | Improves baseline. Reload stable. |

## Diagrams
- Train vs validation loss: `data-science-2-train-vs-val-loss.png`
- Train vs validation main metric (F1 weighted): `data-science-2-train-vs-val-metric.png`

## Notes
- The table is not sorted; it follows experiment creation order.
- Metrics include value and percentage change vs baseline.