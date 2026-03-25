# Model Report - Task 12

## Context
- MNIST loaded from Hugging Face: `ylecun/mnist`.
- Images flattened to 784 features and normalized to [0, 1].

## Experiments Table
| Hypothesis | Architecture / Strategy | Epochs | Batch Size | Learning Rate | Optimizer | Test CE Loss (Δ vs baseline) | Test Accuracy (Δ vs baseline) | Comments |
|---|---|---:|---:|---:|---|---|---|---|
| baseline_majority_class | Predict the most common class (1) for every image | N/A | N/A | N/A | N/A | 2.347651 (+0.00%) | 0.1135 (+0.00%) | Baseline model required by reporting standard. |
| nn_linear_mnist | Linear(784->256)->ReLU->Linear(256->128)->ReLU->Linear(128->10) | 5 | 128 | 0.001 | AdamW | 0.080691 (+96.56%) | 0.9748 (+758.85%) | Trained model with strong improvement over baseline. |

## Best Model
Best model: **nn_linear_mnist**, because it improved both core test metrics versus baseline (CE loss `0.080691` vs `2.347651`, accuracy `0.9748` vs `0.1135`).

## Diagrams
- Train vs validation loss curve: `data-science-12-training-curves.png` (left panel).
- Train vs validation accuracy curve: `data-science-12-training-curves.png` (right panel).