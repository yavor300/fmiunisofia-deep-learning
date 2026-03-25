# Model Report - Task 9

## Context
- Dataset: `water_train.csv` + `water_test.csv`.
- Missing values handled with median imputation.
- Selected features by absolute correlation with target: Solids, Turbidity, Organic_carbon, Conductivity, Sulfate, Chloramines, ph.

## Experiments Table
| Hypothesis | Architecture / Strategy | Epochs | Batch Size | Learning Rate | Optimizer | Test BCE Loss (Δ vs baseline) | Test F1 (Δ vs baseline) | Test Accuracy (Δ vs baseline) | Comments |
|---|---|---:|---:|---:|---|---|---|---|---|
| baseline_majority_class | Predict the most frequent class (0) | N/A | N/A | N/A | N/A | 4.0955 (+0.00%) | 0.0000 (+0.00%) | 0.5905 (+0.00%) | Baseline model required by the reporting standard. |
| nn_adamw_model | Linear(7->32)->ReLU->Dropout(0.2)->Linear(32->16)->ReLU->Dropout(0.2)->Linear(16->1) | 30 | 8 | 0.001 | AdamW | 0.5928 (+85.52%) | 0.5104 (N/A) | 0.6720 (+13.80%) | Trained neural network. Better test F1/accuracy than baseline. |

## Best Model
Best model: **nn_adamw_model**, because it achieved stronger test-set predictive performance than the baseline (F1 `0.5104` vs `0.0000`, accuracy `0.6720` vs `0.5905`).

## Diagrams
- Train vs validation loss curve: `data-science-9-training-curves.png` (left panel).
- Train vs validation F1 curve: `data-science-9-training-curves.png` (right panel).