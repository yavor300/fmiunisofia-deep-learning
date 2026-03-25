# Model Report - Task 6

## Context
- Features: `experience_level`, `employment_type`, `remote_ratio`, `company_size`.
- Target: `salary_in_usd`.
- Task constraint: full dataset was used for training (no separate test split in this exercise).

## Experiments Table
| Hypothesis | Activation | Epochs | Batch Size | Learning Rate | Optimizer | Final Train MSE (Δ vs baseline) | Comments |
|---|---|---:|---:|---:|---|---|---|
| nn_with_sigmoid | Sigmoid | 20 | 8 | 0.001 | AdamW | 0.776745 (+0.00%) | Baseline model. |
| nn_with_relu | ReLU | 20 | 8 | 0.001 | AdamW | 0.759358 (+2.24%) | Best final train MSE among tested activations. |
| nn_with_leakyrelu | LeakyReLU | 20 | 8 | 0.001 | AdamW | 0.763900 (+1.65%) | Better than baseline, but not the best. |

## Best Model
Best model: **nn_with_relu**, because it achieved the lowest final train MSE (`0.759358`).

## Diagrams
- Train loss curve (all hypotheses): `data-science-6-losses.png`.