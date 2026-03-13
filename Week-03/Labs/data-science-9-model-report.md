# Task 9 Model Report

## Data and Feature Selection
- Dataset: `water_train.csv` + `water_test.csv`.
- Missing values handled with median imputation.
- Selected features by absolute correlation with target: Solids, Turbidity, Organic_carbon, Conductivity, Sulfate, Chloramines, ph.

## Model and Training
- Architecture: Linear(7->32) + ReLU + Dropout(0.2) + Linear(32->16) + ReLU + Dropout(0.2) + Linear(16->1).
- Epochs: 30
- Batch size: 8
- Learning rate: 0.001
- Optimizer: AdamW
- Loss: BCEWithLogitsLoss
- Metric: F1 score

## Final Results
- Final validation loss: 0.6092544513869571
- Final validation F1: 0.4615384615384615
- Test loss: 0.5928210987011434
- Test F1: 0.5103857566765578
- Curves saved at `data-science-9-training-curves.png`.