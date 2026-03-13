# Task 6 Model Report

## Setup
- Features: `experience_level`, `employment_type`, `remote_ratio`, `company_size`.
- Target: `salary_in_usd`.
- Categorical encoding: One-hot encoding.
- Normalization: StandardScaler on features and target.
- Training: 20 epochs, batch size 8, AdamW, lr=0.001.

## Final Epoch Losses
- nn_with_sigmoid: 0.776745462496864
- nn_with_relu: 0.7593583499814601
- nn_with_leakyrelu: 0.7638996735532233

## Best Model
- nn_with_relu (0.7593583499814601).

Plot saved at `data-science-6-losses.png`.