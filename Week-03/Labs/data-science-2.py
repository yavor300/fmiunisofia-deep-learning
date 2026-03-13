import torch
import torch.nn as nn


torch.manual_seed(42)

temperature_observation = torch.tensor([2, 3, 6, 7, 9, 3, 2, 1], dtype=torch.float32).unsqueeze(0)

model = nn.Sequential(
    nn.Linear(8, 12),
    nn.ReLU(),
    nn.Linear(12, 1),
)

logit = model(temperature_observation)
print(logit)
