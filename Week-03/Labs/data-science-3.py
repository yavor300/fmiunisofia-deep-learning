import torch
import torch.nn as nn


torch.manual_seed(42)

temperature_observation = torch.tensor([3, 4, 6, 2, 3, 6, 8, 9], dtype=torch.float32).unsqueeze(0)

model = nn.Sequential(
    nn.Linear(8, 1),
    nn.Sigmoid(),
)

# B
spring_confidence = model(temperature_observation)
print(spring_confidence.item())
