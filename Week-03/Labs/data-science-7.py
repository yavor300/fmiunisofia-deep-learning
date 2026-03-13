import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


network_1 = nn.Sequential(
    nn.Linear(8, 6),
    nn.Linear(6, 4),
    nn.Linear(4, 2),
)

network_2 = nn.Sequential(
    nn.Linear(8, 12),
    nn.Linear(12, 10),
    nn.Linear(10, 6),
    nn.Linear(6, 2),
)

print(f"Number of parameters in network 1: {count_parameters(network_1)}")
print(f"Number of parameters in network 2: {count_parameters(network_2)}")
