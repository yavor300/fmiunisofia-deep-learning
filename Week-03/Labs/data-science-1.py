import torch


temperatures = torch.tensor([[72, 75, 78], [70, 73, 76]])
print(f"Temperatures: {temperatures}")
print(f"Shape of temperatures: {temperatures.shape}")
print(f"Data type of temperatures: {temperatures.dtype}")

corrected_temperatures = temperatures + 2
print(f"Corrected temperatures: {corrected_temperatures}")
