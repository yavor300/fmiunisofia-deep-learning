import numpy as np
import torch
from torch.utils.data import TensorDataset


np.random.seed(42)

data = np.random.uniform(0, 1, size=(12, 9))

features = torch.from_numpy(data[:, :-1])
targets = torch.from_numpy(data[:, -1:])
dataset = TensorDataset(features, targets)

last_sample, last_label = dataset[-1]
print(f"Last sample: {last_sample}")
print(f"Last label: {last_label}")
