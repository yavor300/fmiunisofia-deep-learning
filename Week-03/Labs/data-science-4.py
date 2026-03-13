import torch
import torch.nn.functional as F


torch.set_default_dtype(torch.float64)

y = [2]
scores = torch.tensor([[0.1, 6.0, -2.0, 3.2]])

loss = F.cross_entropy(scores, torch.tensor(y))
print(loss)
