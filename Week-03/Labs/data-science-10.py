import ast
from pathlib import Path

import torch
import torch.nn as nn


def extract_input_tensor() -> torch.Tensor:
    readme_path = Path(__file__).with_name("README.md")
    text = readme_path.read_text(encoding="utf-8")
    marker = "input_tensor = torch.tensor("
    start = text.find(marker)

    if start == -1:
        return torch.rand(1, 128)

    idx = start + len(marker)
    depth = 1
    chars = []

    while idx < len(text) and depth > 0:
        ch = text[idx]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                break
        chars.append(ch)
        idx += 1

    values = ast.literal_eval("".join(chars).strip())
    return torch.tensor(values, dtype=torch.float32)


input_tensor = extract_input_tensor()

model = nn.Sequential(
    nn.Linear(input_tensor.shape[1], 16),
    nn.ReLU(),
    nn.Dropout(p=0.8),
)

model.train()
output = model(input_tensor)
print(output)
