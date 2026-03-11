from __future__ import annotations

import abc
from typing import Any, List

_TORCH = None


def _get_torch():
    global _TORCH
    if _TORCH is None:
        import torch  # type: ignore

        _TORCH = torch
    return _TORCH


class Module(abc.ABC):
    @abc.abstractmethod
    def forward(self, input_tensor: Any) -> Any:
        raise NotImplementedError

    def __call__(self, input_tensor: Any) -> Any:
        return self.forward(input_tensor)


class Sigmoid(Module):
    def forward(self, input_tensor: Any) -> Any:
        torch = _get_torch()
        return 1.0 / (1.0 + torch.exp(-input_tensor))


class Tanh(Module):
    def forward(self, input_tensor: Any) -> Any:
        torch = _get_torch()
        exp_2x = torch.exp(2.0 * input_tensor)
        return (exp_2x - 1.0) / (exp_2x + 1.0)


class ReLU(Module):
    def forward(self, input_tensor: Any) -> Any:
        torch = _get_torch()
        zeros = torch.zeros_like(input_tensor)
        return torch.where(input_tensor > 0.0, input_tensor, zeros)


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 1e-2) -> None:
        self.negative_slope = negative_slope

    def forward(self, input_tensor: Any) -> Any:
        torch = _get_torch()
        return torch.where(input_tensor >= 0.0, input_tensor, self.negative_slope * input_tensor)


class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        self.modules: List[Module] = list(modules)

    def forward(self, input_tensor: Any) -> Any:
        output = input_tensor
        for module in self.modules:
            output = module(output)
        return output

    def append(self, module: Module) -> None:
        self.modules.append(module)

    def extend(self, sequential: "Sequential") -> None:
        self.modules.extend(sequential.modules)

    def insert(self, index: int, module: Module) -> None:
        self.modules.insert(index, module)
