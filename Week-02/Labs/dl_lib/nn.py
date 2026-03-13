from __future__ import annotations

import abc
import math
from typing import Any, Dict, Iterable, List

_TORCH = None


# safe module import until torch is actually needed.
def _get_torch():
    global _TORCH
    if _TORCH is None:
        import torch

        _TORCH = torch
    return _TORCH


class Module(abc.ABC):
    def __init__(self) -> None:
        self.training = True

    @abc.abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def _set_mode_recursive(self, value: Any, mode: bool) -> None:
        if isinstance(value, Module):
            value.train(mode)
        elif isinstance(value, dict):
            for sub_value in value.values():
                self._set_mode_recursive(sub_value, mode)
        elif isinstance(value, (list, tuple, set)):
            for sub_value in value:
                self._set_mode_recursive(sub_value, mode)

    def train(self, mode: bool = True) -> "Module":
        self.training = mode
        for value in self.__dict__.values():
            self._set_mode_recursive(value, mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def _collect_parameters(self, value: Any, params: List[Any], seen: set[int]) -> None:
        torch = _get_torch()
        if isinstance(value, Module):
            for parameter in value.parameters():
                param_id = id(parameter)
                if param_id not in seen:
                    seen.add(param_id)
                    params.append(parameter)
        elif isinstance(value, dict):
            for sub_value in value.values():
                self._collect_parameters(sub_value, params, seen)
        elif isinstance(value, (list, tuple, set)):
            for sub_value in value:
                self._collect_parameters(sub_value, params, seen)
        elif torch.is_tensor(value) and value.requires_grad:
            param_id = id(value)
            if param_id not in seen:
                seen.add(param_id)
                params.append(value)

    def parameters(self) -> List[Any]:
        params: List[Any] = []
        seen: set[int] = set()
        for value in self.__dict__.values():
            self._collect_parameters(value, params, seen)
        return params


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
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input_tensor: Any) -> Any:
        torch = _get_torch()
        return torch.where(input_tensor >= 0.0, input_tensor, self.negative_slope * input_tensor)


class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
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


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive integers")

        torch = _get_torch()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        k = 1.0 / in_features
        bound = math.sqrt(k)
        self.weight = torch.empty((out_features, in_features), dtype=torch.float32, requires_grad=True)
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

        if bias:
            self.bias = torch.empty((out_features,), dtype=torch.float32, requires_grad=True)
            with torch.no_grad():
                self.bias.uniform_(-bound, bound)
        else:
            self.bias = None

    def forward(self, input_tensor: Any) -> Any:
        output = input_tensor.matmul(self.weight.t())
        if self.bias is not None:
            output = output + self.bias
        return output

    def parameters(self) -> List[Any]:
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params


class Softmax(Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input_tensor: Any) -> Any:
        torch = _get_torch()
        return torch.softmax(input_tensor, dim=self.dim)


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if p < 0.0 or p > 1.0:
            raise ValueError("Dropout probability has to be between 0 and 1")
        self.p = p

    def forward(self, input_tensor: Any) -> Any:
        torch = _get_torch()
        if not self.training or self.p == 0.0:
            return input_tensor
        if self.p == 1.0:
            return torch.zeros_like(input_tensor)
        keep_prob = 1.0 - self.p
        mask = (torch.rand_like(input_tensor) < keep_prob).to(input_tensor.dtype) / keep_prob
        return input_tensor * mask


class BCEWithLogitsLoss(Module):
    def __init__(self, reduction: str = "mean", pos_weight: Any = None) -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of: 'none', 'mean', 'sum'")
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, input_tensor: Any, target_tensor: Any) -> Any:
        torch = _get_torch()
        return torch.nn.functional.binary_cross_entropy_with_logits(
            input_tensor,
            target_tensor.to(input_tensor.dtype),
            reduction=self.reduction,
            pos_weight=self.pos_weight,
        )


class CrossEntropyLoss(Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of: 'none', 'mean', 'sum'")
        self.reduction = reduction

    def forward(self, input_tensor: Any, target_tensor: Any) -> Any:
        torch = _get_torch()
        return torch.nn.functional.cross_entropy(
            input_tensor, target_tensor.to(torch.long), reduction=self.reduction
        )
