from __future__ import annotations

import abc
import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

import torch


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
        return 1.0 / (1.0 + torch.exp(-input_tensor))


class Tanh(Module):
    def forward(self, input_tensor: Any) -> Any:
        exp_2x = torch.exp(2.0 * input_tensor)
        return (exp_2x - 1.0) / (exp_2x + 1.0)


class ReLU(Module):
    def forward(self, input_tensor: Any) -> Any:
        zeros = torch.zeros_like(input_tensor)
        return torch.where(input_tensor > 0.0, input_tensor, zeros)


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 1e-2) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input_tensor: Any) -> Any:
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
        return torch.softmax(input_tensor, dim=self.dim)


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if p < 0.0 or p > 1.0:
            raise ValueError("Dropout probability has to be between 0 and 1")
        self.p = p

    def forward(self, input_tensor: Any) -> Any:
        if not self.training or self.p == 0.0:
            return input_tensor
        if self.p == 1.0:
            return torch.zeros_like(input_tensor)
        keep_prob = 1.0 - self.p
        mask = (torch.rand_like(input_tensor) < keep_prob).to(input_tensor.dtype) / keep_prob
        return input_tensor * mask


def _to_ntuple(
    value: Union[int, Sequence[int]],
    n: int,
    name: str,
    *,
    allow_zero: bool = True,
) -> Tuple[int, ...]:
    if isinstance(value, int):
        tuple_value = (value,) * n
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        tuple_value = tuple(value)
        if len(tuple_value) != n:
            raise ValueError(f"{name} must have length {n}")
    else:
        raise TypeError(f"{name} must be an int or a tuple/list of length {n}")

    for item in tuple_value:
        if not isinstance(item, int):
            raise TypeError(f"all values in {name} must be integers")
        if allow_zero:
            if item < 0:
                raise ValueError(f"all values in {name} must be >= 0")
        elif item <= 0:
            raise ValueError(f"all values in {name} must be > 0")
    return tuple_value


def _compute_same_padding(input_size: int, kernel_size: int, stride: int) -> Tuple[int, int]:
    output_size = math.ceil(input_size / stride)
    needed = max((output_size - 1) * stride + kernel_size - input_size, 0)
    pad_left = needed // 2
    pad_right = needed - pad_left
    return pad_left, pad_right


def _pad_same_nd(input_tensor: Any, kernel_size: Tuple[int, ...], stride: Tuple[int, ...]) -> Any:
    spatial_shape = input_tensor.shape[-len(kernel_size) :]
    pads: List[int] = []
    for input_size, kernel, step in zip(reversed(spatial_shape), reversed(kernel_size), reversed(stride)):
        pad_left, pad_right = _compute_same_padding(int(input_size), kernel, step)
        pads.extend([pad_left, pad_right])
    return torch.nn.functional.pad(input_tensor, tuple(pads))


class MaxPool2d(Module):
    def __init__(
        self,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int], None] = None,
        padding: Union[int, Sequence[int]] = 0,
    ) -> None:
        super().__init__()
        self.kernel_size = _to_ntuple(kernel_size, 2, "kernel_size", allow_zero=False)
        self.stride = (
            self.kernel_size
            if stride is None
            else _to_ntuple(stride, 2, "stride", allow_zero=False)
        )
        self.padding = _to_ntuple(padding, 2, "padding", allow_zero=True)

    def forward(self, input_tensor: Any) -> Any:
        if not torch.is_tensor(input_tensor):
            raise TypeError("input_tensor must be a torch tensor")

        if self.padding != (0, 0):
            pad_h, pad_w = self.padding
            input_tensor = torch.nn.functional.pad(
                input_tensor,
                (pad_w, pad_w, pad_h, pad_h),
                mode="constant",
                value=float("-inf"),
            )
        return torch.nn.functional.max_pool2d(
            input_tensor,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,
        )


class AvgPool2d(Module):
    def __init__(
        self,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int], None] = None,
        padding: Union[int, Sequence[int]] = 0,
    ) -> None:
        super().__init__()
        self.kernel_size = _to_ntuple(kernel_size, 2, "kernel_size", allow_zero=False)
        self.stride = (
            self.kernel_size
            if stride is None
            else _to_ntuple(stride, 2, "stride", allow_zero=False)
        )
        self.padding = _to_ntuple(padding, 2, "padding", allow_zero=True)

    def forward(self, input_tensor: Any) -> Any:
        if not torch.is_tensor(input_tensor):
            raise TypeError("input_tensor must be a torch tensor")

        if self.padding != (0, 0):
            pad_h, pad_w = self.padding
            input_tensor = torch.nn.functional.pad(
                input_tensor,
                (pad_w, pad_w, pad_h, pad_h),
                mode="constant",
                value=0.0,
            )
        return torch.nn.functional.avg_pool2d(
            input_tensor,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,
        )


class Conv1d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[str, int, Sequence[int]] = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive integers")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _to_ntuple(kernel_size, 1, "kernel_size", allow_zero=False)
        self.stride = _to_ntuple(stride, 1, "stride", allow_zero=False)
        self.padding = padding
        self.use_bias = bias

        fan_in = in_channels * self.kernel_size[0]
        bound = 1.0 / math.sqrt(fan_in)
        self.weight = torch.empty(
            (out_channels, in_channels, self.kernel_size[0]),
            dtype=torch.float32,
            requires_grad=True,
        )
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

        if bias:
            self.bias = torch.empty((out_channels,), dtype=torch.float32, requires_grad=True)
            with torch.no_grad():
                self.bias.uniform_(-bound, bound)
        else:
            self.bias = None

    def _resolve_padding(self, input_tensor: Any) -> Tuple[Any, Union[int, Tuple[int, ...]]]:
        if isinstance(self.padding, str):
            padding_mode = self.padding.lower()
            if padding_mode == "valid":
                return input_tensor, 0
            if padding_mode == "same":
                padded = _pad_same_nd(input_tensor, self.kernel_size, self.stride)
                return padded, 0
            raise ValueError("padding must be 'valid', 'same', an int or a tuple of ints")
        padding_tuple = _to_ntuple(self.padding, 1, "padding", allow_zero=True)
        return input_tensor, padding_tuple[0]

    def forward(self, input_tensor: Any) -> Any:
        if not torch.is_tensor(input_tensor):
            raise TypeError("input_tensor must be a torch tensor")

        input_tensor, padding_value = self._resolve_padding(input_tensor)
        return torch.nn.functional.conv1d(
            input_tensor,
            self.weight,
            self.bias,
            stride=self.stride[0],
            padding=padding_value,
        )


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[str, int, Sequence[int]] = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive integers")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _to_ntuple(kernel_size, 2, "kernel_size", allow_zero=False)
        self.stride = _to_ntuple(stride, 2, "stride", allow_zero=False)
        self.padding = padding
        self.use_bias = bias

        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        bound = 1.0 / math.sqrt(fan_in)
        self.weight = torch.empty(
            (out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]),
            dtype=torch.float32,
            requires_grad=True,
        )
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

        if bias:
            self.bias = torch.empty((out_channels,), dtype=torch.float32, requires_grad=True)
            with torch.no_grad():
                self.bias.uniform_(-bound, bound)
        else:
            self.bias = None

    def _resolve_padding(self, input_tensor: Any) -> Tuple[Any, Union[int, Tuple[int, ...]]]:
        if isinstance(self.padding, str):
            padding_mode = self.padding.lower()
            if padding_mode == "valid":
                return input_tensor, 0
            if padding_mode == "same":
                padded = _pad_same_nd(input_tensor, self.kernel_size, self.stride)
                return padded, 0
            raise ValueError("padding must be 'valid', 'same', an int or a tuple of ints")
        padding_tuple = _to_ntuple(self.padding, 2, "padding", allow_zero=True)
        return input_tensor, padding_tuple

    def forward(self, input_tensor: Any) -> Any:
        if not torch.is_tensor(input_tensor):
            raise TypeError("input_tensor must be a torch tensor")

        input_tensor, padding_value = self._resolve_padding(input_tensor)
        return torch.nn.functional.conv2d(
            input_tensor,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=padding_value,
        )


class Conv3d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[str, int, Sequence[int]] = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive integers")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _to_ntuple(kernel_size, 3, "kernel_size", allow_zero=False)
        self.stride = _to_ntuple(stride, 3, "stride", allow_zero=False)
        self.padding = padding
        self.use_bias = bias

        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        bound = 1.0 / math.sqrt(fan_in)
        self.weight = torch.empty(
            (
                out_channels,
                in_channels,
                self.kernel_size[0],
                self.kernel_size[1],
                self.kernel_size[2],
            ),
            dtype=torch.float32,
            requires_grad=True,
        )
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

        if bias:
            self.bias = torch.empty((out_channels,), dtype=torch.float32, requires_grad=True)
            with torch.no_grad():
                self.bias.uniform_(-bound, bound)
        else:
            self.bias = None

    def _resolve_padding(self, input_tensor: Any) -> Tuple[Any, Union[int, Tuple[int, ...]]]:
        if isinstance(self.padding, str):
            padding_mode = self.padding.lower()
            if padding_mode == "valid":
                return input_tensor, 0
            if padding_mode == "same":
                padded = _pad_same_nd(input_tensor, self.kernel_size, self.stride)
                return padded, 0
            raise ValueError("padding must be 'valid', 'same', an int or a tuple of ints")
        padding_tuple = _to_ntuple(self.padding, 3, "padding", allow_zero=True)
        return input_tensor, padding_tuple

    def forward(self, input_tensor: Any) -> Any:
        if not torch.is_tensor(input_tensor):
            raise TypeError("input_tensor must be a torch tensor")

        input_tensor, padding_value = self._resolve_padding(input_tensor)
        return torch.nn.functional.conv3d(
            input_tensor,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=padding_value,
        )


class BCEWithLogitsLoss(Module):
    def __init__(self, reduction: str = "mean", pos_weight: Any = None) -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of: 'none', 'mean', 'sum'")
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, input_tensor: Any, target_tensor: Any) -> Any:
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
        return torch.nn.functional.cross_entropy(
            input_tensor, target_tensor.to(torch.long), reduction=self.reduction
        )
