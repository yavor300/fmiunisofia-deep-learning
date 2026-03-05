from __future__ import annotations

from numbers import Real
from typing import Iterable, List, Sequence, Union

import numpy as np

from value import Value


InputValue = Union[Real, Value]


class Neuron:
    def __init__(self, in_features: int) -> None:
        self.w = [Value(np.random.uniform(-1, 1)) for _ in range(in_features)]
        self.b = Value(np.random.uniform(-1, 1))

    def __call__(self, x: Sequence[InputValue]) -> Value:
        if len(x) != len(self.w):
            raise ValueError(f"expected {len(self.w)} inputs but got {len(x)}")

        activation = self.b
        for wi, xi in zip(self.w, x):
            activation = activation + wi * xi

        return activation.tanh()

    def parameters(self) -> List[Value]:
        return self.w + [self.b]


class Linear:
    def __init__(self, in_features: int, out_features: int) -> None:
        self.neurons = [Neuron(in_features) for _ in range(out_features)]

    def __call__(self, x: Union[Sequence[InputValue], Value]) -> Union[List[Value], Value]:
        if isinstance(x, Value):
            x = [x]

        outputs = [neuron(x) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self) -> List[Value]:
        params: List[Value] = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params


Layer = Linear


class MLP:
    def __init__(self, in_channels: int, hidden_channels: Sequence[int]) -> None:
        # keep each layer wired with in_channels
        self.layers = [
            Linear(in_features=in_channels, out_features=out_channels)
            for out_channels in hidden_channels
        ]

    def __call__(self, x: Union[Sequence[InputValue], Value]) -> Union[List[Value], Value]:
        output: Union[Sequence[InputValue], Value] = x
        for layer in self.layers:
            if not isinstance(output, Value):
                # if a previous layer produced more features
                # than this layer expects, use the first expected features only
                in_features = len(layer.neurons[0].w)
                output = list(output)[:in_features]
            output = layer(output)
        return output

    def parameters(self) -> List[Value]:
        params: List[Value] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


def zero_grad(parameters: Iterable[Value]) -> None:
    for parameter in parameters:
        parameter.grad = 0.0
