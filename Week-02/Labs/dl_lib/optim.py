from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from .nn import Module, _get_torch


class Optimizer(Module):
    def __init__(self, params: Iterable[Any], lr: float = 1e-3) -> None:
        super().__init__()
        self.params = list(params)
        self.lr = lr

    def forward(self, input_tensor: Any = None) -> Any:
        self.step()
        return input_tensor

    def step(self) -> None:
        raise NotImplementedError

    def zero_grad(self) -> None:
        for parameter in self.params:
            if parameter.grad is not None:
                parameter.grad.zero_()

    def parameters(self) -> List[Any]:
        return self.params


class SGD(Optimizer):
    def __init__(self, params: Iterable[Any], lr: float = 1e-3, momentum: float = 0.0) -> None:
        super().__init__(params=params, lr=lr)
        self.momentum = momentum
        self._velocity: Dict[int, Any] = {}

    def step(self) -> None:
        torch = _get_torch()
        for parameter in self.params:
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach()

            if self.momentum != 0.0:
                velocity = self._velocity.get(id(parameter))
                if velocity is None:
                    velocity = torch.zeros_like(parameter)
                velocity = self.momentum * velocity + grad
                self._velocity[id(parameter)] = velocity
                update = velocity
            else:
                update = grad

            with torch.no_grad():
                parameter.add_(update, alpha=-self.lr)


class AdaGrad(Optimizer):
    def __init__(self, params: Iterable[Any], lr: float = 1e-2, eps: float = 1e-10) -> None:
        super().__init__(params=params, lr=lr)
        self.eps = eps
        self._accumulator: Dict[int, Any] = {}

    def step(self) -> None:
        torch = _get_torch()
        for parameter in self.params:
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach()
            accumulator = self._accumulator.get(id(parameter))
            if accumulator is None:
                accumulator = torch.zeros_like(parameter)
            accumulator = accumulator + grad.pow(2)
            self._accumulator[id(parameter)] = accumulator

            update = grad / (accumulator.sqrt() + self.eps)
            with torch.no_grad():
                parameter.add_(update, alpha=-self.lr)


class RMSprop(Optimizer):
    def __init__(
        self, params: Iterable[Any], lr: float = 1e-2, alpha: float = 0.99, eps: float = 1e-8
    ) -> None:
        super().__init__(params=params, lr=lr)
        self.alpha = alpha
        self.eps = eps
        self._square_avg: Dict[int, Any] = {}

    def step(self) -> None:
        torch = _get_torch()
        for parameter in self.params:
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach()
            square_avg = self._square_avg.get(id(parameter))
            if square_avg is None:
                square_avg = torch.zeros_like(parameter)
            square_avg = self.alpha * square_avg + (1.0 - self.alpha) * grad.pow(2)
            self._square_avg[id(parameter)] = square_avg

            update = grad / (square_avg.sqrt() + self.eps)
            with torch.no_grad():
                parameter.add_(update, alpha=-self.lr)


class Adam(Optimizer):
    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        super().__init__(params=params, lr=lr)
        self.betas = betas
        self.eps = eps
        self._step_count = 0
        self._m: Dict[int, Any] = {}
        self._v: Dict[int, Any] = {}

    def step(self) -> None:
        torch = _get_torch()
        beta1, beta2 = self.betas
        self._step_count += 1

        for parameter in self.params:
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach()
            m = self._m.get(id(parameter))
            v = self._v.get(id(parameter))
            if m is None:
                m = torch.zeros_like(parameter)
            if v is None:
                v = torch.zeros_like(parameter)

            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * grad.pow(2)
            self._m[id(parameter)] = m
            self._v[id(parameter)] = v

            m_hat = m / (1.0 - beta1**self._step_count)
            v_hat = v / (1.0 - beta2**self._step_count)
            update = m_hat / (v_hat.sqrt() + self.eps)

            with torch.no_grad():
                parameter.add_(update, alpha=-self.lr)


class AdamW(Adam):
    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        super().__init__(params=params, lr=lr, betas=betas, eps=eps)
        self.weight_decay = weight_decay

    def step(self) -> None:
        torch = _get_torch()
        for parameter in self.params:
            if parameter.grad is not None:
                with torch.no_grad():
                    parameter.mul_(1.0 - self.lr * self.weight_decay)
        super().step()
