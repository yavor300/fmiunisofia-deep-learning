from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Week-02" / "Labs"))

from dl_lib import nn, optim  # noqa: E402


def optimize_scalar(optimizer: optim.Optimizer, parameter: torch.Tensor, steps: int = 300) -> float:
    target = torch.tensor([3.0], dtype=torch.float32)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = ((parameter - target) ** 2).mean()
        loss.backward()
        optimizer.step()
    return parameter.item()


class TestOptimizers(unittest.TestCase):
    def test_optimizer_classes_inherit_module(self) -> None:
        parameter = torch.tensor([0.0], requires_grad=True)
        self.assertIsInstance(optim.SGD([parameter]), nn.Module)
        self.assertIsInstance(optim.AdaGrad([parameter]), nn.Module)
        self.assertIsInstance(optim.RMSprop([parameter]), nn.Module)
        self.assertIsInstance(optim.Adam([parameter]), nn.Module)
        self.assertIsInstance(optim.AdamW([parameter]), nn.Module)

    def test_sgd_with_momentum_optimizes_scalar(self) -> None:
        parameter = torch.tensor([0.0], requires_grad=True)
        optimizer = optim.SGD([parameter], lr=0.1, momentum=0.9)
        value = optimize_scalar(optimizer, parameter, steps=250)
        self.assertLess(abs(value - 3.0), 0.2)

    def test_adagrad_optimizes_scalar(self) -> None:
        parameter = torch.tensor([0.0], requires_grad=True)
        optimizer = optim.AdaGrad([parameter], lr=0.8)
        value = optimize_scalar(optimizer, parameter, steps=300)
        self.assertLess(abs(value - 3.0), 0.25)

    def test_rmsprop_optimizes_scalar(self) -> None:
        parameter = torch.tensor([0.0], requires_grad=True)
        optimizer = optim.RMSprop([parameter], lr=0.1)
        value = optimize_scalar(optimizer, parameter, steps=300)
        self.assertLess(abs(value - 3.0), 0.2)

    def test_adam_optimizes_scalar(self) -> None:
        parameter = torch.tensor([0.0], requires_grad=True)
        optimizer = optim.Adam([parameter], lr=0.1)
        value = optimize_scalar(optimizer, parameter, steps=200)
        self.assertLess(abs(value - 3.0), 0.1)

    def test_adamw_optimizes_scalar(self) -> None:
        parameter = torch.tensor([0.0], requires_grad=True)
        optimizer = optim.AdamW([parameter], lr=0.1, weight_decay=0.01)
        value = optimize_scalar(optimizer, parameter, steps=250)
        self.assertLess(abs(value - 3.0), 0.25)


if __name__ == "__main__":
    unittest.main()
