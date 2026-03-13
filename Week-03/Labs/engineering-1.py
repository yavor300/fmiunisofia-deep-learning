from pathlib import Path
import math
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Week-02" / "Labs"))

from dl_lib import nn  # noqa: E402


class TestLinear(unittest.TestCase):
    def test_linear_inherits_module(self) -> None:
        layer = nn.Linear(in_features=3, out_features=2)
        self.assertIsInstance(layer, nn.Module)

    def test_linear_forward_accepts_and_returns_tensor(self) -> None:
        layer = nn.Linear(in_features=4, out_features=3)
        x = torch.randn(5, 4)
        y = layer(x)
        self.assertTrue(torch.is_tensor(y))
        self.assertEqual(tuple(y.shape), (5, 3))

    def test_linear_uniform_initialization_range(self) -> None:
        in_features = 6
        layer = nn.Linear(in_features=in_features, out_features=2)
        bound = math.sqrt(1.0 / in_features)
        self.assertLessEqual(torch.max(layer.weight).item(), bound + 1e-6)
        self.assertGreaterEqual(torch.min(layer.weight).item(), -bound - 1e-6)
        self.assertLessEqual(torch.max(layer.bias).item(), bound + 1e-6)
        self.assertGreaterEqual(torch.min(layer.bias).item(), -bound - 1e-6)

    def test_linear_without_bias(self) -> None:
        layer = nn.Linear(in_features=3, out_features=1, bias=False)
        self.assertIsNone(layer.bias)
        self.assertEqual(len(layer.parameters()), 1)


if __name__ == "__main__":
    unittest.main()
