from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


LABS_ROOT = Path(__file__).resolve().parents[1]
if str(LABS_ROOT) not in sys.path:
    sys.path.insert(0, str(LABS_ROOT))

from dl_lib.nn import AvgPool2d, MaxPool2d, Module


class TestPoolingLayers(unittest.TestCase):
    def test_maxpool2d_inherits_module(self) -> None:
        layer = MaxPool2d(kernel_size=2)
        self.assertIsInstance(layer, Module)

    def test_avgpool2d_inherits_module(self) -> None:
        layer = AvgPool2d(kernel_size=2)
        self.assertIsInstance(layer, Module)

    def test_maxpool2d_without_padding_matches_torch(self) -> None:
        layer = MaxPool2d(kernel_size=2, stride=2, padding=0)
        x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        output = layer(x)
        expected = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        self.assertTrue(torch.is_tensor(output))
        self.assertTrue(torch.allclose(output, expected))

    def test_maxpool2d_with_padding_uses_negative_infinity(self) -> None:
        layer = MaxPool2d(kernel_size=(2, 2), stride=(1, 1), padding=(1, 1))
        x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        output = layer(x)

        padded = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="constant", value=float("-inf"))
        expected = torch.nn.functional.max_pool2d(padded, kernel_size=2, stride=1, padding=0)
        self.assertTrue(torch.is_tensor(output))
        self.assertTrue(torch.allclose(output, expected))

    def test_avgpool2d_with_padding_uses_zeros(self) -> None:
        layer = AvgPool2d(kernel_size=2, stride=1, padding=1)
        x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        output = layer(x)

        padded = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="constant", value=0.0)
        expected = torch.nn.functional.avg_pool2d(padded, kernel_size=2, stride=1, padding=0)
        self.assertTrue(torch.is_tensor(output))
        self.assertTrue(torch.allclose(output, expected))


if __name__ == "__main__":
    unittest.main()
