from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from typing import Sequence, Tuple

import torch


LABS_ROOT = Path(__file__).resolve().parents[1]
if str(LABS_ROOT) not in sys.path:
    sys.path.insert(0, str(LABS_ROOT))

from dl_lib.nn import Conv1d, Conv2d, Conv3d, Module


def same_pad_pairs(input_sizes: Sequence[int], kernel_sizes: Sequence[int], strides: Sequence[int]) -> list[Tuple[int, int]]:
    pairs: list[Tuple[int, int]] = []
    for size, kernel, stride in zip(input_sizes, kernel_sizes, strides):
        out = math.ceil(size / stride)
        needed = max((out - 1) * stride + kernel - size, 0)
        left = needed // 2
        right = needed - left
        pairs.append((left, right))
    return pairs


def apply_same_padding(input_tensor: torch.Tensor, kernel_sizes: Sequence[int], strides: Sequence[int]) -> torch.Tensor:
    pairs = same_pad_pairs(input_tensor.shape[-len(kernel_sizes) :], kernel_sizes, strides)
    pads: list[int] = []
    for left, right in reversed(pairs):
        pads.extend([left, right])
    return torch.nn.functional.pad(input_tensor, tuple(pads))


class TestConvolutionLayers(unittest.TestCase):
    def test_conv_layers_inherit_module(self) -> None:
        self.assertIsInstance(Conv1d(1, 1, 3), Module)
        self.assertIsInstance(Conv2d(1, 1, 3), Module)
        self.assertIsInstance(Conv3d(1, 1, 3), Module)

    def test_conv1d_matches_torch_for_valid_padding(self) -> None:
        layer = Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding="valid", bias=True)
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=torch.float32)

        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[[1.0, 0.0, -1.0]]], dtype=torch.float32))
            layer.bias.copy_(torch.tensor([0.5], dtype=torch.float32))

        output = layer(x)
        expected = torch.nn.functional.conv1d(x, layer.weight, layer.bias, stride=1, padding=0)
        self.assertTrue(torch.is_tensor(output))
        self.assertTrue(torch.allclose(output, expected))

    def test_conv1d_same_padding_shape(self) -> None:
        layer = Conv1d(in_channels=2, out_channels=3, kernel_size=3, stride=2, padding="same", bias=False)
        x = torch.randn(4, 2, 10)
        output = layer(x)
        self.assertEqual(output.shape, (4, 3, math.ceil(10 / 2)))

    def test_conv2d_matches_torch_for_numeric_padding(self) -> None:
        layer = Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        x = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5)

        with torch.no_grad():
            layer.weight.fill_(0.1)
            layer.bias.fill_(0.2)

        output = layer(x)
        expected = torch.nn.functional.conv2d(x, layer.weight, layer.bias, stride=(1, 1), padding=(1, 1))
        self.assertTrue(torch.is_tensor(output))
        self.assertTrue(torch.allclose(output, expected))

    def test_conv2d_same_padding_matches_manual_pad(self) -> None:
        layer = Conv2d(in_channels=1, out_channels=2, kernel_size=(3, 5), stride=(2, 2), padding="same", bias=False)
        x = torch.randn(2, 1, 11, 13)
        output = layer(x)

        padded = apply_same_padding(x, kernel_sizes=(3, 5), strides=(2, 2))
        expected = torch.nn.functional.conv2d(
            padded,
            layer.weight,
            None,
            stride=(2, 2),
            padding=0,
        )
        self.assertTrue(torch.allclose(output, expected))
        self.assertEqual(output.shape, (2, 2, math.ceil(11 / 2), math.ceil(13 / 2)))

    def test_conv3d_matches_torch_for_numeric_padding(self) -> None:
        layer = Conv3d(
            in_channels=1,
            out_channels=1,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=True,
        )
        x = torch.randn(1, 1, 4, 5, 6)

        with torch.no_grad():
            layer.weight.fill_(0.05)
            layer.bias.fill_(0.1)

        output = layer(x)
        expected = torch.nn.functional.conv3d(
            x,
            layer.weight,
            layer.bias,
            stride=(1, 1, 1),
            padding=(1, 1, 1),
        )
        self.assertTrue(torch.is_tensor(output))
        self.assertTrue(torch.allclose(output, expected))


if __name__ == "__main__":
    unittest.main()
