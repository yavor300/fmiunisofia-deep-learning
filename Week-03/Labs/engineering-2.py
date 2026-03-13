from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Week-02" / "Labs"))

from dl_lib import nn  # noqa: E402


class TestSoftmax(unittest.TestCase):
    def test_softmax_inherits_module(self) -> None:
        layer = nn.Softmax(dim=1)
        self.assertIsInstance(layer, nn.Module)

    def test_softmax_forward_returns_tensor_same_shape(self) -> None:
        layer = nn.Softmax(dim=1)
        x = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
        y = layer(x)
        self.assertTrue(torch.is_tensor(y))
        self.assertEqual(tuple(y.shape), tuple(x.shape))

    def test_softmax_outputs_sum_to_one_along_dimension(self) -> None:
        layer = nn.Softmax(dim=1)
        x = torch.randn(4, 6)
        y = layer(x)
        row_sums = y.sum(dim=1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
