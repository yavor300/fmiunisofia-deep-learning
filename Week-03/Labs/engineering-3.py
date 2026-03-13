from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Week-02" / "Labs"))

from dl_lib import nn  # noqa: E402


class TestDropout(unittest.TestCase):
    def test_dropout_inherits_module(self) -> None:
        layer = nn.Dropout(p=0.2)
        self.assertIsInstance(layer, nn.Module)

    def test_dropout_train_mode_randomly_zeroes_values(self) -> None:
        torch.manual_seed(42)
        layer = nn.Dropout(p=0.8)
        layer.train()
        x = torch.ones(1000)
        y = layer(x)
        self.assertTrue(torch.is_tensor(y))
        self.assertGreater((y == 0).sum().item(), 0)
        self.assertGreater((y != 0).sum().item(), 0)

    def test_dropout_eval_mode_passes_input_unchanged(self) -> None:
        layer = nn.Dropout(p=0.9)
        layer.eval()
        x = torch.randn(10, 5)
        y = layer(x)
        self.assertTrue(torch.allclose(x, y))


if __name__ == "__main__":
    unittest.main()
