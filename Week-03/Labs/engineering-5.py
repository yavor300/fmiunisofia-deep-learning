from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Week-02" / "Labs"))

from dl_lib import nn  # noqa: E402


class TestCrossEntropyLoss(unittest.TestCase):
    def setUp(self) -> None:
        self.logits = torch.tensor(
            [[0.3, 1.2, -0.7], [2.1, -0.5, 0.2], [1.0, 0.5, 0.3]], dtype=torch.float32
        )
        self.targets = torch.tensor([1, 0, 2], dtype=torch.long)

    def test_loss_inherits_module(self) -> None:
        criterion = nn.CrossEntropyLoss()
        self.assertIsInstance(criterion, nn.Module)

    def test_reduction_none_matches_torch(self) -> None:
        criterion = nn.CrossEntropyLoss(reduction="none")
        result = criterion(self.logits, self.targets)
        expected = torch.nn.functional.cross_entropy(self.logits, self.targets, reduction="none")
        self.assertTrue(torch.allclose(result, expected, atol=1e-7))

    def test_reduction_mean_matches_torch(self) -> None:
        criterion = nn.CrossEntropyLoss(reduction="mean")
        result = criterion(self.logits, self.targets)
        expected = torch.nn.functional.cross_entropy(self.logits, self.targets, reduction="mean")
        self.assertTrue(torch.allclose(result, expected, atol=1e-7))

    def test_reduction_sum_matches_torch(self) -> None:
        criterion = nn.CrossEntropyLoss(reduction="sum")
        result = criterion(self.logits, self.targets)
        expected = torch.nn.functional.cross_entropy(self.logits, self.targets, reduction="sum")
        self.assertTrue(torch.allclose(result, expected, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
