from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Week-02" / "Labs"))

from dl_lib import nn  # noqa: E402


class TestBCEWithLogitsLoss(unittest.TestCase):
    def setUp(self) -> None:
        self.input_tensor = torch.tensor([0.2, -1.1, 3.0, 0.7], dtype=torch.float32)
        self.target_tensor = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)

    def test_loss_inherits_module(self) -> None:
        criterion = nn.BCEWithLogitsLoss()
        self.assertIsInstance(criterion, nn.Module)

    def test_reduction_none_matches_torch(self) -> None:
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        result = criterion(self.input_tensor, self.target_tensor)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            self.input_tensor, self.target_tensor, reduction="none"
        )
        self.assertTrue(torch.allclose(result, expected, atol=1e-7))

    def test_reduction_mean_matches_torch(self) -> None:
        criterion = nn.BCEWithLogitsLoss(reduction="mean")
        result = criterion(self.input_tensor, self.target_tensor)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            self.input_tensor, self.target_tensor, reduction="mean"
        )
        self.assertTrue(torch.allclose(result, expected, atol=1e-7))

    def test_reduction_sum_with_pos_weight_matches_torch(self) -> None:
        pos_weight = torch.tensor([2.0], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(reduction="sum", pos_weight=pos_weight)
        result = criterion(self.input_tensor, self.target_tensor)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            self.input_tensor, self.target_tensor, reduction="sum", pos_weight=pos_weight
        )
        self.assertTrue(torch.allclose(result, expected, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
