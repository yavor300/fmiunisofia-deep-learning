from __future__ import annotations

import unittest

import torch

from src.cityseg.constants import IGNORE_INDEX
from src.cityseg.training.losses import (
    CrossEntropyDiceLoss,
    CrossEntropyLoss,
    DiceLoss,
    FocalDiceLoss,
    FocalLoss,
)


def _logits_and_target() -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.tensor(
        [
            [
                [[2.0, 0.1], [0.1, 1.5]],
                [[0.1, 2.0], [1.8, 0.1]],
                [[0.1, 0.1], [0.1, 0.1]],
            ]
        ],
        requires_grad=True,
    )
    target = torch.tensor([[[0, 1], [1, IGNORE_INDEX]]], dtype=torch.long)
    return logits, target


class TestCrossEntropyLoss(unittest.TestCase):
    def test_when_called_with_ignored_pixels_then_finite_scalar_is_returned(self) -> None:
        logits, target = _logits_and_target()
        loss_fn = CrossEntropyLoss(ignore_index=IGNORE_INDEX)

        loss = loss_fn(logits, target)

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())

    def test_when_backward_is_called_then_logits_receive_gradients(self) -> None:
        logits, target = _logits_and_target()
        loss_fn = CrossEntropyLoss(ignore_index=IGNORE_INDEX)

        loss_fn(logits, target).backward()

        self.assertIsNotNone(logits.grad)

    def test_when_all_pixels_are_ignored_then_zero_scalar_is_returned(self) -> None:
        logits, _ = _logits_and_target()
        target = torch.full((1, 2, 2), IGNORE_INDEX, dtype=torch.long)
        loss_fn = CrossEntropyLoss(ignore_index=IGNORE_INDEX)

        loss = loss_fn(logits, target)

        self.assertEqual(loss.ndim, 0)
        self.assertEqual(loss.item(), 0.0)


class TestDiceLoss(unittest.TestCase):
    def test_when_called_with_ignored_pixels_then_finite_scalar_is_returned(self) -> None:
        logits, target = _logits_and_target()
        loss_fn = DiceLoss(mode="multiclass", ignore_index=IGNORE_INDEX)

        loss = loss_fn(logits, target)

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())

    def test_when_backward_is_called_then_logits_receive_gradients(self) -> None:
        logits, target = _logits_and_target()
        loss_fn = DiceLoss(mode="multiclass", ignore_index=IGNORE_INDEX)

        loss_fn(logits, target).backward()

        self.assertIsNotNone(logits.grad)


class TestFocalLoss(unittest.TestCase):
    def test_when_called_with_ignored_pixels_then_finite_scalar_is_returned(self) -> None:
        logits, target = _logits_and_target()
        loss_fn = FocalLoss(gamma=2.0, alpha=None, ignore_index=IGNORE_INDEX)

        loss = loss_fn(logits, target)

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())

    def test_when_backward_is_called_then_logits_receive_gradients(self) -> None:
        logits, target = _logits_and_target()
        loss_fn = FocalLoss(gamma=2.0, alpha=None, ignore_index=IGNORE_INDEX)

        loss_fn(logits, target).backward()

        self.assertIsNotNone(logits.grad)


class TestCrossEntropyDiceLoss(unittest.TestCase):
    def test_when_called_then_finite_scalar_is_returned(self) -> None:
        logits, target = _logits_and_target()
        loss_fn = CrossEntropyDiceLoss(ignore_index=IGNORE_INDEX)

        loss = loss_fn(logits, target)

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())


class TestFocalDiceLoss(unittest.TestCase):
    def test_when_called_then_finite_scalar_is_returned(self) -> None:
        logits, target = _logits_and_target()
        loss_fn = FocalDiceLoss(ignore_index=IGNORE_INDEX)

        loss = loss_fn(logits, target)

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())


if __name__ == "__main__":
    unittest.main()
