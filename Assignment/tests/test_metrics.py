from __future__ import annotations

import unittest

import torch

from src.cityseg.constants import IGNORE_INDEX
from src.cityseg.training.metrics import (
    SegmentationMetrics,
    compute_confusion_matrix,
    mean_dice,
    mean_iou,
    per_class_iou,
    pixel_accuracy,
)


class TestComputeConfusionMatrix(unittest.TestCase):
    def test_when_predictions_are_perfect_then_diagonal_counts_are_returned(self) -> None:
        prediction = torch.tensor([[[0, 1], [1, 2]]])
        target = torch.tensor([[[0, 1], [1, 2]]])

        matrix = compute_confusion_matrix(prediction, target, num_classes=3)

        expected = torch.tensor([[1, 0, 0], [0, 2, 0], [0, 0, 1]])
        self.assertTrue(torch.equal(matrix.cpu(), expected))

    def test_when_target_has_ignore_index_then_ignored_pixels_are_excluded(self) -> None:
        prediction = torch.tensor([[[0, 1], [1, 2]]])
        target = torch.tensor([[[0, IGNORE_INDEX], [1, IGNORE_INDEX]]])

        matrix = compute_confusion_matrix(prediction, target, num_classes=3)

        expected = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        self.assertTrue(torch.equal(matrix.cpu(), expected))


class TestMeanIou(unittest.TestCase):
    def test_when_prediction_is_perfect_then_mean_iou_is_one(self) -> None:
        prediction = torch.tensor([[[0, 1], [1, 2]]])
        target = torch.tensor([[[0, 1], [1, 2]]])

        score = mean_iou(prediction, target, num_classes=3)

        self.assertAlmostEqual(score.item(), 1.0)

    def test_when_prediction_is_completely_wrong_then_mean_iou_is_zero(self) -> None:
        prediction = torch.tensor([[[1, 0], [0, 1]]])
        target = torch.tensor([[[0, 1], [1, 0]]])

        score = mean_iou(prediction, target, num_classes=2)

        self.assertAlmostEqual(score.item(), 0.0)


class TestMeanDice(unittest.TestCase):
    def test_when_prediction_is_perfect_then_mean_dice_is_one(self) -> None:
        prediction = torch.tensor([[[0, 1], [1, 2]]])
        target = torch.tensor([[[0, 1], [1, 2]]])

        score = mean_dice(prediction, target, num_classes=3)

        self.assertAlmostEqual(score.item(), 1.0)

    def test_when_prediction_is_completely_wrong_then_mean_dice_is_zero(self) -> None:
        prediction = torch.tensor([[[1, 0], [0, 1]]])
        target = torch.tensor([[[0, 1], [1, 0]]])

        score = mean_dice(prediction, target, num_classes=2)

        self.assertAlmostEqual(score.item(), 0.0)


class TestPixelAccuracy(unittest.TestCase):
    def test_when_half_of_valid_pixels_match_then_accuracy_is_half(self) -> None:
        prediction = torch.tensor([[[0, 1], [0, 1]]])
        target = torch.tensor([[[0, 0], [IGNORE_INDEX, 1]]])

        score = pixel_accuracy(prediction, target)

        self.assertAlmostEqual(score.item(), 2 / 3)


class TestPerClassIou(unittest.TestCase):
    def test_when_predictions_are_mixed_then_iou_is_returned_per_class(self) -> None:
        prediction = torch.tensor([[[0, 1], [0, 1]]])
        target = torch.tensor([[[0, 0], [1, 1]]])

        scores = per_class_iou(prediction, target, num_classes=2)

        self.assertAlmostEqual(scores[0].item(), 1 / 3)
        self.assertAlmostEqual(scores[1].item(), 1 / 3)


class TestSegmentationMetrics(unittest.TestCase):
    def test_when_logits_are_provided_then_main_metrics_are_returned(self) -> None:
        logits = torch.tensor(
            [
                [
                    [[2.0, 0.1], [0.1, 2.0]],
                    [[0.1, 2.0], [2.0, 0.1]],
                ]
            ]
        )
        target = torch.tensor([[[0, 1], [1, 0]]])
        metrics = SegmentationMetrics(num_classes=2)

        result = metrics(logits, target)

        self.assertAlmostEqual(result["mean_iou"].item(), 1.0)
        self.assertAlmostEqual(result["mean_dice"].item(), 1.0)
        self.assertAlmostEqual(result["pixel_accuracy"].item(), 1.0)

    def test_when_tensors_are_on_device_then_metric_tensor_stays_on_device(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        prediction = torch.tensor([[[0, 1], [1, 0]]], device=device)
        target = torch.tensor([[[0, 1], [1, 0]]], device=device)

        score = mean_iou(prediction, target, num_classes=2)

        self.assertEqual(score.device, device)


if __name__ == "__main__":
    unittest.main()
