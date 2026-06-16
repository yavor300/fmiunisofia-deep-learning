"""Stable segmentation metrics for Cityscapes experiments."""

from __future__ import annotations

import torch
from torch import nn


def prepare_predictions(prediction: torch.Tensor) -> torch.Tensor:
    """Convert logits/probabilities to class IDs when needed."""
    if prediction.ndim == 4:
        return prediction.argmax(dim=1)
    return prediction.long()


def compute_confusion_matrix(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Return a `[num_classes, num_classes]` matrix with rows=target and cols=prediction."""
    prediction = prepare_predictions(prediction)
    target = target.long()
    valid_mask = (target != ignore_index) & (target >= 0) & (target < num_classes)
    valid_target = target[valid_mask]
    valid_prediction = prediction[valid_mask]

    if valid_target.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.long, device=target.device)

    valid_prediction = valid_prediction.clamp(min=0, max=num_classes - 1)
    encoded = valid_target * num_classes + valid_prediction
    matrix = torch.bincount(encoded, minlength=num_classes * num_classes)
    return matrix.reshape(num_classes, num_classes)


def per_class_iou(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    matrix = compute_confusion_matrix(prediction, target, num_classes, ignore_index)
    matrix = matrix.to(dtype=torch.float32)
    intersection = torch.diag(matrix)
    target_count = matrix.sum(dim=1)
    prediction_count = matrix.sum(dim=0)
    union = target_count + prediction_count - intersection
    return torch.where(union > 0, intersection / union.clamp_min(1.0), torch.nan)


def mean_iou(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    scores = per_class_iou(prediction, target, num_classes, ignore_index)
    return _nanmean_or_zero(scores)


def per_class_dice(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    matrix = compute_confusion_matrix(prediction, target, num_classes, ignore_index)
    matrix = matrix.to(dtype=torch.float32)
    intersection = torch.diag(matrix)
    denominator = matrix.sum(dim=1) + matrix.sum(dim=0)
    return torch.where(denominator > 0, 2.0 * intersection / denominator.clamp_min(1.0), torch.nan)


def mean_dice(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    scores = per_class_dice(prediction, target, num_classes, ignore_index)
    return _nanmean_or_zero(scores)


def pixel_accuracy(
    prediction: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
) -> torch.Tensor:
    prediction = prepare_predictions(prediction)
    target = target.long()
    valid_mask = target != ignore_index
    valid_count = valid_mask.sum()
    if valid_count == 0:
        return torch.zeros((), dtype=torch.float32, device=target.device)
    correct_count = (prediction[valid_mask] == target[valid_mask]).sum()
    return correct_count.to(dtype=torch.float32) / valid_count.to(dtype=torch.float32)


class SegmentationMetrics(nn.Module):
    """Callable metric bundle returning the three report metrics plus per-class IoU."""

    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        class_ids = prepare_predictions(prediction)
        return {
            "mean_iou": mean_iou(class_ids, target, self.num_classes, self.ignore_index),
            "mean_dice": mean_dice(class_ids, target, self.num_classes, self.ignore_index),
            "pixel_accuracy": pixel_accuracy(class_ids, target, self.ignore_index),
            "per_class_iou": per_class_iou(class_ids, target, self.num_classes, self.ignore_index),
            "confusion_matrix": compute_confusion_matrix(
                class_ids,
                target,
                self.num_classes,
                self.ignore_index,
            ),
        }


def _nanmean_or_zero(values: torch.Tensor) -> torch.Tensor:
    valid_values = values[~torch.isnan(values)]
    if valid_values.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=values.device)
    return valid_values.mean()
