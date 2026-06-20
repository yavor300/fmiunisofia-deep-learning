"""Loss functions for semantic segmentation training."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class CrossEntropyLoss(nn.Module):
    """Pixel-wise cross-entropy loss for `[B, C, H, W]` logits."""

    def __init__(self, ignore_index: int = 255) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not torch.any(target != self.ignore_index):
            return logits.sum() * 0.0
        return F.cross_entropy(logits, target, ignore_index=self.ignore_index)


class DiceLoss(nn.Module):
    """Multiclass Dice loss operating on softmax probabilities."""

    def __init__(
        self,
        mode: str = "multiclass",
        ignore_index: int = 255,
        smooth: float = 1.0,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        if mode != "multiclass":
            raise ValueError("Only mode='multiclass' is supported.")
        self.mode = mode
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        valid_mask = target != self.ignore_index
        if not torch.any(valid_mask):
            return logits.sum() * 0.0

        probabilities = torch.softmax(logits, dim=1)
        safe_target = target.masked_fill(~valid_mask, 0)
        target_one_hot = F.one_hot(safe_target, num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).to(dtype=probabilities.dtype)
        valid_mask = valid_mask.unsqueeze(1).to(dtype=probabilities.dtype)

        probabilities = probabilities * valid_mask
        target_one_hot = target_one_hot * valid_mask

        reduce_dims = (0, 2, 3)
        intersection = torch.sum(probabilities * target_one_hot, dim=reduce_dims)
        cardinality = torch.sum(probabilities + target_one_hot, dim=reduce_dims)
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth + self.eps)
        valid_classes = cardinality > 0
        if not torch.any(valid_classes):
            return logits.sum() * 0.0
        return 1.0 - dice_score[valid_classes].mean()


class FocalLoss(nn.Module):
    """Multiclass focal loss for dense segmentation."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | Sequence[float] | torch.Tensor | None = None,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid_mask = target != self.ignore_index
        if not torch.any(valid_mask):
            return logits.sum() * 0.0

        ce_loss = F.cross_entropy(
            logits,
            target,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        ce_loss = ce_loss[valid_mask]
        pt = torch.exp(-ce_loss)
        focal_loss = (1.0 - pt).pow(self.gamma) * ce_loss
        alpha_factor = self._alpha_factor(target[valid_mask], logits.device, logits.dtype)
        if alpha_factor is not None:
            focal_loss = focal_loss * alpha_factor
        return focal_loss.mean()

    def _alpha_factor(
        self,
        target: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.alpha is None:
            return None
        if isinstance(self.alpha, int | float):
            return torch.as_tensor(self.alpha, device=device, dtype=dtype)

        alpha = torch.as_tensor(self.alpha, device=device, dtype=dtype)
        return alpha[target]


class LovaszSoftmaxLoss(nn.Module):
    """Multiclass Lovasz-Softmax loss as a differentiable IoU surrogate."""

    def __init__(self, ignore_index: int = 255) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid_mask = target != self.ignore_index
        if not torch.any(valid_mask):
            return logits.sum() * 0.0

        probabilities = torch.softmax(logits, dim=1).permute(0, 2, 3, 1)
        probabilities = probabilities[valid_mask]
        target = target[valid_mask]
        losses = []
        for class_id in range(logits.shape[1]):
            foreground = (target == class_id).to(dtype=probabilities.dtype)
            if not torch.any(foreground):
                continue
            class_errors = (foreground - probabilities[:, class_id]).abs()
            sorted_errors, permutation = torch.sort(class_errors, descending=True)
            sorted_foreground = foreground[permutation]
            losses.append(torch.dot(sorted_errors, _lovasz_gradient(sorted_foreground)))
        if not losses:
            return logits.sum() * 0.0
        return torch.stack(losses).mean()


def _lovasz_gradient(sorted_foreground: torch.Tensor) -> torch.Tensor:
    foreground_sum = sorted_foreground.sum()
    intersection = foreground_sum - sorted_foreground.cumsum(dim=0)
    union = foreground_sum + (1.0 - sorted_foreground).cumsum(dim=0)
    jaccard = 1.0 - intersection / union.clamp_min(1e-7)
    if sorted_foreground.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


class CrossEntropyDiceLoss(nn.Module):
    """Weighted sum of cross-entropy and Dice loss."""

    def __init__(
        self,
        ignore_index: int = 255,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.cross_entropy = CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (
            self.ce_weight * self.cross_entropy(logits, target)
            + self.dice_weight * self.dice(logits, target)
        )


class CrossEntropyLovaszLoss(nn.Module):
    """Weighted sum of cross-entropy and Lovasz-Softmax loss."""

    def __init__(
        self,
        ignore_index: int = 255,
        ce_weight: float = 1.0,
        lovasz_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.lovasz_weight = lovasz_weight
        self.cross_entropy = CrossEntropyLoss(ignore_index=ignore_index)
        self.lovasz = LovaszSoftmaxLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (
            self.ce_weight * self.cross_entropy(logits, target)
            + self.lovasz_weight * self.lovasz(logits, target)
        )


class FocalDiceLoss(nn.Module):
    """Weighted sum of focal and Dice loss."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | Sequence[float] | torch.Tensor | None = None,
        ignore_index: int = 255,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal = FocalLoss(gamma=gamma, alpha=alpha, ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (
            self.focal_weight * self.focal(logits, target)
            + self.dice_weight * self.dice(logits, target)
        )
