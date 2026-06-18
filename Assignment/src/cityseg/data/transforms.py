"""Albumentations preprocessing and augmentation pipelines."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import numpy as np
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CITYSCAPES_MEAN = (0.28689554, 0.32513303, 0.28389177)
CITYSCAPES_STD = (0.18696375, 0.19017339, 0.18720214)


def build_transforms(
    strategy: str,
    image_size: tuple[int, int],
    crop_size: tuple[int, int] | None = None,
    normalization: str | None = None,
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
) -> A.Compose:
    """Build an Albumentations pipeline for image/mask segmentation pairs."""
    strategy = _normalize_name(strategy)
    transforms: list[Any] = []
    resize_height, resize_width = image_size

    if strategy in {"resize_only", "imagenet_normalization", "cityscapes_normalization"}:
        transforms.append(_resize(resize_height, resize_width))
    elif strategy == "random_crop":
        transforms.extend(
            [
                _resize(resize_height, resize_width),
                _random_crop(crop_size, image_size),
            ]
        )
    elif strategy == "basic_aug":
        transforms.extend(
            [
                _resize(resize_height, resize_width),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                _random_crop(crop_size, image_size),
            ]
        )
    elif strategy == "strong_aug":
        transforms.extend(
            [
                _resize(resize_height, resize_width),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.25),
                A.Affine(
                    scale=(0.85, 1.15),
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    p=0.4,
                ),
                _random_crop(crop_size, image_size),
            ]
        )
    else:
        raise ValueError(f"Unsupported preprocessing strategy: {strategy}")

    normalization = _normalization_from_strategy(strategy, normalization)
    if normalization not in {None, "none"}:
        norm_mean, norm_std = _normalization_values(normalization, mean, std)
        transforms.append(A.Normalize(mean=norm_mean, std=norm_std))

    return A.Compose(transforms)


def create_transforms_from_config(
    config: dict[str, Any],
    split: str,
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
) -> A.Compose:
    preprocessing = config.get("preprocessing", {})
    image_size = (
        int(preprocessing.get("resize_height", 512)),
        int(preprocessing.get("resize_width", 1024)),
    )
    crop_size = _optional_size(preprocessing.get("crop_height"), preprocessing.get("crop_width"))
    normalization = preprocessing.get("normalize", "none")
    if split == "train":
        strategy = preprocessing.get("augmentations", "resize_only")
    else:
        strategy = preprocessing.get("eval_augmentations", "resize_only")

    return build_transforms(
        strategy=strategy,
        image_size=image_size,
        crop_size=crop_size,
        normalization=normalization,
        mean=mean,
        std=std,
    )


def compute_image_mean_std(
    root: str | Path,
    split: str = "train",
    max_samples: int | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Compute channel mean/std from Cityscapes images in `[0, 1]` scale."""
    image_paths = sorted((Path(root) / "leftImg8bit" / split).glob("*/*_leftImg8bit.png"))
    if max_samples is not None:
        image_paths = image_paths[:max_samples]
    if not image_paths:
        raise ValueError(f"No images found for mean/std computation in split '{split}'")

    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_square_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    for image_path in image_paths:
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        pixels = image.reshape(-1, 3)
        pixel_sum += pixels.sum(axis=0)
        pixel_square_sum += np.square(pixels).sum(axis=0)
        pixel_count += pixels.shape[0]

    mean = pixel_sum / pixel_count
    variance = pixel_square_sum / pixel_count - np.square(mean)
    std = np.sqrt(np.maximum(variance, 1e-12))
    return tuple(mean.tolist()), tuple(std.tolist())


def _resize(height: int, width: int) -> A.Resize:
    return A.Resize(
        height=height,
        width=width,
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
    )


def _random_crop(
    crop_size: tuple[int, int] | None,
    fallback_size: tuple[int, int],
) -> A.RandomCrop:
    crop_height, crop_width = crop_size or fallback_size
    return A.RandomCrop(height=crop_height, width=crop_width)


def _optional_size(height: Any, width: Any) -> tuple[int, int] | None:
    if height is None or width is None:
        return None
    return int(height), int(width)


def _normalize_name(name: str | None) -> str:
    normalized = str(name or "resize_only").lower()
    aliases = {
        "basic": "basic_aug",
        "strong": "strong_aug",
        "imagenet": "imagenet_normalization",
        "cityscapes": "cityscapes_normalization",
    }
    return aliases.get(normalized, normalized)


def _normalization_from_strategy(strategy: str, normalization: str | None) -> str | None:
    if strategy == "imagenet_normalization":
        return "imagenet"
    if strategy == "cityscapes_normalization":
        return "cityscapes"
    if normalization is None:
        return None
    return _normalize_normalization_name(normalization)


def _normalize_normalization_name(name: str | None) -> str:
    normalized = str(name or "none").lower()
    aliases = {
        "imagenet_normalization": "imagenet",
        "cityscapes_normalization": "cityscapes",
    }
    return aliases.get(normalized, normalized)


def _normalization_values(
    normalization: str,
    mean: tuple[float, float, float] | None,
    std: tuple[float, float, float] | None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if normalization == "imagenet":
        return IMAGENET_MEAN, IMAGENET_STD
    if normalization == "cityscapes":
        return mean or CITYSCAPES_MEAN, std or CITYSCAPES_STD
    raise ValueError(f"Unsupported normalization: {normalization}")
