"""PyTorch dataset for Cityscapes semantic segmentation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.cityseg.data.label_mapping import convert_label_ids_to_train_ids

ImageMaskTransform = Callable[..., Any]


class CityscapesDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load Cityscapes images and label ID masks as tensors."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: tuple[int, int] | None = None,
        crop_size: tuple[int, int] | None = None,
        transforms: ImageMaskTransform | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.crop_size = crop_size
        self.transforms = transforms
        self.samples = self._find_samples()

        if not self.samples:
            raise ValueError(f"No Cityscapes samples found for split '{split}' in {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.image_size is not None:
            image, mask = self._resize_pair(image, mask, self.image_size)

        if self.crop_size is not None:
            image, mask = self._center_crop_pair(image, mask, self.crop_size)

        image_array = np.asarray(image, dtype=np.uint8).copy()
        mask_array = np.asarray(mask).copy()
        mask_array = convert_label_ids_to_train_ids(mask_array)

        if self.transforms is not None:
            image_array, mask_array = self._apply_transforms(image_array, mask_array)

        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float().div(255.0)
        mask_tensor = torch.from_numpy(mask_array.astype(np.int64, copy=False)).long()
        return image_tensor, mask_tensor

    def _find_samples(self) -> list[tuple[Path, Path]]:
        image_dir = self.root / "leftImg8bit" / self.split
        mask_dir = self.root / "gtFine" / self.split
        image_paths = sorted(image_dir.glob("*/*_leftImg8bit.png"))
        samples: list[tuple[Path, Path]] = []

        for image_path in image_paths:
            city = image_path.parent.name
            mask_name = image_path.name.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
            mask_path = mask_dir / city / mask_name
            if mask_path.exists():
                samples.append((image_path, mask_path))

        return samples

    @staticmethod
    def _resize_pair(
        image: Image.Image,
        mask: Image.Image,
        size: tuple[int, int],
    ) -> tuple[Image.Image, Image.Image]:
        height, width = size
        return (
            image.resize((width, height), Image.Resampling.BILINEAR),
            mask.resize((width, height), Image.Resampling.NEAREST),
        )

    @staticmethod
    def _center_crop_pair(
        image: Image.Image,
        mask: Image.Image,
        size: tuple[int, int],
    ) -> tuple[Image.Image, Image.Image]:
        crop_height, crop_width = size
        width, height = image.size
        left = max((width - crop_width) // 2, 0)
        top = max((height - crop_height) // 2, 0)
        right = min(left + crop_width, width)
        bottom = min(top + crop_height, height)
        return image.crop((left, top, right, bottom)), mask.crop((left, top, right, bottom))

    @staticmethod
    def _apply_transforms(
        image: np.ndarray,
        mask: np.ndarray,
        transforms: ImageMaskTransform,
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            transformed = transforms(image=image, mask=mask)
        except TypeError:
            transformed = transforms(image, mask)

        if isinstance(transformed, dict):
            return transformed["image"], transformed["mask"]

        image_result, mask_result = transformed
        return image_result, mask_result
