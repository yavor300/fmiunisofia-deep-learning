"""Dataloader factories for Cityscapes training and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from src.cityseg.data.cityscapes_dataset import CityscapesDataset


def create_dataloader(
    root: str | Path,
    split: str,
    batch_size: int,
    num_workers: int,
    image_size: tuple[int, int] | None = None,
    crop_size: tuple[int, int] | None = None,
    pin_memory: bool = False,
    shuffle: bool = False,
    transforms: Any | None = None,
) -> DataLoader[tuple[Any, Any]]:
    dataset = CityscapesDataset(
        root=root,
        split=split,
        image_size=image_size,
        crop_size=crop_size,
        transforms=transforms,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def create_train_val_dataloaders(config: dict[str, Any]) -> tuple[DataLoader[Any], DataLoader[Any]]:
    paths = config.get("paths", {})
    training = config.get("training", {})
    preprocessing = config.get("preprocessing", {})
    root = paths.get("data_root", "data/raw/cityscapes")
    image_size = _optional_size(
        preprocessing.get("resize_height"),
        preprocessing.get("resize_width"),
    )
    crop_size = _optional_size(
        preprocessing.get("crop_height"),
        preprocessing.get("crop_width"),
    )
    batch_size = int(training.get("batch_size", 4))
    num_workers = int(training.get("num_workers", 0))
    pin_memory = bool(training.get("pin_memory", False))

    train_loader = create_dataloader(
        root=root,
        split="train",
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        crop_size=crop_size,
        pin_memory=pin_memory,
        shuffle=bool(training.get("shuffle", True)),
    )
    val_loader = create_dataloader(
        root=root,
        split="val",
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        crop_size=crop_size,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return train_loader, val_loader


def _optional_size(height: Any, width: Any) -> tuple[int, int] | None:
    if height is None or width is None:
        return None
    return int(height), int(width)
