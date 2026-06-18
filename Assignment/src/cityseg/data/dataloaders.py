"""Dataloader factories for Cityscapes training and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from src.cityseg.data.cityscapes_dataset import CityscapesDataset
from src.cityseg.data.transforms import compute_image_mean_std, create_transforms_from_config


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
    batch_size = int(training.get("batch_size", 4))
    num_workers = int(training.get("num_workers", 0))
    pin_memory = bool(training.get("pin_memory", False))
    mean, std = _cityscapes_stats_if_needed(root, preprocessing)
    train_transforms = create_transforms_from_config(config, split="train", mean=mean, std=std)
    val_transforms = create_transforms_from_config(config, split="val", mean=mean, std=std)

    train_loader = create_dataloader(
        root=root,
        split="train",
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=bool(training.get("shuffle", True)),
        transforms=train_transforms,
    )
    val_loader = create_dataloader(
        root=root,
        split="val",
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        transforms=val_transforms,
    )
    return train_loader, val_loader


def _optional_size(height: Any, width: Any) -> tuple[int, int] | None:
    if height is None or width is None:
        return None
    return int(height), int(width)


def _cityscapes_stats_if_needed(
    root: str | Path,
    preprocessing: dict[str, Any],
) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
    if str(preprocessing.get("normalize", "none")).lower() != "cityscapes":
        return None, None
    if "mean" in preprocessing and "std" in preprocessing:
        return tuple(preprocessing["mean"]), tuple(preprocessing["std"])
    max_samples = preprocessing.get("normalization_max_samples")
    max_samples = int(max_samples) if max_samples is not None else None
    return compute_image_mean_std(root, max_samples=max_samples)
