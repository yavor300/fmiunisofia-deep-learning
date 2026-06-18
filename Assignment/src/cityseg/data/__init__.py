"""Data loading and preprocessing utilities."""

from __future__ import annotations

from typing import Any

__all__ = [
    "CityscapesDataset",
    "convert_label_ids_to_train_ids",
    "create_dataloader",
    "create_train_val_dataloaders",
    "create_transforms_from_config",
    "decode_train_ids_to_colors",
    "get_class_names",
    "get_palette",
]


def __getattr__(name: str) -> Any:
    if name == "CityscapesDataset":
        from src.cityseg.data.cityscapes_dataset import CityscapesDataset

        return CityscapesDataset
    if name in {"create_dataloader", "create_train_val_dataloaders"}:
        from src.cityseg.data.dataloaders import create_dataloader, create_train_val_dataloaders

        return {
            "create_dataloader": create_dataloader,
            "create_train_val_dataloaders": create_train_val_dataloaders,
        }[name]
    if name in {
        "convert_label_ids_to_train_ids",
        "create_transforms_from_config",
        "decode_train_ids_to_colors",
        "get_class_names",
        "get_palette",
    }:
        if name == "create_transforms_from_config":
            from src.cityseg.data.transforms import create_transforms_from_config

            return create_transforms_from_config
        from src.cityseg.data.label_mapping import (
            convert_label_ids_to_train_ids,
            decode_train_ids_to_colors,
            get_class_names,
            get_palette,
        )

        return {
            "convert_label_ids_to_train_ids": convert_label_ids_to_train_ids,
            "decode_train_ids_to_colors": decode_train_ids_to_colors,
            "get_class_names": get_class_names,
            "get_palette": get_palette,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
