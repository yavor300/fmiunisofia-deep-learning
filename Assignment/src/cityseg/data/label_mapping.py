"""Utilities for converting and visualizing Cityscapes labels."""

from __future__ import annotations

import numpy as np

from src.cityseg.constants import (
    CITYSCAPES_CLASSES,
    CITYSCAPES_PALETTE,
    IGNORE_INDEX,
    LABEL_ID_TO_TRAIN_ID,
)


def convert_label_ids_to_train_ids(mask: np.ndarray) -> np.ndarray:
    """Convert original Cityscapes label IDs to the 19-class train ID space."""
    converted = np.full(mask.shape, IGNORE_INDEX, dtype=np.uint8)
    for label_id, train_id in LABEL_ID_TO_TRAIN_ID.items():
        converted[mask == label_id] = train_id
    return converted


def decode_train_ids_to_colors(mask: np.ndarray) -> np.ndarray:
    """Decode a train ID mask into an RGB visualization."""
    decoded = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for train_id, color in enumerate(CITYSCAPES_PALETTE):
        decoded[mask == train_id] = color
    return decoded


def get_class_names() -> tuple[str, ...]:
    """Return the 19 Cityscapes train classes in train ID order."""
    return CITYSCAPES_CLASSES


def get_palette() -> tuple[tuple[int, int, int], ...]:
    """Return the RGB palette in train ID order."""
    return CITYSCAPES_PALETTE
