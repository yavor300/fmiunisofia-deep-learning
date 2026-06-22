from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.cityseg.data.dataloaders import create_dataloader, create_train_val_dataloaders


def _write_synthetic_cityscapes_pair(root: Path, split: str, city: str) -> None:
    image_dir = root / "leftImg8bit" / split / city
    mask_dir = root / "gtFine" / split / city
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    image = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.full((4, 4), 7, dtype=np.uint8)

    Image.fromarray(image).save(image_dir / f"{city}_000000_000000_leftImg8bit.png")
    Image.fromarray(mask).save(mask_dir / f"{city}_000000_000000_gtFine_labelIds.png")


class TestCreateDataloader(unittest.TestCase):
    def test_when_dataloader_is_created_then_batch_contains_images_and_masks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_synthetic_cityscapes_pair(root, split="train", city="train_city")

            dataloader = create_dataloader(root=root, split="train", batch_size=1, num_workers=0)
            images, masks = next(iter(dataloader))

            self.assertEqual(tuple(images.shape), (1, 3, 4, 4))
            self.assertEqual(tuple(masks.shape), (1, 4, 4))


class TestCreateTrainValDataloaders(unittest.TestCase):
    def test_when_config_is_provided_then_train_and_val_loaders_are_returned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_synthetic_cityscapes_pair(root, split="train", city="train_city")
            _write_synthetic_cityscapes_pair(root, split="val", city="val_city")
            config = {
                "paths": {"data_root": str(root)},
                "training": {
                    "batch_size": 1,
                    "num_workers": 0,
                    "pin_memory": False,
                    "shuffle": True,
                },
                "preprocessing": {
                    "resize_height": 4,
                    "resize_width": 4,
                    "crop_height": None,
                    "crop_width": None,
                },
            }

            train_loader, val_loader = create_train_val_dataloaders(config)

            self.assertEqual(tuple(next(iter(train_loader))[0].shape), (1, 3, 4, 4))
            self.assertEqual(tuple(next(iter(val_loader))[0].shape), (1, 3, 4, 4))
