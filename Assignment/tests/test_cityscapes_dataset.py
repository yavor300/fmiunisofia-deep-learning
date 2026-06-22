from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.cityseg.constants import IGNORE_INDEX
from src.cityseg.data.cityscapes_dataset import CityscapesDataset


def _write_synthetic_cityscapes_pair(root: Path, split: str = "train") -> None:
    image_dir = root / "leftImg8bit" / split / "demo_city"
    mask_dir = root / "gtFine" / split / "demo_city"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[:, :, 0] = 128
    mask = np.array(
        [
            [7, 7, 8, 8, 0, 0],
            [7, 7, 8, 8, 0, 0],
            [24, 24, 26, 26, 33, 33],
            [24, 24, 26, 26, 33, 33],
        ],
        dtype=np.uint8,
    )

    Image.fromarray(image).save(image_dir / "demo_city_000000_000000_leftImg8bit.png")
    Image.fromarray(mask).save(mask_dir / "demo_city_000000_000000_gtFine_labelIds.png")


class TestCityscapesDatasetInit(unittest.TestCase):
    def test_when_split_has_one_pair_then_dataset_length_is_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_synthetic_cityscapes_pair(root)

            dataset = CityscapesDataset(root=root, split="train")

            self.assertEqual(len(dataset), 1)

    def test_when_split_has_no_images_then_value_error_is_raised(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                CityscapesDataset(root=directory, split="train")


class TestCityscapesDatasetGetItem(unittest.TestCase):
    def test_when_item_is_loaded_then_image_tensor_is_channel_first_float(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_synthetic_cityscapes_pair(root)
            dataset = CityscapesDataset(root=root, split="train")

            image, _ = dataset[0]

            self.assertEqual(tuple(image.shape), (3, 4, 6))
            self.assertEqual(image.dtype, torch.float32)
            self.assertGreaterEqual(float(image.min()), 0.0)
            self.assertLessEqual(float(image.max()), 1.0)

    def test_when_item_is_loaded_then_mask_tensor_is_long_with_train_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_synthetic_cityscapes_pair(root)
            dataset = CityscapesDataset(root=root, split="train")

            _, mask = dataset[0]

            self.assertEqual(tuple(mask.shape), (4, 6))
            self.assertEqual(mask.dtype, torch.long)
            valid_ids = {0, 1, 11, 13, 18, IGNORE_INDEX}

            self.assertTrue(set(torch.unique(mask).tolist()).issubset(valid_ids))

    def test_when_image_size_is_provided_then_image_and_mask_are_resized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_synthetic_cityscapes_pair(root)
            dataset = CityscapesDataset(root=root, split="train", image_size=(8, 10))

            image, mask = dataset[0]

            self.assertEqual(tuple(image.shape), (3, 8, 10))
            self.assertEqual(tuple(mask.shape), (8, 10))

    def test_when_crop_size_is_provided_then_image_and_mask_are_center_cropped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_synthetic_cityscapes_pair(root)
            dataset = CityscapesDataset(root=root, split="train", crop_size=(2, 4))

            image, mask = dataset[0]

            self.assertEqual(tuple(image.shape), (3, 2, 4))
            self.assertEqual(tuple(mask.shape), (2, 4))
