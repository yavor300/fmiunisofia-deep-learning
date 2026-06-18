from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.cityseg.data.cityscapes_dataset import CityscapesDataset
from src.cityseg.data.transforms import (
    CITYSCAPES_MEAN,
    CITYSCAPES_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_transforms,
    compute_image_mean_std,
    create_transforms_from_config,
)


def _image_and_mask() -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    image[:, :, 0] = 120
    mask = np.zeros((8, 10), dtype=np.uint8)
    mask[:, 5:] = 1
    return image, mask


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((4, 4, 3), value, dtype=np.uint8)
    Image.fromarray(image).save(path)


class TestBuildTransforms(unittest.TestCase):
    def test_when_strategy_is_resize_only_then_image_and_mask_are_resized(self) -> None:
        image, mask = _image_and_mask()
        transforms = build_transforms("resize_only", image_size=(4, 6))

        result = transforms(image=image, mask=mask)

        self.assertEqual(result["image"].shape[:2], (4, 6))
        self.assertEqual(result["mask"].shape, (4, 6))

    def test_when_strategy_is_random_crop_then_crop_size_is_returned(self) -> None:
        image, mask = _image_and_mask()
        transforms = build_transforms("random_crop", image_size=(8, 10), crop_size=(4, 4))

        result = transforms(image=image, mask=mask)

        self.assertEqual(result["image"].shape[:2], (4, 4))
        self.assertEqual(result["mask"].shape, (4, 4))

    def test_when_strategy_is_basic_aug_then_mask_values_remain_class_ids(self) -> None:
        image, mask = _image_and_mask()
        transforms = build_transforms("basic_aug", image_size=(8, 10), crop_size=(4, 4))

        result = transforms(image=image, mask=mask)

        self.assertTrue(set(np.unique(result["mask"])).issubset({0, 1}))

    def test_when_strategy_alias_is_used_then_transform_is_built(self) -> None:
        image, mask = _image_and_mask()
        transforms = build_transforms("basic", image_size=(8, 10), crop_size=(4, 4))

        result = transforms(image=image, mask=mask)

        self.assertEqual(result["image"].shape[:2], (4, 4))

    def test_when_strategy_is_strong_aug_then_mask_values_remain_class_ids(self) -> None:
        image, mask = _image_and_mask()
        transforms = build_transforms("strong_aug", image_size=(8, 10), crop_size=(4, 4))

        result = transforms(image=image, mask=mask)

        self.assertTrue(set(np.unique(result["mask"])).issubset({0, 1}))

    def test_when_imagenet_normalization_is_used_then_float_image_is_returned(self) -> None:
        image, mask = _image_and_mask()
        transforms = build_transforms(
            "resize_only",
            image_size=(4, 4),
            normalization="imagenet",
        )

        result = transforms(image=image, mask=mask)

        self.assertEqual(result["image"].dtype, np.float32)
        self.assertEqual(tuple(IMAGENET_MEAN), (0.485, 0.456, 0.406))
        self.assertEqual(tuple(IMAGENET_STD), (0.229, 0.224, 0.225))

    def test_when_cityscapes_normalization_is_used_then_float_image_is_returned(self) -> None:
        image, mask = _image_and_mask()
        transforms = build_transforms(
            "resize_only",
            image_size=(4, 4),
            normalization="cityscapes",
        )

        result = transforms(image=image, mask=mask)

        self.assertEqual(result["image"].dtype, np.float32)
        self.assertEqual(len(CITYSCAPES_MEAN), 3)
        self.assertEqual(len(CITYSCAPES_STD), 3)


class TestComputeImageMeanStd(unittest.TestCase):
    def test_when_images_exist_then_mean_and_std_are_computed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_image(root / "leftImg8bit" / "train" / "city" / "a_leftImg8bit.png", 64)
            _write_image(root / "leftImg8bit" / "train" / "city" / "b_leftImg8bit.png", 128)

            mean, std = compute_image_mean_std(root, max_samples=2)

            self.assertEqual(len(mean), 3)
            self.assertEqual(len(std), 3)
            self.assertTrue(all(value > 0 for value in std))


class TestCreateTransformsFromConfig(unittest.TestCase):
    def test_when_config_uses_basic_aug_then_train_transform_has_crop_size(self) -> None:
        config = {
            "preprocessing": {
                "resize_height": 8,
                "resize_width": 10,
                "crop_height": 4,
                "crop_width": 4,
                "augmentations": "basic_aug",
                "normalize": "none",
            }
        }

        transforms = create_transforms_from_config(config, split="train")
        result = transforms(**dict(zip(("image", "mask"), _image_and_mask(), strict=True)))

        self.assertEqual(result["image"].shape[:2], (4, 4))


class TestCityscapesDatasetWithTransforms(unittest.TestCase):
    def test_when_normalized_transform_is_used_then_image_is_not_scaled_twice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "leftImg8bit" / "train" / "city"
            mask_dir = root / "gtFine" / "train" / "city"
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            Image.fromarray(np.full((4, 4, 3), 128, dtype=np.uint8)).save(
                image_dir / "city_000000_000000_leftImg8bit.png"
            )
            Image.fromarray(np.full((4, 4), 7, dtype=np.uint8)).save(
                mask_dir / "city_000000_000000_gtFine_labelIds.png"
            )
            transforms = build_transforms(
                "resize_only",
                image_size=(4, 4),
                normalization="imagenet",
            )

            image, mask = CityscapesDataset(root, split="train", transforms=transforms)[0]

            self.assertEqual(tuple(image.shape), (3, 4, 4))
            self.assertGreater(float(image.abs().mean()), 0.1)
            self.assertEqual(tuple(mask.shape), (4, 4))


if __name__ == "__main__":
    unittest.main()
