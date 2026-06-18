from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.cityseg.training.evaluate import evaluate_checkpoint
from src.cityseg.training.train import train_model


def _write_cityscapes_pair(
    root: Path,
    split: str,
    city: str,
    frame_id: str,
    mask_values: np.ndarray,
) -> None:
    image_dir = root / "leftImg8bit" / split / city
    mask_dir = root / "gtFine" / split / city
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    image = np.zeros((*mask_values.shape, 3), dtype=np.uint8)
    image[:, :, 1] = 128
    Image.fromarray(image).save(image_dir / f"{city}_{frame_id}_leftImg8bit.png")
    Image.fromarray(mask_values.astype(np.uint8)).save(
        mask_dir / f"{city}_{frame_id}_gtFine_labelIds.png"
    )


def _write_tiny_dataset(root: Path) -> None:
    train_mask = np.array([[7, 7, 8, 8], [7, 7, 8, 8], [7, 7, 8, 8], [7, 7, 8, 8]])
    val_mask = np.array([[7, 8, 7, 8], [7, 8, 7, 8], [7, 8, 7, 8], [7, 8, 7, 8]])
    _write_cityscapes_pair(root, "train", "train_city", "000000_000000", train_mask)
    _write_cityscapes_pair(root, "val", "val_city", "000000_000000", val_mask)


def _tiny_config(workspace: Path) -> dict:
    return {
        "seed": 42,
        "paths": {
            "data_root": str(workspace / "data" / "raw" / "cityscapes"),
            "output_dir": str(workspace / "outputs"),
            "reports_dir": str(workspace / "reports"),
        },
        "training": {
            "epochs": 1,
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "shuffle": False,
            "mixed_precision": False,
            "device": "cpu",
            "gradient_clip_norm": None,
        },
        "optimizer": {"name": "adamw", "lr": 0.001, "weight_decay": 0.0},
        "scheduler": {"name": "none"},
        "model": {
            "architecture": "tiny_unet",
            "in_channels": 3,
            "num_classes": 19,
            "base_channels": 4,
        },
        "loss": {"name": "cross_entropy", "ignore_index": 255},
        "preprocessing": {
            "resize_height": 16,
            "resize_width": 16,
            "crop_height": None,
            "crop_width": None,
            "augmentations": "resize_only",
            "eval_augmentations": "resize_only",
            "normalize": "none",
        },
    }


class TestEvaluateCheckpoint(unittest.TestCase):
    def test_when_checkpoint_is_evaluated_then_metrics_and_figures_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            _write_tiny_dataset(workspace / "data" / "raw" / "cityscapes")
            config = _tiny_config(workspace)
            train_output_dir, _ = train_model(config, run_name="train")
            checkpoint = train_output_dir / "checkpoints" / "best.pt"

            output_dir, metrics = evaluate_checkpoint(
                config=config,
                checkpoint_path=checkpoint,
                split="val",
                run_name="eval",
                max_examples=1,
            )

            figures_dir = workspace / "reports" / "figures" / "eval"
            self.assertTrue((output_dir / "global_metrics.csv").exists())
            self.assertTrue((output_dir / "per_class_iou.csv").exists())
            self.assertTrue((output_dir / "confusion_matrix.csv").exists())
            self.assertTrue((output_dir / "error_analysis.md").exists())
            self.assertTrue((figures_dir / "example_000_original.png").exists())
            self.assertTrue((figures_dir / "example_000_error_map.png").exists())
            self.assertIn("mean_iou", metrics)

    def test_when_global_metrics_are_written_then_only_main_metrics_are_in_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            _write_tiny_dataset(workspace / "data" / "raw" / "cityscapes")
            config = _tiny_config(workspace)
            train_output_dir, _ = train_model(config, run_name="train")

            output_dir, _ = evaluate_checkpoint(
                config=config,
                checkpoint_path=train_output_dir / "checkpoints" / "best.pt",
                split="val",
                run_name="eval",
                max_examples=0,
            )

            with (output_dir / "global_metrics.csv").open("r", encoding="utf-8") as file:
                row = next(csv.DictReader(file))
            self.assertEqual(set(row), {"mean_iou", "mean_dice", "pixel_accuracy"})


if __name__ == "__main__":
    unittest.main()
