from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.cityseg.training.train import build_loss, build_optimizer, build_scheduler, train_model


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
    image[:, :, 0] = 120

    Image.fromarray(image).save(image_dir / f"{city}_{frame_id}_leftImg8bit.png")
    Image.fromarray(mask_values.astype(np.uint8)).save(
        mask_dir / f"{city}_{frame_id}_gtFine_labelIds.png"
    )


def _write_tiny_dataset(root: Path) -> None:
    train_mask = np.array(
        [
            [7, 7, 8, 8],
            [7, 7, 8, 8],
            [7, 7, 8, 8],
            [7, 7, 8, 8],
        ],
        dtype=np.uint8,
    )
    val_mask = np.array(
        [
            [7, 8, 7, 8],
            [7, 8, 7, 8],
            [7, 8, 7, 8],
            [7, 8, 7, 8],
        ],
        dtype=np.uint8,
    )
    _write_cityscapes_pair(root, "train", "train_city", "000000_000000", train_mask)
    _write_cityscapes_pair(root, "val", "val_city", "000000_000000", val_mask)


def _tiny_config(workspace: Path, epochs: int = 1) -> dict:
    return {
        "seed": 42,
        "paths": {
            "data_root": str(workspace / "data" / "raw" / "cityscapes"),
            "output_dir": str(workspace / "outputs"),
            "reports_dir": str(workspace / "reports"),
        },
        "training": {
            "epochs": epochs,
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "shuffle": False,
            "mixed_precision": False,
            "progress_bar": False,
            "device": "cpu",
            "early_stopping_patience": None,
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
        },
        "metrics": {"main_metric": "mean_iou"},
    }


class TestBuildLoss(unittest.TestCase):
    def test_when_loss_name_is_cross_entropy_then_loss_module_is_returned(self) -> None:
        loss = build_loss({"name": "cross_entropy", "ignore_index": 255})

        self.assertIsInstance(loss, torch.nn.Module)

    def test_when_loss_name_is_cross_entropy_lovasz_then_loss_module_is_returned(self) -> None:
        loss = build_loss({"name": "cross_entropy_lovasz", "ignore_index": 255})

        self.assertIsInstance(loss, torch.nn.Module)


class TestBuildOptimizer(unittest.TestCase):
    def test_when_optimizer_name_is_adamw_then_optimizer_is_returned(self) -> None:
        model = torch.nn.Conv2d(3, 2, kernel_size=1)

        optimizer = build_optimizer(model, {"name": "adamw", "lr": 0.001})

        self.assertIsInstance(optimizer, torch.optim.AdamW)


class TestBuildScheduler(unittest.TestCase):
    def test_when_scheduler_name_is_none_then_none_is_returned(self) -> None:
        model = torch.nn.Conv2d(3, 2, kernel_size=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        scheduler = build_scheduler(optimizer, {"name": "none"})

        self.assertIsNone(scheduler)


class TestTrainModel(unittest.TestCase):
    def test_when_training_runs_one_epoch_then_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            data_root = workspace / "data" / "raw" / "cityscapes"
            _write_tiny_dataset(data_root)

            output_dir, history = train_model(_tiny_config(workspace), run_name="one_epoch")

            self.assertEqual(len(history), 1)
            self.assertTrue((output_dir / "checkpoints" / "best.pt").exists())
            self.assertTrue((output_dir / "checkpoints" / "last.pt").exists())
            self.assertTrue((output_dir / "history.csv").exists())
            self.assertTrue((output_dir / "train_val_loss.png").exists())
            self.assertTrue((output_dir / "train_val_mean_iou.png").exists())
            self.assertTrue((output_dir / "learning_rate.png").exists())
            self.assertTrue((output_dir / "predictions_preview.png").exists())
            self.assertTrue((output_dir / "config.yaml").exists())
            self.assertTrue((output_dir / "resolved_config.yaml").exists())

    def test_when_resume_checkpoint_is_passed_then_training_continues_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            data_root = workspace / "data" / "raw" / "cityscapes"
            _write_tiny_dataset(data_root)
            first_config = _tiny_config(workspace, epochs=1)
            output_dir, _ = train_model(first_config, run_name="resume")
            resume_path = output_dir / "checkpoints" / "last.pt"

            second_config = _tiny_config(workspace, epochs=2)
            output_dir, history = train_model(second_config, run_name="resume", resume=resume_path)

            self.assertEqual(len(history), 2)
            with (output_dir / "history.csv").open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[-1]["epoch"], "2")
