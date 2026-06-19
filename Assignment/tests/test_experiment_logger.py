from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.cityseg.training.experiment_logger import (
    BASELINE_EXPERIMENT_ID,
    EXPERIMENT_RESULT_FIELDS,
    append_experiment_result,
    build_experiment_result_row,
)


def _config() -> dict:
    return {
        "paths": {"reports_dir": "reports"},
        "training": {"epochs": 3, "batch_size": 2},
        "optimizer": {"name": "adamw", "lr": 0.001},
        "scheduler": {"name": "step_decay"},
        "model": {
            "architecture": "tiny_unet",
            "encoder_name": "none",
            "encoder_weights": None,
        },
        "loss": {"name": "cross_entropy"},
        "preprocessing": {
            "resize_height": 16,
            "resize_width": 32,
            "crop_height": 8,
            "crop_width": 8,
            "augmentations": "basic_aug",
            "normalize": "imagenet",
        },
    }


class TestBuildExperimentResultRow(unittest.TestCase):
    def test_when_config_is_logged_then_phase_twelve_columns_are_populated(self) -> None:
        row = build_experiment_result_row(
            config=_config(),
            metrics={"mean_iou": 0.1, "mean_dice": 0.2, "pixel_accuracy": 0.3},
            experiment_id="001_demo",
            checkpoint_path="outputs/train/checkpoints/best.pt",
            comments="demo",
            date="2026-06-18T00:00:00",
        )

        self.assertEqual(list(row), EXPERIMENT_RESULT_FIELDS)
        self.assertEqual(row["architecture"], "tiny_unet")
        self.assertEqual(row["image_size"], "16x32")
        self.assertEqual(row["crop_size"], "8x8")
        self.assertEqual(row["checkpoint_path"], "outputs/train/checkpoints/best.pt")


class TestAppendExperimentResult(unittest.TestCase):
    def test_when_baseline_is_written_after_other_rows_then_baseline_is_first(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results_path = Path(directory) / "experiment_results.csv"
            append_experiment_result(
                config=_config(),
                metrics={"mean_iou": 0.2, "mean_dice": 0.3, "pixel_accuracy": 0.4},
                results_path=results_path,
                experiment_id="001_other",
                comments="other",
                date="2026-06-18T00:00:01",
            )

            append_experiment_result(
                config=_config(),
                metrics={"mean_iou": 0.1, "mean_dice": 0.2, "pixel_accuracy": 0.3},
                results_path=results_path,
                experiment_id=BASELINE_EXPERIMENT_ID,
                comments="baseline",
                date="2026-06-18T00:00:00",
            )

            with results_path.open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[0]["experiment_id"], BASELINE_EXPERIMENT_ID)
            self.assertEqual(rows[1]["experiment_id"], "001_other")

    def test_when_baseline_exists_then_percentage_changes_are_computed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results_path = Path(directory) / "experiment_results.csv"
            append_experiment_result(
                config=_config(),
                metrics={"mean_iou": 0.5, "mean_dice": 0.25, "pixel_accuracy": 0.8},
                results_path=results_path,
                experiment_id=BASELINE_EXPERIMENT_ID,
                date="2026-06-18T00:00:00",
            )

            append_experiment_result(
                config=_config(),
                metrics={"mean_iou": 0.75, "mean_dice": 0.5, "pixel_accuracy": 0.9},
                results_path=results_path,
                experiment_id="001_other",
                date="2026-06-18T00:00:01",
            )

            with results_path.open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[1]["mean_iou_change_vs_baseline_pct"], "50.000000")
            self.assertEqual(rows[1]["mean_dice_change_vs_baseline_pct"], "100.000000")
            self.assertEqual(rows[1]["pixel_accuracy_change_vs_baseline_pct"], "12.500000")


if __name__ == "__main__":
    unittest.main()
