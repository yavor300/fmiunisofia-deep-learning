from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.cityseg.models.majority_baseline import (
    compute_constant_prediction_metrics,
    compute_majority_class,
    predict_majority_mask,
    run_majority_baseline,
    write_experiment_result,
)


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
    Image.fromarray(image).save(image_dir / f"{city}_{frame_id}_leftImg8bit.png")
    Image.fromarray(mask_values.astype(np.uint8)).save(
        mask_dir / f"{city}_{frame_id}_gtFine_labelIds.png"
    )


class TestComputeMajorityClass(unittest.TestCase):
    def test_when_training_masks_have_frequent_class_then_majority_class_is_returned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask = np.array([[7, 7, 7], [8, 8, 0]], dtype=np.uint8)
            _write_cityscapes_pair(root, "train", "demo_city", "000000_000000", mask)

            majority_class = compute_majority_class(root, split="train")

            self.assertEqual(majority_class, 0)

    def test_when_training_masks_have_only_ignored_pixels_then_value_error_is_raised(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask = np.zeros((2, 2), dtype=np.uint8)
            _write_cityscapes_pair(root, "train", "demo_city", "000000_000000", mask)

            with self.assertRaises(ValueError):
                compute_majority_class(root, split="train")

    def test_when_mask_limit_is_used_then_only_limited_masks_are_counted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_mask = np.full((2, 2), 7, dtype=np.uint8)
            second_mask = np.full((2, 2), 8, dtype=np.uint8)
            _write_cityscapes_pair(root, "train", "demo_city", "000000_000000", first_mask)
            _write_cityscapes_pair(root, "train", "demo_city", "000001_000000", second_mask)

            majority_class = compute_majority_class(root, split="train", max_masks=1)

            self.assertEqual(majority_class, 0)


class TestPredictMajorityMask(unittest.TestCase):
    def test_when_shape_is_provided_then_mask_is_filled_with_majority_class(self) -> None:
        prediction = predict_majority_mask(shape=(2, 3), majority_class=4)

        np.testing.assert_array_equal(prediction, np.full((2, 3), 4, dtype=np.uint8))


class TestComputeConstantPredictionMetrics(unittest.TestCase):
    def test_when_prediction_matches_some_pixels_then_expected_metrics_are_returned(self) -> None:
        target = np.array([[0, 0], [1, 255]], dtype=np.uint8)

        metrics = compute_constant_prediction_metrics(
            targets=[target],
            majority_class=0,
            num_classes=2,
        )

        self.assertAlmostEqual(metrics["pixel_accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["mean_iou"], (2 / 3 + 0.0) / 2)
        self.assertAlmostEqual(metrics["mean_dice"], (4 / 5 + 0.0) / 2)


class TestWriteExperimentResult(unittest.TestCase):
    def test_when_baseline_result_is_written_then_it_is_first_csv_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results_path = Path(directory) / "experiment_results.csv"
            results_path.write_text(
                "experiment_id,model,split,mean_iou,mean_dice,pixel_accuracy,comments\n"
                "001_other,unet,val,0.5,0.6,0.7,other\n",
                encoding="utf-8",
            )
            result = {
                "experiment_id": "000_baseline_majority",
                "model": "majority_baseline",
                "split": "val",
                "mean_iou": 0.1,
                "mean_dice": 0.2,
                "pixel_accuracy": 0.3,
                "comments": "baseline",
            }

            write_experiment_result(result, results_path)

            with results_path.open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[0]["experiment_id"], "000_baseline_majority")
            self.assertEqual(rows[1]["experiment_id"], "001_other")


class TestRunMajorityBaseline(unittest.TestCase):
    def test_when_baseline_runs_then_results_csv_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / "data" / "raw" / "cityscapes"
            reports_dir = workspace / "reports"
            output_dir = workspace / "outputs"
            train_mask = np.array([[7, 7], [8, 0]], dtype=np.uint8)
            val_mask = np.array([[7, 8], [7, 0]], dtype=np.uint8)
            _write_cityscapes_pair(root, "train", "train_city", "000000_000000", train_mask)
            _write_cityscapes_pair(root, "val", "val_city", "000000_000000", val_mask)
            config = {
                "seed": 42,
                "paths": {
                    "data_root": str(root),
                    "reports_dir": str(reports_dir),
                    "output_dir": str(output_dir),
                },
                "model": {"num_classes": 19},
            }

            result = run_majority_baseline(config, eval_split="val", run_name="test_run")

            self.assertEqual(result["experiment_id"], "000_baseline_majority")
            self.assertTrue((reports_dir / "experiment_results.csv").exists())
            self.assertTrue((reports_dir / "model_report.xlsx").exists())
            resolved_config_path = output_dir / "baseline" / "test_run" / "resolved_config.yaml"

            self.assertTrue(resolved_config_path.exists())


if __name__ == "__main__":
    unittest.main()
