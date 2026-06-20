from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.cityseg.eda.analyze_dataset import (
    analyze_cityscapes_dataset,
    build_dataset_report,
    run_analysis_from_config,
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
    image[:, :, 1] = 120

    Image.fromarray(image).save(image_dir / f"{city}_{frame_id}_leftImg8bit.png")
    Image.fromarray(mask_values.astype(np.uint8)).save(
        mask_dir / f"{city}_{frame_id}_gtFine_labelIds.png"
    )


class TestAnalyzeCityscapesDataset(unittest.TestCase):
    def test_when_dataset_has_pairs_then_split_counts_are_returned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask = np.array([[7, 7], [8, 24]], dtype=np.uint8)
            _write_cityscapes_pair(root, "train", "train_city", "000000_000000", mask)
            _write_cityscapes_pair(root, "val", "val_city", "000000_000000", mask)

            analysis = analyze_cityscapes_dataset(root)

            self.assertEqual(analysis.split_image_counts["train"], 1)
            self.assertEqual(analysis.split_image_counts["val"], 1)

    def test_when_masks_are_analyzed_then_class_percentages_are_returned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask = np.array([[7, 7], [8, 24]], dtype=np.uint8)
            _write_cityscapes_pair(root, "train", "train_city", "000000_000000", mask)

            analysis = analyze_cityscapes_dataset(root)

            self.assertAlmostEqual(analysis.class_percentages["road"], 50.0)
            self.assertAlmostEqual(analysis.class_percentages["sidewalk"], 25.0)
            self.assertAlmostEqual(analysis.class_percentages["person"], 25.0)

    def test_when_mask_file_is_missing_then_anomaly_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "leftImg8bit" / "train" / "demo_city"
            image_dir.mkdir(parents=True)
            Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(
                image_dir / "demo_city_000000_000000_leftImg8bit.png"
            )

            analysis = analyze_cityscapes_dataset(root)

            self.assertEqual(analysis.anomaly_counts["missing_masks"], 1)

    def test_when_mask_contains_unknown_label_then_invalid_value_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask = np.array([[7, 99], [8, 24]], dtype=np.uint8)
            _write_cityscapes_pair(root, "train", "train_city", "000000_000000", mask)

            analysis = analyze_cityscapes_dataset(root)

            self.assertEqual(analysis.anomaly_counts["invalid_mask_values"], 1)

    def test_when_sample_limit_is_used_then_only_limited_pairs_are_analyzed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            road_mask = np.full((2, 2), 7, dtype=np.uint8)
            sidewalk_mask = np.full((2, 2), 8, dtype=np.uint8)
            _write_cityscapes_pair(root, "train", "train_city", "000000_000000", road_mask)
            _write_cityscapes_pair(root, "train", "train_city", "000001_000000", sidewalk_mask)

            analysis = analyze_cityscapes_dataset(root, max_samples_per_split=1)

            self.assertEqual(analysis.split_image_counts["train"], 2)
            self.assertEqual(analysis.analyzed_pair_counts["train"], 1)
            self.assertEqual(analysis.class_percentages["road"], 100.0)
            self.assertEqual(analysis.class_percentages["sidewalk"], 0.0)


class TestBuildDatasetReport(unittest.TestCase):
    def test_when_report_is_built_then_class_imbalance_discussion_is_included(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask = np.array([[7, 7], [7, 24]], dtype=np.uint8)
            _write_cityscapes_pair(root, "train", "train_city", "000000_000000", mask)
            analysis = analyze_cityscapes_dataset(root)

            report = build_dataset_report(analysis)

            self.assertIn("Class Imbalance", report)
            self.assertIn("focal loss", report)
            self.assertIn("Dice loss", report)


class TestRunAnalysisFromConfig(unittest.TestCase):
    def test_when_config_is_provided_then_report_and_figures_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / "data" / "raw" / "cityscapes"
            reports_dir = workspace / "reports"
            docs_dir = workspace / "docs"
            mask = np.array([[7, 7], [8, 24]], dtype=np.uint8)
            _write_cityscapes_pair(root, "train", "train_city", "000000_000000", mask)
            config = {
                "paths": {
                    "data_root": str(root),
                    "reports_dir": str(reports_dir),
                    "docs_dir": str(docs_dir),
                }
            }

            run_analysis_from_config(config)

            self.assertTrue((docs_dir / "dataset_analysis.md").exists())
            self.assertTrue((reports_dir / "figures" / "class_distribution.png").exists())
            self.assertTrue((reports_dir / "figures" / "image_size_distribution.png").exists())
            self.assertTrue((reports_dir / "figures" / "sample_overlays.png").exists())
            self.assertTrue((reports_dir / "figures" / "rare_classes_examples.png").exists())
            self.assertTrue(
                (reports_dir / "figures" / "paper_class_pixels_by_category.png").exists()
            )


if __name__ == "__main__":
    unittest.main()
