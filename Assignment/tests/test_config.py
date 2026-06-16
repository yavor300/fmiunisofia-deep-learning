from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import yaml

from src.cityseg.config import (
    create_experiment_output_dir,
    deep_merge,
    load_config,
    save_resolved_config,
    seed_everything,
)


class TestDeepMerge(unittest.TestCase):
    def test_when_override_changes_nested_value_then_other_defaults_are_preserved(self) -> None:
        defaults = {"training": {"epochs": 30, "batch_size": 4}, "model": {"num_classes": 19}}
        overrides = {"training": {"batch_size": 2}}

        merged = deep_merge(defaults, overrides)

        self.assertEqual(merged["training"]["epochs"], 30)
        self.assertEqual(merged["training"]["batch_size"], 2)
        self.assertEqual(merged["model"]["num_classes"], 19)

    def test_when_merge_is_performed_then_inputs_are_not_mutated(self) -> None:
        defaults = {"training": {"epochs": 30}}
        overrides = {"training": {"epochs": 5}}

        deep_merge(defaults, overrides)

        self.assertEqual(defaults["training"]["epochs"], 30)
        self.assertEqual(overrides["training"]["epochs"], 5)


class TestLoadConfig(unittest.TestCase):
    def test_when_default_config_is_loaded_then_yaml_values_are_returned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "default.yaml"
            config_path.write_text("seed: 123\ntraining:\n  epochs: 7\n", encoding="utf-8")

            config = load_config(config_path)

            self.assertEqual(config["seed"], 123)
            self.assertEqual(config["training"]["epochs"], 7)

    def test_when_experiment_config_is_loaded_then_it_overrides_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            default_path = root / "default.yaml"
            experiment_path = root / "experiment.yaml"
            default_path.write_text(
                "seed: 42\ntraining:\n  epochs: 30\n  batch_size: 4\n",
                encoding="utf-8",
            )
            experiment_path.write_text("training:\n  batch_size: 2\n", encoding="utf-8")

            config = load_config(experiment_path, default_path=default_path)

            self.assertEqual(config["seed"], 42)
            self.assertEqual(config["training"]["epochs"], 30)
            self.assertEqual(config["training"]["batch_size"], 2)

    def test_when_config_file_is_missing_then_file_not_found_error_is_raised(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_config(Path(directory) / "missing.yaml")


class TestCreateExperimentOutputDir(unittest.TestCase):
    def test_when_run_name_is_provided_then_named_output_directory_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = {"paths": {"output_dir": directory}}

            output_dir = create_experiment_output_dir(config, stage="train", run_name="demo")

            self.assertTrue(output_dir.exists())
            self.assertEqual(output_dir.name, "demo")
            self.assertEqual(output_dir.parent.name, "train")


class TestSaveResolvedConfig(unittest.TestCase):
    def test_when_config_is_saved_then_resolved_config_file_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            config = {"seed": 42, "training": {"epochs": 1}}

            saved_path = save_resolved_config(config, output_dir)

            self.assertEqual(saved_path.name, "resolved_config.yaml")
            self.assertEqual(yaml.safe_load(saved_path.read_text(encoding="utf-8")), config)


class TestSeedEverything(unittest.TestCase):
    def test_when_same_seed_is_used_then_random_values_are_reproducible(self) -> None:
        seed_everything(123)
        first = random.random()

        seed_everything(123)
        second = random.random()

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
