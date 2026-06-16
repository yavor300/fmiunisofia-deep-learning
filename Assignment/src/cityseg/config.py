"""Configuration loading, run-output setup, and reproducibility helpers."""

from __future__ import annotations

import copy
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def load_config(
    config_path: str | Path,
    default_path: str | Path | None = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Load a YAML config and merge it over the default config when appropriate."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    config = _load_yaml_mapping(path)
    if default_path is None:
        return config

    default_config_path = Path(default_path)
    if not default_config_path.exists() or path.resolve() == default_config_path.resolve():
        return config

    defaults = _load_yaml_mapping(default_config_path)
    return deep_merge(defaults, config)


def deep_merge(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge overrides into defaults without mutating either input."""
    merged = copy.deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def create_experiment_output_dir(
    config: dict[str, Any],
    stage: str,
    run_name: str | None = None,
) -> Path:
    """Create and return an output directory for a train/eval/report run."""
    output_root = Path(config.get("paths", {}).get("output_dir", "outputs"))
    safe_stage = _safe_path_part(stage)
    safe_run_name = _safe_path_part(run_name) if run_name else _timestamp_run_name()
    output_dir = output_root / safe_stage / safe_run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_resolved_config(
    config: dict[str, Any],
    output_dir: str | Path,
    filename: str = "resolved_config.yaml",
) -> Path:
    """Write the exact resolved config used by a run."""
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    return path


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch when those libraries are available."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
    except ImportError:
        np = None

    if np is not None:
        np.random.seed(seed)

    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def prepare_run(
    config: dict[str, Any],
    stage: str,
    run_name: str | None = None,
) -> tuple[Path, Path]:
    """Seed the process, create a run directory, and save the resolved config."""
    seed_everything(int(config.get("seed", 42)))
    output_dir = create_experiment_output_dir(config, stage=stage, run_name=run_name)
    config_path = save_resolved_config(config, output_dir)
    return output_dir, config_path


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return loaded


def _timestamp_run_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_path_part(value: str | None) -> str:
    if not value:
        return _timestamp_run_name()

    safe = "".join(
        character if character.isalnum() or character in "-_." else "-"
        for character in value
    )
    return safe.strip("-") or _timestamp_run_name()
