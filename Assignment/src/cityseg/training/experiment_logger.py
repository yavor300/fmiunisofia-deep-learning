"""Shared CSV experiment tracking for training, evaluation, and baselines."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

BASELINE_EXPERIMENT_ID = "000_baseline_majority"

EXPERIMENT_RESULT_FIELDS = [
    "experiment_id",
    "date",
    "architecture",
    "encoder",
    "pretrained_weights",
    "loss",
    "optimizer",
    "learning_rate",
    "scheduler",
    "epochs",
    "batch_size",
    "image_size",
    "crop_size",
    "augmentation",
    "normalization",
    "mean_iou",
    "mean_iou_change_vs_baseline_pct",
    "mean_dice",
    "mean_dice_change_vs_baseline_pct",
    "pixel_accuracy",
    "pixel_accuracy_change_vs_baseline_pct",
    "checkpoint_path",
    "comments",
]

_CHANGE_COLUMNS = {
    "mean_iou": "mean_iou_change_vs_baseline_pct",
    "mean_dice": "mean_dice_change_vs_baseline_pct",
    "pixel_accuracy": "pixel_accuracy_change_vs_baseline_pct",
}


def build_experiment_result_row(
    config: dict[str, Any],
    metrics: dict[str, Any],
    experiment_id: str,
    checkpoint_path: str | Path = "",
    comments: str = "",
    date: str | None = None,
) -> dict[str, str]:
    training = config.get("training", {})
    optimizer = config.get("optimizer", {})
    scheduler = config.get("scheduler", {})
    model = config.get("model", {})
    loss = config.get("loss", {})
    preprocessing = config.get("preprocessing", {})

    row = {
        "experiment_id": experiment_id,
        "date": date or datetime.now().isoformat(timespec="seconds"),
        "architecture": _string_value(model.get("architecture", "")),
        "encoder": _string_value(model.get("encoder_name", "")),
        "pretrained_weights": _string_value(model.get("encoder_weights", "")),
        "loss": _string_value(loss.get("name", "")),
        "optimizer": _string_value(optimizer.get("name", "")),
        "learning_rate": _format_number(optimizer.get("lr", "")),
        "scheduler": _string_value(scheduler.get("name", "")),
        "epochs": _string_value(training.get("epochs", "")),
        "batch_size": _string_value(training.get("batch_size", "")),
        "image_size": _format_size(
            preprocessing.get("resize_height"),
            preprocessing.get("resize_width"),
        ),
        "crop_size": _format_size(
            preprocessing.get("crop_height"),
            preprocessing.get("crop_width"),
        ),
        "augmentation": _string_value(preprocessing.get("augmentations", "")),
        "normalization": _string_value(preprocessing.get("normalize", "")),
        "mean_iou": _format_number(metrics.get("mean_iou", "")),
        "mean_iou_change_vs_baseline_pct": "",
        "mean_dice": _format_number(metrics.get("mean_dice", "")),
        "mean_dice_change_vs_baseline_pct": "",
        "pixel_accuracy": _format_number(metrics.get("pixel_accuracy", "")),
        "pixel_accuracy_change_vs_baseline_pct": "",
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else "",
        "comments": comments,
    }
    return {field: row[field] for field in EXPERIMENT_RESULT_FIELDS}


def append_experiment_result(
    config: dict[str, Any],
    metrics: dict[str, Any],
    results_path: str | Path,
    experiment_id: str,
    checkpoint_path: str | Path = "",
    comments: str = "",
    date: str | None = None,
) -> Path:
    path = Path(results_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [_normalize_existing_row(row) for row in read_experiment_results(path)]
    row = build_experiment_result_row(
        config=config,
        metrics=metrics,
        experiment_id=experiment_id,
        checkpoint_path=checkpoint_path,
        comments=comments,
        date=date,
    )

    if experiment_id == BASELINE_EXPERIMENT_ID:
        rows = [row, *[existing for existing in rows if not _is_baseline(existing)]]
    else:
        rows = _baseline_first(rows) + [row]
    _populate_all_change_columns(rows)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=EXPERIMENT_RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_experiment_results(results_path: str | Path) -> list[dict[str, str]]:
    path = Path(results_path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _baseline_first(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    baseline = _find_baseline_row(rows)
    non_baseline = [row for row in rows if not _is_baseline(row)]
    return [baseline, *non_baseline] if baseline is not None else non_baseline


def _find_baseline_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    for row in rows:
        if _is_baseline(row):
            return row
    return None


def _is_baseline(row: dict[str, str]) -> bool:
    return row.get("experiment_id") == BASELINE_EXPERIMENT_ID


def _normalize_existing_row(row: dict[str, Any]) -> dict[str, str]:
    normalized = {field: _string_value(row.get(field, "")) for field in EXPERIMENT_RESULT_FIELDS}
    if not normalized["architecture"]:
        normalized["architecture"] = _string_value(row.get("model", ""))
    return normalized


def _populate_change_columns(
    row: dict[str, str],
    baseline: dict[str, str] | None,
) -> None:
    for metric_column, change_column in _CHANGE_COLUMNS.items():
        if _is_baseline(row):
            row[change_column] = "0.000000"
            continue
        if baseline is None:
            row[change_column] = ""
            continue
        row[change_column] = _format_percentage_change(
            row.get(metric_column, ""),
            baseline.get(metric_column, ""),
        )


def _populate_all_change_columns(rows: list[dict[str, str]]) -> None:
    baseline = _find_baseline_row(rows)
    for row in rows:
        _populate_change_columns(row, baseline=baseline)


def _format_percentage_change(value: Any, baseline: Any) -> str:
    value_float = _parse_float(value)
    baseline_float = _parse_float(baseline)
    if value_float is None or baseline_float is None or baseline_float == 0.0:
        return ""
    return f"{((value_float - baseline_float) / baseline_float) * 100.0:.6f}"


def _format_size(height: Any, width: Any) -> str:
    if height in (None, "") or width in (None, ""):
        return ""
    return f"{height}x{width}"


def _format_number(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _parse_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _string_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)
