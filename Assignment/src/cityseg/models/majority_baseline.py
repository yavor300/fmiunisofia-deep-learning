"""CPU-only majority-class semantic segmentation baseline."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.cityseg.config import load_config, prepare_run
from src.cityseg.constants import CITYSCAPES_CLASSES, IGNORE_INDEX
from src.cityseg.data.label_mapping import convert_label_ids_to_train_ids
from src.cityseg.reporting.build_model_report import build_model_report
from src.cityseg.training.experiment_logger import (
    BASELINE_EXPERIMENT_ID,
    EXPERIMENT_RESULT_FIELDS,
    append_experiment_result,
)

EXPERIMENT_ID = BASELINE_EXPERIMENT_ID
MODEL_NAME = "majority_baseline"
RESULT_FIELDNAMES = EXPERIMENT_RESULT_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the majority-class baseline.")
    parser.add_argument(
        "--config",
        default="configs/experiments/000_baseline_majority.yaml",
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=("val", "test"),
        help="Evaluation split for the baseline row.",
    )
    parser.add_argument(
        "--run-name",
        default=EXPERIMENT_ID,
        help="Output run directory name.",
    )
    parser.add_argument(
        "--max-train-masks",
        type=int,
        default=None,
        help="Limit training masks used to estimate the majority class. Use 0 for all masks.",
    )
    parser.add_argument(
        "--max-eval-masks",
        type=int,
        default=None,
        help="Limit evaluation masks used for metrics. Use 0 for all masks.",
    )
    return parser.parse_args()


def compute_majority_class(
    root: str | Path,
    split: str = "train",
    max_masks: int | None = None,
) -> int:
    counts = np.zeros(len(CITYSCAPES_CLASSES), dtype=np.int64)
    for mask in iter_train_id_masks(root, split, max_masks=max_masks):
        valid_pixels = mask[mask != IGNORE_INDEX]
        if valid_pixels.size:
            counts += np.bincount(valid_pixels, minlength=len(CITYSCAPES_CLASSES))

    if int(counts.sum()) == 0:
        raise ValueError(f"No non-ignored pixels found in split '{split}'")
    return int(counts.argmax())


def predict_majority_mask(shape: tuple[int, int], majority_class: int) -> np.ndarray:
    return np.full(shape, majority_class, dtype=np.uint8)


def compute_constant_prediction_metrics(
    targets: Iterable[np.ndarray],
    majority_class: int,
    num_classes: int,
) -> dict[str, float]:
    intersections = np.zeros(num_classes, dtype=np.float64)
    unions = np.zeros(num_classes, dtype=np.float64)
    prediction_counts = np.zeros(num_classes, dtype=np.float64)
    target_counts = np.zeros(num_classes, dtype=np.float64)
    correct_pixels = 0
    total_pixels = 0

    for target in targets:
        valid_mask = target != IGNORE_INDEX
        valid_target = target[valid_mask]
        if valid_target.size == 0:
            continue

        prediction = predict_majority_mask(target.shape, majority_class)[valid_mask]
        correct_pixels += int((prediction == valid_target).sum())
        total_pixels += int(valid_target.size)

        for class_id in range(num_classes):
            pred_is_class = prediction == class_id
            target_is_class = valid_target == class_id
            intersections[class_id] += float(np.logical_and(pred_is_class, target_is_class).sum())
            unions[class_id] += float(np.logical_or(pred_is_class, target_is_class).sum())
            prediction_counts[class_id] += float(pred_is_class.sum())
            target_counts[class_id] += float(target_is_class.sum())

    valid_iou_classes = unions > 0
    valid_dice_classes = (prediction_counts + target_counts) > 0
    ious = np.divide(
        intersections,
        unions,
        out=np.full(num_classes, np.nan, dtype=np.float64),
        where=valid_iou_classes,
    )
    dice_scores = np.divide(
        2 * intersections,
        prediction_counts + target_counts,
        out=np.full(num_classes, np.nan, dtype=np.float64),
        where=valid_dice_classes,
    )
    return {
        "mean_iou": float(np.nanmean(ious)) if np.any(valid_iou_classes) else 0.0,
        "mean_dice": float(np.nanmean(dice_scores)) if np.any(valid_dice_classes) else 0.0,
        "pixel_accuracy": float(correct_pixels / total_pixels) if total_pixels else 0.0,
    }


def run_majority_baseline(
    config: dict[str, Any],
    eval_split: str = "val",
    run_name: str = EXPERIMENT_ID,
    max_train_masks: int | None = None,
    max_eval_masks: int | None = None,
) -> dict[str, Any]:
    baseline_config = config.get("baseline", {})
    train_split = baseline_config.get("train_split", "train")
    max_train_masks = _coalesce_limit(max_train_masks, baseline_config.get("max_train_masks"))
    max_eval_masks = _coalesce_limit(max_eval_masks, baseline_config.get("max_eval_masks"))
    config.setdefault("runtime", {})["stage"] = "baseline"
    config["runtime"]["eval_split"] = eval_split
    config["runtime"]["max_train_masks"] = max_train_masks
    config["runtime"]["max_eval_masks"] = max_eval_masks
    output_dir, _ = prepare_run(config, stage="baseline", run_name=run_name)

    paths = config.get("paths", {})
    data_root = Path(paths.get("data_root", "data/raw/cityscapes"))
    reports_dir = Path(paths.get("reports_dir", "reports"))
    results_path = reports_dir / "experiment_results.csv"
    model_report_path = reports_dir / "model_report.xlsx"
    num_classes = int(config.get("model", {}).get("num_classes", len(CITYSCAPES_CLASSES)))

    majority_class = compute_majority_class(
        data_root,
        split=train_split,
        max_masks=max_train_masks,
    )
    metrics = compute_constant_prediction_metrics(
        targets=iter_train_id_masks(data_root, eval_split, max_masks=max_eval_masks),
        majority_class=majority_class,
        num_classes=num_classes,
    )
    result = {
        "experiment_id": EXPERIMENT_ID,
        "model": MODEL_NAME,
        "split": eval_split,
        "mean_iou": _format_metric(metrics["mean_iou"]),
        "mean_dice": _format_metric(metrics["mean_dice"]),
        "pixel_accuracy": _format_metric(metrics["pixel_accuracy"]),
        "majority_class_id": majority_class,
        "majority_class_name": CITYSCAPES_CLASSES[majority_class],
        "comments": _build_comment(majority_class, max_train_masks, max_eval_masks),
    }
    append_experiment_result(
        config=config,
        metrics=metrics,
        results_path=results_path,
        experiment_id=EXPERIMENT_ID,
        comments=result["comments"],
    )
    build_model_report(results_path=results_path, output_path=model_report_path)
    print(f"Output directory: {output_dir}")
    return result


def write_experiment_result(result: dict[str, Any], results_path: str | Path) -> Path:
    config = {
        "model": {"architecture": result.get("model", MODEL_NAME)},
        "loss": {"name": "constant_majority"},
        "optimizer": {"name": "none"},
        "scheduler": {"name": "none"},
    }
    comments = str(result.get("comments", ""))
    if result.get("split"):
        comments = f"{comments} Split: {result['split']}.".strip()
    return append_experiment_result(
        config=config,
        metrics={
            "mean_iou": result.get("mean_iou", ""),
            "mean_dice": result.get("mean_dice", ""),
            "pixel_accuracy": result.get("pixel_accuracy", ""),
        },
        results_path=results_path,
        experiment_id=str(result.get("experiment_id", EXPERIMENT_ID)),
        comments=comments,
    )


def iter_train_id_masks(
    root: str | Path,
    split: str,
    max_masks: int | None = None,
) -> Iterable[np.ndarray]:
    for index, mask_path in enumerate(_find_mask_paths(Path(root), split)):
        if max_masks is not None and index >= max_masks:
            break
        with Image.open(mask_path) as mask_image:
            label_mask = np.asarray(mask_image).copy()
        yield convert_label_ids_to_train_ids(label_mask)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    result = run_majority_baseline(
        config,
        eval_split=args.split,
        run_name=args.run_name,
        max_train_masks=_normalize_limit(args.max_train_masks),
        max_eval_masks=_normalize_limit(args.max_eval_masks),
    )
    print(
        f"{result['experiment_id']} {result['split']}: "
        f"mIoU={result['mean_iou']}, "
        f"mDice={result['mean_dice']}, "
        f"pixel_acc={result['pixel_accuracy']}"
    )


def _find_mask_paths(root: Path, split: str) -> list[Path]:
    return sorted((root / "gtFine" / split).glob("*/*_gtFine_labelIds.png"))


def _format_metric(value: float) -> str:
    return f"{value:.6f}"


def _normalize_limit(value: int | None) -> int | None:
    if value is None or value <= 0:
        return None
    return value


def _coalesce_limit(cli_value: int | None, config_value: Any) -> int | None:
    if cli_value is not None:
        return cli_value
    if config_value is None:
        return None
    return _normalize_limit(int(config_value))


def _build_comment(
    majority_class: int,
    max_train_masks: int | None,
    max_eval_masks: int | None,
) -> str:
    comment = f"Predicts `{CITYSCAPES_CLASSES[majority_class]}` for every valid pixel."
    if max_train_masks is not None or max_eval_masks is not None:
        comment += (
            f" Smoke-limited masks: train={max_train_masks or 'all'}, "
            f"eval={max_eval_masks or 'all'}."
        )
    return comment


if __name__ == "__main__":
    main()
