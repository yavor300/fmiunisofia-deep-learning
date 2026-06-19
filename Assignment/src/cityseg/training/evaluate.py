"""Checkpoint evaluation and qualitative error analysis."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.cityseg.config import load_config, prepare_run, save_resolved_config
from src.cityseg.constants import CITYSCAPES_CLASSES, IGNORE_INDEX
from src.cityseg.data.cityscapes_dataset import CityscapesDataset
from src.cityseg.data.label_mapping import (
    convert_label_ids_to_train_ids,
    decode_train_ids_to_colors,
)
from src.cityseg.data.transforms import create_transforms_from_config
from src.cityseg.models.factory import create_model
from src.cityseg.reporting.build_model_report import build_model_report
from src.cityseg.training.experiment_logger import append_experiment_result
from src.cityseg.training.metrics import compute_confusion_matrix, prepare_predictions
from src.cityseg.training.train import resolve_device

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a semantic segmentation model.")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to a YAML config file.",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a model checkpoint.")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional output run directory name.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="Number of qualitative examples to save.",
    )
    return parser.parse_args()


def evaluate_checkpoint(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    split: str = "val",
    run_name: str | None = None,
    max_examples: int = 5,
) -> tuple[Path, dict[str, float]]:
    checkpoint_path = Path(checkpoint_path)
    config.setdefault("runtime", {})["stage"] = "eval"
    config["runtime"]["checkpoint"] = str(checkpoint_path)
    config["runtime"]["split"] = split
    output_dir, _ = prepare_run(config, stage="eval", run_name=run_name)
    save_resolved_config(config, output_dir, filename="config.yaml")

    device = resolve_device(config.get("training", {}).get("device", "cuda"))
    model = create_model(config).to(device)
    checkpoint = _torch_load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = _build_eval_dataset(config, split)
    dataloader = DataLoader(
        dataset,
        batch_size=int(config.get("training", {}).get("batch_size", 1)),
        shuffle=False,
        num_workers=int(config.get("training", {}).get("num_workers", 0)),
        pin_memory=bool(config.get("training", {}).get("pin_memory", False)),
    )
    num_classes = int(config.get("model", {}).get("num_classes", len(CITYSCAPES_CLASSES)))
    ignore_index = int(config.get("loss", {}).get("ignore_index", IGNORE_INDEX))
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            predictions = prepare_predictions(logits)
            confusion_matrix += compute_confusion_matrix(
                predictions,
                masks,
                num_classes=num_classes,
                ignore_index=ignore_index,
            )

    metrics = _metrics_from_confusion_matrix(confusion_matrix)
    per_class = _per_class_iou_from_confusion_matrix(confusion_matrix)
    write_global_metrics(metrics, output_dir / "global_metrics.csv")
    write_per_class_iou(per_class, output_dir / "per_class_iou.csv")
    write_confusion_matrix(confusion_matrix.cpu(), output_dir / "confusion_matrix.csv")
    write_error_analysis(
        metrics,
        per_class,
        confusion_matrix.cpu(),
        output_dir / "error_analysis.md",
    )
    figures_dir = Path(config.get("paths", {}).get("reports_dir", "reports"))
    figures_dir = figures_dir / "figures" / output_dir.name
    save_prediction_examples(
        model=model,
        dataset=dataset,
        config=config,
        split=split,
        output_dir=figures_dir,
        device=device,
        max_examples=max_examples,
    )
    log_evaluation_experiment(
        config=config,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        split=split,
        metrics=metrics,
    )
    return output_dir, metrics


def log_evaluation_experiment(
    config: dict[str, Any],
    output_dir: Path,
    checkpoint_path: Path,
    split: str,
    metrics: dict[str, float],
) -> None:
    reports_dir = Path(config.get("paths", {}).get("reports_dir", "reports"))
    results_path = reports_dir / "experiment_results.csv"
    model_report_path = reports_dir / "model_report.xlsx"
    append_experiment_result(
        config=config,
        metrics=metrics,
        results_path=results_path,
        experiment_id=output_dir.name,
        checkpoint_path=checkpoint_path,
        comments=f"Evaluation run on the {split} split.",
    )
    build_model_report(results_path=results_path, output_path=model_report_path)


def write_global_metrics(metrics: dict[str, float], output_path: str | Path) -> Path:
    path = Path(output_path)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["mean_iou", "mean_dice", "pixel_accuracy"])
        writer.writeheader()
        writer.writerow({key: f"{metrics[key]:.6f}" for key in writer.fieldnames})
    return path


def write_per_class_iou(per_class: dict[str, float], output_path: str | Path) -> Path:
    path = Path(output_path)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["class_name", "iou"])
        writer.writeheader()
        for class_name, value in per_class.items():
            writer.writerow({"class_name": class_name, "iou": f"{value:.6f}"})
    return path


def write_confusion_matrix(matrix: torch.Tensor, output_path: str | Path) -> Path:
    path = Path(output_path)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["target/prediction", *CITYSCAPES_CLASSES[: matrix.shape[0]]])
        for class_name, row in zip(CITYSCAPES_CLASSES, matrix.tolist(), strict=False):
            writer.writerow([class_name, *row])
    return path


def write_error_analysis(
    metrics: dict[str, float],
    per_class_iou: dict[str, float],
    confusion_matrix: torch.Tensor,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    road_sidewalk = _class_confusion(confusion_matrix, "road", "sidewalk")
    small_objects = ["pole", "traffic sign", "traffic light", "rider"]
    rare_classes = sorted(per_class_iou.items(), key=lambda item: item[1])[:5]
    small_object_text = ", ".join(
        f"{name}={per_class_iou.get(name, float('nan')):.3f}" for name in small_objects
    )
    rare_text = ", ".join(f"{name}={value:.3f}" for name, value in rare_classes)
    content = "\n".join(
        [
            "# Evaluation Error Analysis",
            "",
            f"- Mean IoU: {metrics['mean_iou']:.4f}",
            f"- Mean Dice: {metrics['mean_dice']:.4f}",
            f"- Pixel accuracy: {metrics['pixel_accuracy']:.4f}",
            "",
            "## Common Error Checks",
            "",
            f"- Road/sidewalk confusion pixels: {road_sidewalk}",
            f"- Small-object IoU snapshot: {small_object_text}",
            f"- Lowest-IoU classes: {rare_text}",
            "- Boundary and distant-object errors should be inspected in the saved error maps.",
            "- Rare classes with low IoU should be prioritized in qualitative review.",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")
    return path


def save_prediction_examples(
    model: torch.nn.Module,
    dataset: CityscapesDataset,
    config: dict[str, Any],
    split: str,
    output_dir: str | Path,
    device: torch.device,
    max_examples: int,
) -> None:
    if max_examples <= 0:
        return

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    for index in range(min(max_examples, len(dataset))):
        image_path, mask_path = dataset.samples[index]
        image_tensor, target_tensor = dataset[index]
        with torch.no_grad():
            prediction = model(image_tensor.unsqueeze(0).to(device)).argmax(dim=1)[0].cpu().numpy()

        original = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        raw_target = np.asarray(Image.open(mask_path)).copy()
        target = convert_label_ids_to_train_ids(raw_target)
        target_color = decode_train_ids_to_colors(target)
        prediction_color = decode_train_ids_to_colors(prediction.astype(np.uint8))
        overlay = _overlay(original, prediction_color)
        error_map = _error_map(target, prediction)

        prefix = path / f"example_{index:03d}"
        Image.fromarray(original).save(prefix.with_name(prefix.name + "_original.png"))
        Image.fromarray(target_color).save(prefix.with_name(prefix.name + "_ground_truth.png"))
        Image.fromarray(prediction_color).save(prefix.with_name(prefix.name + "_prediction.png"))
        Image.fromarray(overlay).save(prefix.with_name(prefix.name + "_overlay.png"))
        Image.fromarray(error_map).save(prefix.with_name(prefix.name + "_error_map.png"))
        _save_panel(
            [original, target_color, prediction_color, overlay, error_map],
            ["Original", "Ground truth", "Prediction", "Overlay", "Error map"],
            prefix.with_name(prefix.name + "_panel.png"),
        )


def main() -> None:
    args = parse_args()
    output_dir, metrics = evaluate_checkpoint(
        config=load_config(args.config),
        checkpoint_path=args.checkpoint,
        split=args.split,
        run_name=args.run_name,
        max_examples=args.max_examples,
    )
    print(f"Evaluation output directory: {output_dir}")
    print(
        f"mean_iou={metrics['mean_iou']:.6f}, "
        f"mean_dice={metrics['mean_dice']:.6f}, "
        f"pixel_accuracy={metrics['pixel_accuracy']:.6f}"
    )


def _build_eval_dataset(config: dict[str, Any], split: str) -> CityscapesDataset:
    paths = config.get("paths", {})
    transforms = create_transforms_from_config(config, split="val" if split != "train" else "train")
    return CityscapesDataset(
        root=paths.get("data_root", "data/raw/cityscapes"),
        split=split,
        transforms=transforms,
    )


def _metrics_from_confusion_matrix(matrix: torch.Tensor) -> dict[str, float]:
    matrix = matrix.to(dtype=torch.float32)
    intersection = torch.diag(matrix)
    target_count = matrix.sum(dim=1)
    prediction_count = matrix.sum(dim=0)
    union = target_count + prediction_count - intersection
    iou = torch.where(union > 0, intersection / union.clamp_min(1.0), torch.nan)
    dice_denominator = target_count + prediction_count
    dice = torch.where(
        dice_denominator > 0,
        2.0 * intersection / dice_denominator.clamp_min(1.0),
        torch.nan,
    )
    total = matrix.sum()
    correct = intersection.sum()
    return {
        "mean_iou": _nanmean(iou),
        "mean_dice": _nanmean(dice),
        "pixel_accuracy": float(correct / total) if total > 0 else 0.0,
    }


def _per_class_iou_from_confusion_matrix(matrix: torch.Tensor) -> dict[str, float]:
    matrix = matrix.to(dtype=torch.float32)
    intersection = torch.diag(matrix)
    union = matrix.sum(dim=1) + matrix.sum(dim=0) - intersection
    iou = torch.where(union > 0, intersection / union.clamp_min(1.0), torch.nan)
    return {
        class_name: (0.0 if torch.isnan(value) else float(value))
        for class_name, value in zip(CITYSCAPES_CLASSES, iou, strict=False)
    }


def _class_confusion(matrix: torch.Tensor, class_a: str, class_b: str) -> int:
    class_to_index = {name: index for index, name in enumerate(CITYSCAPES_CLASSES)}
    index_a = class_to_index[class_a]
    index_b = class_to_index[class_b]
    return int(matrix[index_a, index_b] + matrix[index_b, index_a])


def _overlay(original: np.ndarray, mask_color: np.ndarray) -> np.ndarray:
    if original.shape[:2] != mask_color.shape[:2]:
        mask_color = np.asarray(Image.fromarray(mask_color).resize(original.shape[1::-1]))
    blended = original.astype(np.float32) * 0.6 + mask_color.astype(np.float32) * 0.4
    return np.clip(blended, 0, 255).astype(np.uint8)


def _error_map(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    if target.shape != prediction.shape:
        prediction = Image.fromarray(prediction.astype(np.uint8)).resize(target.shape[1::-1])
        prediction = np.asarray(prediction)
    valid = target != IGNORE_INDEX
    error = valid & (target != prediction)
    image = np.zeros((*target.shape, 3), dtype=np.uint8)
    image[valid & ~error] = (40, 180, 80)
    image[error] = (220, 40, 40)
    return image


def _save_panel(images: list[np.ndarray], titles: list[str], output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
    for axis, image, title in zip(axes, images, titles, strict=True):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _nanmean(values: torch.Tensor) -> float:
    valid = values[~torch.isnan(values)]
    if valid.numel() == 0:
        return 0.0
    return float(valid.mean())


def _torch_load(path: Path, map_location: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


if __name__ == "__main__":
    main()
