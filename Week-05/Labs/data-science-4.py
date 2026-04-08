from __future__ import annotations

import argparse
import copy
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


@dataclass(frozen=True)
class DetectionSample:
    image_path: Path
    label_path: Path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float, image_size: int) -> tuple[float, float, float, float]:
    x1 = (cx - w / 2.0) * image_size
    y1 = (cy - h / 2.0) * image_size
    x2 = (cx + w / 2.0) * image_size
    y2 = (cy + h / 2.0) * image_size
    x1 = max(0.0, min(float(image_size - 1), x1))
    y1 = max(0.0, min(float(image_size - 1), y1))
    x2 = max(0.0, min(float(image_size - 1), x2))
    y2 = max(0.0, min(float(image_size - 1), y2))
    return x1, y1, x2, y2


def parse_yolo_label_file(label_path: Path, image_size: int) -> list[tuple[int, list[float]]]:
    entries: list[tuple[int, list[float]]] = []
    text = label_path.read_text().strip()
    if not text:
        return entries

    for line in text.splitlines():
        tokens = line.strip().split()
        if len(tokens) != 5:
            continue
        class_id = int(float(tokens[0]))
        cx, cy, w, h = map(float, tokens[1:])
        x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, image_size)
        if x2 <= x1 or y2 <= y1:
            continue
        entries.append((class_id, [x1, y1, x2, y2]))

    return entries


class CarDetectionDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        image_size: int,
        train_mode: bool,
        keep_only_car: bool = True,
        max_samples: int | None = None,
        seed: int = 42,
    ):
        self.image_size = image_size
        self.train_mode = train_mode
        self.keep_only_car = keep_only_car
        self.color_jitter = torch.nn.Identity()

        image_files = sorted([p for p in images_dir.glob("*.jpg") if p.is_file()])
        samples: list[DetectionSample] = []
        for image_path in image_files:
            label_path = labels_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                samples.append(DetectionSample(image_path=image_path, label_path=label_path))

        if max_samples is not None and max_samples < len(samples):
            rng = random.Random(seed)
            rng.shuffle(samples)
            samples = samples[:max_samples]

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size))

        parsed_entries = parse_yolo_label_file(sample.label_path, image_size=self.image_size)
        boxes: list[list[float]] = []
        labels: list[int] = []

        for class_id, box in parsed_entries:
            if self.keep_only_car and class_id != 0:
                continue
            boxes.append(box)
            labels.append(1)

        if self.train_mode and random.random() < 0.5:
            image = TF.hflip(image)
            flipped_boxes: list[list[float]] = []
            for x1, y1, x2, y2 in boxes:
                new_x1 = self.image_size - x2
                new_x2 = self.image_size - x1
                flipped_boxes.append([new_x1, y1, new_x2, y2])
            boxes = flipped_boxes

        image_tensor = TF.to_tensor(image)

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        if boxes_tensor.numel() == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)

        if labels_tensor.numel() == 0:
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
        iscrowd = torch.zeros((labels_tensor.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
        }
        return image_tensor, target


def detection_collate(batch):
    return tuple(zip(*batch))


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, (xa2 - xa1) * (ya2 - ya1))
    area_b = max(0.0, (xb2 - xb1) * (yb2 - yb1))
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def match_predictions(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float,
) -> tuple[int, int, int, list[float]]:
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes), []

    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]

    matched_gt: set[int] = set()
    tp = 0
    fp = 0
    matched_ious: list[float] = []

    for pred_box in pred_boxes:
        best_iou = 0.0
        best_gt = -1
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_idx

        if best_gt >= 0 and best_iou >= iou_threshold:
            matched_gt.add(best_gt)
            tp += 1
            matched_ious.append(best_iou)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn, matched_ious


def evaluate_detector(
    model,
    loader: DataLoader,
    device: torch.device,
    score_threshold: float,
    iou_threshold: float,
    description: str,
) -> dict[str, float]:
    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_ious: list[float] = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc=description, leave=False):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                gt_boxes = target["boxes"].cpu().numpy()

                pred_scores = output["scores"].detach().cpu().numpy()
                pred_boxes = output["boxes"].detach().cpu().numpy()
                keep_mask = pred_scores >= score_threshold

                pred_boxes = pred_boxes[keep_mask]
                pred_scores = pred_scores[keep_mask]

                tp, fp, fn, matched_ious = match_predictions(
                    pred_boxes=pred_boxes,
                    pred_scores=pred_scores,
                    gt_boxes=gt_boxes,
                    iou_threshold=iou_threshold,
                )
                total_tp += tp
                total_fp += fp
                total_fn += fn
                all_ious.extend(matched_ious)

    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_iou": mean_iou,
    }


def train_detector(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    score_threshold: float,
    iou_threshold: float,
    model_name: str,
) -> tuple[object, dict[str, float], dict[str, list[float]]]:
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate)

    best_state = copy.deepcopy(model.state_dict())
    best_val_f1 = -1.0
    best_val_metrics: dict[str, float] = {}
    train_losses: list[float] = []
    val_f1_scores: list[float] = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, targets in tqdm(train_loader, desc=f"{model_name} train {epoch + 1}/{epochs}", leave=False):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())
            total_loss.backward()
            optimizer.step()

            running_loss += float(total_loss.item())

        avg_train_loss = running_loss / max(1, len(train_loader))
        val_metrics = evaluate_detector(
            model=model,
            loader=val_loader,
            device=device,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            description=f"{model_name} val",
        )

        print(
            f"[{model_name}] epoch {epoch + 1}/{epochs} | "
            f"train loss={avg_train_loss:.4f} | "
            f"val precision={val_metrics['precision']:.4f} "
            f"val recall={val_metrics['recall']:.4f} "
            f"val f1={val_metrics['f1']:.4f}"
        )
        train_losses.append(avg_train_loss)
        val_f1_scores.append(val_metrics["f1"])

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = copy.deepcopy(model.state_dict())
            best_val_metrics = val_metrics

    model.load_state_dict(best_state)
    history = {
        "train_loss": train_losses,
        "val_f1": val_f1_scores,
    }
    return model, best_val_metrics, history


def build_rcnn_like_model(num_classes: int, pretrained: bool):
    if pretrained:
        try:
            model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
        except Exception:
            model = fasterrcnn_mobilenet_v3_large_fpn(weights=None)
    else:
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def build_faster_rcnn_model(num_classes: int, pretrained: bool):
    if pretrained:
        try:
            model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        except Exception:
            model = fasterrcnn_resnet50_fpn(weights=None)
    else:
        model = fasterrcnn_resnet50_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def collect_raw_train_distribution(labels_dir: Path) -> Counter:
    counter: Counter = Counter()
    for label_path in labels_dir.glob("*.txt"):
        text = label_path.read_text().strip()
        if not text:
            continue
        for line in text.splitlines():
            class_id = int(float(line.strip().split()[0]))
            counter[class_id] += 1
    return counter


def save_train_distribution_plot(counter: Counter, output_path: Path) -> None:
    class_names = ["car", "background"]
    values = [counter.get(0, 0), counter.get(1, 0)]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(class_names, values, color=["#2a9d8f", "#e76f51"])
    ax.set_title("Training Class Distribution (Bounding Boxes)")
    ax.set_ylabel("Box count")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def image_paths_by_class(labels_dir: Path, images_dir: Path) -> dict[int, list[Path]]:
    class_to_paths: dict[int, list[Path]] = defaultdict(list)

    for label_path in sorted(labels_dir.glob("*.txt")):
        text = label_path.read_text().strip()
        if not text:
            continue
        classes_in_image = {int(float(line.split()[0])) for line in text.splitlines() if line.strip()}
        image_path = images_dir / f"{label_path.stem}.jpg"
        if not image_path.exists():
            continue
        for class_id in classes_in_image:
            class_to_paths[class_id].append(image_path)

    return class_to_paths


def draw_gt_boxes(ax, image_path: Path, label_path: Path, image_size: int, target_class: int) -> None:
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    ax.imshow(image)
    ax.axis("off")

    entries = parse_yolo_label_file(label_path, image_size=image_size)
    for class_id, box in entries:
        if class_id != target_class:
            continue
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)


def save_class_examples_plot(
    labels_dir: Path,
    images_dir: Path,
    output_path: Path,
    image_size: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    by_class = image_paths_by_class(labels_dir=labels_dir, images_dir=images_dir)

    classes = [(0, "car"), (1, "background")]
    fig, axes = plt.subplots(len(classes), 5, figsize=(14, 6.5))

    for row, (class_id, class_name) in enumerate(classes):
        candidates = by_class.get(class_id, [])[:]
        rng.shuffle(candidates)
        selected = candidates[:5]

        for col in range(5):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(selected):
                image_path = selected[col]
                label_path = labels_dir / f"{image_path.stem}.txt"
                draw_gt_boxes(ax, image_path, label_path, image_size=image_size, target_class=class_id)
            if col == 0:
                ax.set_title(class_name)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def save_bbox_area_histogram(labels_dir: Path, output_path: Path) -> None:
    car_areas: list[float] = []
    bg_areas: list[float] = []

    for label_path in labels_dir.glob("*.txt"):
        text = label_path.read_text().strip()
        if not text:
            continue
        for line in text.splitlines():
            class_id, _, _, w, h = map(float, line.split())
            area = w * h
            if int(class_id) == 0:
                car_areas.append(area)
            elif int(class_id) == 1:
                bg_areas.append(area)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if car_areas:
        ax.hist(car_areas, bins=30, alpha=0.7, label="car")
    if bg_areas:
        ax.hist(bg_areas, bins=30, alpha=0.7, label="background")

    ax.set_title("Bounding Box Area Distribution (normalized)")
    ax.set_xlabel("box area (width * height)")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def format_metric_with_change(value: float, baseline: float) -> str:
    if baseline == 0:
        return f"{value:.4f} (n/a)"
    delta_pct = ((value - baseline) / baseline) * 100.0
    sign = "+" if delta_pct >= 0 else ""
    return f"{value:.4f} ({sign}{delta_pct:.2f}%)"


def save_training_curves(
    histories: dict[str, dict[str, list[float]]],
    output_loss_path: Path,
    output_metric_path: Path,
) -> None:
    model_names = list(histories.keys())
    if not model_names:
        return

    loss_fig, loss_axes = plt.subplots(len(model_names), 1, figsize=(8, 3.2 * len(model_names)))
    metric_fig, metric_axes = plt.subplots(len(model_names), 1, figsize=(8, 3.2 * len(model_names)))

    if len(model_names) == 1:
        loss_axes = [loss_axes]
        metric_axes = [metric_axes]

    for idx, model_name in enumerate(model_names):
        history = histories[model_name]
        train_loss_values = history.get("train_loss", [])
        val_f1_values = history.get("val_f1", [])
        epochs = range(1, len(train_loss_values) + 1)

        loss_ax = loss_axes[idx]
        if len(train_loss_values) == 0:
            loss_ax.text(
                0.5,
                0.5,
                "Loaded from checkpoint.\nTraining skipped in this run.",
                ha="center",
                va="center",
                fontsize=10,
            )
            loss_ax.set_title(f"{model_name} - Train Loss")
            loss_ax.set_xticks([])
            loss_ax.set_yticks([])
        else:
            loss_ax.plot(epochs, train_loss_values, label="train loss")
            loss_ax.set_title(f"{model_name} - Train Loss")
            loss_ax.set_xlabel("Epoch")
            loss_ax.set_ylabel("Loss")
            loss_ax.grid(True, alpha=0.25)
            loss_ax.legend()

        metric_ax = metric_axes[idx]
        if len(val_f1_values) == 0:
            metric_ax.text(
                0.5,
                0.5,
                "Loaded from checkpoint.\nValidation history unavailable in this run.",
                ha="center",
                va="center",
                fontsize=10,
            )
            metric_ax.set_title(f"{model_name} - Validation F1")
            metric_ax.set_xticks([])
            metric_ax.set_yticks([])
        else:
            metric_ax.plot(range(1, len(val_f1_values) + 1), val_f1_values, label="validation F1")
            metric_ax.set_title(f"{model_name} - Validation F1")
            metric_ax.set_xlabel("Epoch")
            metric_ax.set_ylabel("F1")
            metric_ax.grid(True, alpha=0.25)
            metric_ax.legend()

    loss_fig.tight_layout()
    loss_fig.savefig(output_loss_path, dpi=180)
    plt.close(loss_fig)

    metric_fig.tight_layout()
    metric_fig.savefig(output_metric_path, dpi=180)
    plt.close(metric_fig)


def save_model_report(
    output_path: Path,
    model_order: list[str],
    metrics_before: dict[str, dict[str, float]],
    metrics_after: dict[str, dict[str, float]],
    model_hyperparams: dict[str, dict[str, str]],
    yolo_note: str,
    loss_plot_name: str,
    metric_plot_name: str,
) -> None:
    baseline_metrics = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }

    best_model_name = max(model_order, key=lambda name: metrics_before[name]["f1"])
    best_model_f1 = metrics_before[best_model_name]["f1"]

    lines = [
        "# Model Report - Task 04",
        "",
        f"Best model: **{best_model_name}** because it has the highest test F1 score ({best_model_f1:.4f}).",
        "",
        "## Context",
        f"- {yolo_note}",
        "",
        "## Main Experiment Table",
        "Rows are kept in experiment order. First row is the baseline model.",
        "",
        "| Hypothesis | Architecture | Epochs | Batch Size | Learning Rate | Optimizer | Test Precision (vs baseline) | Test Recall (vs baseline) | Test F1 (vs baseline) | Comments |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |",
    ]

    lines.append(
        "| "
        + " | ".join(
            [
                "baseline_no_detection",
                "Predict no objects",
                "n/a",
                "n/a",
                "n/a",
                "n/a",
                format_metric_with_change(baseline_metrics["precision"], baseline_metrics["precision"]),
                format_metric_with_change(baseline_metrics["recall"], baseline_metrics["recall"]),
                format_metric_with_change(baseline_metrics["f1"], baseline_metrics["f1"]),
                "Greedy statistical baseline for detection.",
            ]
        )
        + " |"
    )

    for model_name in model_order:
        values = metrics_before[model_name]
        reloaded = metrics_after.get(model_name, values)
        params = model_hyperparams[model_name]
        delta_reload = abs(values["f1"] - reloaded["f1"])
        reload_comment = "Reload stable." if delta_reload <= 1e-6 else f"Reload delta F1={delta_reload:.6f}."
        mean_iou_note = f"Mean IoU={values.get('mean_iou', 0.0):.4f}."

        lines.append(
            "| "
            + " | ".join(
                [
                    model_name,
                    params["architecture"],
                    params["epochs"],
                    params["batch_size"],
                    params["learning_rate"],
                    params["optimizer"],
                    format_metric_with_change(values["precision"], baseline_metrics["precision"]),
                    format_metric_with_change(values["recall"], baseline_metrics["recall"]),
                    format_metric_with_change(values["f1"], baseline_metrics["f1"]),
                    f"{mean_iou_note} {reload_comment}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Diagrams",
            f"- Train vs validation loss: `{loss_plot_name}`",
            f"- Train vs validation main metric (F1): `{metric_plot_name}`",
            "",
            "## Notes",
            "- Metrics include value and percentage change vs baseline.",
            "- Table is intentionally not sorted.",
            "- Best row should be highlighted/bolded when moved to Excel.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_torchvision_inference(model, image_path: Path, image_size: int, device: torch.device, score_thr: float):
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    tensor = TF.to_tensor(image).to(device)

    model.eval()
    with torch.no_grad():
        output = model([tensor])[0]

    scores = output["scores"].detach().cpu().numpy()
    boxes = output["boxes"].detach().cpu().numpy()
    keep = scores >= score_thr
    return image, boxes[keep], scores[keep]


def run_yolo_inference(yolo_model, image_path: Path, image_size: int, score_thr: float):
    results = yolo_model.predict(source=str(image_path), imgsz=image_size, conf=score_thr, verbose=False)
    result = results[0]

    image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    if result.boxes is None or len(result.boxes) == 0:
        return image, np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    return image, boxes, scores


def draw_predictions(ax, image: Image.Image, boxes: np.ndarray, scores: np.ndarray, title: str) -> None:
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(title, fontsize=9)

    for box, score in zip(boxes[:8], scores[:8]):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.8, edgecolor="yellow", facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1, f"{score:.2f}", fontsize=8, color="black", bbox=dict(facecolor="yellow", alpha=0.75))


def save_qualitative_grid(
    test_image_paths: list[Path],
    loaded_models: dict[str, object],
    yolo_model,
    output_path: Path,
    image_size: int,
    device: torch.device,
    score_threshold: float,
) -> None:
    model_names = list(loaded_models.keys())
    if yolo_model is not None:
        model_names.append("yolo")

    fig, axes = plt.subplots(len(test_image_paths), len(model_names), figsize=(4.8 * len(model_names), 3.8 * len(test_image_paths)))
    if len(test_image_paths) == 1:
        axes = np.array([axes])

    for row, image_path in enumerate(test_image_paths):
        for col, model_name in enumerate(model_names):
            ax = axes[row, col]
            if model_name == "yolo":
                image, boxes, scores = run_yolo_inference(
                    yolo_model=yolo_model,
                    image_path=image_path,
                    image_size=image_size,
                    score_thr=score_threshold,
                )
            else:
                image, boxes, scores = run_torchvision_inference(
                    model=loaded_models[model_name],
                    image_path=image_path,
                    image_size=image_size,
                    device=device,
                    score_thr=score_threshold,
                )

            draw_predictions(ax, image=image, boxes=boxes, scores=scores, title=f"{image_path.name}\n{model_name}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def write_yolo_data_yaml(output_path: Path, dataset_root: Path) -> None:
    lines = [
        f"path: {dataset_root}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        "  0: car",
        "  1: background",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 04 - Car detection with R-CNN/Faster R-CNN/YOLO")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--image-size", type=int, default=416)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument(
        "--use-saved-models",
        action="store_true",
        help="Load existing .pt checkpoints when available and skip training for those models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    root = get_project_root()
    data_root = root / "DATA" / "car_bee_detection" / "datasets"
    output_root = Path(__file__).resolve().parent

    train_images = data_root / "images" / "train"
    val_images = data_root / "images" / "val"
    test_images = data_root / "images" / "test"

    train_labels = data_root / "labels" / "train"
    val_labels = data_root / "labels" / "val"
    test_labels = data_root / "labels" / "test"

    save_class_examples_plot(
        labels_dir=train_labels,
        images_dir=train_images,
        output_path=output_root / "data-science-4-eda-class-samples.png",
        image_size=args.image_size,
        seed=args.seed,
    )
    distribution = collect_raw_train_distribution(train_labels)
    save_train_distribution_plot(
        counter=distribution,
        output_path=output_root / "data-science-4-eda-class-distribution.png",
    )
    save_bbox_area_histogram(
        labels_dir=train_labels,
        output_path=output_root / "data-science-4-eda-bbox-area-histogram.png",
    )

    train_dataset = CarDetectionDataset(
        images_dir=train_images,
        labels_dir=train_labels,
        image_size=args.image_size,
        train_mode=True,
        keep_only_car=True,
        max_samples=args.max_train_samples,
        seed=args.seed,
    )
    val_dataset = CarDetectionDataset(
        images_dir=val_images,
        labels_dir=val_labels,
        image_size=args.image_size,
        train_mode=False,
        keep_only_car=True,
        max_samples=args.max_val_samples,
        seed=args.seed,
    )
    test_dataset = CarDetectionDataset(
        images_dir=test_images,
        labels_dir=test_labels,
        image_size=args.image_size,
        train_mode=False,
        keep_only_car=True,
        max_samples=args.max_test_samples,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=detection_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detection_collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detection_collate,
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    print(f"Train/val/test sizes: {len(train_dataset)} / {len(val_dataset)} / {len(test_dataset)}")

    num_classes = 2
    models_to_train = {
        "rcnn_like": build_rcnn_like_model(num_classes=num_classes, pretrained=not args.no_pretrained),
        "faster_rcnn": build_faster_rcnn_model(num_classes=num_classes, pretrained=not args.no_pretrained),
    }

    metrics_before: dict[str, dict[str, float]] = {}
    metrics_after: dict[str, dict[str, float]] = {}
    loaded_models: dict[str, object] = {}
    histories: dict[str, dict[str, list[float]]] = {}
    model_hyperparams: dict[str, dict[str, str]] = {}
    model_order = ["rcnn_like", "faster_rcnn"]

    for model_name, model in models_to_train.items():
        model_path = output_root / f"data-science-4-{model_name}.pt"
        model = model.to(device)

        if args.use_saved_models and model_path.exists():
            print(f"\n===== Loading saved {model_name} from {model_path} (training skipped) =====")
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["state_dict"])
            history = {"train_loss": [], "val_f1": []}
        else:
            if args.use_saved_models and not model_path.exists():
                print(f"\n===== No saved checkpoint for {model_name}. Training now =====")
            else:
                print(f"\n===== Training {model_name} =====")

            model, _, history = train_detector(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                score_threshold=args.score_threshold,
                iou_threshold=args.iou_threshold,
                model_name=model_name,
            )
            torch.save({"model_name": model_name, "state_dict": model.state_dict()}, model_path)

        histories[model_name] = history

        test_metrics = evaluate_detector(
            model=model,
            loader=test_loader,
            device=device,
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold,
            description=f"{model_name} test",
        )
        metrics_before[model_name] = test_metrics
        print(f"[{model_name}] test metrics: {test_metrics}")

        if model_name == "rcnn_like":
            reloaded = build_rcnn_like_model(num_classes=num_classes, pretrained=False)
        else:
            reloaded = build_faster_rcnn_model(num_classes=num_classes, pretrained=False)

        checkpoint = torch.load(model_path, map_location=device)
        reloaded.load_state_dict(checkpoint["state_dict"])
        reloaded = reloaded.to(device)

        reloaded_metrics = evaluate_detector(
            model=reloaded,
            loader=test_loader,
            device=device,
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold,
            description=f"{model_name} reloaded test",
        )
        metrics_after[model_name] = reloaded_metrics
        loaded_models[model_name] = reloaded
        print(f"[{model_name}] reloaded test metrics: {reloaded_metrics}")
        model_hyperparams[model_name] = {
            "architecture": "FasterRCNN-MobileNetV3" if model_name == "rcnn_like" else "FasterRCNN-ResNet50",
            "epochs": str(args.epochs) if len(history.get("train_loss", [])) > 0 else "loaded checkpoint",
            "batch_size": str(args.batch_size),
            "learning_rate": f"{args.learning_rate}",
            "optimizer": "AdamW",
        }

    yolo_note = "YOLO not executed."
    loaded_yolo = None

    if not args.skip_yolo and YOLO is not None:
        yolo_yaml = output_root / "data-science-4-yolo-dataset.yaml"
        write_yolo_data_yaml(yolo_yaml, dataset_root=data_root)
        final_path = output_root / "data-science-4-yolo.pt"
        device_arg = 0 if device.type == "cuda" else "cpu"
        model_order.append("yolo")
        model_hyperparams["yolo"] = {
            "architecture": "YOLOv8n",
            "epochs": str(args.epochs),
            "batch_size": str(args.batch_size),
            "learning_rate": "internal (ultralytics default)",
            "optimizer": "internal (ultralytics default)",
        }

        if args.use_saved_models and final_path.exists():
            print(f"\n===== Loading saved yolo from {final_path} (training skipped) =====")
            loaded_yolo = YOLO(str(final_path))
            test_result = loaded_yolo.val(
                data=str(yolo_yaml),
                split="test",
                imgsz=args.image_size,
                batch=args.batch_size,
                device=device_arg,
                verbose=False,
            )
            metrics_before["yolo"] = {
                "precision": float(getattr(test_result.box, "mp", 0.0)),
                "recall": float(getattr(test_result.box, "mr", 0.0)),
                "f1": float((2 * getattr(test_result.box, "mp", 0.0) * getattr(test_result.box, "mr", 0.0)) / max(1e-8, getattr(test_result.box, "mp", 0.0) + getattr(test_result.box, "mr", 0.0))),
                "mean_iou": float(getattr(test_result.box, "map50", 0.0)),
            }

            reloaded_yolo = YOLO(str(final_path))
            test_result_loaded = reloaded_yolo.val(
                data=str(yolo_yaml),
                split="test",
                imgsz=args.image_size,
                batch=args.batch_size,
                device=device_arg,
                verbose=False,
            )
            metrics_after["yolo"] = {
                "precision": float(getattr(test_result_loaded.box, "mp", 0.0)),
                "recall": float(getattr(test_result_loaded.box, "mr", 0.0)),
                "f1": float((2 * getattr(test_result_loaded.box, "mp", 0.0) * getattr(test_result_loaded.box, "mr", 0.0)) / max(1e-8, getattr(test_result_loaded.box, "mp", 0.0) + getattr(test_result_loaded.box, "mr", 0.0))),
                "mean_iou": float(getattr(test_result_loaded.box, "map50", 0.0)),
            }
            loaded_yolo = reloaded_yolo
            yolo_note = "YOLO loaded from existing checkpoint and evaluated."
            model_hyperparams["yolo"]["epochs"] = "loaded checkpoint"
        else:
            if args.use_saved_models and not final_path.exists():
                print("\n===== No saved yolo checkpoint found. Training yolo =====")
            else:
                print("\n===== Training yolo =====")

            yolo_init_note = "YOLO initialized from yolov8n.yaml (random weights)."
            if args.no_pretrained:
                yolo_model = YOLO("yolov8n.yaml")
            else:
                try:
                    yolo_model = YOLO("yolov8n.pt")
                    yolo_init_note = "YOLO initialized from yolov8n.pt (transfer learning)."
                except Exception:
                    yolo_model = YOLO("yolov8n.yaml")

            yolo_result = yolo_model.train(
                data=str(yolo_yaml),
                epochs=args.epochs,
                imgsz=args.image_size,
                batch=args.batch_size,
                project=str(output_root),
                name="data-science-4-yolo-runs",
                exist_ok=True,
                device=device_arg,
                verbose=False,
            )
            best_path = Path(yolo_result.save_dir) / "weights" / "best.pt"
            if best_path.exists():
                shutil.copy2(best_path, final_path)
            else:
                yolo_model.save(str(final_path))

            test_result = yolo_model.val(
                data=str(yolo_yaml),
                split="test",
                imgsz=args.image_size,
                batch=args.batch_size,
                device=device_arg,
                verbose=False,
            )
            metrics_before["yolo"] = {
                "precision": float(getattr(test_result.box, "mp", 0.0)),
                "recall": float(getattr(test_result.box, "mr", 0.0)),
                "f1": float((2 * getattr(test_result.box, "mp", 0.0) * getattr(test_result.box, "mr", 0.0)) / max(1e-8, getattr(test_result.box, "mp", 0.0) + getattr(test_result.box, "mr", 0.0))),
                "mean_iou": float(getattr(test_result.box, "map50", 0.0)),
            }

            loaded_yolo = YOLO(str(final_path))
            test_result_loaded = loaded_yolo.val(
                data=str(yolo_yaml),
                split="test",
                imgsz=args.image_size,
                batch=args.batch_size,
                device=device_arg,
                verbose=False,
            )
            metrics_after["yolo"] = {
                "precision": float(getattr(test_result_loaded.box, "mp", 0.0)),
                "recall": float(getattr(test_result_loaded.box, "mr", 0.0)),
                "f1": float((2 * getattr(test_result_loaded.box, "mp", 0.0) * getattr(test_result_loaded.box, "mr", 0.0)) / max(1e-8, getattr(test_result_loaded.box, "mp", 0.0) + getattr(test_result_loaded.box, "mr", 0.0))),
                "mean_iou": float(getattr(test_result_loaded.box, "map50", 0.0)),
            }
            yolo_note = f"YOLO executed with ultralytics YOLOv8n architecture. {yolo_init_note}"

    else:
        if args.skip_yolo:
            yolo_note = "YOLO branch skipped via --skip-yolo."
        else:
            yolo_note = "YOLO branch skipped because ultralytics is unavailable in the environment."

    rng = random.Random(args.seed)
    test_paths = [sample.image_path for sample in test_dataset.samples]
    rng.shuffle(test_paths)
    selected_test_paths = test_paths[:5]

    if selected_test_paths:
        save_qualitative_grid(
            test_image_paths=selected_test_paths,
            loaded_models=loaded_models,
            yolo_model=loaded_yolo,
            output_path=output_root / "data-science-4-model-comparison.png",
            image_size=args.image_size,
            device=device,
            score_threshold=args.score_threshold,
        )

    loss_curve_path = output_root / "data-science-4-train-vs-val-loss.png"
    metric_curve_path = output_root / "data-science-4-train-vs-val-metric.png"
    save_training_curves(
        histories=histories,
        output_loss_path=loss_curve_path,
        output_metric_path=metric_curve_path,
    )

    save_model_report(
        output_path=output_root / "data-science-4-model-report.md",
        model_order=model_order,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        model_hyperparams=model_hyperparams,
        yolo_note=yolo_note,
        loss_plot_name=loss_curve_path.name,
        metric_plot_name=metric_curve_path.name,
    )

    print("\nTask 04 completed.")
    print(f"Artifacts saved to: {output_root}")


if __name__ == "__main__":
    main()
