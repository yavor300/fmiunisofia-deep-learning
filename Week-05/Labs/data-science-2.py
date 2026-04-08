from __future__ import annotations

import argparse
import copy
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm


INCEPTION_MIN_IMAGE_SIZE = 299


@dataclass(frozen=True)
class ImageSample:
    path: Path
    label: int
    class_name: str


class BrainTumorDataset(Dataset):
    def __init__(self, samples: list[ImageSample], transform: transforms.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        image = Image.open(sample.path).convert("RGB")
        return self.transform(image), sample.label


class CustomTumorCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def maybe_limit_samples(samples: list[ImageSample], limit: int | None, seed: int) -> list[ImageSample]:
    if limit is None or limit >= len(samples):
        return samples
    rng = random.Random(seed)
    sampled = samples[:]
    rng.shuffle(sampled)
    return sampled[:limit]


def collect_samples_from_split(split_dir: Path, class_to_idx: dict[str, int]) -> list[ImageSample]:
    samples: list[ImageSample] = []
    for class_name, label in class_to_idx.items():
        class_dir = split_dir / class_name
        for image_path in sorted(class_dir.glob("*")):
            if image_path.is_file():
                samples.append(ImageSample(path=image_path, label=label, class_name=class_name))
    return samples


def split_testing_into_validation_and_test(
    testing_dir: Path,
    class_to_idx: dict[str, int],
    validation_fraction: float,
    seed: int,
) -> tuple[list[ImageSample], list[ImageSample]]:
    rng = random.Random(seed)
    validation_samples: list[ImageSample] = []
    test_samples: list[ImageSample] = []

    for class_name, label in class_to_idx.items():
        image_paths = [p for p in sorted((testing_dir / class_name).glob("*")) if p.is_file()]
        rng.shuffle(image_paths)

        split_index = int(len(image_paths) * validation_fraction)
        split_index = max(1, min(split_index, len(image_paths) - 1))

        val_paths = image_paths[:split_index]
        test_paths = image_paths[split_index:]

        validation_samples.extend(ImageSample(path=p, label=label, class_name=class_name) for p in val_paths)
        test_samples.extend(ImageSample(path=p, label=label, class_name=class_name) for p in test_paths)

    rng.shuffle(validation_samples)
    rng.shuffle(test_samples)
    return validation_samples, test_samples


def class_distribution(samples: list[ImageSample], class_names: list[str]) -> dict[str, int]:
    counts = Counter(sample.class_name for sample in samples)
    return {name: counts.get(name, 0) for name in class_names}


def save_class_examples_plot(
    samples: list[ImageSample],
    class_names: list[str],
    output_path: Path,
    seed: int,
    per_class: int = 5,
) -> None:
    rng = random.Random(seed)
    class_to_samples: dict[str, list[ImageSample]] = defaultdict(list)
    for sample in samples:
        class_to_samples[sample.class_name].append(sample)

    fig, axes = plt.subplots(len(class_names), per_class, figsize=(2.4 * per_class, 2.4 * len(class_names)))
    if len(class_names) == 1:
        axes = np.array([axes])

    for row, class_name in enumerate(class_names):
        class_items = class_to_samples[class_name][:]
        rng.shuffle(class_items)
        selected = class_items[:per_class]

        for col in range(per_class):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(selected):
                image = Image.open(selected[col].path).convert("RGB")
                ax.imshow(image)
            if col == 0:
                ax.set_title(class_name)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def save_distribution_plot(
    train_samples: list[ImageSample],
    val_samples: list[ImageSample],
    test_samples: list[ImageSample],
    class_names: list[str],
    output_path: Path,
) -> None:
    train_counts = class_distribution(train_samples, class_names)
    val_counts = class_distribution(val_samples, class_names)
    test_counts = class_distribution(test_samples, class_names)

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(x - width, [train_counts[c] for c in class_names], width=width, label="train")
    ax.bar(x, [val_counts[c] for c in class_names], width=width, label="validation")
    ax.bar(x + width, [test_counts[c] for c in class_names], width=width, label="test")

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20)
    ax.set_ylabel("Number of images")
    ax.set_title("Class Distribution Across Splits")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def save_image_size_plot(samples: list[ImageSample], output_path: Path) -> None:
    widths: list[int] = []
    heights: list[int] = []

    for sample in samples:
        try:
            with Image.open(sample.path) as image:
                width, height = image.size
            widths.append(width)
            heights.append(height)
        except Exception:
            continue

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.scatter(widths, heights, alpha=0.4, s=8)
    ax.set_xlabel("Image width")
    ax.set_ylabel("Image height")
    ax.set_title("Image Resolution Scatter Plot")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def get_train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def freeze_all_but(model: nn.Module, trainable_keys: tuple[str, ...]) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    for name, parameter in model.named_parameters():
        if any(key in name for key in trainable_keys):
            parameter.requires_grad = True


def build_custom_cnn(num_classes: int, pretrained: bool) -> tuple[nn.Module, bool]:
    _ = pretrained
    return CustomTumorCNN(num_classes=num_classes), False


def build_vgg11(num_classes: int, pretrained: bool) -> tuple[nn.Module, bool]:
    got_pretrained = False
    if pretrained:
        try:
            model = models.vgg11(weights=models.VGG11_Weights.DEFAULT)
            got_pretrained = True
        except Exception:
            model = models.vgg11(weights=None)
    else:
        model = models.vgg11(weights=None)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    if got_pretrained:
        freeze_all_but(model, trainable_keys=("classifier.6",))
    return model, got_pretrained


def build_inception_v3(num_classes: int, pretrained: bool) -> tuple[nn.Module, bool]:
    got_pretrained = False
    if pretrained:
        try:
            model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, aux_logits=True)
            got_pretrained = True
        except Exception:
            model = models.inception_v3(weights=None, aux_logits=True)
    else:
        model = models.inception_v3(weights=None, aux_logits=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if model.AuxLogits is not None:
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
    if got_pretrained:
        freeze_all_but(model, trainable_keys=("fc", "AuxLogits.fc"))
    return model, got_pretrained


def build_resnet18(num_classes: int, pretrained: bool) -> tuple[nn.Module, bool]:
    got_pretrained = False
    if pretrained:
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            got_pretrained = True
        except Exception:
            model = models.resnet18(weights=None)
    else:
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if got_pretrained:
        freeze_all_but(model, trainable_keys=("fc",))
    return model, got_pretrained


def unpack_inception_outputs(outputs):
    if hasattr(outputs, "logits"):
        return outputs.logits, getattr(outputs, "aux_logits", None)
    if isinstance(outputs, tuple):
        if len(outputs) == 0:
            raise ValueError("Inception output tuple is empty")
        first = outputs[0]
        second = outputs[1] if len(outputs) > 1 else None
        return first, second
    return outputs, None


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    model_name: str,
    epoch_idx: int,
    epochs: int,
    optimizer: torch.optim.Optimizer | None = None,
    use_inception_aux: bool = False,
) -> tuple[float, dict[str, float]]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    progress_label = "train" if is_training else "eval"
    bar = tqdm(loader, desc=f"{model_name} [{epoch_idx + 1}/{epochs}] {progress_label}", leave=False)

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for images, labels in bar:
            images = images.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad()

            outputs = model(images)
            if use_inception_aux and is_training:
                logits, aux_logits = unpack_inception_outputs(outputs)
                loss = criterion(logits, labels)
                if aux_logits is not None:
                    loss = loss + 0.4 * criterion(aux_logits, labels)
            else:
                logits, _ = unpack_inception_outputs(outputs)
                loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            predictions = torch.argmax(logits, dim=1)
            all_targets.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader.dataset)
    metrics = {
        "accuracy": float(accuracy_score(all_targets, all_predictions)),
        "f1_weighted": float(f1_score(all_targets, all_predictions, average="weighted", zero_division=0)),
        "precision_weighted": float(precision_score(all_targets, all_predictions, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(all_targets, all_predictions, average="weighted", zero_division=0)),
    }
    return avg_loss, metrics


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    model_name: str,
    use_inception_aux: bool,
) -> tuple[float, dict[str, float]]:
    return run_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        model_name=model_name,
        epoch_idx=0,
        epochs=1,
        optimizer=None,
        use_inception_aux=use_inception_aux,
    )


def predict_single_image(
    model: nn.Module,
    image_path: Path,
    transform: transforms.Compose,
    class_names: list[str],
    device: torch.device,
) -> tuple[str, float]:
    model.eval()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        logits, _ = unpack_inception_outputs(outputs)
        probs = torch.softmax(logits, dim=1)
        confidence, label_idx = torch.max(probs, dim=1)

    return class_names[int(label_idx.item())], float(confidence.item())


def save_prediction_grid(
    samples: list[ImageSample],
    loaded_models: dict[str, nn.Module],
    class_names: list[str],
    transform: transforms.Compose,
    device: torch.device,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(samples), figsize=(4.2 * len(samples), 5.2))
    if len(samples) == 1:
        axes = [axes]

    for idx, sample in enumerate(samples):
        ax = axes[idx]
        image = Image.open(sample.path).convert("RGB")
        ax.imshow(image)
        ax.axis("off")

        lines = [f"true: {sample.class_name}"]
        for model_name, model in loaded_models.items():
            predicted, confidence = predict_single_image(
                model=model,
                image_path=sample.path,
                transform=transform,
                class_names=class_names,
                device=device,
            )
            lines.append(f"{model_name}: {predicted} ({confidence:.2f})")

        ax.set_title("\n".join(lines), fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


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
        epochs = range(1, len(history["train_loss"]) + 1)

        loss_ax = loss_axes[idx]
        loss_ax.plot(epochs, history["train_loss"], label="train loss")
        loss_ax.plot(epochs, history["val_loss"], label="validation loss")
        loss_ax.set_title(f"{model_name} - Train vs Validation Loss")
        loss_ax.set_xlabel("Epoch")
        loss_ax.set_ylabel("Loss")
        loss_ax.grid(True, alpha=0.25)
        loss_ax.legend()

        metric_ax = metric_axes[idx]
        metric_ax.plot(epochs, history["train_f1"], label="train F1 weighted")
        metric_ax.plot(epochs, history["val_f1"], label="validation F1 weighted")
        metric_ax.set_title(f"{model_name} - Train vs Validation F1 Weighted")
        metric_ax.set_xlabel("Epoch")
        metric_ax.set_ylabel("F1 weighted")
        metric_ax.grid(True, alpha=0.25)
        metric_ax.legend()

    loss_fig.tight_layout()
    loss_fig.savefig(output_loss_path, dpi=180)
    plt.close(loss_fig)

    metric_fig.tight_layout()
    metric_fig.savefig(output_metric_path, dpi=180)
    plt.close(metric_fig)


def format_metric_with_change(value: float, baseline: float) -> str:
    if baseline == 0:
        return f"{value:.4f} (n/a)"
    delta_pct = ((value - baseline) / baseline) * 100.0
    sign = "+" if delta_pct >= 0 else ""
    return f"{value:.4f} ({sign}{delta_pct:.2f}%)"


def compute_majority_baseline_metrics(
    train_samples: list[ImageSample],
    test_samples: list[ImageSample],
) -> tuple[dict[str, float], int]:
    train_labels = [sample.label for sample in train_samples]
    test_labels = [sample.label for sample in test_samples]

    majority_label = Counter(train_labels).most_common(1)[0][0]
    predictions = [majority_label] * len(test_labels)

    baseline_metrics = {
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "f1_weighted": float(f1_score(test_labels, predictions, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(test_labels, predictions, average="weighted", zero_division=0)),
    }
    return baseline_metrics, majority_label


def write_model_report(
    report_path: Path,
    model_order: list[str],
    train_samples: list[ImageSample],
    test_samples: list[ImageSample],
    train_size: int,
    val_size: int,
    test_size: int,
    class_names: list[str],
    initial_results: dict[str, dict[str, float]],
    reloaded_results: dict[str, dict[str, float]],
    model_hyperparams: dict[str, dict[str, str]],
    used_pretrained: dict[str, bool],
    loss_plot_name: str,
    metric_plot_name: str,
) -> None:
    baseline_metrics, majority_label = compute_majority_baseline_metrics(train_samples, test_samples)

    best_model_name = max(model_order, key=lambda name: initial_results[name]["f1_weighted"])
    best_model_metrics = initial_results[best_model_name]

    lines = [
        "# Model Report - Task 02",
        "",
        (
            "Best model: "
            f"**{best_model_name}** because it has the highest weighted F1 on the test set "
            f"({best_model_metrics['f1_weighted']:.4f}) while maintaining strong accuracy "
            f"({best_model_metrics['accuracy']:.4f})"
        ),
        "",
        "## Dataset Setup",
        f"- Classes: {', '.join(class_names)}",
        f"- Train images: {train_size}",
        f"- Validation images: {val_size}",
        f"- Test images: {test_size}",
        "",
        "## Transfer Learning",
    ]

    for model_name, pretrained_used in used_pretrained.items():
        lines.append(f"- {model_name}: {'pretrained backbone used' if pretrained_used else 'fallback to random initialization'}")

    lines.extend(
        [
            "",
            "## Main Experiment Table",
            "Rows are kept in experiment order. First row is the baseline model",
            "",
            "| Hypothesis | Architecture | Epochs | Batch Size | Learning Rate | Optimizer | Test Accuracy (vs baseline) | Test F1 Weighted (vs baseline) | Test Recall Weighted (vs baseline) | Comments |",
            "| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |",
        ]
    )

    lines.append(
        "| "
        + " | ".join(
            [
                "baseline_majority_class",
                f"Predict most common train class ({class_names[majority_label]})",
                "n/a",
                "n/a",
                "n/a",
                "n/a",
                format_metric_with_change(baseline_metrics["accuracy"], baseline_metrics["accuracy"]),
                format_metric_with_change(baseline_metrics["f1_weighted"], baseline_metrics["f1_weighted"]),
                format_metric_with_change(baseline_metrics["recall_weighted"], baseline_metrics["recall_weighted"]),
                "Greedy statistical baseline; no learning",
            ]
        )
        + " |"
    )

    for model_name in model_order:
        metrics = initial_results[model_name]
        reload_metrics = reloaded_results[model_name]
        params = model_hyperparams[model_name]

        delta_reload = abs(metrics["f1_weighted"] - reload_metrics["f1_weighted"])
        reload_comment = "Reload stable" if delta_reload <= 1e-6 else f"Reload delta F1={delta_reload:.6f}"
        gain_comment = "Improves baseline" if metrics["f1_weighted"] >= baseline_metrics["f1_weighted"] else "Under baseline on F1"

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
                    format_metric_with_change(metrics["accuracy"], baseline_metrics["accuracy"]),
                    format_metric_with_change(metrics["f1_weighted"], baseline_metrics["f1_weighted"]),
                    format_metric_with_change(metrics["recall_weighted"], baseline_metrics["recall_weighted"]),
                    f"{gain_comment} {reload_comment}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Diagrams",
            f"- Train vs validation loss: `{loss_plot_name}`",
            f"- Train vs validation main metric (F1 weighted): `{metric_plot_name}`",
            "",
            "## Notes",
            "- The table is not sorted; it follows experiment creation order",
            "- Metrics include value and percentage change vs baseline",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 02 - Brain tumor multiclass classification")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--validation-fraction", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-pretrained", action="store_true", help="Disable transfer learning weights")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    root = get_project_root()
    data_root = root / "DATA" / "brain_tumor_dataset"
    output_root = Path(__file__).resolve().parent

    class_names = sorted([p.name for p in (data_root / "Training").iterdir() if p.is_dir()])
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    effective_image_size = args.image_size
    if effective_image_size < INCEPTION_MIN_IMAGE_SIZE:
        effective_image_size = INCEPTION_MIN_IMAGE_SIZE
        print(
            "InceptionV3 requires larger inputs for stable training "
            f"Overriding image size from {args.image_size} to {effective_image_size}"
        )

    train_samples = collect_samples_from_split(data_root / "Training", class_to_idx)
    val_samples, test_samples = split_testing_into_validation_and_test(
        testing_dir=data_root / "Testing",
        class_to_idx=class_to_idx,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )

    train_samples = maybe_limit_samples(train_samples, args.max_train_samples, args.seed)
    val_samples = maybe_limit_samples(val_samples, args.max_val_samples, args.seed + 1)
    test_samples = maybe_limit_samples(test_samples, args.max_test_samples, args.seed + 2)

    print(f"Classes: {class_names}")
    print(f"Train size: {len(train_samples)}")
    print(f"Validation size: {len(val_samples)}")
    print(f"Test size: {len(test_samples)}")

    save_class_examples_plot(
        samples=train_samples,
        class_names=class_names,
        output_path=output_root / "data-science-2-eda-class-samples.png",
        seed=args.seed,
        per_class=5,
    )
    save_distribution_plot(
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        class_names=class_names,
        output_path=output_root / "data-science-2-eda-class-distribution.png",
    )
    save_image_size_plot(
        samples=train_samples + val_samples + test_samples,
        output_path=output_root / "data-science-2-eda-image-sizes.png",
    )

    train_transform = get_train_transform(effective_image_size)
    eval_transform = get_eval_transform(effective_image_size)

    train_dataset = BrainTumorDataset(train_samples, transform=train_transform)
    val_dataset = BrainTumorDataset(val_samples, transform=eval_transform)
    test_dataset = BrainTumorDataset(test_samples, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    model_builders = {
        "custom_cnn": build_custom_cnn,
        "inception_v3": build_inception_v3,
        "vgg11": build_vgg11,
        "resnet18": build_resnet18,
    }
    model_arch_names = {
        "custom_cnn": "CustomTumorCNN",
        "inception_v3": "InceptionV3",
        "vgg11": "VGG11",
        "resnet18": "ResNet18",
    }
    model_order = list(model_builders.keys())

    criterion = nn.CrossEntropyLoss()
    trained_results: dict[str, dict[str, float]] = {}
    reloaded_results: dict[str, dict[str, float]] = {}
    used_pretrained: dict[str, bool] = {}
    loaded_models: dict[str, nn.Module] = {}
    histories: dict[str, dict[str, list[float]]] = {}
    model_hyperparams: dict[str, dict[str, str]] = {}

    for model_name, builder in model_builders.items():
        print(f"\n===== Training {model_name} =====")
        model, got_pretrained = builder(num_classes=len(class_names), pretrained=not args.no_pretrained)
        used_pretrained[model_name] = got_pretrained

        model = model.to(device)
        use_inception_aux = model_name == "inception_v3"

        trainable_params = [param for param in model.parameters() if param.requires_grad]
        if len(trainable_params) == 0:
            trainable_params = list(model.parameters())

        optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
        best_state = copy.deepcopy(model.state_dict())
        best_val_f1 = -1.0
        train_losses: list[float] = []
        val_losses: list[float] = []
        train_f1_scores: list[float] = []
        val_f1_scores: list[float] = []

        for epoch_idx in range(args.epochs):
            train_loss, train_metrics = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                model_name=model_name,
                epoch_idx=epoch_idx,
                epochs=args.epochs,
                optimizer=optimizer,
                use_inception_aux=use_inception_aux,
            )
            val_loss, val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                model_name=f"{model_name}-val",
                epoch_idx=epoch_idx,
                epochs=args.epochs,
                optimizer=None,
                use_inception_aux=use_inception_aux,
            )

            print(
                f"[{model_name}] epoch {epoch_idx + 1}/{args.epochs} | "
                f"train loss={train_loss:.4f} f1={train_metrics['f1_weighted']:.4f} | "
                f"val loss={val_loss:.4f} f1={val_metrics['f1_weighted']:.4f}"
            )

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_f1_scores.append(train_metrics["f1_weighted"])
            val_f1_scores.append(val_metrics["f1_weighted"])

            if val_metrics["f1_weighted"] > best_val_f1:
                best_val_f1 = val_metrics["f1_weighted"]
                best_state = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_state)
        histories[model_name] = {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "train_f1": train_f1_scores,
            "val_f1": val_f1_scores,
        }

        test_loss, test_metrics = evaluate_model(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            model_name=f"{model_name}-test",
            use_inception_aux=use_inception_aux,
        )
        print(f"[{model_name}] test loss={test_loss:.4f}, metrics={test_metrics}")
        trained_results[model_name] = test_metrics

        save_path = output_root / f"data-science-2-{model_name}.pt"
        torch.save(
            {
                "model_name": model_name,
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "image_size": effective_image_size,
                "pretrained_used": got_pretrained,
            },
            save_path,
        )
        print(f"Saved {model_name} to {save_path}")

        reloaded_model, _ = builder(num_classes=len(class_names), pretrained=False)
        checkpoint = torch.load(save_path, map_location=device)
        reloaded_model.load_state_dict(checkpoint["state_dict"])
        reloaded_model = reloaded_model.to(device)

        _, reloaded_metrics = evaluate_model(
            model=reloaded_model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            model_name=f"{model_name}-reloaded-test",
            use_inception_aux=use_inception_aux,
        )
        reloaded_results[model_name] = reloaded_metrics
        loaded_models[model_name] = reloaded_model
        print(f"Reloaded {model_name} metrics: {reloaded_metrics}")

        model_hyperparams[model_name] = {
            "architecture": f"{model_arch_names[model_name]} ({'pretrained' if got_pretrained else 'random init'})",
            "input_size": f"{effective_image_size}x{effective_image_size}",
            "epochs": str(args.epochs),
            "batch_size": str(args.batch_size),
            "learning_rate": f"{args.learning_rate}",
            "optimizer": "AdamW",
        }

    rng = random.Random(args.seed)
    sampled_test_images = test_samples[:]
    rng.shuffle(sampled_test_images)
    sampled_test_images = sampled_test_images[:5]

    save_prediction_grid(
        samples=sampled_test_images,
        loaded_models=loaded_models,
        class_names=class_names,
        transform=eval_transform,
        device=device,
        output_path=output_root / "data-science-2-loaded-model-predictions.png",
    )

    loss_curve_path = output_root / "data-science-2-train-vs-val-loss.png"
    metric_curve_path = output_root / "data-science-2-train-vs-val-metric.png"
    save_training_curves(
        histories=histories,
        output_loss_path=loss_curve_path,
        output_metric_path=metric_curve_path,
    )

    write_model_report(
        report_path=output_root / "data-science-2-model-report.md",
        model_order=model_order,
        train_samples=train_samples,
        test_samples=test_samples,
        train_size=len(train_samples),
        val_size=len(val_samples),
        test_size=len(test_samples),
        class_names=class_names,
        initial_results=trained_results,
        reloaded_results=reloaded_results,
        model_hyperparams=model_hyperparams,
        used_pretrained=used_pretrained,
        loss_plot_name=loss_curve_path.name,
        metric_plot_name=metric_curve_path.name,
    )

    print("\nTask 02 completed")
    print(f"Artifacts saved in: {output_root}")


if __name__ == "__main__":
    main()
