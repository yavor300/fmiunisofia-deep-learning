"""Analyze Cityscapes data and write assignment-ready EDA outputs."""

from __future__ import annotations

import argparse
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np
from PIL import Image, UnidentifiedImageError

from src.cityseg.config import load_config
from src.cityseg.constants import CITYSCAPES_CLASSES, IGNORE_INDEX, LABEL_ID_TO_TRAIN_ID
from src.cityseg.data.label_mapping import (
    convert_label_ids_to_train_ids,
    decode_train_ids_to_colors,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

SPLITS = ("train", "val", "test")
FIGURE_NAMES = {
    "class_distribution": "class_distribution.png",
    "image_size_distribution": "image_size_distribution.png",
    "sample_overlays": "sample_overlays.png",
    "rare_classes_examples": "rare_classes_examples.png",
}


@dataclass
class DatasetAnalysis:
    """Container for Cityscapes EDA results."""

    root: Path
    split_image_counts: dict[str, int]
    split_mask_counts: dict[str, int]
    image_size_counts: Counter[tuple[int, int]]
    class_pixel_counts: dict[str, int]
    class_percentages: dict[str, float]
    anomaly_counts: dict[str, int]
    anomaly_messages: list[str] = field(default_factory=list)
    sample_pairs: list[tuple[Path, Path]] = field(default_factory=list)
    rare_class_examples: dict[str, tuple[Path, Path]] = field(default_factory=dict)
    max_samples_per_split: int | None = None
    analyzed_pair_counts: dict[str, int] = field(default_factory=dict)

    @property
    def total_images(self) -> int:
        return sum(self.split_image_counts.values())

    @property
    def total_masks(self) -> int:
        return sum(self.split_mask_counts.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the Cityscapes dataset.")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=None,
        help="Limit pixel/figure analysis per split. Use 0 for a full scan.",
    )
    return parser.parse_args()


def run_analysis_from_config(config: dict[str, Any]) -> DatasetAnalysis:
    paths = config.get("paths", {})
    eda_config = config.get("eda", {})
    data_root = Path(paths.get("data_root", "data/raw/cityscapes"))
    reports_dir = Path(paths.get("reports_dir", "reports"))
    docs_dir = Path(paths.get("docs_dir", "docs"))
    figures_dir = reports_dir / "figures"
    max_samples_per_split = _normalize_sample_limit(eda_config.get("max_samples_per_split"))

    analysis = analyze_cityscapes_dataset(
        data_root,
        max_samples_per_split=max_samples_per_split,
    )
    write_figures(analysis, figures_dir)
    write_dataset_report(analysis, docs_dir / "dataset_analysis.md")
    return analysis


def analyze_cityscapes_dataset(
    root: str | Path,
    max_samples_per_split: int | None = None,
) -> DatasetAnalysis:
    root_path = Path(root)
    split_image_counts: dict[str, int] = {}
    split_mask_counts: dict[str, int] = {}
    image_size_counts: Counter[tuple[int, int]] = Counter()
    class_counts = np.zeros(len(CITYSCAPES_CLASSES), dtype=np.int64)
    anomaly_counts = {
        "missing_images": 0,
        "missing_masks": 0,
        "unreadable_files": 0,
        "invalid_mask_values": 0,
        "mismatched_dimensions": 0,
    }
    anomaly_messages: list[str] = []
    sample_pairs: list[tuple[Path, Path]] = []
    class_examples: dict[int, tuple[Path, Path]] = {}
    analyzed_pair_counts: dict[str, int] = {}

    for split in SPLITS:
        image_paths = _find_images(root_path, split)
        mask_paths = _find_masks(root_path, split)
        split_image_counts[split] = len(image_paths)
        split_mask_counts[split] = len(mask_paths)
        image_paths_to_analyze = _limit_paths(image_paths, max_samples_per_split)
        analyzed_pair_counts[split] = len(image_paths_to_analyze)

        mask_path_set = set(mask_paths)
        image_path_set = set(image_paths)

        for image_path in image_paths:
            mask_path = _expected_mask_path(root_path, image_path, split)
            if mask_path not in mask_path_set:
                anomaly_counts["missing_masks"] += 1
                anomaly_messages.append(f"Missing mask for image: {image_path}")

                if image_path in image_paths_to_analyze:
                    _record_image_size(
                        image_path,
                        image_size_counts,
                        anomaly_counts,
                        anomaly_messages,
                    )
                continue

        for image_path in image_paths_to_analyze:
            mask_path = _expected_mask_path(root_path, image_path, split)
            if mask_path not in mask_path_set:
                continue

            _analyze_pair(
                image_path=image_path,
                mask_path=mask_path,
                image_size_counts=image_size_counts,
                class_counts=class_counts,
                anomaly_counts=anomaly_counts,
                anomaly_messages=anomaly_messages,
                sample_pairs=sample_pairs,
                class_examples=class_examples,
            )

        for mask_path in mask_paths:
            image_path = _expected_image_path(root_path, mask_path, split)
            if image_path not in image_path_set:
                anomaly_counts["missing_images"] += 1
                anomaly_messages.append(f"Missing image for mask: {mask_path}")

    total_pixels = int(class_counts.sum())
    class_pixel_counts = {
        class_name: int(count)
        for class_name, count in zip(CITYSCAPES_CLASSES, class_counts, strict=True)
    }
    class_percentages = {
        class_name: (int(count) / total_pixels * 100.0 if total_pixels else 0.0)
        for class_name, count in zip(CITYSCAPES_CLASSES, class_counts, strict=True)
    }
    rare_class_examples = _select_rare_class_examples(class_counts, class_examples)

    return DatasetAnalysis(
        root=root_path,
        split_image_counts=split_image_counts,
        split_mask_counts=split_mask_counts,
        image_size_counts=image_size_counts,
        class_pixel_counts=class_pixel_counts,
        class_percentages=class_percentages,
        anomaly_counts=anomaly_counts,
        anomaly_messages=anomaly_messages,
        sample_pairs=sample_pairs,
        rare_class_examples=rare_class_examples,
        max_samples_per_split=max_samples_per_split,
        analyzed_pair_counts=analyzed_pair_counts,
    )


def build_dataset_report(analysis: DatasetAnalysis) -> str:
    dominant_class, dominant_pct = _dominant_class(analysis)
    rare_classes = _rare_classes(analysis)
    anomaly_total = sum(analysis.anomaly_counts.values())

    lines = [
        "# Dataset Analysis",
        "",
        "## Overview",
        "",
        f"- Dataset root: `{analysis.root}`",
        f"- Total images found: {analysis.total_images}",
        f"- Total masks found: {analysis.total_masks}",
        f"- Number of train classes: {len(CITYSCAPES_CLASSES)}",
        f"- Pixel/figure sample limit per split: {_format_sample_limit(analysis)}",
        "",
        "## Images Per Split",
        "",
    ]
    for split in SPLITS:
        image_count = analysis.split_image_counts.get(split, 0)
        analyzed_count = analysis.analyzed_pair_counts.get(split, 0)
        lines.append(f"- `{split}`: {image_count} images; {analyzed_count} sampled for pixel EDA")

    lines.extend(
        [
            "",
            "## Image Size Distribution",
            "",
            _format_size_distribution(analysis.image_size_counts),
            "",
            "## Class Distribution",
            "",
            "| Class | Pixels | Percentage |",
            "| --- | ---: | ---: |",
        ]
    )
    for class_name in CITYSCAPES_CLASSES:
        pixels = analysis.class_pixel_counts[class_name]
        percentage = analysis.class_percentages[class_name]
        lines.append(f"| {class_name} | {pixels} | {percentage:.4f}% |")

    lines.extend(
        [
            "",
            "## Class Imbalance",
            "",
            _class_imbalance_commentary(analysis, dominant_class, dominant_pct, rare_classes),
            "",
            "## Anomaly Checks",
            "",
        ]
    )
    for anomaly_name, count in analysis.anomaly_counts.items():
        lines.append(f"- `{anomaly_name}`: {count}")

    lines.extend(["", _anomaly_commentary(anomaly_total), ""])
    if analysis.anomaly_messages:
        lines.extend(["### Anomaly Details", ""])
        for message in analysis.anomaly_messages[:30]:
            lines.append(f"- {message}")
        if len(analysis.anomaly_messages) > 30:
            lines.append(f"- ... {len(analysis.anomaly_messages) - 30} more anomalies omitted")
        lines.append("")

    lines.extend(
        [
            "## Figures",
            "",
            f"- `reports/figures/{FIGURE_NAMES['class_distribution']}`",
            f"- `reports/figures/{FIGURE_NAMES['image_size_distribution']}`",
            f"- `reports/figures/{FIGURE_NAMES['sample_overlays']}`",
            f"- `reports/figures/{FIGURE_NAMES['rare_classes_examples']}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_dataset_report(analysis: DatasetAnalysis, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_dataset_report(analysis), encoding="utf-8")


def write_figures(analysis: DatasetAnalysis, figures_dir: str | Path) -> None:
    path = Path(figures_dir)
    path.mkdir(parents=True, exist_ok=True)
    _plot_class_distribution(analysis, path / FIGURE_NAMES["class_distribution"])
    _plot_image_size_distribution(analysis, path / FIGURE_NAMES["image_size_distribution"])
    _plot_sample_overlays(analysis.sample_pairs, path / FIGURE_NAMES["sample_overlays"])
    _plot_rare_class_examples(
        analysis.rare_class_examples,
        path / FIGURE_NAMES["rare_classes_examples"],
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.max_samples_per_split is not None:
        config.setdefault("eda", {})["max_samples_per_split"] = args.max_samples_per_split

    analysis = run_analysis_from_config(config)
    print(f"Analyzed {analysis.total_images} images from {analysis.root}")
    print("Wrote docs/dataset_analysis.md and reports/figures/*.png")


def _normalize_sample_limit(value: Any) -> int | None:
    if value is None:
        return None
    limit = int(value)
    if limit <= 0:
        return None
    return limit


def _limit_paths(paths: list[Path], max_samples_per_split: int | None) -> list[Path]:
    if max_samples_per_split is None:
        return paths
    return paths[:max_samples_per_split]


def _format_sample_limit(analysis: DatasetAnalysis) -> str:
    if analysis.max_samples_per_split is None:
        return "full scan"
    return str(analysis.max_samples_per_split)


def _find_images(root: Path, split: str) -> list[Path]:
    return sorted((root / "leftImg8bit" / split).glob("*/*_leftImg8bit.png"))


def _find_masks(root: Path, split: str) -> list[Path]:
    return sorted((root / "gtFine" / split).glob("*/*_gtFine_labelIds.png"))


def _expected_mask_path(root: Path, image_path: Path, split: str) -> Path:
    city = image_path.parent.name
    name = image_path.name.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
    return root / "gtFine" / split / city / name


def _expected_image_path(root: Path, mask_path: Path, split: str) -> Path:
    city = mask_path.parent.name
    name = mask_path.name.replace("_gtFine_labelIds.png", "_leftImg8bit.png")
    return root / "leftImg8bit" / split / city / name


def _analyze_pair(
    image_path: Path,
    mask_path: Path,
    image_size_counts: Counter[tuple[int, int]],
    class_counts: np.ndarray,
    anomaly_counts: dict[str, int],
    anomaly_messages: list[str],
    sample_pairs: list[tuple[Path, Path]],
    class_examples: dict[int, tuple[Path, Path]],
) -> None:
    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_size = image.size
        with Image.open(mask_path) as mask_image:
            mask = np.asarray(mask_image).copy()
            mask_size = mask_image.size
    except (OSError, UnidentifiedImageError) as error:
        anomaly_counts["unreadable_files"] += 1
        anomaly_messages.append(f"Unreadable file in pair {image_path} / {mask_path}: {error}")
        return

    image_size_counts[image_size] += 1

    if image_size != mask_size:
        anomaly_counts["mismatched_dimensions"] += 1
        anomaly_messages.append(
            f"Mismatched dimensions for {image_path.name}: image={image_size}, mask={mask_size}"
        )

    unknown_values = _unknown_mask_values(mask)
    if unknown_values:
        anomaly_counts["invalid_mask_values"] += 1
        anomaly_messages.append(f"Invalid mask values in {mask_path}: {unknown_values}")

    train_id_mask = convert_label_ids_to_train_ids(mask)
    valid_pixels = train_id_mask[train_id_mask != IGNORE_INDEX]
    if valid_pixels.size:
        class_counts += np.bincount(valid_pixels, minlength=len(CITYSCAPES_CLASSES))
        for train_id in np.unique(valid_pixels):
            class_examples.setdefault(int(train_id), (image_path, mask_path))

    if len(sample_pairs) < 3:
        sample_pairs.append((image_path, mask_path))


def _record_image_size(
    image_path: Path,
    image_size_counts: Counter[tuple[int, int]],
    anomaly_counts: dict[str, int],
    anomaly_messages: list[str],
) -> None:
    try:
        with Image.open(image_path) as image:
            image_size_counts[image.size] += 1
    except (OSError, UnidentifiedImageError) as error:
        anomaly_counts["unreadable_files"] += 1
        anomaly_messages.append(f"Unreadable image {image_path}: {error}")


def _unknown_mask_values(mask: np.ndarray) -> list[int]:
    valid_label_ids = {label_id for label_id in LABEL_ID_TO_TRAIN_ID if label_id >= 0}
    values = set(int(value) for value in np.unique(mask))
    return sorted(values - valid_label_ids)


def _select_rare_class_examples(
    class_counts: np.ndarray,
    class_examples: dict[int, tuple[Path, Path]],
) -> dict[str, tuple[Path, Path]]:
    present_train_ids = [train_id for train_id, count in enumerate(class_counts) if count > 0]
    rare_train_ids = sorted(present_train_ids, key=lambda train_id: int(class_counts[train_id]))[:3]
    return {
        CITYSCAPES_CLASSES[train_id]: class_examples[train_id]
        for train_id in rare_train_ids
        if train_id in class_examples
    }


def _dominant_class(analysis: DatasetAnalysis) -> tuple[str, float]:
    if not analysis.class_percentages:
        return "none", 0.0
    return max(analysis.class_percentages.items(), key=lambda item: item[1])


def _rare_classes(analysis: DatasetAnalysis) -> list[str]:
    present_classes = [
        class_name
        for class_name, count in analysis.class_pixel_counts.items()
        if count > 0
    ]
    return sorted(present_classes, key=lambda name: analysis.class_pixel_counts[name])[:5]


def _format_size_distribution(size_counts: Counter[tuple[int, int]]) -> str:
    if not size_counts:
        return "No readable images were found, so image-size statistics could not be computed."

    lines = ["| Width | Height | Images |", "| ---: | ---: | ---: |"]
    for (width, height), count in size_counts.most_common():
        lines.append(f"| {width} | {height} | {count} |")
    return "\n".join(lines)


def _class_imbalance_commentary(
    analysis: DatasetAnalysis,
    dominant_class: str,
    dominant_pct: float,
    rare_classes: list[str],
) -> str:
    if analysis.total_masks == 0 or sum(analysis.class_pixel_counts.values()) == 0:
        return (
            "No labeled pixels were found. After the Cityscapes masks are added, this section "
            "will quantify how strongly dominant road-scene classes outweigh rare objects."
        )

    rare_text = ", ".join(rare_classes) if rare_classes else "none"
    return (
        f"The dominant class is `{dominant_class}` with {dominant_pct:.2f}% of labeled pixels. "
        f"The rarest observed classes are: {rare_text}. This imbalance matters for training: "
        "plain pixel-wise cross-entropy can over-reward predictions of frequent classes, while "
        "focal loss can emphasize hard or under-represented pixels and Dice loss can optimize "
        "region overlap so small classes are not judged only by raw pixel volume."
    )


def _anomaly_commentary(anomaly_total: int) -> str:
    if anomaly_total == 0:
        return "No structural anomalies were found in the scanned files."
    return (
        "These anomalies should be fixed or intentionally filtered before training. Missing "
        "pairs reduce usable data, invalid labels can poison class statistics, and mismatched "
        "dimensions can break image-mask alignment."
    )


def _plot_class_distribution(analysis: DatasetAnalysis, output_path: Path) -> None:
    names = list(CITYSCAPES_CLASSES)
    percentages = [analysis.class_percentages[name] for name in names]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(names, percentages, color="#4c78a8")
    ax.set_ylabel("Pixels (%)")
    ax.set_title("Cityscapes Class Distribution")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_image_size_distribution(analysis: DatasetAnalysis, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    if analysis.image_size_counts:
        labels = [f"{width}x{height}" for width, height in analysis.image_size_counts]
        counts = [analysis.image_size_counts[size] for size in analysis.image_size_counts]
        ax.bar(labels, counts, color="#59a14f")
        ax.set_ylabel("Images")
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.text(0.5, 0.5, "No readable images found", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("Image Size Distribution")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_sample_overlays(sample_pairs: list[tuple[Path, Path]], output_path: Path) -> None:
    if not sample_pairs:
        _write_placeholder_figure(output_path, "No image/mask pairs found")
        return

    rows = len(sample_pairs)
    fig, axes = plt.subplots(rows, 3, figsize=(10, 3.5 * rows), squeeze=False)
    for row_index, (image_path, mask_path) in enumerate(sample_pairs):
        image, color_mask, overlay = _load_visualization_arrays(image_path, mask_path)
        for col_index, (array, title) in enumerate(
            ((image, "Image"), (color_mask, "Mask"), (overlay, "Overlay"))
        ):
            axes[row_index][col_index].imshow(array)
            axes[row_index][col_index].set_title(title)
            axes[row_index][col_index].axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_rare_class_examples(
    examples: dict[str, tuple[Path, Path]],
    output_path: Path,
) -> None:
    if not examples:
        _write_placeholder_figure(output_path, "No rare class examples available")
        return

    rows = len(examples)
    fig, axes = plt.subplots(rows, 3, figsize=(10, 3.5 * rows), squeeze=False)
    for row_index, (class_name, (image_path, mask_path)) in enumerate(examples.items()):
        image, color_mask, overlay = _load_visualization_arrays(image_path, mask_path)
        for col_index, (array, title) in enumerate(
            ((image, "Image"), (color_mask, "Mask"), (overlay, class_name))
        ):
            axes[row_index][col_index].imshow(array)
            axes[row_index][col_index].set_title(title)
            axes[row_index][col_index].axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _load_visualization_arrays(
    image_path: Path,
    mask_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    mask = np.asarray(Image.open(mask_path)).copy()
    train_id_mask = convert_label_ids_to_train_ids(mask)
    color_mask = decode_train_ids_to_colors(train_id_mask)
    overlay = np.clip(image.astype(np.float32) * 0.6 + color_mask.astype(np.float32) * 0.4, 0, 255)
    return image, color_mask, overlay.astype(np.uint8)


def _write_placeholder_figure(output_path: Path, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
