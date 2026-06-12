from __future__ import annotations

import random
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_image_and_mask(image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    return image, mask


def segment_foreground(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # In this dataset: 1=foreground, 2=background, 3=not classified.
    binary_mask = (mask == 1).astype(np.uint8)
    return image * binary_mask[..., None]


def main() -> None:
    root = get_project_root()
    images_dir = root / "DATA" / "segmentation_cats_dogs" / "images"
    masks_dir = root / "DATA" / "segmentation_cats_dogs" / "annotations"

    image_paths = sorted([p for p in images_dir.glob("*.jpg") if (masks_dir / f"{p.stem}.png").exists()])
    chosen = random.sample(image_paths, k=5)

    fig, axes = plt.subplots(5, 2, figsize=(12, 13))
    fig.suptitle("Exploring the dataset", fontsize=16)

    for row, image_path in enumerate(chosen):
        mask_path = masks_dir / f"{image_path.stem}.png"
        image, mask = load_image_and_mask(image_path, mask_path)
        segmented = segment_foreground(image, mask)

        axes[row, 0].imshow(image)
        axes[row, 0].set_title(image_path.stem)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(segmented)
        axes[row, 1].set_title(f"{image_path.stem} segmented")
        axes[row, 1].axis("off")

    plt.tight_layout()
    if matplotlib.get_backend().lower() == "agg":
        output_path = Path(__file__).resolve().parent / "data-science-1-result.png"
        plt.savefig(output_path, dpi=180)
        print(f"Saved figure to: {output_path}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
