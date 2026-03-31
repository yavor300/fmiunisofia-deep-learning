from __future__ import annotations

import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    train_dir = root / "DATA" / "clouds" / "clouds_train"

    image_paths = [path for path in train_dir.rglob("*.jpg")]
    selected_paths = random.sample(image_paths, 6)

    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    for axis, image_path in zip(axes.flatten(), selected_paths):
        image = Image.open(image_path).convert("RGB")
        axis.imshow(image)
        axis.set_title(image_path.parent.name)
        axis.axis("off")

    figure.suptitle("Six Randomly Selected Cloud Images", fontsize=14)
    figure.tight_layout()
    figure.savefig(Path(__file__).with_name("data-science-5-random-clouds.png"))


if __name__ == "__main__":
    main()
