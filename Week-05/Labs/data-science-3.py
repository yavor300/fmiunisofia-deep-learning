from __future__ import annotations

from pathlib import Path

import torch
from torchvision import io, transforms, utils
from torchvision.transforms import InterpolationMode


TARGET_SIZE = 224
ESPRESSO_BOX = torch.tensor([[60, 88, 179, 204]], dtype=torch.int64)
BOX_LABELS = ["espresso"]


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    input_path = root / "DATA" / "w05_task03_start.jpeg"
    output_path = Path(__file__).with_name("data-science-3-result.jpeg")

    image_tensor = io.read_image(str(input_path))
    resize = transforms.Resize(
        (TARGET_SIZE, TARGET_SIZE),
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    resized_image = resize(image_tensor)

    boxed_image = utils.draw_bounding_boxes(
        resized_image,
        ESPRESSO_BOX,
        labels=BOX_LABELS,
        colors="red",
        width=3,
    )

    io.write_jpeg(boxed_image, str(output_path), quality=95)

    print(f"Loaded image: {input_path}")
    print(f"Resized image to: {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"Drew espresso bounding box (xyxy): {ESPRESSO_BOX.tolist()[0]}")
    print(f"Saved result to: {output_path}")


if __name__ == "__main__":
    main()
