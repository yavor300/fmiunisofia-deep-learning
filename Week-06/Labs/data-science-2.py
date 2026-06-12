from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn


COCO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "trafficlight",
    "firehydrant", "streetsign", "stopsign", "parkingmeter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eyeglasses",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball", "kite", "baseballbat",
    "baseballglove", "skateboard", "surfboard", "tennisracket", "bottle", "plate", "wineglass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hotdog",
    "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "mirror", "diningtable", "window", "desk",
    "toilet", "door", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cellphone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors", "teddybear", "hairdrier",
    "toothbrush", "hairbrush",
]


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    root = get_project_root()
    image_path = root / "DATA" / "w06_task02.jpg"
    image = Image.open(image_path).convert("RGB")

    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    model.eval()

    image_tensor = transforms.ToTensor()(image).unsqueeze(0)

    with torch.no_grad():
        prediction = model(image_tensor)[0]

    labels = prediction["labels"]
    scores = prediction["scores"]
    masks = prediction["masks"]

    if len(labels) < 2:
        raise RuntimeError("Mask R-CNN returned fewer than 2 detections for this image.")

    top1_label = COCO_CLASS_NAMES[int(labels[0].item()) - 1]
    top2_label = COCO_CLASS_NAMES[int(labels[1].item()) - 1]
    top1_score = float(scores[0].item())
    top2_score = float(scores[1].item())

    print(f'Object with highest confidence ({top1_score:.4f}) is "{top1_label}".')
    print(f'Second object with highest confidence ({top2_score:.4f}) is "{top2_label}".')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i in range(2):
        axes[i].imshow(image)
        axes[i].imshow(masks[i, 0].cpu().numpy(), cmap="jet", alpha=0.5)
        axes[i].set_title(f"Object: {COCO_CLASS_NAMES[int(labels[i].item()) - 1]}")
        axes[i].axis("off")

    plt.tight_layout()
    output_path = Path(__file__).resolve().parent / "data-science-2-result.png"
    plt.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
