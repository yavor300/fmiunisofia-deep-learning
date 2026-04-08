from __future__ import annotations

import torch
import torch.nn as nn


IMAGE_SIZE = (64, 64)
MULTICLASS_EXAMPLE_CLASSES = 4


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


class BinaryImageCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.act1 = nn.ReLU(inplace=True)

        self.extra_conv: nn.Conv2d | None = None
        self.extra_act: nn.ReLU | None = None

        self.features_tail = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, 1)

    def add_intermediate_conv_layer(self) -> None:
        if self.extra_conv is None:
            self.extra_conv = nn.Conv2d(16, 16, kernel_size=3, padding=1)
            self.extra_act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.conv1(x))
        if self.extra_conv is not None and self.extra_act is not None:
            x = self.extra_act(self.extra_conv(x))
        x = self.features_tail(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


class MultiClassImageCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


def main() -> None:
    binary_cnn = BinaryImageCNN()
    multiclass_cnn = MultiClassImageCNN(num_classes=MULTICLASS_EXAMPLE_CLASSES)

    print(f"Input resolution: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} RGB")
    print(
        "Number of parameters in CNN for binary classification: "
        f"{count_parameters(binary_cnn):,}"
    )
    print(
        "Number of parameters in CNN for multiclass classification "
        f"({MULTICLASS_EXAMPLE_CLASSES} classes): {count_parameters(multiclass_cnn):,}"
    )

    binary_cnn.add_intermediate_conv_layer()
    print(
        "Number of parameters in CNN for binary classification with two "
        f"convolutional layers at the start: {count_parameters(binary_cnn):,}"
    )

    print("Updated binary classifier:")
    print(binary_cnn)


if __name__ == "__main__":
    main()
