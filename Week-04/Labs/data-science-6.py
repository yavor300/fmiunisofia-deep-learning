from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm


SEED = 42
EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 0.001


torch.manual_seed(SEED)


class CloudCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Linear(64 * 16 * 16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    train_dir = root / "DATA" / "clouds" / "clouds_train"

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(0, 45)),
            transforms.RandomAutocontrast(),
            transforms.ToTensor(),
            transforms.Resize((64, 64)),
        ]
    )

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    dataloader_train = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    first_image, _ = train_dataset[0]
    plt.figure(figsize=(4, 4))
    plt.imshow(first_image.permute(1, 2, 0).clamp(0, 1))
    plt.title("First preprocessed training image")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(Path(__file__).with_name("data-science-6-first-preprocessed-image.png"))

    model = CloudCNN(num_classes=len(train_dataset.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    loss_per_epoch: list[float] = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(dataloader_train, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        average_loss = running_loss / len(dataloader_train)
        loss_per_epoch.append(average_loss)
        print(f"Average training loss per batch: {average_loss}")

    print(f"Average training loss per epoch: {sum(loss_per_epoch) / len(loss_per_epoch)}")

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, EPOCHS + 1), loss_per_epoch)
    plt.xlabel("Epoch")
    plt.ylabel("Average training loss")
    plt.title("Task 06 Training Loss Per Epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(__file__).with_name("data-science-6-loss-per-epoch.png"))


if __name__ == "__main__":
    main()
