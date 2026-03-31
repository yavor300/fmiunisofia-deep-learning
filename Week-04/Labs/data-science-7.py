from __future__ import annotations

import time
from pathlib import Path
from pprint import pprint

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
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


def evaluate(model: nn.Module, dataloader_test: DataLoader, class_names: list[str]) -> None:
    model.eval()
    all_targets: list[int] = []
    all_predictions: list[int] = []

    with torch.no_grad():
        for images, labels in dataloader_test:
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            all_targets.extend(labels.tolist())
            all_predictions.extend(predictions.tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0,
    )
    _, _, class_f1, _ = precision_recall_fscore_support(
        all_targets,
        all_predictions,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    per_class_f1 = {
        class_name: round(float(class_score), 4)
        for class_name, class_score in zip(class_names, class_f1)
    }

    print("\nSummary statistics:")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
    print("\nPer class F1 score")
    pprint(per_class_f1)

    # The model learns strong signals for some classes but underperforms on other
    # Data augmentation helps generalization, but class confusion remains high


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    train_dir = root / "DATA" / "clouds" / "clouds_train"
    test_dir = root / "DATA" / "clouds" / "clouds_test"

    prep_start = time.time()
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(0, 45)),
            transforms.RandomAutocontrast(),
            transforms.ToTensor(),
            transforms.Resize((64, 64)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((64, 64)),
        ]
    )

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    dataloader_train = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dataloader_test = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    data_prep_time = time.time() - prep_start
    print(f"Total time taken for data preparation in seconds: {data_prep_time}")

    model = CloudCNN(num_classes=len(train_dataset.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    training_start = time.time()
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

    training_time = time.time() - training_start
    print(f"\nTotal time taken to train the model in seconds: {training_time}")
    print(f"Average training loss per epoch: {float(np.mean(loss_per_epoch))}")

    evaluate(model, dataloader_test, train_dataset.classes)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, EPOCHS + 1), loss_per_epoch)
    plt.xlabel("Epoch")
    plt.ylabel("Average training loss")
    plt.title("Task 07 Training Loss Per Epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(__file__).with_name("data-science-7-loss-per-epoch.png"))


if __name__ == "__main__":
    main()
