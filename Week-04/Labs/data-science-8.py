from __future__ import annotations

from pathlib import Path
from pprint import pprint

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm.auto import tqdm


SEED = 42
EPOCHS = 35
BATCH_SIZE = 32
LEARNING_RATE = 0.001


torch.manual_seed(SEED)
np.random.seed(SEED)


class BetterCloudCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_samples = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        iterator = tqdm(loader, disable=not is_training)
        for images, labels in iterator:
            if is_training:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            all_targets.extend(labels.tolist())
            all_predictions.extend(torch.argmax(logits, dim=1).tolist())

    average_loss = total_loss / total_samples
    macro_f1 = float(f1_score(all_targets, all_predictions, average="macro", zero_division=0))
    return average_loss, macro_f1


def evaluate_test(model: nn.Module, loader: DataLoader, class_names: list[str]) -> tuple[float, dict[str, float]]:
    model.eval()
    all_targets: list[int] = []
    all_predictions: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            all_targets.extend(labels.tolist())
            all_predictions.extend(predictions.tolist())

    macro_f1 = float(f1_score(all_targets, all_predictions, average="macro", zero_division=0))
    _, _, per_class_f1, _ = precision_recall_fscore_support(
        all_targets,
        all_predictions,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    per_class = {
        class_name: round(float(score), 4)
        for class_name, score in zip(class_names, per_class_f1)
    }
    return macro_f1, per_class


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    train_dir = root / "DATA" / "clouds" / "clouds_train"
    test_dir = root / "DATA" / "clouds" / "clouds_test"

    train_transform = transforms.Compose(
        [
            transforms.Resize((96, 96)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(35),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.RandomAutocontrast(),
            transforms.ToTensor(),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
        ]
    )

    full_train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    indices = np.arange(len(full_train_dataset))
    np.random.shuffle(indices)
    split_idx = int(0.85 * len(indices))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_subset = Subset(full_train_dataset, train_indices.tolist())
    val_dataset = datasets.ImageFolder(train_dir, transform=eval_transform)
    val_subset = Subset(val_dataset, val_indices.tolist())
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = BetterCloudCNN(num_classes=len(full_train_dataset.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    train_losses: list[float] = []
    val_losses: list[float] = []
    train_f1s: list[float] = []
    val_f1s: list[float] = []

    best_val_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(EPOCHS):
        train_loss, train_f1 = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_f1 = run_epoch(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        print(f"Epoch [{epoch + 1}/{EPOCHS}]:")
        print(f" Average training loss: {train_loss}")
        print(f" Average validation loss: {val_loss}")
        print(f" Training macro F1: {train_f1}")
        print(f" Validation macro F1: {val_f1}")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_macro_f1, per_class_f1 = evaluate_test(model, test_loader, full_train_dataset.classes)
    print(f"\nBest validation macro F1: {best_val_f1}")
    print(f"Test macro F1: {test_macro_f1}")
    print("Per class F1 score")
    pprint(per_class_f1)

    figure_path = Path(__file__).with_name("data-science-8-training-curves.png")
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train_f1s, label="Train")
    plt.plot(val_f1s, label="Validation")
    plt.title("Macro F1")
    plt.xlabel("Epoch")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.bar(range(len(per_class_f1)), list(per_class_f1.values()))
    plt.xticks(range(len(per_class_f1)), list(per_class_f1.keys()), rotation=60, ha="right")
    plt.title("Per-Class F1 on Test")
    plt.ylim(0, 1)
    plt.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig(figure_path)


if __name__ == "__main__":
    main()
