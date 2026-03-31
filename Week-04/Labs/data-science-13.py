from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


SEED = 42
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001


np.random.seed(SEED)
torch.manual_seed(SEED)


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(32 * 2 * 2, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def format_value_with_change(value: float, baseline: float) -> str:
    if baseline == 0:
        return f"{value:.4f} (change: n/a)"
    change_pct = ((value - baseline) / baseline) * 100.0
    sign = "+" if change_pct >= 0 else ""
    return f"{value:.4f} ({sign}{change_pct:.2f}%)"


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, disable=not is_training):
            if is_training:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            predictions = torch.argmax(logits, dim=1)
            all_targets.extend(labels.tolist())
            all_predictions.extend(predictions.tolist())

    average_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_targets, all_predictions)
    return average_loss, float(accuracy)


def main() -> None:
    digits = load_digits()
    X = digits.images.astype(np.float32) / 16.0
    y = digits.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=SEED,
        stratify=y_train,
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train).unsqueeze(1),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val).unsqueeze(1),
        torch.tensor(y_val, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test).unsqueeze(1),
        torch.tensor(y_test, dtype=torch.long),
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = DigitCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accuracies: list[float] = []
    val_accuracies: list[float] = []

    best_val_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(EPOCHS):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        print(f"Epoch [{epoch + 1}/{EPOCHS}]")
        print(f" Train loss: {train_loss:.6f}, Train accuracy: {train_accuracy:.4f}")
        print(f" Val loss: {val_loss:.6f}, Val accuracy: {val_accuracy:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    test_targets: list[int] = []
    test_predictions: list[int] = []
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            test_targets.extend(labels.tolist())
            test_predictions.extend(predictions.tolist())

    test_accuracy = float(accuracy_score(test_targets, test_predictions))
    test_macro_f1 = float(f1_score(test_targets, test_predictions, average="macro"))
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test macro F1: {test_macro_f1:.4f}")

    majority_class = int(np.bincount(y_train).argmax())
    baseline_predictions = np.full_like(y_test, majority_class)
    baseline_accuracy = float(accuracy_score(y_test, baseline_predictions))
    baseline_macro_f1 = float(f1_score(y_test, baseline_predictions, average="macro"))

    plot_path = Path(__file__).with_name("data-science-13-training-curves.png")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS + 1), train_losses, label="Train loss")
    plt.plot(range(1, EPOCHS + 1), val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Task 13 CNN Loss Curves")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS + 1), train_accuracies, label="Train accuracy")
    plt.plot(range(1, EPOCHS + 1), val_accuracies, label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Task 13 CNN Accuracy Curves")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_path)

    report_path = Path(__file__).with_name("data-science-13-model-report.md")
    best_model_reason = (
        "DigitCNN_v1 is the best model because it improves both test accuracy and "
        "test macro F1 over the baseline."
    )

    baseline_accuracy_cell = format_value_with_change(baseline_accuracy, baseline_accuracy)
    baseline_macro_f1_cell = format_value_with_change(baseline_macro_f1, baseline_macro_f1)
    model_accuracy_cell = format_value_with_change(test_accuracy, baseline_accuracy)
    model_macro_f1_cell = format_value_with_change(test_macro_f1, baseline_macro_f1)

    report_lines = [
        "# Model Report - Task 13",
        "",
        f"Best model: **DigitCNN_v1**. Why: {best_model_reason}",
        "",
        "## Experiment Table",
        "Each row is a hypothesis. Baseline is first. Metrics are on the test set.",
        "",
        "| Hypothesis | Architecture | Epochs | Batch Size | Learning Rate | Optimizer | Test Accuracy (vs baseline) | Test Macro F1 (vs baseline) | Comments |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |",
        f"| Baseline_majority_class | Predict most frequent training class ({majority_class}) | n/a | n/a | n/a | n/a | {baseline_accuracy_cell} | {baseline_macro_f1_cell} | Greedy statistical baseline; no learning. |",
        f"| DigitCNN_v1 | Conv2d(1->16) + ReLU + MaxPool2d(2) + Conv2d(16->32) + ReLU + MaxPool2d(2) + Linear(128->10) | {EPOCHS} | {BATCH_SIZE} | {LEARNING_RATE} | AdamW | {model_accuracy_cell} | {model_macro_f1_cell} | Learns spatial patterns and outperforms baseline on both metrics. |",
        "",
        "## Data And Setup",
        "- Dataset: `sklearn.datasets.load_digits`",
        "- Input: 8x8 grayscale images, 10 classes.",
        "- Splits: train/validation/test from stratified splits.",
        "- Main metric: Accuracy. Secondary metric: Macro F1.",
        "",
        "## Diagrams",
        "- Train vs validation loss: `data-science-13-training-curves.png` (left panel).",
        "- Train vs validation metric (accuracy): `data-science-13-training-curves.png` (right panel).",
        "",
        "## Notes",
        "- Rows are kept in experiment order (baseline first), not sorted.",
        "- The best model row should be highlighted when moving this table to Excel.",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
