from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


SEED = 42
EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 0.001


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def format_delta(current: float, baseline: float, higher_is_better: bool) -> str:
    if abs(baseline) < 1e-12:
        if abs(current - baseline) < 1e-12:
            return "0.00%"
        return "N/A"
    if higher_is_better:
        change = (current - baseline) / abs(baseline) * 100.0
    else:
        change = (baseline - current) / abs(baseline) * 100.0
    return f"{change:+.2f}%"


def load_mnist_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from datasets import load_dataset

        ds = load_dataset("ylecun/mnist")
        train_images = np.stack([np.asarray(img, dtype=np.float32) for img in ds["train"]["image"]])
        train_labels = np.asarray(ds["train"]["label"], dtype=np.int64)
        test_images = np.stack([np.asarray(img, dtype=np.float32) for img in ds["test"]["image"]])
        test_labels = np.asarray(ds["test"]["label"], dtype=np.int64)
    except Exception as exc:
        raise RuntimeError(
            "failed to load MNIST from Hugging Face dataset 'ylecun/mnist'"
        ) from exc

    train_images = train_images.reshape(train_images.shape[0], -1) / 255.0
    test_images = test_images.reshape(test_images.shape[0], -1) / 255.0
    return train_images, train_labels, test_images, test_labels


class MNISTLinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for xb, yb in tqdm(loader):
            if is_training:
                optimizer.zero_grad()

            logits = model(xb)
            loss = loss_fn(logits, yb)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == yb).sum().item()
            total_samples += xb.size(0)

    return total_loss / total_samples, total_correct / total_samples


def main() -> None:
    set_seed(SEED)

    X_train_full, y_train_full, X_test, y_test = load_mnist_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.1,
        random_state=SEED,
        stratify=y_train_full,
    )

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = MNISTLinearClassifier()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(EPOCHS):
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch + 1}/{EPOCHS}]")
        print(f" Train loss: {train_loss:.6f}, Train accuracy: {train_acc:.4f}")
        print(f" Val loss: {val_loss:.6f}, Val accuracy: {val_acc:.4f}")

    test_loss, test_acc = run_epoch(model, test_loader, loss_fn)
    print(f"Test loss: {test_loss:.6f}")
    print(f"Test accuracy: {test_acc:.4f}")

    majority_class = int(np.bincount(y_train).argmax())
    baseline_preds = np.full_like(y_test, fill_value=majority_class)
    baseline_acc = float((baseline_preds == y_test).mean())
    baseline_logits = torch.zeros((len(y_test), 10), dtype=torch.float32)
    baseline_logits[:, majority_class] = 1.0
    baseline_loss = float(loss_fn(baseline_logits, torch.tensor(y_test, dtype=torch.long)).item())

    plot_path = Path(__file__).with_name("data-science-12-training-curves.png")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS + 1), train_losses, label="Train loss")
    plt.plot(range(1, EPOCHS + 1), val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("MNIST Loss Curves")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS + 1), train_accuracies, label="Train accuracy")
    plt.plot(range(1, EPOCHS + 1), val_accuracies, label="Val accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("MNIST Accuracy Curves")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_path)

    report_path = Path(__file__).with_name("data-science-12-model-report.md")
    report_lines = [
        "# Model Report - Task 12",
        "",
        "## Context",
        "- MNIST loaded from Hugging Face: `ylecun/mnist`.",
        "- Images flattened to 784 features and normalized to [0, 1].",
        "",
        "## Experiments Table",
        "| Hypothesis | Architecture / Strategy | Epochs | Batch Size | Learning Rate | Optimizer | Test CE Loss (Δ vs baseline) | Test Accuracy (Δ vs baseline) | Comments |",
        "|---|---|---:|---:|---:|---|---|---|---|",
        f"| baseline_majority_class | Predict the most common class ({majority_class}) for every image | N/A | N/A | N/A | N/A | {baseline_loss:.6f} (+0.00%) | {baseline_acc:.4f} (+0.00%) | Baseline model required by reporting standard. |",
        (
            "| nn_linear_mnist | Linear(784->256)->ReLU->Linear(256->128)->ReLU->Linear(128->10) | "
            f"{EPOCHS} | {BATCH_SIZE} | {LEARNING_RATE} | AdamW | "
            f"{test_loss:.6f} ({format_delta(test_loss, baseline_loss, higher_is_better=False)}) | "
            f"{test_acc:.4f} ({format_delta(test_acc, baseline_acc, higher_is_better=True)}) | "
            "Trained model with strong improvement over baseline. |"
        ),
        "",
        "## Best Model",
        "Best model: **nn_linear_mnist**, because it improved both core test metrics versus baseline "
        f"(CE loss `{test_loss:.6f}` vs `{baseline_loss:.6f}`, accuracy `{test_acc:.4f}` vs `{baseline_acc:.4f}`).",
        "",
        "## Diagrams",
        f"- Train vs validation loss curve: `{plot_path.name}` (left panel).",
        f"- Train vs validation accuracy curve: `{plot_path.name}` (right panel).",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
