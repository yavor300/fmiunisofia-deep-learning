from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


SEED = 42
EPOCHS = 300
BATCH_SIZE = 32
LEARNING_RATE = 0.001


torch.manual_seed(SEED)
np.random.seed(SEED)


class BetterWaterNet(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.activation(self.bn1(self.fc1(x))))
        x = self.dropout(self.activation(self.bn2(self.fc2(x))))
        x = self.activation(self.fc3(x))
        return self.fc4(x)


def prepare_data() -> tuple[DataLoader, DataLoader, DataLoader]:
    root = Path(__file__).resolve().parents[2]
    train_path = root / "DATA" / "water_train.csv"
    test_path = root / "DATA" / "water_test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=SEED,
        stratify=train_df["Potability"],
    )

    feature_columns = [column for column in train_df.columns if column != "Potability"]

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = imputer.fit_transform(train_df[feature_columns])
    X_val = imputer.transform(val_df[feature_columns])
    X_test = imputer.transform(test_df[feature_columns])

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    y_train = train_df["Potability"].to_numpy(dtype=np.float32)
    y_val = val_df["Potability"].to_numpy(dtype=np.float32)
    y_test = test_df["Potability"].to_numpy(dtype=np.float32)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader


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
        for xb, yb in loader:
            if is_training:
                optimizer.zero_grad()

            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * xb.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            all_targets.extend(yb.long().tolist())
            all_predictions.extend(preds.tolist())

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = float(f1_score(all_targets, all_predictions, average="macro"))
    return avg_loss, macro_f1


def evaluate_on_test(model: nn.Module, loader: DataLoader) -> tuple[float, dict[str, float]]:
    model.eval()
    all_targets: list[int] = []
    all_predictions: list[int] = []

    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb).squeeze(1)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            all_targets.extend(yb.long().tolist())
            all_predictions.extend(preds.tolist())

    macro_f1 = float(f1_score(all_targets, all_predictions, average="macro"))
    _, _, f1_values, _ = precision_recall_fscore_support(
        all_targets,
        all_predictions,
        labels=[0, 1],
        zero_division=0,
    )
    per_class = {
        "not potable (0)": float(f1_values[0]),
        "potable (1)": float(f1_values[1]),
    }
    return macro_f1, per_class


def main() -> None:
    train_loader, val_loader, test_loader = prepare_data()
    model = BetterWaterNet(input_size=9)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    train_losses: list[float] = []
    val_losses: list[float] = []
    train_f1s: list[float] = []
    val_f1s: list[float] = []

    best_val_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    patience = 35
    epochs_without_improvement = 0

    for epoch in tqdm(range(EPOCHS)):
        train_loss, train_f1 = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_f1 = run_epoch(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_macro_f1, per_class_f1 = evaluate_on_test(model, test_loader)

    print(f"Best validation macro F1: {best_val_f1}")
    print(f"Test macro F1: {test_macro_f1}")
    print(f"Per-class F1: {per_class_f1}")

    figure_path = Path(__file__).with_name("data-science-4-training-curves.png")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCEWithLogits loss")
    plt.title("Water Potability Loss Curves")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_f1s, label="Train macro F1")
    plt.plot(val_f1s, label="Validation macro F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title("Water Potability F1 Curves")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(figure_path)


if __name__ == "__main__":
    main()
