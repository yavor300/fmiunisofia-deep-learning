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
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


SEED = 42
EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 0.001


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_data_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[2]
    train_path = root / "DATA" / "water_train.csv"
    test_path = root / "DATA" / "water_test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"could not find required files {train_path} and/or {test_path}"
        )
    return train_path, test_path


def print_distribution(df: pd.DataFrame, label: str) -> None:
    counts = df["Potability"].value_counts().sort_index()
    proportions = counts / counts.sum()
    table = pd.DataFrame({"count": counts, "proportion": proportions})
    table.index.name = "Potability"
    print(f"Distribution of target values in {label} set:")
    print(table)


class WaterPotabilityClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    all_probs = []
    all_targets = []
    total_loss = 0.0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for xb, yb in tqdm(loader):
            if is_training:
                optimizer.zero_grad()

            logits = model(xb).squeeze(1)
            loss = loss_fn(logits, yb)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * xb.size(0)
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(probabilities.tolist())
            all_targets.extend(yb.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    preds = (np.array(all_probs) >= 0.5).astype(int)
    metric = f1_score(np.array(all_targets), preds, zero_division=0)
    return avg_loss, float(metric)


def main() -> None:
    set_seed(SEED)

    train_path, test_path = get_data_paths()
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_split, val_split = train_test_split(
        train_df,
        test_size=0.166,
        random_state=SEED,
        stratify=train_df["Potability"],
    )

    print_distribution(train_split, "training")
    print_distribution(val_split, "validation")
    print_distribution(test_df, "testing")

    feature_columns = [column for column in train_df.columns if column != "Potability"]

    # Simple EDA choice: pick features with highest absolute correlation to target.
    eda_df = train_split.copy()
    for col in feature_columns:
        eda_df[col] = eda_df[col].fillna(eda_df[col].median())
    correlations = (
        eda_df[feature_columns + ["Potability"]]
        .corr(numeric_only=True)["Potability"]
        .drop("Potability")
        .abs()
        .sort_values(ascending=False)
    )
    selected_features = correlations.head(min(7, len(correlations))).index.tolist()

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = imputer.fit_transform(train_split[selected_features])
    X_val = imputer.transform(val_split[selected_features])
    X_test = imputer.transform(test_df[selected_features])

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    y_train = train_split["Potability"].to_numpy(dtype=np.float32)
    y_val = val_split["Potability"].to_numpy(dtype=np.float32)
    y_test = test_df["Potability"].to_numpy(dtype=np.float32)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = WaterPotabilityClassifier(input_dim=len(selected_features))
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    train_losses, val_losses = [], []
    train_metrics, val_metrics = [], []

    for epoch in range(EPOCHS):
        train_loss, train_metric = evaluate_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_metric = evaluate_epoch(model, val_loader, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_metrics.append(train_metric)
        val_metrics.append(val_metric)

        print(f"Epoch [{epoch + 1}/{EPOCHS}]:")
        print(f" Average training loss: {train_loss}")
        print(f" Average validation loss: {val_loss}")
        print(f" Training metric score: {train_metric}")
        print(f" Validation metric score: {val_metric}")

    test_loss, test_metric = evaluate_epoch(model, test_loader, loss_fn)
    print(f"Test loss: {test_loss}")
    print(f"Test F1 score: {test_metric}")

    plot_path = Path(__file__).with_name("data-science-9-training-curves.png")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS + 1), train_losses, label="Train BCE")
    plt.plot(range(1, EPOCHS + 1), val_losses, label="Val BCE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Binary Cross Entropy Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS + 1), train_metrics, label="Train F1")
    plt.plot(range(1, EPOCHS + 1), val_metrics, label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("F1 Score")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_path)

    report_path = Path(__file__).with_name("data-science-9-model-report.md")
    report_lines = [
        "# Task 9 Model Report",
        "",
        "## Data and Feature Selection",
        "- Dataset: `water_train.csv` + `water_test.csv`.",
        "- Missing values handled with median imputation.",
        f"- Selected features by absolute correlation with target: {', '.join(selected_features)}.",
        "",
        "## Model and Training",
        f"- Architecture: Linear({len(selected_features)}->32) + ReLU + Dropout(0.2) + Linear(32->16) + ReLU + Dropout(0.2) + Linear(16->1).",
        f"- Epochs: {EPOCHS}",
        f"- Batch size: {BATCH_SIZE}",
        f"- Learning rate: {LEARNING_RATE}",
        "- Optimizer: AdamW",
        "- Loss: BCEWithLogitsLoss",
        "- Metric: F1 score",
        "",
        "## Final Results",
        f"- Final validation loss: {val_losses[-1]}",
        f"- Final validation F1: {val_metrics[-1]}",
        f"- Test loss: {test_loss}",
        f"- Test F1: {test_metric}",
        f"- Curves saved at `{plot_path.name}`.",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
