from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


SEED = 42
BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS = 20


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_data_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    data_path = root / "DATA" / "ds_salaries.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"could not find dataset at {data_path}")
    return data_path


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


class SalaryRegressor(nn.Module):
    def __init__(self, input_dim: int, activation: nn.Module):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            activation,
            nn.Linear(32, 16),
            activation,
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_model(model: nn.Module, dataloader: DataLoader) -> list[float]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    epoch_losses: list[float] = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for xb, yb in tqdm(dataloader):
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{EPOCHS}]: Average loss: {avg_loss}")

    return epoch_losses


def main() -> None:
    set_seed(SEED)

    df = pd.read_csv(get_data_path())

    feature_columns = [
        "experience_level",
        "employment_type",
        "remote_ratio",
        "company_size",
    ]
    target_column = "salary_in_usd"

    X = df[feature_columns]
    y = df[[target_column]]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                make_one_hot_encoder(),
                ["experience_level", "employment_type", "company_size"],
            ),
            ("num", MinMaxScaler(), ["remote_ratio"]),
        ]
    )

    X_processed = preprocessor.fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_processed)
    y_scaled = StandardScaler().fit_transform(y)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    models = {
        "nn_with_sigmoid": SalaryRegressor(X_tensor.shape[1], nn.Sigmoid()),
        "nn_with_relu": SalaryRegressor(X_tensor.shape[1], nn.ReLU()),
        "nn_with_leakyrelu": SalaryRegressor(X_tensor.shape[1], nn.LeakyReLU()),
    }

    all_losses: dict[str, list[float]] = {}
    final_losses: dict[str, float] = {}

    for name, model in models.items():
        print(f"\nTraining model: {name}")
        losses = train_model(model, dataloader)
        all_losses[name] = losses
        final_losses[name] = losses[-1]

    best_model = min(final_losses, key=final_losses.get)
    print(f"Lowest loss of {final_losses[best_model]} was achieved by model {best_model}.")

    plt.figure(figsize=(10, 6))
    for name, losses in all_losses.items():
        plt.plot(range(1, EPOCHS + 1), losses, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Average Training MSE Loss")
    plt.title("Activation Comparison on Salary Prediction")
    plt.legend()
    plt.grid(True)
    plot_path = Path(__file__).with_name("data-science-6-losses.png")
    plt.tight_layout()
    plt.savefig(plot_path)

    report_path = Path(__file__).with_name("data-science-6-model-report.md")
    report_lines = [
        "# Task 6 Model Report",
        "",
        "## Setup",
        "- Features: `experience_level`, `employment_type`, `remote_ratio`, `company_size`.",
        "- Target: `salary_in_usd`.",
        "- Categorical encoding: One-hot encoding.",
        "- Normalization: StandardScaler on features and target.",
        f"- Training: {EPOCHS} epochs, batch size {BATCH_SIZE}, AdamW, lr={LEARNING_RATE}.",
        "",
        "## Final Epoch Losses",
    ]
    for name, value in final_losses.items():
        report_lines.append(f"- {name}: {value}")
    report_lines.extend(
        [
            "",
            f"## Best Model\n- {best_model} ({final_losses[best_model]}).",
            f"\nPlot saved at `{plot_path.name}`.",
        ]
    )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
