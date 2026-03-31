from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


torch.manual_seed(42)


class WaterDataset(Dataset):
    def __init__(self, csv_path: Path):
        self.data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.float64]:
        row = self.data[index]
        return row[:-1], row[-1]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


def train_model(
    dataloader_train: DataLoader,
    optimizer: torch.optim.Optimizer,
    net: nn.Module,
    num_epochs: int,
    create_plot: bool = False,
) -> float:
    criterion = nn.BCELoss()
    epoch_losses: list[float] = []

    net.train()
    for _ in tqdm(range(num_epochs)):
        running_loss = 0.0

        for features, labels in dataloader_train:
            features = features.float()
            labels = labels.float()

            optimizer.zero_grad()
            predictions = net(features).squeeze(1)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_losses.append(running_loss / len(dataloader_train))

    average_loss = float(np.mean(epoch_losses))
    print(f"Average loss: {average_loss}")

    if create_plot:
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, num_epochs + 1), epoch_losses)
        plt.xlabel("Epoch")
        plt.ylabel("Average BCE loss per epoch")
        plt.title("Task 02 Loss Per Epoch")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(Path(__file__).with_name("data-science-2-loss-per-epoch.png"))

    return average_loss


def evaluate_f1(net: nn.Module, dataloader_test: DataLoader) -> float:
    net.eval()
    all_targets: list[int] = []
    all_predictions: list[int] = []

    with torch.no_grad():
        for features, labels in dataloader_test:
            features = features.float()
            predictions = net(features).squeeze(1)
            predicted_classes = (predictions >= 0.5).long()

            all_targets.extend(labels.long().tolist())
            all_predictions.extend(predicted_classes.tolist())

    return float(f1_score(all_targets, all_predictions))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    train_path = root / "DATA" / "water_train.csv"
    test_path = root / "DATA" / "water_test.csv"

    train_dataset = WaterDataset(train_path)
    test_dataset = WaterDataset(test_path)

    dataloader_train = DataLoader(train_dataset, batch_size=8, shuffle=True)
    dataloader_test = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for optimizer_name, optimizer_class in [
        ("SGD", torch.optim.SGD),
        ("RMSprop", torch.optim.RMSprop),
        ("Adam", torch.optim.Adam),
        ("AdamW", torch.optim.AdamW),
    ]:
        torch.manual_seed(42)
        net = Net()
        optimizer = optimizer_class(net.parameters(), lr=0.001)
        print(f"Using the {optimizer_name} optimizer:")
        train_model(
            dataloader_train=dataloader_train,
            optimizer=optimizer,
            net=net,
            num_epochs=10,
        )

    torch.manual_seed(42)
    net = Net()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
    train_model(
        dataloader_train=dataloader_train,
        optimizer=optimizer,
        net=net,
        num_epochs=1000,
        create_plot=True,
    )

    test_f1 = evaluate_f1(net, dataloader_test)
    print(f"\nF1 score on test set: {test_f1}")

    # The model does not perform especially well, F1 around 0.55 is moderately
    # useful for a binary classifier and leaves a lot of room for improvement


if __name__ == "__main__":
    main()
