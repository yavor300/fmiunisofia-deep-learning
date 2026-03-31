from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class WaterDataset(Dataset):
    def __init__(self, csv_path: Path):
        self.data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.float64]:
        row = self.data[index]
        features = row[:-1]
        label = row[-1]
        return features, label


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    train_path = root / "DATA" / "water_train.csv"

    dataset = WaterDataset(train_path)

    print(f"Number of instances: {len(dataset)}")
    print(f"Fifth item: {dataset[4]}")

    torch.manual_seed(42)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch_features, batch_labels = next(iter(dataloader))
    print(batch_features, batch_labels)


if __name__ == "__main__":
    main()
