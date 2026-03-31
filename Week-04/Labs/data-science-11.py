from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm


SEED = 42
EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 0.001


torch.manual_seed(SEED)


class OmniglotDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        transform: transforms.Compose | None = None,
        alphabet_to_index: dict[str, int] | None = None,
        character_to_label: dict[str, int] | None = None,
    ):
        self.root_dir = root_dir
        self.transform = transform

        self.samples: list[tuple[Path, str, str]] = []
        for alphabet_path in sorted(root_dir.iterdir()):
            if not alphabet_path.is_dir() or alphabet_path.name.startswith("."):
                continue
            for character_path in sorted(alphabet_path.iterdir()):
                if not character_path.is_dir() or character_path.name.startswith("."):
                    continue
                for image_path in sorted(character_path.glob("*.png")):
                    self.samples.append((image_path, alphabet_path.name, character_path.name))

        if alphabet_to_index is None:
            alphabets = sorted({alphabet for _, alphabet, _ in self.samples})
            self.alphabet_to_index = {alphabet: idx for idx, alphabet in enumerate(alphabets)}
        else:
            self.alphabet_to_index = alphabet_to_index

        if character_to_label is None:
            characters = sorted({f"{alphabet}/{character}" for _, alphabet, character in self.samples})
            self.character_to_label = {character: idx for idx, character in enumerate(characters)}
        else:
            self.character_to_label = character_to_label

        self.num_alphabets = len(self.alphabet_to_index)
        self.num_characters = len(self.character_to_label)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        image_path, alphabet_name, character_name = self.samples[index]

        image = Image.open(image_path).convert("L")
        image_tensor = self.transform(image) if self.transform is not None else transforms.ToTensor()(image)

        alphabet_idx = self.alphabet_to_index[alphabet_name]
        alphabet_one_hot = torch.zeros(self.num_alphabets, dtype=torch.float32)
        alphabet_one_hot[alphabet_idx] = 1.0

        character_key = f"{alphabet_name}/{character_name}"
        character_label = self.character_to_label[character_key]
        return image_tensor, alphabet_one_hot, character_label


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.activation = nn.ReLU()
        self.fc = nn.Linear(128 * 8 * 8, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = self.pool(self.activation(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc(x))
        return x


class AlphabetEncoder(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return x


class MultiInputOmniglotClassifier(nn.Module):
    def __init__(self, num_alphabets: int, num_characters: int):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.alphabet_encoder = AlphabetEncoder(num_alphabets)
        self.fc1 = nn.Linear(320, 512)
        self.fc2 = nn.Linear(512, num_characters)
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.ReLU()

    def forward(self, image: torch.Tensor, alphabet_one_hot: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(image)
        alphabet_features = self.alphabet_encoder(alphabet_one_hot)
        x = torch.cat([image_features, alphabet_features], dim=1)
        x = self.dropout(self.activation(self.fc1(x)))
        return self.fc2(x)


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
    total_correct = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for images, alphabets, labels in tqdm(loader):
            if is_training:
                optimizer.zero_grad()

            logits = model(images, alphabets)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += images.size(0)

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return average_loss, accuracy


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    train_dir = root / "DATA" / "omniglot_train"
    val_dir = root / "DATA" / "omniglot_test"

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((64, 64)),
        ]
    )

    train_dataset = OmniglotDataset(train_dir, transform=transform)
    val_dataset = OmniglotDataset(
        val_dir,
        transform=transform,
        alphabet_to_index=train_dataset.alphabet_to_index,
        character_to_label=train_dataset.character_to_label,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = MultiInputOmniglotClassifier(
        num_alphabets=train_dataset.num_alphabets,
        num_characters=train_dataset.num_characters,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accuracies: list[float] = []
    val_accuracies: list[float] = []

    for epoch in range(EPOCHS):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch + 1}/{EPOCHS}]:")
        print(f" Average training loss: {train_loss}")
        print(f" Average validation loss: {val_loss}")
        print(f" Training metric score: {train_acc}")
        print(f" Validation metric score: {val_acc}")

    plot_path = Path(__file__).with_name("data-science-11-training-curves.png")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS + 1), train_losses, label="Training loss")
    plt.plot(range(1, EPOCHS + 1), val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Task 11 Loss Curves")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS + 1), train_accuracies, label="Training accuracy")
    plt.plot(range(1, EPOCHS + 1), val_accuracies, label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Task 11 Accuracy Curves")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_path)


if __name__ == "__main__":
    main()
