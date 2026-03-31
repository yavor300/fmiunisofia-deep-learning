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


class OmniglotMultiOutputDataset(Dataset):
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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        image_path, alphabet_name, character_name = self.samples[index]
        image = Image.open(image_path).convert("L")
        image_tensor = self.transform(image) if self.transform is not None else transforms.ToTensor()(image)

        alphabet_label = self.alphabet_to_index[alphabet_name]
        character_label = self.character_to_label[f"{alphabet_name}/{character_name}"]
        return image_tensor, alphabet_label, character_label


class SharedImageEncoder(nn.Module):
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
        return self.activation(self.fc(x))


class MultiOutputOmniglotModel(nn.Module):
    def __init__(self, num_characters: int, num_alphabets: int):
        super().__init__()
        self.encoder = SharedImageEncoder()
        self.dropout = nn.Dropout(0.2)
        self.character_head = nn.Linear(256, num_characters)
        self.alphabet_head = nn.Linear(256, num_alphabets)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.dropout(self.encoder(image))
        character_logits = self.character_head(features)
        alphabet_logits = self.alphabet_head(features)
        return character_logits, alphabet_logits


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion_characters: nn.Module,
    criterion_alphabets: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_samples = 0
    correct_characters = 0
    correct_alphabets = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for images, alphabet_labels, character_labels in tqdm(loader):
            if is_training:
                optimizer.zero_grad()

            character_logits, alphabet_logits = model(images)
            loss_char = criterion_characters(character_logits, character_labels)
            loss_alpha = criterion_alphabets(alphabet_logits, alphabet_labels)
            loss = loss_char + loss_alpha

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            character_preds = torch.argmax(character_logits, dim=1)
            alphabet_preds = torch.argmax(alphabet_logits, dim=1)
            correct_characters += (character_preds == character_labels).sum().item()
            correct_alphabets += (alphabet_preds == alphabet_labels).sum().item()

    average_loss = total_loss / total_samples
    character_accuracy = correct_characters / total_samples
    alphabet_accuracy = correct_alphabets / total_samples
    return average_loss, character_accuracy, alphabet_accuracy


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

    train_dataset = OmniglotMultiOutputDataset(train_dir, transform=transform)
    val_dataset = OmniglotMultiOutputDataset(
        val_dir,
        transform=transform,
        alphabet_to_index=train_dataset.alphabet_to_index,
        character_to_label=train_dataset.character_to_label,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = MultiOutputOmniglotModel(
        num_characters=train_dataset.num_characters,
        num_alphabets=train_dataset.num_alphabets,
    )
    criterion_characters = nn.CrossEntropyLoss()
    criterion_alphabets = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    train_losses: list[float] = []
    val_losses: list[float] = []
    train_char_scores: list[float] = []
    val_char_scores: list[float] = []
    train_alpha_scores: list[float] = []
    val_alpha_scores: list[float] = []

    for epoch in range(EPOCHS):
        train_loss, train_char_acc, train_alpha_acc = run_epoch(
            model,
            train_loader,
            criterion_characters,
            criterion_alphabets,
            optimizer,
        )
        val_loss, val_char_acc, val_alpha_acc = run_epoch(
            model,
            val_loader,
            criterion_characters,
            criterion_alphabets,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_char_scores.append(train_char_acc)
        val_char_scores.append(val_char_acc)
        train_alpha_scores.append(train_alpha_acc)
        val_alpha_scores.append(val_alpha_acc)

        print(f"Epoch [{epoch + 1}/{EPOCHS}]:")
        print(f" Average training loss: {train_loss}")
        print(f" Average validation loss: {val_loss}")
        print(f" Training metric score characters: {train_char_acc}")
        print(f" Validation metric score characters: {val_char_acc}")
        print(f" Training metric score alphabets: {train_alpha_acc}")
        print(f" Validation metric score alphabets: {val_alpha_acc}")

    plot_path = Path(__file__).with_name("data-science-12-training-curves.png")
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.plot(range(1, EPOCHS + 1), train_losses, label="Training loss")
    plt.plot(range(1, EPOCHS + 1), val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Combined Loss")
    plt.title("Task 12 Loss Curves")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(range(1, EPOCHS + 1), train_char_scores, label="Training")
    plt.plot(range(1, EPOCHS + 1), val_char_scores, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Character Accuracy")
    plt.title("Character Head Metric")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(range(1, EPOCHS + 1), train_alpha_scores, label="Training")
    plt.plot(range(1, EPOCHS + 1), val_alpha_scores, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Alphabet Accuracy")
    plt.title("Alphabet Head Metric")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_path)


if __name__ == "__main__":
    main()
