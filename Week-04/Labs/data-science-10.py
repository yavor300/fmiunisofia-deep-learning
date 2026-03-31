from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class OmniglotDataset(Dataset):
    def __init__(self, root_dir: Path, transform: transforms.Compose | None = None):
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

        alphabets = sorted({alphabet for _, alphabet, _ in self.samples})
        characters = sorted({f"{alphabet}/{character}" for _, alphabet, character in self.samples})

        self.alphabet_to_index = {alphabet: index for index, alphabet in enumerate(alphabets)}
        self.character_to_label = {character: index for index, character in enumerate(characters)}
        self.num_alphabets = len(self.alphabet_to_index)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        image_path, alphabet_name, character_name = self.samples[index]
        image = Image.open(image_path).convert("L")
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        alphabet_index = self.alphabet_to_index[alphabet_name]
        alphabet_one_hot = torch.zeros(self.num_alphabets, dtype=torch.float32)
        alphabet_one_hot[alphabet_index] = 1.0

        character_key = f"{alphabet_name}/{character_name}"
        character_label = self.character_to_label[character_key]
        return image_tensor, alphabet_one_hot, character_label


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    train_dir = root / "DATA" / "omniglot_train"

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((64, 64)),
        ]
    )

    train_dataset = OmniglotDataset(train_dir, transform=transform)
    print(f"Number of instances: {len(train_dataset)}")

    last_item = train_dataset[-1]
    print(f"Last item: {last_item}")
    print(f"Shape of the last image: {last_item[0].shape}")


if __name__ == "__main__":
    main()
