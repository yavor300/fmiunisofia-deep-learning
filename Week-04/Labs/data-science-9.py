from __future__ import annotations

import torch
import torch.nn as nn


class FourLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(5, 50)
        self.layer2 = nn.Linear(50, 2)
        self.layer3 = nn.Linear(2, 2)
        self.layer4 = nn.Linear(2, 1)

        self._init_uniform()
        self._freeze_first_two_layers()

    def _init_uniform(self) -> None:
        for parameter in self.parameters():
            nn.init.uniform_(parameter, a=0.0, b=1.0)

    def _freeze_first_two_layers(self) -> None:
        for parameter in self.layer1.parameters():
            parameter.requires_grad = False
        for parameter in self.layer2.parameters():
            parameter.requires_grad = False


def print_first_five_neurons(model: FourLayerModel) -> None:
    layers = [model.layer1, model.layer2, model.layer3, model.layer4]

    for index, layer in enumerate(layers, start=1):
        print(f"\nLayer {index}:")
        print(layer.weight[:5])
        print(layer.bias[:5])


def main() -> None:
    model = FourLayerModel()
    print_first_five_neurons(model)


if __name__ == "__main__":
    main()
