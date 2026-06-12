from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.initial = ConvBlock(in_channels, 64)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc1 = ConvBlock(64, 128)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = ConvBlock(128, 256)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = ConvBlock(256, 512)

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.initial(x)

        x1 = self.enc1(self.pool1(x0))
        x2 = self.enc2(self.pool2(x1))
        x3 = self.enc3(self.pool3(x2))

        y = self.up1(x3)
        y = torch.cat([y, x2], dim=1)
        y = self.dec1(y)

        y = self.up2(y)
        y = torch.cat([y, x1], dim=1)
        y = self.dec2(y)

        y = self.up3(y)
        y = torch.cat([y, x0], dim=1)
        y = self.dec3(y)

        return self.final_conv(y)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def main() -> None:
    model = UNet(in_channels=16, out_channels=5)
    total_params = count_parameters(model)
    print(f"Total number of parameters in UNet(16, 5): {total_params:,}")


if __name__ == "__main__":
    main()
