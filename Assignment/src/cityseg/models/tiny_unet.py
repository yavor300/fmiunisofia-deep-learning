"""Small custom U-Net used for lightweight experiments."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two Conv-BN-ReLU layers."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class TinyUNet(nn.Module):
    """Two-level encoder-decoder U-Net with concatenation skip connections."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 19,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.encoder1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(base_channels * 2, base_channels)
        self.classifier = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        skip1 = self.encoder1(inputs)
        skip2 = self.encoder2(self.pool1(skip1))
        bottleneck = self.bottleneck(self.pool2(skip2))

        decoded2 = self.up2(bottleneck)
        decoded2 = _match_spatial_size(decoded2, skip2)
        decoded2 = self.decoder2(torch.cat([decoded2, skip2], dim=1))

        decoded1 = self.up1(decoded2)
        decoded1 = _match_spatial_size(decoded1, skip1)
        decoded1 = self.decoder1(torch.cat([decoded1, skip1], dim=1))
        return self.classifier(decoded1)


def _match_spatial_size(inputs: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if inputs.shape[-2:] == reference.shape[-2:]:
        return inputs
    return F.interpolate(inputs, size=reference.shape[-2:], mode="bilinear", align_corners=False)
