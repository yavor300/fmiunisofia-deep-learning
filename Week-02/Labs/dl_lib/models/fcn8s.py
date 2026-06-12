from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import models


def _make_bilinear_kernel(kernel_size: int) -> Tensor:
    factor = (kernel_size + 1) // 2
    center = factor - 1 if kernel_size % 2 == 1 else factor - 0.5
    og = torch.arange(kernel_size, dtype=torch.float32)
    filt = (1 - torch.abs(og - center) / factor)
    return torch.outer(filt, filt)


def _init_bilinear_deconv(layer: nn.ConvTranspose2d) -> None:
    if layer.in_channels != layer.out_channels:
        raise ValueError("Bilinear initialization requires matching in/out channels.")

    kernel = _make_bilinear_kernel(layer.kernel_size[0])
    weight = torch.zeros_like(layer.weight.data)
    for channel in range(layer.in_channels):
        weight[channel, channel] = kernel
    layer.weight.data.copy_(weight)


class FCN8s(nn.Module):
    SUPPORTED_BACKBONES = ("vgg16", "resnet50")

    def __init__(self, backbone: str = "vgg16", *, num_classes: int = 21) -> None:
        super().__init__()

        if backbone not in self.SUPPORTED_BACKBONES:
            supported = ", ".join(self.SUPPORTED_BACKBONES)
            raise ValueError(f"Unsupported backbone '{backbone}'. Supported: {supported}")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        self.backbone_name = backbone
        self.num_classes = num_classes

        if backbone == "vgg16":
            self.features = self._build_vgg16_backbone()
            pool3_channels, pool4_channels, pool5_channels = 256, 512, 512
        else:
            self.features = self._build_resnet50_backbone()
            pool3_channels, pool4_channels, pool5_channels = 512, 1024, 2048

        self.score_pool3 = nn.Conv2d(pool3_channels, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(pool4_channels, num_classes, kernel_size=1)
        self.score_pool5 = nn.Conv2d(pool5_channels, num_classes, kernel_size=1)

        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, padding=4, bias=False)

        _init_bilinear_deconv(self.upscore2)
        _init_bilinear_deconv(self.upscore_pool4)
        _init_bilinear_deconv(self.upscore8)

    @staticmethod
    def _build_vgg16_backbone() -> nn.Module:
        return models.vgg16(weights=None).features

    @staticmethod
    def _build_resnet50_backbone() -> nn.Module:
        return models.resnet50(weights=None)

    def _forward_vgg16(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        pool3 = pool4 = pool5 = x
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx == 16:
                pool3 = x
            elif idx == 23:
                pool4 = x
            elif idx == 30:
                pool5 = x
        return pool3, pool4, pool5

    def _forward_resnet50(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        backbone = self.features
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)

        x = backbone.layer1(x)
        pool3 = backbone.layer2(x)
        pool4 = backbone.layer3(pool3)
        pool5 = backbone.layer4(pool4)
        return pool3, pool4, pool5

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.ndim != 4:
            raise ValueError("Input tensor must have shape [N, C, H, W] or [C, H, W]")

        input_hw = x.shape[-2:]

        if self.backbone_name == "vgg16":
            pool3, pool4, pool5 = self._forward_vgg16(x)
        else:
            pool3, pool4, pool5 = self._forward_resnet50(x)

        score_pool5 = self.score_pool5(pool5)
        upscore2 = self.upscore2(score_pool5)
        upscore2 = F.interpolate(upscore2, size=pool4.shape[-2:], mode="bilinear", align_corners=False)

        score_pool4 = self.score_pool4(pool4)
        fuse_pool4 = upscore2 + score_pool4

        upscore_pool4 = self.upscore_pool4(fuse_pool4)
        upscore_pool4 = F.interpolate(upscore_pool4, size=pool3.shape[-2:], mode="bilinear", align_corners=False)

        score_pool3 = self.score_pool3(pool3)
        fuse_pool3 = upscore_pool4 + score_pool3

        logits = self.upscore8(fuse_pool3)
        logits = F.interpolate(logits, size=input_hw, mode="bilinear", align_corners=False)
        return logits


def load(model_name: str, **kwargs: Any) -> FCN8s:
    normalized = model_name.strip().lower()
    if normalized != "fcn8s":
        raise ValueError(f"Unknown model '{model_name}'. Supported models: ['fcn8s']")

    kwargs.pop("pretrained", None)
    return FCN8s(**kwargs)


__all__ = ["FCN8s", "load"]
