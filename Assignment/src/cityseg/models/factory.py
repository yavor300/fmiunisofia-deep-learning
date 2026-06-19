"""Model factory for Cityscapes segmentation experiments."""

from __future__ import annotations

import argparse
from typing import Any

import torch
from torch import nn

from src.cityseg.config import load_config
from src.cityseg.env import load_env_file
from src.cityseg.models.tiny_unet import TinyUNet


class MajorityBaselineModel(nn.Module):
    """Module wrapper that emits constant logits for the majority class."""

    def __init__(self, num_classes: int = 19, majority_class: int = 0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.majority_class = majority_class

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = inputs.shape
        logits = inputs.new_full((batch_size, self.num_classes, height, width), -20.0)
        logits[:, self.majority_class, :, :] = 20.0
        return logits


def create_model(config: dict[str, Any]) -> nn.Module:
    """Instantiate a segmentation model from a config mapping."""
    load_env_file()
    model_config = _model_config(config)
    architecture = str(model_config.get("architecture", "unet")).lower()
    in_channels = int(model_config.get("in_channels", 3))
    num_classes = int(model_config.get("num_classes", 19))

    if architecture == "majority_baseline":
        return MajorityBaselineModel(
            num_classes=num_classes,
            majority_class=int(model_config.get("majority_class", 0)),
        )
    if architecture == "tiny_unet":
        return TinyUNet(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=int(model_config.get("base_channels", 32)),
        )
    if architecture == "segformer":
        raise NotImplementedError("SegFormer requires a separate implementation path.")

    return _create_smp_model(
        architecture=architecture,
        model_config=model_config,
        in_channels=in_channels,
        num_classes=num_classes,
    )


def _create_smp_model(
    architecture: str,
    model_config: dict[str, Any],
    in_channels: int,
    num_classes: int,
) -> nn.Module:
    try:
        import segmentation_models_pytorch as smp
    except ImportError as error:
        raise ImportError(
            "segmentation-models-pytorch is required for SMP architectures."
        ) from error

    smp_classes = {
        "unet": "Unet",
        "deeplabv3plus": "DeepLabV3Plus",
        "fpn": "FPN",
        "pspnet": "PSPNet",
        "unetplusplus": "UnetPlusPlus",
    }
    if architecture not in smp_classes:
        supported = ", ".join(["majority_baseline", "tiny_unet", *smp_classes])
        raise ValueError(f"Unsupported architecture '{architecture}'. Supported: {supported}")

    class_name = smp_classes[architecture]
    if not hasattr(smp, class_name):
        raise ValueError(f"SMP architecture '{class_name}' is not supported by this SMP version.")

    model_class = getattr(smp, class_name)
    return model_class(
        encoder_name=model_config.get("encoder_name", "resnet34"),
        encoder_weights=model_config.get("encoder_weights"),
        in_channels=in_channels,
        classes=num_classes,
    )


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("model")
    if isinstance(nested, dict):
        return nested
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Instantiate a model and print output shape.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to a YAML config.")
    parser.add_argument("--height", type=int, default=64, help="Input image height.")
    parser.add_argument("--width", type=int, default=64, help="Input image width.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model = create_model(config)
    model.eval()
    in_channels = int(_model_config(config).get("in_channels", 3))
    inputs = torch.randn(1, in_channels, args.height, args.width)
    with torch.no_grad():
        logits = model(inputs)
    print(f"{model.__class__.__name__}: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
