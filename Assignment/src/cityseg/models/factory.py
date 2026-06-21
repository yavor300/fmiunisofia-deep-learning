"""Model factory for Cityscapes segmentation experiments."""

from __future__ import annotations

import argparse
from typing import Any

import torch
import torch.nn.functional as F
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


class SegFormerSegmentationModel(nn.Module):
    """Hugging Face SegFormer wrapper that returns `[B, C, H, W]` logits."""

    def __init__(
        self,
        encoder_name: str,
        encoder_weights: str | None,
        num_classes: int,
    ) -> None:
        super().__init__()
        try:
            from transformers import SegformerForSemanticSegmentation
        except ImportError as error:
            raise ImportError(
                "The optional SegFormer experiment requires `transformers`. "
                "Install dependencies with `make install` before running it."
            ) from error

        model_id = _segformer_model_id(encoder_name, encoder_weights)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_id,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=inputs)
        return F.interpolate(
            outputs.logits,
            size=inputs.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


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
        if in_channels != 3:
            raise ValueError("SegFormer supports only 3-channel RGB inputs.")
        return SegFormerSegmentationModel(
            encoder_name=str(model_config.get("encoder_name", "mit_b1")),
            encoder_weights=model_config.get("encoder_weights"),
            num_classes=num_classes,
        )

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


def _segformer_model_id(encoder_name: str, encoder_weights: str | None) -> str:
    normalized = encoder_name.replace("-", "_").lower()
    if encoder_weights in {None, "none", ""}:
        return f"nvidia/{normalized.replace('_', '-')}"
    model_ids = {
        "mit_b0": "nvidia/mit-b0",
        "mit_b1": "nvidia/mit-b1",
        "mit_b2": "nvidia/mit-b2",
        "mit_b3": "nvidia/mit-b3",
        "mit_b4": "nvidia/mit-b4",
        "mit_b5": "nvidia/mit-b5",
    }
    if normalized not in model_ids:
        supported = ", ".join(sorted(model_ids))
        raise ValueError(f"Unsupported SegFormer encoder '{encoder_name}'. Supported: {supported}")
    return model_ids[normalized]


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
