from __future__ import annotations

import unittest

import torch
from torch import nn

from src.cityseg.models.factory import MajorityBaselineModel, _segformer_model_id, create_model
from src.cityseg.models.tiny_unet import TinyUNet


class TestTinyUNetForward(unittest.TestCase):
    def test_when_input_is_image_batch_then_logits_keep_spatial_shape(self) -> None:
        model = TinyUNet(in_channels=3, num_classes=19, base_channels=8)
        inputs = torch.randn(2, 3, 32, 32)

        logits = model(inputs)

        self.assertEqual(tuple(logits.shape), (2, 19, 32, 32))


class TestCreateModel(unittest.TestCase):
    def test_when_architecture_is_tiny_unet_then_tiny_unet_is_returned(self) -> None:
        config = {
            "architecture": "tiny_unet",
            "in_channels": 3,
            "num_classes": 19,
            "base_channels": 8,
        }

        model = create_model(config)

        self.assertIsInstance(model, TinyUNet)

    def test_when_architecture_is_majority_baseline_then_baseline_model_is_returned(self) -> None:
        config = {"architecture": "majority_baseline", "num_classes": 19, "majority_class": 0}

        model = create_model(config)
        logits = model(torch.randn(1, 3, 16, 16))

        self.assertIsInstance(model, MajorityBaselineModel)
        self.assertEqual(tuple(logits.shape), (1, 19, 16, 16))

    def test_when_architecture_is_unet_then_smp_model_returns_expected_shape(self) -> None:
        config = {
            "architecture": "unet",
            "encoder_name": "resnet18",
            "encoder_weights": None,
            "in_channels": 3,
            "num_classes": 19,
        }

        model = create_model(config)
        model.eval()
        with torch.no_grad():
            logits = model(torch.randn(1, 3, 64, 64))

        self.assertEqual(tuple(logits.shape), (1, 19, 64, 64))

    def test_when_architecture_is_unknown_then_value_error_is_raised(self) -> None:
        with self.assertRaises(ValueError):
            create_model({"architecture": "not_a_model", "num_classes": 19})

    def test_when_config_contains_model_key_then_nested_model_config_is_used(self) -> None:
        config = {
            "model": {
                "architecture": "tiny_unet",
                "in_channels": 3,
                "num_classes": 19,
                "base_channels": 8,
            }
        }

        model = create_model(config)

        self.assertIsInstance(model, nn.Module)

    def test_when_segformer_encoder_is_mit_b1_then_hf_model_id_is_returned(self) -> None:
        model_id = _segformer_model_id("mit_b1", "imagenet")

        self.assertEqual(model_id, "nvidia/mit-b1")


if __name__ == "__main__":
    unittest.main()
