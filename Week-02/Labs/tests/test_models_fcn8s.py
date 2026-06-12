from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


LABS_ROOT = Path(__file__).resolve().parents[1]
if str(LABS_ROOT) not in sys.path:
    sys.path.insert(0, str(LABS_ROOT))

import dl_lib
from dl_lib.models import FCN8s


class TestFCN8s(unittest.TestCase):
    def test_factory_loads_supported_backbones(self) -> None:
        for backbone in ("vgg16", "resnet50"):
            with self.subTest(backbone=backbone):
                model = dl_lib.models.load("fcn8s", backbone=backbone, num_classes=7)
                self.assertIsInstance(model, FCN8s)
                self.assertEqual(model.backbone_name, backbone)
                self.assertEqual(model.num_classes, 7)

    def test_factory_rejects_unknown_model(self) -> None:
        with self.assertRaises(ValueError):
            dl_lib.models.load("unknown_segmentation_model")

    def test_factory_rejects_unknown_backbone(self) -> None:
        with self.assertRaises(ValueError):
            dl_lib.models.load("fcn8s", backbone="efficientnet")

    def test_forward_preserves_spatial_resolution_for_vgg16(self) -> None:
        model = dl_lib.models.load("fcn8s", backbone="vgg16", num_classes=5)
        batch = torch.rand(2, 3, 224, 224)

        output = model(batch)

        self.assertTrue(torch.is_tensor(output))
        self.assertEqual(output.shape, (2, 5, 224, 224))

    def test_forward_preserves_spatial_resolution_for_resnet50(self) -> None:
        model = dl_lib.models.load("fcn8s", backbone="resnet50", num_classes=3)
        batch = torch.rand(1, 3, 161, 193)

        output = model(batch)

        self.assertTrue(torch.is_tensor(output))
        self.assertEqual(output.shape, (1, 3, 161, 193))


if __name__ == "__main__":
    unittest.main()
