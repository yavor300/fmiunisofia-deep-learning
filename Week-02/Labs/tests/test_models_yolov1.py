from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


LABS_ROOT = Path(__file__).resolve().parents[1]
if str(LABS_ROOT) not in sys.path:
    sys.path.insert(0, str(LABS_ROOT))

import dl_lib
from dl_lib.models import YOLOv1


class TestYOLOv1(unittest.TestCase):
    def test_factory_loads_all_supported_backbones(self) -> None:
        for backbone in ("vgg19", "googlenet", "inception_v3", "resnet50"):
            with self.subTest(backbone=backbone):
                model = dl_lib.models.load("yolov1", backbone=backbone, num_classes=20)
                self.assertIsInstance(model, YOLOv1)
                self.assertEqual(model.backbone_name, backbone)

    def test_factory_rejects_unknown_model(self) -> None:
        with self.assertRaises(ValueError):
            dl_lib.models.load("unknown_model")

    def test_factory_rejects_unknown_backbone(self) -> None:
        with self.assertRaises(ValueError):
            dl_lib.models.load("yolov1", backbone="efficientnet")

    def test_tensor_forward_has_yolov1_output_shape(self) -> None:
        model = dl_lib.models.load("yolov1", backbone="googlenet", grid_size=7, boxes_per_cell=2, num_classes=20)
        batch = torch.rand(2, 3, 224, 224)

        output = model(batch)

        self.assertTrue(torch.is_tensor(output))
        self.assertEqual(output.shape, (2, 7, 7, 2 * 5 + 20))

    def test_inference_api_contract_matches_readme(self) -> None:
        model = dl_lib.models.load("yolov1", backbone="googlenet", conf_threshold=0.0)
        model.eval()

        results = model([torch.rand(3, 128, 128)])

        self.assertTrue(hasattr(results, "print"))
        self.assertTrue(hasattr(results, "xyxy"))
        self.assertEqual(len(results.xyxy), 1)
        self.assertTrue(torch.is_tensor(results.xyxy[0]))
        self.assertEqual(results.xyxy[0].shape[1], 6)

        pandas_results = results.pandas().xyxy
        self.assertEqual(len(pandas_results), 1)

        first = pandas_results[0]
        if hasattr(first, "columns"):
            self.assertEqual(
                list(first.columns),
                ["xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"],
            )
        else:
            if first:
                self.assertIsInstance(first[0], dict)
                self.assertEqual(
                    sorted(first[0].keys()),
                    sorted(["xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"]),
                )


if __name__ == "__main__":
    unittest.main()
