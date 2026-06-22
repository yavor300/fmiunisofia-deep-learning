from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from src.cityseg.data.label_mapping import get_palette
from src.cityseg.inference import predict_color_mask, predict_train_ids


class _ConstantClassModel(nn.Module):
    def __init__(self, class_id: int, num_classes: int = 19) -> None:
        super().__init__()
        self.class_id = class_id
        self.bias = nn.Parameter(torch.zeros(1))
        self.num_classes = num_classes

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = image.shape
        logits = torch.zeros(
            batch_size,
            self.num_classes,
            height,
            width,
            device=image.device,
        )
        logits[:, self.class_id] = 1.0 + self.bias
        return logits


class TestPredictTrainIds(unittest.TestCase):
    def test_when_single_image_is_predicted_then_train_id_mask_is_returned(self) -> None:
        model = _ConstantClassModel(class_id=2)
        image = torch.zeros(3, 4, 5)

        prediction = predict_train_ids(model=model, image_tensor=image, device="cpu")

        self.assertEqual(prediction.shape, (4, 5))
        self.assertEqual(prediction.dtype, np.uint8)
        np.testing.assert_array_equal(prediction, np.full((4, 5), 2, dtype=np.uint8))

    def test_when_image_shape_is_invalid_then_value_error_is_raised(self) -> None:
        model = _ConstantClassModel(class_id=2)
        image = torch.zeros(2, 3, 4, 5)

        with self.assertRaises(ValueError):
            predict_train_ids(model=model, image_tensor=image, device="cpu")


class TestPredictColorMask(unittest.TestCase):
    def test_when_single_image_is_predicted_then_color_mask_is_returned(self) -> None:
        class_id = 3
        model = _ConstantClassModel(class_id=class_id)
        image = torch.zeros(3, 4, 5)

        color_mask = predict_color_mask(model=model, image_tensor=image, device="cpu")

        self.assertEqual(color_mask.shape, (4, 5, 3))
        self.assertEqual(color_mask.dtype, np.uint8)
        np.testing.assert_array_equal(
            color_mask[0, 0],
            np.array(get_palette()[class_id], dtype=np.uint8),
        )
