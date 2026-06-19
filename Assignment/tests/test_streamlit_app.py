from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


def _load_streamlit_app():
    app_path = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    spec = importlib.util.spec_from_file_location("streamlit_app", app_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load streamlit_app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


streamlit_app = _load_streamlit_app()


class TestPreprocessImage(unittest.TestCase):
    def test_when_image_is_preprocessed_then_tensor_is_channel_first(self) -> None:
        image = Image.fromarray(np.zeros((8, 10, 3), dtype=np.uint8))
        config = {
            "preprocessing": {
                "resize_height": 16,
                "resize_width": 20,
                "normalize": "none",
            }
        }

        tensor = streamlit_app.preprocess_image(image, config)

        self.assertEqual(tuple(tensor.shape), (3, 16, 20))


class TestCreateOverlay(unittest.TestCase):
    def test_when_overlay_is_created_then_output_matches_original_size(self) -> None:
        image = Image.fromarray(np.full((8, 10, 3), 100, dtype=np.uint8))
        mask = np.full((4, 5, 3), 200, dtype=np.uint8)

        overlay = streamlit_app.create_overlay(image, mask, opacity=0.5)

        self.assertEqual(overlay.shape, (8, 10, 3))
        self.assertEqual(overlay.dtype, np.uint8)


class TestTopClassesByPercentage(unittest.TestCase):
    def test_when_mask_has_multiple_classes_then_largest_classes_are_returned_first(self) -> None:
        mask = np.array([[0, 0, 1], [1, 1, 2]], dtype=np.uint8)

        classes = streamlit_app.top_classes_by_percentage(mask, top_k=2)

        self.assertEqual(classes[0][0], "sidewalk")
        self.assertAlmostEqual(classes[0][1], 50.0)
        self.assertEqual(len(classes), 2)


class TestImageToPngBytes(unittest.TestCase):
    def test_when_image_is_encoded_then_png_bytes_are_returned(self) -> None:
        image = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))

        png_bytes = streamlit_app.image_to_png_bytes(image)

        self.assertTrue(png_bytes.startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
