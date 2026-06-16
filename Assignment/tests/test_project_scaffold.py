from __future__ import annotations

import importlib
import unittest


class TestImportCitysegPackage(unittest.TestCase):
    def test_when_package_is_imported_then_version_is_available(self) -> None:
        module = importlib.import_module("src.cityseg")

        self.assertEqual(module.__version__, "0.1.0")


class TestSyntheticSegmentationSample(unittest.TestCase):
    def setUp(self) -> None:
        self.image = [[[0.0 for _ in range(16)] for _ in range(16)] for _ in range(3)]
        self.mask = [[0 for _ in range(16)] for _ in range(16)]
        for row in range(4, 8):
            for col in range(4, 8):
                self.mask[row][col] = 1

    def test_when_synthetic_image_is_created_then_shape_is_channel_first(self) -> None:
        self.assertEqual(len(self.image), 3)
        self.assertEqual(len(self.image[0]), 16)
        self.assertEqual(len(self.image[0][0]), 16)

    def test_when_synthetic_mask_is_created_then_shape_matches_image_spatial_size(self) -> None:
        self.assertEqual(len(self.mask), 16)
        self.assertEqual(len(self.mask[0]), 16)

    def test_when_synthetic_mask_is_created_then_expected_class_ids_are_present(self) -> None:
        self.assertEqual({value for row in self.mask for value in row}, {0, 1})


if __name__ == "__main__":
    unittest.main()
