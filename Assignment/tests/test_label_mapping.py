from __future__ import annotations

import unittest

import numpy as np

from src.cityseg.constants import IGNORE_INDEX
from src.cityseg.data.label_mapping import (
    convert_label_ids_to_train_ids,
    decode_train_ids_to_colors,
    get_class_names,
    get_palette,
)


class TestConvertLabelIdsToTrainIds(unittest.TestCase):
    def test_when_mask_contains_valid_label_ids_then_train_ids_are_returned(self) -> None:
        mask = np.array([[7, 8, 11], [24, 26, 33]], dtype=np.uint8)

        converted = convert_label_ids_to_train_ids(mask)

        expected = np.array([[0, 1, 2], [11, 13, 18]], dtype=np.uint8)

        np.testing.assert_array_equal(converted, expected)

    def test_when_mask_contains_ignored_label_ids_then_ignore_index_is_returned(self) -> None:
        mask = np.array([[0, 1, 3], [255, 99, 7]], dtype=np.uint8)

        converted = convert_label_ids_to_train_ids(mask)

        np.testing.assert_array_equal(
            converted,
            np.array(
                [
                    [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX],
                    [IGNORE_INDEX, IGNORE_INDEX, 0],
                ],
                dtype=np.uint8,
            ),
        )


class TestDecodeTrainIdsToColors(unittest.TestCase):
    def test_when_train_id_mask_is_decoded_then_rgb_image_is_returned(self) -> None:
        mask = np.array([[0, 1], [2, IGNORE_INDEX]], dtype=np.uint8)

        decoded = decode_train_ids_to_colors(mask)

        self.assertEqual(decoded.shape, (2, 2, 3))
        np.testing.assert_array_equal(decoded[0, 0], np.array(get_palette()[0], dtype=np.uint8))

    def test_when_mask_contains_ignore_index_then_black_pixel_is_returned(self) -> None:
        mask = np.array([[IGNORE_INDEX]], dtype=np.uint8)

        decoded = decode_train_ids_to_colors(mask)

        np.testing.assert_array_equal(decoded[0, 0], np.array([0, 0, 0], dtype=np.uint8))


class TestGetClassNames(unittest.TestCase):
    def test_when_names_are_requested_then_train_classes_are_returned(self) -> None:
        class_names = get_class_names()

        self.assertEqual(len(class_names), 19)
        self.assertEqual(class_names[0], "road")
        self.assertEqual(class_names[-1], "bicycle")


class TestGetPalette(unittest.TestCase):
    def test_when_palette_is_requested_then_one_rgb_color_exists_per_class(self) -> None:
        palette = get_palette()

        self.assertEqual(len(palette), 19)
        self.assertTrue(all(len(color) == 3 for color in palette))


if __name__ == "__main__":
    unittest.main()
