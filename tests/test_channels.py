import unittest

import numpy as np
from PIL import Image

from forensics_app.core.channels import MODE_CHANNELS, NUMERIC_MODES, to_multichannel


class ModeChannelsTests(unittest.TestCase):
    def test_covers_every_pillow_mode(self) -> None:
        self.assertEqual(set(MODE_CHANNELS), set(Image.MODES))

    def test_channel_counts_match_pillow(self) -> None:
        for mode, names in MODE_CHANNELS.items():
            with self.subTest(mode=mode):
                self.assertEqual(len(names), len(Image.new(mode, (1, 1)).getbands()))

    def test_numeric_modes_are_the_ones_wider_than_8_bits(self) -> None:
        for mode in Image.MODES:
            itemsize = np.asarray(Image.new(mode, (1, 1))).dtype.itemsize
            with self.subTest(mode=mode):
                self.assertEqual(mode in NUMERIC_MODES, itemsize > 1)


class ToMultichannelTests(unittest.TestCase):
    def test_expands_palettes(self) -> None:
        transparent = Image.new("P", (2, 2))
        transparent.info["transparency"] = 0
        cases = [
            (Image.new("P", (2, 2)), "RGB"),
            (transparent, "RGBA"),
            (Image.new("PA", (2, 2)), "RGBA"),
        ]
        for image, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(to_multichannel(image).mode, expected)

    def test_keeps_multichannel_modes(self) -> None:
        for mode in ("RGB", "RGBA", "CMYK", "LAB", "LA"):
            with self.subTest(mode=mode):
                self.assertEqual(to_multichannel(Image.new(mode, (2, 2))).mode, mode)

    def test_rejects_single_channel_modes(self) -> None:
        for mode, names in MODE_CHANNELS.items():
            if len(names) == 1 and mode != "P":
                image = Image.new(mode, (2, 2))
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    to_multichannel(image)


if __name__ == "__main__":
    unittest.main()
