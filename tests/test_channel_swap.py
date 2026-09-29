import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.channel_swap import ChannelSwapTool, swap_channels

ASK_ORDER = "forensics_app.tools.channel_swap.ask_channel_order"


class ChannelSwapTests(unittest.TestCase):
    def test_swap_channels(self) -> None:
        array = np.zeros((1, 1, 3), dtype=np.uint8)
        array[0, 0] = (10, 20, 30)
        result = swap_channels(array, (2, 1, 0))
        self.assertEqual(result[0, 0].tolist(), [30, 20, 10])

    def test_identity_order_keeps_array(self) -> None:
        array = np.zeros((1, 1, 3), dtype=np.uint8)
        array[0, 0] = (10, 20, 30)
        result = swap_channels(array, (0, 1, 2))
        self.assertTrue(np.array_equal(result, array))

    def test_swap_rejects_grayscale_array(self) -> None:
        array = np.zeros((2, 2), dtype=np.uint8)
        with self.assertRaises(ValueError):
            swap_channels(array, (0, 1, 2))

    def test_swaps_four_channels(self) -> None:
        array = np.zeros((1, 1, 4), dtype=np.uint8)
        array[0, 0] = (10, 20, 30, 40)
        self.assertEqual(swap_channels(array, (3, 2, 1, 0))[0, 0].tolist(), [40, 30, 20, 10])

    def test_rejects_invalid_orders(self) -> None:
        array = np.zeros((1, 1, 3), dtype=np.uint8)
        for order in ((0, 0, 1), (0, 1), (0, 1, 2, 3), (1, 2, 3)):
            with self.subTest(order=order), self.assertRaises(ValueError):
                swap_channels(array, order)


class ChannelSwapToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = ImageDocument()
        self.document.current = Image.new("RGBA", (4, 3), (10, 20, 30, 128))

    def test_returns_swapped_channels_without_mutating_document(self) -> None:
        with patch(ASK_ORDER, return_value=(2, 1, 0)) as ask:
            result = ChannelSwapTool().run(None, self.document)

        self.assertEqual(ask.call_args.args[2], ("Red", "Green", "Blue"))
        self.assertEqual(result.image.mode, "RGBA")
        self.assertEqual(result.image.getpixel((0, 0)), (30, 20, 10, 128))
        self.assertEqual(result.details["Order"], "Red, Green, Blue -> Blue, Green, Red")
        self.assertEqual(self.document.current.mode, "RGBA")

    def test_returns_none_when_dialog_is_cancelled(self) -> None:
        with patch(ASK_ORDER, return_value=None):
            result = ChannelSwapTool().run(None, self.document)
        self.assertIsNone(result)

    def test_accepts_color_modes(self) -> None:
        palette_image = Image.new("P", (4, 3), 0)
        palette_image.putpalette([10, 20, 30])
        for image in (Image.new("RGB", (4, 3), (10, 20, 30)), palette_image):
            self.document.current = image
            with self.subTest(mode=image.mode), patch(ASK_ORDER, return_value=(2, 1, 0)):
                result = ChannelSwapTool().run(None, self.document)
                self.assertEqual(result.image.mode, "RGB")
                self.assertEqual(result.image.getpixel((0, 0)), (30, 20, 10))

    def test_keeps_cmyk_mode(self) -> None:
        self.document.current = Image.new("CMYK", (4, 3), (10, 20, 30, 40))
        with patch(ASK_ORDER, return_value=(3, 2, 1, 0)):
            result = ChannelSwapTool().run(None, self.document)
        self.assertEqual(result.image.mode, "CMYK")
        self.assertEqual(result.image.getpixel((0, 0)), (40, 30, 20, 10))

    def test_rejects_modes_without_two_color_channels(self) -> None:
        tool = ChannelSwapTool()
        for mode in ("1", "L", "I", "F", "I;16", "LA", "La"):
            self.document.current = Image.new(mode, (4, 3))
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                tool.run(None, self.document)


if __name__ == "__main__":
    unittest.main()
