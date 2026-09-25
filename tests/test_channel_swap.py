import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.channel_swap import (
    ChannelSwapTool,
    parse_channel_order,
    swap_channels,
)

ASK_ORDER = "forensics_app.tools.channel_swap.simpledialog.askstring"


class ChannelSwapTests(unittest.TestCase):
    def test_parse_channel_order(self) -> None:
        result = parse_channel_order(" bgr ")
        self.assertEqual(result, (2, 1, 0))

    def test_parse_rejects_invalid_orders(self) -> None:
        for text in ("RRG", "RG", "RGBA", "XYZ", ""):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_channel_order(text)

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


class ChannelSwapToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = ImageDocument()
        self.document.current = Image.new("RGBA", (4, 3), (10, 20, 30, 128))

    def test_returns_swapped_channels_without_mutating_document(self) -> None:
        with patch(ASK_ORDER, return_value="BGR"):
            result = ChannelSwapTool().run(None, self.document)

        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.image.getpixel((0, 0)), (30, 20, 10))
        self.assertEqual(result.details["Order"], "RGB -> BGR")
        self.assertEqual(self.document.current.mode, "RGBA")

    def test_returns_none_when_dialog_is_cancelled(self) -> None:
        with patch(ASK_ORDER, return_value=None):
            result = ChannelSwapTool().run(None, self.document)
        self.assertIsNone(result)

    def test_rejects_grayscale_modes(self) -> None:
        tool = ChannelSwapTool()
        for mode in ("1", "L", "LA", "I", "F"):
            self.document.current = Image.new(mode, (4, 3))
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                tool.run(None, self.document)


if __name__ == "__main__":
    unittest.main()
