import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.channel_split import ChannelSplitTool, extract_channel

ASK_CHANNEL = "forensics_app.tools.channel_split.simpledialog.askinteger"


class ChannelSplitTests(unittest.TestCase):
    def test_extract_each_channel(self) -> None:
        array = np.zeros((2, 2, 3), dtype=np.uint8)
        array[:, :, 0] = 10
        array[:, :, 1] = 20
        array[:, :, 2] = 30

        for index, expected in [(0, 10), (1, 20), (2, 30)]:
            channel = extract_channel(array, index)
            self.assertEqual(channel.shape, (2, 2))
            self.assertTrue(np.all(channel == expected))

    def test_rejects_grayscale_array(self) -> None:
        gray = np.zeros((2, 2), dtype=np.uint8)
        with self.assertRaises(ValueError):
            extract_channel(gray, 0)

    def test_rejects_invalid_index(self) -> None:
        array = np.zeros((2, 2, 3), dtype=np.uint8)
        for bad_index in (-1, 100):
            with self.assertRaises(ValueError):
                extract_channel(array, bad_index)


class ChannelSplitToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = ImageDocument()
        self.document.current = Image.new("RGBA", (4, 3), (10, 20, 30, 128))

    def test_returns_selected_channel_without_mutating_document(self) -> None:
        with patch(ASK_CHANNEL, return_value=1):
            result = ChannelSplitTool().run(None, self.document)

        self.assertEqual(result.image.mode, "L")
        self.assertTrue(np.all(np.asarray(result.image) == 20))
        self.assertEqual(result.details["Channel"], "Green")
        self.assertEqual(result.details["Mean"], 20.0)
        self.assertEqual(self.document.current.mode, "RGBA")

    def test_returns_none_when_dialog_is_cancelled(self) -> None:
        with patch(ASK_CHANNEL, return_value=None):
            self.assertIsNone(ChannelSplitTool().run(None, self.document))

    def test_rejects_grayscale_image(self) -> None:
        self.document.current = Image.new("L", (4, 3))
        with self.assertRaises(ValueError):
            ChannelSplitTool().run(None, self.document)


if __name__ == "__main__":
    unittest.main()