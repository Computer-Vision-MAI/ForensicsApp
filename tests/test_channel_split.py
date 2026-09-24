import unittest

import numpy as np

from forensics_app.tools.channel_split import extract_channel


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


if __name__ == "__main__":
    unittest.main()