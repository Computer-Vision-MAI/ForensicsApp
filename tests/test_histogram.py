import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.histogram import HistogramTool, calculate_histograms


class HistogramCalculationTests(unittest.TestCase):
    def assert_counts(self, image, expected):
        original = image.tobytes()
        histograms = calculate_histograms(image)
        self.assertEqual(list(histograms), list(expected))
        for name, bins in expected.items():
            counts = np.zeros(256, dtype=int)
            for intensity, count in bins.items():
                counts[intensity] = count
            np.testing.assert_array_equal(histograms[name], counts)
        self.assertEqual(image.tobytes(), original)

    def test_binary_white_pixels_use_bin_255(self):
        image = Image.new("1", (3, 1))
        image.putdata([0, 1, 1])
        self.assert_counts(image, {"Intensity": {0: 1, 255: 2}})

    def test_supported_direct_modes(self):
        cases = [
            ("L", [0, 128, 255], {"Intensity": {0: 1, 128: 1, 255: 1}}),
            ("LA", [(20, 255), (20, 0), (80, 255)],
             {"Intensity": {20: 2, 80: 1}, "Alpha": {0: 1, 255: 2}}),
            ("RGB", [(0, 20, 255), (0, 40, 255), (128, 40, 0)],
             {"Red": {0: 2, 128: 1}, "Green": {20: 1, 40: 2}, "Blue": {0: 1, 255: 2}}),
            ("RGBA", [(0, 20, 255, 0), (0, 40, 255, 128), (128, 40, 0, 255)],
             {"Red": {0: 2, 128: 1}, "Green": {20: 1, 40: 2},
              "Blue": {0: 1, 255: 2}, "Alpha": {0: 1, 128: 1, 255: 1}}),
            ("CMYK", [(0, 20, 255, 10), (0, 40, 255, 10), (128, 40, 0, 90)],
             {"Cyan": {0: 2, 128: 1}, "Magenta": {20: 1, 40: 2},
              "Yellow": {0: 1, 255: 2}, "Black": {10: 2, 90: 1}}),
        ]
        for mode, pixels, expected in cases:
            with self.subTest(mode=mode):
                image = Image.new(mode, (3, 1))
                image.putdata(pixels)
                self.assert_counts(image, expected)

    def test_palette_modes_use_colors_and_transparency(self):
        for mode in ("P", "PA"):
            with self.subTest(mode=mode):
                image = Image.new(mode, (3, 1))
                image.putpalette([10, 20, 30, 100, 150, 200] + [0] * 762)
                if mode == "P":
                    image.putdata([0, 1, 1])
                    image.info["transparency"] = 0
                    alpha = {0: 1, 255: 2}
                else:
                    image.putdata([(0, 0), (1, 128), (1, 255)])
                    alpha = {0: 1, 128: 1, 255: 1}
                self.assert_counts(image, {
                    "Red": {10: 1, 100: 2}, "Green": {20: 1, 150: 2},
                    "Blue": {30: 1, 200: 2}, "Alpha": alpha,
                })

    def test_unsupported_modes_are_rejected(self):
        for mode in ("I", "F", "I;16", "HSV", "YCbCr"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "does not support image mode"):
                    calculate_histograms(Image.new(mode, (2, 2)))


class HistogramToolTests(unittest.TestCase):
    def setUp(self):
        self.tool = HistogramTool()
        self.document = ImageDocument()

    def test_run_fails_without_current_image(self):
        with self.assertRaises(AssertionError):
            self.tool.run(None, self.document)

    def test_supported_modes_plot_calculated_counts(self):
        for mode in ("1", "L", "LA", "RGB", "RGBA", "P", "PA", "CMYK"):
            with self.subTest(mode=mode):
                self.document.current = Image.new(mode, (2, 2))
                expected = calculate_histograms(self.document.current)
                with patch("matplotlib.axes.Axes.plot") as plot:
                    result = self.tool.run(None, self.document)
                self.assertEqual(plot.call_count, len(expected))
                for call, (name, counts) in zip(plot.call_args_list, expected.items()):
                    np.testing.assert_array_equal(call.args[0], np.arange(256))
                    np.testing.assert_array_equal(call.args[1], counts)
                    self.assertEqual(call.kwargs["label"], name)
                self.assertEqual(result.details["Channels Plotted"], len(expected))
                self.assertEqual(result.details["Original Mode"], mode)
                self.assertTrue(result.preview_only)

    def test_unsupported_mode_fails_before_plotting(self):
        self.document.current = Image.new("F", (2, 2), 1000)
        with patch("forensics_app.tools.histogram.Figure") as figure:
            with self.assertRaises(ValueError):
                self.tool.run(None, self.document)
        figure.assert_not_called()

    def test_output_image_dimensions_and_type(self):
        self.document.current = Image.new("L", (2, 2), color=0)
        source = self.document.current
        result = self.tool.run(None, self.document)
        self.assertEqual(result.image.size, (800, 600))
        self.assertEqual(result.image.mode, "RGBA")
        self.assertIs(self.document.current, source)


if __name__ == "__main__":
    unittest.main()
