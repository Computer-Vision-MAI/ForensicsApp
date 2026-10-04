import unittest
from unittest.mock import patch, MagicMock

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.histogram import HistogramTool, calculate_histograms

class HistogramCalculationTests(unittest.TestCase):
    def assert_counts(self, image, expected, bins=256):
        original = image.tobytes()
        histograms = calculate_histograms(image, bins=bins)
        
        self.assertEqual(list(histograms), list(expected))
        for name, expected_bins in expected.items():
            # Construir el array de conteos esperado
            expected_counts = np.zeros(bins, dtype=int)
            for intensity, count in expected_bins.items():
                expected_counts[intensity] = count
            
            # Desempaquetar la nueva tupla (counts, edges)
            counts, edges = histograms[name]
            
            np.testing.assert_array_equal(counts, expected_counts)
            self.assertEqual(len(edges), bins + 1) # Los bordes siempre son N+1
            
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

    def test_high_depth_and_continuous_modes_are_supported(self):
        # Asegurar que los modos matemáticos que antes fallaban ahora calculan dinámicamente
        for mode in ("I", "F", "I;16", "HSV", "YCbCr"):
            with self.subTest(mode=mode):
                image = Image.new(mode, (2, 2))
                histograms = calculate_histograms(image, bins=10)
                self.assertTrue(len(histograms) > 0)
                # Seleccionar el primer canal devuelto
                counts, edges = list(histograms.values())[0]
                self.assertEqual(len(counts), 10)
                self.assertEqual(len(edges), 11)

    def test_invalid_mode_is_rejected(self):
        # Create a mock object that simulates an Image with an unsupported mode
        mock_image = MagicMock()
        mock_image.mode = "UNKNOWN_FORMAT"
        with self.assertRaisesRegex(ValueError, "does not support image mode"):
            calculate_histograms(mock_image)


class HistogramToolTests(unittest.TestCase):
    def setUp(self):
        self.tool = HistogramTool()
        self.document = ImageDocument()

    def test_run_fails_without_current_image(self):
        with self.assertRaises(AssertionError):
            self.tool.run(None, self.document)

    @patch("forensics_app.tools.histogram.simpledialog.askinteger", return_value=None)
    def test_run_returns_none_when_dialog_cancelled(self, mock_ask):
        self.document.current = Image.new("RGB", (2, 2))
        self.assertIsNone(self.tool.run(None, self.document))

    @patch("forensics_app.tools.histogram.simpledialog.askinteger", return_value=256)
    def test_supported_modes_plot_calculated_counts(self, mock_ask):
        for mode in ("1", "L", "LA", "RGB", "RGBA", "P", "PA", "CMYK"):
            with self.subTest(mode=mode):
                self.document.current = Image.new(mode, (2, 2))
                expected = calculate_histograms(self.document.current, bins=256)
                
                with patch("matplotlib.axes.Axes.plot") as plot:
                    result = self.tool.run(None, self.document)
                    
                self.assertEqual(plot.call_count, len(expected))
                
                for call, (name, (counts, edges)) in zip(plot.call_args_list, expected.items()):
                    # Verificar que al plot se le pasan los centros geométricos, no los bordes
                    bin_centers = (edges[:-1] + edges[1:]) / 2
                    np.testing.assert_array_equal(call.args[0], bin_centers)
                    np.testing.assert_array_equal(call.args[1], counts)
                    self.assertEqual(call.kwargs["label"], name)
                    
                self.assertEqual(result.details["Channels Plotted"], len(expected))
                self.assertEqual(result.details["Original Mode"], mode)
                self.assertEqual(result.details["Bins"], 256)
                self.assertTrue(result.preview_only)

    @patch("forensics_app.tools.histogram.simpledialog.askinteger", return_value=1024)
    def test_dynamic_bins_adjust_plotting(self, mock_ask):
        self.document.current = Image.new("I;16", (2, 2))
        with patch("matplotlib.axes.Axes.plot") as plot:
            result = self.tool.run(None, self.document)
            
        # Verificar que el array en el eje X tiene la longitud del bin dinámico pedido (1024)
        x_axis_data = plot.call_args_list[0].args[0]
        self.assertEqual(len(x_axis_data), 1024)
        self.assertEqual(result.details["Bins"], 1024)
    
    @patch("forensics_app.tools.histogram.simpledialog.askinteger", return_value=256)
    def test_active_data_range_calculation(self, mock_ask):
        # --- Scenario 1: Restricted contrast image (low contrast) ---
        # Create a gray image (128) and force a minimum of 50 and a maximum of 200
        img_restricted = Image.new("L", (10, 10), color=128)
        img_restricted.putpixel((0, 0), 50)
        img_restricted.putpixel((9, 9), 200)
        
        self.document.current = img_restricted
        
        # Intercept plot so the test runs ultra-fast and doesn't attempt to draw anything real
        with patch("matplotlib.axes.Axes.plot"):
            result_restricted = self.tool.run(None, self.document)
            
        self.assertIsNotNone(result_restricted)
        # Pixel 200 falls into the [200.0, 201.0) bin, so the upper edge must be 201.0
        self.assertEqual(result_restricted.details["Active Data Range (Edges)"], "[50.0, 201.0]")
        
        # --- Scenario 2: Full range image ---
        # Modify the image to include pure black (0) and pure white (255)
        img_full = img_restricted.copy()
        img_full.putpixel((0, 1), 0)
        img_full.putpixel((9, 8), 255)
        
        self.document.current = img_full
        
        with patch("matplotlib.axes.Axes.plot"):
            result_full = self.tool.run(None, self.document)
            
        self.assertIsNotNone(result_full)
        # Pixel 255 falls into the [255.0, 256.0) bin, so the upper edge must be 256.0
        self.assertEqual(result_full.details["Active Data Range (Edges)"], "[0.0, 256.0]")

    @patch("forensics_app.tools.histogram.simpledialog.askinteger", return_value=256)
    def test_output_image_dimensions_and_type(self, mock_ask):
        self.document.current = Image.new("L", (2, 2), color=0)
        source = self.document.current
        result = self.tool.run(None, self.document)
        
        self.assertEqual(result.image.size, (800, 600))
        self.assertEqual(result.image.mode, "RGBA")
        self.assertIs(self.document.current, source)


if __name__ == "__main__":
    unittest.main()