import unittest
from unittest.mock import patch, MagicMock

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.histogram import HistogramTool

class HistogramToolTests(unittest.TestCase):
    """Unit tests for the HistogramTool ensuring correct plotting logic across all image modes."""
    
    def setUp(self) -> None:
        self.tool = HistogramTool()
        self.document = ImageDocument()

    def test_run_fails_without_current_image(self) -> None:
        self.document.current = None
        with self.assertRaises(AssertionError):
            self.tool.run(None, self.document)

    @patch("matplotlib.axes.Axes.plot")
    def test_rgb_image_plots_three_lines(self, mock_plot: MagicMock) -> None:
        # Standard 3-channel image
        self.document.current = Image.new("RGB", (4, 4), color="red")
        
        result = self.tool.run(None, self.document)
        
        self.assertIsNotNone(result)
        # Should call ax.plot 3 times (Red, Green, Blue)
        self.assertEqual(mock_plot.call_count, 3)
        self.assertEqual(result.details["Channels Plotted"], 3)
        self.assertEqual(result.details["Original Mode"], "RGB")
        self.assertEqual(result.image.mode, "RGBA") # Output graph must always be RGBA

    @patch("matplotlib.axes.Axes.plot")
    def test_rgba_image_plots_four_lines(self, mock_plot: MagicMock) -> None:
        # Standard 4-channel image
        self.document.current = Image.new("RGBA", (4, 4), color=(10, 20, 30, 255))
        
        result = self.tool.run(None, self.document)
        
        self.assertIsNotNone(result)
        # Should call ax.plot 4 times (Red, Green, Blue, Alpha)
        self.assertEqual(mock_plot.call_count, 4)
        self.assertEqual(result.details["Channels Plotted"], 4) # Numpy ndim for RGBA is 4

    @patch("matplotlib.axes.Axes.hist")
    def test_grayscale_image_plots_single_histogram(self, mock_hist: MagicMock) -> None:
        # 1-channel grayscale image
        self.document.current = Image.new("L", (4, 4), color=128)
        
        result = self.tool.run(None, self.document)
        
        self.assertIsNotNone(result)
        # Should call ax.hist exactly once with black color
        self.assertEqual(mock_hist.call_count, 1)
        self.assertEqual(result.details["Channels Plotted"], 1)
        
        # Verify it passed the correct color to the histogram
        mock_hist.assert_called_with(
            unittest.mock.ANY, bins=256, range=(0, 256), color="black", alpha=0.7, label="Intensity"
        )

    @patch("matplotlib.axes.Axes.hist")
    def test_binary_image_plots_single_histogram(self, mock_hist: MagicMock) -> None:
        # 1-channel binary image (pure black and white)
        self.document.current = Image.new("1", (4, 4), color=1)
        
        result = self.tool.run(None, self.document)
        
        self.assertIsNotNone(result)
        self.assertEqual(mock_hist.call_count, 1)
        self.assertEqual(result.details["Original Mode"], "1")

    @patch("matplotlib.axes.Axes.plot")
    def test_palette_image_converts_to_rgba_and_plots_lines(self, mock_plot: MagicMock) -> None:
        # Create a Palette (P) image
        base_p = Image.new("P", (4, 4), color=0)
        palette = [0, 0, 0, 255, 0, 0] + [0] * 762 # Black and Red palette
        base_p.putpalette(palette)
        
        self.document.current = base_p
        
        result = self.tool.run(None, self.document)
        
        self.assertIsNotNone(result)
        # Because P converts to RGBA internally, it must call plot 4 times
        self.assertEqual(mock_plot.call_count, 4)
        self.assertEqual(result.details["Original Mode"], "P")
        
    def test_output_image_dimensions_and_type(self) -> None:
        # Ensure that regardless of mocking, Matplotlib generates a valid PIL Image buffer
        self.document.current = Image.new("L", (2, 2), color=0)
        
        result = self.tool.run(None, self.document)
        
        # The default Figure(figsize=(8, 6)) at default 100 dpi yields an 800x600 image
        self.assertEqual(result.image.size, (800, 600))
        self.assertEqual(result.image.mode, "RGBA")


if __name__ == "__main__":
    unittest.main()