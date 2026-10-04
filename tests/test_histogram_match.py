import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.histogram_match import apply_histogram_match, HistogramMatchTool

class HistogramMatchFunctionTests(unittest.TestCase):
    """Unit tests for the apply_histogram_match mathematical orchestration."""

    def test_match_grayscale_to_grayscale(self) -> None:
        # Base is a dark grayscale image
        base_img = Image.new("L", (10, 10), color=50)
        # Reference is a bright grayscale image
        ref_img = Image.new("L", (10, 10), color=200)

        result = apply_histogram_match(base_img, ref_img)
        result_arr = np.array(result)

        self.assertEqual(result.mode, "L")
        self.assertEqual(result.size, (10, 10))
        
        # The base image should be shifted entirely towards the bright reference
        self.assertTrue(np.all(result_arr > 150))

    def test_match_rgb_to_rgb(self) -> None:
        base_img = Image.new("RGB", (4, 4), color=(100, 50, 50))
        ref_img = Image.new("RGB", (4, 4), color=(200, 200, 200))

        result = apply_histogram_match(base_img, ref_img)

        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, (4, 4))
        # Verify output array is correctly formatted as a 3D RGB array
        self.assertEqual(np.array(result).shape, (4, 4, 3))

    def test_histogram_match_reassembles_alpha_channel(self) -> None:
        # Although alpha extraction is tested in image_utils, we must test that 
        # apply_histogram_match correctly re-stacks it (np.dstack) before returning.
        base_img = Image.new("RGBA", (2, 2), color=(100, 100, 100, 128))
        base_img.putpixel((0, 0), (150, 150, 150, 0))
        ref_img = Image.new("RGB", (2, 2), color=(200, 200, 200))

        input_alpha = np.array(base_img.getchannel("A"))
        result = apply_histogram_match(base_img, ref_img)
        
        self.assertEqual(result.mode, "RGBA")
        result_arr = np.array(result)
        np.testing.assert_array_equal(result_arr[..., 3], input_alpha)

    def test_match_cross_mode_rgb_base_to_l_reference(self) -> None:
        # Base is color, reference is grayscale
        base_img = Image.new("RGB", (2, 2), color=(100, 50, 50))
        ref_img = Image.new("L", (2, 2), color=200)
        
        result = apply_histogram_match(base_img, ref_img)
        
        # The output must perfectly preserve the base image's mode and size
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, (2, 2))


class HistogramMatchToolTests(unittest.TestCase):
    """Unit tests for the HistogramMatchTool UI wrapper."""

    def setUp(self) -> None:
        self.tool = HistogramMatchTool()
        self.document = ImageDocument()

    def test_run_fails_without_current_image(self) -> None:
        self.document.current = None
        with self.assertRaises(AssertionError):
            self.tool.run(MagicMock(), self.document)

    @patch("forensics_app.tools.histogram_match.HistogramMatchTool.load_reference_image")
    def test_run_cancels_gracefully_when_no_reference_selected(self, mock_load) -> None:
        self.document.current = Image.new("RGB", (2, 2), color=(50, 50, 50))
        mock_load.return_value = None  # Simulate user canceling the file dialog
        
        result = self.tool.run(MagicMock(), self.document)
        
        self.assertIsNone(result)
        mock_load.assert_called_once()

    @patch("forensics_app.tools.histogram_match.HistogramMatchTool.load_reference_image")
    def test_run_executes_successfully(self, mock_load) -> None:
        self.document.current = Image.new("RGB", (2, 2), color=(50, 50, 50))
        ref_img = Image.new("RGB", (2, 2), color=(200, 200, 200))
        
        # Simulate a valid file selection
        mock_load.return_value = (Path("reference_evidence.png"), ref_img)
        
        result = self.tool.run(MagicMock(), self.document)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.details["Operation"], "Histogram Match")
        self.assertEqual(result.details["Reference File"], "reference_evidence.png")
        self.assertEqual(result.details["Original Mode"], "RGB")
        self.assertEqual(result.details["Output Mode"], "RGB")

if __name__ == "__main__":
    unittest.main()