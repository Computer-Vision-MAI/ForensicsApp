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
        self.assertTrue(np.all(result_arr == 200))

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

    def test_native_hsv_matching(self) -> None:
        base_img = Image.new("HSV", (2, 2), color=(120, 100, 50))  # Very dark image
        ref_img = Image.new("RGB", (2, 2), color=(50, 200, 200))   # Very bright reference
        
        result = apply_histogram_match(base_img, ref_img)
        result_arr = np.array(result)
        
        self.assertEqual(result.mode, "HSV")
        
        # Hue and Saturation must remain strictly from the BASE image
        self.assertTrue(np.all(result_arr[..., 0] == 120))
        self.assertTrue(np.all(result_arr[..., 1] == 100))
        
        # Value (2) must be pulled heavily toward the bright reference
        self.assertTrue(np.all(result_arr[..., 2] > 150))

    def test_native_ycbcr_matching_with_rgb_reference(self) -> None:
        base_img = Image.new("YCbCr", (2, 2), color=(50, 100, 150))  # Dark YCbCr image
        ref_img = Image.new("RGB", (2, 2), color=(200, 200, 200))    # Bright RGB reference
        
        result = apply_histogram_match(base_img, ref_img)
        result_arr = np.array(result)
        
        self.assertEqual(result.mode, "YCbCr")
        
        # Cb (1) and Cr (2) must remain strictly from the BASE image
        self.assertTrue(np.all(result_arr[..., 1] == 100))
        self.assertTrue(np.all(result_arr[..., 2] == 150))
        
        # Luma (Y - index 0) must be pulled toward the bright RGB reference
        self.assertTrue(np.all(result_arr[..., 0] > 150))

    def test_lab_mode_matching(self) -> None:
        base_img = Image.new("LAB", (2, 2), color=(50, 5, 5))
        ref_img = Image.new("RGB", (2, 2), color=(200, 200, 200))

        # 1. We extract the raw array of the base image to compare channels later 
        # as the np array of the LAB mode where the A B values from -128..127 range are
        # converted to 0..255 range in the array, thus, +128 to the A and B channels.
        base_arr = np.array(base_img)

        result = apply_histogram_match(base_img, ref_img)
        result_arr = np.array(result)

        self.assertEqual(result.mode, "LAB")
        

        # 2. The A and B channels must remain mathematically identical to the base image
        np.testing.assert_array_equal(result_arr[..., 1], base_arr[..., 1])
        np.testing.assert_array_equal(result_arr[..., 2], base_arr[..., 2])
        
        # 3. The Luminance channel must be pulled toward the bright reference
        self.assertTrue(np.all(result_arr[..., 0] > 150))

        #4. The A and B channels are +128 in the array representation
        self.assertTrue(np.all(result_arr[..., 1] == base_img.getpixel((0, 0))[1] + 128))
        self.assertTrue(np.all(result_arr[..., 2] == base_img.getpixel((0, 0))[2] + 128))

    def test_match_la_preserves_alpha(self) -> None:
        # Base is grayscale with alpha. Ref is standard grayscale.
        base_img = Image.new("LA", (2, 2), color=(50, 128))
        base_img.putpixel((0, 0), (100, 0))
        ref_img = Image.new("L", (2, 2), color=200)

        input_alpha = np.array(base_img.getchannel("A"))
        result = apply_histogram_match(base_img, ref_img)
        result_arr = np.array(result)

        self.assertEqual(result.mode, "LA")
        # Check that the alpha channel is perfectly intact
        np.testing.assert_array_equal(result_arr[..., 1], input_alpha)
        # Check that the luminance channel was matched to the bright reference
        self.assertTrue(np.all(result_arr[..., 0] > 150))

    def test_match_palette_with_transparency(self) -> None:
        # Palette image with transparency must be processed and output as RGBA
        base_img = Image.new("P", (2, 2))
        base_img.putpalette([50, 50, 50, 100, 100, 100] + [0] * 762)
        base_img.putdata([0, 1, 0, 1])
        base_img.info["transparency"] = 0  # Index 0 is transparent
        
        ref_img = Image.new("RGB", (2, 2), color=(200, 200, 200))
        
        input_alpha = np.array(base_img.convert("RGBA").getchannel("A"))
        result = apply_histogram_match(base_img, ref_img)
        result_arr = np.array(result)
        
        self.assertEqual(result.mode, "RGBA")
        # Alpha channel must survive the LAB conversion trip
        np.testing.assert_array_equal(result_arr[..., 3], input_alpha)
        # RGB channels must be matched to the bright reference
        self.assertTrue(np.all(result_arr[..., :3] > 150))

    def test_match_high_depth_base_produces_l(self) -> None:
        # 32-bit Float image (e.g. scientific or medical evidence)
        f_data = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        base_img = Image.fromarray(f_data, mode="F")
        
        # Match against a bright 8-bit reference
        ref_img = Image.new("L", (2, 2), color=200)
        
        result = apply_histogram_match(base_img, ref_img)
        result_arr = np.array(result)
        
        # The result must be squeezed safely into an 8-bit L image
        self.assertEqual(result.mode, "L")
        self.assertEqual(result_arr.dtype, np.uint8)
        self.assertTrue(np.all(result_arr > 150))

    def test_match_cross_mode_l_base_to_rgb_reference(self) -> None:
        # Base is grayscale, reference is color
        base_img = Image.new("L", (2, 2), color=50)
        ref_img = Image.new("RGB", (2, 2), color=(200, 200, 200))
        
        result = apply_histogram_match(base_img, ref_img)
        result_arr = np.array(result)
        
        # The output must strictly respect the base image's mode
        self.assertEqual(result.mode, "L")
        self.assertEqual(result.size, (2, 2))
        self.assertTrue(np.all(result_arr > 150))

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