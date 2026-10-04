import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.contrast_stretch import apply_contrast_enhancement, ContrastStretchTool

class ContrastStretchFunctionTests(unittest.TestCase):
    """Unit tests for the apply_contrast_enhancement public function using scikit-image."""
    
    def test_percentile_stretching_on_grayscale(self) -> None:
        # Create a low-contrast grayscale image (values between 100 and 150)
        img = Image.new("L", (10, 10), color=125)
        img.putpixel((0, 0), 100)
        img.putpixel((9, 9), 150)
        
        # Apply percentile stretch (0% forces absolute min/max stretching)
        result = apply_contrast_enhancement(img, method="percentile", clip_percent=0)
        result_arr = np.array(result)
        
        # The lowest value (100) should now be pure black (0)
        self.assertEqual(result_arr[0, 0], 0)
        # The highest value (150) should now be pure white (255)
        self.assertEqual(result_arr[9, 9], 255)
        self.assertEqual(result.mode, "L")

    def test_equalize_histogram_on_rgb_preserves_color_structure(self) -> None:
        img = Image.new("RGB", (4, 4), color=(100, 50, 50)) # Reddish image
        img.putpixel((0, 0), (50, 25, 25))
        img.putpixel((3, 3), (200, 100, 100))
        
        result = apply_contrast_enhancement(img, method="equalize", clip_percent=0)
        
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, (4, 4))
        
        # Check that the dominant red channel remains strictly greater than green/blue
        # ensuring the hue was preserved during LAB conversions
        result_arr = np.array(result)
        self.assertTrue(np.all(result_arr[..., 0] >= result_arr[..., 1]))
        self.assertTrue(np.all(result_arr[..., 0] >= result_arr[..., 2]))

    def test_adaptive_clahe_method(self) -> None:
        img = Image.new("L", (10, 10), color=128)
        
        result = apply_contrast_enhancement(img, method="adaptive", clip_percent=0)
        
        self.assertEqual(result.mode, "L")
        self.assertEqual(result.size, (10, 10))

    def test_rgba_preserves_alpha(self) -> None:
        # Alpha should remain completely untouched by the LAB luminance stretching
        img = Image.new("RGBA", (2, 2), color=(100, 100, 100, 128))
        img.putpixel((0, 0), (150, 150, 150, 0))
        img.putpixel((1, 0), (125, 125, 125, 64))
        
        input_alpha = np.array(img.getchannel("A"))
        result = apply_contrast_enhancement(img, method="percentile", clip_percent=2)
        result_arr = np.array(result)
        
        self.assertEqual(result.mode, "RGBA")
        np.testing.assert_array_equal(result_arr[..., 3], input_alpha)

    def test_palette_image_preserves_alpha_and_converts_to_rgba(self) -> None:
        # Palette image with transparency should be processed and returned as RGBA
        img = Image.new("P", (2, 2))
        img.putpalette([50, 50, 50, 150, 150, 150] + [0] * 762)
        img.putdata([0, 1, 0, 1])
        img.info["transparency"] = 0 # Index 0 is transparent
        
        input_alpha = np.array(img.convert("RGBA").getchannel("A"))
        result = apply_contrast_enhancement(img, method="percentile", clip_percent=0)
        
        self.assertEqual(result.mode, "RGBA")
        np.testing.assert_array_equal(np.array(result.getchannel("A")), input_alpha)

    def test_numeric_modes_process_and_return_to_l_mode(self) -> None:
        # Float and 32-bit integer modes should process safely and return as Grayscale
        for mode in ("I", "F", "1"):
            with self.subTest(mode=mode):
                img = Image.new(mode, (2, 2), color=100)
                img.putpixel((0, 0), 0)
                
                result = apply_contrast_enhancement(img, method="equalize", clip_percent=0)
                
                self.assertEqual(result.mode, "L")
                self.assertEqual(result.size, (2, 2))

    def test_flat_image_prevents_math_crashes(self) -> None:
        # Images where all pixels are identical should not raise division by zero errors
        img = Image.new("RGB", (2, 2), color=(100, 100, 100))
        result = apply_contrast_enhancement(img, method="percentile", clip_percent=5)
        self.assertEqual(result.size, (2, 2))


class ContrastStretchToolTests(unittest.TestCase):
    """Unit tests for the ContrastStretchTool wrapper and UI integration."""
    
    def setUp(self) -> None:
        self.tool = ContrastStretchTool()
        self.document = ImageDocument()

    def test_run_fails_without_current_image(self) -> None:
        self.document.current = None
        with self.assertRaises(AssertionError):
            self.tool.run(None, self.document)

    @patch("forensics_app.tools.contrast_stretch.ask_choice", return_value=None)
    def test_run_cancels_gracefully_at_method_selection(self, mock_ask_choice) -> None:
        self.document.current = Image.new("RGB", (2, 2), color=(50, 50, 50))
        
        # User closes the initial ChoiceDialog
        result = self.tool.run(None, self.document)
        self.assertIsNone(result)

    @patch("forensics_app.tools.contrast_stretch.ask_choice", return_value=0) # 0 is Percentile
    @patch("forensics_app.tools.contrast_stretch.simpledialog.askinteger", return_value=None)
    def test_run_cancels_gracefully_at_percentile_input(self, mock_askinteger, mock_ask_choice) -> None:
        self.document.current = Image.new("RGB", (2, 2), color=(50, 50, 50))
        
        # User selects Percentile but cancels the subsequent integer dialog
        result = self.tool.run(None, self.document)
        
        self.assertIsNone(result)
        mock_askinteger.assert_called_once()

    @patch("forensics_app.tools.contrast_stretch.ask_choice", return_value=0) # 0 is Percentile
    @patch("forensics_app.tools.contrast_stretch.simpledialog.askinteger", return_value=3)
    def test_run_executes_percentile_successfully(self, mock_askinteger, mock_ask_choice) -> None:
        self.document.current = Image.new("RGB", (2, 2), color=(50, 50, 50))
        
        result = self.tool.run(None, self.document)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.details["Operation"], "Contrast Enhancement")
        self.assertEqual(result.details["Method"], "Percentile")
        self.assertEqual(result.details["Clip Percentage"], "3%")
        self.assertEqual(result.details["Original Mode"], "RGB")

    @patch("forensics_app.tools.contrast_stretch.ask_choice", return_value=1) # 1 is Equalize
    @patch("forensics_app.tools.contrast_stretch.simpledialog.askinteger")
    def test_run_executes_equalize_without_asking_percent(self, mock_askinteger, mock_ask_choice) -> None:
        self.document.current = Image.new("L", (2, 2), color=50)
        
        result = self.tool.run(None, self.document)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.details["Method"], "Equalize")
        self.assertEqual(result.details["Clip Percentage"], "N/A")
        # Ensure askinteger was NEVER called since Equalize doesn't need it
        mock_askinteger.assert_not_called()

    @patch("forensics_app.tools.contrast_stretch.ask_choice", return_value=2) # 2 is Adaptive (CLAHE)
    @patch("forensics_app.tools.contrast_stretch.simpledialog.askinteger")
    def test_run_executes_adaptive_without_asking_percent(self, mock_askinteger, mock_ask_choice) -> None:
        self.document.current = Image.new("RGB", (2, 2), color=(50, 50, 50))
        
        result = self.tool.run(None, self.document)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.details["Method"], "Adaptive")
        self.assertEqual(result.details["Clip Percentage"], "N/A")
        mock_askinteger.assert_not_called()

if __name__ == "__main__":
    unittest.main()