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
    
    def test_flat_image_prevents_math_percentile(self) -> None:
        img = Image.new("RGB", (2, 2), color=(100, 100, 100))
        result = apply_contrast_enhancement(img, method="percentile", clip_percent=5)
        result_arr = np.array(result)
        
        self.assertEqual(result.size, (2, 2))
        # A uniform processed channel leaves the source image unchanged.
        self.assertTrue(np.all(result_arr == 100))
    
    def test_flat_image_prevents_math_adaptive(self) -> None:
        img = Image.new("RGB", (2, 2), color=(100, 100, 100))
        result = apply_contrast_enhancement(img, method="adaptive", clip_percent=0)
        result_arr = np.array(result)

        self.assertEqual(result.size, (2, 2))
        # A uniform processed channel leaves the source image unchanged.
        self.assertTrue(np.all(result_arr == 100))
    
    def test_flat_image_prevents_math_equalize(self) -> None:
        img = Image.new("RGB", (2, 2), color=(100, 100, 100))
        result = apply_contrast_enhancement(img, method="equalize", clip_percent=0)
        result_arr = np.array(result)

        self.assertEqual(result.size, (2, 2))

        # A uniform processed channel leaves the source image unchanged.
        self.assertTrue(np.all(result_arr == 100))
    
    def test_uniform_non_gray_image_is_unchanged(self) -> None:
        for method in ("percentile", "equalize", "adaptive"):
            with self.subTest(method=method):
                img = Image.new("RGB", (2, 2), color=(100, 50, 25))
                self.assertIs(apply_contrast_enhancement(img, method, 0), img)

    def test_uniform_intensity_ignores_variation_in_other_channels(self) -> None:
        for mode in ("HSV", "LAB", "YCbCr", "RGBA"):
            for method in ("percentile", "equalize", "adaptive"):
                with self.subTest(mode=mode, method=method):
                    img = Image.new("RGB", (2, 2), color=(100, 50, 25)).convert(mode)
                    pixel = list(img.getpixel((0, 0)))
                    idx = 3 if mode == "RGBA" else 1
                    pixel[idx] -= 1
                    img.putpixel((0, 0), tuple(pixel))
                    self.assertIs(apply_contrast_enhancement(img, method, 0), img)

    def test_unsupported_method_raises_for_flat_image(self) -> None:
        for mode in ("L", "RGB", "HSV", "LAB", "YCbCr"):
            with self.subTest(mode=mode):
                img = Image.new("RGB", (2, 2), color=(100, 100, 100)).convert(mode)
                with self.assertRaisesRegex(ValueError, "Unsupported contrast enhancement method"):
                    apply_contrast_enhancement(img, "invalid_method", 0)

    def test_unsupported_method_raises_value_error(self) -> None:
        img = Image.new("RGB", (2, 2), color=(100, 100, 100))
        img.putpixel((0, 0), (50, 50, 50)) # Add a pixel with a different value to avoid flat image detection
        with self.assertRaisesRegex(ValueError, "Unsupported contrast enhancement method"):
            apply_contrast_enhancement(img, method="invalid_method", clip_percent=0)
    
    def test_nan_values_in_high_depth_image_are_handled(self) -> None:
        # Create a high-depth image with NaN and Inf values
        float_data = np.array([[np.nan, 1.0], [2.0, np.inf]], dtype=np.float32)
        img = Image.fromarray(float_data)
        
        result = apply_contrast_enhancement(img, method="percentile", clip_percent=0)
        result_arr = np.array(result)
        
        self.assertEqual(result.mode, "L")
        # The NaN should be mapped to the minimum (0), Inf to maximum (255)
        self.assertEqual(result_arr[0, 0], 0)   # NaN -> 0
        self.assertEqual(result_arr[1, 1], 255) # Inf -> 255

    def test_native_hsv_stretching(self) -> None:
        # Base color with a dark and bright pixel to force stretching
        img = Image.new("HSV", (2, 2), color=(120, 100, 128))
        img.putpixel((0, 0), (120, 100, 50))   # Lowest Value
        img.putpixel((1, 1), (120, 100, 200))  # Highest Value
        
        result = apply_contrast_enhancement(img, method="percentile", clip_percent=0)
        result_arr = np.array(result)
        
        self.assertEqual(result.mode, "HSV")
        
        # Hue (0) and Saturation (1) must remain mathematically identical
        self.assertTrue(np.all(result_arr[..., 0] == 120))
        self.assertTrue(np.all(result_arr[..., 1] == 100))
        
        # Value (2) should be stretched to absolute min/max bounds (50->0, 200->255)
        self.assertEqual(result_arr[0, 0, 2], 0)
        self.assertEqual(result_arr[1, 1, 2], 255)

    def test_native_ycbcr_stretching(self) -> None:
        # Base color with a dark and bright pixel
        img = Image.new("YCbCr", (2, 2), color=(128, 100, 150))
        img.putpixel((0, 0), (50, 100, 150))   # Lowest Luma
        img.putpixel((1, 1), (200, 100, 150))  # Highest Luma
        
        result = apply_contrast_enhancement(img, method="percentile", clip_percent=0)
        result_arr = np.array(result)
        
        self.assertEqual(result.mode, "YCbCr")
        
        # Luma (Y - index 0) should be stretched (50->0, 200->255)
        self.assertEqual(result_arr[0, 0, 0], 0)
        self.assertEqual(result_arr[1, 1, 0], 255)
        
        # Cb (1) and Cr (2) must remain mathematically identical
        self.assertTrue(np.all(result_arr[..., 1] == 100))
        self.assertTrue(np.all(result_arr[..., 2] == 150))

    def test_lab_mode_processing(self) -> None:
        # LAB falls to the standard branch, converting safely to RGB on output
        img = Image.new("LAB", (2, 2), color=(100, 5, 5))
        img.putpixel((0, 0), (50, 5, 5))
        img.putpixel((1, 1), (200, 5, 5))
        
        result = apply_contrast_enhancement(img, method="percentile", clip_percent=0)
        
        self.assertEqual(result.mode, "LAB")
        # Luminance channel should be stretched to full range (50->0, 200->255)
        result_arr = np.array(result)
        self.assertEqual(result_arr[0, 0, 0], 0)
        self.assertEqual(result_arr[1, 1, 0], 255)

    def test_rgba_preserves_alpha_on_reassembly(self) -> None:
        # Verifies that apply_contrast_enhancement correctly re-stacks the alpha channel 
        # and outputs the "RGBA" mode using np.dstack
        img = Image.new("RGBA", (2, 2), color=(100, 100, 100, 128))
        img.putpixel((0, 0), (150, 150, 150, 0))
        
        input_alpha = np.array(img.getchannel("A"))
        result = apply_contrast_enhancement(img, method="percentile", clip_percent=0)
        result_arr = np.array(result)
        
        self.assertEqual(result.mode, "RGBA")
        # Check that the 4th channel (index 3) is perfectly intact
        np.testing.assert_array_equal(result_arr[..., 3], input_alpha)
        # Check that the RGB channels were still stretched (100 -> 0)
        self.assertEqual(result_arr[1, 1, 0], 0)

    def test_la_preserves_alpha_on_reassembly(self) -> None:
        # Verifies that apply_contrast_enhancement correctly handles 2D grayscale + alpha
        # and outputs the "LA" mode using np.dstack
        img = Image.new("LA", (2, 2), color=(125, 128))
        img.putpixel((0, 0), (50, 0))
        
        input_alpha = np.array(img.getchannel("A"))
        result = apply_contrast_enhancement(img, method="percentile", clip_percent=0)
        result_arr = np.array(result)
        
        self.assertEqual(result.mode, "LA")
        # Check that the 2nd channel (index 1) is perfectly intact
        np.testing.assert_array_equal(result_arr[..., 1], input_alpha)
        # Check that the L channel was still stretched (50 -> 0)
        self.assertEqual(result_arr[0, 0, 0], 0)

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