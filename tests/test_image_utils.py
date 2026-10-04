import unittest
import numpy as np
from PIL import Image

from forensics_app.tools.image_utils import extract_target_channel

class ImageUtilsTests(unittest.TestCase):
    """Unit tests for the shared image channel extraction utility."""

    def test_rgba_preserves_alpha(self) -> None:
        # Alpha should be correctly separated from the RGB working array
        img = Image.new("RGBA", (2, 2), color=(100, 100, 100, 128))
        img.putpixel((0, 0), (150, 150, 150, 0))
        input_alpha = np.array(img.getchannel("A"))
        
        target, lab, alpha = extract_target_channel(img, as_grayscale=False)
        
        self.assertIsNotNone(alpha)
        np.testing.assert_array_equal(alpha, input_alpha)
        self.assertIsNotNone(lab)

    def test_la_preserves_alpha(self) -> None:
        img = Image.new("LA", (2, 2), color=(125, 128))
        img.putpixel((0, 0), (50, 0))
        input_alpha = np.array(img.getchannel("A"))
        
        target, lab, alpha = extract_target_channel(img, as_grayscale=True)
        
        self.assertIsNotNone(alpha)
        np.testing.assert_array_equal(alpha, input_alpha)
        self.assertIsNone(lab)  # LAB is not generated for grayscale workflows

    def test_palette_transparency_extracts_alpha(self) -> None:
        for transparency in (0, bytes([0, 64, 128, 255])):
            with self.subTest(transparency=transparency):
                img = Image.new("P", (2, 2))
                img.putpalette([50, 50, 50, 150, 150, 150] + [0] * 762)
                img.putdata([0, 1, 0, 1])
                img.info["transparency"] = transparency
                input_alpha = np.array(img.convert("RGBA").getchannel("A"))

                target, lab, alpha = extract_target_channel(img, as_grayscale=False)
                
                self.assertIsNotNone(alpha)
                np.testing.assert_array_equal(alpha, input_alpha)

    def test_high_depth_image_prevents_premature_clipping(self) -> None:
        # Array with values far exceeding standard 8-bit bounds (0-255)
        int32_data = np.array([[100, 500], [1000, 2000]], dtype=np.int32)
        img = Image.fromarray(int32_data)
        
        target, lab, alpha = extract_target_channel(img, as_grayscale=True)
        
        self.assertIsNone(alpha)
        self.assertIsNone(lab)
        
        # The manual normalization should map 100 to 0.0 and 2000 to 1.0 exactly
        self.assertEqual(target[0, 0], 0.0)
        self.assertEqual(target[1, 1], 1.0)
        
        # 500 mapped in range 100-2000 (span 1900): 400/1900 = ~0.2105
        self.assertAlmostEqual(target[0, 1], 400 / 1900, places=4)
        # 1000 mapped in range 100-2000 (span 1900): 900/1900 = ~0.4736
        self.assertAlmostEqual(target[1, 0], 900 / 1900, places=4)

    def test_high_depth_flat_image_becomes_zero(self) -> None:
        # Array with identical values exceeding 8-bit bounds
        int32_data = np.full((2, 2), 1500, dtype=np.int32)
        img = Image.fromarray(int32_data)
        
        target, lab, alpha = extract_target_channel(img, as_grayscale=True)
        
        # Custom normalization maps flat high-depth arrays to 0 to prevent division by zero
        self.assertTrue(np.all(target == 0.0))

    def test_high_depth_image_filters_nan_and_inf(self) -> None:
        # Float array containing corrupted non-finite values
        float_data = np.array([[np.nan, 100.0], [200.0, np.inf]], dtype=np.float32)
        img = Image.fromarray(float_data)
        
        target, lab, alpha = extract_target_channel(img, as_grayscale=True)
        
        # Min valid is 100.0 (becomes 0.0), Max valid is 200.0 (becomes 1.0)
        # NaN is safely replaced by the minimum. Inf is safely replaced by the maximum.
        np.testing.assert_almost_equal(target, [[0.0, 0.0], [1.0, 1.0]])

    def test_high_depth_image_as_color_translates_to_lab(self) -> None:
        # Tests the failsafe where a 1-channel high depth image is used as a reference 
        # for a color image during histogram matching.
        int32_data = np.array([[100, 2000]], dtype=np.int32)
        img = Image.fromarray(int32_data)
        
        # Forcing as_grayscale=False simulates matching it against an RGB base image
        target, lab, alpha = extract_target_channel(img, as_grayscale=False)
        
        self.assertIsNotNone(lab)
        # Verify it was successfully padded to a 3D LAB image (Height, Width, Channels)
        self.assertEqual(lab.shape, (1, 2, 3))
        # The target should remain a 2D luminance array
        self.assertEqual(target.shape, (1, 2))

if __name__ == "__main__":
    unittest.main()