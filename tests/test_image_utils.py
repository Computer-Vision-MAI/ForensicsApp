import unittest
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock

from forensics_app.tools.image_utils import (
    extract_target_channel,
    image_to_unit_array,
    normalize_high_depth_image,
    per_channel,
    unit_array_to_image,
)

class ExtractTargetChannelTest(unittest.TestCase):
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

    def test_float_range_and_degenerate_ranges(self) -> None:
        cases = (
            ([100.0, 150.0, 200.0], [0.0, 0.5, 1.0]),
            ([100.0, 100.0, 100.0], [0.0, 0.0, 0.0]),
            ([np.nan, np.inf, -np.inf], [0.0, 0.0, 0.0]),
            ([-np.inf, 100.0, np.nan], [0.0, 0.0, 0.0]),
        )
        for values, expected in cases:
            with self.subTest(values=values):
                img = Image.fromarray(np.array([values], dtype=np.float32))
                target, _, _ = extract_target_channel(img, as_grayscale=True)
                np.testing.assert_array_equal(target, [expected])

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
    
    def test_rgb_returns_no_alpha(self) -> None:
        img = Image.new("RGB", (2, 2), color=(100, 150, 200))
        target, lab, alpha = extract_target_channel(img, as_grayscale=False)
        
        self.assertIsNone(alpha)
        self.assertIsNotNone(lab)
        # RGB is converted to LAB, so lab array should be 3D and target 2D
        self.assertEqual(lab.shape, (2, 2, 3))
        self.assertEqual(target.shape, (2, 2))

    def test_l_returns_no_alpha(self) -> None:
        img = Image.new("L", (2, 2), color=128)
        target, lab, alpha = extract_target_channel(img, as_grayscale=True)
        
        self.assertIsNone(alpha)
        self.assertIsNone(lab)  # Grayscale workflows skip LAB conversion
        self.assertEqual(target.shape, (2, 2))

    def test_1_returns_no_alpha(self) -> None:
        # Binary (1-bit) images should be treated as opaque grayscale
        img = Image.new("1", (2, 2), color=1)
        target, lab, alpha = extract_target_channel(img, as_grayscale=True)
        
        self.assertIsNone(alpha)
        self.assertIsNone(lab)
        # Target must be successfully normalized to [0, 1] float
        self.assertTrue(np.all(target == 1.0))

    def test_palette_without_transparency_returns_no_alpha(self) -> None:
        # A standard palette image without a transparency key in its info dictionary
        img = Image.new("P", (2, 2))
        img.putpalette([50, 50, 50, 150, 150, 150] + [0] * 762)
        img.putdata([0, 1, 0, 1])
        
        target, lab, alpha = extract_target_channel(img, as_grayscale=False)
        
        self.assertIsNone(alpha)
        self.assertIsNotNone(lab)

    def test_unsupported_mode_raises_value_error(self) -> None:
        # "XYZ" is a valid Pillow color space, but it is explicitly NOT in our 
        # forensic supported_modes allowlist, so it must be rejected to prevent corruption.
        img = MagicMock(spec=Image.Image)
        img.mode = "XYZ"
        
        with self.assertRaisesRegex(ValueError, "Unsupported image mode: 'XYZ'"):
            extract_target_channel(img, as_grayscale=False)

class NormalizeHighDepthImageTest(unittest.TestCase):
    """Unit tests for the normalize_high_depth_image function."""

    def test_high_depth_image_prevents_premature_clipping(self) -> None:
        # Array with values far exceeding standard 8-bit bounds (0-255)
        int32_data = np.array([[100, 500], [1000, 2000]], dtype=np.int32)
        img = Image.fromarray(int32_data)

        target = normalize_high_depth_image(img)
        
        # The smallest value should map to 0.0, the largest to 1.0, and intermediate values proportionally
        self.assertEqual(target[0, 0], 0.0)
        self.assertEqual(target[0, 1], 400 / 1900)
        self.assertEqual(target[1, 0], 900 / 1900)
        self.assertEqual(target[1, 1], 1.0)
    
    def test_high_depth_flat_image_becomes_zero(self) -> None:
        # Array with identical values exceeding 8-bit bounds
        int32_data = np.full((2, 2), 1500, dtype=np.int32)
        img = Image.fromarray(int32_data)
        
        target = normalize_high_depth_image(img)
        
        # Custom normalization maps flat high-depth arrays to 0 to prevent division by zero
        self.assertEqual(target[0, 0], 0.0)

    def test_high_depth_image_filters_nan_and_inf(self) -> None:
        # Float array containing corrupted non-finite values
        float_data = np.array([[np.nan, 100.0], [200.0, np.inf]], dtype=np.float32)
        img = Image.fromarray(float_data)
        
        target = normalize_high_depth_image(img)
        
        # Finite values define the range; NaN/Inf map to its endpoints.
        np.testing.assert_array_equal(target, [[0.0, 0.0], [1.0, 1.0]])
        self.assertLess(target[0, 1], target[1, 0])


class UnitArrayTest(unittest.TestCase):
    """Unit tests for the float conversions shared by the filtering tools."""

    def test_grayscale_modes_give_a_2d_array(self) -> None:
        for mode in ("1", "L"):
            with self.subTest(mode=mode):
                pixels, alpha = image_to_unit_array(Image.new(mode, (3, 2), 255))
                self.assertEqual(pixels.shape, (2, 3))
                self.assertTrue(np.all(pixels == 1.0))
                self.assertIsNone(alpha)

    def test_color_modes_are_converted_to_rgb(self) -> None:
        source = Image.new("RGB", (3, 2), (255, 0, 51))
        for mode in ("RGB", "RGBX", "CMYK", "P"):
            with self.subTest(mode=mode):
                pixels, alpha = image_to_unit_array(source.convert(mode))
                self.assertEqual(pixels.shape, (2, 3, 3))
                np.testing.assert_allclose(pixels[0, 0], [1.0, 0.0, 0.2])
                self.assertIsNone(alpha)

    def test_alpha_is_returned_apart(self) -> None:
        cases = {
            "RGBA": Image.new("RGBA", (2, 2), (10, 20, 30, 77)),
            "LA": Image.new("LA", (2, 2), (10, 77)),
            "La": Image.new("LA", (2, 2), (10, 77)).convert("La"),
        }
        for mode, image in cases.items():
            with self.subTest(mode=mode):
                pixels, alpha = image_to_unit_array(image)
                self.assertEqual(pixels.ndim, 3 if mode == "RGBA" else 2)
                self.assertTrue(np.all(alpha == 77))

    def test_palette_transparency_becomes_alpha(self) -> None:
        image = Image.new("P", (2, 2))
        image.putpalette([50, 50, 50, 150, 150, 150] + [0] * 762)
        image.putdata([0, 1, 0, 1])
        image.info["transparency"] = 0
        pixels, alpha = image_to_unit_array(image)
        self.assertEqual(pixels.shape, (2, 2, 3))
        np.testing.assert_array_equal(alpha, [[0, 255], [0, 255]])

    def test_numeric_modes_are_stretched_to_the_unit_range(self) -> None:
        image = Image.fromarray(np.array([[1000, 2000], [3000, 5000]], dtype=np.int32))
        pixels, alpha = image_to_unit_array(image)
        np.testing.assert_allclose(pixels, [[0.0, 0.25], [0.5, 1.0]])
        self.assertIsNone(alpha)

    def test_unknown_mode_is_rejected(self) -> None:
        image = MagicMock()
        image.mode = "XYZ"
        with self.assertRaisesRegex(ValueError, "Unsupported image mode"):
            image_to_unit_array(image)

    def test_round_trip_keeps_8bit_images(self) -> None:
        data = np.random.default_rng(0).integers(0, 256, (5, 6, 4), dtype=np.uint8)
        for mode, array in (("L", data[..., 0]), ("LA", data[..., :2]), ("RGB", data[..., :3]), ("RGBA", data)):
            with self.subTest(mode=mode):
                image = Image.fromarray(array, mode=mode)
                result = unit_array_to_image(*image_to_unit_array(image))
                self.assertEqual(result.mode, mode)
                self.assertEqual(result.tobytes(), image.tobytes())

    def test_values_outside_the_unit_range_are_clipped(self) -> None:
        result = unit_array_to_image(np.array([[-0.5, 0.5, 1.5]]))
        np.testing.assert_array_equal(np.asarray(result), [[0, 128, 255]])

    def test_per_channel_applies_the_function_to_each_channel(self) -> None:
        gray = np.arange(6.0).reshape(2, 3)
        np.testing.assert_array_equal(per_channel(np.fliplr, gray), np.fliplr(gray))

        color = np.arange(18.0).reshape(2, 3, 3)
        seen = []
        result = per_channel(lambda channel: seen.append(channel.shape) or channel * 2, color)
        self.assertEqual(seen, [(2, 3)] * 3)
        np.testing.assert_array_equal(result, color * 2)

if __name__ == "__main__":
    unittest.main()