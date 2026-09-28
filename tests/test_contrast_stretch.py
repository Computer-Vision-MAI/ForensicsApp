import unittest
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.contrast_stretch import apply_contrast_stretch, ContrastStretchTool

class ContrastStretchFunctionTests(unittest.TestCase):
    """Unit tests for the apply_contrast_stretch public function."""
    
    def test_rgb_stretching_math(self) -> None:
        # Create an image with restricted pixel ranges (Min: 50, Max: 150)
        img = Image.new("RGB", (2, 2), color=(50, 0, 150))
        img.putpixel((0, 0), (150, 255, 50)) 
        
        result = apply_contrast_stretch(img)
        result_arr = np.array(result)
        
        # Red channel: min was 50 (becomes 0), max was 150 (becomes 255)
        self.assertEqual(result_arr[1, 1, 0], 0)
        self.assertEqual(result_arr[0, 0, 0], 255)

        # Green channel: min was 0 (becomes 0), max was 255 (becomes 255)
        self.assertEqual(result_arr[1, 1, 1], 0)
        self.assertEqual(result_arr[0, 0, 1], 255)

        # Blue channel: min was 50 (becomes 0), max was 150 (becomes 255)
        self.assertEqual(result_arr[1, 1, 2], 255)
        self.assertEqual(result_arr[0, 0, 2], 0)

    def test_rgba_preserves_alpha(self) -> None:
        # Alpha should remain completely untouched by the stretching
        img = Image.new("RGBA", (2, 2), color=(100, 100, 100, 128))
        img.putpixel((0, 0), (150, 150, 150, 0))
        img.putpixel((1, 0), (125, 125, 125, 64))
        img.putpixel((0, 1), (100, 100, 100, 255))
        input_alpha = np.array(img.getchannel("A"))
        
        result = apply_contrast_stretch(img)
        result_arr = np.array(result)
        
        self.assertEqual(result.mode, "RGBA")
        np.testing.assert_array_equal(result_arr[..., 3], input_alpha)

    def test_grayscale_stretching(self) -> None:
        # Ensure 2D arrays (L mode) do not crash the min/max dimensional logic
        img = Image.new("L", (2, 2), color=100)
        img.putpixel((0, 0), 200)
        
        result = apply_contrast_stretch(img)
        result_arr = np.array(result)
        
        self.assertEqual(result.mode, "L")
        self.assertEqual(result_arr[1, 1], 0)
        self.assertEqual(result_arr[0, 0], 255)

    def test_flat_image_prevents_division_by_zero(self) -> None:
        # When all pixels are the same, max - min = 0
        img = Image.new("RGB", (2, 2), color=(100, 100, 100))
        
        result = apply_contrast_stretch(img)
        result_arr = np.array(result)
        
        # The fallback logic should handle it without NaN/Inf corruption
        self.assertTrue(np.all(result_arr == 0))

    def test_numeric_modes_produce_stretched_grayscale_pixels(self) -> None:
        for mode, dtype in (("I", np.int32), ("F", np.float32)):
            with self.subTest(mode=mode):
                img = Image.fromarray(np.array([[-100, 0], [100, 200]], dtype=dtype))
                self.assertEqual(img.mode, mode)

                result = apply_contrast_stretch(img)

                self.assertEqual(result.mode, "L")
                np.testing.assert_array_equal(np.array(result), [[0, 85], [170, 255]])

    def test_palette_image_converts_to_rgb(self) -> None:
        # Palette images should convert to RGB to evaluate actual colors
        img = Image.new("P", (2, 2), color=0)
        palette = [50, 50, 50, 150, 150, 150] + [0] * 762
        img.putpalette(palette)
        img.putpixel((0, 0), 1)
        
        result = apply_contrast_stretch(img)
        
        self.assertEqual(result.mode, "RGB")

    def test_transparent_palette_preserves_alpha(self) -> None:
        for transparency in (0, bytes([0, 64, 128, 255])):
            with self.subTest(transparency=transparency):
                img = Image.new("P", (2, 2))
                img.putpalette([50, 50, 50, 150, 150, 150] + [0] * 762)
                img.putdata([0, 1, 2, 3])
                img.info["transparency"] = transparency
                input_alpha = np.array(img.convert("RGBA").getchannel("A"))

                result = apply_contrast_stretch(img)

                self.assertEqual(result.mode, "RGBA")
                np.testing.assert_array_equal(np.array(result.getchannel("A")), input_alpha)
                self.assertEqual(result.getpixel((1, 0))[:3], (255, 255, 255))

    def test_pa_preserves_alpha(self) -> None:
        img = Image.new("PA", (2, 2))
        img.putpalette([50, 50, 50, 150, 150, 150] + [0] * 762)
        img.putdata([(0, 0), (1, 64), (0, 128), (1, 255)])
        input_alpha = np.array(img.convert("RGBA").getchannel("A"))

        result = apply_contrast_stretch(img)

        self.assertEqual(result.mode, "RGBA")
        np.testing.assert_array_equal(np.array(result.getchannel("A")), input_alpha)
        self.assertEqual(result.getpixel((1, 0))[:3], (255, 255, 255))


class ContrastStretchToolTests(unittest.TestCase):
    """Unit tests for the ContrastStretchTool wrapper."""
    
    def setUp(self) -> None:
        self.tool = ContrastStretchTool()
        self.document = ImageDocument()

    def test_run_fails_without_current_image(self) -> None:
        self.document.current = None
        with self.assertRaises(AssertionError):
            self.tool.run(None, self.document)

    def test_run_executes_successfully(self) -> None:
        self.document.current = Image.new("RGB", (2, 2), color=(50, 50, 50))
        result = self.tool.run(None, self.document)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.details["Operation"], "Contrast Stretch")
        self.assertEqual(result.details["Original Mode"], "RGB")
        self.assertEqual(result.details["Output Mode"], "RGB")

if __name__ == "__main__":
    unittest.main()
