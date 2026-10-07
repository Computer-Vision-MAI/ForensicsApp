import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from skimage import filters

from forensics_app.core import ImageDocument
from forensics_app.tools import sharpen as sharpen_module
from forensics_app.tools.sharpen import (
    DETAIL,
    MANUAL,
    METHODS,
    SKIMAGE,
    SharpenTool,
    detail_layer,
    sharpen,
    sharpen_manual,
    sharpen_skimage,
)

ASK_CHOICE = "forensics_app.tools.sharpen.ask_choice"
ASKFLOAT = "forensics_app.tools.sharpen.simpledialog.askfloat"


def noise(height: int, width: int, channels: int | None = None) -> np.ndarray:
    """Return reproducible random floats in [0, 1]."""
    shape = (height, width) if channels is None else (height, width, channels)
    return np.random.default_rng(1).random(shape)


def step_edge() -> np.ndarray:
    """Return a gray image that is dark on the left half and bright on the right."""
    pixels = np.full((16, 16), 0.3)
    pixels[:, 8:] = 0.7
    return pixels


class SharpenFunctionTests(unittest.TestCase):
    def test_manual_and_skimage_agree(self) -> None:
        for pixels in (noise(20, 24), noise(20, 24, 3)):
            with self.subTest(ndim=pixels.ndim):
                np.testing.assert_allclose(
                    sharpen_manual(pixels, 2.0, 1.5), sharpen_skimage(pixels, 2.0, 1.5), atol=1e-12
                )

    def test_skimage_color_matches_each_channel_alone(self) -> None:
        pixels = noise(12, 14, 3)
        result = sharpen_skimage(pixels, 2.0, 1.5)
        for index in range(3):
            expected = filters.unsharp_mask(pixels[..., index], radius=2.0, amount=1.5)
            np.testing.assert_allclose(result[..., index], expected)

    def test_zero_amount_changes_nothing(self) -> None:
        pixels = noise(10, 10, 3)
        np.testing.assert_allclose(sharpen_manual(pixels, 2.0, 0.0), pixels)
        np.testing.assert_allclose(sharpen_skimage(pixels, 2.0, 0.0), pixels)

    def test_sharpening_increases_the_contrast_across_an_edge(self) -> None:
        result = sharpen_manual(step_edge(), 2.0, 1.5)
        self.assertLess(result[8, 7], 0.3)
        self.assertGreater(result[8, 8], 0.7)
        # Far from the edge nothing changes
        self.assertAlmostEqual(result[8, 0], 0.3, places=3)

    def test_larger_amount_sharpens_more(self) -> None:
        mild = sharpen_manual(step_edge(), 2.0, 0.5)
        strong = sharpen_manual(step_edge(), 2.0, 2.0)
        self.assertGreater(strong[8, 8], mild[8, 8])

    def test_result_stays_inside_the_unit_range(self) -> None:
        for function in (sharpen_manual, sharpen_skimage):
            result = function(noise(10, 10), 2.0, 10.0)
            self.assertGreaterEqual(result.min(), 0.0)
            self.assertLessEqual(result.max(), 1.0)

    def test_detail_of_a_flat_image_is_zero(self) -> None:
        np.testing.assert_allclose(detail_layer(np.full((8, 8, 3), 0.4), 2.0), 0.0, atol=1e-12)

    def test_detail_is_signed_around_an_edge(self) -> None:
        detail = detail_layer(step_edge(), 2.0)
        self.assertLess(detail[8, 7], 0.0)
        self.assertGreater(detail[8, 8], 0.0)


class SharpenImageTests(unittest.TestCase):
    def test_rejects_unknown_methods(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported sharpening method"):
            sharpen(Image.new("L", (4, 4)), "laplacian", 2.0)

    def test_detail_of_a_flat_image_is_mid_gray(self) -> None:
        result = sharpen(Image.new("L", (6, 6), 90), DETAIL, 2.0)
        self.assertEqual(result.getcolors(), [(36, 128)])

    def test_alpha_is_left_untouched(self) -> None:
        image = Image.fromarray((noise(8, 8, 4) * 255).astype(np.uint8), mode="RGBA")
        for method in METHODS:
            with self.subTest(method=method):
                result = sharpen(image, method, 2.0)
                self.assertEqual(result.mode, "RGBA")
                np.testing.assert_array_equal(np.asarray(result)[..., 3], np.asarray(image)[..., 3])

    def test_output_mode_for_every_kind_of_input(self) -> None:
        base = Image.fromarray((noise(8, 8, 3) * 255).astype(np.uint8))
        expected = {
            "1": "L", "L": "L", "I": "L", "F": "L", "LA": "LA",
            "P": "RGB", "RGB": "RGB", "CMYK": "RGB", "YCbCr": "RGB",
            "LAB": "RGB", "HSV": "RGB", "RGBA": "RGBA",
        }
        for mode, output_mode in expected.items():
            for method in METHODS:
                with self.subTest(mode=mode, method=method):
                    result = sharpen(base.convert(mode), method, 2.0)
                    self.assertEqual(result.mode, output_mode)
                    self.assertEqual(result.size, base.size)


class SharpenToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = SharpenTool()
        self.document = ImageDocument()
        self.document.current = Image.fromarray((noise(12, 12, 3) * 255).astype(np.uint8))

    def run_tool(self, choice: int | None, floats: list[float | None]):
        with patch(ASK_CHOICE, return_value=choice), patch(ASKFLOAT, side_effect=floats) as ask:
            return self.tool.run(None, self.document), ask

    def test_every_parameter_has_a_default_to_accept(self) -> None:
        _result, ask = self.run_tool(METHODS.index(MANUAL), [2.0, 1.5])
        defaults = [call.kwargs["initialvalue"] for call in ask.call_args_list]
        self.assertEqual(defaults, [sharpen_module.DEFAULT_SIGMA, sharpen_module.DEFAULT_AMOUNT])

    def test_manual_and_skimage_apply_the_sharpened_image(self) -> None:
        source = self.document.current
        for method, key in ((MANUAL, "Sigma"), (SKIMAGE, "Radius")):
            with self.subTest(method=method):
                result, _ask = self.run_tool(METHODS.index(method), [3.0, 0.8])
                self.assertEqual(result.image.tobytes(), sharpen(source, method, 3.0, 0.8).tobytes())
                self.assertEqual(result.details[key], 3.0)
                self.assertEqual(result.details["Amount"], 0.8)
                self.assertFalse(result.preview_only)
        self.assertIs(self.document.current, source)

    def test_detail_view_is_a_preview_and_skips_the_amount(self) -> None:
        result, ask = self.run_tool(METHODS.index(DETAIL), [2.0])
        self.assertTrue(result.preview_only)
        self.assertNotIn("Amount", result.details)
        self.assertEqual(ask.call_count, 1)
        self.assertEqual(result.image.tobytes(), sharpen(self.document.current, DETAIL, 2.0).tobytes())

    def test_cancelling_any_dialog_returns_none(self) -> None:
        for choice, floats in ((None, []), (0, [None]), (0, [2.0, None]), (2, [None])):
            with self.subTest(choice=choice, floats=floats):
                result, _ask = self.run_tool(choice, floats)
                self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
