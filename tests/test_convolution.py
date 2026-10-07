import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from scipy import ndimage

from forensics_app.core import ImageDocument
from forensics_app.tools.convolution import (
    GAUSSIAN,
    GRADIENT_HORIZONTAL,
    GRADIENT_VERTICAL,
    HIGH_PASS,
    KERNEL_LABELS,
    MEAN,
    RANDOM,
    SHARPEN,
    ConvolutionTool,
    build_kernel,
    convolve_image,
    convolve_pixels,
    is_zero_sum,
)

ASK_CHOICE = "forensics_app.tools.convolution.ask_choice"
ASKINTEGER = "forensics_app.tools.convolution.simpledialog.askinteger"
NAMES = tuple(KERNEL_LABELS)
ZERO_SUM = (HIGH_PASS, GRADIENT_HORIZONTAL, GRADIENT_VERTICAL)


def noise(height: int, width: int, channels: int | None = None) -> np.ndarray:
    """Return reproducible random floats in [0, 1]."""
    shape = (height, width) if channels is None else (height, width, channels)
    return np.random.default_rng(0).random(shape)


def noise_image(height: int, width: int, channels: int = 3) -> Image.Image:
    return Image.fromarray((noise(height, width, channels) * 255).astype(np.uint8))


class BuildKernelTests(unittest.TestCase):
    def test_every_preset_has_the_requested_shape(self) -> None:
        for name in NAMES:
            for rows, cols in ((2, 2), (10, 15), (3, 8)):
                with self.subTest(name=name, rows=rows, cols=cols):
                    self.assertEqual(build_kernel(name, rows, cols).shape, (rows, cols))

    def test_presets_add_up_to_one_or_to_zero(self) -> None:
        for name in NAMES:
            with self.subTest(name=name):
                kernel = build_kernel(name, 4, 7)
                self.assertAlmostEqual(kernel.sum(), 0.0 if name in ZERO_SUM else 1.0)
                self.assertEqual(is_zero_sum(kernel), name in ZERO_SUM)

    def test_mean_weights_are_all_equal(self) -> None:
        np.testing.assert_allclose(build_kernel(MEAN, 10, 15), 1 / 150)

    def test_gaussian_peaks_at_the_center_and_is_symmetric(self) -> None:
        kernel = build_kernel(GAUSSIAN, 5, 7)
        self.assertEqual(np.unravel_index(kernel.argmax(), kernel.shape), (2, 3))
        np.testing.assert_allclose(kernel, kernel[::-1, ::-1])
        self.assertGreater(kernel[2, 3], kernel[0, 0])

    def test_sharpen_and_high_pass_are_negative_except_at_the_center(self) -> None:
        for name in (SHARPEN, HIGH_PASS):
            with self.subTest(name=name):
                kernel = build_kernel(name, 3, 5)
                self.assertGreater(kernel[1, 2], 0.0)
                kernel[1, 2] = -1.0
                self.assertTrue(np.all(kernel < 0.0))

    def test_gradient_halves_have_opposite_signs(self) -> None:
        horizontal = build_kernel(GRADIENT_HORIZONTAL, 3, 5)
        np.testing.assert_array_equal(np.sign(horizontal[0]), [-1, -1, 0, 1, 1])
        vertical = build_kernel(GRADIENT_VERTICAL, 4, 2)
        np.testing.assert_array_equal(np.sign(vertical[:, 0]), [-1, -1, 1, 1])

    def test_random_is_positive_and_repeats_with_the_same_seed(self) -> None:
        first = build_kernel(RANDOM, 4, 4, seed=7)
        self.assertTrue(np.all(first > 0.0))
        np.testing.assert_array_equal(first, build_kernel(RANDOM, 4, 4, seed=7))
        self.assertFalse(np.array_equal(first, build_kernel(RANDOM, 4, 4, seed=8)))

    def test_rejects_unknown_presets_and_empty_sizes(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported kernel"):
            build_kernel("laplacian", 3, 3)
        for rows, cols in ((0, 3), (3, 0), (-1, 2)):
            with self.subTest(rows=rows, cols=cols), self.assertRaises(ValueError):
                build_kernel(MEAN, rows, cols)

    def test_rejects_sizes_that_would_give_an_all_zero_kernel(self) -> None:
        for name, rows, cols in ((HIGH_PASS, 1, 1), (GRADIENT_HORIZONTAL, 5, 1), (GRADIENT_VERTICAL, 1, 5)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                build_kernel(name, rows, cols)


class ConvolvePixelsTests(unittest.TestCase):
    def test_matches_scipy_for_any_kernel_shape(self) -> None:
        pixels = noise(12, 9)
        rng = np.random.default_rng(3)
        for shape in ((1, 1), (3, 3), (1, 5), (5, 1), (2, 2), (4, 6), (5, 4), (12, 9)):
            with self.subTest(shape=shape):
                kernel = rng.normal(size=shape)
                np.testing.assert_allclose(
                    convolve_pixels(pixels, kernel), ndimage.convolve(pixels, kernel, mode="reflect"), atol=1e-9
                )

    def test_identity_kernel_changes_nothing(self) -> None:
        pixels = noise(6, 7, 3)
        np.testing.assert_allclose(convolve_pixels(pixels, np.ones((1, 1))), pixels, atol=1e-12)

    def test_color_channels_are_convolved_independently(self) -> None:
        pixels = noise(8, 10, 3)
        kernel = build_kernel(MEAN, 3, 5)
        result = convolve_pixels(pixels, kernel)
        for index in range(3):
            np.testing.assert_allclose(result[..., index], ndimage.convolve(pixels[..., index], kernel), atol=1e-9)

    def test_flips_the_kernel_unlike_correlation(self) -> None:
        impulse = np.zeros((5, 5))
        impulse[2, 2] = 1.0
        kernel = np.arange(9, dtype=float).reshape(3, 3)
        # Convolving an impulse reproduces the kernel itself
        np.testing.assert_allclose(convolve_pixels(impulse, kernel)[1:4, 1:4], kernel, atol=1e-12)

    def test_rejects_kernels_that_are_not_2d(self) -> None:
        with self.assertRaises(ValueError):
            convolve_pixels(noise(4, 4), np.ones(3))


class ConvolveImageTests(unittest.TestCase):
    def test_unit_sum_kernels_keep_a_flat_image(self) -> None:
        image = Image.new("RGB", (6, 5), (10, 120, 250))
        for name in (MEAN, GAUSSIAN, SHARPEN, RANDOM):
            with self.subTest(name=name):
                result = convolve_image(image, build_kernel(name, 3, 4))
                self.assertEqual(result.getcolors(), [(30, (10, 120, 250))])

    def test_zero_sum_kernels_turn_a_flat_image_mid_gray(self) -> None:
        image = Image.new("L", (6, 5), 200)
        for name in ZERO_SUM:
            with self.subTest(name=name):
                result = convolve_image(image, build_kernel(name, 3, 4))
                self.assertEqual(result.getcolors(), [(30, 128)])

    def test_horizontal_kernel_only_mixes_along_rows(self) -> None:
        image = Image.new("L", (9, 9), 0)
        image.putpixel((4, 4), 255)
        result = np.asarray(convolve_image(image, build_kernel(MEAN, 1, 3)))
        self.assertEqual(set(np.flatnonzero(result.any(axis=1))), {4})
        self.assertEqual(set(np.flatnonzero(result.any(axis=0))), {3, 4, 5})

    def test_vertical_kernel_only_mixes_along_columns(self) -> None:
        image = Image.new("L", (9, 9), 0)
        image.putpixel((4, 4), 255)
        result = np.asarray(convolve_image(image, build_kernel(MEAN, 3, 1)))
        self.assertEqual(set(np.flatnonzero(result.any(axis=1))), {3, 4, 5})
        self.assertEqual(set(np.flatnonzero(result.any(axis=0))), {4})

    def test_mean_blurs_and_sharpen_adds_contrast(self) -> None:
        image = noise_image(20, 20, 3).convert("L")
        spread = np.asarray(image).std()
        self.assertLess(np.asarray(convolve_image(image, build_kernel(MEAN, 5, 5))).std(), spread)
        self.assertGreater(np.asarray(convolve_image(image, build_kernel(SHARPEN, 5, 5))).std(), spread)

    def test_horizontal_gradient_responds_to_a_vertical_edge_only(self) -> None:
        pixels = np.zeros((10, 10), dtype=np.uint8)
        pixels[:, 5:] = 200
        vertical_edge = Image.fromarray(pixels)
        across = np.asarray(convolve_image(vertical_edge, build_kernel(GRADIENT_HORIZONTAL, 3, 2)))
        along = np.asarray(convolve_image(vertical_edge, build_kernel(GRADIENT_VERTICAL, 2, 3)))
        self.assertTrue(np.any(across != 128))
        self.assertTrue(np.all(along == 128))

    def test_kernel_as_large_as_the_image(self) -> None:
        image = noise_image(6, 8)
        self.assertEqual(convolve_image(image, build_kernel(MEAN, 6, 8)).size, image.size)

    def test_alpha_is_left_untouched(self) -> None:
        image = Image.fromarray((noise(7, 7, 4) * 255).astype(np.uint8), mode="RGBA")
        result = convolve_image(image, build_kernel(MEAN, 3, 3))
        self.assertEqual(result.mode, "RGBA")
        np.testing.assert_array_equal(np.asarray(result)[..., 3], np.asarray(image)[..., 3])

    def test_output_mode_for_every_kind_of_input(self) -> None:
        base = noise_image(6, 6)
        expected = {
            "1": "L", "L": "L", "I": "L", "F": "L", "LA": "LA",
            "P": "RGB", "RGB": "RGB", "RGBX": "RGB", "CMYK": "RGB",
            "YCbCr": "RGB", "LAB": "RGB", "HSV": "RGB", "RGBA": "RGBA",
        }
        for mode, output_mode in expected.items():
            with self.subTest(mode=mode):
                result = convolve_image(base.convert(mode), build_kernel(MEAN, 3, 3))
                self.assertEqual(result.mode, output_mode)
                self.assertEqual(result.size, base.size)


class ConvolutionToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = ConvolutionTool()
        self.document = ImageDocument()
        self.document.current = noise_image(20, 30)

    def run_tool(self, name: str | None, sizes: list[int | None]):
        choice = None if name is None else NAMES.index(name)
        with patch(ASK_CHOICE, return_value=choice) as ask_choice, patch(ASKINTEGER, side_effect=sizes) as ask:
            return self.tool.run(None, self.document), ask_choice, ask

    def test_offers_every_preset_with_mean_first(self) -> None:
        _result, ask_choice, _ask = self.run_tool(None, [])
        self.assertEqual(ask_choice.call_args.args[3], tuple(KERNEL_LABELS.values()))
        self.assertEqual(NAMES[0], MEAN)

    def test_asks_rows_then_columns_bounded_by_the_image(self) -> None:
        _result, _ask_choice, ask = self.run_tool(MEAN, [2, 5])
        rows, cols = (call.kwargs for call in ask.call_args_list)
        self.assertEqual((rows["minvalue"], rows["maxvalue"], rows["initialvalue"]), (1, 20, 15))
        self.assertEqual((cols["minvalue"], cols["maxvalue"], cols["initialvalue"]), (1, 30, 15))

    def test_default_never_exceeds_a_small_image(self) -> None:
        self.document.current = Image.new("L", (4, 3))
        _result, _ask_choice, ask = self.run_tool(MEAN, [1, 1])
        self.assertEqual([call.kwargs["initialvalue"] for call in ask.call_args_list], [3, 4])

    def test_returns_the_convolved_image_without_mutating_the_document(self) -> None:
        source = self.document.current
        result, _ask_choice, _ask = self.run_tool(MEAN, [2, 5])
        self.assertIs(self.document.current, source)
        self.assertEqual(result.image.tobytes(), convolve_image(source, build_kernel(MEAN, 2, 5)).tobytes())
        self.assertEqual(result.details["Kernel"], KERNEL_LABELS[MEAN])
        self.assertEqual(result.details["Size"], "2 x 5 (rows x columns)")
        self.assertNotIn("Seed", result.details)
        self.assertNotIn("Values", result.details)
        self.assertFalse(result.preview_only)

    def test_zero_sum_presets_explain_the_gray_background(self) -> None:
        for name in ZERO_SUM:
            with self.subTest(name=name):
                result, _ask_choice, _ask = self.run_tool(name, [3, 3])
                self.assertIn("mid-gray", result.details["Values"])

    def test_random_reports_the_seed_that_reproduces_the_result(self) -> None:
        source = self.document.current
        result, _ask_choice, _ask = self.run_tool(RANDOM, [4, 4])
        kernel = build_kernel(RANDOM, 4, 4, seed=result.details["Seed"])
        self.assertEqual(result.image.tobytes(), convolve_image(source, kernel).tobytes())

    def test_cancelling_any_dialog_returns_none(self) -> None:
        for name, sizes in ((None, []), (MEAN, [None]), (MEAN, [3, None])):
            with self.subTest(name=name, sizes=sizes):
                result, _ask_choice, _ask = self.run_tool(name, sizes)
                self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
