import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from skimage import filters as skfilters

from forensics_app.core import ImageDocument
from forensics_app.tools import filters
from forensics_app.tools.filters import (
    FILTER_LABELS,
    GAUSSIAN,
    MEDIAN,
    OTSU,
    SOBEL,
    UNSHARP,
    FiltersTool,
    apply_filter,
    edges,
    gaussian,
    median,
    otsu,
)

ASK_CHOICE = "forensics_app.tools.filters.ask_choice"
ASKFLOAT = "forensics_app.tools.filters.simpledialog.askfloat"
ASKINTEGER = "forensics_app.tools.filters.simpledialog.askinteger"
NAMES = tuple(FILTER_LABELS)


def noise(height: int, width: int, channels: int | None = None) -> np.ndarray:
    """Return reproducible random floats in [0, 1]."""
    shape = (height, width) if channels is None else (height, width, channels)
    return np.random.default_rng(2).random(shape)


def step_edge() -> np.ndarray:
    """Return a gray image that is black on the left half and white on the right."""
    pixels = np.zeros((12, 12))
    pixels[:, 6:] = 1.0
    return pixels


class FilterFunctionTests(unittest.TestCase):
    def test_gaussian_matches_skimage_on_each_channel(self) -> None:
        pixels = noise(12, 14, 3)
        result = gaussian(pixels, 2.0)
        for index in range(3):
            np.testing.assert_allclose(result[..., index], skfilters.gaussian(pixels[..., index], sigma=2.0))

    def test_larger_sigma_smooths_more(self) -> None:
        pixels = noise(20, 20)
        self.assertLess(gaussian(pixels, 3.0).std(), gaussian(pixels, 1.0).std())

    def test_gaussian_keeps_a_flat_image(self) -> None:
        np.testing.assert_allclose(gaussian(np.full((6, 6, 3), 0.25), 2.0), 0.25)

    def test_median_removes_an_isolated_pixel(self) -> None:
        pixels = np.full((7, 7), 0.2)
        pixels[3, 3] = 1.0
        np.testing.assert_allclose(median(pixels, 3), 0.2)

    def test_median_keeps_a_sharp_edge(self) -> None:
        np.testing.assert_allclose(median(step_edge(), 3), step_edge())

    def test_median_of_one_pixel_changes_nothing(self) -> None:
        pixels = np.round(noise(6, 6, 3) * 255) / 255
        np.testing.assert_allclose(median(pixels, 1), pixels)

    def test_median_matches_skimage_median_away_from_the_borders(self) -> None:
        pixels = np.round(noise(20, 20) * 255) / 255
        expected = skfilters.median(pixels, np.ones((5, 5), dtype=bool))
        np.testing.assert_allclose(median(pixels, 5)[2:-2, 2:-2], expected[2:-2, 2:-2])

    def test_median_rejects_an_empty_window(self) -> None:
        pixels = noise(4, 4)
        with self.assertRaises(ValueError):
            median(pixels, 0)

    def test_edge_operators_respond_only_at_the_edge(self) -> None:
        for operator in filters.EDGE_OPERATORS:
            with self.subTest(operator=operator):
                result = edges(step_edge(), operator)
                self.assertGreater(result[6, 5], 0.0)
                self.assertGreater(result[6, 6], 0.0)
                np.testing.assert_allclose(result[:, :4], 0.0, atol=1e-12)
                np.testing.assert_allclose(result[:, 8:], 0.0, atol=1e-12)

    def test_edges_match_skimage_on_each_channel(self) -> None:
        pixels = noise(10, 10, 3)
        result = edges(pixels, SOBEL)
        for index in range(3):
            np.testing.assert_allclose(result[..., index], skfilters.sobel(pixels[..., index]))

    def test_edges_reject_unknown_operators(self) -> None:
        pixels = noise(4, 4)
        with self.assertRaisesRegex(ValueError, "Unsupported edge operator"):
            edges(pixels, "roberts")

    def test_otsu_separates_two_gray_levels(self) -> None:
        pixels = np.full((8, 8), 0.2)
        pixels[:, 4:] = 0.8
        binary, threshold = otsu(pixels)
        self.assertGreaterEqual(threshold, 0.2)
        self.assertLess(threshold, 0.8)
        np.testing.assert_array_equal(binary, pixels > 0.5)

    def test_otsu_ignores_fully_transparent_pixels(self) -> None:
        # Visible levels 0.4 and 0.6, and a large transparent area hiding white
        pixels = np.full((10, 10), 1.0)
        pixels[:2, :5] = 0.4
        pixels[:2, 5:] = 0.6
        alpha = np.zeros((10, 10), dtype=np.uint8)
        alpha[:2] = 255

        _binary, blind_threshold = otsu(pixels)
        self.assertGreaterEqual(blind_threshold, 0.6)

        binary, threshold = otsu(pixels, alpha)
        self.assertGreaterEqual(threshold, 0.4)
        self.assertLess(threshold, 0.6)
        np.testing.assert_array_equal(binary[:2, :5], 0.0)
        np.testing.assert_array_equal(binary[:2, 5:], 1.0)

    def test_otsu_of_a_fully_transparent_image_uses_every_pixel(self) -> None:
        pixels = np.full((4, 4), 0.2)
        pixels[:, 2:] = 0.8
        _binary, expected = otsu(pixels)
        _binary, threshold = otsu(pixels, np.zeros((4, 4), dtype=np.uint8))
        self.assertEqual(threshold, expected)

    def test_otsu_reduces_color_to_one_channel(self) -> None:
        binary, _threshold = otsu(noise(8, 9, 3))
        self.assertEqual(binary.shape, (8, 9))
        self.assertLessEqual(set(np.unique(binary)), {0.0, 1.0})


class ApplyFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = Image.fromarray((noise(12, 12, 3) * 255).astype(np.uint8))

    def test_rejects_unknown_filters(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported filter"):
            apply_filter(self.image, "laplace")

    def test_parameters_change_the_result(self) -> None:
        cases = (
            (GAUSSIAN, {"sigma": 1.0}, {"sigma": 3.0}),
            (MEDIAN, {"size": 3}, {"size": 5}),
            (UNSHARP, {"radius": 1.0, "amount": 1.0}, {"radius": 1.0, "amount": 2.0}),
        )
        for name, first, second in cases:
            with self.subTest(name=name):
                self.assertNotEqual(
                    apply_filter(self.image, name, **first)[0].tobytes(),
                    apply_filter(self.image, name, **second)[0].tobytes(),
                )

    def test_edges_on_grayscale_give_a_single_channel(self) -> None:
        gray = self.image.convert("L")
        for name in filters.EDGE_OPERATORS:
            with self.subTest(name=name):
                output, _measured = apply_filter(self.image, name, grayscale=True)
                self.assertEqual(output.mode, "L")
                # Pillow rounds the gray image to 8 bits before the filter sees it
                expected = np.asarray(apply_filter(gray, name)[0], dtype=float)
                np.testing.assert_allclose(np.asarray(output), expected, atol=2)
                self.assertEqual(apply_filter(self.image, name)[0].mode, "RGB")

    def test_grayscale_edges_keep_the_alpha(self) -> None:
        image = Image.fromarray((noise(8, 8, 4) * 255).astype(np.uint8), mode="RGBA")
        output, _measured = apply_filter(image, SOBEL, grayscale=True)
        self.assertEqual(output.mode, "LA")
        np.testing.assert_array_equal(np.asarray(output)[..., 1], np.asarray(image)[..., 3])

    def test_grayscale_only_affects_the_edge_operators(self) -> None:
        self.assertEqual(apply_filter(self.image, GAUSSIAN, grayscale=True)[0].mode, "RGB")

    def test_otsu_gives_a_black_and_white_image_and_its_threshold(self) -> None:
        output, measured = apply_filter(self.image, OTSU)
        self.assertEqual(output.mode, "L")
        self.assertEqual(set(np.unique(output)), {0, 255})
        self.assertRegex(measured["Threshold"], r"^\d+\.\d of 255$")

    def test_only_otsu_reports_a_measurement(self) -> None:
        for name in NAMES:
            with self.subTest(name=name):
                _output, measured = apply_filter(self.image, name)
                self.assertEqual(bool(measured), name == OTSU)

    def test_alpha_is_left_untouched(self) -> None:
        image = Image.fromarray((noise(8, 8, 4) * 255).astype(np.uint8), mode="RGBA")
        for name in NAMES:
            with self.subTest(name=name):
                output, _measured = apply_filter(image, name)
                self.assertEqual(output.mode, "LA" if name == OTSU else "RGBA")
                np.testing.assert_array_equal(np.asarray(output)[..., -1], np.asarray(image)[..., 3])

    def test_every_filter_accepts_every_kind_of_input(self) -> None:
        color_modes = ("P", "RGB", "RGBX", "CMYK", "YCbCr", "LAB", "HSV")
        for mode in ("1", "L", "I", "F") + color_modes:
            for name in NAMES:
                with self.subTest(mode=mode, name=name):
                    output, _measured = apply_filter(self.image.convert(mode), name)
                    is_color = mode in color_modes and name != OTSU
                    self.assertEqual(output.mode, "RGB" if is_color else "L")
                    self.assertEqual(output.size, self.image.size)


class FiltersToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = FiltersTool()
        self.document = ImageDocument()
        self.document.current = Image.fromarray((noise(12, 16, 3) * 255).astype(np.uint8))

    def run_tool(self, name: str, floats=(), integers=(), edges_on: int | None = 0):
        with (
            patch(ASK_CHOICE, side_effect=[NAMES.index(name), edges_on]) as self.ask_choice,
            patch(ASKFLOAT, side_effect=list(floats)) as askfloat,
            patch(ASKINTEGER, side_effect=list(integers)) as askinteger,
        ):
            return self.tool.run(None, self.document), askfloat, askinteger

    def test_offers_every_filter(self) -> None:
        with patch(ASK_CHOICE, return_value=None) as ask:
            self.assertIsNone(self.tool.run(None, self.document))
        self.assertEqual(ask.call_args.args[3], tuple(FILTER_LABELS.values()))
        self.assertEqual(len(NAMES), 7)

    def test_gaussian_asks_for_sigma(self) -> None:
        source = self.document.current
        result, askfloat, _askinteger = self.run_tool(GAUSSIAN, floats=[3.0])
        self.assertEqual(askfloat.call_args.kwargs["initialvalue"], filters.DEFAULT_SIGMA)
        self.assertEqual(result.details["Sigma"], 3.0)
        self.assertEqual(result.image.tobytes(), apply_filter(source, GAUSSIAN, sigma=3.0)[0].tobytes())
        self.assertIs(self.document.current, source)

    def test_median_asks_for_a_window_bounded_by_the_image(self) -> None:
        result, _askfloat, askinteger = self.run_tool(MEDIAN, integers=[5])
        self.assertEqual(askinteger.call_args.kwargs["maxvalue"], 12)
        self.assertEqual(result.details["Size"], 5)

        self.document.current = Image.new("L", (300, 200))
        _result, _askfloat, askinteger = self.run_tool(MEDIAN, integers=[3])
        self.assertEqual(askinteger.call_args.kwargs["maxvalue"], filters.MAX_MEDIAN_SIZE)

        self.document.current = Image.new("L", (2, 9))
        _result, _askfloat, askinteger = self.run_tool(MEDIAN, integers=[1])
        self.assertEqual(askinteger.call_args.kwargs["maxvalue"], 2)
        self.assertEqual(askinteger.call_args.kwargs["initialvalue"], 2)

    def test_unsharp_mask_asks_for_radius_then_amount(self) -> None:
        result, askfloat, _askinteger = self.run_tool(UNSHARP, floats=[2.0, 1.5])
        self.assertEqual(askfloat.call_count, 2)
        self.assertEqual((result.details["Radius"], result.details["Amount"]), (2.0, 1.5))

    def test_edge_filters_ask_whether_to_use_grayscale_on_color_images(self) -> None:
        for name in filters.EDGE_OPERATORS:
            for edges_on, mode in ((0, "L"), (1, "RGB")):
                with self.subTest(name=name, edges_on=edges_on):
                    result, _askfloat, _askinteger = self.run_tool(name, edges_on=edges_on)
                    self.assertEqual(self.ask_choice.call_count, 2)
                    self.assertEqual(self.ask_choice.call_args.args[3], filters.EDGE_SOURCE_LABELS)
                    self.assertEqual(result.image.mode, mode)
                    self.assertEqual(result.details["Edges on"], filters.EDGE_SOURCE_LABELS[edges_on])

    def test_edge_filters_do_not_ask_on_grayscale_images(self) -> None:
        for mode in ("L", "LA", "1", "F"):
            with self.subTest(mode=mode):
                self.document.current = Image.new(mode, (8, 8))
                result, _askfloat, _askinteger = self.run_tool(SOBEL)
                self.assertEqual(self.ask_choice.call_count, 1)
                self.assertNotIn("Edges on", result.details)

    def test_only_edge_filters_ask_about_grayscale(self) -> None:
        self.run_tool(OTSU)
        self.assertEqual(self.ask_choice.call_count, 1)

    def test_cancelling_the_grayscale_question_returns_none(self) -> None:
        result, _askfloat, _askinteger = self.run_tool(SOBEL, edges_on=None)
        self.assertIsNone(result)

    def test_filters_without_parameters_ask_nothing(self) -> None:
        for name in (SOBEL, filters.PREWITT, filters.SCHARR, OTSU):
            with self.subTest(name=name):
                result, askfloat, askinteger = self.run_tool(name)
                askfloat.assert_not_called()
                askinteger.assert_not_called()
                self.assertEqual(result.details["Filter"], FILTER_LABELS[name])

    def test_otsu_shows_the_threshold(self) -> None:
        result, _askfloat, _askinteger = self.run_tool(OTSU)
        self.assertIn("Threshold", result.details)
        self.assertEqual(result.details["Mode"], "RGB -> L")

    def test_cancelling_a_parameter_returns_none(self) -> None:
        cases = ((GAUSSIAN, [None], []), (MEDIAN, [], [None]), (UNSHARP, [None], []), (UNSHARP, [1.0, None], []))
        for name, floats, integers in cases:
            with self.subTest(name=name, floats=floats):
                result, _askfloat, _askinteger = self.run_tool(name, floats, integers)
                self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
