from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.hog import HogParameters
from forensics_app.tools.person_detection import (
    DOT_COLOR,
    EUCLIDEAN_COLOR,
    MAX_WINDOWS,
    Detection,
    PersonDetectionTool,
    detect_person,
    draw_detections,
    smallest_step,
    window_positions,
)

LOAD_IMAGE = "forensics_app.tools.person_detection.load_image_file"
# Both modules use the same tkinter.simpledialog, so one patch answers every question
ASK_INTEGER = "forensics_app.tools.person_detection.simpledialog.askinteger"
SHOW_ERROR = "forensics_app.tools.person_detection.messagebox.showerror"

PARAMETERS = HogParameters(16, 8, 2)


def textured(height: int, width: int, channels: int = 0) -> np.ndarray:
    """Return reproducible random pixels in [0, 1], gray or with ``channels`` channels."""
    shape = (height, width, channels) if channels else (height, width)
    return np.random.default_rng(7).random(shape)


class WindowPositionTests(unittest.TestCase):
    def test_includes_the_last_window_that_fits(self):
        positions = window_positions((10, 12), (4, 5), 3)
        self.assertEqual(sorted({row for row, _ in positions}), [0, 3, 6])
        self.assertEqual(sorted({col for _, col in positions}), [0, 3, 6])
        self.assertEqual(window_positions((10, 12), (10, 12), 3), [(0, 0)])

    def test_step_one_tries_every_position(self):
        self.assertEqual(len(window_positions((10, 12), (4, 5), 1)), 7 * 8)

    def test_rejects_a_template_larger_than_the_image(self):
        with self.assertRaisesRegex(ValueError, "does not fit"):
            window_positions((10, 12), (11, 5), 1)
        with self.assertRaisesRegex(ValueError, "does not fit"):
            window_positions((10, 12), (4, 13), 1)

    def test_rejects_a_step_below_one(self):
        with self.assertRaisesRegex(ValueError, "at least one pixel"):
            window_positions((10, 12), (4, 5), 0)

    def test_smallest_step_keeps_the_window_count_under_the_limit(self):
        for image_shape in [(150, 113), (600, 800), (2000, 3000)]:
            with self.subTest(image_shape=image_shape):
                step = smallest_step(image_shape, (129, 49))
                self.assertLessEqual(len(window_positions(image_shape, (129, 49), step)), MAX_WINDOWS)
                if step > 1:
                    self.assertGreater(len(window_positions(image_shape, (129, 49), step - 1)), MAX_WINDOWS)

    def test_small_searches_use_every_pixel(self):
        self.assertEqual(smallest_step((150, 113), (129, 49)), 1)


class DetectPersonTests(unittest.TestCase):
    def test_both_metrics_find_the_window_the_template_was_cut_from(self):
        for channels in (0, 3):
            with self.subTest(channels=channels):
                pixels = textured(80, 64, channels)
                template = pixels[12:44, 20:44]
                euclidean, dot, windows = detect_person(pixels, template, PARAMETERS, 4)
                self.assertEqual((euclidean.top, euclidean.left), (12, 20))
                self.assertEqual((dot.top, dot.left), (12, 20))
                self.assertAlmostEqual(euclidean.score, 0.0)
                self.assertEqual(windows, len(window_positions(pixels.shape, template.shape, 4)))

    def test_scores_are_the_distance_and_the_dot_product(self):
        pixels = textured(40, 40)
        template = textured(32, 32)[::-1]
        euclidean, dot, _ = detect_person(pixels, template, PARAMETERS, 8)
        self.assertGreater(euclidean.score, 0.0)
        self.assertGreater(dot.score, 0.0)


class DrawDetectionsTests(unittest.TestCase):
    def test_draws_the_window_border_and_leaves_the_rest(self):
        image = Image.new("L", (30, 30), 100)
        output = draw_detections(image, (8, 10), [(Detection(5, 6, 0.0), EUCLIDEAN_COLOR, 1)])
        self.assertEqual(output.mode, "RGB")
        self.assertEqual(output.size, image.size)
        # Corners of an 8 wide, 10 tall window whose top-left pixel is x 6, y 5
        for corner in [(6, 5), (13, 5), (6, 14), (13, 14)]:
            self.assertEqual(output.getpixel(corner), EUCLIDEAN_COLOR)
        self.assertEqual(output.getpixel((9, 9)), (100, 100, 100))
        self.assertEqual(output.getpixel((14, 15)), (100, 100, 100))
        self.assertEqual(image.mode, "L")

    def test_a_wider_first_line_stays_visible_around_the_second(self):
        same = Detection(5, 5, 0.0)
        output = draw_detections(
            Image.new("RGB", (30, 30)), (12, 12), [(same, EUCLIDEAN_COLOR, 4), (same, DOT_COLOR, 2)]
        )
        self.assertEqual(output.getpixel((5, 10)), DOT_COLOR)
        self.assertEqual(output.getpixel((8, 10)), EUCLIDEAN_COLOR)

    def test_numeric_images_are_rescaled_before_drawing(self):
        image = Image.fromarray(np.linspace(0, 60000, 400).reshape(20, 20).astype(np.uint16))
        output = draw_detections(image, (5, 5), [(Detection(0, 0, 0.0), DOT_COLOR, 1)])
        self.assertEqual(output.getpixel((19, 19)), (255, 255, 255))


class PersonDetectionToolTests(unittest.TestCase):
    def setUp(self):
        pixels = (textured(80, 64, 3) * 255).astype(np.uint8)
        self.document = ImageDocument()
        self.document.current = Image.fromarray(pixels)
        self.template = Image.fromarray(pixels[12:44, 20:44])
        self.loaded = (Path("person_template.bmp"), self.template)
        self.tool = PersonDetectionTool()

    def run_tool(self, answers):
        """Run the tool answering orientations, cell, block and step in that order."""
        with patch(LOAD_IMAGE, return_value=self.loaded), patch(ASK_INTEGER, side_effect=answers) as ask:
            result = self.tool.run(None, self.document)
        return result, ask.call_args_list

    def test_finds_the_person_and_reports_both_metrics(self):
        before = self.document.current.copy()
        result, _ = self.run_tool([16, 8, 2, 4])
        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.image.size, before.size)
        self.assertEqual(result.image.getpixel((20, 12)), DOT_COLOR)
        self.assertEqual(result.image.getpixel((23, 15)), EUCLIDEAN_COLOR)
        self.assertEqual(result.details["Euclidean (red)"], "x 20, y 12 | distance 0.000")
        self.assertTrue(result.details["Dot product (blue)"].startswith("x 20, y 12 | score "))
        self.assertEqual(result.details["Same window"], "yes")
        self.assertEqual(result.details["Template"], "person_template.bmp (24 x 32)")
        self.assertEqual(self.document.current.tobytes(), before.tobytes())

    def test_offers_the_course_parameters_and_a_bounded_step(self):
        _, calls = self.run_tool([16, 8, 2, 4])
        defaults = [call.kwargs["initialvalue"] for call in calls]
        self.assertEqual(defaults, [16, 8, 2, 4])
        # The HOG limits come from the template, the shortest side of which is 24 pixels
        self.assertEqual(calls[1].kwargs["maxvalue"], 24)
        # Step 1 would try 49 x 41 = 2009 windows, just over MAX_WINDOWS
        self.assertEqual(calls[3].kwargs["minvalue"], 2)
        self.assertEqual(calls[3].kwargs["maxvalue"], 32)

    def test_cancelling_the_template_returns_none(self):
        with patch(LOAD_IMAGE, return_value=None):
            self.assertIsNone(self.tool.run(None, self.document))

    def test_cancelling_any_parameter_returns_none(self):
        for cancelled in range(4):
            answers = [16, 8, 2, 4]
            answers[cancelled] = None
            with self.subTest(dialog=cancelled):
                self.assertIsNone(self.run_tool(answers)[0])

    def test_a_template_larger_than_the_image_is_reported(self):
        self.loaded = (Path("big.png"), Image.new("RGB", (65, 20)))
        with patch(SHOW_ERROR) as show_error:
            result, calls = self.run_tool([16, 8, 2, 4])
        self.assertIsNone(result)
        show_error.assert_called_once()
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
