import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from skimage import feature

from forensics_app.core import ImageDocument
from forensics_app.tools.hog import (
    HogParameters,
    HogTool,
    compute_hog,
    descriptor_length,
    grid_size,
    hog_image,
)

ASK_INTEGER = "forensics_app.tools.hog.simpledialog.askinteger"


def vertical_edge(size: int = 64) -> np.ndarray:
    """Return a gray square, black on the left half and white on the right half."""
    pixels = np.zeros((size, size))
    pixels[:, size // 2:] = 1.0
    return pixels


class GridTests(unittest.TestCase):
    def test_counts_whole_cells_and_sliding_blocks(self):
        cells, blocks = grid_size((50, 37), HogParameters(9, 8, 3))
        self.assertEqual(cells, (6, 4))
        self.assertEqual(blocks, (4, 2))

    def test_descriptor_length_matches_skimage(self):
        cases = [
            ((64, 48), HogParameters()),
            ((129, 49), HogParameters(16, 8, 2)),
            ((50, 37), HogParameters(5, 4, 2)),
            ((30, 30), HogParameters(1, 30, 1)),
        ]
        for shape, parameters in cases:
            with self.subTest(shape=shape, parameters=parameters):
                pixels = np.random.default_rng(0).random(shape)
                expected = feature.hog(
                    pixels,
                    orientations=parameters.orientations,
                    pixels_per_cell=(parameters.pixels_per_cell,) * 2,
                    cells_per_block=(parameters.cells_per_block,) * 2,
                ).size
                self.assertEqual(descriptor_length(shape, parameters), expected)


class ComputeHogTests(unittest.TestCase):
    def test_more_orientations_give_a_longer_descriptor(self):
        pixels = vertical_edge()
        lengths = [compute_hog(pixels, HogParameters(bins, 8, 2))[0].size for bins in (2, 4, 9)]
        self.assertEqual(lengths, sorted(lengths))
        self.assertEqual(lengths[1], 2 * lengths[0])

    def test_drawing_only_when_asked(self):
        _, drawing = compute_hog(vertical_edge(), HogParameters())
        self.assertIsNone(drawing)
        _, drawing = compute_hog(vertical_edge(), HogParameters(), visualize=True)
        self.assertEqual(drawing.shape, (64, 64))

    def test_flat_image_has_no_gradients(self):
        descriptor, drawing = compute_hog(np.full((32, 32), 0.5), HogParameters(), visualize=True)
        self.assertFalse(descriptor.any())
        self.assertFalse(drawing.any())

    def test_color_uses_the_same_layout_as_gray(self):
        gray = vertical_edge()
        color = np.dstack([gray, gray * 0.5, gray * 0.2])
        self.assertEqual(
            compute_hog(color, HogParameters())[0].size, compute_hog(gray, HogParameters())[0].size
        )

    def test_rejects_an_image_smaller_than_one_block(self):
        with self.assertRaisesRegex(ValueError, "smaller than one HOG block"):
            compute_hog(np.zeros((20, 40)), HogParameters(9, 8, 3))


class HogImageTests(unittest.TestCase):
    def test_lines_appear_only_around_the_edge(self):
        image = Image.fromarray((vertical_edge() * 255).astype(np.uint8))
        output, _ = hog_image(image, HogParameters(9, 8, 1))
        drawing = np.asarray(output)
        # The edge sits between columns 31 and 32; cells farther than one cell see a flat area
        self.assertEqual(drawing[:, :16].max(), 0)
        self.assertEqual(drawing[:, 48:].max(), 0)
        # The strongest line is stretched to white
        self.assertEqual(drawing.max(), 255)

    def test_flat_image_draws_black(self):
        output, _ = hog_image(Image.new("RGB", (40, 40), "orange"), HogParameters())
        self.assertFalse(np.asarray(output).any())

    def test_reports_the_descriptor_and_the_grid(self):
        _, measured = hog_image(Image.new("L", (48, 64)), HogParameters(9, 8, 3))
        self.assertEqual(measured["Descriptor length"], str(descriptor_length((64, 48), HogParameters())))
        self.assertEqual(measured["Cells"], "8 x 6 (rows x columns)")
        self.assertEqual(measured["Blocks"], "6 x 4 (rows x columns)")

    def test_every_kind_of_input_gives_a_gray_drawing_of_the_same_size(self):
        images = [
            Image.new("RGB", (40, 30), "red"),
            Image.new("RGBA", (40, 30), (0, 0, 0, 0)),
            Image.new("LA", (40, 30)),
            Image.new("L", (40, 30)).convert("P"),
            Image.new("1", (40, 30)),
            Image.new("CMYK", (40, 30)),
            Image.new("I;16", (40, 30), 4000),
            Image.new("F", (40, 30), 0.25),
        ]
        for image in images:
            with self.subTest(mode=image.mode):
                output, _ = hog_image(image, HogParameters())
                self.assertEqual(output.mode, "L")
                self.assertEqual(output.size, image.size)


class HogToolTests(unittest.TestCase):
    def setUp(self):
        self.document = ImageDocument()
        self.document.current = Image.fromarray((vertical_edge() * 255).astype(np.uint8)).convert("RGB")
        self.tool = HogTool()

    def test_asks_each_parameter_with_its_default(self):
        with patch(ASK_INTEGER, side_effect=[9, 8, 3]) as ask:
            self.tool.run(None, self.document)
        defaults = [call.kwargs["initialvalue"] for call in ask.call_args_list]
        self.assertEqual(defaults, [9, 8, 3])

    def test_block_limit_depends_on_the_chosen_cell(self):
        with patch(ASK_INTEGER, side_effect=[9, 16, 2]) as ask:
            self.tool.run(None, self.document)
        # 64 pixels hold 4 cells of 16, so a block can have up to 4 cells per side
        self.assertEqual(ask.call_args_list[1].kwargs["maxvalue"], 64)
        self.assertEqual(ask.call_args_list[2].kwargs["maxvalue"], 4)

    def test_returns_the_drawing_without_mutating_the_document(self):
        before = self.document.current.copy()
        with patch(ASK_INTEGER, side_effect=[4, 8, 2]):
            result = self.tool.run(None, self.document)
        self.assertEqual(result.image.mode, "L")
        self.assertEqual(result.image.size, before.size)
        self.assertFalse(result.preview_only)
        self.assertEqual(self.document.current.tobytes(), before.tobytes())
        self.assertEqual(result.details["Orientations"], 4)
        self.assertEqual(result.details["Pixels per cell"], "8 x 8")
        self.assertEqual(result.details["Cells per block"], "2 x 2")
        self.assertEqual(result.details["Mode"], "RGB -> L")

    def test_cancelling_any_dialog_returns_none(self):
        for cancelled in range(3):
            answers = [9, 8, 3]
            answers[cancelled] = None
            with self.subTest(dialog=cancelled), patch(ASK_INTEGER, side_effect=answers):
                self.assertIsNone(self.tool.run(None, self.document))


if __name__ == "__main__":
    unittest.main()
