import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.convert_mode import (
    TARGET_MODES,
    ConvertModeTool,
    convert_mode,
    has_transparency,
    rescale_to_8bit,
)

ASK_TARGET = "forensics_app.tools.convert_mode.ask_choice"


class ConvertModeTests(unittest.TestCase):
    def test_converts_every_mode_to_every_target(self) -> None:
        for source in Image.MODES:
            for target in TARGET_MODES:
                if source == target:
                    continue
                image = Image.new(source, (2, 2))
                with self.subTest(source=source, target=target):
                    self.assertEqual(convert_mode(image, target).mode, target)

    def test_drops_alpha_without_blending(self) -> None:
        image = Image.new("RGBA", (2, 2), (200, 20, 30, 0))
        self.assertEqual(convert_mode(image, "RGB").getpixel((0, 0)), (200, 20, 30))

    def test_keeps_alpha_when_the_target_has_it(self) -> None:
        image = Image.new("RGBA", (2, 2), (200, 20, 30, 77))
        self.assertEqual(convert_mode(image, "LA").getpixel((0, 0))[1], 77)

    def test_uses_screen_color_for_cmyk(self) -> None:
        image = Image.new("CMYK", (2, 2), (255, 0, 0, 0))  # full cyan ink
        self.assertEqual(convert_mode(image, "RGB").getpixel((0, 0)), (0, 255, 255))

    def test_rescales_numeric_images_instead_of_clipping(self) -> None:
        image = Image.new("I;16", (3, 1))
        image.putpixel((1, 0), 20000)
        image.putpixel((2, 0), 40000)
        self.assertEqual(np.asarray(convert_mode(image, "L")).tolist(), [[0, 128, 255]])

    def test_rejects_the_current_mode(self) -> None:
        image = Image.new("RGB", (2, 2))
        with self.assertRaises(ValueError):
            convert_mode(image, "RGB")

    def test_rejects_unsupported_targets(self) -> None:
        image = Image.new("RGB", (2, 2))
        for target in ("P", "1", "I", "XYZ"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                convert_mode(image, target)


class RescaleTests(unittest.TestCase):
    def test_keeps_precision_for_large_close_values(self) -> None:
        image = Image.new("I", (2, 1), 2_000_000_000)
        image.putpixel((1, 0), 2_000_000_100)
        self.assertEqual(np.asarray(rescale_to_8bit(image)).tolist(), [[0, 255]])

    def test_rescales_float_images(self) -> None:
        image = Image.new("F", (3, 1), 0.0)
        image.putpixel((1, 0), 0.5)
        image.putpixel((2, 0), 1.0)
        result = rescale_to_8bit(image)
        self.assertEqual(result.mode, "L")
        self.assertEqual(np.asarray(result).tolist(), [[0, 128, 255]])

    def test_flat_image_becomes_black(self) -> None:
        result = rescale_to_8bit(Image.new("F", (2, 2), 0.5))
        self.assertEqual(result.size, (2, 2))
        self.assertTrue(np.all(np.asarray(result) == 0))


class HasTransparencyTests(unittest.TestCase):
    def test_detects_alpha_channels_and_palette_transparency(self) -> None:
        palette = Image.new("P", (2, 2))
        palette.info["transparency"] = 0
        for image in (Image.new("RGBA", (2, 2)), Image.new("LA", (2, 2)), Image.new("PA", (2, 2)), palette):
            with self.subTest(mode=image.mode):
                self.assertTrue(has_transparency(image))

    def test_opaque_modes_have_none(self) -> None:
        for mode in ("RGB", "L", "CMYK", "P"):
            with self.subTest(mode=mode):
                self.assertFalse(has_transparency(Image.new(mode, (2, 2))))


class ConvertModeToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = ImageDocument()
        self.document.current = Image.new("RGBA", (4, 3), (200, 20, 30, 0))

    def test_converts_to_the_selected_target_without_mutating_document(self) -> None:
        with patch(ASK_TARGET, return_value=0):
            result = ConvertModeTool().run(None, self.document)

        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.image.getpixel((0, 0)), (200, 20, 30))
        self.assertEqual(result.details["From"], "RGBA")
        self.assertEqual(result.details["To"], "RGB")
        self.assertEqual(result.details["Alpha"], "discarded")
        self.assertNotIn("Values", result.details)
        self.assertEqual(self.document.current.mode, "RGBA")

    def test_offers_every_target_except_the_current_mode(self) -> None:
        with patch(ASK_TARGET, return_value=0) as ask:
            ConvertModeTool().run(None, self.document)

        labels = ask.call_args.args[3]
        self.assertEqual(len(labels), len(TARGET_MODES) - 1)
        self.assertFalse(any(label.startswith("RGBA ") for label in labels))
        self.assertIn("CMYK (Cyan, Magenta, Yellow, Black)", labels)

    def test_each_label_selects_its_own_target(self) -> None:
        for source in ("RGBA", "L", "I;16"):
            self.document.current = Image.new(source, (4, 3))
            offered = len([mode for mode in TARGET_MODES if mode != source])
            for index in range(offered):
                with self.subTest(source=source, index=index), patch(ASK_TARGET, return_value=index) as ask:
                    result = ConvertModeTool().run(None, self.document)
                    label = ask.call_args.args[3][index]
                    self.assertTrue(label.startswith(f"{result.image.mode} ("))

    def test_reports_what_happened_to_alpha(self) -> None:
        cases = [("RGBA", 2, "kept"), ("RGBA", 0, "discarded"), ("RGB", 0, "none")]
        for source, index, expected in cases:
            self.document.current = Image.new(source, (4, 3))
            with self.subTest(source=source, index=index), patch(ASK_TARGET, return_value=index):
                result = ConvertModeTool().run(None, self.document)
                self.assertEqual(result.details["Alpha"], expected)

    def test_reports_rescaled_values_for_numeric_images(self) -> None:
        self.document.current = Image.new("I;16", (4, 3))
        with patch(ASK_TARGET, return_value=0):
            result = ConvertModeTool().run(None, self.document)
        self.assertEqual(result.details["Values"], "rescaled to 0-255")

    def test_returns_none_when_dialog_is_cancelled(self) -> None:
        with patch(ASK_TARGET, return_value=None):
            self.assertIsNone(ConvertModeTool().run(None, self.document))


if __name__ == "__main__":
    unittest.main()
