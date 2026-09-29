import unittest

from PIL import Image

from forensics_app.ui.image_view import _displayable


class DisplayableTests(unittest.TestCase):
    def test_converts_modes_tk_cannot_draw(self) -> None:
        for mode, expected in (("LAB", "RGB"), ("La", "LA")):
            with self.subTest(mode=mode):
                self.assertEqual(_displayable(Image.new(mode, (2, 2))).mode, expected)

    def test_keeps_drawable_modes_untouched(self) -> None:
        image = Image.new("CMYK", (2, 2))
        self.assertIs(_displayable(image), image)


if __name__ == "__main__":
    unittest.main()
