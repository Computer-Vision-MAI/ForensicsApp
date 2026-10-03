import unittest

from PIL import Image

from forensics_app.core.pixel_info import pixel_color, describe_pixel


class PixelInfoTests(unittest.TestCase):
    def test_rgba_drops_alpha(self) -> None:
        image = Image.new("RGBA", (2, 2), (200, 20, 30, 128))
        self.assertEqual(pixel_color(image, 1, 1), (200, 20, 30))

    def test_grayscale_repeats_value(self) -> None:
        image = Image.new("L", (2, 2), 128)
        self.assertEqual(pixel_color(image, 0, 0), (128, 128, 128))

    def test_palette_uses_palette_color(self) -> None:
        image = Image.new("P", (2, 2), 1)
        image.putpalette([0, 0, 0, 200, 20, 30])
        self.assertEqual(pixel_color(image, 1, 1), (200, 20, 30))

    def test_float_image_has_no_color(self) -> None:
        image = Image.new("F", (2, 2), 0.5)
        self.assertIsNone(pixel_color(image, 0, 0))

    def test_sixteen_bit_image_has_no_color(self) -> None:
        self.assertIsNone(pixel_color(Image.new("I;16", (2, 2)), 0, 0))

    def test_integer_image_has_no_color(self) -> None:
        self.assertIsNone(pixel_color(Image.new("I", (2, 2), 70000), 0, 0))

    def test_binary_image_uses_black_or_white(self) -> None:
        self.assertEqual(pixel_color(Image.new("1", (2, 2), 1), 0, 0), (255, 255, 255))

    def test_luminance_alpha_uses_gray(self) -> None:
        image = Image.new("LA", (2, 2), (77, 200))
        self.assertEqual(pixel_color(image, 0, 0), (77, 77, 77))

    def test_premultiplied_luminance_alpha_uses_gray(self) -> None:
        image = Image.new("La", (2, 2), (77, 255))
        self.assertEqual(pixel_color(image, 0, 0), (77, 77, 77))

    def test_cmyk_is_converted_to_screen_color(self) -> None:
        image = Image.new("CMYK", (2, 2), (255, 0, 0, 0))  # full cyan ink
        self.assertEqual(pixel_color(image, 0, 0), (0, 255, 255))

    def test_reads_the_requested_pixel(self) -> None:
        image = Image.new("RGB", (2, 1), (0, 0, 0))
        image.putpixel((1, 0), (200, 20, 30))
        self.assertEqual(pixel_color(image, 1, 0), (200, 20, 30))


class DescribePixelTests(unittest.TestCase):
    def test_describe_pixel_rgb(self) -> None:
        image = Image.new("RGB", (2, 2), (200, 20, 30))
        self.assertEqual(describe_pixel(image, 1, 1), "x 1, y 1 | RGB (200, 20, 30)")

    def test_describe_pixel_palette(self) -> None:
        image = Image.new("P", (2, 2), 1)
        image.putpalette([0, 0, 0, 200, 20, 30])
        self.assertEqual(describe_pixel(image, 0, 0), "x 0, y 0 | P index 1 -> RGB (200, 20, 30)")

    def test_describe_pixel_grayscale(self) -> None:
        image = Image.new("L", (2, 2), 128)
        self.assertEqual(describe_pixel(image, 1, 1), "x 1, y 1 | L 128")

    def test_describe_pixel_float(self) -> None:
        image = Image.new("F", (2, 2), 0.5)
        self.assertEqual(describe_pixel(image, 0, 0), "x 0, y 0 | F 0.500")


if __name__ == "__main__":
    unittest.main()
