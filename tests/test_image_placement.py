import unittest

from forensics_app.ui.image_placement import ImagePlacement


class ImagePlacementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.placement = ImagePlacement(
            left=150, top=150, shown_size=(500, 300), source_size=(1000, 600)
        )

    def test_maps_center_point(self) -> None:
        self.assertEqual(self.placement.to_image_coords(400, 300), (500, 300))

    def test_maps_corners(self) -> None:
        self.assertEqual(self.placement.to_image_coords(150, 150), (0, 0))
        self.assertEqual(self.placement.to_image_coords(649, 449), (998, 598))

    def test_returns_none_outside(self) -> None:
        for point in ((149, 300), (650, 300), (400, 149), (400, 450)):
            with self.subTest(point=point):
                self.assertIsNone(self.placement.to_image_coords(*point))

    def test_unscaled_image(self) -> None:
        placement = ImagePlacement(left=10, top=20, shown_size=(100, 80), source_size=(100, 80))
        self.assertEqual(placement.to_image_coords(15, 25), (5, 5))


if __name__ == "__main__":
    unittest.main()
