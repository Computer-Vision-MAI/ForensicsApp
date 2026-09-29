import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

from forensics_app.ui.image_placement import ImagePlacement
from forensics_app.ui.image_view import ImageView


class ImageViewHoverTests(unittest.TestCase):
    def setUp(self) -> None:
        # Exercise the real hover logic without creating a Tk window.
        self.view = ImageView.__new__(ImageView)
        self.view.canvas = MagicMock()
        self.view.on_hover = MagicMock()
        self.view._source = None
        self.view._pointer = None
        self.view._placement = ImagePlacement(
            left=150, top=150, shown_size=(500, 300), source_size=(1000, 600)
        )

    def test_motion_over_image_reports_source_pixel(self) -> None:
        self.view._on_motion(SimpleNamespace(x=400, y=300))
        self.view.on_hover.assert_called_with((500, 300))
        self.view.canvas.configure.assert_called_with(cursor="crosshair")

    def test_motion_over_margin_reports_none(self) -> None:
        self.view._on_motion(SimpleNamespace(x=100, y=100))
        self.view.on_hover.assert_called_with(None)
        self.view.canvas.configure.assert_called_with(cursor="")

    def test_leaving_canvas_reports_none(self) -> None:
        self.view._on_motion(SimpleNamespace(x=400, y=300))
        self.view._on_leave(SimpleNamespace())
        self.view.on_hover.assert_called_with(None)

    def test_without_image_reports_none(self) -> None:
        self.view._placement = None
        self.view._on_motion(SimpleNamespace(x=400, y=300))
        self.view.on_hover.assert_called_with(None)

    def test_without_callback_does_not_fail(self) -> None:
        self.view.on_hover = None
        self.view._on_motion(SimpleNamespace(x=400, y=300))
        self.view.canvas.configure.assert_called_with(cursor="crosshair")


class ImageViewRenderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.view = ImageView.__new__(ImageView)
        self.view.canvas = MagicMock()
        self.view.canvas.winfo_width.return_value = 824
        self.view.canvas.winfo_height.return_value = 624
        self.view.on_hover = MagicMock()
        self.view._pointer = None

    def test_render_centers_preview_and_records_placement(self) -> None:
        with patch("forensics_app.ui.image_view.ImageTk.PhotoImage"):
            self.view.show(Image.new("RGB", (1600, 1200)))

        # The 800x600 space (canvas minus padding) fits a half-size preview, centered.
        self.assertEqual(
            self.view._placement,
            ImagePlacement(left=12, top=12, shown_size=(800, 600), source_size=(1600, 1200)),
        )
        self.assertEqual(self.view.image.size, (1600, 1200))
        self.view.on_hover.assert_called_with(None)

    def test_render_without_image_clears_placement(self) -> None:
        self.view._placement = ImagePlacement(0, 0, (1, 1), (1, 1))
        self.view.show(None)
        self.assertIsNone(self.view._placement)
        self.assertIsNone(self.view.image)


if __name__ == "__main__":
    unittest.main()
