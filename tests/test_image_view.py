import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

from forensics_app.ui.icons import magnifier_icon
from forensics_app.ui.image_placement import ImagePlacement
from forensics_app.ui.image_view import ZOOM_LEVELS, ImageView


def make_view(width: int = 824, height: int = 624, scroll: tuple[float, float] = (0, 0)) -> ImageView:
    """An ImageView running its real logic on a fake canvas, without a Tk window."""
    view = ImageView.__new__(ImageView)
    view.canvas = MagicMock()
    view.canvas.winfo_width.return_value = width
    view.canvas.winfo_height.return_value = height
    view.canvas.canvasx.side_effect = lambda x: x + scroll[0]
    view.canvas.canvasy.side_effect = lambda y: y + scroll[1]
    view._init_state(on_hover=MagicMock(), on_zoom=MagicMock())
    return view


class ImageViewHoverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.view = make_view()
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

    def test_hover_accounts_for_scroll_offset(self) -> None:
        view = make_view(scroll=(100, 50))
        view._placement = self.view._placement
        # Window point (300, 250) is canvas point (400, 300) once scrolled.
        view._on_motion(SimpleNamespace(x=300, y=250))
        view.on_hover.assert_called_with((500, 300))


class ImageViewRenderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.view = make_view()

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
        self.view.on_zoom.assert_called_with(self.view._placement, True)

    def test_render_without_image_clears_placement(self) -> None:
        self.view._placement = ImagePlacement(0, 0, (1, 1), (1, 1))
        self.view.show(None)
        self.assertIsNone(self.view._placement)
        self.assertIsNone(self.view.image)
        self.view.on_zoom.assert_called_with(None, True)

    def test_small_image_is_not_enlarged_when_fitted(self) -> None:
        with patch("forensics_app.ui.image_view.ImageTk.PhotoImage"):
            self.view.show(Image.new("L", (180, 232)))
        self.assertEqual(self.view.scale, 1.0)
        self.assertEqual(self.view._placement.shown_size, (180, 232))


class ImageViewZoomTests(unittest.TestCase):
    def setUp(self) -> None:
        self.view = make_view(width=400, height=300)
        self.source = Image.new("RGB", (200, 100), "red")
        self.photo = patch("forensics_app.ui.image_view.ImageTk.PhotoImage").start()
        self.addCleanup(patch.stopall)
        self.view.show(self.source)

    def test_zoom_in_steps_to_next_level_and_scrolls(self) -> None:
        self.assertEqual(self.view.scale, 1.0)
        self.assertTrue(self.view.zoom_in())
        self.assertEqual(self.view.scale, 1.5)
        self.assertFalse(self.view.is_fitted)
        # 300x150 plus margins still fits the 400x300 canvas: centered, no scrolling.
        self.assertEqual(self.view._placement, ImagePlacement(50, 75, (300, 150), (200, 100)))
        self.view.canvas.configure.assert_any_call(scrollregion=(0, 0, 400, 300))

        self.assertTrue(self.view.zoom_in())
        self.assertEqual(self.view.scale, 2.0)
        # 400x200 plus margins is wider than the canvas: it scrolls horizontally only.
        self.assertEqual(self.view._placement, ImagePlacement(12, 50, (400, 200), (200, 100)))
        self.view.canvas.configure.assert_any_call(scrollregion=(0, 0, 424, 300))
        self.view.on_zoom.assert_called_with(self.view._placement, False)

    def test_zoom_out_then_reset_returns_to_fit(self) -> None:
        self.assertTrue(self.view.zoom_out())
        self.assertAlmostEqual(self.view.scale, 0.75)
        self.assertEqual(self.view._placement.shown_size, (150, 75))
        self.view.reset_zoom()
        self.assertTrue(self.view.is_fitted)
        self.assertEqual(self.view._placement.shown_size, (200, 100))

    def test_zoom_never_modifies_the_source_image(self) -> None:
        before = self.source.tobytes()
        for _ in range(5):
            self.view.zoom_in()
        self.view.zoom_out()
        self.assertIs(self.view.image, self.source)
        self.assertEqual(self.source.size, (200, 100))
        self.assertEqual(self.source.tobytes(), before)

    def test_zoom_stops_at_the_limits(self) -> None:
        while self.view.zoom_in():
            pass
        self.assertEqual(self.view.scale, ZOOM_LEVELS[-1])
        while self.view.zoom_out():
            pass
        self.assertEqual(self.view.scale, ZOOM_LEVELS[0])

    def test_zoomed_view_draws_only_visible_pixels_as_blocks(self) -> None:
        self.view._set_zoom(8.0)
        drawn = self.photo.call_args.args[0]
        # The 400x300 canvas holds at most 51x38 source pixels at 8x.
        self.assertLessEqual(drawn.width, 400 + 8)
        self.assertLessEqual(drawn.height, 300 + 8)
        self.assertEqual(drawn.getpixel((0, 0)), (255, 0, 0))

    def test_same_size_image_keeps_zoom_other_size_resets_it(self) -> None:
        self.view.zoom_in()
        self.view.show(Image.new("RGB", (200, 100), "blue"))
        self.assertEqual(self.view.scale, 1.5)
        self.view.show(Image.new("RGB", (50, 50)))
        self.assertTrue(self.view.is_fitted)

    def test_zoom_without_image_does_nothing(self) -> None:
        view = make_view()
        view.show(None)
        self.assertFalse(view.zoom_in())
        self.assertFalse(view.zoom_out())


class VisibleRegionTests(unittest.TestCase):
    def test_whole_image_visible(self) -> None:
        placement = ImagePlacement(12, 12, (400, 200), (200, 100))
        self.assertEqual(
            placement.visible_region((0, 0, 500, 300)),
            ((0, 0, 200, 100), (12, 12), (400, 200)),
        )

    def test_partially_scrolled_view_crops_whole_source_pixels(self) -> None:
        placement = ImagePlacement(12, 12, (800, 400), (200, 100))  # 4x zoom
        box, position, size = placement.visible_region((113, 0, 313, 100))
        # Canvas x 113..313 is image x 101..301 -> source columns 25..76.
        self.assertEqual(box, (25, 0, 76, 22))
        self.assertEqual(position, (112, 12))
        self.assertEqual(size, (204, 88))

    def test_view_outside_image_returns_none(self) -> None:
        placement = ImagePlacement(12, 12, (100, 100), (100, 100))
        self.assertIsNone(placement.visible_region((200, 200, 300, 300)))


class MagnifierIconTests(unittest.TestCase):
    def test_icons_are_small_rgba_and_differ(self) -> None:
        plus, minus = magnifier_icon("+"), magnifier_icon("-")
        self.assertEqual((plus.mode, plus.size), ("RGBA", (18, 18)))
        self.assertNotEqual(plus.tobytes(), minus.tobytes())

    def test_rejects_unknown_sign(self) -> None:
        with self.assertRaises(ValueError):
            magnifier_icon("x")


if __name__ == "__main__":
    unittest.main()
