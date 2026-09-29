from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.grayscale import GrayscaleTool
from forensics_app.tools.histogram import HistogramTool
from forensics_app.ui.main_window import MainWindow


class MainWindowToolTests(unittest.TestCase):
    def setUp(self):
        # Exercise the real coordinator and document without creating a Tk window.
        self.window = MainWindow.__new__(MainWindow)
        self.window.document = ImageDocument()
        self.window.document.current = Image.new("RGB", (3, 2), "red")
        for name in ("root", "status", "image_view", "undo_button", "redo_button", "results"):
            setattr(self.window, name, MagicMock())

    def test_histogram_preserves_working_image_history_saves_and_next_tool(self):
        # Create a working image history with two changes, then run the histogram tool and verify that it does not modify the document or its history. Then save the current image and verify that it matches the expected state. Finally, test undo/redo functionality and run another tool to ensure the document state is as expected.
        document = self.window.document
        # Create a working image history with two changes
        document.apply(Image.new("RGB", (3, 2), "green"))
        document.apply(Image.new("RGB", (3, 2), "blue"))
        # current after the undo should be the green image, with the red image as undo and the blue image as redo
        document.undo()
        source = document.current
        # Run the histogram tool
        with patch.object(document, "apply", wraps=document.apply) as apply:
            self.window.run_tool(HistogramTool())
        # Verify that the document's apply method was not called, meaning the histogram tool did not modify the document or its history
        apply.assert_not_called()
        # Verify that the document's current image is still the green image
        self.assertIs(document.current, source)
        # Verify that the histogram tool's result preview is displayed in the image view with the expected size
        preview = self.window.image_view.show.call_args.args[0]
        self.assertEqual(preview.size, (800, 600))
        # Verify that the document's undo and redo capabilities are still intact
        self.assertTrue(document.can_undo)
        # Verify that the document's redo capability is still intact
        self.assertTrue(document.can_redo)

        # Save the current image and verify that it matches the expected state not the preview image
        with TemporaryDirectory() as directory:
            target = Path(directory) / "saved.png"
            with patch("forensics_app.ui.main_window.filedialog.asksaveasfilename", return_value=str(target)):
                self.window.save_image()
            with Image.open(target) as saved:
                self.assertEqual(saved.size, source.size)
                self.assertEqual(saved.tobytes(), source.tobytes())

        # Test undo/redo functionality and run another tool to ensure the document state is as expected
        self.assertTrue(document.redo())
        # Verify that the current image is now the blue image
        self.assertEqual(document.current.getpixel((0, 0)), (0, 0, 255))
        self.assertTrue(document.undo())
        # Verify that the current image is now the green image again
        self.assertEqual(document.current.tobytes(), source.tobytes())
        # Run another tool (GrayscaleTool) and verify that it modifies the document as expected
        self.window.run_tool(GrayscaleTool())
        self.assertEqual(document.current.size, source.size)
        self.assertEqual(document.current.tobytes(), source.convert("L").tobytes())
        # Verify that the document's undo and redo capabilities are still intact after running the GrayscaleTool
        self.assertFalse(document.can_redo)
        self.assertTrue(document.undo())
        self.assertEqual(document.current.tobytes(), source.tobytes())


class PixelHoverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.window = MainWindow.__new__(MainWindow)
        self.window.document = ImageDocument()
        self.window.document.current = Image.new("RGB", (3, 2), "red")
        self.window.image_view = MagicMock(image=Image.new("RGBA", (800, 600), (0, 0, 255, 255)))
        self.window.pixel_info = MagicMock()
        self.window.pixel_swatch = MagicMock()
        self.window._swatch_idle = "gray"

    def test_hover_reads_the_displayed_preview_not_the_document(self) -> None:
        self.window._on_pixel_hover((700, 500))  # outside the 3x2 document image

        self.assertIn("RGBA (0, 0, 255, 255)", self.window.pixel_info.set.call_args.args[0])
        self.window.pixel_swatch.configure.assert_called_with(background="#0000ff")

    def test_hover_outside_image_resets_the_inspector(self) -> None:
        self.window._on_pixel_hover(None)

        self.window.pixel_info.set.assert_called_with("Hover over the image to inspect pixels.")
        self.window.pixel_swatch.configure.assert_called_with(background="gray")

    def test_numeric_image_shows_value_without_swatch(self) -> None:
        self.window.image_view = MagicMock(image=Image.new("F", (2, 2), 0.5))
        self.window._on_pixel_hover((1, 1))

        self.window.pixel_info.set.assert_called_with("x 1, y 1 | F 0.500")
        self.window.pixel_swatch.configure.assert_called_with(background="gray")


if __name__ == "__main__":
    unittest.main()
