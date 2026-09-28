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
        document = self.window.document
        document.apply(Image.new("RGB", (3, 2), "green"))
        document.apply(Image.new("RGB", (3, 2), "blue"))
        document.undo()
        source = document.current
        with patch.object(document, "apply", wraps=document.apply) as apply:
            self.window.run_tool(HistogramTool())
        apply.assert_not_called()
        self.assertIs(document.current, source)
        preview = self.window.image_view.show.call_args.args[0]
        self.assertEqual(preview.size, (800, 600))
        self.assertTrue(document.can_undo)
        self.assertTrue(document.can_redo)

        with TemporaryDirectory() as directory:
            target = Path(directory) / "saved.png"
            with patch("forensics_app.ui.main_window.filedialog.asksaveasfilename", return_value=str(target)):
                self.window.save_image()
            with Image.open(target) as saved:
                self.assertEqual(saved.size, source.size)
                self.assertEqual(saved.tobytes(), source.tobytes())

        self.assertTrue(document.redo())
        self.assertEqual(document.current.getpixel((0, 0)), (0, 0, 255))
        self.assertTrue(document.undo())
        self.assertEqual(document.current.tobytes(), source.tobytes())
        self.window.run_tool(GrayscaleTool())
        self.assertEqual(document.current.size, source.size)
        self.assertEqual(document.current.tobytes(), source.convert("L").tobytes())
        self.assertFalse(document.can_redo)
        self.assertTrue(document.undo())
        self.assertEqual(document.current.tobytes(), source.tobytes())


if __name__ == "__main__":
    unittest.main()
