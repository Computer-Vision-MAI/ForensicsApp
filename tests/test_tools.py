import unittest
from unittest.mock import MagicMock, patch

from PIL import Image, UnidentifiedImageError

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools.grayscale import GrayscaleTool
from forensics_app.tools.registry import ToolRegistry


class ToolTests(unittest.TestCase):
    def test_grayscale_returns_image_without_mutating_document(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (4, 3), "red")
        result = GrayscaleTool().run(None, document)  # parent is unused by this tool
        self.assertEqual(result.image.mode, "L")
        self.assertEqual(document.current.mode, "RGB")

    def test_registry_rejects_duplicate_ids(self) -> None:
        with self.assertRaises(ValueError):
            ToolRegistry([GrayscaleTool(), GrayscaleTool()])

    def test_build_tool_registry_contains_template_matching(self) -> None:
        registry = build_tool_registry()
        tool_ids = [tool.tool_id for tool in registry.all()]
        self.assertIn("template_matching", tool_ids)
        self.assertIn("corner_detection", tool_ids)
        self.assertIn("keypoint_detection", tool_ids)


class LoadImageFileTests(unittest.TestCase):
    @patch("forensics_app.tools.dialogs.filedialog.askopenfilename", return_value="")
    def test_load_image_file_cancellation(self, mock_ask: MagicMock) -> None:
        from forensics_app.tools.dialogs import load_image_file

        result = load_image_file(None, title="Test Open")
        self.assertIsNone(result)

    @patch("forensics_app.tools.dialogs.messagebox.showerror")
    @patch("forensics_app.tools.dialogs.Image.open", side_effect=UnidentifiedImageError("corrupt"))
    @patch("forensics_app.tools.dialogs.filedialog.askopenfilename", return_value="broken.png")
    def test_load_image_file_corrupt(
        self, mock_ask: MagicMock, mock_open: MagicMock, mock_error: MagicMock
    ) -> None:
        from forensics_app.tools.dialogs import load_image_file

        result = load_image_file(None, title="Test Open")
        self.assertIsNone(result)
        mock_error.assert_called_once()

    @patch("forensics_app.tools.dialogs.Image.open")
    @patch("forensics_app.tools.dialogs.filedialog.askopenfilename", return_value="good.png")
    def test_load_image_file_success(
        self, mock_ask: MagicMock, mock_open: MagicMock
    ) -> None:
        from forensics_app.tools.dialogs import load_image_file

        mock_img = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_img
        mock_img.copy.return_value = Image.new("RGB", (2, 2))
        result = load_image_file(None, title="Test Open")
        self.assertIsNotNone(result)
        path, img = result
        self.assertEqual(path.name, "good.png")
        self.assertEqual(img.size, (2, 2))


if __name__ == "__main__":
    unittest.main()


